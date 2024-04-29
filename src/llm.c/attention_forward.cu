/*
Kernels for attention forward pass.

Compile example:
nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./attention_forward 1

version 2 is a naive implementation of flash attention, taken, adapted from
https://github.com/tspeterkim/flash-attention-minimal
and with help from
https://github.com/leloykun/flash-hyperbolic-attention-minimal
sadly, this flash attention version seems about 3X slower than the naive version
./attention_forward 2

version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
cuBLAS is used both to calculate the QK^T and the final weighted sum
the softmax is calculated using a custom, efficient kernel as well
this turns out to be ~20X faster than (1) nice
./attention_forward 3

version 4 is a further optimized kernel that fuses the scale operation,
uses a directly autoregressive softmax, and uses the online softmax algorithm.
./attention_forward 4
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CUDA setup

static cublasHandle_t cublas_handle;

// ----------------------------------------------------------------------------
// CPU code reference
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void attention_forward_cpu(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void attention_query_key_kernel1(float* preatt, const float* inp,
                                           int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        int hs = C / NH; // head size
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0 / sqrtf(hs);

        preatt[idx] = val;
    }
}

__global__ void attention_softmax_kernel1(float* att, const float* preatt,
                                         int B, int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // find maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
            if (preatt_bth[t2] > maxval) {
                maxval = preatt_bth[t2];
            }
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f;
            }
        }
    }
}

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        // subtract max for numerical stability
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}


__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}


__global__ void attention_value_kernel1(float* out, const float* att, const float* inp,
                                       int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_bth = out + b * T * C + t * C + h * hs;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
           const  float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

__global__
void attention_forward_kernel2(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= N) {
                break;
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // implement softmax with causal masking
            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= N) {
                        break;
                    }
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

// direct translation of the CPU kernel. Each warp handles ont (b, h, t) combination.
// The important changes compared to the CPU version:
//  - each inner loop is handled by a warp
//  - don't write non-autoregressive parts
//  - reordered the last loops so that we can do all writing in the outer loop.
__global__ void attention_forward_fused1(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int h = blockIdx.y;
    int b = blockIdx.z;

    if(t >= T) return;

    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // pass 1: calculate query dot key and maxval
    float maxval = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = warp.thread_rank(); i < hs; i += warp.size()) {
            val += query_t[i] * key_t2[i];
        }
        val = cg::reduce(warp, val, cg::plus<float>{});
        val *= scale;
        maxval = max(maxval, val);
        if(warp.thread_rank() == 0) {
            preatt_bth[t2] = val;
        }
    }

    // pass 2: calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        float expv = expf(preatt_bth[t2] - maxval);
        expsum += expv;
    }

    expsum = cg::reduce(warp, expsum, cg::plus<float>{});

    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // pass 3: normalize to get the softmax is combined with the next loop to reduce memory round-trips
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        att_bth[t2] = expf(preatt_bth[t2] - maxval) * expsum_inv;
    }

    // pass 4: accumulate weighted values into the output of attention
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = warp.thread_rank(); i < hs; i += warp.size()) {
        float o = 0.f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            o += att_btht2 * value_t2[i];
        }
        out_bth[i] = o;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}


void attention_forward2(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // TODO there should be no mallocs inside any of these functions!
    // not fixing this because we don't intend to use attention_forward2,
    // it seems to be way too slow as is

    // these are hardcoded to 32 for now
    const int Bc = 32;
    const int Br = 32;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    const int d = C / NH;
    // more
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);
    // create some temporary memory
    float* l;
    float* m;
    cudaCheck(cudaMalloc(&l, B * nh * N * sizeof(float)));
    cudaCheck(cudaMalloc(&m, B * nh * N * sizeof(float)));
    cudaCheck(cudaMemset(l, 0, B * nh * N * sizeof(float)));
    cudaCheck(cudaMemset(m, -10000.0f, B * nh * N * sizeof(float)));

    // calculate SRAM size needed per block, ensure we have enough shared memory
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
        printf("SRAM size exceeds maximum shared memory per block\n");
        printf("Try decreasing col_tile_size or row_tile_size further\n");
        exit(1);
    }

    // grid and block dims
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Br);  // Br threads per block

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    cudaCheck(cudaMalloc(&q, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&k, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&v, B * T * C * sizeof(float)));
    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, N, nh, d);

    // now actually call the flash attention kernel
    attention_forward_kernel2<<<grid_dim, block_dim, sram_size>>>(
        q, k, v,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l, m, out
    );

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, N, nh, d);
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));

    // free memory
    cudaCheck(cudaFree(l));
    cudaCheck(cudaFree(m));
    cudaCheck(cudaFree(q));
    cudaCheck(cudaFree(k));
    cudaCheck(cudaFree(v));
}

void attention_forward3(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

void attention_forward4(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     T, T, HS,
                                     &alpha,
                                     k, HS, T * HS,
                                     q, HS, T * HS,
                                     &beta,
                                     preatt, T, T * T,
                                     B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HS, T, T,
                                     &alpha,
                                     v, HS, T * HS,
                                     att, T, T * T,
                                     &beta,
                                     vaccum, HS, T * HS,
                                     B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

void attention_forward5(float* out, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // attention calculation
    int x_blocks = ceil_div(T, block_size / 32);
    attention_forward_fused1<<<dim3(x_blocks, NH, B), block_size>>>(out, preatt, att, inp, B, T, C, NH);
}

__global__ void flashattention(float *out, float *K, float *Q, float* V, float scaling, int T_r, int T_c, int seq_len)
{   // used by attention_forward6
    // define constants, could be adjusted for different hardware specs
    const int d = 64;
    const int B_c = 32;
    const int B_r = 32;
    const int BK = B_c;
    const int CACHE_Q = 0; // if 1 then cache Q in SMEM otherwise reload it over the tiles

    const int batch_offset = d * seq_len * blockIdx.x;
    const int TN = 4;
    const int TM = 4;
    const int num_tiles = d/32; // or d/BK, number of tiles that the attention computation is split into
    /*
    NOTE: all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
    and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
    */
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    // statically define in SMEM and still address it with indices
    //__shared__ float Q_i[B_r][d]; // uncomment only if you want to cache over full d (if CACHE_Q = 1)
    __shared__ float Q_i[B_r][BK]; // if you want to save SMEM loads and keep the full Q loaded then change this to [B_r][d]
    
    __shared__ float K_j[B_c][BK+1]; // reduce SMEM bank conflicts by adding 1 column as K will be loaded transposed!
    __shared__ float V_j[B_c][BK];
    
    // attention result
    __shared__ float S_i[B_r][B_c+1]; // reduce SMEM bank conflicts by adding 1 column (in the naive softmax part)
    
    const uint totalResultsBlocktile = B_r * B_c; // number of results to calculate per block
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // number of threads needed
    const int threadId_flat = threadIdx.y * blockDim.x + threadIdx.x; // flattened thread id  (used for coalesced loading of tiles)

    // each thread process one block at position:
    const int threadCol = threadId_flat % (B_c / TN);
    const int threadRow = threadId_flat / (B_c / TN);
        
    float l_i[TM]= {0.0};; // storing the intermediate sum of exponentials per row
    float m_i[TM]; // storing the intermediate max value of the rows
    float last_m[TM]; // storing the last max value of the rows
    float O_i[num_tiles * TN * TM] = {0.0}; // storing the intermediate results of the Outputs (each thread stores a chunk TM x TN per tile)
    
    // reset to min
    for (int ii = 0; ii < TM; ii++) {
        m_i[ii] = -INFINITY;
    }

    //WARNING: due to coalsecing I should probably add a second set of variables for using BK+1
    const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
    const uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
    const uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63

    int id;
    // load Q_i, UNCOMMENT only if your Q is caching over full d
    const uint innerRowQ = threadId_flat / d; // 0-63 / 64, 0000000000000...0
    const uint innerColQ = threadId_flat % d; // 0-63 % 64, 0123456789012...63
    const uint nr_loads = B_r * d / numThreadsBlocktile;

    for (int t=0; t<nr_loads; t++){
      // need to load block of size B_r x d (64 x 64) with numThreadsBlocktile threads
      // if (blockIdx.y * B_r + innerRowQ) * d + innerColQ + t * numThreadsBlocktile / d
      id = (blockIdx.y * B_r + innerRowQ) * d + innerColQ + t * numThreadsBlocktile;
      // 4 x 4 then this is 5 thus 5/
      if (id < d*seq_len){
        Q_i[innerRowQ][innerColQ + t * numThreadsBlocktile] = Q[batch_offset + id];
      }
      else {
        Q_i[innerRowQ][innerColQ + t * numThreadsBlocktile] = 0.0;
      }
    }

    __syncthreads();

    // scratchpad register for register-tiling (coarsening of the matrix mults)
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int j = 0; j < T_c && j <= blockIdx.y ; j++) { // iterate of ver the chunks of K and V
        float threadResults[TM * TN] = {0.0}; // storing the intermediate outputs
        
        for (int t=0; t<num_tiles; t++){
            // load K_j and V_j, thread idx, idy loads idy,idx
            // we load a tile
            for (int i=0; i<B_r; i+=strideK){
                // load Q, K and V in tiles (for now we are loading the full V)
                if (not CACHE_Q){Q_i[innerRowK+i][innerColK] = Q[batch_offset + (innerRowK + blockIdx.y * B_r) * d  + i * d + innerColK + t * B_c];
                } // if you cache Q over whole d then remove this line
                id = (innerRowK + j * B_c) * d + i * d + innerColK + t * B_c;
                if (id < d*seq_len){
                    K_j[innerRowK+i][innerColK] = K[batch_offset + id];
                    //V_j[innerRowK+i][innerColK+t*B_c] = V[batch_offset + id];
                } else {
                    K_j[innerRowK+i][innerColK] = 0.0;
                    //V_j[innerRowK+i][innerColK+t*B_c] = 0.0;
                }
        
            }
            __syncthreads();
        
            for (int dd=0; dd<BK; dd++){ // load elements of Q_i and K_j^T into registers
                for (uint i = 0; i < TM; ++i) {
                    if (CACHE_Q){
                        regM[i] = Q_i[(threadRow * TM + i)][dd+t*BK]; // uncomment if you cache Q over full d
                    } else {
                        regM[i] = Q_i[(threadRow * TM + i)][dd];
                    }
                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = K_j[threadCol * TN + i][dd];
                }
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }
        

        // store the results in S_i, account for causal masking
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                if (j*B_c + threadCol * TN + resIdxN <= blockIdx.y * B_r + threadRow * TM + resIdxM){
                    S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN] *scaling;
                } else {
                    S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = -INFINITY;
                }      
            }
        }
        __syncthreads();

        for (int i=0;i<TM;++i){
            last_m[i] = m_i[i];
            float m = m_i[i];
            for (int jj = 0; jj < B_c; jj += 1) {
                if (m < S_i[threadRow*TM+i][jj]) {
                    m = S_i[threadRow*TM+i][jj];
                }
            }
            m_i[i] = m;
        }

        // 2) renormalize current O
        if (j > 0) {
            for (int t = 0; t < num_tiles; t++){
                for (int i=0;i<TM;++i){
                    for (int jj=0;jj<TN;++jj){
                        O_i[t*TN*TM + i*TN + jj] *= exp(last_m[i] - m_i[i]);
                    }
                }
            }
        }

        // 3) renormalize the sum l_i
        for (int i=0;i<TM;++i){
            l_i[i] *= exp(last_m[i] - m_i[i]);
        }

        // // 4) compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1}) // TODO: TO OPTIMIZE
        // for (int dd = 0; dd < B_c; dd++) {
        //     for (int ii = 0; ii < TN; ii++){ 
        //         // calculate new sum and load exp(Attention) weights
        //         //check whether thus is in range or not (if not we set it to 0)
        //         //if (idrow+ii < seq_len && idcol+dd < seq_len){
        //         regM[ii] = exp(S_i[threadRow*TM+ii][dd] - m_i[ii]);
        //         l_i[ii] += regM[ii];
        //     }
        //     for (int t = 0; t < num_tiles; t++){
        //         for (int ii=0;ii<TN;ii++){
        //             for (int jj=0;jj<TM;jj++){ // calculate output elements
        //                 regN[jj] = V_j[dd][t * B_c + threadCol * TN + jj];
        //                 O_i[t*TN*TM + ii*TM + jj] += regM[ii] * regN[jj];
        //             }
        //         }
        //     }
        // __syncthreads();
        // }


        for (int t = 0; t < num_tiles; t++){
            // load V
            __syncthreads();
            for (int i=0; i<B_r; i+=strideK){
                id = (innerRowK + j * B_c) * d + i * d + innerColK + t * B_c;
                if (id < d*seq_len){
                    V_j[innerRowK+i][innerColK] = V[batch_offset + id];
                } else {
                    V_j[innerRowK+i][innerColK] = 0.0;
                }
            }
            __syncthreads();

            for (int dd = 0; dd < B_c; dd++) {
                for (int ii = 0; ii < TN; ii++){
                    regM[ii] = exp(S_i[threadRow*TM+ii][dd] - m_i[ii]);
                    if (t==0){
                        l_i[ii] += regM[ii];
                    }
                    regN[ii] = V_j[dd][threadCol * TN + ii];
                }
                for (int ii=0;ii<TN;ii++){
                    for (int jj=0;jj<TM;jj++){ // calculate output elements
                        regN[jj] = V_j[dd][threadCol * TN + jj];
                        O_i[t*TN*TM + ii*TM + jj] += regM[ii] * regN[jj];
                    }
                }
            }
            __syncthreads();
        }
    }

    // normalize by the output sum and write to out matrix
    for (int t = 0; t < num_tiles; t++){
        for (int ii=0;ii<TM;ii++){
            for (int jj=0;jj<TN;jj++){
                if(blockIdx.y*B_r+threadRow*TM+ii < seq_len){
                    out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t * B_c + threadCol*TN + jj] = O_i[t*TN*TM+ii*TM+jj] / l_i[ii];
                }
            }
        } 
    }
}

void attention_forward6(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // these are hardcoded to 32 for now
    const int B_r = 32;
    const int B_c = 32;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    const int d = C / NH;
    // more
    
    int TM = 4;
    int TN = 4;

    const float softmax_scale = 1.0 / sqrt(d);

    // calculate SRAM size needed per block, ensure we have enough shared memory
    int col_tile_size = B_r * d;  // size of Kj, Vj
    int row_tile_size = B_c * d;  // size of Qi
    const int sram_size =
        (col_tile_size * sizeof(float))  // SRAM size for Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (B_c * (B_c+1) * sizeof(float)) // SRAM size for S
        + (B_c * (B_c+1) * sizeof(float)); // SRAM size for Kj, 

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
        printf("SRAM size exceeds maximum shared memory per block\n");
        printf("Try decreasing col_tile_size or row_tile_size further\n");
        exit(1);
    }

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    cudaCheck(cudaMalloc(&q, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&k, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&v, B * T * C * sizeof(float)));

    dim3 blockDim(B_r/TN, B_c/TM);
    dim3 gridDim(B*nh, (N+B_r-1)/B_r);

    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, N, nh, d);

    // now actually call the flash attention kernel
    cudaDeviceSynchronize();
    double start, end;
    start = getTimeStamp();
    flashattention<<<gridDim, blockDim>>>(out, k, q, v, softmax_scale, (N+B_r-1)/B_r, (N+B_c-1)/B_c, N);
    cudaDeviceSynchronize();
    end = getTimeStamp();
    printf("Time taken for attention kernel: %f\n", end-start);

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    num_blocks = ceil_div(B * T * C, block_size);

    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, N, nh, d);
    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
    // free memory
    cudaCheck(cudaFree(q));
    cudaCheck(cudaFree(k));
    cudaCheck(cudaFree(v));
}


// kernel version dispatch
void attention_forward(int kernel_num,
                       float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_forward1(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 2:
            attention_forward2(out, inp, B, T, C, NH, block_size);
            break;
        case 3:
            attention_forward3(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 4:
            attention_forward4(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 5:
            attention_forward5(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 6:
            attention_forward6(out, inp, B, T, C, NH, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 6;
    int T = 4096;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cublasCreate(&cublas_handle);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    int block_sizes[] = {32, 64, 128, 256, 512};

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward(kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        validate_result(d_out, out, "out", B * T * C, 1e-4f);
        // but as for preatt and att, things get a bit more complicated:
        if (kernel_num != 2 && kernel_num != 6) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, 1e-4f);
        }
        if (kernel_num != 2 && kernel_num != 4 && kernel_num != 5 && kernel_num != 6) {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            validate_result(d_preatt, preatt, "preatt", B * NH * T * T, 1e-4f);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                              kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp,
                                              B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

    return 0;
}