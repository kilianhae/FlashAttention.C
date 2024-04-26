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
#include <stdlib.h>
#include <assert.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CUDA setup


// ----------------------------------------------------------------------------
// CPU code reference

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


__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
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

__global__ void flashattention(float *out, float *K, float *Q, float* V, float scaling, int T_r, int T_c, int seq_len)
{   
    printf("within kenrel");

    // define constants
    const int d=64;
    const int B_c = 32;
    const int B_r = 32;
    const int BK = B_c;

    const int batch_offset = d * seq_len * blockIdx.x;
    const int TN = 1;
    const int TM = 1;
    const int num_tiles = 64/32; // d/BK;
  /*
  all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
 // define static smem such that i can still adress it with indices
    int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;


  /*
  all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
  
  __shared__ float Q_i[B_r][d]; // uncomment only if you want to cache over full d (if CACHE_Q = 1), also + 1 seems to help SMEM latency here too
  //__shared__ float Q_i[B_r][BK]; // if you want to save SMEM loads and keep the full Q loaded then change this to [B_r][d]
  __shared__ float K_j[B_c][BK+1]; // reduce SMEM bank conflicts by adding 1 column as K will be loaded transposed!
  __shared__ float V_j[B_c][d];
  
  // attention result
  __shared__ float S_i[B_r][B_c+1]; // reduce SMEM bank conflicts by adding 1 column (in the naive softmax part)
  
  const uint totalResultsBlocktile = B_r * B_c; // number of results to calculate per block
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // number of threads needed
  const int threadId_flat = threadIdx.y * blockDim.x + threadIdx.x; // flattened thread id  (used for loading tiles coalesced)

  // each thread process 1 block at position:
  const int threadCol = threadId_flat % (B_c / TN); // 0-63 % 8 => 0,1,2,3,4...7,0,1,2,3,4...7,... 0,1,2,3,4...7
  const int threadRow = threadId_flat / (B_c / TN); // 0-63 / 8 => 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,..., 7,7,7,7,7,7,7,7

  //const int num_tiles = d/BK; // how many tiles are the computationof the attention is split into
  
  float l_i[TM]= {0.0};; // storing the intermediate sum of exp per row
  float m_i[TM]; // storing the intermediate max of the rows
  float last_m[TM]; // storing the last max of the rows
  float O_i[num_tiles * TN * TM] = {0.0}; // storing the intermediate results of the Outputs (each thread stores a chunk TM x TN per tile)
  
  // rset to min
  for (int ii = 0; ii < TM; ii++) {
    m_i[ii] = -INFINITY;
  }

  //WARNING due to coalsecing I should probably add a second set of variables for using BK+1
  const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
  const uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
  const uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63


  // do if: blockIdx.y * B_r + innerRowK * TM + row < seq_len
  // or: j * B_c + innerColK * TN + col < seq_len

  int id;
  // load Q_i, UNCOMMENT only if your Q is caching over full d
  
    const uint innerRowQ = threadId_flat / d; // 0-63 / 64, 0000000000000...0
    const uint innerColQ = threadId_flat % d; // 0-63 % 64, 0123456789012...63
    const uint nr_loads = B_r * d / numThreadsBlocktile;
    for (int t=0; t<nr_loads; t++){
      // need to laod block of size B_r x d (64 x 64) with numThreadsBlocktile threads
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

  // scratchboard register for registertiling (coarsening of the matrix mults)
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int j = 0; j < T_c; j++) { // iterate of ver the chunks of K and V
    float threadResults[TM * TN] = {0.0}; // storing the intermediate outputs
    S_i[tid_y][tid_x] = 0.f;
    
    for (int t=0; t<num_tiles; t++){
      // load K_j and V_j, thread idx, idy loads idy,idx
      // we load a tile
      for (int i=0; i<B_r; i+=strideK){
        // load Q, K and V in tiles (for now we are loading the full V)
        // if (not CACHE_Q){Q_i[innerRowK+i][innerColK] = Q[batch_offset + (innerRowK + blockIdx.y * B_r) * d  + i * d + innerColK + t * B_c];
        // } // if you cache Q over whole d then remove this line
        id = (innerRowK + j * B_c) * d + i * d + innerColK + t * B_c;
        if (id < d*seq_len){
          K_j[innerRowK+i][innerColK] = K[batch_offset + id];
          V_j[innerRowK+i][innerColK+t*B_c] = V[batch_offset + id];
        }
        else {
          K_j[innerRowK+i][innerColK] = 0.0;
          V_j[innerRowK+i][innerColK+t*B_c] = 0.0;
        }
       
      }
      __syncthreads();
      
      for (int dd=0; dd<BK; dd++){ // load elements of Q_i and K_j^T into registers
        for (uint i = 0; i < TM; ++i) {
            regM[i] = Q_i[(threadRow * TM + i)][dd+t*BK]; // uncomment if you cache Q over full d
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
    

    // store the results in S_i
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN]/scaling;
      }
    }
    __syncthreads();
    
    // tested up to here with different seq length and hidden dim and seems to work fine
    // find max of each row for current j: m^{j} = max(m_{j-1},\max_i S_i)
    // renormalize current A: A^{j} \cdot \exp(m^{j} - m^{j+1})
    // renormalize the sum: l^{j} \cdot \exp(m^{j} - m^{j+1})
    // compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    // sum up new parts: sum \exp(Q_iK^T_{j+1} - m^{j+1})
    // Compute additional A: \exp(Q_iK^T_{j+1} - m^{j+1}) \cdot V_j

    // each tread now needs to find max of the rows its assigned to (WARNING: implemented very naively)
    int bound;
    if (j<T_c-1){
      bound = B_c;
    }
    else{
      bound = (seq_len+31)%B_c + 1;
    }

    for (int i=0;i<TM;++i){
      last_m[i] = m_i[i];
      float m = m_i[i];
      for (int jj = 0; jj < bound; jj += 1) {
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

    // 4) compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1}) // TO OPTIMIZE!!
    int idrow = threadIdx.y*B_r+threadRow*TM;
    int idcol = j*B_c+threadCol*TN;

    for (int dd = 0; dd < bound; dd++) {
      for (int ii=0;ii<TN;ii++){ // calculate new sum and load exp(Attention) weights
        //check wether thus is in range  or not (if not we set it to 0)
        //if (idrow+ii < seq_len && idcol+dd < seq_len){
          regM[ii] = exp(S_i[threadRow*TM+ii][dd] - m_i[ii]);
          l_i[ii] += regM[ii];
      }
      for (int t = 0; t < num_tiles; t++){
        for (int ii=0;ii<TN;ii++){
          for (int jj=0;jj<TM;jj++){ // calculate output elements
            regN[jj] = V_j[dd][t * B_c + threadCol * TN + jj];
            O_i[t*TN*TM + ii*TM + jj] += regM[ii] * regN[jj];
          }
        }
      }
    __syncthreads();
    }

  }

  // normalize the whole thing by the output sum and write to out
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
    // TODO there should be no mallocs inside any of these functions!
    // not fixing this because we don't intend to use attention_forward2,
    // it seems to be way too slow as is

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
    printf("prekernel \n");
    flashattention<<<blockDim, gridDim>>>(out, k, q, v, softmax_scale, (N+B_r-1)/B_r, (N+B_c-1)/B_c, N);
    cudaDeviceSynchronize();

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    num_blocks = ceil_div(B * T * C, block_size);

    //unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, N, nh, d);
    cudaDeviceSynchronize();
    //cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
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
            printf("Running kernel 1\n");
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

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;


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
        if (kernel_num != 2) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, 1e-4f);
        }
        if (kernel_num != 2 && kernel_num != 4 && kernel_num != 5) {
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

    return 0;
}