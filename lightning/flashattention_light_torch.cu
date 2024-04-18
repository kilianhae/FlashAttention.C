#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

# define B_r 32
# define B_c 32
# define o_per_thread_x 32/32

# define d 32
# define o_per_thread_y 32/32

#define NEG_INFINITY __int_as_float(0xff800000)

__global__
void silly_attn(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;

  /*
  all are fully loaded into shared memory, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
  __shared__ float Q_i[B_r][d]; 
  __shared__ float K_j[B_c][d];
  __shared__ float V_j[B_c][d];
  
  // attention result
  __shared__ float S_i[B_r][B_c];


  float l_i[o_per_thread_x];
  float m_i[o_per_thread_x];

  // this will be automatucally be put onto registers (potentially slow if doesnt fit onto regsters and spills to local memory, prob no way around it)
  float O_i[o_per_thread_x][o_per_thread_y];

  for (int i = 0; i < T_r; i++) { // iterate over the chunks of Q

    // o_per_thread_x, o_per_thread_y is a bit like thread coarsening (each thread takes on multiple elements in loading, and potentially storing)
    for (int ii = 0; ii < o_per_thread_x; ii++) {
      for (int dd = 0; dd < o_per_thread_y; dd++) {
        O_i[ii][dd] = 0;
      }
      l_i[ii] = 0.f;
      m_i[ii] = NEG_INFINITY;
    }

    // load Q_i
    for (int ii = tid_y; ii < B_r; ii += blockDim.y) { // each thread loads offsetted to enable memory coalescing
      for (int dd = tid_x; dd < d; dd += blockDim.x) { // each thread loads offsetted to enable memory coalescing
         Q_i[ii][dd] = Q[batch_offset + (ii + i * B_r) * d + dd];
      }
    }

    // T_c is the number of chunks of K, we iterate over them
    for (int j=0; j < T_c; j++) {
        __syncthreads();

        // load K_j and V_j
        for (int jj=tid_y; jj < B_c; jj+= blockDim.y) { // each thread loads offsetted to enable memory coalescing
            for (int dd=tid_x; dd < d; dd += blockDim.x) { // each thread loads offsetted to enable memory coalescing
                K_j[jj][dd] = K[batch_offset + (jj + j * B_c) * d + dd];
                V_j[jj][dd] = V[batch_offset + (jj + j * B_c) * d + dd];
            }
        }
        __syncthreads();

        // S_i = scale * (Q_i @ K_j.T)
        for (int ii = tid_x; ii < B_r; ii += blockDim.x) {
            for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                float S_ij = 0.f;
                for (int dd = 0; dd < d; dd++) {
                    S_ij += Q_i[ii][dd] * K_j[jj][dd];
                }
                S_ij = scaling * S_ij;
                S_i[ii][jj] = S_ij;
            }
        }
        __syncthreads();

        // do softmax
        for (int ii = 0; ii < o_per_thread_x; ii++) {
            float m = m_i[ii];
            float last_m = m;
            for (int jj = 0; jj < B_c; jj += 1) {
                if (m < S_i[ii * blockDim.x + tid_x][jj]) {
                  m = S_i[ii * blockDim.x + tid_x][jj];
                }
            }
            m_i[ii] = m;
            float l = exp(last_m - m) * l_i[ii];
            for (int dd = 0; dd < o_per_thread_y; dd++) {
                O_i[ii][dd] *= exp(last_m - m);
            }
            
            for (int jj = 0; jj < B_c; jj ++) {
                float S_ij = exp(S_i[ii * blockDim.x + tid_x][jj] - m);
                l += S_ij;
                for (int dd = 0; dd < o_per_thread_y; dd++) {
                    O_i[ii][dd] += S_ij * V_j[jj][dd * blockDim.y + tid_y];
                }
            }
            l_i[ii] = l;

       }
    }

    // renormalize and add up to output
    for (int ii = 0; ii < o_per_thread_x; ii++) {
      for (int dd = 0; dd < o_per_thread_y; dd++) {
        out[batch_offset + (ii * blockDim.x + tid_x + i * B_r) * d + dd * blockDim.y + tid_y] = O_i[ii][dd] / l_i[ii];
        out_l[batch_offset / d +  ii * blockDim.x + tid_x + i * B_r] = l_i[ii];
      }
    }
  }
}


// write main function that takes two command line integer arguments
torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, d}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  // this kernel processes a whole Attention head in one block:
  dim3 blockDim(B_r, B_c); // not fully sure what the blocksize is
  dim3 gridDim(batch_size);
  printf("Launching kernel\n");
  silly_attn<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c);
  cudaDeviceSynchronize();
  return O;
}
    





