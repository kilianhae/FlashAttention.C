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

# define d 64
# define o_per_thread_y 64/32

#define NEG_INFINITY __int_as_float(0xff800000)

__global__
void silly_attn_mult(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c, int seq_len)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;
  int i = blockIdx.y;

  /*
  all are fully loaded into shared memory, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
  __shared__ float Q_i[B_r][d]; 
  __shared__ float K_j[B_r][B_c];
  __shared__ float V_j[B_r][d];
  
  // attention result
  __shared__ float S_i[B_r][B_c];
  
  
  
  // assuming B_c = blockdim.x, within a block, number of tiles a thread has to calculate
  const int num_tiles = d/B_c;
  
  float l_i[num_tiles];
  float m_i[num_tiles];

//   assert (B_r == B_c && B_r == blockDim.x && B_r == blockDim.y);
//   assert (num_tiles == 1); // Hack: for now

  // this will be automatucally be put onto registers since very small
  float O_i[num_tiles]; // per register

  // o_per_thread_x, o_per_thread_y is a bit like thread coarsening (each thread takes on multiple elements in loading, and potentially storing)
  for (int t = 0; t < num_tiles; t++) {
    O_i[t] = 0;
  }
  
  // row wise statistics
  for (int t = 0; t < num_tiles; t++) {
    l_i[t] = 0.f;
    m_i[t] = NEG_INFINITY;
  }

  // // load 
  // for (int ii = tid_y; ii < B_r; ii += blockDim.y) { // each thread loads offsetted to enable memory coalescing
  //   for (int dd = tid_x; dd < d; dd += blockDim.x) { // each thread loads offsetted to enable memory coalescing
  //       Q_i[ii][dd] = Q[batch_offset + (ii + i * B_r) * d + dd];
  //   }
  // }

  // load Q_i
  for (int t=0; t<num_tiles; t++){
    Q_i[tid_y][t * B_c + tid_x] = Q[batch_offset + (blockIdx.y * B_r + tid_y) * d + t * B_c + tid_x ];
    // Q[batch_offset + (ii + i * B_r) * d + dd];
  }
  __syncthreads();

  // T_c = seq_len (due to K^T) / B_c, chunk over the d dimension
  // T_c is the number of chunks of K, we iterate over them
  for (int j = 0; j < T_c; j++) {
    S_i[tid_y][tid_x] = 0.f;
    for (int t=0; t<num_tiles; t++){
      // load K_j and V_j, thread idx, idy loads idy,idx
      // we load a tile
      K_j[tid_y][tid_x] = K[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * B_c]; // not with with r and c
      
      // TO OPTIMIZE, just loading the V_j for now
      V_j[tid_y][t * B_c + tid_x] = V[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * B_c]; // not with with r and c
      __syncthreads();

      // tiled matrix mult
      float S_ij = 0.f;
      for (int dd=0; dd<B_c; dd++){
        S_ij += Q_i[tid_y][t * B_c + dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K
      }
      S_i[tid_y][tid_x] += scaling * S_ij;
      __syncthreads();
    }

    //load s to out
    out[batch_offset + blockIdx.y * B_r * seq_len + tid_y * seq_len + tid_x + B_c * j] = S_i[tid_y][tid_x];
    

    
  //   __syncthreads();

  //   // do softmax
  //   for (int ii = 0; ii < num_tiles; ii++) { // replaced from o_x
  //       float m = m_i[ii];
  //       float last_m = m;
  //       // both directions for attention are B_c
  //       for (int jj = 0; jj < B_c; jj += 1) {
  //           // HACK: we will try both ways
  //           if (m < S_i[jj][ii * blockDim.x + tid_x]) {
  //             m = S_i[jj][ii * blockDim.x + tid_x];
  //           }
  //       }
  //       m_i[ii] = m;
  //       float l = exp(last_m - m) * l_i[ii];
  //       // for (int dd = 0; dd < 1; dd++) { // replaced o_y with 1
  //       O_i[ii] *= exp(last_m - m);
  //       // }
        
  //       for (int jj = 0; jj < B_c; jj++) {
  //           float S_ij = exp(S_i[jj][ii * blockDim.x + tid_x] - m);
  //           l += S_ij;
  //           for (int dd = 0; dd < 1; dd++) { // replaced o_y with num_tiles
  //               O_i[ii] += S_ij * V_j[dd * blockDim.y + tid_y][jj];
  //           }
  //       }
  //       l_i[ii] = l;

  //     }
  // }

  // // renormalize and add up to output
  // for (int ii = 0; ii < num_tiles; ii++) {
  //   for (int dd = 0; dd < num_tiles; dd++) {
  //     out[batch_offset + (ii * blockDim.x + tid_x + i * B_r) * d + dd * blockDim.y + tid_y] = O_i[ii] / l_i[ii];
  //     out_l[batch_offset / d +  ii * blockDim.x + tid_x + i * B_r] = l_i[ii];
  //   }
  // }

  }
}





void run_silly_mult_parallel(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_mult<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c, seq_len);
  cudaDeviceSynchronize();
}


// write main function that takes two command line integer arguments
torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, seq_len}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_silly_mult_parallel(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}
    
