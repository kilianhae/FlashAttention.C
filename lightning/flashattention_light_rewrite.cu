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
# define o_per_thread_y d/32

#define NEG_INFINITY __int_as_float(0xff800000)


__global__
void silly_attn_parallel(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c)
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
  __shared__ float V_j[B_r][B_c];
  
  // attention result
  __shared__ float S_i[B_r][B_c];
  
  // assuming B_c = blockdim.x, within a block, number of tiles a thread has to calculate
  const int num_tiles = d/B_c;
  
  float l_i;
  float m_i;

  assert (B_r == B_c && B_r == blockDim.x && B_r == blockDim.y);
  // assert (num_tiles == 1); // Hack: for now

  // this will be automatucally be put onto registers since very small
  float O_i[num_tiles]; // per register

  // o_per_thread_x, o_per_thread_y is a bit like thread coarsening (each thread takes on multiple elements in loading, and potentially storing)
  for (int t = 0; t < num_tiles; t++) {
    O_i[t] = 0;
  }
  
  // row wise statistics
  for (int t = 0; t < num_tiles; t++) {
    l_i = 0.f;
    m_i = NEG_INFINITY;
  }

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
      V_j[tid_y][tid_x] = V[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * B_c]; // not with with r and c
      __syncthreads();

      // tiled matrix mult
      float S_ij = 0.f;
      for (int dd=0; dd<B_c; dd++){
        S_ij += Q_i[tid_y][t * B_c + dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K
      }
      S_i[tid_y][tid_x] += scaling * S_ij;
      __syncthreads();
    }

    // tested up to here with different seq length and hidden dim and seems to work fine
    
    __syncthreads();

    // find max of each row for current j: m^{j} = max(m_{j-1},\max_i S_i)
    // renormalize current A: A^{j} \cdot \exp(m^{j} - m^{j+1})
    // Compute additional A: \exp(Q_iK^T_{j+1} - m^{j+1}) \cdot V_j
    // add up the two A's
    // renormalize the sum: l^{j} \cdot \exp(m^{j} - m^{j+1})
    // sum up new parts: sum \exp(Q_iK^T_{j+1} - m^{j+1})

    // actually: 
    // find max of each row for current j: m^{j} = max(m_{j-1},\max_i S_i)
    // renormalize current A: A^{j} \cdot \exp(m^{j} - m^{j+1})
    // renormalize the sum: l^{j} \cdot \exp(m^{j} - m^{j+1})
    // compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    // sum up new parts: sum \exp(Q_iK^T_{j+1} - m^{j+1})
    // Compute additional A: \exp(Q_iK^T_{j+1} - m^{j+1}) \cdot V_j

    // 1) find the max per row (extremely bad) with smem bank conflicts -> (add padding row to S in future)
    float last_m = m_i;
    float m = m_i;
    for (int jj = 0; jj < B_c; jj += 1) {
      if (m < S_i[tid_y][jj]) {
              m = S_i[tid_y][jj];
            }
    }
    m_i = m;

    // 2) renormalize current O
    for (int t = 0; t < num_tiles; t++){
      O_i[t] *= exp(last_m - m);
    }
    // 3) renormalize the sum
    float l = exp(last_m - m) * l_i;

    // 4) compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    float S_id;
    __syncthreads();
    for (int dd = 0; dd < B_c; dd++) {
      S_id = exp(S_i[tid_y][dd] - m);
      l += S_id;
      for (int t = 0; t < num_tiles; t++){
       // replaced o_y with 1
        O_i[t] += S_id * V_j[dd][tid_x];
      }
    }
    l_i = l;
    __syncthreads();
  }

  // normalize the whole thing by the sum and write to output
  for (int t = 0; t < num_tiles; t++){
    out[batch_offset + (blockIdx.y * B_r + tid_y ) * d + t * B_c + tid_x] = O_i[t] / l_i;
  }
}

void run_silly_attn_parallel(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_parallel<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c);
  cudaDeviceSynchronize();
}

// write main function that takes two command line integer arguments
torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, d}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_silly_attn_parallel(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}
    





