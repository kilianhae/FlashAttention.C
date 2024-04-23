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

// # define B_r 32
// # define B_c 32
# define o_per_thread_x 32/32

# define d 64
# define o_per_thread_y 64/32

#define NEG_INFINITY __int_as_float(0xff800000)


// tiling 
# define B_r 32 // B_r or BM
# define B_c 32 // B_c or BN
# define BK 32 // used to be B_c but now different  due to coarsening

// thread - 2nd level tiling
# define TM 4 // threadblock size
# define TN 4 // threadblock size

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





__global__
void silly_attn_parallel_coarse(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c, int seq_len)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;
  int i = blockIdx.y;

  /*
  all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
  __shared__ float Q_i[B_r][d]; // fully stored 64 x 128
  __shared__ float K_j[B_r][BK]; // used to be B_r x B_c but now is B_r x B_K (BN x BK) 64 x 64
  __shared__ float V_j[B_r][d]; // fully stored  64 x 128
  
  // attention result
  __shared__ float S_i[B_r][B_c]; //64 x 64 
  
  const uint totalResultsBlocktile = B_r * B_c; // number of results to calculate per block, 4096 = 64 x 64
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // (64 * 64) / (8 * 8) = 64
  const int threadId_flat = threadIdx.y * blockDim.x + threadIdx.x;

  // each thread process 1 block
  // TODO: check if this is correct
  const int threadCol = threadId_flat % (B_c / TN); // 0-63 % 8 => 0,1,2,3,4...7,0,1,2,3,4...7,... 0,1,2,3,4...7
  const int threadRow = threadId_flat / (B_c / TN); // 0-63 / 8 => 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,..., 7,7,7,7,7,7,7,7

  // assuming B_c = blockdim.x, within a block, number of tiles a thread has to calculate
  const int num_tiles = d/BK; // how many tiles there are on the outside 128/64 = 2
  
  float l_i;
  float m_i;
  
  // this will be automatucally be put onto registers since very small
  float O_i[num_tiles * TN * TM]; // per register, cache before loading into global memory write // 1*8*8 = 64

  // o_per_thread_x, o_per_thread_y is a bit like thread coarsening (each thread takes on multiple elements in loading, and potentially storing)
  for (int t = 0; t < num_tiles; t++) {
    O_i[t] = 0;
  }
  
  // row wise statistics
  for (int t = 0; t < num_tiles; t++) {
    l_i = 0.f;
    m_i = NEG_INFINITY;
  }
  const uint innerRowQ = threadId_flat / d; // 0-63 / 64, 0000000000000...0
  const uint innerColQ = threadId_flat % d; // 0-63 % 64, 0123456789012...63

  const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
  const uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
  const uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63

  // 
  const int nr_loads = B_r * d / numThreadsBlocktile; //64*128 /64 = 128

  
  // load all of Q_i in coalesced manner 
  for (int t=0; t<nr_loads; t++){
    // need to laod block of size B_r x d (64 x 64) with numThreadsBlocktile threads
    Q_i[innerRowQ][innerColQ + t * numThreadsBlocktile] = Q[batch_offset + (blockIdx.y * B_r + innerRowQ) * d + innerColQ + t * numThreadsBlocktile];
  }
  
  __syncthreads();
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // T_c = seq_len (due to K^T) / B_c, chunk over the d dimension
  // T_c is the number of chunks of K, we iterate over them
  for (int j = 0; j < T_c; j++) {
    S_i[tid_y][tid_x] = 0.f;
    for (int t=0; t<num_tiles; t++){
      // load K_j and V_j, thread idx, idy loads idy,idx
      // we load a tile
      for (int i=0; i<B_r; i+=strideK){ // WARNING: only corret for BK=B_c i think
        // need to load 64 x 64 
        K_j[innerRowK+i][innerColK] = K[batch_offset + (innerRowK + j * B_r) * d  + i * d + innerColK + t * BK]; // not with with r and c
      }
      // if (threadId_flat == 0 && blockIdx.x == 0 && blockIdx.y == 0){
      //   // print K_j
      //   for (int i=0; i<B_r; i++){
      //     for (int j=0; j<BK; j++){
      //       printf("%f ", K_j[i][j]);
      //     }
      //     printf("\n");
      //   }
      // }
      
      // TO OPTIMIZE, just loading the V_j for now
      V_j[tid_y][t * B_c + tid_x] = V[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * BK]; // not with with r and c
      __syncthreads();

      // tiled matrix mult (NEEDS TO BE COARSENED NOW!)
      //float S_ij = 0.f;
      //S_ij += Q_i[tid_y][t * B_c + dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K

      for (int dd=0; dd<BK; dd++){ // 0 to 64
        for (uint i = 0; i < TM; ++i) {
          regM[i] = Q_i[(threadRow * TM + i)][dd+t*BK];
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

    if (threadId_flat==0 && blockIdx.x == 0 && blockIdx.y == 0){
      for (int i=0; i<TM; i++){
        for (int j=0; j<TN; j++){
          printf("%f ", threadResults[i * TN + j]);
        }
        printf("\n");
      }
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
      }
    }
    __syncthreads();

    for (int n=0;n<TN;n++){
      for (int m=0;m<TM;m++){
        out[batch_offset + blockIdx.y * B_r * seq_len + threadRow * TM * seq_len + threadCol * TM + B_c * j + m * seq_len + n] = S_i[threadRow * TM + m][threadCol * TN + n];
        //printf("id %d",batch_offset + blockIdx.y * B_r * seq_len + threadRow * TM * seq_len + threadCol * TM + B_c * j + m * seq_len + n);
      }
    }

  }
}


void run_silly_mult_parallel(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_mult<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c, seq_len);
  cudaDeviceSynchronize();
}

void run_silly_mult_parallel_coarse(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r/TN, B_c/TM);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_parallel_coarse<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c, seq_len);
  cudaDeviceSynchronize();
}


// write main function that takes two command line integer arguments
torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, seq_len}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_silly_mult_parallel_coarse(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}
    
