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

//# define B_r 32
//# define B_c 32
# define o_per_thread_x 32/32

# define d 64
# define o_per_thread_y d/32

#define NEG_INFINITY __int_as_float(0xff800000)


# define B_r 32 // B_r or BM
# define B_c 32 // B_c or BN
# define BK 32 // used to be B_c but now different  due to coarsening

// thread - 2nd level tiling
# define TM 4 // threadblock size
# define TN 4 // threadblock size


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
  __shared__ float V_j[B_r][d];
  
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
    float S_ij = 0.f;
    for (int t=0; t<num_tiles; t++){
      // load K_j and V_j, thread idx, idy loads idy,idx
      // we load a tile
      K_j[tid_y][tid_x] = K[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * B_c]; // not with with r and c
      
      // TO OPTIMIZE, just loading the V_j for now
      V_j[tid_y][t * B_c + tid_x] = V[batch_offset + (tid_y + j * B_c) * d  + tid_x + t * B_c]; // not with with r and c
      __syncthreads();

      // tiled matrix mult
      
      for (int dd=0; dd<B_c; dd++){
        S_ij += Q_i[tid_y][t * B_c + dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K
      }
      __syncthreads();
    }
    S_i[tid_y][tid_x] += scaling * S_ij;
    __syncthreads();

    // tested up to here with different seq length and hidden dim and seems to work fine
    // find max of each row for current j: m^{j} = max(m_{j-1},\max_i S_i)
    // renormalize current A: A^{j} \cdot \exp(m^{j} - m^{j+1})
    // renormalize the sum: l^{j} \cdot \exp(m^{j} - m^{j+1})
    // compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    // sum up new parts: sum \exp(Q_iK^T_{j+1} - m^{j+1})
    // Compute additional A: \exp(Q_iK^T_{j+1} - m^{j+1}) \cdot V_j

    // 1) fin the max per row (extremely bad) with smem bank conflicts -> (add padding row to S in future)
    // all with same tid_y collaborate on a reduce scheme to find max and finally the one at position 0 is the max
    
    

    float last_m = m_i;
    float m = m_i;
    for (int jj = 0; jj < B_c; jj += 1) {
      if (m < S_i[tid_y][jj]) {
              m = S_i[tid_y][jj];
            }
    }
    __syncthreads();
    m_i = m;

    // load urself and load urself + B_c/2
    // if (i%2==0){
    //   for (int jj = B_c; jj > 0; jj /= 2) {
    //   float m_i_jj = S_i[tid_y][tid_x + jj];
    //   if (tid_x < jj) {
    //     if (S_i[tid_y][tid_x] < m_i_jj){
    //       S_i[tid_y][tid_x] = m_i_jj;
    //     }
    //   }
    //   __syncthreads();
    // }
    

    // m_i = S_i[tid_y][0];
    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y==0 && blockIdx.x == 0 && blockIdx.y == 0){
    //   printf("m_i_s: %f\n", m_i);
    // }

    // if (m_i > last_m) {
    //   m = m_i;
    // }
    // }
    
    __syncthreads();
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
        O_i[t] += S_id * V_j[dd][t * B_c + tid_x];
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

// tiling 

__launch_bounds__(1024)
__global__ void silly_attn_parallel_coarse(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;

  /*
  all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
  and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
  */
  __shared__ float Q_i[B_r][d]; // fully stored 
  __shared__ float K_j[B_r][BK]; // used to be B_r x B_c but now is B_r x B_K (BN x BK) 64 x 64
  __shared__ float V_j[B_r][d]; // fully stored  64 x 64
  
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
  const int num_tiles = d/BK; // how many tiles there are on the outside 64/64 = 1
  
  float l_i[TN]; // 4 
  float m_i[TN]; // 4 
  float last_m[TN]; // 4
   float O_i[num_tiles * TN * TM]={0.0}; 
  //float l[TN];
  
  // this will be automatucally be put onto registers since very small
  // 4, per register, cache before loading into global memory write // 1*8*8 = 64

  // o_per_thread_x, o_per_thread_y is a bit like thread coarsening (each thread takes on multiple elements in loading, and potentially storing)
  for (int t = 0; t < num_tiles; t++) {
    for (int i=0; i<TN; i++){
      for (int j=0; j<TM; j++){
        O_i[t*TN*TM + i*TM + j] = 0;
      }
    }
  }
  
  // row wise statistics
  for (int ii = 0; ii < TM; ii++) {
    l_i[ii] = 0.f;
    m_i[ii] = NEG_INFINITY;
  }

  const uint innerRowQ = threadId_flat / d; // 0-63 / 64, 0000000000000...0
  const uint innerColQ = threadId_flat % d; // 0-63 % 64, 0123456789012...63
  const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
  const uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
  const uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63
  const int nr_loads = B_r * d / numThreadsBlocktile; //64*64 /64 = 64

  
  // load all of Q_i in coalesced manner 
  for (int t=0; t<nr_loads; t++){
    // need to laod block of size B_r x d (64 x 64) with numThreadsBlocktile threads
    Q_i[innerRowQ][innerColQ + t * numThreadsBlocktile] = Q[batch_offset + (blockIdx.y * B_r + innerRowQ) * d + innerColQ + t * numThreadsBlocktile];
  }
  
  __syncthreads();
  //float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // T_c = seq_len (due to K^T) / B_c, chunk over the d dimension
  // T_c is the number of chunks of K, we iterate over them
  for (int j = 0; j < T_c; j++) {
    float threadResults[TM * TN] = {0.0};
    S_i[tid_y][tid_x] = 0.f;
    for (int t=0; t<num_tiles; t++){
      // load K_j and V_j, thread idx, idy loads idy,idx
      // we load a tile
      for (int i=0; i<B_r; i+=strideK){ // WARNING: only corret for BK=B_c i think
        // need to load 64 x 64 
        K_j[innerRowK+i][innerColK] = K[batch_offset + (innerRowK + j * B_r) * d  + i * d + innerColK + t * B_c]; // not with with r and c
        V_j[innerRowK+i][innerColK+t*B_c] = V[batch_offset + (innerRowK + j * B_r) * d  + i * d + innerColK + t * B_c]; // not with with r and c

      }
      
      // TO OPTIMIZE, just loading the V_j for now
      __syncthreads();

      // tiled matrix mult (NEEDS TO BE COARSENED NOW!)
      //float S_ij = 0.f;
      //S_ij += Q_i[tid_y][t * B_c + dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K

      for (int dd=0; dd<BK; dd++){ // 0 to 64
        for (uint i = 0; i < TM; ++i) {
          regM[i] = Q_i[(threadRow * TM + i)][dd];
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

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
      }
    }
    __syncthreads();

    // should be correct up to here apart from loading V!


    // tested up to here with different seq length and hidden dim and seems to work fine
    // find max of each row for current j: m^{j} = max(m_{j-1},\max_i S_i)
    // renormalize current A: A^{j} \cdot \exp(m^{j} - m^{j+1})
    // renormalize the sum: l^{j} \cdot \exp(m^{j} - m^{j+1})
    // compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    // sum up new parts: sum \exp(Q_iK^T_{j+1} - m^{j+1})
    // Compute additional A: \exp(Q_iK^T_{j+1} - m^{j+1}) \cdot V_j

    // 1) fin the max per row (extremely bad) with smem bank conflicts -> (add padding row to S in future)
    // all with same tid_y collaborate on a reduce scheme to find max and finally the one at position 0 is the max
    
    //let threa 0 block 0 print S
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
    //   for (int i=0; i<B_r; i++){
    //     for (int jj=0; jj<B_c; jj++){
    //       printf("%f ", S_i[i][jj]);
    //     }
    //     printf("\n");
    //   }
    // }


    // each tread now needs to find amx of the rows its assigned to
    for (int i=0;i<TN;++i){
      last_m[i] = m_i[i];
      float m = m_i[i];
      for (int jj = 0; jj < B_c; jj += 1) {
        if (m < S_i[threadRow*TN+i][jj]) {
          m = S_i[threadRow*TN+i][jj];
        }
      }
      __syncthreads();
      m_i[i] = m;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
      for (int i=0; i<TN; i++){
        printf("m_i: %f\n", m_i[i]);
      }
    }

    // load urself and load urself + B_c/2
    // if (i%2==0){
    //   for (int jj = B_c; jj > 0; jj /= 2) {
    //   float m_i_jj = S_i[tid_y][tid_x + jj];
    //   if (tid_x < jj) {
    //     if (S_i[tid_y][tid_x] < m_i_jj){
    //       S_i[tid_y][tid_x] = m_i_jj;
    //     }
    //   }
    //   __syncthreads();
    // }
    

    // m_i = S_i[tid_y][0];
    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y==0 && blockIdx.x == 0 && blockIdx.y == 0){
    //   printf("m_i_s: %f\n", m_i);
    // }

    // if (m_i > last_m) {
    //   m = m_i;
    // }
    // }
    
    
    __syncthreads();
    // 2) renormalize current O and // 3) renormalize the sum
    for (int t = 0; t < num_tiles; t++){
      for (int i=0;i<TN;++i){
        for (int jj=0;jj<TM;++jj){
          O_i[t*TN*TM + i*TM + j] *= exp(last_m[i] - m_i[i]);
        }
      // l[i] = exp(last_m[i] - m_i[i]) * l_i[i];
      }
    }

    if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x == 0 && blockIdx.y == 0){
      for (int i=0; i<TN; i++){
        for (int jj=0; jj<TM; jj++){
          printf("%f ", O_i[i*TM+jj]);
        }
        printf("\n");
      }
    }


    // 4) compute \exp(Q_iK^T_{j+1} - m^{j+1}) = \exp(S_i-m^{j+1})
    
    __syncthreads();
    for (int dd = 0; dd < B_c; dd++) { // 32 iterates
    __syncthreads();
      for (int ii=0;ii<TN;ii++){
        regM[ii] = exp(S_i[threadRow*TN+ii][dd] - m_i[ii]); // exp(64-64) = 1
        
        l_i[ii] += regM[ii]; // +=1 -> 32
        __syncthreads();
      }
      __syncthreads();
      for (int t = 0; t < num_tiles; t++){ // 2
       // replaced o_y with 1
       for (int ii=0;ii<TN;ii++){ // 4
        regN[ii] = V_j[dd][t * B_c + threadCol * TN + ii]; // set to 1
        printf("regN: %f\n", regN[ii]);
        __syncthreads();
        for (int jj=0;jj<TN;jj++){
          O_i[t*TN*TM+ii*TM+jj] += regM[ii] * regN[jj]; // every =_i is += 1 32 times
        }__syncthreads();}
        __syncthreads();
      }
      __syncthreads();
    }


    
    if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x == 0 && blockIdx.y == 0){
      for (int i=0; i<TN; i++){
        for (int jj=0; jj<TM; jj++){
          printf("%f ", O_i[i*TM+jj]);
        }
        printf("\n");
      }
    }

    
    //print l_i
    
    //l_i = l;
    __syncthreads();
  }
  


  // normalize the whole thing by the sum and write to output
  for (int t = 0; t < num_tiles; t++){
    for (int ii=0;ii<TN;ii++){
      for (int jj=0;jj<TN;jj++){
                out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t * B_c + threadCol*TN+jj] = O_i[t*TN*TM+ii*TM+jj] / l_i[ii];
      }
    }  
  }
  
}


void run_silly_attn_parallel(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_parallel<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c);
  cudaDeviceSynchronize();
}


void run_silly_attn_parallel_coarse(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size, (int) seq_len/B_r);
  silly_attn_parallel_coarse<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c);
  cudaDeviceSynchronize();
}



// write main function that takes two command line integer arguments
torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, d}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_silly_attn_parallel_coarse(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}
    





