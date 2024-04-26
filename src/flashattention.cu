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


# define B_r 32 // How many rows of Q_i are processed by one threadblock
# define B_c 32 // How many rows of K_i and V_i are processed by one threadblock
# define BK 32 // for now = B_c

// thread - 2nd level tiling
# define TM 4 // How many rows f the attention Matrix S are processed by a single thread
# define TN 4 // How many columns of the attention Matrix S are processed by a single thread

# define CACHE_Q 1

__global__
void silly_attn_parallel(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;

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



      for (int dd=0; dd<B_c; dd++){
        S_ij += Q_i[tid_y][t*B_c+dd] * K_j[tid_x][dd]; // this maybe leads to bank conflicts in the K
      }
      __syncthreads();
    }
    S_i[tid_y][tid_x] += scaling * S_ij;
    __syncthreads();

    float last_m = m_i;
    float m = m_i;
    for (int jj = 0; jj < B_c; jj += 1) {
      if (m < S_i[tid_y][jj]) {
              m = S_i[tid_y][jj];
            }
    }
    __syncthreads();
    m_i = m;
    
    // print all of V
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


__global__
void silly_attn_parallel_coarse(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c, int seq_len)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;

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

  const int num_tiles = d/BK; // how many tiles are the computationof the attention is split into
  
  float l_i[TM]= {0.0};; // storing the intermediate sum of exp per row
  float m_i[TM]; // storing the intermediate max of the rows
  float last_m[TM]; // storing the last max of the rows
  float O_i[num_tiles * TN * TM] = {0.0}; // storing the intermediate results of the Outputs (each thread stores a chunk TM x TN per tile)
  
  // rset to min
  for (int ii = 0; ii < TM; ii++) {
    m_i[ii] = NEG_INFINITY;
  }

  //WARNING due to coalsecing I should probably add a second set of variables for using BK+1
  const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
  const uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
  const uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63


  // do if: blockIdx.y * B_r + innerRowK * TM + row < seq_len
  // or: j * B_c + innerColK * TN + col < seq_len

  int id;
  // load Q_i, UNCOMMENT only if your Q is caching over full d
  if (CACHE_Q){
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
    

    // store the results in S_i
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
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



void run_silly_attn_parallel(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r, B_c);
  dim3 gridDim(batch_size,  (seq_len+B_r-1)/B_r);
  
  silly_attn_parallel<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) seq_len/B_r, (int) seq_len/B_c);
  cudaDeviceSynchronize();
}

void run_silly_attn_parallel_coarse(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  dim3 blockDim(B_r/TN, B_c/TM);
  dim3 gridDim(batch_size, (seq_len+B_r-1)/B_r);
  silly_attn_parallel_coarse<<<gridDim, blockDim>>>(O.data_ptr<float>(), O_l.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0, (int) seq_len * d, (int) (seq_len+B_r-1)/B_r, (int) (seq_len+B_c-1)/B_c, seq_len);
  cudaDeviceSynchronize();
}


torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
  assert (Q_d.size(2) == d);

  torch::Tensor O = torch::zeros({batch_size, seq_len, d}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_flash(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}