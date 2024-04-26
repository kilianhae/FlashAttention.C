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



__global__
void FlashAttention2(float *out, float *K, float *Q, float* V, float scaling, int T_r, int T_c, int seq_len)
{   
    // define constants
    const int d=64;
    const int B_c = 32;
    const int B_r = 32;
    const int BK = B_c;

    const int batch_offset = d * seq_len * blockIdx.x;
    const int TN = 4;
    const int TM = 4;
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


void run_flash(torch::Tensor O, torch::Tensor O_l, torch::Tensor K_d, torch::Tensor Q_d, torch::Tensor V_d, int batch_size, int seq_len) {
  int TM = 4;
  int TN =4;
  int B_r = 32;
    int B_c = 32;
    int d = 64;

  dim3 blockDim(B_r/TN, B_c/TM);
  dim3 gridDim(batch_size, (seq_len+B_r-1)/B_r);
  int col_tile_size = B_r * d;  // size of Kj, Vj
    int row_tile_size = B_c * d;  // size of Qi
  const int sram_size =
        (col_tile_size * sizeof(float))  // SRAM size for Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (B_c * (B_c+1) * sizeof(float)) // SRAM size for S
        + (B_c * (B_c+1) * sizeof(float)); // SRAM size for Kj, 

  //void FlashAttention2(float *out, float *K, float *Q, float* V, float scaling, const int d, int T_r, int T_c, int seq_len, int B_r, int B_c)
  FlashAttention2<<<gridDim, blockDim>>>(O.data_ptr<float>(), K_d.data_ptr<float>(), Q_d.data_ptr<float>(), V_d.data_ptr<float>(), (float) 1.0/d, (int) (seq_len+B_r-1)/B_r, (int) (seq_len+B_c-1)/B_c, seq_len);
  cudaDeviceSynchronize();
}

torch::Tensor forward(torch::Tensor Q_d, torch::Tensor K_d, torch::Tensor V_d) {
  int batch_size = Q_d.size(0);
  int seq_len = Q_d.size(1);
    int d = 64;
  torch::Tensor O = torch::zeros({batch_size, seq_len, d}, torch::kCUDA);
  torch::Tensor O_l = torch::zeros({batch_size, seq_len}, torch::kCUDA);

  run_flash(O, O_l, K_d, Q_d, V_d, batch_size, seq_len);
  return O;
}