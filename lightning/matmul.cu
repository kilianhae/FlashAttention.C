#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define H 8
#define BB 1
#define BLKS 32


#define BM 64
#define BN 64
#define BK 8

// thread coarsening
#define TM 8
#define TN 8

// 
# define BD 1

#define tilesize 32

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


template <typename T>
void randomInit(std::vector<T> &x) {
    // Pseudo-random float vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> unif(-1, 1);
    for (int i = 0; i < x.size(); i++) {
        x[i] = unif(gen);
    }
}


__global__ void sgemm_naive(int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the row index and y as the column index (which leads to bad coalescing)
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];

    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp;
  }
}


__global__ void sgemm_naive_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the row index and y as the column index (which leads to bad coalescing)
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;

  if (batch >= L) {
    return;
  }
  

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[offset_A + x * K + i] * B[offset_B + i * N + y];

    }
    // C = α*(A@B)+β*C
    C[offset_C + x * N + y] = tmp;
  }
}


__global__ void sgemm_naive_coalesced(int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[y*K+i] * B[i*N+x];

    }
    // C = α*(A@B)+β*C
    C[y*N+x] = tmp;
  }
}


__global__ void sgemm_naive_coalesced_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;

  if (batch >= L) {
    return;
  }
  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[offset_A+y*K+i] * B[offset_B+i*N+x];

    }
    // C = α*(A@B)+β*C
    C[offset_C+y*N+x] = tmp;
  }
}


__global__ void sgemm_naive_coalesced_tiled(int M, int N, int K, float *A,
                            float *B, float *C) {
    // assign smem
    __shared__ float As[tilesize][tilesize];
    __shared__ float Bs[tilesize][tilesize];

    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;


    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint blocksize=blockDim.x;

    
    float val = 0.0;
    // loop over phases of tiling 
    for(int i=0;i<K;i+=blocksize) {
        //load to shared mem
        
        if (y<M && i+tx<K){

            As[ty][tx] = A[y*K+i+tx];
        }
        else{
            As[ty][tx] = 0;
        }
        if (x<N && i+ty<K){
            Bs[ty][tx] = B[(i+ty)*N+x];
        }
        else{
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        for(int j=0;j<blocksize;j++) {
            val += As[ty][j]*Bs[j][tx];
        }
        __syncthreads();
    }
    C[y*N+x] = val;
}


__global__ void sgemm_naive_coalesced_tiled_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

    const uint offset_A = batch * M * K;
    const uint offset_B = batch * K * N;
    const uint offset_C = batch * M * N;

    if (batch >= L) {
      return;
    }

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint blocksize=blockDim.x;

    assert(blocksize == tilesize);

    __shared__ float As[tilesize][tilesize];
    __shared__ float Bs[tilesize][tilesize];

    float val = 0.0;
    // loop over phases of tiling 
    for(int i=0;i<K;i+=blocksize) {
        //load to shared mem
        
        if (y<M && i+tx<K){

            As[ty][tx] = A[offset_A+y*K+i+tx];
        }
        else{
            As[ty][tx] = 0;
        }
        if (x<N && i+ty<K){
            Bs[ty][tx] = B[offset_B+(i+ty)*N+x];
        }
        else{
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        for(int j=0;j<blocksize;j++) {
            val += As[ty][j]*Bs[j][tx];
        }
        __syncthreads();
    }
    C[offset_C+y*N+x] = val;
}


__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) 
sgemm2DBlocktiling(int M, int N, int K, const float *A, const float *B,float *C) {
  // This kernel is taken from the amazing Matmul guide from Simon Boehm: https://siboehm.com/articles/22/CUDA-MMM
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;
  const uint strideA = numThreadsBlocktile / BK;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  float a;
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
             regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}


__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling_batched(int L, int M, int N, int K, const float *A,
                       const float *B,float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;
  

  if (batch >= L) {
    return;
  }
  
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K + offset_A;
  B += cCol * BN + offset_B;
  C += cRow * BM * N + cCol * BN + offset_C;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;
  const uint strideA = numThreadsBlocktile / BK;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  
  
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
             regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}

void run_sgemm_cublas(float* A, float* B, float* C, int M, int N, int K, bool transpose){

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    const float alpha = 1.0;
    const float beta = 0.0;
    // loop over batchsize and head
    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            // get the i-th batch and j-th head
            float* Aij = &A[i*H*K*N+j*K*N];
            float* Bij = &B[i*H*K*N+j*K*N];
            float* Cij = &C[i*H*K*N+j*K*N];
            // compute the matrix multiplication
            // cublas expects A to be m x k, B to be k x n, and C to be m x n
            // BUT in col major layout
            if(transpose){
                stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, N, M, &alpha, Bij, M, Aij, K, &beta, Cij,K);
            }
            else{
            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, Bij, M, Aij, K, &beta, Cij,M);
            // allocate memory for output on GPU in cuda
            }
        }
    }
}

void run_sgemm_cublas_batched(float* A, float* B, float* C, int M, int N, int K, bool transpose){
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle;
    M=4096;
    N=4096;
    K=64;

    stat = cublasCreate(&handle);
    const float alpha = 1.0;
    const float beta = 0.0;
    // loop over batchsize and head
    // make array of pointers of elelments of A with stride of M*K
    // make array of pointers of elelments of B with stride of K*N
    // make array of pointers of elelments of C with stride of M*N

    float *Aarray[BB*H];
    float *Barray[BB*H];
    float *Carray[BB*H];

    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            Aarray[i*H+j] = &A[i*H*K*N+j*K*N];
            Barray[i*H+j] = &B[i*H*K*N+j*K*N];
            Carray[i*H+j] = &C[i*H*K*N+j*K*N];
        }
    }


    float **Aarray_d;
    float **Barray_d;
    float **Carray_d;
    cudaMalloc((void**)&Aarray_d, BB*H*sizeof(float*));
    cudaMalloc((void**)&Barray_d, BB*H*sizeof(float*));
    cudaMalloc((void**)&Carray_d, BB*H*sizeof(float*));

    cudaMemcpy(Aarray_d, Aarray, BB*H*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(Barray_d, Barray, BB*H*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(Carray_d, Carray, BB*H*sizeof(float*), cudaMemcpyHostToDevice);
    if(transpose){
      //stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3000, 3000, 4000, &alpha, Barray_d, 4000, Aarray_d, 4000, &beta, Carray_d, 3000, BB*H);
      double start, end;
      start = getTimeStamp();
      cudaDeviceSynchronize();
      stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, Barray_d, K , Aarray_d, K, &beta, Carray_d, M, BB*H);
      cudaDeviceSynchronize();
      end = getTimeStamp();
      printf("Time taken for batched cublas: %f\n", end-start);
    }
    else{
      double start, end;
      start = getTimeStamp();
      cudaDeviceSynchronize();
      stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, Barray_d, M, Aarray_d, K, &beta, Carray_d, M, 8);
      cudaDeviceSynchronize();
      end = getTimeStamp();
      printf("Time taken for batched cublas: %f\n", end-start);}
}

void run_sgemm_naive(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS));
    dim3 blockDim(BLKS,BLKS);

    // loop over batchsize and head
    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            // get the i-th batch and j-th head
            float* Aij = &A[i*H*K*N+j*K*N];
            float* Bij = &B[i*H*K*N+j*K*N];
            float* Cij = &C[i*H*K*N+j*K*N];
            // compute the matrix multiplication
            sgemm_naive<<<gridDim, blockDim>>>(N, M, K, Aij, Bij, Cij);
            // allocate memory for output on GPU in cuda
        }
    }

}

void run_sgemm_naive_batched(float* A, float* B, float* C, int M, int N, int K){
    
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS), CEIL_DIV(BB*H, BD));
    dim3 blockDim(BLKS,BLKS,BD);
    int L = BB*H;
    sgemm_naive_batched<<<gridDim, blockDim>>>(L, N, M, K, A, B, C);
    return;

}

void run_sgemm_coalesced(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS));
    dim3 blockDim(BLKS,BLKS);

    // loop over batchsize and head
    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            // get the i-th batch and j-th head
            float* Aij = &A[i*H*K*N+j*K*N];
            float* Bij = &B[i*H*K*N+j*K*N];
            float* Cij = &C[i*H*K*N+j*K*N];
            // compute the matrix multiplication
            sgemm_naive_coalesced<<<gridDim, blockDim>>>(N, M, K, Aij, Bij, Cij);
            // allocate memory for output on GPU in cuda
        }
    }
}

void run_sgemm_coalesced_batched(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS), CEIL_DIV(BB*H,BD));
    dim3 blockDim(BLKS,BLKS,BD);
    int L = BB*H;
    sgemm_naive_coalesced_batched<<<gridDim, blockDim>>>(L, N, M, K, A, B, C);
    return;

}

void run_sgemm_coalesced_tiled(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS));
    dim3 blockDim(BLKS,BLKS);

    // loop over batchsize and head
    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            // get the i-th batch and j-th head
            float* Aij = &A[i*H*K*N+j*K*N];
            float* Bij = &B[i*H*K*N+j*K*N];
            float* Cij = &C[i*H*K*N+j*K*N];
            // compute the matrix multiplication
            sgemm_naive_coalesced_tiled<<<gridDim, blockDim>>>(N, M, K, Aij, Bij, Cij);
            // allocate memory for output on GPU in cuda
        }
    }

}

void run_sgemm_coalesced_tiled_batched(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BLKS), CEIL_DIV(N, BLKS), CEIL_DIV(BB*H,BD));
    dim3 blockDim(BLKS,BLKS,BD);
    int L = BB*H;
    sgemm_naive_coalesced_tiled_batched<<<gridDim, blockDim>>>(L, N, M, K, A, B, C);
    return;
}

void run_sgemm_blocktiling(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BM));
    dim3 blockDim(CEIL_DIV(BM * BN, (TM * TN)));

    // loop over batchsize and head
    for (int i = 0; i < BB; i++) {
        for (int j = 0; j < H; j++) {
            // get the i-th batch and j-th head
            float* Aij = &A[i*H*K*N+j*K*N];
            float* Bij = &B[i*H*K*N+j*K*N];
            float* Cij = &C[i*H*K*N+j*K*N];
            // compute the matrix multiplication
            sgemm2DBlocktiling<<<gridDim, blockDim>>>(N, M, K, Aij, Bij, Cij);
            // allocate memory for output on GPU in cuda
        }
    }
}

void run_sgemm_blocktiling_batched(float* A, float* B, float* C, int M, int N, int K){
    dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BM), CEIL_DIV(BB*H,BD));
    dim3 blockDim((BM * BN)/(TM * TN), BD);
    int L = BB*H;
    sgemm2DBlocktiling_batched<<<gridDim, blockDim>>>(L, N, M, K, A, B, C);
    return;
    }


int main(){

    int N = 4096; // number of rows in dataset
    int M = 4096; // number of columns in dataset
    int K = 64;
    // A is N*K, B is K*M
    std::vector<float> A(BB * H * N * K,1.0);
    std::vector<float> B(BB * H * K * M,1.0);
    std::vector<float> C(BB * H * N * M,0.0);
    randomInit(A);
    randomInit(B);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, BB * H * N * K * sizeof(float));
    cudaMalloc(&d_B, BB * H * K * M * sizeof(float));
    cudaMalloc(&d_C, BB * H * N * M * sizeof(float));
    cudaMemset(d_C, 0, N*sizeof(float));

    cudaMemcpy(d_A,  A.data(),  BB * H * N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,  B.data(),  BB * H * K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    double start, end;
    start = getTimeStamp();
    
    // select which kernel run fucntion to use
    run_sgemm_cublas_batched(d_A,d_B,d_C,M,N,K,false);
    
    cudaDeviceSynchronize();
    end = getTimeStamp();
    std::cout << "Time taken by naive kernel: " << end - start << std::endl;


    cudaMemcpy(C.data(),d_C, BB * H * N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
