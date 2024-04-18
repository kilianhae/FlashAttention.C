# define B_r 32
# define B_c 32
# define o_per_thread_x 1
# define o_per_thread_y 32/32
# define d 32

#define NEG_INFINITY __int_as_float(0xff800000)

__global__
void silly_attn(float *out, float* out_l, float *K, float *Q, float* V, float scaling, int batch_stride, int T_r, int T_c)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int batch_offset = batch_stride * blockIdx.x;

  __shared__ float Q_i[B_r][d]; // all are fully loaded into shared memory
  __shared__ float K_j[B_c][d];
  __shared__ float V_j[B_c][d];
  
  // attention result
  __shared__ float S_i[B_r][B_c];

  float l_i[o_per_thread_x];
  float m_i[o_per_thread_x];

  // this will be automatucally be put onto registers (potentially slow if doesnt fit onto regsters and spills to local memory, prob no way around it)
  float O_i[o_per_thread_x][o_per_thread_y];

  for (int i = 0; i < T_r; i++) {
    for (int ii = 0; ii < o_per_thread_x; ii++) {
      for (int dd = 0; dd < o_per_thread_y; dd++) {
        O_i[ii][dd] = 0;
      }
      l_i[ii] = 0.f;
      m_i[ii] = NEG_INFINITY;
    }
    for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
      for (int dd = tid_x; dd < d; dd += blockDim.x) {
         Q_i[ii][dd] = Q[batch_offset + (ii + i * B_r) * d + dd];
      }
    }
    for (int j=0; j < T_c; j++) {
        __syncthreads();
        for (int jj=tid_y; jj < B_c; jj+= blockDim.y) {
            for (int dd=tid_x; dd < d; dd += blockDim.x) {
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
    for (int ii = 0; ii < o_per_thread_x; ii++) {
      for (int dd = 0; dd < o_per_thread_y; dd++) {
        out[batch_offset + (ii * blockDim.x + tid_x + i * B_r) * d + dd * blockDim.y + tid_y] = O_i[ii][dd] / l_i[ii];
        out_l[batch_offset / d +  ii * blockDim.x + tid_x + i * B_r] = l_i[ii];
      }
    }
  }
}


// write main function that takes two command line integer arguments
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <batch_size> <seq_len>" << std::endl;
    return 1;
  }
  int batch_size = std::stoi(argv[1]);
  int seq_len = std::stoi(argv[2]);
  

  float* Q_h = (float*) malloc(batch_size * seq_len * d * sizeof(float));
  float* K_h = (float*) malloc(batch_size * seq_len * d * sizeof(float));
  float* V_h = (float*) malloc(batch_size * seq_len * d * sizeof(float));
  float* out_h = (float*) malloc(batch_size * seq_len * d * sizeof(float));
  float* out_l_h = (float*) malloc(batch_size * seq_len * sizeof(float));

  float* Q_d, *K_d, *V_d, *out_d, *out_l_d;
  
  // init Q,K,V as ones
  for (int i = 0; i < batch_size * seq_len * d; i++) {
    Q_h[i] = 1.f;
    K_h[i] = 1.f;
    V_h[i] = 1.f;
  }

  // allocat GPU memory
  cudaMalloc(&Q_d, batch_size * seq_len * d * sizeof(float));
  cudaMalloc(&K_d, batch_size * seq_len * d * sizeof(float));
  cudaMalloc(&V_d, batch_size * seq_len * d * sizeof(float));
  cudaMalloc(&out_d, batch_size * seq_len * d * sizeof(float));
  cudaMalloc(&out_l_d, batch_size * seq_len * sizeof(float));

  // copy to device
  cudaMemcpy(Q_d, Q_h, batch_size * seq_len * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(K_d, K_h, batch_size * seq_len * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(V_d, V_h, batch_size * seq_len * d * sizeof(float), cudaMemcpyHostToDevice);

  // this kernel processes a whole Attention head in one block:
  dim3 block(B_r, B_c); // not fully sure what the blocksize is
  dim3 grid(batch_size);

  // call kernel
  silly_attn<<<grid, block>>>(out_d, out_l_d, K_d, Q_d, V_d, 1.f / sqrtf(d), seq_len * d, seq_len, seq_len);
  
  // copy back
  cudaMemcpy(out_h, out_d, batch_size * seq_len * d * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(out_l_h, out_l_d, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(Q_d);
  cudaFree(K_d);
  cudaFree(V_d);
  cudaFree(out_d);
  cudaFree(out_l_d);
}
    





