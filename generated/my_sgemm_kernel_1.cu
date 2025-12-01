__global__ void my_sgemm_kernel_1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, int block_size) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    int bid = __block_id_0;
    {
        float* C_blk = C + (0);
        {
            int __block_id_1 = __block_id_0 % 1;
            int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
            int tid = ((bid * block_size) + __thread_id_1);
            {
                float* C_thrd = C_blk + (0);
                const int x = (tid / N);
                const int y = (tid % N);
                {
                    int __block_id_2 = __block_id_1;
                    int __thread_id_2 = __thread_id_1 % 1;
                    float tmp = 0;
                    for (int i = 0; i < K; i++) {
                        tmp += (A[((x * K) + i)] * B[((i * N) + y)]);
                    }
                    C_thrd[((x * N) + y)] = ((alpha * tmp) + (beta * C[((x * N) + y)]));
                }
            }
        }
    }
}
