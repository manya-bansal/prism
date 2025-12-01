template<int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, int num_blocks_N) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    const unsigned int cRow = (__block_id_0 / num_blocks_N);
    const unsigned int cCol = (__block_id_0 % num_blocks_N);
    {
        const float* A_blk = A + (((cRow * BLOCKSIZE) * K));
        {
            const float* B_blk = B + ((cCol * BLOCKSIZE));
            {
                float* C_blk = C + ((((cRow * BLOCKSIZE) * N) + (cCol * BLOCKSIZE)));
                {
                    int __block_id_1 = __block_id_0 % 1;
                    int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
                    __shared__ float As[(BLOCKSIZE * BLOCKSIZE)];
                    __shared__ float Bs[(BLOCKSIZE * BLOCKSIZE)];
                    const unsigned int threadCol = (__thread_id_1 % BLOCKSIZE);
                    const unsigned int threadRow = (__thread_id_1 / BLOCKSIZE);
                    float tmp = 0;
                    {
                        float* As_t = As + (0);
                        {
                            float* Bs_t = Bs + (0);
                            {
                                float* C_t = C_blk + (0);
                                {
                                    int __block_id_2 = __block_id_1;
                                    int __thread_id_2 = __thread_id_1 % 1;
                                    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
                                        As_t[((threadRow * BLOCKSIZE) + threadCol)] = A_blk[((bkIdx + (threadRow * K)) + threadCol)];
                                        Bs_t[((threadRow * BLOCKSIZE) + threadCol)] = B_blk[(((bkIdx + threadRow) * N) + threadCol)];
                                        __syncthreads();
                                        for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
                                            tmp += (As_t[((threadRow * BLOCKSIZE) + dotIdx)] * Bs_t[((dotIdx * BLOCKSIZE) + threadCol)]);
                                        }
                                        __syncthreads();
                                    }
                                    C_t[((threadRow * N) + threadCol)] = ((alpha * tmp) + (beta * C_t[((threadRow * N) + threadCol)]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
