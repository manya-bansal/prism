template<int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void my_sgemm_kernel_5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C, int num_blocks_N) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    const int cRow = (__block_id_0 / num_blocks_N);
    const int cCol = (__block_id_0 % num_blocks_N);
    constexpr unsigned int WMITER = ((WM * WN) / (((32 * TM) * TN) * WNITER));
    constexpr unsigned int WSUBM = (WM / WMITER);
    constexpr unsigned int WSUBN = (WN / WNITER);
    constexpr int rowStrideA = ((NUM_THREADS * 4) / BK);
    constexpr int rowStrideB = (NUM_THREADS / (BN / 4));
    {
        float* C_blk = C + ((((cRow * BM) * N) + (cCol * BN)));
        {
            float* A_blk = A + (((cRow * BM) * K));
            {
                float* B_blk = B + ((cCol * BN));
                {
                    int __block_id_1 = __block_id_0 % 1;
                    int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
                    const unsigned int warpIdx = (__thread_id_1 / 32);
                    const unsigned int warpCol = (warpIdx % (BN / WN));
                    const unsigned int warpRow = (warpIdx / (BN / WN));
                    float (&As)[2048] = *((float (*)[2048])(__smem + 0));
                    float (&Bs)[2048] = *((float (*)[2048])(__smem + 8192));
                    const int innerRowA = (__thread_id_1 / (BK / 4));
                    const int innerColA = (__thread_id_1 % (BK / 4));
                    const int innerRowB = (__thread_id_1 / (BN / 4));
                    const int innerColB = (__thread_id_1 % (BN / 4));
                    float threadResults[128] = {0};
                    float regM[8] = {0};
                    float regN[16] = {0};
                    int j = 0;
                    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
                        {
                            float* A_offset = A_blk + ((j * BK));
                            {
                                float* B_offset = B_blk + (((j * BK) * N));
                                loadFromGmem_our<BM, BN, BK, rowStrideA, rowStrideB>(__block_id_1, __thread_id_1, __smem, N, K, A_offset, B_offset, As, Bs);
                                __syncthreads();
                                processFromSmem_our<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(__block_id_1, __thread_id_1, __smem, regM, regN, threadResults, As, Bs, warpRow, warpCol);
                                j += 1;
                            }
                        }
                        __syncthreads();
                    }
                    {
                        float* C_warp = C_blk + ((((warpRow * WM) * N) + (warpCol * WN)));
                        {
                            int __block_id_2 = __block_id_1;
                            int __thread_id_2 = __thread_id_1 % 32;
                            int threadIdxInWarp = __thread_id_2;
                            const unsigned int threadColInWarp = (threadIdxInWarp % (WSUBN / TN));
                            const unsigned int threadRowInWarp = (threadIdxInWarp / (WSUBN / TN));
                            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                                for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                                    {
                                        float* C_interim = C_warp + ((((wSubRowIdx * WSUBM) * N) + (wSubColIdx * WSUBN)));
                                        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                                            for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                                                {
                                                    int __block_id_3 = __block_id_2;
                                                    int __thread_id_3 = __thread_id_2 % 1;
                                                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(((((threadRowInWarp * TM) + resIdxM) * N) + (threadColInWarp * TN)) + resIdxN)])[0];
                                                    const int i = (((((wSubRowIdx * TM) + resIdxM) * (WNITER * TN)) + (wSubColIdx * TN)) + resIdxN);
                                                    tmp.x = ((alpha * threadResults[(i + 0)]) + (beta * tmp.x));
                                                    tmp.y = ((alpha * threadResults[(i + 1)]) + (beta * tmp.y));
                                                    tmp.z = ((alpha * threadResults[(i + 2)]) + (beta * tmp.z));
                                                    tmp.w = ((alpha * threadResults[(i + 3)]) + (beta * tmp.w));
                                                    reinterpret_cast<float4 *>(&C_interim[(((((threadRowInWarp * TM) + resIdxM) * N) + (threadColInWarp * TN)) + resIdxN)])[0] = tmp;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
