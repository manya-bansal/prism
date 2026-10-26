template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN), 1) void my_sgemm_kernel_4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C, int num_blocks_N) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    const int cRow = (__block_id_0 / num_blocks_N);
    const int cCol = (__block_id_0 % num_blocks_N);
    const unsigned int totalResultsBlocktile = (BM * BN);
    const unsigned int numThreadsBlocktile = (totalResultsBlocktile / (TM * TN));
    {
        float* C_blk = C + ((((cRow * BM) * N) + (cCol * BN)));
        {
            float* A_blk = A + (((cRow * BM) * K));
            {
                float* B_blk = B + ((cCol * BN));
                {
                    int __block_id_1 = __block_id_0 % 1;
                    int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
                    float (&As)[1024] = *((float (*)[1024])(__smem + 0));
                    float (&Bs)[1024] = *((float (*)[1024])(__smem + 4096));
                    const int threadRow = (__thread_id_1 / (BN / TN));
                    const int threadCol = (__thread_id_1 % (BN / TN));
                    const int innerRowA = (__thread_id_1 / (BK / 4));
                    const int innerColA = (__thread_id_1 % (BK / 4));
                    const int strideA = (numThreadsBlocktile / BK);
                    const int innerRowB = (__thread_id_1 / (BN / 4));
                    const int innerColB = (__thread_id_1 % (BN / 4));
                    const int strideB = (numThreadsBlocktile / BN);
                    float threadResults[64] = {0};
                    float regM[8] = {0};
                    float regN[8] = {0};
                    int i = 0;
                    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
                        {
                            float* As_thrd = As + (0);
                            {
                                float* Bs_thrd = Bs + (0);
                                {
                                    int __block_id_2 = __block_id_1;
                                    int __thread_id_2 = __thread_id_1 % 1;
                                    float4 tmp = reinterpret_cast<float4 *>(&A_blk[(((i * BK) + (innerRowA * K)) + (innerColA * 4))])[0];
                                    As_thrd[((((innerColA * 4) + 0) * BM) + innerRowA)] = tmp.x;
                                    As_thrd[((((innerColA * 4) + 1) * BM) + innerRowA)] = tmp.y;
                                    As_thrd[((((innerColA * 4) + 2) * BM) + innerRowA)] = tmp.z;
                                    As_thrd[((((innerColA * 4) + 3) * BM) + innerRowA)] = tmp.w;
                                    reinterpret_cast<float4 *>(&Bs_thrd[((innerRowB * BN) + (innerColB * 4))])[0] = reinterpret_cast<float4 *>(&B_blk[((((i * BK) * N) + (innerRowB * N)) + (innerColB * 4))])[0];
                                    i += 1;
                                    __syncthreads();
                                    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                                        for (int i = 0; i < TN; i++) {
                                            regM[i] = As_thrd[(((dotIdx * BM) + (threadRow * TM)) + i)];
                                        }
                                        for (int i = 0; i < TN; i++) {
                                            regN[i] = Bs_thrd[(((dotIdx * BN) + (threadCol * TN)) + i)];
                                        }
                                        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                                            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                                                threadResults[((resIdxM * TN) + resIdxN)] += (regM[resIdxM] * regN[resIdxN]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                    }
                    for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                            {
                                float* C_thrd = C_blk + ((((((threadRow * TM) + resIdxM) * N) + (threadCol * TN)) + resIdxN));
                                {
                                    int __block_id_2 = __block_id_1;
                                    int __thread_id_2 = __thread_id_1 % 1;
                                    float4 temp = reinterpret_cast<float4 *>(&C_thrd[0])[0];
                                    temp.x = ((alpha * threadResults[((resIdxM * TN) + resIdxN)]) + (beta * temp.x));
                                    temp.y = ((alpha * threadResults[(((resIdxM * TN) + resIdxN) + 1)]) + (beta * temp.y));
                                    temp.z = ((alpha * threadResults[(((resIdxM * TN) + resIdxN) + 2)]) + (beta * temp.z));
                                    temp.w = ((alpha * threadResults[(((resIdxM * TN) + resIdxN) + 3)]) + (beta * temp.w));
                                    reinterpret_cast<float4 *>(&C_thrd[0])[0] = temp;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
