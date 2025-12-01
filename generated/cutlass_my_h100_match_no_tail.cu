__global__ __launch_bounds__(128*3) void cutlass_my_h100_match_no_tail(int M, int N, int K, __nv_bfloat16* C, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    {
        int __block_id_1 = __block_id_0;
        int __thread_id_1 = __thread_id_0;
        constexpr int BM = 128;
        constexpr int BN = 256;
        constexpr int BK = 64;
        constexpr int NUM_THREADS = (128 * 3);
        constexpr int QSIZE = 4;
        constexpr int NUM_SM = 128;
        constexpr int WGMMA_M = 64;
        constexpr int WGMMA_K = 16;
        constexpr int WGMMA_N = BN;
        constexpr int num_consumers = ((NUM_THREADS / 128) - 1);
        constexpr int B_WG_M = (BM / num_consumers);
        constexpr int TM = 16;
        constexpr int TN = 8;
        int schedule_block = __block_id_1;
        {
            __nv_bfloat16* block_C = C + (0);
            {
                int __block_id_2 = __block_id_1 % 1;
                int __thread_id_2 = __block_id_2 * blockDim.x + threadIdx.x;
                __align__(128) __nv_bfloat16 (&sA_slot0)[(64 * 128)] = *((__nv_bfloat16 (*)[(64 * 128)])(__smem + 0));
                __nv_bfloat16 (&sA_slot1)[(64 * 128)] = *((__nv_bfloat16 (*)[(64 * 128)])(__smem + 16384));
                __nv_bfloat16 (&sA_slot2)[(64 * 128)] = *((__nv_bfloat16 (*)[(64 * 128)])(__smem + 32768));
                __nv_bfloat16 (&sA_slot3)[(64 * 128)] = *((__nv_bfloat16 (*)[(64 * 128)])(__smem + 49152));
                __align__(128) __nv_bfloat16 (&sB_slot0)[(64 * 256)] = *((__nv_bfloat16 (*)[(64 * 256)])(__smem + 65536));
                __nv_bfloat16 (&sB_slot1)[(64 * 256)] = *((__nv_bfloat16 (*)[(64 * 256)])(__smem + 98304));
                __nv_bfloat16 (&sB_slot2)[(64 * 256)] = *((__nv_bfloat16 (*)[(64 * 256)])(__smem + 131072));
                __nv_bfloat16 (&sB_slot3)[(64 * 256)] = *((__nv_bfloat16 (*)[(64 * 256)])(__smem + 163840));
                const int num_blocks_k = (K / BK);
                int wg_idx = (__thread_id_2 / 128);
                int blk_thrd_id = __thread_id_2;
                int tid = (__thread_id_2 % 128);
                int is_producer = (wg_idx == 0);
                int num_block_m = 0;
                int num_block_n = 0;
                {
                    int __block_id_3 = __block_id_2;
                    int __thread_id_3 = __thread_id_2 % 128;
                    if (is_producer) {
                    } else {
                        wg_idx = (wg_idx - 1);
                    }
                }
                int schedule_it = 0;
                int total_blocks_m = (((M + BM) - 1) / BM);
                int total_blocks_n = (((N + BN) - 1) / BN);
                assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);
                __shared__ __align__(8) uint64_t full_0[1];
                __shared__ __align__(8) uint64_t full_1[1];
                __shared__ __align__(8) uint64_t full_2[1];
                __shared__ __align__(8) uint64_t full_3[1];
                __shared__ __align__(8) uint64_t empty_0[1];
                __shared__ __align__(8) uint64_t empty_1[1];
                __shared__ __align__(8) uint64_t empty_2[1];
                __shared__ __align__(8) uint64_t empty_3[1];
                int p0 = 0;
                int p1 = 0;
                int p2 = 0;
                int p3 = 0;
                if (threadIdx.x == 0) {
                    init_barrier(__block_id_2, __thread_id_2, __smem, full_0, 0, 1);
                    init_barrier(__block_id_2, __thread_id_2, __smem, full_1, 0, 1);
                    init_barrier(__block_id_2, __thread_id_2, __smem, full_2, 0, 1);
                    init_barrier(__block_id_2, __thread_id_2, __smem, full_3, 0, 1);
                    init_barrier(__block_id_2, __thread_id_2, __smem, empty_0, 0, 2);
                    init_barrier(__block_id_2, __thread_id_2, __smem, empty_1, 0, 2);
                    init_barrier(__block_id_2, __thread_id_2, __smem, empty_2, 0, 2);
                    init_barrier(__block_id_2, __thread_id_2, __smem, empty_3, 0, 2);
                }
                __syncthreads();
                {
                    int __wg = (__thread_id_2 / 128);
                    if (__wg > 0) {
                        int __lane = __thread_id_2 % 128;
                        if (__lane == 0) {
                            arrive(__block_id_2, __thread_id_2, __smem, empty_0, 1);
                            arrive(__block_id_2, __thread_id_2, __smem, empty_1, 1);
                            arrive(__block_id_2, __thread_id_2, __smem, empty_2, 1);
                            arrive(__block_id_2, __thread_id_2, __smem, empty_3, 1);
                        }
                    }
                }
                __syncthreads();
                while (true) {
                    int num = ((schedule_it * NUM_SM) + schedule_block);
                    if ((num >= (total_blocks_m * total_blocks_n))) {
                        break;
                    }
                    int cur_tile = (num / (TM * TN));
                    int cur_tile_pos = (num % (TM * TN));
                    num_block_m = (TM * (cur_tile / (total_blocks_n / TN)));
                    num_block_n = (TN * (cur_tile % (total_blocks_n / TN)));
                    num_block_m += (cur_tile_pos / TN);
                    num_block_n += (cur_tile_pos % TN);
                    schedule_it += 1;
                    float d[1][16][8];
                    {
                        int __block_id_3 = __block_id_2;
                        int __thread_id_3 = __thread_id_2 % 1;
                        float* d_ptr;
                        d_ptr = (float *)d;
                        memset_wrapper(__block_id_3, __thread_id_3, __smem, d_ptr, 0, (((1 * 16) * 8) * 4));
                    }
                    {
                        int __block_id_3 = __block_id_2;
                        int __thread_id_3 = __thread_id_2 % 128;
                        if (is_producer) {
                            constexpr int num_regs = ((num_consumers <= 2) ? 24 : 32);
                            warpgroup_reg_dealloc<num_regs>(__block_id_3, __thread_id_3, __smem);
                        } else {
                            constexpr int num_regs_p = ((num_consumers == 1) ? 256 : ((num_consumers == 2) ? 240 : 160));
                            warpgroup_reg_alloc<num_regs_p>(__block_id_3, __thread_id_3, __smem);
                        }
                    }
                    const int num_blocks_k_main = ((num_blocks_k / 4) * 4);
                    for (int block_k_iter = 0; block_k_iter < num_blocks_k_main; block_k_iter += 4) {
                        {
                            __nv_bfloat16* sA_new = sA_slot0 + (0);
                            {
                                __nv_bfloat16* sA_new1 = sA_slot1 + (0);
                                {
                                    __nv_bfloat16* sA_new2 = sA_slot2 + (0);
                                    {
                                        __nv_bfloat16* sB_new = sB_slot0 + (0);
                                        {
                                            __nv_bfloat16* sB_new1 = sB_slot1 + (0);
                                            {
                                                __nv_bfloat16* sB_new2 = sB_slot2 + (0);
                                                {
                                                    __nv_bfloat16* sA_new3 = sA_slot3 + (0);
                                                    {
                                                        __nv_bfloat16* sB_new3 = sB_slot3 + (0);
                                                        {
                                                            int __block_id_3 = __block_id_2;
                                                            int __thread_id_3 = __thread_id_2;
                                                            if (__thread_id_3 >= 0 && __thread_id_3 < 128) {
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3;
                                                                    if (__thread_id_4 >= 0 && __thread_id_4 < 1) {
                                                                        const void* ta_void_ptr;
                                                                        ta_void_ptr = &tensorMapA;
                                                                        wait(__block_id_4, __thread_id_4, __smem, empty_0, p0);
                                                                        expect_bytes(__block_id_4, __thread_id_4, __smem, full_0, 49152);
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sA_new, ta_void_ptr, full_0, ((block_k_iter + 0) * BK), (num_block_m * BM));
                                                                        const void* tb_void_ptr;
                                                                        tb_void_ptr = &tensorMapB;
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sB_new, tb_void_ptr, full_0, ((block_k_iter + 0) * BK), (num_block_n * BN));
                                                                        wait(__block_id_4, __thread_id_4, __smem, empty_1, p1);
                                                                        expect_bytes(__block_id_4, __thread_id_4, __smem, full_1, 49152);
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sA_new1, ta_void_ptr, full_1, ((block_k_iter + 1) * BK), (num_block_m * BM));
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sB_new1, tb_void_ptr, full_1, ((block_k_iter + 1) * BK), (num_block_n * BN));
                                                                        wait(__block_id_4, __thread_id_4, __smem, empty_2, p2);
                                                                        expect_bytes(__block_id_4, __thread_id_4, __smem, full_2, 49152);
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sA_new2, ta_void_ptr, full_2, ((block_k_iter + 2) * BK), (num_block_m * BM));
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sB_new2, tb_void_ptr, full_2, ((block_k_iter + 2) * BK), (num_block_n * BN));
                                                                        wait(__block_id_4, __thread_id_4, __smem, empty_3, p3);
                                                                        expect_bytes(__block_id_4, __thread_id_4, __smem, full_3, 49152);
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sA_new3, ta_void_ptr, full_3, ((block_k_iter + 3) * BK), (num_block_m * BM));
                                                                        load_async(__block_id_4, __thread_id_4, __smem, sB_new3, tb_void_ptr, full_3, ((block_k_iter + 3) * BK), (num_block_n * BN));
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
                        {
                            __nv_bfloat16* wgmma_sA = sA_slot0 + (0);
                            {
                                __nv_bfloat16* sA_1_producer = sA_slot1 + (0);
                                {
                                    __nv_bfloat16* sA_2_producer = sA_slot2 + (0);
                                    {
                                        __nv_bfloat16* wgmma_sB = sB_slot0 + (0);
                                        {
                                            __nv_bfloat16* sB_1_producer = sB_slot1 + (0);
                                            {
                                                __nv_bfloat16* sB_2_producer = sB_slot2 + (0);
                                                {
                                                    __nv_bfloat16* sA_3_producer = sA_slot3 + (0);
                                                    {
                                                        __nv_bfloat16* sB_3_producer = sB_slot3 + (0);
                                                        {
                                                            int __block_id_3 = __block_id_2;
                                                            int __thread_id_3 = __thread_id_2;
                                                            if (__thread_id_3 >= 0 && __thread_id_3 < 128) {
                                                            }
                                                        }
                                                        {
                                                            int __block_id_3 = __block_id_2;
                                                            int __thread_id_3 = __thread_id_2 - 128;
                                                            if (__thread_id_3 >= 0 && __thread_id_3 < 256) {
                                                                wait_cluster(__block_id_3, __thread_id_3, __smem, full_0, p0);
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    warpgroup_arrive(__block_id_4, __thread_id_4, __smem);
                                                                }
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    {
                                                                        for (int m_it = 0; m_it < (B_WG_M / WGMMA_M); m_it++) {
                                                                            int index_a = (m_it * BM);
                                                                            int index_b = (m_it * BN);
                                                                            #pragma unroll
                                                                            for (int k_it = 0; k_it < (64 / WGMMA_K); k_it++) {
                                                                                wgmma<256, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[((((64 * (m_it + ((wg_idx * B_WG_M) / WGMMA_M))) * WGMMA_M) + index_a) + (k_it * WGMMA_K))], &wgmma_sB[(index_b + (k_it * WGMMA_K))]);
                                                                            }
                                                                        }
                                                                        warpgroup_commit_batch();
                                                                        warpgroup_wait<0>();
                                                                    }
                                                                }
                                                                if (__thread_id_3 % 128 == 0) arrive(__block_id_3, __thread_id_3, __smem, empty_0, 1);
                                                                wait_cluster(__block_id_3, __thread_id_3, __smem, full_1, p1);
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    warpgroup_arrive(__block_id_4, __thread_id_4, __smem);
                                                                }
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    {
                                                                        for (int m_it = 0; m_it < (B_WG_M / WGMMA_M); m_it++) {
                                                                            int index_a2 = (m_it * BM);
                                                                            int index_b2 = (m_it * BN);
                                                                            #pragma unroll
                                                                            for (int k_it = 0; k_it < (64 / WGMMA_K); k_it++) {
                                                                                wgmma<256, 1, 1, 1, 0, 0>(d[m_it], &sA_1_producer[((((64 * (m_it + ((wg_idx * B_WG_M) / WGMMA_M))) * WGMMA_M) + index_a2) + (k_it * WGMMA_K))], &sB_1_producer[(index_b2 + (k_it * WGMMA_K))]);
                                                                            }
                                                                        }
                                                                        warpgroup_commit_batch();
                                                                        warpgroup_wait<0>();
                                                                    }
                                                                }
                                                                if (__thread_id_3 % 128 == 0) arrive(__block_id_3, __thread_id_3, __smem, empty_1, 1);
                                                                wait_cluster(__block_id_3, __thread_id_3, __smem, full_2, p2);
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    warpgroup_arrive(__block_id_4, __thread_id_4, __smem);
                                                                }
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    {
                                                                        for (int m_it = 0; m_it < (B_WG_M / WGMMA_M); m_it++) {
                                                                            int index_a3 = (m_it * BM);
                                                                            int index_b3 = (m_it * BN);
                                                                            #pragma unroll
                                                                            for (int k_it = 0; k_it < (64 / WGMMA_K); k_it++) {
                                                                                wgmma<256, 1, 1, 1, 0, 0>(d[m_it], &sA_2_producer[((((64 * (m_it + ((wg_idx * B_WG_M) / WGMMA_M))) * WGMMA_M) + index_a3) + (k_it * WGMMA_K))], &sB_2_producer[(index_b3 + (k_it * WGMMA_K))]);
                                                                            }
                                                                        }
                                                                        warpgroup_commit_batch();
                                                                        warpgroup_wait<0>();
                                                                    }
                                                                }
                                                                if (__thread_id_3 % 128 == 0) arrive(__block_id_3, __thread_id_3, __smem, empty_2, 1);
                                                                wait_cluster(__block_id_3, __thread_id_3, __smem, full_3, p3);
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    warpgroup_arrive(__block_id_4, __thread_id_4, __smem);
                                                                }
                                                                {
                                                                    int __block_id_4 = __block_id_3;
                                                                    int __thread_id_4 = __thread_id_3 % 128;
                                                                    {
                                                                        for (int m_it = 0; m_it < (B_WG_M / WGMMA_M); m_it++) {
                                                                            int index_a4 = (m_it * BM);
                                                                            int index_b4 = (m_it * BN);
                                                                            #pragma unroll
                                                                            for (int k_it = 0; k_it < (64 / WGMMA_K); k_it++) {
                                                                                wgmma<256, 1, 1, 1, 0, 0>(d[m_it], &sA_3_producer[((((64 * (m_it + ((wg_idx * B_WG_M) / WGMMA_M))) * WGMMA_M) + index_a4) + (k_it * WGMMA_K))], &sB_3_producer[(index_b4 + (k_it * WGMMA_K))]);
                                                                            }
                                                                        }
                                                                        warpgroup_commit_batch();
                                                                        warpgroup_wait<0>();
                                                                    }
                                                                }
                                                                if (__thread_id_3 % 128 == 0) arrive(__block_id_3, __thread_id_3, __smem, empty_3, 1);
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
                        p0 ^= 1;
                        p1 ^= 1;
                        p2 ^= 1;
                        p3 ^= 1;
                    }
                    {
                        __nv_bfloat16* c_consumer = block_C + ((((num_block_n * BN) * M) + (num_block_m * BM)));
                        {
                            int __block_id_3 = __block_id_2;
                            int __thread_id_3 = __thread_id_2;
                            if (__thread_id_3 >= 0 && __thread_id_3 < 128) {
                            }
                        }
                        {
                            int __block_id_3 = __block_id_2;
                            int __thread_id_3 = __thread_id_2 - 128;
                            if (__thread_id_3 >= 0 && __thread_id_3 < 256) {
                                {
                                    __nv_bfloat16* c_thrd = c_consumer + (0);
                                    {
                                        int __block_id_4 = __block_id_3;
                                        int __thread_id_4 = __thread_id_3 % 128;
                                        int warp_id = (__thread_id_4 / 32);
                                        {
                                            int __block_id_5 = __block_id_4;
                                            int __thread_id_5 = __thread_id_4 % 32;
                                            int lane = __thread_id_5;
                                            int row = ((warp_id * 16) + (lane / 4));
                                            #pragma unroll
                                            for (int m_it = 0; m_it < (B_WG_M / WGMMA_M); m_it++) {
                                                int yo = ((m_it * WGMMA_M) + (wg_idx * B_WG_M));
                                                {
                                                    int __block_id_6 = __block_id_5;
                                                    int __thread_id_6 = __thread_id_5 % 1;
                                                    if ((((row + yo) + (num_block_m * BM)) >= M)) {
                                                        continue;
                                                    }
                                                    for (int w = 0; w < WGMMA_N; w += 16) {
                                                        if (((w < w) < (N - (num_block_n * BN)))) {
                                                            int col = (w + (2 * (tid % 4)));
                                                            c_thrd[(((col * M) + row) + yo)] = d[m_it][(w / 16)][0];
                                                            c_thrd[((((col + 1) * M) + row) + yo)] = d[m_it][(w / 16)][1];
                                                            c_thrd[(((col * M) + (row + 8)) + yo)] = d[m_it][(w / 16)][2];
                                                            c_thrd[((((col + 1) * M) + (row + 8)) + yo)] = d[m_it][(w / 16)][3];
                                                            c_thrd[((((col + 8) * M) + (row + 0)) + yo)] = d[m_it][(w / 16)][4];
                                                            c_thrd[((((col + 9) * M) + (row + 0)) + yo)] = d[m_it][(w / 16)][5];
                                                            c_thrd[((((col + 8) * M) + (row + 8)) + yo)] = d[m_it][(w / 16)][6];
                                                            c_thrd[((((col + 9) * M) + (row + 8)) + yo)] = d[m_it][(w / 16)][7];
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
    }
    return;
}
