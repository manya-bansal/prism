"""Port of cutlass_my_h100_match_no_tail from ccuda_py h100_matmul.py.

4-slot producer/consumer H100 matmul kernel using TMA, WGMMA, mbarrier sync.
Translates warp[N] -> thread[N*32]: warp[4]=thread[128], warp[8]=thread[256], warp[12]=thread[384].
"""

# Import helpers (registers them in the prism registry)
import examples.h100_helpers  # noqa: F401

from prism import (
    prism, grid, block, thread,
    requires, group, split, partition, claim,
    ptr, const, constexpr, shared, unsafe, attr,
    bf16, uint64_t, CUtensorMap,
    intrinsic, wgmma_async,
    generate,
)


@attr("__launch_bounds__(128*3)")
@requires(grid[1], block[1], thread[384], smem=227000)
@prism("global")
def cutlass_my_h100_match_no_tail(
        M: int @ grid[1],
        N: int @ grid[1],
        K: int @ grid[1],
        C: ptr(bf16) @ grid[1],
        tensorMapA: const(CUtensorMap) @ grid[1],
        tensorMapB: const(CUtensorMap) @ grid[1]):

    with group(grid[1]):
        BM: constexpr(int) @ grid[1] = 128
        BN: constexpr(int) @ grid[1] = 256
        BK: constexpr(int) @ grid[1] = 64
        NUM_THREADS: constexpr(int) @ grid[1] = 128 * 3
        QSIZE: constexpr(int) @ grid[1] = 4
        NUM_SM: constexpr(int) @ grid[1] = 128

        WGMMA_M: constexpr(int) @ grid[1] = 64
        WGMMA_K: constexpr(int) @ grid[1] = 16
        WGMMA_N: constexpr(int) @ grid[1] = BN
        num_consumers: constexpr(int) @ grid[1] = (NUM_THREADS / 128) - 1
        B_WG_M: constexpr(int) @ grid[1] = BM / num_consumers

        TM: constexpr(int) @ grid[1] = 16
        TN: constexpr(int) @ grid[1] = 8
        schedule_block: int @ block[1] = id()

        with partition(C, p=block[1], offset=0) as block_C:
            with group(block[1]):

                # Allocate sA[QSIZE][BK*BM]
                sA_slot0: shared(bf16[64 * 128], align=128) @ block[1]
                sA_slot1: shared(bf16[64 * 128]) @ block[1]
                sA_slot2: shared(bf16[64 * 128]) @ block[1]
                sA_slot3: shared(bf16[64 * 128]) @ block[1]

                # Allocate sB[QSIZE][BK*BN]
                sB_slot0: shared(bf16[64 * 256], align=128) @ block[1]
                sB_slot1: shared(bf16[64 * 256]) @ block[1]
                sB_slot2: shared(bf16[64 * 256]) @ block[1]
                sB_slot3: shared(bf16[64 * 256]) @ block[1]

                # Inferred by codegen: barrier decls, phase vars, init, wait, arrive
                # full_0..3: shared(uint64_t[1], align=8, static=True) @ block[1]
                # empty_0..3: shared(uint64_t[1], align=8, static=True) @ block[1]
                # p0..p3: int @ thread[256] = 0

                num_blocks_k: const(int) @ block[1] = K / BK
                # warp[4] = thread[128]
                wg_idx: int @ thread[128] = id()
                blk_thrd_id: int @ thread[1] = id()

                # # Initialize barriers
                # with partition(full_0, p=thread[1], offset=0) as full_0_thrd:
                #     with partition(full_1, p=thread[1], offset=0) as full_1_thrd:
                #         with partition(full_2, p=thread[1], offset=0) as full_2_thrd:
                #             with partition(empty_0, p=thread[1], offset=0) as empty_0_thrd:
                #                 with partition(empty_1, p=thread[1], offset=0) as empty_1_thrd:
                #                     with partition(empty_2, p=thread[1], offset=0) as empty_2_thrd:
                #                         with group(thread[1]):
                #                             if (blk_thrd_id == 0):
                #                                 init_barrier(full_0_thrd, 0, 1)
                #                                 init_barrier(full_1_thrd, 0, 1)
                #                                 init_barrier(full_2_thrd, 0, 1)
                #                                 unsafe("init_barrier(full_3, 0, 1);")
                #                                 init_barrier(empty_0_thrd, 0, num_consumers)
                #                                 init_barrier(empty_1_thrd, 0, num_consumers)
                #                                 init_barrier(empty_2_thrd, 0, num_consumers)
                #                                 unsafe("init_barrier(empty_3, 0, num_consumers);")

                # unsafe("__syncthreads();")
                tid: int @ thread[1] = id() % 128
                is_producer: int @ thread[128] = wg_idx == 0

                num_block_m: int @ block[1] = 0
                num_block_n: int @ block[1] = 0

                # Consumer wg_idx adjustment: consumers use 0-indexed wg_idx
                with group(thread[128]):
                    if (is_producer):
                        pass
                    else:
                        wg_idx = wg_idx - 1

                schedule_it: int @ block[1] = 0
                total_blocks_m: int @ block[1] = (((M + BM) - 1) / BM)
                total_blocks_n: int @ block[1] = (((N + BN) - 1) / BN)
                unsafe("assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);")

                while (True):
                    num: int @ block[1] = ((schedule_it * NUM_SM) + schedule_block)
                    if (num >= (total_blocks_m * total_blocks_n)):
                        break
                    cur_tile: int @ block[1] = (num / (TM * TN))
                    cur_tile_pos: int @ block[1] = (num % (TM * TN))
                    num_block_m = (TM * (cur_tile / (total_blocks_n / TN)))
                    num_block_n = (TN * (cur_tile % (total_blocks_n / TN)))
                    num_block_m += (cur_tile_pos / TN)
                    num_block_n += (cur_tile_pos % TN)
                    schedule_it += 1

                    # Multi-dim accumulator: float d[1][16][8]
                    d: float[1][16][8] @ thread[1]

                    with group(thread[1]):
                        d_ptr: ptr(float) @ thread[1]
                        unsafe("d_ptr = (float *)d;")
                        memset_wrapper(d_ptr, 0, 1 * 16 * 8 * 4)

                    # Register alloc/dealloc (warp[4] group = thread[128])
                    with group(thread[128]):
                        if (is_producer):
                            num_regs: constexpr(int) @ grid[1] = 24 if (num_consumers <= 2) else 32
                            warpgroup_reg_dealloc(template=[num_regs])
                            pass
                        else:
                            num_regs_p: constexpr(int) @ grid[1] = 256 if num_consumers == 1 else (240 if num_consumers == 2 else 160)
                            warpgroup_reg_alloc(template=[num_regs_p])
                            pass

                    num_blocks_k_main: const(int) @ block[1] = (num_blocks_k / 4) * 4

                    for block_k_iter in range(0, num_blocks_k_main, 4):

                        # Producer path: claim at thread[1] scope
                        with claim(sA_slot0, p=thread[1], offset=0) as sA_new:
                            with claim(sA_slot1, p=thread[1], offset=0) as sA_new1:
                                with claim(sA_slot2, p=thread[1], offset=0) as sA_new2:
                                    with claim(sB_slot0, p=thread[1], offset=0) as sB_new:
                                        with claim(sB_slot1, p=thread[1], offset=0) as sB_new1:
                                            with claim(sB_slot2, p=thread[1], offset=0) as sB_new2:
                                                with claim(sA_slot3, p=thread[1], offset=0) as sA_new3:
                                                    with claim(sB_slot3, p=thread[1], offset=0) as sB_new3:
                                                        # match warp -> match split(thread): case 128 (4 warps)
                                                        match split(thread):
                                                            case 128:
                                                                # match thread -> match split(thread): case 1
                                                                match split(thread):
                                                                    case 1:
                                                                        # wait(empty_0, p0)
                                                                        # expect_bytes(full_0, (BK * BN + BK * BM) * 2)
                                                                        ta_void_ptr: ptr(const(void)) @ thread[1]
                                                                        unsafe("ta_void_ptr = &tensorMapA;")
                                                                        # load_async(sA_new, ta_void_ptr, full_0, (block_k_iter + 0) * BK, num_block_m * BM)
                                                                        load_tma(sA_new, ta_void_ptr, (block_k_iter + 0) * BK, num_block_m * BM)
                                                                        tb_void_ptr: ptr(const(void)) @ thread[1]
                                                                        unsafe("tb_void_ptr = &tensorMapB;")
                                                                        # load_async(sB_new, tb_void_ptr, full_0, (block_k_iter + 0) * BK, num_block_n * BN)
                                                                        load_tma(sB_new, tb_void_ptr, (block_k_iter + 0) * BK, num_block_n * BN)

                                                                        # wait(empty_1, p1)
                                                                        # expect_bytes(full_1, (BK * BN + BK * BM) * 2)
                                                                        # load_async(sA_new1, ta_void_ptr, full_1, (block_k_iter + 1) * BK, num_block_m * BM)
                                                                        load_tma(sA_new1, ta_void_ptr, (block_k_iter + 1) * BK, num_block_m * BM)
                                                                        # load_async(sB_new1, tb_void_ptr, full_1, (block_k_iter + 1) * BK, num_block_n * BN)
                                                                        load_tma(sB_new1, tb_void_ptr, (block_k_iter + 1) * BK, num_block_n * BN)

                                                                        # wait(empty_2, p2)
                                                                        # expect_bytes(full_2, (BK * BN + BK * BM) * 2)
                                                                        # load_async(sA_new2, ta_void_ptr, full_2, (block_k_iter + 2) * BK, num_block_m * BM)
                                                                        load_tma(sA_new2, ta_void_ptr, (block_k_iter + 2) * BK, num_block_m * BM)
                                                                        # load_async(sB_new2, tb_void_ptr, full_2, (block_k_iter + 2) * BK, num_block_n * BN)
                                                                        load_tma(sB_new2, tb_void_ptr, (block_k_iter + 2) * BK, num_block_n * BN)

                                                                        # wait(empty_3, p2)
                                                                        # expect_bytes(full_3, (BK * BN + BK * BM) * 2)
                                                                        # load_async(sA_new3, ta_void_ptr, full_3, (block_k_iter + 3) * BK, num_block_m * BM)
                                                                        load_tma(sA_new3, ta_void_ptr, (block_k_iter + 3) * BK, num_block_m * BM)
                                                                        # load_async(sB_new3, tb_void_ptr, full_3, (block_k_iter + 3) * BK, num_block_n * BN)
                                                                        load_tma(sB_new3, tb_void_ptr, (block_k_iter + 3) * BK, num_block_n * BN)

                        # Consumer path: claim at thread[256] scope (warp[8])
                        with claim(sA_slot0, p=thread[256], offset=0) as wgmma_sA:
                            with claim(sA_slot1, p=thread[256], offset=0) as sA_1_producer:
                                with claim(sA_slot2, p=thread[256], offset=0) as sA_2_producer:
                                    with claim(sB_slot0, p=thread[256], offset=0) as wgmma_sB:
                                        with claim(sB_slot1, p=thread[256], offset=0) as sB_1_producer:
                                            with claim(sB_slot2, p=thread[256], offset=0) as sB_2_producer:
                                                with claim(sA_slot3, p=thread[256], offset=0) as sA_3_producer:
                                                    with claim(sB_slot3, p=thread[256], offset=0) as sB_3_producer:
                                                        # match warp -> split thread: case 128 (producer), case 256 (consumer)
                                                        match split(thread):
                                                            case 128:
                                                                pass
                                                            case 256:
                                                                # wait_cluster(full_0, p0)
                                                                with group(thread[128]):
                                                                    warpgroup_arrive()
                                                                with group(thread[128]):
                                                                    with wgmma_async():
                                                                        for m_it in range(0, B_WG_M / WGMMA_M, 1):
                                                                            index_a: int @ thread[128] = m_it * BM
                                                                            index_b: int @ thread[128] = m_it * BN
                                                                            unsafe("#pragma unroll")
                                                                            for k_it in range(0, 64 / WGMMA_K, 1):
                                                                                intrinsic.wgmma256(wgmma_sA[64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M + index_a + k_it * WGMMA_K], wgmma_sB[(index_b + (k_it * WGMMA_K))], 1, 1, 1, 0, 0, out=[d[m_it]])
                                                                # unsafe("if (tid == 0)  arrive(empty_0, 1);")

                                                                # wait_cluster(full_1, p1)
                                                                with group(thread[128]):
                                                                    warpgroup_arrive()
                                                                with group(thread[128]):
                                                                    with wgmma_async():
                                                                        for m_it in range(0, B_WG_M / WGMMA_M, 1):
                                                                            index_a2: int @ thread[128] = m_it * BM
                                                                            index_b2: int @ thread[128] = m_it * BN
                                                                            unsafe("#pragma unroll")
                                                                            for k_it in range(0, 64 / WGMMA_K, 1):
                                                                                intrinsic.wgmma256(sA_1_producer[64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M + index_a2 + k_it * WGMMA_K], sB_1_producer[(index_b2 + (k_it * WGMMA_K))], 1, 1, 1, 0, 0, out=[d[m_it]])
                                                                # unsafe("if (tid == 0)  arrive(empty_1, 1);")

                                                                # wait_cluster(full_2, p2)
                                                                with group(thread[128]):
                                                                    warpgroup_arrive()
                                                                with group(thread[128]):
                                                                    with wgmma_async():
                                                                        for m_it in range(0, B_WG_M / WGMMA_M, 1):
                                                                            index_a3: int @ thread[128] = m_it * BM
                                                                            index_b3: int @ thread[128] = m_it * BN
                                                                            unsafe("#pragma unroll")
                                                                            for k_it in range(0, 64 / WGMMA_K, 1):
                                                                                intrinsic.wgmma256(sA_2_producer[64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M + index_a3 + k_it * WGMMA_K], sB_2_producer[(index_b3 + (k_it * WGMMA_K))], 1, 1, 1, 0, 0, out=[d[m_it]])
                                                                # unsafe("if (tid == 0)  arrive(empty_2, 1);")

                                                                # wait_cluster(full_3, p2)
                                                                with group(thread[128]):
                                                                    warpgroup_arrive()
                                                                with group(thread[128]):
                                                                    with wgmma_async():
                                                                        for m_it in range(0, B_WG_M / WGMMA_M, 1):
                                                                            index_a4: int @ thread[128] = m_it * BM
                                                                            index_b4: int @ thread[128] = m_it * BN
                                                                            unsafe("#pragma unroll")
                                                                            for k_it in range(0, 64 / WGMMA_K, 1):
                                                                                intrinsic.wgmma256(sA_3_producer[64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M + index_a4 + k_it * WGMMA_K], sB_3_producer[(index_b4 + (k_it * WGMMA_K))], 1, 1, 1, 0, 0, out=[d[m_it]])
                                                                # unsafe("if (tid == 0)  arrive(empty_3, 1);")

                        # p0 ^= 1
                        # p1 ^= 1
                        # p2 ^= 1

                    # Epilogue store
                    with claim(block_C, p=thread[256], offset=num_block_n * BN * M + num_block_m * BM) as c_consumer:
                        # match warp -> split thread: case 128 (producer skip), case 256 (consumer store)
                        match split(thread):
                            case 128:
                                pass
                            case 256:
                                with partition(c_consumer, p=thread[1], offset=0) as c_thrd:
                                    # group warp[4] = group thread[128]
                                    with group(thread[128]):
                                        # warp[1] = thread[32]
                                        warp_id: int @ thread[32] = id()
                                        with group(thread[32]):
                                            lane: int @ thread[1] = id()
                                            row: int @ thread[1] = warp_id * 16 + lane / 4
                                            unsafe("#pragma unroll")
                                            for m_it in range(0, B_WG_M / WGMMA_M, 1):
                                                yo: int @ thread[32] = m_it * WGMMA_M + wg_idx * B_WG_M
                                                with group(thread[1]):
                                                    if (row + yo + num_block_m * BM >= M):
                                                        continue
                                                    for w in range(0, WGMMA_N, 16):
                                                        if (w < w < N - num_block_n * BN):
                                                            col: int @ thread[1] = w + 2 * (tid % 4)
                                                            c_thrd[col * M + row + yo] = d[m_it][w / 16][0]
                                                            c_thrd[(col + 1) * M + row + yo] = d[m_it][w / 16][1]
                                                            c_thrd[col * M + (row + 8) + yo] = d[m_it][w / 16][2]
                                                            c_thrd[(col + 1) * M + (row + 8) + yo] = d[m_it][w / 16][3]

                                                            c_thrd[(col + 8) * M + (row + 0) + yo] = d[m_it][w / 16][4]
                                                            c_thrd[(col + 9) * M + (row + 0) + yo] = d[m_it][w / 16][5]
                                                            c_thrd[(col + 8) * M + (row + 8) + yo] = d[m_it][w / 16][6]
                                                            c_thrd[(col + 9) * M + (row + 8) + yo] = d[m_it][w / 16][7]

    return


if __name__ == "__main__":
    print(generate("cutlass_my_h100_match_no_tail"))
