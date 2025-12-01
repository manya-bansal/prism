"""Faithful port of ccuda_py my_sgemm_kernel_5 — no manual sync barriers."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, constexpr, shared, attr,
    uint, float4, float4_cast, generate,
)


@requires(block[1], thread[32])
@prism("device")
def processFromSmem_our(regM: ptr(float) @ thread[1],
                         regN: ptr(float) @ thread[1],
                         threadResults: ptr(float) @ thread[1],
                         As: ptr(const(float)) @ block[1],
                         Bs: ptr(const(float)) @ block[1],
                         warpRow: uint @ thread[32],
                         warpCol: uint @ thread[32],
                         template=[("BM", int), ("BN", int), ("BK", int),
                                   ("WM", int), ("WN", int), ("WMITER", int),
                                   ("WNITER", int), ("WSUBM", int), ("WSUBN", int),
                                   ("TM", int), ("TN", int)]):
    with group(block[1]):
        unsafe("/* stub — real implementation is extern */")
    return


@requires(block[1], thread[32])
@prism("device")
def loadFromGmem_our(N: int @ grid[1], K: int @ grid[1],
                     A: ptr(const(float)) @ block[1],
                     B: ptr(const(float)) @ block[1],
                     As: ptr(float) @ block[1],
                     Bs: ptr(float) @ block[1],
                     template=[("BM", int), ("BN", int), ("BK", int),
                               ("rowStrideA", int), ("rowStrideB", int)]):
    with group(block[1]):
        innerRowA: const(uint) @ thread[1] = id() / (BK / 4)
        innerColA: const(uint) @ thread[1] = id() % (BK / 4)
        innerRowB: const(uint) @ thread[1] = id() / (BN / 4)
        innerColB: const(uint) @ thread[1] = id() % (BN / 4)

        with partition(As, p=thread[1], offset=0) as As_t:
            with partition(Bs, p=thread[1], offset=0) as Bs_t:
                with group(thread[1]):
                    for offset in range(0, BM, rowStrideA):
                        tmp: float4 @ thread[1] = float4_cast(A[(innerRowA + offset) * K + innerColA * 4])
                        As_t[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x
                        As_t[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y
                        As_t[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z
                        As_t[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w

                    for offset in range(0, BK, rowStrideB):
                        float4_cast[Bs_t[(innerRowB + offset) * BN + innerColB * 4]] = float4_cast(B[(innerRowB + offset) * N + innerColB * 4])
    return


@attr("__launch_bounds__(NUM_THREADS)")
@requires(grid[1], block[1], thread[32], smem=49000)
@prism("global")
def my_sgemm_kernel_5(M: int @ grid[1],
                       N: int @ grid[1],
                       K: int @ grid[1],
                       alpha: float @ grid[1],
                       A: ptr(float) @ grid[1],
                       B: ptr(float) @ grid[1],
                       beta: float @ grid[1],
                       C: ptr(float) @ grid[1],
                       num_blocks_N: int @ grid[1],
                       template=[("BM", int), ("BN", int), ("BK", int),
                                 ("WM", int), ("WN", int), ("WNITER", int),
                                 ("TM", int), ("TN", int), ("NUM_THREADS", int)]):
    cRow: const(int) @ block[1] = id() / num_blocks_N
    cCol: const(int) @ block[1] = id() % num_blocks_N

    WMITER: constexpr(uint) @ grid[1] = (WM * WN) / (32 * TM * TN * WNITER)
    WSUBM: constexpr(uint) @ grid[1] = WM / WMITER
    WSUBN: constexpr(uint) @ grid[1] = WN / WNITER

    rowStrideA: constexpr(int) @ grid[1] = (NUM_THREADS * 4) / BK
    rowStrideB: constexpr(int) @ grid[1] = NUM_THREADS / (BN / 4)

    with partition(C, p=block[1], offset=cRow * BM * N + cCol * BN) as C_blk:
        with partition(A, p=block[1], offset=cRow * BM * K) as A_blk:
            with partition(B, p=block[1], offset=cCol * BN) as B_blk:
                with group(block[1]):
                    warpIdx: const(uint) @ thread[32] = id()
                    warpCol: const(uint) @ thread[32] = warpIdx % (BN / WN)
                    warpRow: const(uint) @ thread[32] = warpIdx / (BN / WN)
                    As: shared(float[2048]) @ block[1]
                    Bs: shared(float[2048]) @ block[1]

                    innerRowA: const(int) @ thread[1] = id() / (BK / 4)
                    innerColA: const(int) @ thread[1] = id() % (BK / 4)

                    innerRowB: const(int) @ thread[1] = id() / (BN / 4)
                    innerColB: const(int) @ thread[1] = id() % (BN / 4)

                    threadResults: float[128] @ thread[1] = {0}
                    regM: float[8] @ thread[1] = {0}
                    regN: float[16] @ thread[1] = {0}
                    j: int @ block[1] = 0

                    for bkIdx in range(0, K, BK):
                        with partition(A_blk, p=block[1], offset=j * BK) as A_offset:
                            with partition(B_blk, p=block[1], offset=j * BK * N) as B_offset:
                                loadFromGmem_our(N, K, A_offset, B_offset, As, Bs, template=[BM, BN, BK, rowStrideA, rowStrideB])
                                processFromSmem_our(regM, regN, threadResults, As, Bs, warpRow, warpCol, template=[BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN])
                                j += 1

                    with partition(C_blk, p=thread[32], offset=warpRow * WM * N + warpCol * WN) as C_warp:
                        with group(thread[32]):
                            threadIdxInWarp: int @ thread[1] = id()
                            threadColInWarp: const(uint) @ thread[1] = threadIdxInWarp % (WSUBN / TN)
                            threadRowInWarp: const(uint) @ thread[1] = threadIdxInWarp / (WSUBN / TN)

                            for wSubRowIdx in range(0, WMITER, 1):
                                for wSubColIdx in range(0, WNITER, 1):
                                    with partition(C_warp, p=thread[1], offset=(wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN) as C_interim:
                                        for resIdxM in range(0, TM, 1):
                                            for resIdxN in range(0, TN, 4):
                                                with group(thread[1]):
                                                    tmp: float4 @ thread[1] = float4_cast(C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])
                                                    i: const(int) @ thread[1] = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN
                                                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x
                                                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y
                                                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z
                                                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w
                                                    float4_cast[C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN]] = tmp


if __name__ == "__main__":
    print(generate("my_sgemm_kernel_5", auto_sync=True))
