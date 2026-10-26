"""Faithful port of ccuda_py my_sgemm_kernel_3 — no manual sync barriers."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, shared, attr,
    uint, generate,
)


@attr("__launch_bounds__((BM * BN) / (TM * TN), 1)")
@requires(grid[1], block[1], thread[1], smem=49000)
@prism("global")
def my_sgemm_kernel_3(M: int @ grid[1],
                       N: int @ grid[1],
                       K: int @ grid[1],
                       alpha: float @ grid[1],
                       A: ptr(const(float)) @ grid[1],
                       B: ptr(const(float)) @ grid[1],
                       beta: float @ grid[1],
                       C: ptr(float) @ grid[1],
                       num_blocks_N: int @ grid[1],
                       template=[("BM", int), ("BN", int), ("BK", int), ("TM", int), ("TN", int)]):
    cRow: const(int) @ block[1] = id() / num_blocks_N
    cCol: const(int) @ block[1] = id() % num_blocks_N
    totalResultsBlocktile: const(uint) @ grid[1] = BM * BN
    numThreadsBlocktile: const(uint) @ grid[1] = totalResultsBlocktile / (TM * TN)
    with partition(C, p=block[1], offset=cRow * BM * N + cCol * BN) as C_blk:
        with partition(A, p=block[1], offset=cRow * BM * K) as A_blk:
            with partition(B, p=block[1], offset=cCol * BN) as B_blk:
                with group(block[1]):
                    As: shared(float[1024]) @ block[1]
                    Bs: shared(float[1024]) @ block[1]
                    threadRow: const(int) @ thread[1] = id() / (BN / TN)
                    threadCol: const(int) @ thread[1] = id() % (BN / TN)

                    innerRowA: const(int) @ thread[1] = id() / BK
                    innerColA: const(int) @ thread[1] = id() % BK
                    strideA: const(int) @ thread[1] = numThreadsBlocktile / BK

                    innerRowB: const(int) @ thread[1] = id() / BN
                    innerColB: const(int) @ thread[1] = id() % BN
                    strideB: const(int) @ thread[1] = numThreadsBlocktile / BN

                    threadResults: float[64] @ thread[1] = {0}
                    regM: float[8] @ thread[1] = {0}
                    regN: float[8] @ thread[1] = {0}

                    i: int @ thread[1] = 0

                    for bkIdx in range(0, K, BK):
                        with partition(As, p=thread[1], offset=0) as As_thrd:
                            with partition(Bs, p=thread[1], offset=0) as Bs_thrd:
                                with group(thread[1]):
                                    for loadOffset in range(0, BM, strideA):
                                        As_thrd[(innerRowA + loadOffset) * BK + innerColA] = \
                                            A_blk[i * BK + (innerRowA + loadOffset) * K + innerColA]

                                    for loadOffset in range(0, BK, strideB):
                                        Bs_thrd[(innerRowB + loadOffset) * BN + innerColB] = \
                                            B_blk[(i * BK * N) + (innerRowB + loadOffset) * N + innerColB]

                                    i += 1

                                    for dotIdx in range(0, BK, 1):
                                        for i in range(0, TN, 1):
                                            regM[i] = As_thrd[(threadRow * TM + i) * BK + dotIdx]
                                        for i in range(0, TN, 1):
                                            regN[i] = Bs_thrd[dotIdx * BN + threadCol * TN + i]
                                        for resIdxM in range(0, TM, 1):
                                            for resIdxN in range(0, TN, 1):
                                                threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN]

                    for resIdxM in range(0, TM, 1):
                        for resIdxN in range(0, TN, 1):
                            with partition(C_blk, p=thread[1], offset=(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN) as C_thrd:
                                with group(thread[1]):
                                    C_thrd[0] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C_thrd[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]


if __name__ == "__main__":
    print(generate("my_sgemm_kernel_3", auto_sync=True))
