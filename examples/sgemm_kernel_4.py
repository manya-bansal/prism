"""Faithful port of ccuda_py my_sgemm_kernel_4 — no manual sync barriers."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, shared, attr, float4_cast,
    uint, float4, generate,
)


@attr("__launch_bounds__((BM * BN) / (TM * TN), 1)")
@requires(grid[1], block[1], thread[1], smem=49000)
@prism("global")
def my_sgemm_kernel_4(M: int @ grid[1],
                       N: int @ grid[1],
                       K: int @ grid[1],
                       alpha: float @ grid[1],
                       A: ptr(float) @ grid[1],
                       B: ptr(float) @ grid[1],
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

                    innerRowA: const(int) @ thread[1] = id() / (BK / 4)
                    innerColA: const(int) @ thread[1] = id() % (BK / 4)
                    strideA: const(int) @ thread[1] = numThreadsBlocktile / BK

                    innerRowB: const(int) @ thread[1] = id() / (BN / 4)
                    innerColB: const(int) @ thread[1] = id() % (BN / 4)
                    strideB: const(int) @ thread[1] = numThreadsBlocktile / BN

                    threadResults: float[64] @ thread[1] = {0}
                    regM: float[8] @ thread[1] = {0}
                    regN: float[8] @ thread[1] = {0}

                    i: int @ thread[1] = 0

                    for bkIdx in range(0, K, BK):
                        with partition(As, p=thread[1], offset=0) as As_thrd:
                            with partition(Bs, p=thread[1], offset=0) as Bs_thrd:
                                with group(thread[1]):
                                    tmp: float4 @ thread[1] = float4_cast(A_blk[i * BK + innerRowA * K + innerColA * 4])
                                    As_thrd[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x
                                    As_thrd[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y
                                    As_thrd[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z
                                    As_thrd[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w

                                    float4_cast[Bs_thrd[innerRowB * BN + innerColB * 4]] = \
                                        float4_cast(B_blk[(i * BK * N) + innerRowB * N + innerColB * 4])

                                    i += 1

                                    for dotIdx in range(0, BK, 1):
                                        for i in range(0, TN, 1):
                                            regM[i] = As[dotIdx * BM + threadRow * TM + i]
                                        for i in range(0, TN, 1):
                                            regN[i] = Bs[dotIdx * BN + threadCol * TN + i]
                                        for resIdxM in range(0, TM, 1):
                                            for resIdxN in range(0, TN, 1):
                                                threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN]

                    for resIdxM in range(0, TM, 1):
                        for resIdxN in range(0, TN, 4):
                            with partition(C_blk, p=thread[1], offset=(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN) as C_thrd:
                                with group(thread[1]):
                                    temp: float4 @ thread[1] = float4_cast(C_thrd[0])
                                    temp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * temp.x
                                    temp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * temp.y
                                    temp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * temp.z
                                    temp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * temp.w
                                    float4_cast[C_thrd[0]] = temp


if __name__ == "__main__":
    print(generate("my_sgemm_kernel_4", auto_sync=True))
