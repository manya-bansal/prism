"""Faithful port of ccuda_py my_sgemm_kernel_1."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, generate,
)


@requires(grid[1], block[1], thread[1], smem=49000)
@prism("global")
def my_sgemm_kernel_1(M: int @ grid[1],
                       N: int @ grid[1],
                       K: int @ grid[1],
                       alpha: float @ grid[1],
                       A: ptr(const(float)) @ grid[1],
                       B: ptr(const(float)) @ grid[1],
                       beta: float @ grid[1],
                       C: ptr(float) @ grid[1],
                       block_size: int @ grid[1]):
    bid: int @ block[1] = id()
    with partition(C, p=block[1], offset=0) as C_blk:
        with group(block[1]):
            tid: int @ thread[1] = bid * block_size + id()
            with partition(C_blk, p=thread[1], offset=0) as C_thrd:
                x: const(int) @ thread[1] = tid / N
                y: const(int) @ thread[1] = tid % N
                with group(thread[1]):
                    tmp: float @ thread[1] = 0
                    for i in range(0, K, 1):
                        tmp += A[x * K + i] * B[i * N + y]
                    C_thrd[x * N + y] = alpha * tmp + beta * C[x * N + y]


if __name__ == "__main__":
    print(generate("my_sgemm_kernel_1", auto_sync=True))
