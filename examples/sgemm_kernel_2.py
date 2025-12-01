"""Faithful port of ccuda_py my_sgemm_kernel_2."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, generate,
)


@requires(grid[1], block[1], thread[1], smem=49000)
@prism("global")
def my_sgemm_kernel_2(M: int @ grid[1],
                       N: int @ grid[1],
                       K: int @ grid[1],
                       alpha: float @ grid[1],
                       A: ptr(const(float)) @ grid[1],
                       B: ptr(const(float)) @ grid[1],
                       beta: float @ grid[1],
                       C: ptr(float) @ grid[1],
                       tile_size: int @ grid[1]):
    global_tid: int @ thread[1] = id()
    total_elements: const(int) @ grid[1] = M * N
    num_tiles_per_row: const(int) @ grid[1] = (N + tile_size - 1) / tile_size
    tile_id: const(int) @ thread[1] = global_tid / (tile_size * tile_size)
    tile_row: const(int) @ thread[1] = tile_id / num_tiles_per_row
    tile_col: const(int) @ thread[1] = tile_id % num_tiles_per_row
    local_id: const(int) @ thread[1] = global_tid % (tile_size * tile_size)
    local_row: const(int) @ thread[1] = local_id / tile_size
    local_col: const(int) @ thread[1] = local_id % tile_size
    cRow: const(int) @ thread[1] = tile_row * tile_size + local_row
    cCol: const(int) @ thread[1] = tile_col * tile_size + local_col
    with partition(C, p=thread[1], offset=cRow * N + cCol) as C_thrd:
        with group(thread[1]):
            if global_tid < total_elements:
                if cRow < M and cCol < N:
                    tmp: float @ thread[1] = 0
                    for i in range(0, K, 1):
                        tmp += A[cRow * K + i] * B[i * N + cCol]
                    C_thrd[0] = alpha * tmp + beta * C_thrd[0]


if __name__ == "__main__":
    print(generate("my_sgemm_kernel_2", auto_sync=True))
