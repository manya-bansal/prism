"""Faithful port of ccuda_py mmaTF32NaiveKernel: warp-level MMA naive kernel."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, constexpr, shared, unsafe,
    generate,
)

# Import simple_mma so it's registered
import examples.simple_mma  # noqa: F401


@requires(grid[1], block[1], thread[32], smem=49000)
@prism("global")
def mmaTF32NaiveKernel(A: ptr(const(float)) @ grid[1],
                        B: ptr(const(float)) @ grid[1],
                        C: ptr(float) @ grid[1],
                        M: int @ grid[1],
                        N: int @ grid[1],
                        K: int @ grid[1]):
    MMA_N: constexpr(int) @ grid[1] = 8
    MMA_K: constexpr(int) @ grid[1] = 8
    MMA_M: constexpr(int) @ grid[1] = 16

    num_blocks_n: const(int) @ grid[1] = (N + MMA_N - 1) / MMA_N
    K_tiles: const(int) @ grid[1] = (K + MMA_K - 1) / MMA_K
    block_row: const(int) @ block[1] = id() / num_blocks_n
    block_col: const(int) @ block[1] = id() % num_blocks_n

    warp_row: const(int) @ block[1] = block_row * MMA_M
    warp_col: const(int) @ block[1] = block_col * MMA_N

    with partition(C, p=block[1], offset=warp_row * N + warp_col) as C_blk:
        with group(block[1]):
            A_smem: shared(float[128]) @ block[1]
            B_smem: shared(float[64]) @ block[1]
            C_smem: shared(float[128]) @ block[1]

            with partition(C_smem, p=thread[32], offset=0) as C_smem_warp:
                with group(thread[32]):
                    idx: int @ thread[1] = id() * 4
                    for i in range(0, 4, 1):
                        with partition(C_smem_warp, p=thread[1], offset=idx) as C_thrd:
                            with group(thread[1]):
                                C_thrd[i] = 0
            unsafe("__syncthreads();")

            for i in range(0, K_tiles, 1):
                a_idx: int @ thread[1] = id() * 4
                unsafe("__syncthreads();")
                for j in range(0, 4, 1):
                    flat_idx: int @ thread[1] = a_idx + j
                    row: int @ thread[1] = flat_idx / MMA_K
                    col: int @ thread[1] = flat_idx % MMA_K
                    global_row: int @ thread[1] = warp_row + row
                    global_col: int @ thread[1] = i * MMA_K + col
                    with partition(A_smem, p=thread[1], offset=row * MMA_K + col) as A_smem_thrd:
                        with group(thread[1]):
                            A_smem_thrd[0] = A[global_row * K + global_col]

                b_idx: int @ thread[1] = id() * 2
                unsafe("__syncthreads();")
                for j in range(0, 2, 1):
                    flat_idx_b: int @ thread[1] = b_idx + j
                    row_b: int @ thread[1] = flat_idx_b / MMA_K
                    col_b: int @ thread[1] = flat_idx_b % MMA_K
                    global_row_b: int @ thread[1] = i * MMA_K + row_b
                    global_col_b: int @ thread[1] = warp_col + col_b
                    with partition(B_smem, p=thread[1], offset=row_b * MMA_K + col_b) as B_smem_thrd:
                        with group(thread[1]):
                            B_smem_thrd[0] = B[global_row_b * N + global_col_b]

                unsafe("__syncthreads();")
                with partition(C_smem, p=thread[32], offset=0) as C_smem_warp:
                    with group(thread[32]):
                        simple_mma(A_smem, B_smem, C_smem_warp)

            for j in range(0, 4, 1):
                flat_idx_c: int @ thread[1] = id() * 4 + j
                row_c: int @ thread[1] = flat_idx_c / MMA_K
                col_c: int @ thread[1] = flat_idx_c % MMA_K
                with partition(C_blk, p=thread[1], offset=row_c * N + col_c) as C_thrd:
                    with group(thread[1]):
                        C_thrd[0] = C_smem[row_c * MMA_N + col_c]


if __name__ == "__main__":
    print(generate("simple_mma"))
    print(generate("mmaTF32NaiveKernel"))
