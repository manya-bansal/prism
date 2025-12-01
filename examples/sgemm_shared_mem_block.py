"""Port of sgemm_shared_mem_block to prism — no manual sync barriers.

Reference: CUDA shared memory tiled SGEMM.
Note: 2D grid (blockIdx.x/y) linearized via num_blocks_N parameter.
Pointer advancement (A += BLOCKSIZE) replaced by index arithmetic with bkIdx.
"""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, shared,
    uint, generate,
)


@requires(grid[1], block[1], thread[1])
@prism("global")
def sgemm_shared_mem_block(M: int @ grid[1],
                           N: int @ grid[1],
                           K: int @ grid[1],
                           alpha: float @ grid[1],
                           A: ptr(const(float)) @ grid[1],
                           B: ptr(const(float)) @ grid[1],
                           beta: float @ grid[1],
                           C: ptr(float) @ grid[1],
                           num_blocks_N: int @ grid[1],
                           template=[("BLOCKSIZE", int)]):
    # 2D block index from linear block id
    cRow: const(uint) @ block[1] = id() / num_blocks_N
    cCol: const(uint) @ block[1] = id() % num_blocks_N

    # Advance pointers to the starting positions for this block
    with partition(A, p=block[1], offset=cRow * BLOCKSIZE * K) as A_blk:
        with partition(B, p=block[1], offset=cCol * BLOCKSIZE) as B_blk:
            with partition(C, p=block[1], offset=cRow * BLOCKSIZE * N + cCol * BLOCKSIZE) as C_blk:
                with group(block[1]):
                    # Shared memory tiles (static — size depends on template param)
                    As: shared(float[BLOCKSIZE * BLOCKSIZE], static=True) @ block[1]
                    Bs: shared(float[BLOCKSIZE * BLOCKSIZE], static=True) @ block[1]

                    threadCol: const(uint) @ thread[1] = id() % BLOCKSIZE
                    threadRow: const(uint) @ thread[1] = id() / BLOCKSIZE

                    tmp: float @ thread[1] = 0

                    with partition(As, p=thread[1], offset=0) as As_t:
                        with partition(Bs, p=thread[1], offset=0) as Bs_t:
                            with partition(C_blk, p=thread[1], offset=0) as C_t:
                                with group(thread[1]):
                                    for bkIdx in range(0, K, BLOCKSIZE):
                                        # Load one element per thread into shared memory
                                        As_t[threadRow * BLOCKSIZE + threadCol] = A_blk[bkIdx + threadRow * K + threadCol]
                                        Bs_t[threadRow * BLOCKSIZE + threadCol] = B_blk[(bkIdx + threadRow) * N + threadCol]

                                        # Dot product on the cached block
                                        for dotIdx in range(0, BLOCKSIZE, 1):
                                            tmp += As_t[threadRow * BLOCKSIZE + dotIdx] * Bs_t[dotIdx * BLOCKSIZE + threadCol]

                                    # Write result
                                    C_t[threadRow * N + threadCol] = alpha * tmp + beta * C_t[threadRow * N + threadCol]


if __name__ == "__main__":
    print(generate("sgemm_shared_mem_block", auto_sync=True))
