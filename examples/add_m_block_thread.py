"""Faithful port of ccuda_py add_m_block_thread."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, generate,
)


@requires(grid[1], block[1], thread[1])
@prism("global")
def add_m_block_thread(x: ptr(int) @ grid[1],
                       y: ptr(int) @ grid[1],
                       m: int @ grid[1]):
    bid: int @ block[1] = id()
    tid1: int @ thread[1] = id()
    with partition(x, p=block[1], offset=bid) as x_block:
        with partition(y, p=block[1], offset=bid) as y_block:
            with group(block[1]):
                tid: int @ thread[1] = id()
                with partition(x_block, p=thread[1], offset=tid) as x_thread:
                    with partition(y_block, p=thread[1], offset=tid) as y_thread:
                        with group(thread[1]):
                            y_thread[0] = x_thread[0] + m


if __name__ == "__main__":
    print(generate("add_m_block_thread"))
