"""Faithful port of ccuda_py add_m_thread."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, generate,
)


@requires(grid[1], thread[1])
@prism("global")
def add_m_thread(x: ptr(const(int)) @ grid[1],
                 y: ptr(int) @ grid[1],
                 m: int @ grid[1]):
    tid: int @ thread[1] = id()
    with partition(x, p=thread[1], offset=tid) as x_thread:
        with partition(y, p=thread[1], offset=tid) as y_thread:
            with group(thread[1]):
                y_thread[0] = x_thread[0] + m


if __name__ == "__main__":
    print(generate("add_m_thread"))
