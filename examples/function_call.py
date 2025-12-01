"""Faithful port of ccuda_py helper_add + test_call_helper."""

from prism import (
    prism, grid, block, thread,
    requires, group,
    generate,
)


@requires(block[16])
@prism("device")
def helper_add(x: int @ block[16], y: int @ block[16]) -> int @ block[16]:
    with group(block[16]):
        result: int @ block[16] = x + y
    return result


@requires(grid[1], block[32])
@prism("global")
def test_call_helper():
    with group(block[32]):
        a: int @ block[32] = 10
        with group(block[16]):
            b: int @ block[16] = 20
            c: int @ block[16] = helper_add(a, b)


if __name__ == "__main__":
    print(generate("helper_add"))
    print(generate("test_call_helper"))
