"""Faithful port of ccuda_py thread_store."""

from prism import (
    prism, thread,
    requires, group,
    ptr, const, generate,
)


@requires(thread[1])
@prism("device")
def thread_store(input: ptr(const(int)) @ thread[1],
                 output: ptr(int) @ thread[1],
                 items_per_thread: int @ thread[1]):
    with group(thread[1]):
        for i in range(0, items_per_thread, 1):
            output[i] = input[i]


if __name__ == "__main__":
    print(generate("thread_store"))
