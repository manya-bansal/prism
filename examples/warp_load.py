"""Faithful port of ccuda_py warp_load (with thread_load dependency)."""

from prism import (
    prism, thread,
    requires, group, partition,
    ptr, const, generate,
)

# Import thread_load so it's registered
import examples.thread_load  # noqa: F401


@requires(thread[32])
@prism("device")
def warp_load(input: ptr(const(int)) @ thread[32],
              output: ptr(int) @ thread[1],
              items_per_thread: int @ thread[32]):
    with group(thread[32]):
        tid: int @ thread[1] = id()
        with partition(input, p=thread[1], offset=tid * items_per_thread) as input_thrd:
            with group(thread[1]):
                thread_load(input_thrd, output, items_per_thread)


if __name__ == "__main__":
    print(generate("thread_load"))
    print(generate("warp_load"))
