"""Faithful port of ccuda_py warp_store (with thread_store dependency)."""

from prism import (
    prism, thread,
    requires, group, partition,
    ptr, const, generate,
)

# Import thread_store so it's registered
import examples.thread_store  # noqa: F401


@requires(thread[32])
@prism("device")
def warp_store(input: ptr(const(int)) @ thread[1],
               output: ptr(int) @ thread[32],
               items_per_thread: int @ thread[32]):
    with group(thread[32]):
        tid: int @ thread[1] = id()
        with partition(output, p=thread[1], offset=tid * items_per_thread) as output_thrd:
            with group(thread[1]):
                thread_store(input, output_thrd, items_per_thread)


if __name__ == "__main__":
    print(generate("thread_store"))
    print(generate("warp_store"))
