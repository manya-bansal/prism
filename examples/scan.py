"""Port of ccuda_py scan branch: Single-pass Parallel Prefix Scan with Decoupled Look-back.
No manual sync barriers — auto_sync will insert them.
"""

from prism import (
    prism, grid, block, thread,
    requires, group, partition, split,
    ptr, const, shared, unsafe, uint32_t,
    generate,
)

# Constants (matching ccuda_py reference)
# WORK_PER_BLOCK = (8192 * 2)
# WARPS_PER_BLOCK = 16
# WARPS_PASS_CNT = (8192 * 2) / (16 * 32) = 32
# WARP_ENDS = (8192 * 2) / 32 = 512


# ---------------------------------------------------------------------------
# Stub device functions (handwritten CUDA bodies via unsafe)
# ---------------------------------------------------------------------------

@requires(thread[1])
@prism("device")
def shfl_up_sync_wrapper(val: uint32_t @ thread[1],
                         delta: uint32_t @ thread[1]) -> uint32_t @ thread[1]:
    with group(thread[1]):
        unsafe("return __shfl_up_sync(static_cast<uint32_t>(-1), val, delta);")
    return


@requires(block[1], thread[1])
@prism("device")
def write_block_aggregate(aggregate: uint32_t @ block[1],
                          block_idx: uint32_t @ block[1],
                          block_infos: ptr(int) @ block[1]):
    with group(block[1]):
        unsafe("""
if (threadIdx.x == 0) {
    reinterpret_cast<BlockInfo *>(block_infos)[block_idx].write_aggregate(aggregate);
}
""")
    return


@requires(block[1], thread[1])
@prism("device")
def write_block_prefix(prefix: uint32_t @ block[1],
                       block_idx: uint32_t @ block[1],
                       block_infos: ptr(int) @ block[1]):
    with group(block[1]):
        unsafe("""
if (threadIdx.x == 0) {
    reinterpret_cast<BlockInfo *>(block_infos)[block_idx].write_prefix(prefix);
}
""")
    return


@requires(block[1], thread[1])
@prism("device")
def lookback_to_prev_blocks(block_idx: uint32_t @ block[1],
                            block_infos: ptr(int) @ block[1]) -> uint32_t @ block[1]:
    with group(block[1]):
        unsafe("""
BlockInfo *block_infos_typed = reinterpret_cast<BlockInfo *>(block_infos);
__shared__ uint32_t shared_res;
if (threadIdx.x == 0) {
    uint32_t res = 0;
    for (int id = static_cast<int>(block_idx) - 1; id >= 0; --id) {
        while (!block_infos_typed[id].aggregate.ready) {
        }
        if (block_infos_typed[id].prefix.ready) {
            res = block_infos_typed[id].prefix.data + res;
            break;
        } else {
            res = block_infos_typed[id].aggregate.data + res;
        }
    }
    shared_res = res;
}
__syncthreads();
return shared_res;
""")
    return


@requires(grid[1])
@prism("device")
def get_block_idx_from_global_counter(block_idx_counter: ptr(uint32_t) @ grid[1]) -> uint32_t @ grid[1]:
    unsafe("""
__shared__ uint32_t res;
if (threadIdx.x == 0) {
    res = atomicAdd(block_idx_counter, 1);
}
__syncthreads();
return res;
""")
    return


# ---------------------------------------------------------------------------
# Generated device functions (full prism logic)
# ---------------------------------------------------------------------------

@requires(block[1], thread[512])
@prism("device")
def scan_warps(x: ptr(const(uint32_t)) @ block[1],
               x_shmem: ptr(shared(uint32_t)) @ block[1]):
    with group(block[1]):
        for i in range(0, (8192 * 2) // (16 * 32)):
            warp_id: uint32_t @ thread[32] = id()
            with partition(x, offset=(warp_id + i * 16) * 32, p=thread[32]) as x_warp:
                with partition(x_shmem, offset=(warp_id + i * 16) * 32, p=thread[32]) as x_shmem_warp:
                    with group(thread[32]):
                        thread_id: uint32_t @ thread[1] = id()
                        with partition(x_warp, offset=thread_id, p=thread[1]) as x_thread:
                            with partition(x_shmem_warp, offset=thread_id, p=thread[1]) as x_shmem_thread:
                                with group(thread[1]):
                                    val: uint32_t @ thread[1] = x_thread[0]

                                    stride: uint32_t @ thread[1] = 1
                                    while stride < 32:
                                        received: uint32_t @ thread[1] = shfl_up_sync_wrapper(val, stride)
                                        if thread_id >= stride:
                                            val = received + val
                                        stride *= 2
                                    x_shmem_thread[0] = val


@requires(block[1], thread[512])
@prism("device")
def get_warp_ends_inclusive(x: ptr(const(shared(uint32_t))) @ block[1],
                            warp_ends: ptr(shared(uint32_t)) @ block[1]):
    with group(block[1]):
        warp_id: uint32_t @ thread[32] = id()
        with partition(x, offset=warp_id * 32 * 32, p=thread[32]) as x_warp:
            with partition(warp_ends, offset=warp_id * 32, p=thread[32]) as warp_ends_warp:
                with group(thread[32]):
                    thread_id: uint32_t @ thread[1] = id()
                    with partition(x_warp, offset=thread_id * 32, p=thread[1]) as x_thread:
                        with partition(warp_ends_warp, offset=thread_id, p=thread[1]) as warp_ends_thread:
                            with group(thread[1]):
                                warp_ends_thread[0] = x_thread[31]


@requires(block[1], thread[512])
@prism("device")
def naive_scan_warp_ends(x: ptr(shared(uint32_t)) @ block[1],
                         x_buf: ptr(shared(uint32_t)) @ block[1]) -> ptr(uint32_t) @ block[1]:
    with group(block[1]):
        stride: uint32_t @ block[1] = 1
        while stride < (8192 * 2) // 32:

            warp_id: uint32_t @ thread[32] = id()
            with partition(x, offset=warp_id * 32, p=thread[32]) as x_warp:
                with partition(x_buf, offset=warp_id * 32, p=thread[32]) as x_buf_warp:
                    with group(thread[32]):
                        thread_id: uint32_t @ thread[1] = id()
                        with partition(x_warp, offset=thread_id, p=thread[1]) as x_thread:
                            with partition(x_buf_warp, offset=thread_id, p=thread[1]) as x_buf_thread:
                                with group(thread[1]):
                                    if thread_id + warp_id * 32 >= stride:
                                        x_buf_thread[0] = x_thread[0] + x_thread[-stride]
                                    else:
                                        x_buf_thread[0] = x_thread[0]

            unsafe("uint32_t* tmp = x; x = x_buf; x_buf = tmp; __syncthreads();") # swapping is not safe!

            stride *= 2
    return x


@requires(block[1], thread[512])
@prism("device")
def update_with_new_data_and_writeback(x: ptr(uint32_t) @ block[1],
                                       x_shmem: ptr(const(shared(uint32_t))) @ block[1],
                                       warp_ends: ptr(const(shared(uint32_t))) @ block[1],
                                       prefix: uint32_t @ block[1]):
    with group(block[1]):
        for i in range(0, (8192 * 2) // (16 * 32)):
            warp_id: uint32_t @ thread[32] = id()
            with partition(x, offset=(warp_id + i * 16) * 32, p=thread[32]) as x_warp:
                with partition(x_shmem, offset=(warp_id + i * 16) * 32, p=thread[32]) as x_shmem_warp:
                    with group(thread[32]):
                        thread_id: uint32_t @ thread[1] = id()
                        with partition(x_warp, offset=thread_id, p=thread[1]) as x_thread:
                            with partition(x_shmem_warp, offset=thread_id, p=thread[1]) as x_shmem_thread:
                                with group(thread[1]):
                                    if warp_id + i * 16 == 0:
                                        x_thread[0] = prefix + x_shmem_thread[0]
                                    else:
                                        x_thread[0] = prefix + warp_ends[warp_id + i * 16 - 1] + x_shmem_thread[0]


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

@requires(grid[1], block[1], thread[512], smem=69632)
@prism("global")
def scan_kernel(n: uint32_t @ grid[1],
                x: ptr(uint32_t) @ grid[1],
                block_idx_counter: ptr(uint32_t) @ grid[1],
                block_infos: ptr(int) @ grid[1]):
    block_idx: uint32_t @ grid[1] = get_block_idx_from_global_counter(block_idx_counter)

    with partition(x, offset=(8192 * 2) * block_idx, p=block[1]) as x_block:
        with partition(block_infos, offset=0, p=block[1]) as block_infos_blk:
            with group(block[1]):
                x_shmem: shared(uint32_t[8192 * 2]) @ block[1]

                scan_warps(x_block, x_shmem)

                warp_ends: shared(uint32_t[512]) @ block[1]
                warp_ends_buf: shared(uint32_t[512]) @ block[1]

                get_warp_ends_inclusive(x_shmem, warp_ends)
                warp_ends_res: ptr(uint32_t) @ block[1] = naive_scan_warp_ends(warp_ends, warp_ends_buf)

                aggregate: uint32_t @ block[1] = warp_ends_res[512 - 1]
                write_block_aggregate(aggregate, block_idx, block_infos_blk)

                prefix: uint32_t @ block[1] = lookback_to_prev_blocks(block_idx, block_infos_blk)
                write_block_prefix(aggregate + prefix, block_idx, block_infos_blk)

                update_with_new_data_and_writeback(x_block, x_shmem, warp_ends_res, prefix)


if __name__ == "__main__":
    for name in ["shfl_up_sync_wrapper", "scan_warps", "get_warp_ends_inclusive",
                  "naive_scan_warp_ends", "update_with_new_data_and_writeback",
                  "write_block_aggregate", "write_block_prefix",
                  "lookback_to_prev_blocks", "get_block_idx_from_global_counter",
                  "scan_kernel"]:
        print(generate(name, auto_sync=True))
