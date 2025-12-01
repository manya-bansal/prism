__global__ void scan_kernel(uint32_t n, uint32_t* x, uint32_t* block_idx_counter, int* block_infos) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    uint32_t block_idx = get_block_idx_from_global_counter(__block_id_0, __thread_id_0, __smem, block_idx_counter);
    {
        uint32_t* x_block = x + (((8192 * 2) * block_idx));
        {
            int* block_infos_blk = block_infos + (0);
            {
                int __block_id_1 = __block_id_0 % 1;
                int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
                uint32_t (&x_shmem)[(8192 * 2)] = *((uint32_t (*)[(8192 * 2)])(__smem + 0));
                scan_warps(__block_id_1, __thread_id_1, __smem, x_block, x_shmem);
                uint32_t (&warp_ends)[512] = *((uint32_t (*)[512])(__smem + 65536));
                uint32_t (&warp_ends_buf)[512] = *((uint32_t (*)[512])(__smem + 67584));
                __syncthreads();
                get_warp_ends_inclusive(__block_id_1, __thread_id_1, __smem, x_shmem, warp_ends);
                __syncthreads();
                uint32_t* warp_ends_res = naive_scan_warp_ends(__block_id_1, __thread_id_1, __smem, warp_ends, warp_ends_buf);
                uint32_t aggregate = warp_ends_res[(512 - 1)];
                write_block_aggregate(__block_id_1, __thread_id_1, __smem, aggregate, block_idx, block_infos_blk);
                uint32_t prefix = lookback_to_prev_blocks(__block_id_1, __thread_id_1, __smem, block_idx, block_infos_blk);
                write_block_prefix(__block_id_1, __thread_id_1, __smem, (aggregate + prefix), block_idx, block_infos_blk);
                update_with_new_data_and_writeback(__block_id_1, __thread_id_1, __smem, x_block, x_shmem, warp_ends_res, prefix);
            }
        }
    }
}
