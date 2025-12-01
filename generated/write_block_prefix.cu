__device__ void write_block_prefix(int __block_id, int __thread_id, char* __smem, uint32_t prefix, uint32_t block_idx, int* block_infos) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        
if (threadIdx.x == 0) {
    reinterpret_cast<BlockInfo *>(block_infos)[block_idx].write_prefix(prefix);
}

    }
    return;
}
