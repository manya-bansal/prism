__device__ uint32_t get_block_idx_from_global_counter(int __block_id, int __thread_id, char* __smem, uint32_t* block_idx_counter) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    
__shared__ uint32_t res;
if (threadIdx.x == 0) {
    res = atomicAdd(block_idx_counter, 1);
}
__syncthreads();
return res;

    return;
}
