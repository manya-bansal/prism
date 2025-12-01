__device__ uint32_t shfl_up_sync_wrapper(int __block_id, int __thread_id, char* __smem, uint32_t val, uint32_t delta) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0;
        int __thread_id_1 = __thread_id_0 % 1;
        return __shfl_up_sync(static_cast<uint32_t>(-1), val, delta);
    }
    return;
}
