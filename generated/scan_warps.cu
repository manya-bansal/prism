__device__ void scan_warps(int __block_id, int __thread_id, char* __smem, const uint32_t* x, uint32_t* x_shmem) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        for (int i = 0; i < ((8192 * 2) / (16 * 32)); i++) {
            uint32_t warp_id = (__thread_id_1 / 32);
            {
                const uint32_t* x_warp = x + (((warp_id + (i * 16)) * 32));
                {
                    uint32_t* x_shmem_warp = x_shmem + (((warp_id + (i * 16)) * 32));
                    {
                        int __block_id_2 = __block_id_1;
                        int __thread_id_2 = __thread_id_1 % 32;
                        uint32_t thread_id = __thread_id_2;
                        {
                            const uint32_t* x_thread = x_warp + (thread_id);
                            {
                                uint32_t* x_shmem_thread = x_shmem_warp + (thread_id);
                                {
                                    int __block_id_3 = __block_id_2;
                                    int __thread_id_3 = __thread_id_2 % 1;
                                    uint32_t val = x_thread[0];
                                    uint32_t stride = 1;
                                    while ((stride < 32)) {
                                        uint32_t received = shfl_up_sync_wrapper(__block_id_3, __thread_id_3, __smem, val, stride);
                                        if ((thread_id >= stride)) {
                                            val = (received + val);
                                        }
                                        stride *= 2;
                                    }
                                    x_shmem_thread[0] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
