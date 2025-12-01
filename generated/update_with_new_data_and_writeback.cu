__device__ void update_with_new_data_and_writeback(int __block_id, int __thread_id, char* __smem, uint32_t* x, const uint32_t* x_shmem, const uint32_t* warp_ends, uint32_t prefix) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        for (int i = 0; i < ((8192 * 2) / (16 * 32)); i++) {
            uint32_t warp_id = (__thread_id_1 / 32);
            {
                uint32_t* x_warp = x + (((warp_id + (i * 16)) * 32));
                {
                    const uint32_t* x_shmem_warp = x_shmem + (((warp_id + (i * 16)) * 32));
                    {
                        int __block_id_2 = __block_id_1;
                        int __thread_id_2 = __thread_id_1 % 32;
                        uint32_t thread_id = __thread_id_2;
                        {
                            uint32_t* x_thread = x_warp + (thread_id);
                            {
                                const uint32_t* x_shmem_thread = x_shmem_warp + (thread_id);
                                {
                                    int __block_id_3 = __block_id_2;
                                    int __thread_id_3 = __thread_id_2 % 1;
                                    if (((warp_id + (i * 16)) == 0)) {
                                        x_thread[0] = (prefix + x_shmem_thread[0]);
                                    } else {
                                        x_thread[0] = ((prefix + warp_ends[((warp_id + (i * 16)) - 1)]) + x_shmem_thread[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
