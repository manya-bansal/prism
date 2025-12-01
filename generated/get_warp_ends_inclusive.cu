__device__ void get_warp_ends_inclusive(int __block_id, int __thread_id, char* __smem, const uint32_t* x, uint32_t* warp_ends) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        uint32_t warp_id = (__thread_id_1 / 32);
        {
            const uint32_t* x_warp = x + (((warp_id * 32) * 32));
            {
                uint32_t* warp_ends_warp = warp_ends + ((warp_id * 32));
                {
                    int __block_id_2 = __block_id_1;
                    int __thread_id_2 = __thread_id_1 % 32;
                    uint32_t thread_id = __thread_id_2;
                    {
                        const uint32_t* x_thread = x_warp + ((thread_id * 32));
                        {
                            uint32_t* warp_ends_thread = warp_ends_warp + (thread_id);
                            {
                                int __block_id_3 = __block_id_2;
                                int __thread_id_3 = __thread_id_2 % 1;
                                warp_ends_thread[0] = x_thread[31];
                            }
                        }
                    }
                }
            }
        }
    }
}
