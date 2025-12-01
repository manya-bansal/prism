__device__ uint32_t* naive_scan_warp_ends(int __block_id, int __thread_id, char* __smem, uint32_t* x, uint32_t* x_buf) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        uint32_t stride = 1;
        while ((stride < ((8192 * 2) / 32))) {
            uint32_t warp_id = (__thread_id_1 / 32);
            {
                uint32_t* x_warp = x + ((warp_id * 32));
                {
                    uint32_t* x_buf_warp = x_buf + ((warp_id * 32));
                    {
                        int __block_id_2 = __block_id_1;
                        int __thread_id_2 = __thread_id_1 % 32;
                        uint32_t thread_id = __thread_id_2;
                        {
                            uint32_t* x_thread = x_warp + (thread_id);
                            {
                                uint32_t* x_buf_thread = x_buf_warp + (thread_id);
                                {
                                    int __block_id_3 = __block_id_2;
                                    int __thread_id_3 = __thread_id_2 % 1;
                                    if (((thread_id + (warp_id * 32)) >= stride)) {
                                        x_buf_thread[0] = (x_thread[0] + x_thread[(-stride)]);
                                    } else {
                                        x_buf_thread[0] = x_thread[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            uint32_t* tmp = x; x = x_buf; x_buf = tmp; __syncthreads();
            stride *= 2;
        }
    }
    return x;
}
