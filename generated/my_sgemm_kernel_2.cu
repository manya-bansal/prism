__global__ void my_sgemm_kernel_2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, int tile_size) {
    extern __shared__ __align__(128) char __smem[];
    int __block_id_0 = blockIdx.x;
    int __thread_id_0 = __block_id_0 * blockDim.x + threadIdx.x;
    int global_tid = __thread_id_0;
    const int total_elements = (M * N);
    const int num_tiles_per_row = (((N + tile_size) - 1) / tile_size);
    const int tile_id = (global_tid / (tile_size * tile_size));
    const int tile_row = (tile_id / num_tiles_per_row);
    const int tile_col = (tile_id % num_tiles_per_row);
    const int local_id = (global_tid % (tile_size * tile_size));
    const int local_row = (local_id / tile_size);
    const int local_col = (local_id % tile_size);
    const int cRow = ((tile_row * tile_size) + local_row);
    const int cCol = ((tile_col * tile_size) + local_col);
    {
        float* C_thrd = C + (((cRow * N) + cCol));
        {
            int __block_id_1 = __block_id_0;
            int __thread_id_1 = __thread_id_0 % 1;
            if ((global_tid < total_elements)) {
                if (((cRow < M) && (cCol < N))) {
                    float tmp = 0;
                    for (int i = 0; i < K; i++) {
                        tmp += (A[((cRow * K) + i)] * B[((i * N) + cCol)]);
                    }
                    C_thrd[0] = ((alpha * tmp) + (beta * C_thrd[0]));
                }
            }
        }
    }
}
