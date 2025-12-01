__device__ uint32_t lookback_to_prev_blocks(int __block_id, int __thread_id, char* __smem, uint32_t block_idx, int* block_infos) {
    int __block_id_0 = __block_id;
    int __thread_id_0 = __thread_id;
    {
        int __block_id_1 = __block_id_0 % 1;
        int __thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;
        
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

    }
    return;
}
