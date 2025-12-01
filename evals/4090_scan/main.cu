////////////////////////////////////////////////////////////////////////////////
// main.cu
//
// To update generated kernels, just copy from ~/prism/generated/ into
// the generated/ subdirectory:
//
//   cp ~/prism/generated/{shfl_up_sync_wrapper,scan_warps,get_warp_ends_inclusive,naive_scan_warp_ends,update_with_new_data_and_writeback,write_block_aggregate,write_block_prefix,lookback_to_prev_blocks,get_block_idx_from_global_counter,scan_kernel}.cu generated/
//
////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cuda_runtime.h>

namespace scan_gpu {

////////////////////////////////////////////////////////////////////////////////
// BlockInfo struct (needed by write_block_aggregate, write_block_prefix,
// lookback_to_prev_blocks)
////////////////////////////////////////////////////////////////////////////////

struct BlockInfo {
    union PackedState {
        struct {
            volatile uint32_t ready;
            volatile uint32_t data;
        };
        volatile uint64_t raw;
    };
    PackedState prefix;
    PackedState aggregate;
    __device__ __forceinline__ void write_aggregate(uint32_t aggregate) {
        this->aggregate.raw = static_cast<uint64_t>(true) | (static_cast<uint64_t>(aggregate) << 32);
    }
    __device__ __forceinline__ void write_prefix(uint32_t prefix) {
        this->prefix.raw = static_cast<uint64_t>(true) | (static_cast<uint64_t>(prefix) << 32);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Generated device functions (copy from ~/prism/generated/)
////////////////////////////////////////////////////////////////////////////////

#include "generated/shfl_up_sync_wrapper.cu"
#include "generated/scan_warps.cu"
#include "generated/get_warp_ends_inclusive.cu"
#include "generated/naive_scan_warp_ends.cu"
#include "generated/update_with_new_data_and_writeback.cu"
#include "generated/write_block_aggregate.cu"
#include "generated/write_block_prefix.cu"
#include "generated/lookback_to_prev_blocks.cu"
#include "generated/get_block_idx_from_global_counter.cu"

////////////////////////////////////////////////////////////////////////////////
// Generated global kernel
////////////////////////////////////////////////////////////////////////////////

#include "generated/scan_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Host-side launch glue
////////////////////////////////////////////////////////////////////////////////

size_t get_workspace_size(size_t n) {
    size_t block_cnt = n / (8192 * 2);
    return block_cnt * sizeof(BlockInfo) + sizeof(uint32_t);
}

uint32_t *launch_scan(
    size_t n,
    uint32_t *x,
    void *workspace
) {
    uint32_t block_cnt = n / (8192 * 2);
    uint32_t shmem_needed = (8192 * 2 + 512 * 2) * sizeof(uint32_t);

    cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_needed);

    BlockInfo *block_infos = reinterpret_cast<BlockInfo *>(workspace);
    uint32_t *block_idx_counter = reinterpret_cast<uint32_t *>(block_infos + block_cnt);

    cudaMemset(block_idx_counter, 0, sizeof(uint32_t));
    cudaMemset(block_infos, 0, sizeof(BlockInfo) * block_cnt);

    scan_kernel<<<block_cnt, 512, shmem_needed>>>(n, x, block_idx_counter, reinterpret_cast<int *>(block_infos));
    return x;
}

} // namespace scan_gpu

////////////////////////////////////////////////////////////////////////////////
// Harness + main
////////////////////////////////////////////////////////////////////////////////

#include "scan_harness.cuh"

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        8192 * 2,
        8192 * 2 * 2,
        1 << 20,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: integer sum\n\n");
    run_tests(correctness_sizes, [&](uint32_t n) { return gen_random_data(rng, n); });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}
