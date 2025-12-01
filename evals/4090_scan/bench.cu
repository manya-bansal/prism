////////////////////////////////////////////////////////////////////////////////
// bench.cu
//
// Benchmarks CUB DeviceScan::InclusiveSum vs our custom scan implementation
// across a sweep of sizes.
////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////
// Error checking
////////////////////////////////////////////////////////////////////////////////

inline void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}
#define CUDA_CHECK(x) do { cuda_check((x), __FILE__, __LINE__); } while (0)

////////////////////////////////////////////////////////////////////////////////
// Our custom scan (from generated/ directory)
////////////////////////////////////////////////////////////////////////////////

namespace scan_gpu {

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

#include "generated/shfl_up_sync_wrapper.cu"
#include "generated/scan_warps.cu"
#include "generated/get_warp_ends_inclusive.cu"
#include "generated/naive_scan_warp_ends.cu"
#include "generated/update_with_new_data_and_writeback.cu"
#include "generated/write_block_aggregate.cu"
#include "generated/write_block_prefix.cu"
#include "generated/lookback_to_prev_blocks.cu"
#include "generated/get_block_idx_from_global_counter.cu"
#include "generated/scan_kernel.cu"

size_t get_workspace_size(size_t n) {
    size_t block_cnt = n / (8192 * 2);
    return block_cnt * sizeof(BlockInfo) + sizeof(uint32_t);
}

uint32_t *launch_scan(size_t n, uint32_t *x, void *workspace) {
    uint32_t block_cnt = n / (8192 * 2);
    uint32_t shmem_needed = (8192 * 2 + 512 * 2) * sizeof(uint32_t);

    cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_needed);

    BlockInfo *block_infos = reinterpret_cast<BlockInfo *>(workspace);
    uint32_t *block_idx_counter = reinterpret_cast<uint32_t *>(block_infos + block_cnt);

    cudaMemset(block_idx_counter, 0, sizeof(uint32_t));
    cudaMemset(block_infos, 0, sizeof(BlockInfo) * block_cnt);

    scan_kernel<<<block_cnt, 512, shmem_needed>>>(n, x, block_idx_counter,
                                                   reinterpret_cast<int *>(block_infos));
    return x;
}

} // namespace scan_gpu

////////////////////////////////////////////////////////////////////////////////
// CUB scan wrapper
////////////////////////////////////////////////////////////////////////////////

namespace scan_cub {

struct CubWorkspace {
    void *temp_storage;
    size_t temp_storage_bytes;
};

size_t get_workspace_size(size_t n) {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                  (uint32_t *)nullptr, (uint32_t *)nullptr, n);
    return temp_storage_bytes;
}

void launch_scan(size_t n, uint32_t *d_in, uint32_t *d_out,
                 void *temp_storage, size_t temp_storage_bytes) {
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes,
                                  d_in, d_out, n);
}

} // namespace scan_cub

////////////////////////////////////////////////////////////////////////////////
// CPU reference
////////////////////////////////////////////////////////////////////////////////

void scan_cpu(size_t n, uint32_t const *x, uint32_t *out) {
    uint32_t acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += x[i];
        out[i] = acc;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Benchmark helper — returns best time in ms over target_time_ms window
////////////////////////////////////////////////////////////////////////////////

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best = 1e30;
    double elapsed = 0.0;
    while (elapsed < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        elapsed += ms;
        best = std::min(best, ms);
    }
    return best;
}

////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////

int main() {
    std::vector<size_t> sizes = {
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
        268435456,
        536870912,
        1073741824,
    };

    auto rng = std::mt19937(0xCA7CAFE);
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);

    // Print header
    printf("%-14s | %-10s %-14s | %-10s %-14s\n",
           "Size", "CUB (ms)", "CUB (GB/s)", "Ours (ms)", "Ours (GB/s)");
    printf("%-14s-+-%-10s-%-14s-+-%-10s-%-14s\n",
           "--------------", "----------", "--------------",
           "----------", "--------------");

    for (size_t n : sizes) {
        // Generate random input on host
        std::vector<uint32_t> h_data(n);
        for (size_t i = 0; i < n; i++) h_data[i] = uniform(rng);

        double bytes = (double)n * sizeof(uint32_t) * 2; // read + write

        // Allocate device memory
        uint32_t *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint32_t)));

        // ---- CUB benchmark ----
        uint32_t *d_cub_out;
        CUDA_CHECK(cudaMalloc(&d_cub_out, n * sizeof(uint32_t)));

        size_t cub_ws_bytes = scan_cub::get_workspace_size(n);
        void *d_cub_ws;
        CUDA_CHECK(cudaMalloc(&d_cub_ws, cub_ws_bytes));

        // Correctness check for CUB
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        scan_cub::launch_scan(n, d_data, d_cub_out, d_cub_ws, cub_ws_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> h_expected(n), h_actual(n);
        scan_cpu(n, h_data.data(), h_expected.data());
        CUDA_CHECK(cudaMemcpy(h_actual.data(), d_cub_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) {
            if (h_actual[i] != h_expected[i]) {
                printf("CUB mismatch at %zu: got %u expected %u\n", i, h_actual[i], h_expected[i]);
                return 1;
            }
        }

        double cub_ms = benchmark_ms(
            500.0,
            [&]() {
                CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
            },
            [&]() {
                scan_cub::launch_scan(n, d_data, d_cub_out, d_cub_ws, cub_ws_bytes);
            });
        double cub_bw = bytes / cub_ms / 1e6;

        CUDA_CHECK(cudaFree(d_cub_out));
        CUDA_CHECK(cudaFree(d_cub_ws));

        // ---- Our scan benchmark ----
        size_t our_ws_bytes = scan_gpu::get_workspace_size(n);
        void *d_our_ws;
        CUDA_CHECK(cudaMalloc(&d_our_ws, our_ws_bytes));

        // Correctness check for ours
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_our_ws, 0, our_ws_bytes));
        scan_gpu::launch_scan(n, d_data, d_our_ws);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_actual.data(), d_data, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) {
            if (h_actual[i] != h_expected[i]) {
                printf("Ours mismatch at %zu: got %u expected %u\n", i, h_actual[i], h_expected[i]);
                return 1;
            }
        }

        double our_ms = benchmark_ms(
            500.0,
            [&]() {
                CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_our_ws, 0, our_ws_bytes));
            },
            [&]() {
                scan_gpu::launch_scan(n, d_data, d_our_ws);
            });
        double our_bw = bytes / our_ms / 1e6;

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_our_ws));

        printf("%-14zu | %-10.3f %-14.2f | %-10.3f %-14.2f\n",
               n, cub_ms, cub_bw, our_ms, our_bw);
        fflush(stdout);
    }

    return 0;
}
