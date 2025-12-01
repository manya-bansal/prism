#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

inline void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation

inline void scan_cpu(size_t n, uint32_t const *x, uint32_t *out) {
    uint32_t accumulator = 0;
    for (size_t i = 0; i < n; i++) {
        accumulator = accumulator + x[i];
        out[i] = accumulator;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Print Utility

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

inline void print_array(size_t n, uint32_t const *x) {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<uint32_t>();
    uint32_t const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(cudaMemcpy(x_host_buf.data(), x, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = std::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf("(Reached maximum limit on 'print_array' output; skipping further calls "
               "to 'print_array')\n");
    }

    total_print_array_output++;
}

////////////////////////////////////////////////////////////////////////////////
// Benchmark Infrastructure

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

inline Results run_config(Mode mode, std::vector<uint32_t> const &x) {
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size(n);
    uint32_t *x_gpu;
    uint32_t *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    auto expected = std::vector<uint32_t>(n);
    scan_cpu(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<uint32_t>(n);
    CUDA_CHECK(cudaMemcpy(actual.data(), out_gpu, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = std::to_string(actual.at(i));
            auto expected_str = std::to_string(expected.at(i));
            printf("Mismatch at position %zu: %s != %s\n", i, actual_str.c_str(), expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array(n, x.data());
                printf("\nExpected:\n");
                print_array(n, expected.data());
                printf("\nActual:\n");
                print_array(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        CUDA_CHECK(cudaFree(x_gpu));
        CUDA_CHECK(cudaFree(workspace_gpu));
        return {0.0, 0.0};
    }

    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(uint32_t) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

template <typename Rng>
std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config(Mode::TEST, data);
        printf("  OK\n\n");
    }
}
