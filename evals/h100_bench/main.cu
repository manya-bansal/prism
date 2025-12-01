// Taken from https://github.com/pranjalssh/fast.cu

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// Include the kernel implementation
#include "utils.cuh"

std::default_random_engine generator(69);

void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

cublasHandle_t cublas_handle;

void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  warmupKernel<<<1024, 1024>>>();

  cublasCreate(&cublas_handle);
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int size = 2048*3;
  if (argc > 1) size = atoi(argv[1]);
  long max_size = size;
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;
  int *DB = nullptr; int *dDB = nullptr;

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));

  int repeat_times = 8;

  std::cout << "Testing prism_h100::runmyh100" << std::endl;

  // Verify against cuBLAS
  memset(C, 0, sizeof(bf16) * max_size * max_size);
  cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  memset(DB, ~0, sizeof(int) * max_size * 128);
  cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128, cudaMemcpyHostToDevice));

  runCublasGemmBF16(m, n, k, dA, dB, dC_ref); // cuBLAS reference
  prism_h100::runmyh100(m, n, k, dA, dB, dC, dDB); // Test kernel

  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());

  cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

  if (!verify_matrix(C_ref, C, m * n)) {
    std::cout << "Failed to pass correctness verification against cuBLAS." << std::endl;
  } else {
    std::cout << "Passed correctness verification!" << std::endl;
  }

  // Benchmark
  cudaEventRecord(start);
  for (int j = 0; j < repeat_times; j++) {
    prism_h100::runmyh100(m, n, k, dA, dB, dC, dDB);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  long flops = (2LL * m) * (n * k);
  printf(
      "Kernel:  Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n",
      elapsed_time / 1000.0 / repeat_times,
      (repeat_times * flops * 1e-9) / elapsed_time, m);

  // Benchmark cuBLAS
  cudaEventRecord(start);
  for (int j = 0; j < repeat_times; j++) {
    runCublasGemmBF16(m, n, k, dA, dB, dC_ref);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  printf(
      "cuBLAS:  Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
      elapsed_time / 1000.0 / repeat_times,
      (repeat_times * flops * 1e-9) / elapsed_time, m);

  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  return 0;
}
