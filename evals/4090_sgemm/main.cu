#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv)
{
  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL)
  {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  // Removed verbose device info print

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle))
  {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // Size sweep: 256, 512, 1024, 2048, 4096, 8192
  std::vector<int> SIZE = {256, 512, 1024, 2048, 4096, 8192};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 10;

  // Benchmark all custom kernels
  std::vector<int> my_kernels_to_run = {1, 2, 3, 4, 5};

  // Check correctness only once at size 1024
  int correctness_size = 1024;
  bool correctness_checked = false;

  std::cout << "\n====================================\n";
  std::cout << "Checking correctness at size " << correctness_size << "\n";
  std::cout << "====================================\n";

  m = n = k = correctness_size;

  // Reset dC and dC_ref to original C values
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * m * n,
                       cudaMemcpyHostToDevice));

  // Compute cuBLAS reference for verification
  run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
  cudaCheck(cudaDeviceSynchronize());

  // Verify correctness for each custom kernel
  for (int my_kern_num : my_kernels_to_run)
  {
    std::cout << "\n[My Kernel " << my_kern_num << " - Correctness Check]" << std::endl;

    // Reset dC to original C values
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n,
                         cudaMemcpyHostToDevice));

    // Run kernel
    run_my_kernel(my_kern_num, m, n, k, alpha, dA, dB, beta, dC);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    // Copy results to temporary host arrays for verification
    float *C_test = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(C_test, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    if (!verify_matrix(C_ref, C_test, m * n))
    {
      std::cout
          << "Failed to pass the correctness verification against NVIDIA "
             "cuBLAS."
          << std::endl;
      if (m <= 128)
      {
        std::cout << " Logging faulty output into " << errLogFile << "\n";
        std::ofstream fs;
        fs.open(errLogFile);
        fs << "A:\n";
        print_matrix(A, m, n, fs);
        fs << "B:\n";
        print_matrix(B, m, n, fs);
        fs << "C:\n";
        print_matrix(C_test, m, n, fs);
        fs << "Should:\n";
        print_matrix(C_ref, m, n, fs);
      }
      free(C_test);
      exit(EXIT_FAILURE);
    }
    std::cout << "Correctness: PASSED" << std::endl;
    free(C_test);
  }

  std::cout << "\n====================================\n";
  std::cout << "Starting Performance Sweep\n";
  std::cout << "====================================\n";

  // Performance sweep across all sizes
  for (int size : SIZE)
  {
    m = n = k = size;

    std::cout << "\n====================================\n";
    std::cout << "Size: " << m << "x" << n << "x" << k << "\n";
    std::cout << "====================================\n";

    // Reset dC and dC_ref to original C values for this size
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * m * n,
                         cudaMemcpyHostToDevice));

    // Benchmark cuBLAS
    std::cout << "\n[cuBLAS FP32]" << std::endl;
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++)
    {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);

    // Benchmark custom kernels
    for (int my_kern_num : my_kernels_to_run)
    {
      std::cout << "\n[My Kernel " << my_kern_num << "]" << std::endl;

      // Reset dC to original C values
      cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n,
                           cudaMemcpyHostToDevice));

      // Benchmark
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++)
      {
        run_my_kernel(my_kern_num, m, n, k, alpha, dA, dB, beta, dC);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
    }
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
