#include "my_kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// My Custom Kernels
void run_my_sgemm_1(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const int BLOCK_SIZE = 256;  // 1D block size
  int total_threads = M * N;
  int gridDim = CEIL_DIV(total_threads, BLOCK_SIZE);
  my_sgemm_kernel_1<<<gridDim, BLOCK_SIZE>>>(M, N, K, alpha, A, B, beta, C, BLOCK_SIZE);
}

void run_my_sgemm_2(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const int TILE_SIZE = 32;  // Tile size for coalescing (32x32 tiles)
  const int BLOCK_SIZE = TILE_SIZE * TILE_SIZE;  // 1024 threads per block
  int total_threads = M * N;
  int gridDim = CEIL_DIV(total_threads, BLOCK_SIZE);
  my_sgemm_kernel_2<<<gridDim, BLOCK_SIZE>>>(M, N, K, alpha, A, B, beta, C, TILE_SIZE);
}

void run_my_sgemm_3(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const int BLOCKSIZE = 32;
  const int num_blocks_M = CEIL_DIV(M, BLOCKSIZE);
  const int num_blocks_N = CEIL_DIV(N, BLOCKSIZE);
  const int total_blocks = num_blocks_M * num_blocks_N;
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  my_sgemm_kernel_3<BLOCKSIZE>
      <<<total_blocks, blockDim>>>(M, N, K, alpha, A, B, beta, C, num_blocks_N);
}

void run_my_sgemm_4(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    const int num_blocks_M = CEIL_DIV(M, BM);
    const int num_blocks_N = CEIL_DIV(N, BN);
    const int total_blocks = num_blocks_M * num_blocks_N;
    dim3 blockDim((BM * BN) / (TM * TN));
    const size_t smem_size = (BM * BK + BK * BN) * sizeof(float);
    my_sgemm_kernel_4<BM, BN, BK, TM, TN>
        <<<total_blocks, blockDim, smem_size>>>(M, N, K, alpha, A, B, beta, C, num_blocks_N);
  } else {
    const uint BM = 64;
    const uint BN = 64;
    const int num_blocks_M = CEIL_DIV(M, BM);
    const int num_blocks_N = CEIL_DIV(N, BN);
    const int total_blocks = num_blocks_M * num_blocks_N;
    dim3 blockDim((BM * BN) / (TM * TN));
    const size_t smem_size = (BM * BK + BK * BN) * sizeof(float);
    my_sgemm_kernel_4<BM, BN, BK, TM, TN>
        <<<total_blocks, blockDim, smem_size>>>(M, N, K, alpha, A, B, beta, C, num_blocks_N);
  }
}

void run_my_sgemm_5(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  // Settings for A6000
  const uint K5_NUM_THREADS = 128;
  const uint K5_BN = 128;
  const uint K5_BM = 128;
  const uint K5_BK = 16;
  const uint K5_WN = 64;
  const uint K5_WM = 64;
  const uint K5_WNITER = 4;
  const uint K5_TN = 4;
  const uint K5_TM = 8;
  dim3 blockDim(K5_NUM_THREADS);

  constexpr uint NUM_WARPS = K5_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K5_BN % K5_WN == 0) and (K5_BM % K5_WM == 0));
  static_assert((K5_BN / K5_WN) * (K5_BM / K5_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K5_WM * K5_WN) % (32 * K5_TM * K5_TN * K5_WNITER) ==
                0);
  constexpr uint K5_WMITER =
      (K5_WM * K5_WN) / (32 * K5_TM * K5_TN * K5_WNITER);
  // warpsubtile in warptile
  static_assert((K5_WM % K5_WMITER == 0) and (K5_WN % K5_WNITER == 0));

  static_assert((K5_NUM_THREADS * 4) % K5_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K5_NUM_THREADS * 4) % K5_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K5_BN % (16 * K5_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K5_BM % (16 * K5_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K5_BM * K5_BK) % (4 * K5_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K5_BN * K5_BK) % (4 * K5_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  const int num_blocks_M = CEIL_DIV(M, K5_BM);
  const int num_blocks_N = CEIL_DIV(N, K5_BN);
  const int total_blocks = num_blocks_M * num_blocks_N;
  const size_t shmem_size = (K5_BM * K5_BK + K5_BK * K5_BN) * sizeof(float);
  my_sgemm_kernel_5<K5_BM, K5_BN, K5_BK, K5_WM, K5_WN, K5_WNITER, K5_TM,
                    K5_TN, K5_NUM_THREADS>
      <<<total_blocks, blockDim, shmem_size>>>(M, N, K, alpha, A, B, beta, C, num_blocks_N);
}

void run_my_kernel(int my_kernel_num, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  switch (my_kernel_num) {
  case 1:
    run_my_sgemm_1(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_my_sgemm_2(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_my_sgemm_3(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    run_my_sgemm_4(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    run_my_sgemm_5(M, N, K, alpha, A, B, beta, C);
    break;
  // Add more custom kernels here
  default:
    throw std::invalid_argument("Unknown my_kernel number");
  }
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}