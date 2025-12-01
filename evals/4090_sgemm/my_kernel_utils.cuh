#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
const int WARPSIZE_our = 32; // warpSize is not constexpr

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem_our(int a, int b, char *c, int N, int K, const float *A, const float *B,
                                 float *As, float *Bs)
{

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA)
  {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
  {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem_our(int, int, char*, float *regM, float *regN, float *threadResults, const float *As,
                    const float *Bs, const uint warpRow, const uint warpCol)
{
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE_our;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
  {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    {
      for (uint i = 0; i < TM; ++i)
      {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
    {
      for (uint i = 0; i < TN; ++i)
      {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
      {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
        {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
          {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}
