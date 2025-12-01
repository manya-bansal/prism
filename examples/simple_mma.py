"""Faithful port of ccuda_py simple_mma: warp-level MMA using inline PTX."""

from prism import (
    prism, grid, block, thread,
    requires, group, partition,
    ptr, const, uint32_t, intrinsic,
    generate,
)


@requires(thread[32])
@prism("device")
def simple_mma(a: ptr(const(float)) @ thread[32],
               b: ptr(const(float)) @ thread[32],
               c: ptr(float) @ thread[32]):
    with group(thread[32]):
        stride: int @ thread[32] = 8

        tid: int @ thread[1] = id()

        regA: uint32_t[4] @ thread[1]
        row_a: int @ thread[1] = tid / 4
        col_a: int @ thread[1] = tid % 4
        offsetA: int @ thread[1] = row_a * stride + col_a

        regA[0] = __float_as_uint(a[offsetA + 0])
        regA[1] = __float_as_uint(a[offsetA + 8 * stride])
        regA[2] = __float_as_uint(a[offsetA + 4])
        regA[3] = __float_as_uint(a[offsetA + 8 * stride + 4])

        regB: uint32_t[2] @ thread[1]
        row_b: int @ thread[1] = tid % 4
        col_b: int @ thread[1] = tid / 4
        offsetB: int @ thread[1] = row_b * stride + col_b

        regB[0] = __float_as_uint(b[offsetB + 0])
        regB[1] = __float_as_uint(b[offsetB + 4 * stride])

        regC: uint32_t[4] @ thread[1]
        regC[0] = 0
        regC[1] = 0
        regC[2] = 0
        regC[3] = 0

        intrinsic.mma(
            regA[0], regA[1], regA[2], regA[3],
            regB[0], regB[1],
            regC[0], regC[1], regC[2], regC[3],
            out=[regC[0], regC[1], regC[2], regC[3]]
        )

        row_c: uint32_t @ thread[1] = tid / 4
        col_c: int @ thread[1] = tid % 4
        offsetC: int @ thread[1] = row_c * stride + 2 * col_c

        with partition(c, p=thread[1], offset=offsetC) as c_thread:
            with group(thread[1]):
                c_thread[0] += __uint_as_float(regC[0])
                c_thread[1] += __uint_as_float(regC[1])
                c_thread[8 * stride] += __uint_as_float(regC[2])
                c_thread[8 * stride + 1] += __uint_as_float(regC[3])


if __name__ == "__main__":
    print(generate("simple_mma"))
