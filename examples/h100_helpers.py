"""H100 matmul helper device functions — ported from ccuda_py h100_matmul.py."""

from prism import (
    prism, grid, block, thread,
    requires, group, split, partition, claim,
    ptr, const, constexpr, shared, unsafe, attr,
    bf16, uint32_t, uint64_t, void,
    intrinsic, wgmma_async,
    generate,
)


@attr("static __forceinline__")
@requires(thread[128])
@prism("device")
def warpgroup_arrive():
    with group(thread[128]):
        unsafe("""asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");""")
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def load_async(dst: ptr(bf16) @ thread[1],
               src_tma_map_ptr: ptr(const(void)) @ thread[1],
               bar: ptr(uint64_t) @ block[1],
               global_col_idx: int @ thread[1],
               global_row_idx: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
            uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map_ptr);
            uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
                : "memory"
            );
            """)
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def expect_bytes(bar: ptr(uint64_t) @ block[1],
                 bytes: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
             uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
             asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n"
             :: "r"(bar_ptr), "r"(bytes));
        """)
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def wait(bar: ptr(uint64_t) @ block[1],
         kPhaseBit: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
             uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
             asm volatile (
                 "{\\n"
                 ".reg .pred                P1;\\n"
                 "LAB_WAIT:\\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\\n"
                 "@P1                       bra.uni DONE;\\n"
                 "bra.uni                   LAB_WAIT;\\n"
                 "DONE:\\n"
                 "}\\n"
                 :: "r"(mbar_ptr),
                 "r"(kPhaseBit)
            );
            """)
    return


@attr("static __forceinline__")
@requires(thread[256])
@prism("device")
def wait_cluster(bar: ptr(uint64_t) @ block[1],
                 kPhaseBit: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
             uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
             asm volatile (
                 "{\\n"
                 ".reg .pred                P1;\\n"
                 "LAB_WAIT:\\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\\n"
                 "@P1                       bra.uni DONE;\\n"
                 "bra.uni                   LAB_WAIT;\\n"
                 "DONE:\\n"
                 "}\\n"
                 :: "r"(mbar_ptr),
                 "r"(kPhaseBit)
            );
            """)
    return


@attr("static __forceinline__")
@requires(thread[128])
@prism("device")
def warpgroup_reg_alloc(template=[("RegCount", uint32_t)]):
    with group(thread[128]):
        unsafe("""asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" : : "n"(RegCount));""")
    return


@attr("static __forceinline__")
@requires(thread[128])
@prism("device")
def warpgroup_reg_dealloc(template=[("RegCount", uint32_t)]):
    with group(thread[128]):
        unsafe("""asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" : : "n"(RegCount));""")
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def memset_wrapper(dst: ptr(float) @ thread[1], value: float @ thread[1], size: int @ thread[1]):
    with group(thread[1]):
        unsafe("memset(dst, value, size);")
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def arrive(bar: ptr(uint64_t) @ thread[1],
           count: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            asm volatile (
                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\\n"
                :: "r"(bar_ptr), "r"(count)
                : "memory"
            );
            """)
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def init_barrier(bar: ptr(uint64_t) @ thread[1],
                 thread_count: int @ thread[1],
                 transaction_count: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
           uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;\\n"
                :: "r"(bar_ptr), "r"(thread_count+transaction_count)
            );
            """)
    return


@attr("static __forceinline__")
@requires(thread[1])
@prism("device")
def load_tma(dst: ptr(bf16) @ thread[1],
             src_tma_map_ptr: ptr(const(void)) @ thread[1],
             global_col_idx: int @ thread[1],
             global_row_idx: int @ thread[1]):
    with group(thread[1]):
        unsafe("""
            // Barrier-free TMA load — codegen inserts barrier arg
            uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map_ptr);
            uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3, %4, 0, 0}], [0];"
                :
                : "r"(dst_ptr), "l"(tma_ptr),
                "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
                : "memory"
            );
            """)
    return


if __name__ == "__main__":
    for name in ["warpgroup_arrive", "load_async", "load_tma", "expect_bytes", "wait",
                 "wait_cluster", "warpgroup_reg_alloc", "warpgroup_reg_dealloc",
                 "memset_wrapper", "arrive", "init_barrier"]:
        print(generate(name))
