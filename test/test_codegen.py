"""
Tests for the CUDA C code generator.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    group, split, partition, claim,
    ptr, const, restrict, constexpr,
    shared, double, bf16,
)
from prism.codegen import generate


# ---------------------------------------------------------------------------
# Function Signatures
# ---------------------------------------------------------------------------

class TestFunctionSignatures:

    def test_global_function(self):
        @requires(grid[1], block[4], thread[4])
        @prism("global")
        def kern_global(x: ptr(int) @ thread[1], n: int @ grid[1]):
            pass

        code = generate("kern_global")
        assert "__global__" in code
        assert "void kern_global(int* x, int n)" in code
        assert "extern __shared__ __align__(128) char __smem[];" in code
        assert "blockIdx.x" in code
        assert "blockDim.x" in code

    def test_device_function(self):
        @requires(block[1], thread[4])
        @prism("device")
        def kern_device(x: ptr(int) @ thread[1]):
            pass

        code = generate("kern_device")
        assert "__device__" in code
        assert "int __block_id, int __thread_id, char* __smem, int* x" in code
        assert "__block_id_0 = __block_id;" in code
        assert "__thread_id_0 = __thread_id;" in code

    def test_pointer_params(self):
        @requires(block[1], thread[1])
        @prism("device")
        def kern_ptrs(
            a: ptr(float) @ thread[1],
            b: const(ptr(int)) @ thread[1],
            c: restrict(ptr(double)) @ thread[1],
        ):
            pass

        code = generate("kern_ptrs")
        assert "float* a" in code
        assert "const int* b" in code
        assert "double* __restrict__ c" in code

    def test_template_params(self):
        @requires(block[1], thread[1])
        @prism("device")
        def kern_tmpl(
            x: int @ thread[1],
            template=[("N", int), ("M", constexpr(int))],
        ):
            pass

        code = generate("kern_tmpl")
        assert "template<int N, int M>" in code
        # template param should not appear in the function's C parameter list
        assert "int __block_id, int __thread_id, char* __smem, int x" in code

    def test_return_type(self):
        @requires(block[1], thread[1])
        @prism("device")
        def kern_ret(x: int @ thread[1]) -> int @ thread[1]:
            return x

        code = generate("kern_ret")
        assert "__device__ int kern_ret(" in code
        assert "return x;" in code


# ---------------------------------------------------------------------------
# Variable Declarations
# ---------------------------------------------------------------------------

class TestVariableDeclarations:

    def test_scalar_with_init(self):
        @requires(block[1], thread[1])
        @prism("device")
        def decl_init(x: int @ thread[1]):
            y: int @ thread[1] = 42

        code = generate("decl_init")
        assert "int y = 42;" in code

    def test_scalar_no_init(self):
        @requires(block[1], thread[1])
        @prism("device")
        def decl_noinit():
            y: int @ thread[1]

        code = generate("decl_noinit")
        assert "int y;" in code

    def test_local_array(self):
        @requires(block[1], thread[1])
        @prism("device")
        def decl_arr():
            buf: int[16] @ thread[1]

        code = generate("decl_arr")
        assert "int buf[16];" in code

    def test_shared_memory(self):
        @requires(grid[1], block[1], thread[1], smem=1024)
        @prism("device")
        def decl_smem():
            buf: shared(int[64]) @ block[1]

        code = generate("decl_smem")
        assert "int (&buf)[64]" in code
        assert "__smem + 0" in code

    def test_shared_memory_offset(self):
        @requires(grid[1], block[1], thread[1], smem=2048)
        @prism("device")
        def decl_smem_offset():
            a: shared(int[64]) @ block[1]
            b: shared(float[32]) @ block[1]

        code = generate("decl_smem_offset")
        assert "__smem + 0" in code
        assert "__smem + 256" in code  # 64 * 4 = 256 bytes


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class TestExpressions:

    def test_binary_ops(self):
        @requires(block[1], thread[1])
        @prism("device")
        def expr_binops(x: int @ thread[1], y: int @ thread[1]):
            z: int @ thread[1] = x + y * 2

        code = generate("expr_binops")
        assert "int z = (x + (y * 2));" in code

    def test_id_expression(self):
        @requires(grid[1], block[4], thread[4])
        @prism("device")
        def expr_id():
            x: int @ thread[1] = id()

        code = generate("expr_id")
        assert "__thread_id_0" in code

    def test_id_block_level(self):
        @requires(grid[1], block[4], thread[4])
        @prism("device")
        def expr_id_block():
            x: int @ block[4] = id()

        code = generate("expr_id_block")
        assert "__block_id_0" in code

    def test_comparisons(self):
        @requires(thread[1])
        @prism("device")
        def expr_cmp(x: int @ thread[1]):
            if x > 0:
                pass

        code = generate("expr_cmp")
        assert "(x > 0)" in code

    def test_subscript(self):
        @requires(block[1], thread[1])
        @prism("device")
        def expr_sub(p: ptr(int) @ thread[1]):
            x: int @ thread[1] = p[0]

        code = generate("expr_sub")
        assert "int x = p[0];" in code

    def test_boolean_constant(self):
        @requires(block[1], thread[1])
        @prism("device")
        def expr_bool():
            x: bool @ thread[1] = True

        code = generate("expr_bool")
        assert "bool x = true;" in code


# ---------------------------------------------------------------------------
# Control Flow
# ---------------------------------------------------------------------------

class TestControlFlow:

    def test_for_range_one_arg(self):
        @requires(block[1], thread[1])
        @prism("device")
        def cf_for1(n: int @ grid[1]):
            for i in range(n):
                pass

        code = generate("cf_for1")
        assert "for (int i = 0; i < n; i++)" in code

    def test_for_range_two_args(self):
        @requires(block[1], thread[1])
        @prism("device")
        def cf_for2():
            for i in range(1, 10):
                pass

        code = generate("cf_for2")
        assert "for (int i = 1; i < 10; i++)" in code

    def test_for_range_three_args(self):
        @requires(block[1], thread[1])
        @prism("device")
        def cf_for3():
            for i in range(0, 100, 4):
                pass

        code = generate("cf_for3")
        assert "for (int i = 0; i < 100; i += 4)" in code

    def test_while_loop(self):
        @requires(thread[1])
        @prism("device")
        def cf_while(x: int @ thread[1]):
            while x > 0:
                x = x - 1

        code = generate("cf_while")
        assert "while ((x > 0))" in code
        assert "x = (x - 1);" in code

    def test_if_elif_else(self):
        @requires(block[1], thread[1])
        @prism("device")
        def cf_ifelifelse(x: int @ block[1]):
            if x > 10:
                x = 1
            elif x > 5:
                x = 2
            else:
                x = 3

        code = generate("cf_ifelifelse")
        assert "if ((x > 10))" in code
        assert "} else if ((x > 5))" in code
        assert "} else {" in code

    def test_break_continue(self):
        @requires(block[1], thread[1])
        @prism("device")
        def cf_brk_cont(n: int @ block[1]):
            for i in range(n):
                if n > 5:
                    break
                if n > 3:
                    continue

        code = generate("cf_brk_cont")
        assert "break;" in code
        assert "continue;" in code


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------

class TestGroup:

    def test_single_group_block(self):
        @requires(grid[1], block[4], thread[4])
        @prism("device")
        def grp_block():
            with group(block[2]):
                x: int @ block[2] = id()

        code = generate("grp_block")
        assert "__block_id_1 = __block_id_0 % 2;" in code
        assert "__thread_id_1 = __block_id_1 * blockDim.x + threadIdx.x;" in code

    def test_single_group_thread(self):
        @requires(grid[1], block[4], thread[4])
        @prism("device")
        def grp_thread():
            with group(thread[2]):
                x: int @ thread[2] = id()

        code = generate("grp_thread")
        assert "__block_id_1 = __block_id_0;" in code
        assert "__thread_id_1 = __thread_id_0 % 2;" in code

    def test_nested_groups(self):
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def grp_nested():
            with group(block[2]):
                with group(thread[4]):
                    x: int @ thread[4] = id()

        code = generate("grp_nested")
        assert "__block_id_1 = __block_id_0 % 2;" in code
        assert "__thread_id_2 = __thread_id_1 % 4;" in code


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

class TestSplit:

    def test_two_case_block_split(self):
        @requires(grid[1], block[8], thread[8])
        @prism("device")
        def splt_block():
            match split(block):
                case 2:
                    with group(thread[8]):
                        pass
                case 2:
                    with group(thread[8]):
                        pass
                case 4:
                    with group(thread[8]):
                        pass

        code = generate("splt_block")
        # First case: no offset
        assert "__block_id_1 = __block_id_0;" in code
        assert "__block_id_1 >= 0 && __block_id_1 < 2" in code
        # Second case: offset 2
        assert "__block_id_1 = __block_id_0 - 2;" in code
        # Third case: offset 4
        assert "__block_id_1 = __block_id_0 - 4;" in code
        assert "__block_id_1 < 4" in code

    def test_thread_level_split(self):
        @requires(grid[1], block[1], thread[4])
        @prism("device")
        def splt_thread():
            match split(thread):
                case 2:
                    pass
                case 2:
                    pass

        code = generate("splt_thread")
        assert "__thread_id_1 = __thread_id_0;" in code
        assert "__thread_id_1 >= 0 && __thread_id_1 < 2" in code
        assert "__thread_id_1 = __thread_id_0 - 2;" in code


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

class TestPartition:

    def test_partition_with_offset(self):
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def part_offset(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], offset=id() * n) as my_data:
                my_data[0] = 42

        code = generate("part_offset")
        assert "int* my_data = data + (" in code
        assert "my_data[0] = 42;" in code

    def test_partition_with_lambda(self):
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def part_lambda(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], f=lambda i: i + id() * n) as my_data:
                my_data[0] = 42

        code = generate("part_lambda")
        # No pointer declaration — index is transformed per-access
        assert "int* my_data" not in code
        assert "data[(0 + (__thread_id_0 * n))] = 42;" in code

    def test_partition_lambda_read(self):
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def part_lambda_read(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], f=lambda i: i + id() * n) as my_data:
                x: int @ thread[1] = my_data[3]

        code = generate("part_lambda_read")
        assert "int x = data[(3 + (__thread_id_0 * n))];" in code

    def test_partition_lambda_augassign(self):
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def part_lambda_aug(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], f=lambda i: i + id() * n) as my_data:
                my_data[0] += 1

        code = generate("part_lambda_aug")
        assert "data[(0 + (__thread_id_0 * n))] += 1;" in code

    def test_partition_offset_passed_to_function(self):
        """offset= creates a real pointer that can be passed to functions."""
        @requires(block[1], thread[1])
        @prism("device")
        def callee_off(x: ptr(int) @ thread[1]):
            pass

        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def part_offset_call(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], offset=id() * n) as my_data:
                callee_off(my_data)

        code = generate("part_offset_call")
        assert "int* my_data = data + (" in code
        assert "callee_off(__block_id_0, __thread_id_0, __smem, my_data);" in code


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

class TestClaim:

    def test_claim_generates_same_as_partition(self):
        @requires(block[4], thread[1])
        @prism("device")
        def claim_basic(data: ptr(int) @ block[4]):
            with claim(data, p=thread[1]) as my_data:
                match split(block):
                    case 2:
                        x: int @ thread[1] = my_data[0]
                    case 2:
                        pass

        code = generate("claim_basic")
        # Claim produces pointer arithmetic just like partition
        assert "int* my_data = data + (0);" in code


# ---------------------------------------------------------------------------
# Function Calls
# ---------------------------------------------------------------------------

class TestFunctionCalls:

    def test_device_call(self):
        @requires(block[1], thread[4])
        @prism("device")
        def callee_dev(x: int @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_dev(a: int @ thread[8]):
            callee_dev(a)

        code = generate("caller_dev")
        assert "callee_dev(__block_id_0, __thread_id_0, __smem, a);" in code

    def test_template_call(self):
        @requires(block[1], thread[1])
        @prism("device")
        def callee_tmpl(
            x: int @ thread[1],
            template=[("N", int)],
        ):
            pass

        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def caller_tmpl(n: int @ grid[1]):
            callee_tmpl(n, template=[16])

        code = generate("caller_tmpl")
        assert "callee_tmpl<16>(__block_id_0, __thread_id_0, __smem, n);" in code


# ---------------------------------------------------------------------------
# End-to-End
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_vector_add(self):
        @requires(thread[1])
        @prism("global")
        def vec_add(
            a: ptr(float) @ thread[1],
            b: ptr(float) @ thread[1],
            c: ptr(float) @ thread[1],
            n: int @ thread[1],
        ):
            idx: int @ thread[1] = id()
            if idx < n:
                c[idx] = a[idx] + b[idx]

        code = generate("vec_add")
        assert "__global__" in code
        assert "float* a, float* b, float* c, int n" in code
        assert "int idx = __thread_id_0;" in code
        assert "c[idx] = (a[idx] + b[idx]);" in code
        assert "if ((idx < n))" in code
