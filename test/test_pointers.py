"""
Tests for pointer types and dereference operations.
"""

import pytest
from prism import (
    prism,
    ptr, const, restrict,
    GLOBAL, SHARED, LOCAL,
    grid, block, thread,
    bf16, double,
    requires,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import (
    PerspectiveError,
    PointerDeclarationError,
    PointerAssignmentError,
    InvalidDereferenceError,
    InvalidWriteThroughError,
)


class TestPointerDeclaration:
    """Test that pointers can only come from function parameters."""

    def test_uninitialized_ptr_in_body_allowed(self):
        """Uninitialized pointer declarations are allowed (for use with unsafe)."""
        @requires(grid[1])
        @prism("device")
        def test_ptr_body():
            p: ptr(int, SHARED) @ block[1]

        func_type, ast_node, _ = registry.get("test_ptr_body")
        check_function(func_type, ast_node)  # Should not raise

    def test_uninitialized_ptr_default_global_allowed(self):
        """Uninitialized pointer (default GLOBAL) declarations are allowed."""
        @requires(grid[1])
        @prism("device")
        def test_ptr_default():
            p: ptr(int) @ grid[1]

        func_type, ast_node, _ = registry.get("test_ptr_default")
        check_function(func_type, ast_node)  # Should not raise

    def test_uninitialized_const_ptr_allowed(self):
        """Uninitialized const(ptr(...)) declarations are allowed."""
        @requires(grid[1])
        @prism("device")
        def test_const_ptr():
            p: const(ptr(float, SHARED)) @ block[1]

        func_type, ast_node, _ = registry.get("test_const_ptr")
        check_function(func_type, ast_node)  # Should not raise

    def test_uninitialized_restrict_ptr_allowed(self):
        """Uninitialized restrict(ptr(...)) declarations are allowed."""
        @requires(grid[1])
        @prism("device")
        def test_restrict_ptr():
            p: restrict(ptr(int, GLOBAL)) @ grid[1]

        func_type, ast_node, _ = registry.get("test_restrict_ptr")
        check_function(func_type, ast_node)  # Should not raise

    def test_ptr_as_parameter_allowed(self):
        """Pointers ARE allowed as function parameters."""
        @requires(grid[1])
        @prism("device")
        def test_ptr_param(p: ptr(int, SHARED) @ block[1]):
            x: int @ thread[1]
            x = p[0]

        func_type, ast_node, _ = registry.get("test_ptr_param")
        check_function(func_type, ast_node)  # Should not raise


class TestPointerAssignment:
    """Test that pointers cannot be on LHS of assignment."""

    def test_ptr_on_lhs_fails(self):
        """Cannot assign to a pointer variable."""
        @requires(grid[1])
        @prism("device")
        def test_ptr_lhs(p: ptr(int, SHARED) @ block[1], q: ptr(int, GLOBAL) @ grid[1]):
            p = q

        func_type, ast_node, _ = registry.get("test_ptr_lhs")
        with pytest.raises(PointerAssignmentError):
            check_function(func_type, ast_node)


class TestPointerDereference:
    """Test pointer dereference p[expr] with perspective narrowing."""

    def test_deref_with_constant_index(self):
        """p[0] where p is at block[1] -> result is block[1]."""
        @requires(grid[1])
        @prism("device")
        def test_deref_const(p: ptr(int, SHARED) @ block[1]):
            x: int @ thread[1]
            x = p[0]  # p[0] is int @ block[1], broader than thread[1]

        func_type, ast_node, _ = registry.get("test_deref_const")
        check_function(func_type, ast_node)  # Should not raise

    def test_deref_with_thread_index(self):
        """p[i] where p is block[1], i is thread[1] -> result is thread[1]."""
        @requires(grid[1])
        @prism("device")
        def test_deref_thread_idx(p: ptr(int, SHARED) @ block[1]):
            i: int @ thread[1]
            x: int @ thread[1]
            x = p[i]  # narrowest(block[1], thread[1]) = thread[1]

        func_type, ast_node, _ = registry.get("test_deref_thread_idx")
        check_function(func_type, ast_node)  # Should not raise

    def test_deref_with_block_index(self):
        """p[j] where p is block[1], j is block[1] -> result is block[1]."""
        @requires(grid[1])
        @prism("device")
        def test_deref_block_idx(p: ptr(int, SHARED) @ block[1]):
            j: int @ block[1]
            x: int @ thread[1]
            x = p[j]  # narrowest(block[1], block[1]) = block[1], broader than thread[1]

        func_type, ast_node, _ = registry.get("test_deref_block_idx")
        check_function(func_type, ast_node)  # Should not raise

    def test_deref_narrowed_to_thread_assigned_to_block_fails(self):
        """p[i] where result is thread[1] cannot be assigned to block[1]."""
        @requires(grid[1])
        @prism("device")
        def test_deref_narrow_fail(p: ptr(int, SHARED) @ block[1]):
            i: int @ thread[1]
            y: int @ block[1]
            y = p[i]  # narrowest(block[1], thread[1]) = thread[1], NOT broader than block[1]

        func_type, ast_node, _ = registry.get("test_deref_narrow_fail")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)

    def test_deref_grid_ptr_with_thread_index(self):
        """Global ptr with thread index -> thread perspective."""
        @requires(grid[1])
        @prism("device")
        def test_deref_grid(p: ptr(float, GLOBAL) @ grid[1]):
            i: int @ thread[1]
            x: float @ thread[1]
            x = p[i]  # narrowest(grid[1], thread[1]) = thread[1]

        func_type, ast_node, _ = registry.get("test_deref_grid")
        check_function(func_type, ast_node)  # Should not raise


class TestNonPointerSubscript:
    """Test that subscripting a non-pointer fails."""

    def test_subscript_int_fails(self):
        """Cannot subscript a non-pointer variable."""
        @requires(grid[1])
        @prism("device")
        def test_subscript_int():
            x: int @ thread[1]
            y: int @ thread[1]
            y = x[0]

        func_type, ast_node, _ = registry.get("test_subscript_int")
        with pytest.raises(InvalidDereferenceError):
            check_function(func_type, ast_node)

    def test_subscript_param_int_fails(self):
        """Cannot subscript a non-pointer parameter."""
        @requires(grid[1])
        @prism("device")
        def test_subscript_param(n: int @ grid[1]):
            x: int @ thread[1]
            x = n[0]

        func_type, ast_node, _ = registry.get("test_subscript_param")
        with pytest.raises(InvalidDereferenceError):
            check_function(func_type, ast_node)


class TestConstantPerspective:
    """Test that constants have grid[1] perspective."""

    def test_constant_to_thread(self):
        """Constants (grid[1]) can be assigned to thread perspective."""
        @requires(grid[1])
        @prism("device")
        def test_const_thread():
            x: int @ thread[1]
            x = 5

        func_type, ast_node, _ = registry.get("test_const_thread")
        check_function(func_type, ast_node)  # Should not raise

    def test_constant_to_block(self):
        """Constants (grid[1]) can be assigned to block perspective."""
        @requires(grid[1])
        @prism("device")
        def test_const_block():
            x: int @ block[1]
            x = 42

        func_type, ast_node, _ = registry.get("test_const_block")
        check_function(func_type, ast_node)  # Should not raise

    def test_constant_to_grid(self):
        """Constants (grid[1]) can be assigned to grid perspective."""
        @requires(grid[1])
        @prism("device")
        def test_const_grid():
            x: int @ grid[1]
            x = 100

        func_type, ast_node, _ = registry.get("test_const_grid")
        check_function(func_type, ast_node)  # Should not raise

    def test_constant_in_expression(self):
        """Constants in expressions don't narrow the perspective."""
        @requires(grid[1])
        @prism("device")
        def test_const_expr():
            x: int @ thread[1]
            y: int @ block[1]
            x = y + 5

        func_type, ast_node, _ = registry.get("test_const_expr")
        check_function(func_type, ast_node)  # Should not raise


class TestWriteThrough:
    """Test write-through p[expr] = value."""

    def test_write_through_thread_ptr_allowed(self):
        """Write through pointer at thread[1] is allowed."""
        @requires(grid[1])
        @prism("device")
        def test_write_ok(p: ptr(int, GLOBAL) @ thread[1]):
            x: int @ thread[1]
            p[0] = x

        func_type, ast_node, _ = registry.get("test_write_ok")
        check_function(func_type, ast_node)  # Should not raise

    def test_write_through_non_thread_ptr_fails(self):
        """Cannot write through pointer not at thread[1]."""
        @requires(grid[1])
        @prism("device")
        def test_write_block_ptr(p: ptr(int, SHARED) @ block[1]):
            x: int @ thread[1]
            p[0] = x

        func_type, ast_node, _ = registry.get("test_write_block_ptr")
        with pytest.raises(InvalidWriteThroughError):
            check_function(func_type, ast_node)

    def test_write_through_non_ptr_fails(self):
        """Cannot write through non-pointer."""
        @requires(grid[1])
        @prism("device")
        def test_write_non_ptr(n: int @ thread[1]):
            n[0] = 5

        func_type, ast_node, _ = registry.get("test_write_non_ptr")
        with pytest.raises(InvalidDereferenceError):
            check_function(func_type, ast_node)
