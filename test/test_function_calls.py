"""
Tests for function composition and call checking.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    ptr, const,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import (
    CallSmemError,
    CallBoundError,
    CallArgPerspectiveError,
    PerspectiveError,
)


class TestFunctionCallSmem:
    """Test smem checking for function calls."""

    def test_call_with_sufficient_smem(self):
        """Caller with enough smem can call callee."""
        @requires(block[1], thread[4], smem=512)
        @prism("device")
        def callee_smem():
            pass

        @requires(grid[1], block[2], thread[8], smem=1024)
        @prism("device")
        def caller_sufficient_smem():
            callee_smem()

        func_type, ast_node, _ = registry.get("caller_sufficient_smem")
        check_function(func_type, ast_node)  # Should not raise

    def test_call_exceeds_smem(self):
        """Caller without enough smem cannot call callee."""
        @requires(block[1], thread[4], smem=512)
        @prism("device")
        def callee_needs_smem():
            pass

        @requires(grid[1], block[2], thread[8], smem=256)
        @prism("device")
        def caller_insufficient_smem():
            callee_needs_smem()

        func_type, ast_node, _ = registry.get("caller_insufficient_smem")
        with pytest.raises(CallSmemError):
            check_function(func_type, ast_node)


class TestFunctionCallBound:
    """Test perspective bound checking for function calls."""

    def test_call_with_broader_bound(self):
        """Caller with broader bound can call callee."""
        @requires(block[2], thread[4])
        @prism("device")
        def callee_bound():
            pass

        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def caller_broader_bound():
            callee_bound()

        func_type, ast_node, _ = registry.get("caller_broader_bound")
        check_function(func_type, ast_node)  # Should not raise

    def test_call_with_exact_bound(self):
        """Caller with exact bound can call callee."""
        @requires(block[2], thread[4])
        @prism("device")
        def callee_exact():
            pass

        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def caller_exact_bound():
            callee_exact()

        func_type, ast_node, _ = registry.get("caller_exact_bound")
        check_function(func_type, ast_node)  # Should not raise

    def test_call_with_narrower_bound_fails(self):
        """Caller with narrower bound cannot call callee."""
        @requires(block[4], thread[8])
        @prism("device")
        def callee_wide():
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_narrow_bound():
            callee_wide()

        func_type, ast_node, _ = registry.get("caller_narrow_bound")
        with pytest.raises(CallBoundError):
            check_function(func_type, ast_node)

    def test_call_missing_level_fails(self):
        """Caller missing a level callee needs cannot call."""
        @requires(block[2], thread[4])
        @prism("device")
        def callee_needs_block():
            pass

        @requires(thread[8])
        @prism("device")
        def caller_no_block():
            callee_needs_block()

        func_type, ast_node, _ = registry.get("caller_no_block")
        with pytest.raises(CallBoundError):
            check_function(func_type, ast_node)


class TestFunctionCallArguments:
    """Test argument perspective checking for function calls."""

    def test_non_ptr_arg_broader_perspective(self):
        """Non-ptr argument with broader perspective is valid."""
        @requires(block[1], thread[4])
        @prism("device")
        def callee_int(x: int @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_broader_arg(a: int @ thread[8]):
            callee_int(a)

        func_type, ast_node, _ = registry.get("caller_broader_arg")
        check_function(func_type, ast_node)  # Should not raise

    def test_non_ptr_arg_narrower_perspective_fails(self):
        """Non-ptr argument with narrower perspective fails."""
        @requires(block[1], thread[4])
        @prism("device")
        def callee_int_narrow(x: int @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_narrow_arg(a: int @ thread[2]):
            callee_int_narrow(a)

        func_type, ast_node, _ = registry.get("caller_narrow_arg")
        with pytest.raises(CallArgPerspectiveError):
            check_function(func_type, ast_node)

    def test_const_ptr_arg_broader_perspective(self):
        """Const ptr argument with broader perspective is valid."""
        @requires(block[1], thread[4])
        @prism("device")
        def callee_const_ptr(p: const(ptr(float)) @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_const_broader(p: ptr(float) @ thread[8]):
            callee_const_ptr(p)

        func_type, ast_node, _ = registry.get("caller_const_broader")
        check_function(func_type, ast_node)  # Should not raise

    def test_non_const_ptr_exact_match(self):
        """Non-const ptr argument with exact perspective is valid."""
        @requires(block[1], thread[4])
        @prism("device")
        def callee_mut_ptr(p: ptr(float) @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_mut_exact(p: ptr(float) @ thread[4]):
            callee_mut_ptr(p)

        func_type, ast_node, _ = registry.get("caller_mut_exact")
        check_function(func_type, ast_node)  # Should not raise

    def test_non_const_ptr_mismatch_fails(self):
        """Non-const ptr argument with different perspective fails."""
        @requires(block[1], thread[4])
        @prism("device")
        def callee_mut_ptr_strict(p: ptr(float) @ thread[4]):
            pass

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_mut_mismatch(p: ptr(float) @ thread[8]):
            callee_mut_ptr_strict(p)

        func_type, ast_node, _ = registry.get("caller_mut_mismatch")
        with pytest.raises(CallArgPerspectiveError):
            check_function(func_type, ast_node)


class TestFunctionCallReturn:
    """Test return value perspective checking for function calls."""

    def test_assign_from_call_valid(self):
        """Assigning call result to variable with narrower perspective is valid."""
        @requires(block[1], thread[4])
        @prism("device")
        def get_val() -> int @ thread[4]:
            return id()

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_assign_valid():
            x: int @ thread[4]
            x = get_val()

        func_type, ast_node, _ = registry.get("caller_assign_valid")
        check_function(func_type, ast_node)  # Should not raise

    def test_assign_from_call_broader_return(self):
        """Assigning call result with broader return to narrower var is valid."""
        @requires(block[1], thread[4])
        @prism("device")
        def get_broad() -> int @ thread[4]:
            return id()

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_assign_broader():
            x: int @ thread[2]
            x = get_broad()

        func_type, ast_node, _ = registry.get("caller_assign_broader")
        check_function(func_type, ast_node)  # Should not raise

    def test_assign_from_call_narrower_return_fails(self):
        """Assigning call result with narrower return to broader var fails."""
        @requires(block[1], thread[4])
        @prism("device")
        def get_narrow() -> int @ thread[4]:
            return id()

        @requires(grid[1], block[2], thread[8])
        @prism("device")
        def caller_assign_narrow():
            x: int @ thread[8]
            x = get_narrow()

        func_type, ast_node, _ = registry.get("caller_assign_narrow")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)
