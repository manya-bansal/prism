"""
Tests for the type checker assignment rules.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    bf16, double,
    requires,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import PerspectiveError


class TestValidAssignments:
    """Test valid assignments that should pass."""

    def test_read_from_broader_level(self):
        """thread = block (different levels)"""
        @requires(grid[1])
        @prism("device")
        def test_func():
            x: int @ thread[1]
            y: int @ block[1]
            x = y + id()

        func_type, ast_node, _ = registry.get("test_func")
        check_function(func_type, ast_node)  # Should not raise

    def test_read_from_same_perspective(self):
        """thread[1] = thread[1] (same perspective)"""
        @requires(grid[1])
        @prism("device")
        def test_func2():
            a: float @ thread[1]
            b: float @ thread[1]
            a = b

        func_type, ast_node, _ = registry.get("test_func2")
        check_function(func_type, ast_node)  # Should not raise

    def test_read_from_broader_index(self):
        """block[2] = block[4] (broader index)"""
        @requires(grid[1])
        @prism("device")
        def test_func3():
            p: int @ block[2]
            q: int @ block[4]
            p = q

        func_type, ast_node, _ = registry.get("test_func3")
        check_function(func_type, ast_node)  # Should not raise

    def test_complex_expression_multiple_rhs(self):
        """thread[1] = block[1] + block[2] (multiple RHS variables)"""
        @requires(grid[1])
        @prism("device")
        def test_func4():
            result: int @ thread[1]
            v1: int @ block[1]
            v2: int @ block[2]
            result = v1 + v2

        func_type, ast_node, _ = registry.get("test_func4")
        check_function(func_type, ast_node)  # Should not raise

    def test_grid_to_thread(self):
        """thread[1] = grid[1] (most broad to most narrow)"""
        @requires(grid[1])
        @prism("device")
        def test_func5():
            local: bf16 @ thread[1]
            global_val: bf16 @ grid[1]
            local = global_val

        func_type, ast_node, _ = registry.get("test_func5")
        check_function(func_type, ast_node)  # Should not raise


class TestInvalidAssignments:
    """Test invalid assignments that should fail."""

    def test_write_from_narrower_to_broader_level(self):
        """block = thread (should fail)"""
        @requires(grid[1])
        @prism("device")
        def test_func6():
            broad: int @ block[1]
            narrow: int @ thread[1]
            broad = narrow

        func_type, ast_node, _ = registry.get("test_func6")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)

    def test_write_from_narrower_index(self):
        """block[4] = block[2] (should fail)"""
        @requires(grid[1])
        @prism("device")
        def test_func7():
            w: int @ block[4]
            z: int @ block[2]
            w = z

        func_type, ast_node, _ = registry.get("test_func7")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)

    def test_incomparable_perspectives(self):
        """block[3] = block[5] (incomparable, should fail)"""
        @requires(grid[1])
        @prism("device")
        def test_func8():
            m: float @ block[3]
            n: float @ block[5]
            m = n

        func_type, ast_node, _ = registry.get("test_func8")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)

    def test_thread_to_grid(self):
        """grid[1] = thread[1] (narrow to broad, should fail)"""
        @requires(grid[1])
        @prism("device")
        def test_func9():
            global_var: double @ grid[1]
            local_var: double @ thread[1]
            global_var = local_var

        func_type, ast_node, _ = registry.get("test_func9")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)

    def test_mixed_valid_and_invalid_rhs(self):
        """thread[2] = block[2] + thread[3] (one invalid RHS, should fail)"""
        @requires(grid[1])
        @prism("device")
        def test_func10():
            res: int @ thread[2]
            valid: int @ block[2]
            invalid: int @ thread[3]
            res = valid + invalid

        func_type, ast_node, _ = registry.get("test_func10")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)
