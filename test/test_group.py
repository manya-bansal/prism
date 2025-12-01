"""
Tests for the group construct.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    group,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import InvalidGroupError, PerspectiveBoundError


class TestGroupConstruct:
    """Test the group construct for narrowing perspective bounds."""

    def test_group_with_valid_narrower_perspective(self):
        """group(block[2]) inside @requires(grid[1], block[4]) should pass (block[4] is broader than block[2])."""
        @requires(grid[1], block[4])
        @prism("device")
        def valid_group():
            with group(block[2]):
                pass

        func_type, ast_node, _ = registry.get("valid_group")
        check_function(func_type, ast_node)  # Should not raise

    def test_group_with_invalid_perspective_fails(self):
        """group(block[4]) inside @requires(grid[1], block[2]) should fail (block[2] not broader than block[4])."""
        @requires(grid[1], block[2])
        @prism("device")
        def invalid_group():
            with group(block[4]):
                pass

        func_type, ast_node, _ = registry.get("invalid_group")
        with pytest.raises(InvalidGroupError):
            check_function(func_type, ast_node)

    def test_read_in_narrowed_context_passes(self):
        """Reading grid[2] var in if condition inside group(block[4]) should pass."""
        @requires(grid[2], block[4], thread[8])
        @prism("device")
        def read_broader_in_group(x: int @ grid[2]):
            with group(block[4]):
                if x > 0:  # x is grid[2], broader than block[4]
                    pass

        func_type, ast_node, _ = registry.get("read_broader_in_group")
        check_function(func_type, ast_node)  # Should not raise

    def test_read_in_narrowed_context_fails(self):
        """Reading thread[8] var in if condition inside group(block[4]) should fail."""
        @requires(grid[2], block[4], thread[8])
        @prism("device")
        def read_narrower_in_group(x: int @ thread[8]):
            with group(block[4]):
                if x > 0:  # x is thread[8], NOT broader than block[4]
                    pass

        func_type, ast_node, _ = registry.get("read_narrower_in_group")
        with pytest.raises(PerspectiveBoundError):
            check_function(func_type, ast_node)

    def test_nested_groups(self):
        """Nested groups should work: group(block[2]) then group(thread[4])."""
        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def nested_groups(x: int @ block[2], y: int @ thread[4]):
            with group(block[2]):
                if x > 0:  # x is block[2], equal to block[2] - should pass
                    pass
                with group(thread[4]):
                    if y > 0:  # y is thread[4], equal to thread[4] - should pass
                        pass
                # Back to block[2] context
                if x > 0:  # Still valid
                    pass

        func_type, ast_node, _ = registry.get("nested_groups")
        check_function(func_type, ast_node)  # Should not raise

    def test_group_same_perspective_repeatedly(self):
        """group(block[2]) inside group(block[2]) should pass (same perspective)."""
        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def repeated_same_group():
            with group(block[2]):
                with group(block[2]):
                    with group(block[2]):
                        pass

        func_type, ast_node, _ = registry.get("repeated_same_group")
        check_function(func_type, ast_node)  # Should not raise

    def test_group_broader_than_current_fails(self):
        """group(grid[1]) inside group(block[2]) should fail (can't widen back to grid)."""
        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def widen_group_fails():
            with group(block[2]):
                with group(grid[1]):  # Can't go back to grid from block
                    pass
            
            with group(block[2]):
                with group(grid[1]):  # Can't go back to grid from block
                    pass

        func_type, ast_node, _ = registry.get("widen_group_fails")
        with pytest.raises(InvalidGroupError):
            check_function(func_type, ast_node)
