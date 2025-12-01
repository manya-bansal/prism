"""
Tests for the split construct.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    split,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import InvalidSplitError, PerspectiveBoundError


class TestSplitConstruct:
    """Test the split construct for dividing perspective bounds."""

    def test_valid_split_multiple_cases(self):
        """Split with multiple valid cases should pass."""
        @requires(grid[1], block[8], thread[8])
        @prism("device")
        def valid_split_multi():
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

        func_type, ast_node, _ = registry.get("valid_split_multi")
        check_function(func_type, ast_node)  # Should not raise

    def test_non_divisible_split_allowed(self):
        """Non-divisible splits are valid (split uses subtraction, not division)."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def non_divisible_split():
            match split(block):
                case 3:
                    pass
                case 1:
                    pass

        func_type, ast_node, _ = registry.get("non_divisible_split")
        check_function(func_type, ast_node)  # Should not raise

    def test_invalid_split_case_exceeds_original_bound(self):
        """Split case larger than original bound should fail."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def invalid_split_original():
            match split(block):
                case 5:
                    pass

        func_type, ast_node, _ = registry.get("invalid_split_original")
        with pytest.raises(InvalidSplitError):
            check_function(func_type, ast_node)

    def test_invalid_split_case_exceeds_previous(self):
        """Split case larger than previous case should fail."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def invalid_split_previous():
            match split(block):
                case 2:
                    pass
                case 3:
                    pass

        func_type, ast_node, _ = registry.get("invalid_split_previous")
        with pytest.raises(InvalidSplitError):
            check_function(func_type, ast_node)

    def test_read_narrower_in_split_branch_fails(self):
        """Reading a narrower perspective var in split branch should fail."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def read_narrower_split(x: int @ thread[8]):
            match split(block):
                case 2:
                    if x > 0:  # should fail here
                        pass

        func_type, ast_node, _ = registry.get("read_narrower_split")
        with pytest.raises(PerspectiveBoundError):
            check_function(func_type, ast_node)

    def test_non_divisible_split_increasing_allowed(self):
        """Split with increasing case values is valid (split subtracts, doesn't require alignment)."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def non_div_split_incr():
            match split(block):
                case 1:
                    pass
                case 2:
                    pass

        func_type, ast_node, _ = registry.get("non_div_split_incr")
        check_function(func_type, ast_node)  # Should not raise

    def test_invalid_split_case_exceeds_remaining(self):
        """Split case exceeding remaining count should fail."""
        @requires(grid[1], block[4], thread[8])
        @prism("device")
        def invalid_split_remaining():
            match split(block):
                case 3:
                    pass
                case 2:
                    pass

        func_type, ast_node, _ = registry.get("invalid_split_remaining")
        with pytest.raises(InvalidSplitError):
            check_function(func_type, ast_node)
