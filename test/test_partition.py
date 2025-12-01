"""
Tests for the partition construct.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    partition,
    ptr,
    split,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import (
    InvalidPartitionError,
    PartitionPerspectiveError,
    PartitionNonPointerError,
    PartitionLocalPointerError,
)


class TestPartitionConstruct:
    """Test the partition construct for narrowing pointer perspectives."""

    def test_valid_partition_grid_to_thread(self):
        """Partition a grid[1] pointer to thread[1] with valid lambda."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def valid_partition(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], f=lambda i: i + id() * n) as my_data:
                my_data[0] = 42

        func_type, ast_node, _ = registry.get("valid_partition")
        check_function(func_type, ast_node)  # Should not raise

    def test_partition_fails_when_broadest_bound_mismatches(self):
        """Partition should fail if broadest bound doesn't equal pointer's perspective."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def partition_bound_mismatch(data: ptr(int) @ block[1]):
            # broadest bound is grid[1], but data is at block[1]
            with partition(data, p=thread[1], f=lambda i: i) as my_data:
                my_data[0] = 42

        func_type, ast_node, _ = registry.get("partition_bound_mismatch")
        with pytest.raises(InvalidPartitionError, match="broadest bound"):
            check_function(func_type, ast_node)

    def test_partition_allows_reading_original_pointer(self):
        """Original pointer can still be read inside partition block."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def use_original_after_partition(data: ptr(int) @ grid[1]):
            with partition(data, p=thread[1], f=lambda i: i) as my_data:
                my_data[0] = data[0]

        func_type, ast_node, _ = registry.get("use_original_after_partition")
        check_function(func_type, ast_node)  # Should not raise

    def test_partition_lambda_rejects_narrow_perspective_var(self):
        """Lambda cannot use variables narrower than target perspective."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def lambda_narrow_var(data: ptr(int) @ grid[1], offset: int @ thread[1]):
            # offset is at thread[1], same as target - but we're partitioning to block[1]
            # offset is NOT broader than block[1], so this should fail
            with partition(data, p=block[1], f=lambda i: i + offset) as my_data:
                pass

        func_type, ast_node, _ = registry.get("lambda_narrow_var")
        with pytest.raises(InvalidPartitionError):
            check_function(func_type, ast_node)

    def test_partition_same_var_in_split_siblings_allowed(self):
        """Same variable can be partitioned in multiple split branches."""
        @requires(block[4], thread[1])
        @prism("device")
        def partition_in_siblings(data: ptr(int) @ block[2]):
            match split(block):
                case 2:
                    with partition(data, p=thread[1], f=lambda i: i) as my_data:
                        my_data[0] = 1
                case 2:
                    with partition(data, p=thread[1], f=lambda i: i) as my_data:
                        my_data[0] = 2

        func_type, ast_node, _ = registry.get("partition_in_siblings")
        check_function(func_type, ast_node)  # Should not raise

    def test_valid_partition_with_offset(self):
        """Partition with offset expression instead of lambda."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def valid_partition_offset(data: ptr(int) @ grid[1], n: int @ grid[1]):
            with partition(data, p=thread[1], offset=id() * n) as my_data:
                my_data[0] = 42

        func_type, ast_node, _ = registry.get("valid_partition_offset")
        check_function(func_type, ast_node)  # Should not raise

    def test_partition_offset_rejects_narrow_perspective_var(self):
        """Offset expression cannot use variables narrower than target perspective."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def offset_narrow_var(data: ptr(int) @ grid[1], narrow: int @ thread[1]):
            with partition(data, p=block[1], offset=narrow) as my_data:
                pass

        func_type, ast_node, _ = registry.get("offset_narrow_var")
        with pytest.raises(InvalidPartitionError):
            check_function(func_type, ast_node)
