"""
Tests for shared memory allocation (shared(type[size]) syntax).
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    shared,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import SharedMemoryPerspectiveError, SharedMemoryExceededError


class TestSharedMemory:
    """Test shared memory allocation."""

    def test_valid_shared_allocation(self):
        """Valid shared memory allocation at block[1] with sufficient smem."""
        @requires(grid[1], block[1], thread[1], smem=1024)
        @prism("device")
        def valid_shared():
            buf: shared(int[64]) @ block[1]  # 64 * 4 = 256 bytes

        func_type, ast_node, _ = registry.get("valid_shared")
        check_function(func_type, ast_node)  # Should not raise

    def test_shared_exceeds_budget(self):
        """Shared memory allocation exceeding smem budget should fail."""
        @requires(grid[1], block[1], thread[1], smem=100)
        @prism("device")
        def exceed_budget():
            buf: shared(int[64]) @ block[1]  # 64 * 4 = 256 bytes > 100

        func_type, ast_node, _ = registry.get("exceed_budget")
        with pytest.raises(SharedMemoryExceededError):
            check_function(func_type, ast_node)

    def test_shared_wrong_perspective(self):
        """Shared memory at non-block[1] perspective should fail."""
        @requires(grid[1], block[2], thread[1], smem=1024)
        @prism("device")
        def wrong_persp():
            buf: shared(int[64]) @ block[2]

        func_type, ast_node, _ = registry.get("wrong_persp")
        with pytest.raises(SharedMemoryPerspectiveError):
            check_function(func_type, ast_node)

    def test_shared_at_thread_perspective_fails(self):
        """Shared memory at thread perspective should fail."""
        @requires(grid[1], block[1], thread[4], smem=1024)
        @prism("device")
        def shared_at_thread():
            buf: shared(int[64]) @ thread[4]

        func_type, ast_node, _ = registry.get("shared_at_thread")
        with pytest.raises(SharedMemoryPerspectiveError):
            check_function(func_type, ast_node)

    def test_multiple_shared_allocations(self):
        """Multiple shared allocations should subtract from available smem."""
        @requires(grid[1], block[1], thread[1], smem=512)
        @prism("device")
        def multi_shared():
            buf1: shared(int[64]) @ block[1]  # 256 bytes
            buf2: shared(int[64]) @ block[1]  # 256 bytes, total = 512

        func_type, ast_node, _ = registry.get("multi_shared")
        check_function(func_type, ast_node)  # Should not raise

    def test_multiple_shared_exceed_total(self):
        """Multiple shared allocations that together exceed budget should fail."""
        @requires(grid[1], block[1], thread[1], smem=400)
        @prism("device")
        def multi_exceed():
            buf1: shared(int[64]) @ block[1]  # 256 bytes
            buf2: shared(int[64]) @ block[1]  # 256 bytes, total = 512 > 400

        func_type, ast_node, _ = registry.get("multi_exceed")
        with pytest.raises(SharedMemoryExceededError):
            check_function(func_type, ast_node)

    def test_shared_different_types(self):
        """Shared memory with different types should use correct sizes."""
        @requires(grid[1], block[1], thread[1], smem=1024)
        @prism("device")
        def shared_types():
            buf_int: shared(int[64]) @ block[1]      # 64 * 4 = 256 bytes
            buf_float: shared(float[64]) @ block[1]  # 64 * 4 = 256 bytes
            buf_double: shared(double[32]) @ block[1]  # 32 * 8 = 256 bytes
            buf_bf16: shared(bf16[128]) @ block[1]   # 128 * 2 = 256 bytes
            # Total = 1024 bytes

        func_type, ast_node, _ = registry.get("shared_types")
        check_function(func_type, ast_node)  # Should not raise

    def test_shared_no_smem_budget(self):
        """Shared memory with no smem budget (default 0) should fail."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def no_budget():
            buf: shared(int[1]) @ block[1]  # 4 bytes > 0

        func_type, ast_node, _ = registry.get("no_budget")
        with pytest.raises(SharedMemoryExceededError):
            check_function(func_type, ast_node)
