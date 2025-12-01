"""
Tests for stack-allocated local arrays (type[constant] syntax).
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import PerspectiveError


class TestLocalArrays:
    """Test stack-allocated local arrays."""

    def test_local_array_write_at_block_perspective(self):
        """LOCAL arrays can be written at any perspective (not just thread[1])."""
        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def local_write_block():
            buffer: int[16] @ block[1]
            x: int @ block[1]
            buffer[0] = x 

        func_type, ast_node, _ = registry.get("local_write_block")
        check_function(func_type, ast_node)  # Should not raise

    def test_local_array_read_narrower_fails(self):
        """Reading from LOCAL array with narrower index perspective must respect rules."""
        @requires(grid[1], block[2], thread[4])
        @prism("device")
        def local_read_narrower(idx: int @ thread[4]):
            buffer: int[16] @ block[2]
            result: int @ block[1]
            result = buffer[idx]

        func_type, ast_node, _ = registry.get("local_read_narrower")
        with pytest.raises(PerspectiveError):
            check_function(func_type, ast_node)
