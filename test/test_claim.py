"""
Tests for the claim construct.
"""

import pytest
from prism import (
    prism,
    grid, block, thread,
    requires,
    claim,
    ptr,
    split,
)
from prism.type_checker import check_function
from prism.registry import registry
from prism.errors import (
    InvalidClaimError,
    ClaimMultipleBranchError,
    UndeclaredVariableError,
    InvalidPartitionError,
)


class TestClaimConstruct:
    """Test the claim construct for exclusive pointer access in split branches."""

    def test_valid_claim_used_in_one_branch(self):
        """Claimed pointer used in only one split branch should pass."""
        @requires(block[4], thread[1])
        @prism("device")
        def valid_claim(data: ptr(int) @ block[4]):
            with claim(data, p=thread[1]) as my_data:
                match split(block):
                    case 2:
                        x: int @ thread[1] = my_data[0]  # only used here
                    case 2:
                        pass  # not used here

        func_type, ast_node, _ = registry.get("valid_claim")
        check_function(func_type, ast_node)  # Should not raise

    def test_claim_used_in_multiple_branches_fails(self):
        """Claimed pointer used in multiple split branches should fail."""
        @requires(block[4], thread[1])
        @prism("device")
        def claim_multi_branch(data: ptr(int) @ block[4]):
            with claim(data, p=thread[1]) as my_data:
                match split(block):
                    case 2:
                        x: int @ thread[1] = my_data[0]  # read here
                    case 2:
                        y: int @ thread[1] = my_data[0]  # ERROR: also read here

        func_type, ast_node, _ = registry.get("claim_multi_branch")
        with pytest.raises(ClaimMultipleBranchError):
            check_function(func_type, ast_node)

    def test_claim_accessed_outside_split_fails(self):
        """Claimed pointer accessed outside split branch should fail."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def claim_outside_split(data: ptr(int) @ grid[1]):
            with claim(data, p=thread[1]) as my_data:
                x: int @ thread[1] = my_data[0]  # ERROR: not inside a split branch

        func_type, ast_node, _ = registry.get("claim_outside_split")
        with pytest.raises(InvalidClaimError, match="only be accessed inside a split branch"):
            check_function(func_type, ast_node)

    def test_claim_with_offset(self):
        """Claim with offset expression should work."""
        @requires(block[4], thread[1])
        @prism("device")
        def claim_with_offset(data: ptr(int) @ block[4], n: int @ block[4]):
            with claim(data, p=thread[1], offset=id() * n) as my_data:
                match split(block):
                    case 2:
                        x: int @ thread[1] = my_data[0]
                    case 2:
                        pass

        func_type, ast_node, _ = registry.get("claim_with_offset")
        check_function(func_type, ast_node)  # Should not raise

    def test_claim_allows_reading_original_pointer(self):
        """Original pointer can still be read inside claim block."""
        @requires(block[4], thread[1])
        @prism("device")
        def use_original_after_claim(data: ptr(int) @ block[4]):
            with claim(data, p=thread[1]) as my_data:
                match split(block):
                    case 2:
                        my_data[0] = data[0]
                    case 2:
                        pass

        func_type, ast_node, _ = registry.get("use_original_after_claim")
        check_function(func_type, ast_node)  # Should not raise


    def test_claim_old_variable_unavailable(self):
        """Cannot use old variable name inside claim."""
        @requires(block[4], thread[1])
        @prism("device")
        def variable_unavailable(data: ptr(int) @ block[4]):
            with claim(data, p=thread[1]) as my_data:
                match split(block):
                    case 2:
                        x: int @ thread[1] = data[0] 
                    case 2:
                        pass 

        func_type, ast_node, _ = registry.get("variable_unavailable")
        with pytest.raises(UndeclaredVariableError):
            check_function(func_type, ast_node)