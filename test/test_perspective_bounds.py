"""
Tests for perspective bounds and @requires decorator.
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
    PerspectiveBoundError,
    MissingRequiresError,
    InvalidRequiresError,
    InvalidAnnotationError,
)


class TestRequiresDecorator:
    """Test @requires decorator parsing."""

    def test_missing_requires_raises_error(self):
        """Function without @requires should raise MissingRequiresError."""
        @prism("device")
        def no_requires():
            x: int @ thread[1]

        func_type, ast_node, _ = registry.get("no_requires")
        with pytest.raises(MissingRequiresError):
            check_function(func_type, ast_node)

    def test_valid_requires_with_all_levels(self):
        """Function with @requires(grid[1], block[1], thread[1]) should pass."""
        @requires(grid[1], block[1], thread[1])
        @prism("device")
        def valid_requires():
            x: int @ thread[1]

        func_type, ast_node, _ = registry.get("valid_requires")
        check_function(func_type, ast_node)  # Should not raise

    def test_duplicate_level_raises_error(self):
        """@requires with duplicate levels should raise InvalidRequiresError."""
        @requires(grid[1], grid[2])
        @prism("device")
        def duplicate_grid():
            x: int @ thread[1]

        func_type, ast_node, _ = registry.get("duplicate_grid")
        with pytest.raises(InvalidRequiresError):
            check_function(func_type, ast_node)


class TestIfConditionPerspective:
    """Test perspective checking in if conditions."""

    def test_if_condition_broader_than_bound_passes(self):
        """If condition with perspective broader than bound should pass."""
        @requires(thread[1])
        @prism("device")
        def if_broader(x: int @ grid[1]):
            if x > 0:
                pass

        func_type, ast_node, _ = registry.get("if_broader")
        check_function(func_type, ast_node)  # Should not raise

    def test_if_condition_narrower_than_bound_fails(self):
        """If condition with perspective narrower than bound should fail."""
        @requires(grid[1])
        @prism("device")
        def if_narrower(x: int @ thread[1]):
            if x > 0:
                pass

        func_type, ast_node, _ = registry.get("if_narrower")
        with pytest.raises(PerspectiveBoundError):
            check_function(func_type, ast_node)


class TestForLoopPerspective:
    """Test perspective checking in for loop range arguments."""

    def test_for_range_with_broader_perspective_passes(self):
        """For loop with range args broader than bound should pass."""
        @requires(thread[1])
        @prism("device")
        def for_broader(n: int @ grid[1]):
            for i in range(n):
                pass

        func_type, ast_node, _ = registry.get("for_broader")
        check_function(func_type, ast_node)  # Should not raise

    def test_for_range_with_narrower_perspective_fails(self):
        """For loop with range args narrower than bound should fail."""
        @requires(grid[1])
        @prism("device")
        def for_narrower(n: int @ thread[1]):
            for i in range(n):
                pass

        func_type, ast_node, _ = registry.get("for_narrower")
        with pytest.raises(PerspectiveBoundError):
            check_function(func_type, ast_node)

    def test_for_not_range_raises_error(self):
        """For loop without range() should raise InvalidAnnotationError."""
        @requires(thread[1])
        @prism("device")
        def for_no_range(items: int @ grid[1]):
            for i in items:
                pass

        func_type, ast_node, _ = registry.get("for_no_range")
        with pytest.raises(InvalidAnnotationError):
            check_function(func_type, ast_node)


class TestWhileConditionPerspective:
    """Test perspective checking in while conditions."""

    def test_while_condition_broader_than_bound_passes(self):
        """While condition with perspective broader than bound should pass."""
        @requires(thread[1])
        @prism("device")
        def while_broader(x: int @ block[1]):
            while x > 0:
                pass

        func_type, ast_node, _ = registry.get("while_broader")
        check_function(func_type, ast_node)  # Should not raise

    def test_while_condition_narrower_than_bound_fails(self):
        """While condition with perspective narrower than bound should fail."""
        @requires(block[1])
        @prism("device")
        def while_narrower(x: int @ thread[1]):
            while x > 0:
                pass

        func_type, ast_node, _ = registry.get("while_narrower")
        with pytest.raises(PerspectiveBoundError):
            check_function(func_type, ast_node)
