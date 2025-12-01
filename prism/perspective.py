"""
Perspective system for tracking variable execution levels.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Level(Enum):
    """Execution levels in the CUDA hierarchy."""
    grid = "grid"
    block = "block"
    thread = "thread"

    def _order(self):
        """Return the hierarchical order: grid (0) is broader than block (1) is broader than thread (2)."""
        order_map = {Level.grid: 0, Level.block: 1, Level.thread: 2}
        return order_map[self]

    def __lt__(self, other):
        """Compare levels: grid < block < thread (grid is broader)."""
        return self._order() < other._order()

    def __le__(self, other):
        return self._order() <= other._order()

    def __gt__(self, other):
        return self._order() > other._order()

    def __ge__(self, other):
        return self._order() >= other._order()


@dataclass(frozen=True)
class Perspective:
    """
    Represents a perspective: a level plus a static integer index.

    Examples:
        grid[0], block[1], thread[2]
    """
    level: Level
    index: int

    def __str__(self):
        return f"{self.level.value}[{self.index}]"

    def __repr__(self):
        return f"Perspective({self.level.value}, {self.index})"

    def is_broader(self, other: 'Perspective') -> bool:
        """
        Check if this perspective is broader than or equal to another.

        A perspective is broader if:
        1. Its level is higher in the hierarchy (grid > block > thread), OR
        2. Same level and this index is divisible by the other index
        """
        if self.index == 0 or other.index == 0:
            raise ValueError("Perspective index cannot be 0")

        if self.level < other.level:
            return True
        if self.level == other.level:
            return self.index % other.index == 0
        return False

    def is_narrower(self, other: 'Perspective') -> bool:
        """
        Check if this perspective is narrower than another.

        A perspective is narrower if the other is broader.
        """
        return other.is_broader(self)

    def narrowest(self, other: 'Perspective') -> 'Perspective':
        """
        Return the narrower of two perspectives.

        Used for pointer dereference: p[expr] has the narrowest
        perspective of p and expr.
        """
        if self.is_broader(other):
            return other
        if other.is_broader(self):
            return self
        # If neither is broader, they're incomparable - return self as fallback
        # (type checker should catch invalid cases)
        return self


@dataclass
class PerspectiveBound:
    """
    A perspective bound is a collection of perspectives, at most one per level.

    Used with @requires decorator to specify function-level perspective constraints.

    Example:
        @requires(grid[2], block[1], thread[4])
        def kernel(...):
            ...
    """
    grid: Optional[Perspective] = None
    block: Optional[Perspective] = None
    thread: Optional[Perspective] = None

    def __post_init__(self):
        """Validate that perspectives match their expected levels and auto-fill narrower levels."""
        if self.grid is not None and self.grid.level != Level.grid:
            raise ValueError(f"Grid perspective must have grid level, got {self.grid.level}")
        if self.block is not None and self.block.level != Level.block:
            raise ValueError(f"Block perspective must have block level, got {self.block.level}")
        if self.thread is not None and self.thread.level != Level.thread:
            raise ValueError(f"Thread perspective must have thread level, got {self.thread.level}")

        # Auto-fill narrower levels: grid implies block[1] and thread[1], block implies thread[1]
        if self.grid is not None:
            if self.block is None:
                object.__setattr__(self, 'block', Perspective(Level.block, 1))
            if self.thread is None:
                object.__setattr__(self, 'thread', Perspective(Level.thread, 1))
        elif self.block is not None:
            if self.thread is None:
                object.__setattr__(self, 'thread', Perspective(Level.thread, 1))

    def broadest(self) -> Optional[Perspective]:
        """
        Return the broadest perspective in this bound.

        Order: grid > block > thread (grid is broadest).
        Returns None if the bound is empty.
        """
        if self.grid is not None:
            return self.grid
        if self.block is not None:
            return self.block
        if self.thread is not None:
            return self.thread
        return None

    def update(self, perspective: Perspective) -> 'PerspectiveBound':
        """
        Return a new PerspectiveBound with the given perspective as the broadest.

        Drops broader levels, updates the requested level, preserves narrower levels.

        Raises:
            InvalidGroupError: If current bound at that level is not broader than requested.
        """
        from prism.errors import InvalidGroupError

        level = perspective.level

        # Get current perspective at this level
        if level == Level.grid:
            current = self.grid
        elif level == Level.block:
            current = self.block
        else:
            current = self.thread

        # Check broader condition (if current is None, that level was dropped so can't group into it)
        if current is None or not current.is_broader(perspective):
            raise InvalidGroupError(perspective, current)

        # Build new bound: drop broader levels, update this level, preserve narrower
        if level == Level.grid:
            return PerspectiveBound(grid=perspective, block=self.block, thread=self.thread)
        elif level == Level.block:
            return PerspectiveBound(grid=None, block=perspective, thread=self.thread)
        else:  # thread
            return PerspectiveBound(grid=None, block=None, thread=perspective)

    def __str__(self):
        parts = []
        if self.grid is not None:
            parts.append(str(self.grid))
        if self.block is not None:
            parts.append(str(self.block))
        if self.thread is not None:
            parts.append(str(self.thread))
        return f"PerspectiveBound({', '.join(parts)})"
