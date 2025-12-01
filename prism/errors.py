"""
Error types for the prism type checker.
"""

from dataclasses import dataclass
from typing import Optional
from prism.perspective import Perspective, PerspectiveBound
from prism.types import PrismType


class PrismError(Exception):
    """Base class for all prism errors."""
    pass


class PrismTypeError(PrismError):
    """Base class for type checking errors."""
    pass


@dataclass
class PerspectiveError(PrismTypeError):
    """RHS perspective is not broader than or equal to LHS perspective."""
    lhs_name: str
    lhs_perspective: Perspective
    rhs_perspective: Perspective

    def __str__(self):
        return (
            f"Cannot assign to '{self.lhs_name}' at {self.lhs_perspective} "
            f"from RHS at {self.rhs_perspective}: "
            f"RHS perspective must be broader than or equal to LHS perspective"
        )


@dataclass
class UndeclaredVariableError(PrismTypeError):
    """Variable used but not declared."""
    var_name: str

    def __str__(self):
        return f"Variable '{self.var_name}' not declared"


@dataclass
class PointerDeclarationError(PrismTypeError):
    """Pointer declared in function body instead of as parameter."""
    var_name: str

    def __str__(self):
        return (
            f"Cannot declare pointer '{self.var_name}' in function body. "
            "Pointers must be passed as function parameters."
        )


@dataclass
class PointerAssignmentError(PrismTypeError):
    """Pointer used on LHS of assignment."""
    var_name: str

    def __str__(self):
        return f"Cannot assign to pointer '{self.var_name}'. Pointers are not assignable."


@dataclass
class InvalidDereferenceError(PrismTypeError):
    """Subscript operator used on non-pointer type."""
    var_name: str
    var_type: Optional[PrismType] = None

    def __str__(self):
        if self.var_type:
            return f"Cannot subscript non-pointer '{self.var_name}' of type {self.var_type}"
        return f"Cannot subscript non-pointer '{self.var_name}'"


@dataclass
class InvalidTypeError(PrismTypeError):
    """Invalid or unknown type."""
    type_name: str
    valid_types: list[str]

    def __str__(self):
        return f"Unknown type '{self.type_name}'. Must be one of: {self.valid_types}"


@dataclass
class InvalidMemorySpaceError(PrismTypeError):
    """Invalid memory space for pointer."""
    space_name: str
    valid_spaces: list[str]

    def __str__(self):
        return f"Unknown memory space '{self.space_name}'. Must be one of: {self.valid_spaces}"


@dataclass
class InvalidAnnotationError(PrismTypeError):
    """Invalid type annotation format."""
    message: str

    def __str__(self):
        return self.message


@dataclass
class InvalidWriteThroughError(PrismTypeError):
    """Write-through to pointer not at thread[1] perspective."""
    ptr_name: str
    ptr_perspective: Perspective

    def __str__(self):
        return (
            f"Cannot write through pointer '{self.ptr_name}' at {self.ptr_perspective}. "
            f"Write-through is only allowed for pointers at thread[1] perspective."
        )


@dataclass
class PerspectiveBoundError(PrismTypeError):
    """Expression perspective violates the function's perspective bound."""
    context: str  # e.g., "if condition", "while condition", "for iterator"
    expr_perspective: Perspective
    bound_perspective: Perspective

    def __str__(self):
        return (
            f"Perspective violation in {self.context}: "
            f"expression has perspective {self.expr_perspective}, "
            f"but must be broader than or equal to {self.bound_perspective}"
        )


@dataclass
class MissingRequiresError(PrismTypeError):
    """Function is missing @requires decorator."""
    func_name: str

    def __str__(self):
        return (
            f"Function '{self.func_name}' is missing @requires decorator. "
            "Example: @requires(grid[1], block[1], thread[1])"
        )


@dataclass
class InvalidRequiresError(PrismTypeError):
    """Invalid @requires decorator format."""
    message: str

    def __str__(self):
        return f"Invalid @requires decorator: {self.message}"


@dataclass
class InvalidGroupError(PrismTypeError):
    """Invalid group construct."""
    requested: Perspective
    current: Optional[Perspective]

    def __str__(self):
        if self.current is None:
            return (
                f"Cannot enter group({self.requested}): "
                f"{self.requested.level.value} level is not available in current bound"
            )
        return (
            f"Cannot enter group({self.requested}): "
            f"current bound at {self.requested.level.value} level is {self.current}, "
            f"which is not broader than {self.requested}"
        )


@dataclass
class InvalidSplitError(PrismTypeError):
    """Invalid split construct."""
    case_value: int
    level: str
    previous: Perspective

    def __str__(self):
        return (
            f"Cannot split case {self.case_value}: "
            f"{self.level}[{self.case_value}] is not broader than or equal to "
            f"previous case's perspective {self.previous}"
        )


@dataclass
class SharedMemoryPerspectiveError(PrismTypeError):
    """Shared memory declared at wrong perspective."""
    var_name: str
    perspective: Perspective

    def __str__(self):
        return (
            f"Cannot allocate shared memory '{self.var_name}' at {self.perspective}. "
            f"Shared memory must be declared at block[1] perspective."
        )


@dataclass
class SharedMemoryExceededError(PrismTypeError):
    """Shared memory allocation exceeds available budget."""
    var_name: str
    requested: int
    available: int

    def __str__(self):
        return (
            f"Cannot allocate {self.requested} bytes for shared memory '{self.var_name}'. "
            f"Only {self.available} bytes available (set via @requires smem parameter)."
        )


@dataclass
class CallSmemError(PrismTypeError):
    """Function call requires more smem than available."""
    func_name: str
    requested: int
    available: int

    def __str__(self):
        return (
            f"Cannot call '{self.func_name}': requires {self.requested} bytes smem, "
            f"but only {self.available} bytes available."
        )


@dataclass
class CallBoundError(PrismTypeError):
    """Function call perspective bound not satisfied."""
    func_name: str
    level: str
    caller_perspective: Optional[Perspective]
    callee_perspective: Perspective

    def __str__(self):
        if self.caller_perspective is None:
            return (
                f"Cannot call '{self.func_name}': requires {self.callee_perspective}, "
                f"but caller has no {self.level} level in current bound."
            )
        return (
            f"Cannot call '{self.func_name}': requires {self.callee_perspective}, "
            f"but caller has {self.caller_perspective} which is not broader."
        )


@dataclass
class CallArgPerspectiveError(PrismTypeError):
    """Argument perspective doesn't match parameter requirement."""
    func_name: str
    param_name: str
    arg_perspective: Perspective
    param_perspective: Perspective
    requires_exact: bool

    def __str__(self):
        if self.requires_exact:
            return (
                f"Cannot call '{self.func_name}': argument for non-const ptr param '{self.param_name}' "
                f"has perspective {self.arg_perspective}, but must exactly match {self.param_perspective}."
            )
        return (
            f"Cannot call '{self.func_name}': argument for param '{self.param_name}' "
            f"has perspective {self.arg_perspective}, "
            f"but must be broader than or equal to {self.param_perspective}."
        )


@dataclass
class InvalidPartitionError(PrismTypeError):
    """Invalid partition construct."""
    message: str

    def __str__(self):
        return f"Invalid partition: {self.message}"


@dataclass
class PartitionPerspectiveError(PrismTypeError):
    """Partition target perspective is not narrower than source."""
    ptr_name: str
    ptr_perspective: Perspective
    target_perspective: Perspective

    def __str__(self):
        return (
            f"Cannot partition '{self.ptr_name}' at {self.ptr_perspective} "
            f"to {self.target_perspective}: target must be strictly narrower than source."
        )


@dataclass
class PartitionLambdaVarError(PrismTypeError):
    """Variable in partition lambda has perspective not broader than target."""
    var_name: str
    var_perspective: Perspective
    target_perspective: Perspective

    def __str__(self):
        return (
            f"Variable '{self.var_name}' in partition lambda has perspective {self.var_perspective}, "
            f"which is not broader than or equal to target perspective {self.target_perspective}."
        )


@dataclass
class PartitionNonPointerError(PrismTypeError):
    """Partition applied to non-pointer type."""
    var_name: str
    var_type: PrismType

    def __str__(self):
        return f"Cannot partition non-pointer '{self.var_name}' of type {self.var_type}."


@dataclass
class PartitionLocalPointerError(PrismTypeError):
    """Partition applied to LOCAL pointer."""
    var_name: str

    def __str__(self):
        return f"Cannot partition LOCAL pointer '{self.var_name}'. Only GLOBAL and SHARED pointers can be partitioned."


@dataclass
class InvalidClaimError(PrismTypeError):
    """Invalid claim construct."""
    message: str

    def __str__(self):
        return f"Invalid claim: {self.message}"


@dataclass
class ClaimMultipleBranchError(PrismTypeError):
    """Claimed variable accessed in multiple split branches."""
    var_name: str

    def __str__(self):
        return f"Claimed variable '{self.var_name}' accessed in multiple split branches. Claimed pointers can only be used in one branch."
