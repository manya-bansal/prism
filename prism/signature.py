"""
Function signatures for prism functions.
"""

from dataclasses import dataclass
from prism.perspective import Perspective, PerspectiveBound
from prism.types import PrismType


@dataclass
class ParamSignature:
    """Signature for a function parameter."""
    name: str
    param_type: PrismType
    perspective: Perspective


@dataclass
class PerspectiveSignature:
    """Complete perspective signature for a prism function."""
    name: str
    bound: PerspectiveBound
    smem: int
    params: list[ParamSignature]
    return_type: PrismType | None
    return_perspective: Perspective | None
