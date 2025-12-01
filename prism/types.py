"""
Type system for prism variables.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Union


class MemorySpace(Enum):
    """Memory spaces for pointers in CUDA."""
    GLOBAL = "global"
    SHARED = "shared"
    LOCAL = "local"

    def __str__(self):
        return self.value


class Type(Enum):
    """Supported types in prism."""
    int = "int"
    float = "float"
    bf16 = "bf16"
    double = "double"
    bool = "bool"
    uint = "uint"
    uint32_t = "uint32_t"
    float4 = "float4"
    uint64_t = "uint64_t"
    void = "void"
    CUtensorMap = "CUtensorMap"

    def __str__(self):
        return self.value

    def get_bytes(self) -> int:
        """Return the size in bytes for this type."""
        sizes = {
            Type.int: 4,
            Type.float: 4,
            Type.bf16: 2,
            Type.double: 8,
            Type.bool: 1,
            Type.uint: 4,
            Type.uint32_t: 4,
            Type.float4: 16,
            Type.uint64_t: 8,
            Type.void: 0,
            Type.CUtensorMap: 0,
        }
        return sizes[self]

    def get_c_str(self) -> str:
        """Return the C/CUDA type string for this type."""
        c_types = {
            Type.int: "int",
            Type.float: "float",
            Type.bf16: "__nv_bfloat16",
            Type.double: "double",
            Type.bool: "bool",
            Type.uint: "unsigned int",
            Type.uint32_t: "uint32_t",
            Type.float4: "float4",
            Type.uint64_t: "uint64_t",
            Type.void: "void",
            Type.CUtensorMap: "__grid_constant__ CUtensorMap",
        }
        return c_types[self]


@dataclass
class PtrType:
    """Pointer type with memory space."""
    inner: PrismType
    memory_space: MemorySpace = MemorySpace.GLOBAL

    def __str__(self):
        return f"ptr({self.inner}, {self.memory_space})"


@dataclass
class ConstType:
    """Const-qualified type."""
    inner: PrismType

    def __str__(self):
        return f"const({self.inner})"


@dataclass
class RestrictType:
    """Restrict-qualified pointer type."""
    inner: PrismType

    def __str__(self):
        return f"restrict({self.inner})"


@dataclass
class ConstExprType:
    """Constexpr-qualified type (compile-time constant)."""
    inner: PrismType

    def __str__(self):
        return f"constexpr({self.inner})"


# Union of all prism types
PrismType = Union[Type, PtrType, ConstType, RestrictType, ConstExprType]
