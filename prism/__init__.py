"""
Prism: A simple embedded language for Python with CUDA code generation.
"""

from prism.decorator import prism
from prism.registry import PrismRegistry
from prism.codegen import generate


# Runtime placeholders for type annotations.
# These are only used so Python doesn't raise NameError when parsing annotations.
# The actual type parsing happens in the type checker via AST analysis.

class _Placeholder:
    """Placeholder for annotation symbols."""
    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def __matmul__(self, other):
        return self

    def __rmatmul__(self, other):
        return self


# Type modifiers
ptr = _Placeholder("ptr")
const = _Placeholder("const")
restrict = _Placeholder("restrict")
constexpr = _Placeholder("constexpr")

# Memory spaces
GLOBAL = _Placeholder("GLOBAL")
SHARED = _Placeholder("SHARED")
LOCAL = _Placeholder("LOCAL")

# Perspective levels
grid = _Placeholder("grid")
block = _Placeholder("block")
thread = _Placeholder("thread")

# Perspective bound decorator
requires = _Placeholder("requires")

# Group construct
group = _Placeholder("group")

# Split construct
split = _Placeholder("split")

# Partition construct
partition = _Placeholder("partition")

# Claim construct
claim = _Placeholder("claim")

# Shared memory allocation
shared = _Placeholder("shared")

# Base types (shadowing builtins for annotation purposes)
# Note: These shadow Python builtins within prism-decorated functions
bf16 = _Placeholder("bf16")
double = _Placeholder("double")
uint = _Placeholder("uint")
uint32_t = _Placeholder("uint32_t")
float4 = _Placeholder("float4")

# Inline raw CUDA
unsafe = _Placeholder("unsafe")

# Function attributes
attr = _Placeholder("attr")

# Vector load/store cast
float4_cast = _Placeholder("float4_cast")

# Inline PTX assembly
intrinsic = _Placeholder("intrinsic")

# Extra types and opaque types
uint64_t = _Placeholder("uint64_t")
void = _Placeholder("void")
CUtensorMap = _Placeholder("CUtensorMap")

# Context managers
wgmma_async = _Placeholder("wgmma_async")


__all__ = [
    'prism',
    'PrismRegistry',
    'generate',
    # Type modifiers
    'ptr', 'const', 'restrict', 'constexpr',
    # Memory spaces
    'GLOBAL', 'SHARED', 'LOCAL',
    # Perspective levels
    'grid', 'block', 'thread',
    # Perspective bound decorator
    'requires',
    # Group construct
    'group',
    # Split construct
    'split',
    # Partition construct
    'partition',
    # Claim construct
    'claim',
    # Shared memory allocation
    'shared',
    # Extra types
    'bf16', 'double', 'uint', 'uint32_t', 'uint64_t', 'float4',
    'void', 'CUtensorMap',
    # Inline raw CUDA
    'unsafe',
    # Function attributes
    'attr',
    # Vector cast
    'float4_cast',
    # Intrinsics
    'intrinsic',
    # Context managers
    'wgmma_async',
]
