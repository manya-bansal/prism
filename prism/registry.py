"""
Registry for storing prism functions and their ASTs.
"""

from __future__ import annotations
import ast
from typing import Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from prism.signature import PerspectiveSignature


class PrismRegistry:
    """Stores all prism-decorated functions with their ASTs and metadata."""

    def __init__(self):
        # Maps function name to (function_type, ast_node, source_code)
        self.functions: Dict[str, Tuple[str, ast.FunctionDef, str]] = {}
        # Cache for computed signatures
        self.signatures: Dict[str, PerspectiveSignature] = {}

    def register(self, name: str, func_type: str, ast_node: ast.FunctionDef, source: str):
        """Register a prism function with its AST."""
        self.functions[name] = (func_type, ast_node, source)

    def get(self, name: str) -> Tuple[str, ast.FunctionDef, str]:
        """Retrieve a registered function's metadata."""
        return self.functions.get(name)

    def get_all(self) -> Dict[str, Tuple[str, ast.FunctionDef, str]]:
        """Get all registered functions."""
        return self.functions

    def get_signature(self, name: str) -> PerspectiveSignature | None:
        """Get cached signature for a function."""
        return self.signatures.get(name)

    def set_signature(self, name: str, signature: PerspectiveSignature):
        """Cache a function's signature."""
        self.signatures[name] = signature


# Global registry instance
registry = PrismRegistry()
