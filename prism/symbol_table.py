"""
Symbol table for tracking variables and their perspectives across scopes.
"""

from typing import Optional, Dict
from dataclasses import dataclass
from prism.perspective import Perspective
from prism.types import PrismType


@dataclass
class VariableDeclaration:
    """
    Represents a variable declaration with its type and perspective.

    Example: x: int @ grid[0]
    """
    name: str
    var_type: PrismType
    perspective: Perspective

    def __str__(self):
        return f"{self.name}: {self.var_type} @ {self.perspective}"


class Scope:
    """Represents a single scope in the program."""

    def __init__(self, parent: Optional['Scope'] = None):
        self.parent = parent
        self.symbols: Dict[str, VariableDeclaration] = {}

    def define(self, name: str, decl: VariableDeclaration):
        """Define a variable in this scope."""
        self.symbols[name] = decl

    def lookup_local(self, name: str) -> Optional[VariableDeclaration]:
        """Look up a variable in this scope only."""
        return self.symbols.get(name)

    def lookup(self, name: str) -> Optional[VariableDeclaration]:
        """Look up a variable in this scope and parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None


class SymbolTable:
    """Manages nested scopes and variable declarations."""

    def __init__(self):
        self.current_scope = Scope()
        self.global_scope = self.current_scope

    def enter_scope(self):
        """Enter a new nested scope."""
        self.current_scope = Scope(parent=self.current_scope)

    def exit_scope(self):
        """Exit the current scope and return to parent."""
        if self.current_scope.parent is None:
            raise RuntimeError("Cannot exit global scope")
        self.current_scope = self.current_scope.parent

    def define(self, name: str, var_type: PrismType, perspective: Perspective):
        """
        Define a variable in the current scope.

        All variables must have both a type and a perspective.
        Validation should happen in the type checker before calling this.
        """
        decl = VariableDeclaration(name, var_type, perspective)
        self.current_scope.define(name, decl)

    def lookup(self, name: str) -> Optional[VariableDeclaration]:
        """Look up a variable in current and parent scopes."""
        return self.current_scope.lookup(name)

    def lookup_local(self, name: str) -> Optional[VariableDeclaration]:
        """Look up a variable in current scope only."""
        return self.current_scope.lookup_local(name)

    def reset(self):
        """Reset to global scope."""
        self.current_scope = Scope()
        self.global_scope = self.current_scope
