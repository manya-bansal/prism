"""
The @prism decorator for marking device and global functions.
"""

import ast
import inspect
import textwrap
from functools import wraps
from prism.registry import registry


def prism(func_type: str):
    """
    Decorator to mark functions as prism device or global functions.

    Args:
        func_type: Either "device" or "global"

    Usage:
        @prism("device")
        def my_kernel(x):
            return x * 2
    """
    if func_type not in ["device", "global"]:
        raise ValueError(f"Invalid prism type: {func_type}. Must be 'device' or 'global'")

    def decorator(func):
        # Get the source code of the function
        source = inspect.getsource(func)

        # Remove any common leading whitespace (dedent)
        source = textwrap.dedent(source)

        # Parse the source to get the AST
        module_ast = ast.parse(source)

        # Extract the function definition node
        func_ast = module_ast.body[0]

        if not isinstance(func_ast, ast.FunctionDef):
            raise ValueError(f"Expected function definition, got {type(func_ast)}")

        # Register the function in the global registry
        registry.register(func.__name__, func_type, func_ast, source)

        # Return a wrapper that notes this is a prism function
        # For now, we'll still allow calling it normally for testing
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: In future, this might dispatch to CUDA execution
            # For now, just call the original function
            return func(*args, **kwargs)

        # Attach metadata to the wrapper
        wrapper._prism_type = func_type
        wrapper._prism_ast = func_ast
        wrapper._prism_source = source

        return wrapper

    return decorator
