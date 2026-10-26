"""
Type checker for prism functions.
"""

import ast
from typing import Any, Optional
from prism.symbol_table import SymbolTable, VariableDeclaration
from prism.types import Type, PrismType, PtrType, ConstType, RestrictType, ConstExprType, MemorySpace
from prism.perspective import Perspective, PerspectiveBound, Level
from prism.errors import (
    PerspectiveError,
    UndeclaredVariableError,
    PointerDeclarationError,
    PointerAssignmentError,
    InvalidDereferenceError,
    InvalidTypeError,
    InvalidMemorySpaceError,
    InvalidAnnotationError,
    InvalidWriteThroughError,
    PerspectiveBoundError,
    MissingRequiresError,
    InvalidRequiresError,
    InvalidGroupError,
    InvalidSplitError,
    SharedMemoryPerspectiveError,
    SharedMemoryExceededError,
    CallSmemError,
    CallBoundError,
    CallArgPerspectiveError,
    InvalidPartitionError,
    PartitionPerspectiveError,
    PartitionLambdaVarError,
    PartitionNonPointerError,
    PartitionLocalPointerError,
    InvalidClaimError,
    ClaimMultipleBranchError,
)
from prism.signature import ParamSignature, PerspectiveSignature


class TypeChecker(ast.NodeVisitor):
    """
    AST visitor for type checking prism functions.

    This visitor walks through the AST of prism functions and validates
    type constraints and language rules.
    """

    def __init__(self, func_type: str):
        """
        Initialize the type checker.

        Args:
            func_type: Either "device" or "global"
        """
        self.func_type = func_type
        self.symbol_table = SymbolTable()
        self.perspective_bound: PerspectiveBound | None = None
        self.smem: int = 0
        self.smem_available: int = 0
        # Track partitioned variables at each scope level
        self.partitioned_stack: list[set[str]] = [set()]
        # Track claimed variables at each scope level (maps new_name -> original_name)
        self.claimed_stack: list[dict[str, str]] = [{}]
        # Track which claimed variables have been accessed in current split branch
        self.claimed_accessed_in_branch: set[str] = set()
        # Track whether we're inside a split branch (depth counter)
        self.split_branch_depth: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition node."""
        # Parse @requires decorator for perspective bound and smem
        self.perspective_bound, self.smem = self._parse_requires_decorator(node)
        self.smem_available = self.smem

        # Find template parameter if present (arg with default value that's a list)
        template_arg_index = self._find_template_arg_index(node.args)

        # Parse template parameters first (they're at grid[1])
        if template_arg_index is not None:
            self._parse_template_params(node.args, template_arg_index)

        # Parse function parameters and add them to the symbol table
        params: list[ParamSignature] = []
        for i, arg in enumerate(node.args.args):
            # Skip the template parameter
            if i == template_arg_index:
                continue

            if arg.annotation is None:
                raise InvalidAnnotationError(
                    f"Parameter '{arg.arg}' must have a type annotation with perspective. "
                    "Example: def kernel(x: int @ thread[1])"
                )

            param_name = arg.arg
            param_type, perspective = self._parse_annotation(arg.annotation)

            # Add parameter to symbol table (pointers are allowed as parameters)
            self.symbol_table.define(param_name, param_type, perspective)

            # Build param signature
            params.append(ParamSignature(param_name, param_type, perspective))

        # Parse return type if present
        return_type = None
        return_perspective = None
        if node.returns is not None:
            return_type, return_perspective = self._parse_annotation(node.returns)

        # Build the function signature
        self.signature = PerspectiveSignature(
            name=node.name,
            bound=self.perspective_bound,
            smem=self.smem,
            params=params,
            return_type=return_type,
            return_perspective=return_perspective,
        )

        self.generic_visit(node)

    def _find_template_arg_index(self, args: ast.arguments) -> int | None:
        """
        Find the index of the template parameter in the function arguments.

        The template parameter is identified by:
        - Name is 'template'
        - Has a default value that is a list

        Returns:
            Index of template arg, or None if not found.
        """
        # defaults align to the end of args
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_index = num_args - num_defaults

        for i, arg in enumerate(args.args):
            if arg.arg == "template":
                # Check if it has a default that's a list
                default_index = i - first_default_index
                if default_index >= 0 and default_index < len(args.defaults):
                    default = args.defaults[default_index]
                    if isinstance(default, ast.List):
                        return i
        return None

    def _parse_template_params(self, args: ast.arguments, template_index: int) -> None:
        """
        Parse template parameters and add them to the symbol table at grid[1].

        Template syntax: template=[(N, int), (M, constexpr(int))]
        """
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_index = num_args - num_defaults
        default_index = template_index - first_default_index

        template_list = args.defaults[default_index]
        if not isinstance(template_list, ast.List):
            raise InvalidAnnotationError("template must be a list of (name, type) tuples")

        grid_1 = Perspective(Level.grid, 1)

        for item in template_list.elts:
            if not isinstance(item, ast.Tuple) or len(item.elts) != 2:
                raise InvalidAnnotationError(
                    "Each template parameter must be a tuple of (name, type). "
                    "Example: template=[(N, int), (M, constexpr(int))]"
                )

            name_node, type_node = item.elts

            if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
                raise InvalidAnnotationError(
                    "Template parameter name must be a string. "
                    "Example: template=[('N', int)]"
                )

            param_name = name_node.value
            param_type = self._parse_type(type_node)

            # Add to symbol table at grid[1] perspective
            self.symbol_table.define(param_name, param_type, grid_1)

    def _check_template_args(self, node: ast.Call, func_name: str) -> None:
        """
        Check that all template arguments in a function call are at grid[1].

        Template args are passed as: func(x, y, template=[a, b, c])
        """
        grid_1 = Perspective(Level.grid, 1)

        for kw in node.keywords:
            if kw.arg == "template":
                if not isinstance(kw.value, ast.List):
                    raise InvalidAnnotationError(
                        f"template argument in call to '{func_name}' must be a list"
                    )

                for i, elem in enumerate(kw.value.elts):
                    elem_perspective = self._get_expr_perspective(elem)
                    if elem_perspective != grid_1:
                        raise CallArgPerspectiveError(
                            func_name,
                            f"template[{i}]",
                            elem_perspective,
                            grid_1,
                            requires_exact=True
                        )
                break

    def _parse_requires_decorator(self, node: ast.FunctionDef) -> tuple[PerspectiveBound, int]:
        """
        Parse the @requires decorator from a function definition.

        Returns:
            Tuple of (PerspectiveBound, smem).

        Raises:
            MissingRequiresError: If @requires decorator is not found.
        """
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "requires":
                    return self._parse_requires_args(decorator.args, decorator.keywords)
        raise MissingRequiresError(node.name)

    def _parse_requires_args(self, args: list[ast.AST], keywords: list[ast.keyword]) -> tuple[PerspectiveBound, int]:
        """
        Parse the arguments of @requires(grid[n], block[m], thread[k], smem=1024).

        Returns:
            Tuple of (PerspectiveBound, smem). smem defaults to 0 if not provided.
        """
        grid_persp = None
        block_persp = None
        thread_persp = None
        smem = 0

        # Parse positional args (perspectives)
        for arg in args:
            perspective = self._parse_perspective(arg)

            if perspective.level == Level.grid:
                if grid_persp is not None:
                    raise InvalidRequiresError("Duplicate grid perspective")
                grid_persp = perspective
            elif perspective.level == Level.block:
                if block_persp is not None:
                    raise InvalidRequiresError("Duplicate block perspective")
                block_persp = perspective
            elif perspective.level == Level.thread:
                if thread_persp is not None:
                    raise InvalidRequiresError("Duplicate thread perspective")
                thread_persp = perspective

        # Parse keyword args (smem)
        for kw in keywords:
            if kw.arg == "smem":
                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, int):
                    raise InvalidRequiresError("smem must be a constant integer")
                smem = kw.value.value
            else:
                raise InvalidRequiresError(f"Unknown keyword argument '{kw.arg}'. Only 'smem' is supported.")

        return PerspectiveBound(grid=grid_persp, block=block_persp, thread=thread_persp), smem

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """
        Visit an annotated assignment (variable declaration).

        Parses: x: int @ grid[0]
        """
        if not isinstance(node.target, ast.Name):
            raise InvalidAnnotationError("Annotated assignment target must be a simple variable")

        var_name = node.target.id
        var_type, perspective = self._parse_annotation(node.annotation)

        # Check for shared memory allocation
        if self._is_shared_allocation(node.annotation):
            # Shared memory must be at block[1] perspective
            required_perspective = Perspective(Level.block, 1)
            if perspective != required_perspective:
                raise SharedMemoryPerspectiveError(var_name, perspective)

            # Get allocation size and check against budget
            alloc_size = self._get_shared_alloc_size(node.annotation)
            if alloc_size > self.smem_available:
                raise SharedMemoryExceededError(var_name, alloc_size, self.smem_available)
            self.smem_available -= alloc_size

        # Non-LOCAL pointers: allow declarations (initialized or not).
        # LOCAL pointers (stack-allocated arrays) are allowed.
        # SHARED pointers via shared() are also allowed.

        # Add to symbol table
        self.symbol_table.define(var_name, var_type, perspective)

        if node.value:
            # Skip type-check for array zero-init: {0} parsed as ast.Set
            if isinstance(node.value, ast.Set):
                pass
            elif self._contains_non_local_ptr(var_type):
                # Pointer initialization: skip synthetic visit_Assign (which forbids ptr LHS).
                # Just validate RHS perspective directly.
                rhs_perspective = self._get_expr_perspective(node.value, perspective)
                if not rhs_perspective.is_broader(perspective):
                    raise PerspectiveError(var_name, perspective, rhs_perspective)
            else:
                # Create a temporary Assign node to reuse the assignment checking logic
                assign = ast.Assign(targets=[node.target], value=node.value)
                self.visit_Assign(assign)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        """Visit a return statement node."""
        if node.value is not None and self.signature.return_perspective is not None:
            # Same rule as assignment: return expr perspective must be broader than or equal to return perspective
            return_perspective = self.signature.return_perspective
            expr_perspective = self._get_expr_perspective(node.value, return_perspective)

            if not expr_perspective.is_broader(return_perspective):
                raise PerspectiveError("return", return_perspective, expr_perspective)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        """
        Visit an assignment node.

        Two cases:
        1. x = expr: RHS perspective must be broader than or equal to LHS perspective
        2. p[expr] = value: Write-through, pointer must be at thread[1] perspective
        """
        if len(node.targets) != 1:
            raise InvalidAnnotationError("Multiple assignment targets not yet supported")

        target = node.targets[0]

        # Case 1: Write-through p[expr] = value
        if isinstance(target, ast.Subscript):
            # float4_cast[expr] = value — raw cast store, skip type-check
            if isinstance(target.value, ast.Name) and target.value.id == "float4_cast":
                self.generic_visit(node)
                return
            self._check_write_through(target, node.value)
            self.generic_visit(node)
            return

        # Case 1b: Attribute target (e.g. tmp.x = expr)
        if isinstance(target, ast.Attribute):
            self.generic_visit(node)
            return

        # Case 2: Simple assignment x = expr
        if not isinstance(target, ast.Name):
            raise InvalidAnnotationError("Assignment target must be a simple variable or pointer dereference")

        lhs_name = target.id
        lhs_decl = self.lookup_with_partition_guard(lhs_name)

        # Pointers cannot be on LHS of assignment
        if self._contains_ptr(lhs_decl.var_type):
            raise PointerAssignmentError(lhs_name)

        # Get the perspective of the RHS expression (pass LHS perspective for id() resolution)
        rhs_perspective = self._get_expr_perspective(node.value, lhs_decl.perspective)

        # Check that RHS perspective is broader than or equal to LHS
        if not rhs_perspective.is_broader(lhs_decl.perspective):
            raise PerspectiveError(lhs_name, lhs_decl.perspective, rhs_perspective)

        self.generic_visit(node)

    def _check_write_through(self, target: ast.Subscript, value: ast.AST) -> None:
        """
        Check write-through assignment: p[expr] = value

        Rules:
        1. The base must be a pointer
        2. For GLOBAL/SHARED pointers: must be at thread[1] perspective
           For LOCAL pointers: any perspective is allowed
        3. RHS perspective must be broader than or equal to the write target perspective
        """
        # Get the pointer variable
        if not isinstance(target.value, ast.Name):
            raise InvalidAnnotationError("Write-through target must be a simple pointer dereference")

        ptr_name = target.value.id
        ptr_decl = self.lookup_with_partition_guard(ptr_name)

        # Check that it's a pointer
        if not self._contains_ptr(ptr_decl.var_type):
            raise InvalidDereferenceError(ptr_name, ptr_decl.var_type)

        # Check perspective requirement (only for non-LOCAL pointers)
        if not self._is_local_ptr(ptr_decl.var_type):
            required_perspective = Perspective(Level.thread, 1)
            if ptr_decl.perspective != required_perspective:
                raise InvalidWriteThroughError(ptr_name, ptr_decl.perspective)

        # The write target perspective is narrowest(ptr_perspective, index_perspective)
        idx_perspective = self._get_expr_perspective(target.slice)
        target_perspective = ptr_decl.perspective.narrowest(idx_perspective)

        # Check RHS perspective
        rhs_perspective = self._get_expr_perspective(value)
        if not rhs_perspective.is_broader(target_perspective):
            raise PerspectiveError(f"{ptr_name}[...]", target_perspective, rhs_perspective)

    def _contains_ptr(self, t: PrismType) -> bool:
        """Check if a type is or contains a pointer type."""
        if isinstance(t, PtrType):
            return True
        if isinstance(t, (ConstType, RestrictType)):
            return self._contains_ptr(t.inner)
        return False

    def _contains_non_local_ptr(self, t: PrismType) -> bool:
        """Check if a type is or contains a non-LOCAL pointer type."""
        if isinstance(t, PtrType):
            return t.memory_space != MemorySpace.LOCAL
        if isinstance(t, (ConstType, RestrictType)):
            return self._contains_non_local_ptr(t.inner)
        return False

    def _is_local_ptr(self, t: PrismType) -> bool:
        """Check if a type is a LOCAL pointer."""
        if isinstance(t, PtrType):
            return t.memory_space == MemorySpace.LOCAL
        if isinstance(t, (ConstType, RestrictType)):
            return self._is_local_ptr(t.inner)
        return False

    def _is_non_const_ptr(self, t: PrismType) -> bool:
        """Check if a type is a pointer that allows writing (inner is not const)."""
        if isinstance(t, ConstType):
            return False
        if isinstance(t, PtrType):
            # ptr(const(...)) = read-only pointer, doesn't require exact match
            if isinstance(t.inner, ConstType):
                return False
            return True
        if isinstance(t, RestrictType):
            return self._is_non_const_ptr(t.inner)
        return False

    def _is_shared_allocation(self, annotation: ast.AST) -> bool:
        """Check if an annotation is a shared() allocation."""
        if not isinstance(annotation, ast.BinOp) or not isinstance(annotation.op, ast.MatMult):
            return False
        type_node = annotation.left
        if not isinstance(type_node, ast.Call):
            return False
        if not isinstance(type_node.func, ast.Name):
            return False
        return type_node.func.id == "shared"

    def _get_shared_alloc_size(self, annotation: ast.AST) -> int:
        """Get the allocation size in bytes from a shared() annotation."""
        type_node = annotation.left  # shared(type[size], align=N, static=True)
        # Static shared memory doesn't count against dynamic smem budget
        for kw in type_node.keywords:
            if kw.arg == "static" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return 0
        arg = type_node.args[0]  # type[size]
        size = self._eval_const_expr(arg.slice)
        type_name = arg.value.id
        return size * Type[type_name].get_bytes()

    def _eval_const_expr(self, node: ast.AST) -> int:
        """Evaluate a constant expression at type-check time."""
        if isinstance(node, ast.Constant):
            return int(node.value)
        if isinstance(node, ast.BinOp):
            left = self._eval_const_expr(node.left)
            right = self._eval_const_expr(node.right)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Div: lambda a, b: a // b,
            }
            op_fn = ops.get(type(node.op))
            if op_fn:
                return op_fn(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_const_expr(node.operand)
        # Fallback for names (constexpr vars) - return 0 to avoid crashing
        if isinstance(node, ast.Name):
            return 0
        raise ValueError(f"Cannot evaluate constant expression: {ast.dump(node)}")

    def _unwrap_ptr(self, t: PrismType) -> PrismType:
        """Unwrap a pointer type to get the inner type."""
        if isinstance(t, PtrType):
            return t.inner
        if isinstance(t, (ConstType, RestrictType)):
            inner = self._unwrap_ptr(t.inner)
            if isinstance(t, ConstType):
                return ConstType(inner)
            else:
                return RestrictType(inner)
        raise TypeError("Cannot dereference non-pointer type")

    def lookup_with_partition_guard(self, ptr_name) -> Optional[VariableDeclaration]:
        # Look up the pointer in symbol table
        ptr_decl = self.symbol_table.lookup(ptr_name)
        if not ptr_decl:
            raise UndeclaredVariableError(ptr_name)

        # Check that this variable has not been partitioned 
        self._check_not_partitioned(ptr_name)

        return ptr_decl


    def _get_expr_perspective(self, node: ast.AST, target_perspective: Perspective | None = None) -> Perspective:
        """
        Compute the perspective of an expression.

        - Variable: returns its declared perspective
        - Constant: returns grid[1] (broadest)
        - BinOp: returns narrowest of left and right
        - Subscript (dereference): returns narrowest of pointer and index perspectives
        - id(): returns target_perspective (from LHS) or broadest bound (vacuum)

        Args:
            node: The AST node to compute perspective for
            target_perspective: Optional perspective from LHS of assignment (for id() resolution)
        """
        if isinstance(node, ast.Name):
            decl = self.lookup_with_partition_guard(node.id)
            if not decl:
                raise UndeclaredVariableError(node.id)
            # Track access to claimed variables
            self._check_claimed_access(node.id)
            return decl.perspective

        elif isinstance(node, ast.Constant):
            # Constants are at grid[1] - broadest perspective
            return Perspective(Level.grid, 1)

        elif isinstance(node, ast.Call):
            # Check for id() call
            if isinstance(node.func, ast.Name) and node.func.id == "id" and len(node.args) == 0:
                # Determine perspective: use target_perspective or broadest bound
                if target_perspective is not None:
                    persp = target_perspective
                else:
                    persp = self.perspective_bound.broadest()

                # Update AST: add perspective as argument
                # Transform id() -> id(level[index])
                level_node = ast.Name(id=persp.level.name, ctx=ast.Load())
                subscript_node = ast.Subscript(
                    value=level_node,
                    slice=ast.Constant(value=persp.index),
                    ctx=ast.Load()
                )
                node.args.append(subscript_node)

                return persp

            # Check if this is a prism function call - use its return perspective
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                from prism.registry import registry
                result = registry.get(func_name)
                if result is not None:
                    callee_func_type, callee_ast, _ = result
                    callee_sig = check_function(callee_func_type, callee_ast)
                    if callee_sig.return_perspective is not None:
                        return callee_sig.return_perspective

            # Other function calls: compute from arguments
            perspectives = [self._get_expr_perspective(arg, target_perspective) for arg in node.args]
            if not perspectives:
                return Perspective(Level.grid, 1)
            result = perspectives[0]
            for p in perspectives[1:]:
                result = result.narrowest(p)
            return result

        elif isinstance(node, ast.BinOp):
            left_persp = self._get_expr_perspective(node.left, target_perspective)
            right_persp = self._get_expr_perspective(node.right, target_perspective)
            return left_persp.narrowest(right_persp)

        elif isinstance(node, ast.UnaryOp):
            return self._get_expr_perspective(node.operand, target_perspective)

        elif isinstance(node, ast.Subscript):
            # Pointer dereference: p[expr]
            # Result perspective = narrowest(pointer perspective, index perspective)

            # Verify that the base is actually a pointer
            if isinstance(node.value, ast.Name):
                decl = self.lookup_with_partition_guard(node.value.id)
                if not self._contains_ptr(decl.var_type):
                    raise InvalidDereferenceError(node.value.id, decl.var_type)

            ptr_persp = self._get_expr_perspective(node.value, target_perspective)
            idx_persp = self._get_expr_perspective(node.slice, target_perspective)
            return ptr_persp.narrowest(idx_persp)

        else:
            # For other node types, find narrowest among children
            perspectives = []
            for child in ast.iter_child_nodes(node):
                perspectives.append(self._get_expr_perspective(child, target_perspective))

            if not perspectives:
                # No children - treat as constant
                return Perspective(Level.grid, 1)

            result = perspectives[0]
            for p in perspectives[1:]:
                result = result.narrowest(p)
            return result

    def _parse_type(self, node: ast.AST) -> PrismType:
        """Parse a type node and return a PrismType."""
        # Base type: int, float, bf16, double, bool
        if isinstance(node, ast.Name):
            type_name = node.id
            try:
                return Type[type_name]
            except KeyError:
                raise InvalidTypeError(type_name, [t.name for t in Type])

        # Stack-allocated array: type[constant] or type[s1][s2][s3] -> ptr(type, LOCAL)
        if isinstance(node, ast.Subscript):
            # Unwrap nested subscripts to find the base type name
            current = node
            while isinstance(current, ast.Subscript):
                current = current.value
            if not isinstance(current, ast.Name):
                raise InvalidAnnotationError("Stack array type must be a simple name (int, float, etc.)")

            type_name = current.id
            try:
                inner_type = Type[type_name]
            except KeyError:
                raise InvalidTypeError(type_name, [t.name for t in Type])

            # Stack-allocated arrays become LOCAL pointers
            return PtrType(inner_type, MemorySpace.LOCAL)

        # Wrapper types: ptr(...), const(...), restrict(...)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise InvalidAnnotationError("Type modifier must be a simple name (ptr, const, restrict)")

            modifier = node.func.id

            if modifier == "ptr":
                # ptr(inner_type) or ptr(inner_type, MEMORY_SPACE)
                if len(node.args) < 1 or len(node.args) > 2:
                    raise InvalidAnnotationError("ptr() takes 1 or 2 arguments: ptr(type) or ptr(type, MEMORY_SPACE)")

                inner = self._parse_type(node.args[0])
                memory_space = MemorySpace.GLOBAL  # default

                if len(node.args) == 2:
                    memory_space = self._parse_memory_space(node.args[1])

                return PtrType(inner, memory_space)

            elif modifier == "const":
                # const(inner_type)
                if len(node.args) != 1:
                    raise InvalidAnnotationError("const() takes exactly 1 argument")

                inner = self._parse_type(node.args[0])
                return ConstType(inner)

            elif modifier == "restrict":
                # restrict(inner_type)
                if len(node.args) != 1:
                    raise InvalidAnnotationError("restrict() takes exactly 1 argument")

                inner = self._parse_type(node.args[0])
                return RestrictType(inner)

            elif modifier == "constexpr":
                # constexpr(inner_type)
                if len(node.args) != 1:
                    raise InvalidAnnotationError("constexpr() takes exactly 1 argument")

                inner = self._parse_type(node.args[0])
                return ConstExprType(inner)

            elif modifier == "shared":
                # shared(type[size], align=N, static=True) -> ptr(type, SHARED)
                # shared(type) without size -> ptr(type, SHARED)  (for parameter annotations)
                if len(node.args) != 1:
                    raise InvalidAnnotationError("shared() takes exactly 1 argument: shared(type[size]) or shared(type)")

                arg = node.args[0]
                if isinstance(arg, ast.Name):
                    # shared(type) — no size, used as parameter annotation
                    type_name = arg.id
                    try:
                        inner_type = Type[type_name]
                    except KeyError:
                        raise InvalidTypeError(type_name, [t.name for t in Type])
                    return PtrType(inner_type, MemorySpace.SHARED)
                if not isinstance(arg, ast.Subscript):
                    raise InvalidAnnotationError("shared() argument must be type[size] or type. Example: shared(int[256])")
                if not isinstance(arg.value, ast.Name):
                    raise InvalidAnnotationError("shared() type must be a simple name (int, float, etc.)")
                # Allow expression sizes (e.g. 64*128), not just constants

                type_name = arg.value.id
                try:
                    inner_type = Type[type_name]
                except KeyError:
                    raise InvalidTypeError(type_name, [t.name for t in Type])

                # Return SHARED pointer (smem tracking handled in visit_AnnAssign)
                return PtrType(inner_type, MemorySpace.SHARED)

            else:
                raise InvalidTypeError(modifier, ["ptr", "const", "restrict", "shared"])

        raise InvalidAnnotationError("Type must be a simple name (int, float, etc.) or modifier (ptr, const, restrict)")

    def _parse_memory_space(self, node: ast.AST) -> MemorySpace:
        """Parse a memory space node (GLOBAL, SHARED, LOCAL)."""
        if not isinstance(node, ast.Name):
            raise InvalidAnnotationError("Memory space must be GLOBAL, SHARED, or LOCAL")

        space_name = node.id
        try:
            return MemorySpace[space_name]
        except KeyError:
            raise InvalidMemorySpaceError(space_name, [m.name for m in MemorySpace])

    def _parse_perspective(self, node: ast.AST) -> Perspective:
        """Parse a perspective node and return a Perspective object."""
        if not isinstance(node, ast.Subscript):
            raise InvalidAnnotationError("Perspective must be in format 'level[index]'. Example: grid[1]")

        if not isinstance(node.value, ast.Name):
            raise InvalidAnnotationError("Level must be a simple name (grid, block, thread)")

        level_name = node.value.id
        try:
            level = Level[level_name]
        except KeyError:
            raise InvalidAnnotationError(f"Unknown level '{level_name}'. Must be one of: {[l.name for l in Level]}")

        if not isinstance(node.slice, ast.Constant):
            raise InvalidAnnotationError("Perspective index must be a constant integer")

        index = node.slice.value
        if not isinstance(index, int):
            raise InvalidAnnotationError("Perspective index must be an integer")

        return Perspective(level, index)

    def _parse_annotation(self, annotation: ast.AST) -> tuple[PrismType, Perspective]:
        """
        Parse a type annotation with perspective.

        Expected format: type @ level[index]
        Example: int @ grid[1]
        """
        if not isinstance(annotation, ast.BinOp) or not isinstance(annotation.op, ast.MatMult):
            raise InvalidAnnotationError(
                "Variable annotation must be in format 'type @ level[index]'. "
                "Example: x: int @ grid[1]"
            )

        var_type = self._parse_type(annotation.left)
        perspective = self._parse_perspective(annotation.right)

        return var_type, perspective

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Visit a binary operation node."""
        # TODO: Add type checking rules
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Visit a function call node."""
        # Skip attribute-based calls like intrinsic.mma(...)
        if isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        # Skip built-in functions, decorators, and type constructors
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ("id", "range", "group", "split", "requires", "prism", "shared", "ptr", "const", "restrict", "unsafe", "float4_cast", "wgmma_async"):
                self.generic_visit(node)
                return

            # Look up callee's signature from registry
            from prism.registry import registry
            import warnings
            result = registry.get(func_name)
            if result is None:
                warnings.warn(f"Function '{func_name}' not found in registry, skipping call check")
                self.generic_visit(node)
                return
            callee_func_type, callee_ast, _ = result

            # Get callee's signature by type checking it
            callee_sig = check_function(callee_func_type, callee_ast)

            # Check smem: caller must have enough available
            if callee_sig.smem > self.smem_available:
                raise CallSmemError(func_name, callee_sig.smem, self.smem_available)

            # Check perspective bound: for each level callee requires,
            # caller must have a broader perspective at that level
            callee_bound = callee_sig.bound

            if callee_bound.grid is not None:
                caller_grid = self.perspective_bound.grid
                if caller_grid is None or not caller_grid.is_broader(callee_bound.grid):
                    raise CallBoundError(func_name, "grid", caller_grid, callee_bound.grid)

            if callee_bound.block is not None:
                caller_block = self.perspective_bound.block
                if caller_block is None or not caller_block.is_broader(callee_bound.block):
                    raise CallBoundError(func_name, "block", caller_block, callee_bound.block)

            if callee_bound.thread is not None:
                caller_thread = self.perspective_bound.thread
                if caller_thread is None or not caller_thread.is_broader(callee_bound.thread):
                    raise CallBoundError(func_name, "thread", caller_thread, callee_bound.thread)

            # Check argument perspectives against param signatures
            for i, (arg, param) in enumerate(zip(node.args, callee_sig.params)):
                arg_perspective = self._get_expr_perspective(arg)

                # Check if param is a non-const ptr (requires exact match)
                if self._is_non_const_ptr(param.param_type):
                    # Non-const ptr: perspective must match exactly
                    if arg_perspective != param.perspective:
                        raise CallArgPerspectiveError(
                            func_name, param.name, arg_perspective, param.perspective, requires_exact=True
                        )
                else:
                    # Non-ptr or const ptr: argument must be broader than or equal to param
                    if not arg_perspective.is_broader(param.perspective):
                        raise CallArgPerspectiveError(
                            func_name, param.name, arg_perspective, param.perspective, requires_exact=False
                        )

            # Check template arguments: all must be at grid[1]
            self._check_template_args(node, func_name)

    def visit_If(self, node: ast.If) -> Any:
        """
        Visit an if statement node.

        The condition is read in a "vacuum" - we check its perspective
        against the broadest perspective in the function's perspective bound.
        """
        self._check_vacuum_expression(node.test, "if condition")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        """
        Visit a for loop node.

        For `for i in range(start, stop, step):`, we check each argument
        to range() against the perspective bound.
        """
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            raise InvalidAnnotationError("For loops must use range()")

        for arg in node.iter.args:
            self._check_vacuum_expression(arg, "for range argument")

        # Register loop variable at the broadest perspective bound
        loop_var = node.target.id
        broadest = self.perspective_bound.broadest()
        self.symbol_table.enter_scope()
        self.symbol_table.define(loop_var, Type.int, broadest)

        for stmt in node.body:
            self.visit(stmt)

        self.symbol_table.exit_scope()
        return  # skip generic_visit, we already visited the body

    def visit_While(self, node: ast.While) -> Any:
        """
        Visit a while loop node.

        The condition is read in a "vacuum" - we check its perspective
        against the broadest perspective in the function's perspective bound.
        """
        self._check_vacuum_expression(node.test, "while condition")
        self.generic_visit(node)

    def _check_vacuum_expression(self, node: ast.AST, context: str) -> None:
        """
        Check an expression read in a "vacuum" (not as part of an assignment RHS).

        Uses the broadest perspective from the function's perspective bound
        as the "logical LHS" and checks that the expression's perspective
        is broader than or equal to it.
        """
        if self.perspective_bound is None:
            raise InvalidAnnotationError("Cannot check vacuum expression without a perspective bound")

        bound_perspective = self.perspective_bound.broadest()
        if bound_perspective is None:
            raise InvalidRequiresError("Perspective bound is empty - @requires must specify at least one perspective")

        expr_perspective = self._get_expr_perspective(node)

        if not expr_perspective.is_broader(bound_perspective):
            raise PerspectiveBoundError(context, expr_perspective, bound_perspective)

    def visit_With(self, node: ast.With) -> Any:
        """
        Visit a with statement node.

        Handles:
        - with group(perspective): <body>
        - with partition(ptr, p=perspective, f=lambda) as new_name: <body>
        - with claim(ptr, p=perspective) as new_name: <body>
        """
        if len(node.items) != 1:
            raise InvalidAnnotationError("with statement must have exactly one item")

        item = node.items[0]
        context_expr = item.context_expr

        if not isinstance(context_expr, ast.Call):
            raise InvalidAnnotationError("with statement must use group(), partition(), or claim()")
        if not isinstance(context_expr.func, ast.Name):
            raise InvalidAnnotationError("with statement must use group(), partition(), or claim()")

        func_name = context_expr.func.id

        if func_name == "group":
            self._visit_with_group(node, context_expr, item)
        elif func_name == "partition":
            self._visit_with_partition(node, context_expr, item)
        elif func_name == "claim":
            self._visit_with_claim(node, context_expr, item)
        elif func_name == "wgmma_async":
            # wgmma_async is purely a codegen wrapper, just visit the body
            for stmt in node.body:
                self.visit(stmt)
        else:
            raise InvalidAnnotationError("with statement must use group(), partition(), claim(), or wgmma_async()")

    def _visit_with_group(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """Handle with group(perspective): <body>"""
        if len(context_expr.args) != 1:
            raise InvalidAnnotationError("group() takes exactly one argument")

        # Parse the perspective
        requested_perspective = self._parse_perspective(context_expr.args[0])

        # Save current bound and update to new bound
        old_bound = self.perspective_bound
        self.perspective_bound = self.perspective_bound.update(requested_perspective)

        # Visit body
        for stmt in node.body:
            self.visit(stmt)

        # Restore old bound
        self.perspective_bound = old_bound

    def _visit_with_partition(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """
        Handle with partition(ptr, p=perspective, f=lambda i: expr) as new_name: <body>

        Rules:
        1. ptr must be a non-local pointer
        2. p (target perspective) must be strictly narrower than ptr's perspective
        3. All variables in f (the lambda) must have perspective broader than or equal to p
        4. new_name is added to scope with ptr's type at perspective p
        5. ptr is removed from scope for the duration of the partition
        """
        # Must have 'as new_name'
        if item.optional_vars is None:
            raise InvalidPartitionError("partition requires 'as <name>' to bind the result")
        if not isinstance(item.optional_vars, ast.Name):
            raise InvalidPartitionError("partition 'as' target must be a simple name")
        new_name = item.optional_vars.id

        # Parse arguments: partition(ptr, p=perspective, f=lambda)
        if len(context_expr.args) != 1:
            raise InvalidPartitionError("partition() requires exactly one positional argument (the pointer)")

        # Get the pointer argument
        ptr_arg = context_expr.args[0]
        if not isinstance(ptr_arg, ast.Name):
            raise InvalidPartitionError("partition() first argument must be a variable name")
        ptr_name = ptr_arg.id

        # Look up the pointer in symbol table
        ptr_decl = self.lookup_with_partition_guard(ptr_name)

        # Check it's a pointer
        if not self._contains_ptr(ptr_decl.var_type):
            raise PartitionNonPointerError(ptr_name, ptr_decl.var_type)

        # Check it's not a LOCAL pointer
        if self._is_local_ptr(ptr_decl.var_type):
            raise PartitionLocalPointerError(ptr_name)

        # Parse keyword arguments: p=perspective, f=lambda or offset=expr
        target_perspective = None
        lambda_node = None
        offset_node = None

        for kw in context_expr.keywords:
            if kw.arg == "p":
                target_perspective = self._parse_perspective(kw.value)
            elif kw.arg == "f":
                if not isinstance(kw.value, ast.Lambda):
                    raise InvalidPartitionError("partition 'f' argument must be a lambda")
                lambda_node = kw.value
            elif kw.arg == "offset":
                offset_node = kw.value
            else:
                raise InvalidPartitionError(f"Unknown keyword argument '{kw.arg}'. Expected 'p', 'f', or 'offset'.")

        if target_perspective is None:
            raise InvalidPartitionError("partition() requires 'p' keyword argument for target perspective")
        if lambda_node is None and offset_node is None:
            raise InvalidPartitionError("partition() requires either 'f' (lambda) or 'offset' (expression) argument")
        if lambda_node is not None and offset_node is not None:
            raise InvalidPartitionError("partition() cannot have both 'f' and 'offset' arguments")

        # Check target perspective is narrower than or equal to ptr's perspective
        if not ptr_decl.perspective.is_broader(target_perspective) and ptr_decl.perspective != target_perspective:
            raise PartitionPerspectiveError(ptr_name, ptr_decl.perspective, target_perspective)

        # Check that broadest bound equals pointer's perspective
        broadest = self.perspective_bound.broadest()
        if broadest != ptr_decl.perspective:
            raise InvalidPartitionError(
                f"Cannot partition '{ptr_name}' at {ptr_decl.perspective}: "
                f"current broadest bound is {broadest}, must equal pointer's perspective"
            )

        # Check variable is not already partitioned
        if self._is_partitioned(ptr_name):
            raise InvalidPartitionError(f"'{ptr_name}' is already partitioned")

        # Validate lambda or offset: all variables must have perspective broader than or equal to target
        if lambda_node is not None:
            self._validate_partition_lambda(lambda_node, target_perspective)
        else:
            self._validate_partition_offset(offset_node, target_perspective)

        # Mark as partitioned and push new scope
        self.partitioned_stack[-1].add(ptr_name)
        self.symbol_table.enter_scope()
        self.partitioned_stack.append(set())
        self.symbol_table.define(new_name, ptr_decl.var_type, target_perspective)

        # Visit body
        for stmt in node.body:
            self.visit(stmt)

        # Pop scope and allow re-partitioning after scope exits
        self.partitioned_stack.pop()
        self.partitioned_stack[-1].discard(ptr_name)
        self.symbol_table.exit_scope()

    def _visit_with_claim(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """
        Handle with claim(ptr, p=perspective, f=lambda or offset=expr) as new_name: <body>

        Like partition, but with additional constraint: new_name can only be
        accessed in one split branch.
        """
        # Must have 'as new_name'
        if item.optional_vars is None:
            raise InvalidClaimError("claim requires 'as <name>' to bind the result")
        if not isinstance(item.optional_vars, ast.Name):
            raise InvalidClaimError("claim 'as' target must be a simple name")
        new_name = item.optional_vars.id

        # Parse arguments: claim(ptr, p=perspective, f=lambda or offset=expr)
        if len(context_expr.args) != 1:
            raise InvalidClaimError("claim() requires exactly one positional argument (the pointer)")

        # Get the pointer argument
        ptr_arg = context_expr.args[0]
        if not isinstance(ptr_arg, ast.Name):
            raise InvalidClaimError("claim() first argument must be a variable name")
        ptr_name = ptr_arg.id

        # Look up the pointer in symbol table
        ptr_decl = self.lookup_with_partition_guard(ptr_name)

        # Check it's a pointer
        if not self._contains_ptr(ptr_decl.var_type):
            raise InvalidClaimError(f"'{ptr_name}' is not a pointer")

        # Check it's not a LOCAL pointer
        if self._is_local_ptr(ptr_decl.var_type):
            raise InvalidClaimError(f"Cannot claim LOCAL pointer '{ptr_name}'")

        # Parse keyword arguments: p=perspective, f=lambda or offset=expr (both optional)
        target_perspective = None
        lambda_node = None
        offset_node = None

        for kw in context_expr.keywords:
            if kw.arg == "p":
                target_perspective = self._parse_perspective(kw.value)
            elif kw.arg == "f":
                if not isinstance(kw.value, ast.Lambda):
                    raise InvalidClaimError("claim 'f' argument must be a lambda")
                lambda_node = kw.value
            elif kw.arg == "offset":
                offset_node = kw.value
            else:
                raise InvalidClaimError(f"Unknown keyword argument '{kw.arg}'. Expected 'p', 'f', or 'offset'.")

        if target_perspective is None:
            raise InvalidClaimError("claim() requires 'p' keyword argument for target perspective")
        if lambda_node is not None and offset_node is not None:
            raise InvalidClaimError("claim() cannot have both 'f' and 'offset' arguments")

        # Check target perspective is strictly narrower than ptr's perspective
        if not ptr_decl.perspective.is_broader(target_perspective) or ptr_decl.perspective == target_perspective:
            raise InvalidClaimError(
                f"Cannot claim '{ptr_name}' at {ptr_decl.perspective} "
                f"to {target_perspective}: target must be strictly narrower"
            )

        # Check that broadest bound equals pointer's perspective
        broadest = self.perspective_bound.broadest()
        if broadest != ptr_decl.perspective:
            raise InvalidClaimError(
                f"Cannot claim '{ptr_name}' at {ptr_decl.perspective}: "
                f"current broadest bound is {broadest}, must equal pointer's perspective"
            )

        # Check variable is not already partitioned
        if self._is_partitioned(ptr_name):
            raise InvalidClaimError(f"'{ptr_name}' is already partitioned")

        # Validate lambda or offset if provided
        if lambda_node is not None:
            self._validate_partition_lambda(lambda_node, target_perspective)
        elif offset_node is not None:
            self._validate_partition_offset(offset_node, target_perspective)

        # Mark as partitioned (original pointer invalidated) and push new scope
        self.partitioned_stack[-1].add(ptr_name)
        self.symbol_table.enter_scope()
        self.partitioned_stack.append(set())
        self.claimed_stack.append({})
        self.symbol_table.define(new_name, ptr_decl.var_type, target_perspective)

        # Register this as a claimed variable (maps new_name to original ptr_name)
        self.claimed_stack[-1][new_name] = ptr_name

        # Visit body
        for stmt in node.body:
            self.visit(stmt)

        # Pop scope and allow re-claiming after scope exits
        self.claimed_stack.pop()
        self.partitioned_stack.pop()
        self.partitioned_stack[-1].discard(ptr_name)
        self.symbol_table.exit_scope()

    def _is_partitioned(self, var_name: str) -> bool:
        """Check if a variable is partitioned at any scope level."""
        for partitioned_set in self.partitioned_stack:
            if var_name in partitioned_set:
                return True
        return False

    def _check_not_partitioned(self, var_name: str) -> None:
        """Raise an error if the variable has been partitioned."""
        if self._is_partitioned(var_name):
            raise InvalidPartitionError(f"'{var_name}' has been partitioned and cannot be used")

    def _is_claimed(self, var_name: str) -> bool:
        """Check if a variable is a claimed variable at any scope level."""
        for claimed_dict in self.claimed_stack:
            if var_name in claimed_dict:
                return True
        return False

    def _check_claimed_access(self, var_name: str) -> None:
        """Track access to a claimed variable for single-branch enforcement."""
        if self._is_claimed(var_name):
            if self.split_branch_depth == 0:
                raise InvalidClaimError(
                    f"Claimed variable '{var_name}' can only be accessed inside a split branch"
                )
            self.claimed_accessed_in_branch.add(var_name)

    def _validate_partition_lambda(self, lambda_node: ast.Lambda, target_perspective: Perspective) -> None:
        """
        Validate that all variables in the partition lambda have perspective
        broader than or equal to the target perspective.

        Also transforms id() calls to id(level[index]).
        """
        # Temporarily define lambda params in a new scope so they're recognized
        self.symbol_table.enter_scope()
        for arg in lambda_node.args.args:
            # Lambda params are treated as having the target perspective
            self.symbol_table.define(arg.arg, Type.int, target_perspective)

        # Use _get_expr_perspective to validate and transform id() calls
        body_perspective = self._get_expr_perspective(lambda_node.body, target_perspective)

        self.symbol_table.exit_scope()

        if not body_perspective.is_broader(target_perspective):
            raise InvalidPartitionError(
                f"Lambda body has perspective {body_perspective}, "
                f"which is not broader than or equal to target perspective {target_perspective}"
            )

    def _validate_partition_offset(self, offset_node: ast.AST, target_perspective: Perspective) -> None:
        """
        Validate that the partition offset expression has perspective
        broader than or equal to the target perspective.
        """
        offset_perspective = self._get_expr_perspective(offset_node, target_perspective)
        if not offset_perspective.is_broader(target_perspective):
            raise InvalidPartitionError(
                f"Offset expression has perspective {offset_perspective}, "
                f"which is not broader than or equal to target perspective {target_perspective}"
            )

    def visit_Match(self, node: ast.Match) -> Any:
        """
        Visit a match statement node.

        Handles: match split(level):
                     case N:
                         <body>
        """
        # Check it's a split() call
        subject = node.subject
        if not isinstance(subject, ast.Call):
            raise InvalidAnnotationError("match statement must use split()")
        if not isinstance(subject.func, ast.Name) or subject.func.id != "split":
            raise InvalidAnnotationError("match statement must use split()")
        if len(subject.args) != 1:
            raise InvalidAnnotationError("split() takes exactly one argument (grid, block, or thread)")

        # Get the level being split
        level_arg = subject.args[0]
        if not isinstance(level_arg, ast.Name):
            raise InvalidAnnotationError("split() argument must be grid, block, or thread")
        level_name = level_arg.id
        if level_name not in ("grid", "block", "thread"):
            raise InvalidAnnotationError(f"split() argument must be grid, block, or thread, got {level_name}")
        level = Level[level_name]

        # Get the current perspective at this level from the bound
        if level == Level.grid:
            current_perspective = self.perspective_bound.grid
        elif level == Level.block:
            current_perspective = self.perspective_bound.block
        else:
            current_perspective = self.perspective_bound.thread

        if current_perspective is None:
            raise InvalidAnnotationError(f"Cannot split on {level_name}: level not available in current bound")

        # Track the previous case's perspective (starts with current bound's perspective)
        previous_perspective = current_perspective
        old_bound = self.perspective_bound

        # Track partitioned variables across sibling branches
        sibling_partitioned: set[str] = set()
        # Track claimed variables accessed across sibling branches
        sibling_claimed_accessed: set[str] = set()

        for case in node.cases:
            # Each case pattern must be a constant integer
            pattern = case.pattern
            if not isinstance(pattern, ast.MatchValue):
                raise InvalidAnnotationError("split case must be a constant integer")
            if not isinstance(pattern.value, ast.Constant) or not isinstance(pattern.value.value, int):
                raise InvalidAnnotationError("split case must be a constant integer")

            case_value = pattern.value.value

            # Create the requested perspective for this case
            requested = Perspective(level, case_value)

            # Check that case_value fits within available threads/blocks
            # Split uses subtraction (not division), so we check index >= case_value
            if current_perspective.index < case_value:
                raise InvalidSplitError(case_value, level_name, current_perspective)
            if previous_perspective.index < case_value:
                raise InvalidSplitError(case_value, level_name, previous_perspective)

            # Compute the bound for this branch: previous.index - case_value
            remaining_count = previous_perspective.index - case_value
            remaining = Perspective(level, remaining_count)

            if level == Level.grid:
                self.perspective_bound = PerspectiveBound(
                    grid=requested,
                    block=old_bound.block,
                    thread=old_bound.thread
                )
            elif level == Level.block:
                self.perspective_bound = PerspectiveBound(
                    grid=None,
                    block=requested,
                    thread=old_bound.thread
                )
            else:
                self.perspective_bound = PerspectiveBound(
                    grid=None,
                    block=None,
                    thread=requested
                )

            # Push scope for this branch
            self.partitioned_stack.append(set())

            # Reset claimed access tracking for this branch
            old_claimed_accessed = self.claimed_accessed_in_branch
            self.claimed_accessed_in_branch = set()

            # Enter split branch context
            self.split_branch_depth += 1

            # Visit the body of this case
            for stmt in case.body:
                self.visit(stmt)

            # Exit split branch context
            self.split_branch_depth -= 1

            # Check for conflicts with previous siblings (partitioned)
            branch_partitioned = self.partitioned_stack[-1]
            conflicts = branch_partitioned & sibling_partitioned
            if conflicts:
                raise InvalidPartitionError(
                    f"Variable '{next(iter(conflicts))}' partitioned in multiple split branches"
                )

            # Check for conflicts with previous siblings (claimed access)
            claimed_conflicts = self.claimed_accessed_in_branch & sibling_claimed_accessed
            if claimed_conflicts:
                raise ClaimMultipleBranchError(next(iter(claimed_conflicts)))

            # Accumulate for next sibling
            sibling_partitioned |= branch_partitioned
            sibling_claimed_accessed |= self.claimed_accessed_in_branch

            # Restore claimed access tracking
            self.claimed_accessed_in_branch = old_claimed_accessed

            # Pop scope
            self.partitioned_stack.pop()

            # Update previous_perspective for the next case
            previous_perspective = remaining

        # Restore old bound
        self.perspective_bound = old_bound

    def check(self, ast_node: ast.AST):
        """
        Run the type checker on an AST node.

        Args:
            ast_node: The AST node to check

        Raises:
            TypeError: If type checking fails
        """
        self.visit(ast_node)


def check_function(func_type: str, ast_node: ast.FunctionDef) -> PerspectiveSignature:
    """
    Type check a prism function.

    Args:
        func_type: Either "device" or "global"
        ast_node: The function's AST node

    Returns:
        The function's PerspectiveSignature.

    Raises:
        TypeError: If type checking fails
    """
    from prism.registry import registry

    # Check cache first
    func_name = ast_node.name
    cached_sig = registry.get_signature(func_name)
    if cached_sig is not None:
        return cached_sig

    # Type check and cache
    checker = TypeChecker(func_type)
    checker.check(ast_node)
    registry.set_signature(func_name, checker.signature)
    return checker.signature
