"""
Code generator: produces CUDA C source from type-checked prism ASTs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from prism.types import Type, PrismType, PtrType, ConstType, RestrictType, ConstExprType, MemorySpace
from prism.perspective import Level
from prism.signature import PerspectiveSignature, ParamSignature


# ---------------------------------------------------------------------------
# Context snapshot — saved/restored around group and split
# ---------------------------------------------------------------------------

@dataclass
class _Context:
    block_id: str   # e.g. "__block_id_0"
    thread_id: str  # e.g. "__thread_id_0"
    depth: int      # nesting depth for unique variable naming


# ---------------------------------------------------------------------------
# CodeGenerator
# ---------------------------------------------------------------------------

class CodeGenerator(ast.NodeVisitor):
    """Walk a type-checked prism AST and emit CUDA C source."""

    def __init__(self, func_type: str, signature: PerspectiveSignature):
        self._func_type = func_type
        self._signature = signature
        self._lines: list[str] = []
        self._indent: int = 0
        self._ctx = _Context(block_id="__block_id_0", thread_id="__thread_id_0", depth=0)
        self._smem_offset: int = 0
        self._var_types: dict[str, PrismType] = {}
        # Lambda parameter bindings: name -> C expression string
        self._lambda_bindings: dict[str, str] = {}
        # Partition bindings for f=lambda: name -> (original_ptr, lambda_node, perspective)
        self._partition_bindings: dict[str, tuple[str, ast.Lambda, ast.AST]] = {}
        # Split-barrier state: current SplitExpect node (for load_tma → load_async)
        self._current_expect = None

    # -- output helpers -----------------------------------------------------

    def _emit(self, line: str = "") -> None:
        if line:
            self._lines.append("    " * self._indent + line)
        else:
            self._lines.append("")

    def get_code(self) -> str:
        return "\n".join(self._lines) + "\n"

    # -- type helpers -------------------------------------------------------

    def _cuda_type(self, t: PrismType) -> str:
        if isinstance(t, Type):
            return t.get_c_str()
        if isinstance(t, PtrType):
            # shared(T) already produces PtrType(T, SHARED); when wrapped in
            # ptr(), this creates PtrType(PtrType(T, SHARED)) — collapse to
            # avoid emitting double pointers (e.g. uint32_t**).
            inner = self._unwrap_shared_ptr(t.inner)
            return f"{self._cuda_type(inner)}*"
        if isinstance(t, ConstType):
            return f"const {self._cuda_type(t.inner)}"
        if isinstance(t, RestrictType):
            inner = self._cuda_type(t.inner)
            # __restrict__ applies to the pointer, so insert before the last *
            if inner.endswith("*"):
                return f"{inner} __restrict__"
            return f"{inner} __restrict__"
        if isinstance(t, ConstExprType):
            return self._cuda_type(t.inner)
        return str(t)

    def _unwrap_shared_ptr(self, t: PrismType) -> PrismType:
        """Strip PtrType(T, SHARED) when nested inside an outer PtrType.

        shared(T) in the type system returns PtrType(T, SHARED) — already a
        pointer.  When the user writes ptr(shared(T)) or ptr(const(shared(T))),
        the extra PtrType layer is an artefact, not a real indirection level.
        """
        if isinstance(t, PtrType) and t.memory_space == MemorySpace.SHARED:
            return t.inner
        if isinstance(t, (ConstType, RestrictType)):
            unwrapped = self._unwrap_shared_ptr(t.inner)
            if unwrapped is not t.inner:
                return type(t)(unwrapped)
        return t

    def _base_type_from_prism(self, t: PrismType) -> Type:
        """Unwrap modifiers to find the base Type enum."""
        if isinstance(t, Type):
            return t
        if isinstance(t, (PtrType, ConstType, RestrictType, ConstExprType)):
            return self._base_type_from_prism(t.inner)
        raise ValueError(f"Cannot find base type of {t}")

    def _type_from_annotation(self, annotation: ast.AST) -> str:
        """Extract the C type string from an AST annotation (type @ perspective)."""
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
            type_node = annotation.left
        else:
            type_node = annotation
        return self._type_from_annotation_type_node(type_node)

    def _type_from_annotation_type_node(self, type_node: ast.AST) -> str:
        """Resolve a type AST node to a C type string."""
        # Simple name → scalar
        if isinstance(type_node, ast.Name):
            return Type[type_node.id].get_c_str()
        # Subscript → local array type (e.g. int[16] or int[1][16][8])
        if isinstance(type_node, ast.Subscript):
            # Unwrap nested subscripts to find base type
            current = type_node
            while isinstance(current, ast.Subscript):
                current = current.value
            if isinstance(current, ast.Name):
                return Type[current.id].get_c_str()
        # Call-based types: const(...), constexpr(...), shared(...), ptr(...)
        if isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name):
            modifier = type_node.func.id
            if modifier == "shared":
                arg = type_node.args[0]  # type[size]
                # Unwrap to find base type name
                current = arg
                while isinstance(current, ast.Subscript):
                    current = current.value
                return Type[current.id].get_c_str()
            if modifier == "const":
                inner = self._type_from_annotation_type_node(type_node.args[0])
                return f"const {inner}"
            if modifier == "constexpr":
                inner = self._type_from_annotation_type_node(type_node.args[0])
                return f"constexpr {inner}"
            if modifier == "ptr":
                inner = self._type_from_annotation_type_node(type_node.args[0])
                return f"{inner}*"
        return "int"

    # -- ID expressions -----------------------------------------------------

    def _id_expr(self, level_name: str, index: int) -> str:
        if level_name == "grid":
            return "0"
        if level_name == "block":
            if index == 1:
                return self._ctx.block_id
            return f"({self._ctx.block_id} / {index})"
        if level_name == "thread":
            if index == 1:
                return self._ctx.thread_id
            return f"({self._ctx.thread_id} / {index})"
        return "0"

    # -- expression generation ----------------------------------------------

    def _expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            return str(node.value)

        if isinstance(node, ast.Name):
            # Check lambda bindings first
            if node.id in self._lambda_bindings:
                return self._lambda_bindings[node.id]
            return node.id

        if isinstance(node, ast.BinOp):
            left = self._expr(node.left)
            right = self._expr(node.right)
            op = self._op(node.op)
            return f"({left} {op} {right})"

        if isinstance(node, ast.UnaryOp):
            operand = self._expr(node.operand)
            op = self._unary_op(node.op)
            if isinstance(node.op, ast.Not):
                return f"(!{operand})"
            return f"({op}{operand})"

        if isinstance(node, ast.Compare):
            result = self._expr(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                result = f"({result} {self._cmp_op(op)} {self._expr(comparator)})"
            return result

        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || "
            parts = [self._expr(v) for v in node.values]
            return f"({op.join(parts)})"

        if isinstance(node, ast.Subscript):
            # Partition-bound variable: transform index through lambda
            if isinstance(node.value, ast.Name) and node.value.id in self._partition_bindings:
                ptr_name, lambda_node, _ = self._partition_bindings[node.value.id]
                saved = self._lambda_bindings.copy()
                param_name = lambda_node.args.args[0].arg
                self._lambda_bindings[param_name] = self._expr(node.slice)
                transformed_index = self._expr(lambda_node.body)
                self._lambda_bindings = saved
                return f"{ptr_name}[{transformed_index}]"
            base = self._expr(node.value)
            index = self._expr(node.slice)
            return f"{base}[{index}]"

        if isinstance(node, ast.Attribute):
            return f"{self._expr(node.value)}.{node.attr}"

        if isinstance(node, ast.Call):
            return self._call_expr(node)

        if isinstance(node, ast.IfExp):
            test = self._expr(node.test)
            body = self._expr(node.body)
            orelse = self._expr(node.orelse)
            return f"({test} ? {body} : {orelse})"

        return f"/* unsupported: {ast.dump(node)} */"

    def _call_expr(self, node: ast.Call) -> str:
        """Generate a C expression for a function call."""
        # Attribute-based calls: intrinsic.mma(...) etc.
        if isinstance(node.func, ast.Attribute):
            return f"/* unsupported attribute call */"

        if not isinstance(node.func, ast.Name):
            return f"/* unsupported call */"

        func_name = node.func.id

        # float4_cast(expr) → reinterpret_cast<float4 *>(&expr)[0]
        if func_name == "float4_cast" and node.args:
            inner = self._expr(node.args[0])
            return f"reinterpret_cast<float4 *>(&{inner})[0]"

        # id(level[index]) — rewritten by type checker
        if func_name == "id" and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                level_name = arg.value.id
                index = arg.slice.value if isinstance(arg.slice, ast.Constant) else 1
                return self._id_expr(level_name, index)

        # Check if this is a registered prism function call
        from prism.registry import registry
        result = registry.get(func_name)
        if result is not None:
            return self._prism_call_expr(node, func_name)

        # Generic C function call
        template_str = ""
        for kw in node.keywords:
            if kw.arg == "template" and isinstance(kw.value, ast.List):
                targs = [self._expr(e) for e in kw.value.elts]
                template_str = f"<{', '.join(targs)}>"
        args = [self._expr(a) for a in node.args]
        return f"{func_name}{template_str}({', '.join(args)})"

    def _prism_call_expr(self, node: ast.Call, func_name: str) -> str:
        """Generate a call to another prism function with context args."""
        # load_tma → load_async rewrite when SplitExpect is in scope
        if func_name == "load_tma" and self._current_expect is not None:
            return self._rewrite_load_tma_to_async(node)

        # Template arguments
        template_str = ""
        for kw in node.keywords:
            if kw.arg == "template" and isinstance(kw.value, ast.List):
                targs = [self._expr(e) for e in kw.value.elts]
                template_str = f"<{', '.join(targs)}>"

        # Context args: block_id, thread_id, smem pointer
        ctx_args = [self._ctx.block_id, self._ctx.thread_id, "__smem"]
        # User args
        user_args = [self._expr(a) for a in node.args]
        all_args = ctx_args + user_args

        return f"{func_name}{template_str}({', '.join(all_args)})"

    def _rewrite_load_tma_to_async(self, node: ast.Call) -> str:
        """Rewrite load_tma(dst, src, col, row) → load_async(dst, src, barrier, col, row)."""
        ctx_args = [self._ctx.block_id, self._ctx.thread_id, "__smem"]
        user_args = [self._expr(a) for a in node.args]
        # Insert barrier name as 3rd user arg (after dst, src_tma_map_ptr)
        barrier = self._current_expect.barrier_name
        rewritten = user_args[:2] + [barrier] + user_args[2:]
        all_args = ctx_args + rewritten
        return f"load_async({', '.join(all_args)})"

    def _op(self, op: ast.operator) -> str:
        ops = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
            ast.Div: "/", ast.Mod: "%", ast.FloorDiv: "/",
            ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
            ast.LShift: "<<", ast.RShift: ">>",
        }
        return ops.get(type(op), "?")

    def _cmp_op(self, op: ast.cmpop) -> str:
        ops = {
            ast.Eq: "==", ast.NotEq: "!=",
            ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=",
        }
        return ops.get(type(op), "?")

    def _unary_op(self, op: ast.unaryop) -> str:
        ops = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~", ast.Not: "!"}
        return ops.get(type(op), "?")

    # -- visitor methods ----------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        sig = self._signature

        # Find template parameter index (same logic as type checker)
        template_params = self._get_template_params(node)

        # Build parameter list
        params: list[str] = []
        if self._func_type == "device":
            params.extend([
                "int __block_id", "int __thread_id", "char* __smem"
            ])

        for ps in sig.params:
            params.append(f"{self._cuda_type(ps.param_type)} {ps.name}")

        param_str = ", ".join(params)

        # Emit template line if needed
        if template_params:
            tparams = [f"{self._cuda_type(t)} {n}" for n, t in template_params]
            self._emit(f"template<{', '.join(tparams)}>")

        qualifier = "__device__" if self._func_type == "device" else "__global__"

        # Check for @attr("...") decorators
        attr_str = ""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "attr":
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    attr_str = dec.args[0].value

        ret_type = "void"
        if sig.return_type is not None:
            ret_type = self._cuda_type(sig.return_type)

        if attr_str:
            self._emit(f"{qualifier} {attr_str} {ret_type} {sig.name}({param_str}) {{")
        else:
            self._emit(f"{qualifier} {ret_type} {sig.name}({param_str}) {{")
        self._indent += 1

        # Shared memory declaration (global functions only)
        if self._func_type == "global":
            self._emit("extern __shared__ __align__(128) char __smem[];")

        # Initialize ID variables
        if self._func_type == "global":
            self._emit(f"int {self._ctx.block_id} = blockIdx.x;")
            self._emit(f"int {self._ctx.thread_id} = {self._ctx.block_id} * blockDim.x + threadIdx.x;")
        else:
            self._emit(f"int {self._ctx.block_id} = __block_id;")
            self._emit(f"int {self._ctx.thread_id} = __thread_id;")

        # Track parameter types
        for ps in sig.params:
            self._var_types[ps.name] = ps.param_type

        # Visit body (skip decorators and args — they're handled above)
        for stmt in node.body:
            self.visit(stmt)

        self._indent -= 1
        self._emit("}")

    def _get_template_params(self, node: ast.FunctionDef) -> list[tuple[str, PrismType]]:
        """Extract template parameter names and types from the AST."""
        from prism.types import Type as PType
        args = node.args
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_index = num_args - num_defaults

        for i, arg in enumerate(args.args):
            if arg.arg == "template":
                default_index = i - first_default_index
                if default_index >= 0 and default_index < len(args.defaults):
                    default = args.defaults[default_index]
                    if isinstance(default, ast.List):
                        result = []
                        for item in default.elts:
                            if isinstance(item, ast.Tuple) and len(item.elts) == 2:
                                name_node, type_node = item.elts
                                name = name_node.value
                                t = self._parse_template_type(type_node)
                                result.append((name, t))
                        return result
        return []

    def _parse_template_type(self, node: ast.AST) -> PrismType:
        """Parse a type from template parameter declaration."""
        if isinstance(node, ast.Name):
            return Type[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "constexpr":
                inner = self._parse_template_type(node.args[0])
                return ConstExprType(inner)
        return Type.int

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        var_name = node.target.id
        annotation = node.annotation

        # Detect which kind of declaration
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
            type_node = annotation.left
        else:
            type_node = annotation

        # Case 1: Shared memory — shared(type[size]) @ block[1]
        if isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name) and type_node.func.id == "shared":
            self._emit_shared_decl(var_name, type_node)
            return

        # Case 2: Local array — type[size] or type[s1][s2][s3] @ thread[1]
        if isinstance(type_node, ast.Subscript):
            # Unwrap nested subscripts to find base type and collect dimensions
            current = type_node
            dims = []
            while isinstance(current, ast.Subscript):
                dims.append(current.slice)
                current = current.value
            # Check if it's a type name (int, float, etc.) not a level name
            if isinstance(current, ast.Name):
                type_name = current.id
                _array_types = ("int", "float", "double", "bf16", "bool", "uint", "uint32_t", "float4")
                if type_name in _array_types:
                    c_type = Type[type_name].get_c_str()
                    dims.reverse()  # outer-most dimension first
                    dim_str = "".join(f"[{self._expr(d)}]" for d in dims)
                    # Check for zero-init: = {0} parsed as ast.Set(elts=[Constant(0)])
                    if (node.value is not None and isinstance(node.value, ast.Set)
                            and len(node.value.elts) == 1
                            and isinstance(node.value.elts[0], ast.Constant)
                            and node.value.elts[0].value == 0):
                        self._emit(f"{c_type} {var_name}{dim_str} = {{0}};")
                    else:
                        self._emit(f"{c_type} {var_name}{dim_str};")
                    self._var_types[var_name] = PtrType(Type[type_name], MemorySpace.LOCAL)
                    return

        # Case 3: Scalar
        c_type = self._type_from_annotation(annotation)
        if node.value is not None:
            self._emit(f"{c_type} {var_name} = {self._expr(node.value)};")
        else:
            self._emit(f"{c_type} {var_name};")

        # Track type for scalar or pointer
        if isinstance(type_node, ast.Name):
            try:
                self._var_types[var_name] = Type[type_node.id]
            except KeyError:
                pass
        elif isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name) and type_node.func.id == "ptr":
            # ptr(type) or ptr(type, SPACE)
            if type_node.args and isinstance(type_node.args[0], ast.Name):
                try:
                    self._var_types[var_name] = PtrType(Type[type_node.args[0].id], MemorySpace.GLOBAL)
                except KeyError:
                    pass

    def _emit_shared_decl(self, var_name: str, type_node: ast.Call) -> None:
        """Emit a shared memory declaration from shared(type[size], align=N, static=True)."""
        arg = type_node.args[0]  # type[size]
        type_name = arg.value.id
        c_type = Type[type_name].get_c_str()
        elem_bytes = Type[type_name].get_bytes()
        size_expr = self._expr(arg.slice)
        size_val = self._eval_const_expr(arg.slice)
        total_bytes = size_val * elem_bytes

        # Parse keyword arguments
        align = None
        is_static = False
        for kw in type_node.keywords:
            if kw.arg == "align" and isinstance(kw.value, ast.Constant):
                align = kw.value.value
            elif kw.arg == "static" and isinstance(kw.value, ast.Constant):
                is_static = kw.value.value

        if is_static:
            # Static shared memory: __shared__ __align__(N) type name[size];
            align_str = f"__align__({align}) " if align else ""
            self._emit(f"__shared__ {align_str}{c_type} {var_name}[{size_expr}];")
        else:
            if align:
                # Align the smem offset
                self._smem_offset = (self._smem_offset + (align - 1)) & ~(align - 1)
            self._emit(
                f"{'__align__(' + str(align) + ') ' if align else ''}"
                f"{c_type} (&{var_name})[{size_expr}] = "
                f"*(({c_type} (*)[{size_expr}])(__smem + {self._smem_offset}));"
            )
            self._smem_offset += total_bytes
        self._var_types[var_name] = PtrType(Type[type_name], MemorySpace.SHARED)

    def _eval_const_expr(self, node: ast.AST) -> int:
        """Evaluate a constant expression at codegen time."""
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
        if isinstance(node, ast.Name):
            return 0  # fallback for constexpr vars
        raise ValueError(f"Cannot evaluate constant expression: {ast.dump(node)}")

    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            # float4_cast[lhs] = rhs → reinterpret_cast
            if isinstance(target.value, ast.Name) and target.value.id == "float4_cast":
                lhs = self._expr(target.slice)
                rhs = self._expr(node.value)
                self._emit(f"reinterpret_cast<float4 *>(&{lhs})[0] = {rhs};")
                return
            # Partition-bound write-through: transform index through lambda
            if isinstance(target.value, ast.Name) and target.value.id in self._partition_bindings:
                ptr_name, lambda_node, _ = self._partition_bindings[target.value.id]
                saved = self._lambda_bindings.copy()
                param_name = lambda_node.args.args[0].arg
                self._lambda_bindings[param_name] = self._expr(target.slice)
                transformed_index = self._expr(lambda_node.body)
                self._lambda_bindings = saved
                value = self._expr(node.value)
                self._emit(f"{ptr_name}[{transformed_index}] = {value};")
                return
            # Write-through: p[expr] = value
            base = self._expr(target.value)
            index = self._expr(target.slice)
            value = self._expr(node.value)
            self._emit(f"{base}[{index}] = {value};")
        elif isinstance(target, ast.Attribute):
            obj = self._expr(target.value)
            self._emit(f"{obj}.{target.attr} = {self._expr(node.value)};")
        elif isinstance(target, ast.Name):
            self._emit(f"{target.id} = {self._expr(node.value)};")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        op = self._op(node.op)
        if isinstance(node.target, ast.Subscript):
            # Partition-bound aug-assign: transform index through lambda
            if isinstance(node.target.value, ast.Name) and node.target.value.id in self._partition_bindings:
                ptr_name, lambda_node, _ = self._partition_bindings[node.target.value.id]
                saved = self._lambda_bindings.copy()
                param_name = lambda_node.args.args[0].arg
                self._lambda_bindings[param_name] = self._expr(node.target.slice)
                transformed_index = self._expr(lambda_node.body)
                self._lambda_bindings = saved
                self._emit(f"{ptr_name}[{transformed_index}] {op}= {self._expr(node.value)};")
                return
            base = self._expr(node.target.value)
            index = self._expr(node.target.slice)
            self._emit(f"{base}[{index}] {op}= {self._expr(node.value)};")
        elif isinstance(node.target, ast.Attribute):
            obj = self._expr(node.target.value)
            self._emit(f"{obj}.{node.target.attr} {op}= {self._expr(node.value)};")
        elif isinstance(node.target, ast.Name):
            self._emit(f"{node.target.id} {op}= {self._expr(node.value)};")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self._emit(f"return {self._expr(node.value)};")
        else:
            self._emit("return;")

    def visit_Break(self, node: ast.Break) -> None:
        self._emit("break;")

    def visit_Continue(self, node: ast.Continue) -> None:
        self._emit("continue;")

    def visit_Pass(self, node: ast.Pass) -> None:
        pass  # no-op in C

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "unsafe" and node.value.args:
                if isinstance(node.value.args[0], ast.Constant):
                    self._emit(node.value.args[0].value)
                    return
        # intrinsic.mma(...) or intrinsic.wgmma256(...) call
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if (isinstance(node.value.func.value, ast.Name) and
                    node.value.func.value.id == "intrinsic"):
                if node.value.func.attr == "mma":
                    self._emit_mma(node.value)
                    return
                if node.value.func.attr == "wgmma256":
                    self._emit_wgmma256(node.value)
                    return
        self._emit(f"{self._expr(node.value)};")

    def _emit_mma(self, node: ast.Call) -> None:
        """Emit inline PTX mma.sync assembly from intrinsic.mma(...) call."""
        # Extract the 10 input args (a0..a3, b0..b1, c0..c3)
        inputs = [self._expr(a) for a in node.args]
        # Extract out=[...] keyword for 4 outputs
        outputs = []
        for kw in node.keywords:
            if kw.arg == "out" and isinstance(kw.value, ast.List):
                outputs = [self._expr(e) for e in kw.value.elts]

        a_args = inputs[0:4]
        b_args = inputs[4:6]
        c_args = inputs[6:10]

        self._emit("asm(")
        self._indent += 1
        self._emit('"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "')
        self._emit('"{%0, %1, %2, %3}, "')
        self._emit('"{%4, %5, %6, %7}, "')
        self._emit('"{%8, %9}, "')
        self._emit('"{%10, %11, %12, %13};"')

        # Output operands
        out_parts = [f'"+r"({o})' for o in outputs]
        self._emit(f': {", ".join(out_parts)}')

        # Input operands: a (4 regs), b (2 regs), c (4 regs)
        a_parts = [f'"r"({a})' for a in a_args]
        b_parts = [f'"r"({b})' for b in b_args]
        c_parts = [f'"r"({c})' for c in c_args]
        all_in = a_parts + b_parts + c_parts
        self._emit(f': {", ".join(all_in)}')

        self._indent -= 1
        self._emit(");")

    def _emit_wgmma256(self, node: ast.Call) -> None:
        """Emit wgmma<256, ...> call from intrinsic.wgmma256(ptrA, ptrB, ScaleD, ScaleA, ScaleB, TransA, TransB, out=[d])."""
        ptr_a = self._expr(node.args[0])
        ptr_b = self._expr(node.args[1])
        scale_d = self._expr(node.args[2])
        scale_a = self._expr(node.args[3])
        scale_b = self._expr(node.args[4])
        trans_a = self._expr(node.args[5])
        trans_b = self._expr(node.args[6])

        # Extract out= keyword
        out_expr = ""
        for kw in node.keywords:
            if kw.arg == "out" and isinstance(kw.value, ast.List):
                out_expr = self._expr(kw.value.elts[0])

        self._emit(f"wgmma<256, {scale_d}, {scale_a}, {scale_b}, {trans_a}, {trans_b}>({out_expr}, &{ptr_a}, &{ptr_b});")

    # -- control flow -------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        self._emit(f"if ({self._expr(node.test)}) {{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._indent -= 1

        # Handle elif/else chain
        orelse = node.orelse
        while orelse:
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                elif_node = orelse[0]
                self._emit(f"}} else if ({self._expr(elif_node.test)}) {{")
                self._indent += 1
                for stmt in elif_node.body:
                    self.visit(stmt)
                self._indent -= 1
                orelse = elif_node.orelse
            else:
                self._emit("} else {")
                self._indent += 1
                for stmt in orelse:
                    self.visit(stmt)
                self._indent -= 1
                orelse = None

        self._emit("}")

    def visit_For(self, node: ast.For) -> None:
        # for i in range(...)
        loop_var = node.target.id
        range_call = node.iter
        args = range_call.args

        if len(args) == 1:
            start, stop, step = "0", self._expr(args[0]), "1"
        elif len(args) == 2:
            start, stop, step = self._expr(args[0]), self._expr(args[1]), "1"
        else:
            start, stop, step = self._expr(args[0]), self._expr(args[1]), self._expr(args[2])

        if step == "1":
            self._emit(f"for (int {loop_var} = {start}; {loop_var} < {stop}; {loop_var}++) {{")
        else:
            self._emit(f"for (int {loop_var} = {start}; {loop_var} < {stop}; {loop_var} += {step}) {{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._indent -= 1
        self._emit("}")

    def visit_While(self, node: ast.While) -> None:
        self._emit(f"while ({self._expr(node.test)}) {{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._indent -= 1
        self._emit("}")

    # -- with constructs: group, partition, claim ---------------------------

    def visit_With(self, node: ast.With) -> None:
        item = node.items[0]
        context_expr = item.context_expr
        func_name = context_expr.func.id

        if func_name == "group":
            self._visit_with_group(node, context_expr)
        elif func_name == "partition":
            self._visit_with_partition(node, context_expr, item)
        elif func_name == "claim":
            self._visit_with_claim(node, context_expr, item)
        elif func_name == "wgmma_async":
            self._visit_with_wgmma_async(node)

    def _visit_with_group(self, node: ast.With, context_expr: ast.Call) -> None:
        """Generate code for `with group(level[N]): ...`"""
        perspective_node = context_expr.args[0]
        level_name = perspective_node.value.id
        index = perspective_node.slice.value

        saved_ctx = self._ctx
        new_depth = saved_ctx.depth + 1
        new_block_id = f"__block_id_{new_depth}"
        new_thread_id = f"__thread_id_{new_depth}"

        self._emit("{")
        self._indent += 1

        if level_name == "block":
            self._emit(f"int {new_block_id} = {saved_ctx.block_id} % {index};")
            self._emit(f"int {new_thread_id} = {new_block_id} * blockDim.x + threadIdx.x;")
        elif level_name == "thread":
            self._emit(f"int {new_block_id} = {saved_ctx.block_id};")
            self._emit(f"int {new_thread_id} = {saved_ctx.thread_id} % {index};")
        elif level_name == "grid":
            self._emit(f"int {new_block_id} = {saved_ctx.block_id};")
            self._emit(f"int {new_thread_id} = {saved_ctx.thread_id};")

        self._ctx = _Context(block_id=new_block_id, thread_id=new_thread_id, depth=new_depth)

        for stmt in node.body:
            self.visit(stmt)

        self._indent -= 1
        self._emit("}")
        self._ctx = saved_ctx

    def _visit_with_partition(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """Generate code for `with partition(ptr, p=..., f=lambda/offset=...) as new_name: ...`"""
        self._emit_partition_or_claim(node, context_expr, item)

    def _visit_with_claim(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """Generate code for `with claim(ptr, p=..., f=lambda/offset=...) as new_name: ...`"""
        self._emit_partition_or_claim(node, context_expr, item)

    def _emit_partition_or_claim(self, node: ast.With, context_expr: ast.Call, item: ast.withitem) -> None:
        """Shared codegen for partition and claim — both produce pointer arithmetic."""
        new_name = item.optional_vars.id
        ptr_name = context_expr.args[0].id

        # Parse keyword arguments
        target_perspective = None
        lambda_node = None
        offset_node = None

        for kw in context_expr.keywords:
            if kw.arg == "p":
                target_perspective = kw.value
            elif kw.arg == "f":
                lambda_node = kw.value
            elif kw.arg == "offset":
                offset_node = kw.value

        if lambda_node is not None:
            # f=lambda case: no pointer declaration, transform index on every access
            self._partition_bindings[new_name] = (ptr_name, lambda_node, target_perspective)
            ptr_type = self._var_types.get(ptr_name)
            self._var_types[new_name] = ptr_type

            for stmt in node.body:
                self.visit(stmt)

            del self._partition_bindings[new_name]
        else:
            # offset= case: emit pointer shift as before
            if offset_node is not None:
                offset_c = self._expr(offset_node)
            else:
                offset_c = "0"

            ptr_type = self._var_types.get(ptr_name)
            if ptr_type is not None:
                c_type = self._cuda_type(ptr_type)
            else:
                c_type = "int*"

            self._emit("{")
            self._indent += 1
            self._emit(f"{c_type} {new_name} = {ptr_name} + ({offset_c});")
            self._var_types[new_name] = ptr_type

            for stmt in node.body:
                self.visit(stmt)

            self._indent -= 1
            self._emit("}")

    def _visit_with_wgmma_async(self, node: ast.With) -> None:
        """Generate code for `with wgmma_async(): ...` — wraps body with commit/wait."""
        self._emit("{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._emit("warpgroup_commit_batch();")
        self._emit("warpgroup_wait<0>();")
        self._indent -= 1
        self._emit("}")

    def _lambda_offset_expr(self, lambda_node: ast.Lambda, perspective_node: ast.AST) -> str:
        """Generate the C expression for a partition lambda body.

        Temporarily binds lambda parameters to the appropriate id expression.
        """
        # Get the target level/index from the perspective node
        level_name = perspective_node.value.id
        index = perspective_node.slice.value

        # Bind lambda params to id expression
        saved_bindings = self._lambda_bindings.copy()
        for arg in lambda_node.args.args:
            self._lambda_bindings[arg.arg] = self._id_expr(level_name, index)

        result = self._expr(lambda_node.body)

        self._lambda_bindings = saved_bindings
        return result

    # -- sync nodes ---------------------------------------------------------

    def visit_BlockSync(self, node) -> None:
        self._emit("__syncthreads();")

    def visit_SplitWait(self, node) -> None:
        self._emit(f"wait_cluster({self._ctx.block_id}, {self._ctx.thread_id}, __smem, {node.barrier_name}, {node.phase_var});")

    def visit_SplitArrive(self, node) -> None:
        # One arrive per consumer warp group (thread_id % 128 == 0)
        self._emit(f"if ({self._ctx.thread_id} % 128 == 0) arrive({self._ctx.block_id}, {self._ctx.thread_id}, __smem, {node.barrier_name}, 1);")

    def visit_SplitExpect(self, node) -> None:
        self._emit(f"expect_bytes({self._ctx.block_id}, {self._ctx.thread_id}, __smem, {node.barrier_name}, {node.total_bytes});")
        self._current_expect = node

    def visit_SplitProducerWait(self, node) -> None:
        self._current_expect = None  # clear previous slot's expect scope
        self._emit(f"wait({self._ctx.block_id}, {self._ctx.thread_id}, __smem, {node.barrier_name}, {node.phase_var});")

    def visit_PhaseFlip(self, node) -> None:
        self._emit(f"{node.phase_var} ^= 1;")

    def visit_BarrierSetup(self, node) -> None:
        # Barrier shared declarations
        for slot in range(node.num_slots):
            self._emit(f"__shared__ __align__(8) uint64_t full_{slot}[1];")
        for slot in range(node.num_slots):
            self._emit(f"__shared__ __align__(8) uint64_t empty_{slot}[1];")
        # Phase variables
        for slot in range(node.num_slots):
            self._emit(f"int p{slot} = 0;")
        # Init barriers
        self._emit("if (threadIdx.x == 0) {")
        self._indent += 1
        for slot in range(node.num_slots):
            self._emit(f"init_barrier({self._ctx.block_id}, {self._ctx.thread_id}, __smem, full_{slot}, 0, 1);")
        for slot in range(node.num_slots):
            self._emit(f"init_barrier({self._ctx.block_id}, {self._ctx.thread_id}, __smem, empty_{slot}, 0, {node.num_consumers});")
        self._indent -= 1
        self._emit("}")
        self._emit("__syncthreads();")
        # Initial consumer arrive (consumers signal they're ready to be written to)
        self._emit("{")
        self._indent += 1
        self._emit(f"int __wg = ({self._ctx.thread_id} / 128);")
        self._emit("if (__wg > 0) {")
        self._indent += 1
        self._emit(f"int __lane = {self._ctx.thread_id} % 128;")
        self._emit("if (__lane == 0) {")
        self._indent += 1
        for slot in range(node.num_slots):
            self._emit(f"arrive({self._ctx.block_id}, {self._ctx.thread_id}, __smem, empty_{slot}, 1);")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        self._emit("}")
        self._emit("__syncthreads();")

    # -- split (match/case) -------------------------------------------------

    def visit_Match(self, node: ast.Match) -> None:
        """Generate code for `match split(level): case N: ...`"""
        subject = node.subject
        level_name = subject.args[0].id

        cumulative_offset = 0
        saved_ctx = self._ctx

        for case in node.cases:
            case_value = case.pattern.value.value
            new_depth = saved_ctx.depth + 1
            new_block_id = f"__block_id_{new_depth}"
            new_thread_id = f"__thread_id_{new_depth}"

            self._emit("{")
            self._indent += 1

            if level_name == "block":
                if cumulative_offset > 0:
                    self._emit(f"int {new_block_id} = {saved_ctx.block_id} - {cumulative_offset};")
                else:
                    self._emit(f"int {new_block_id} = {saved_ctx.block_id};")
                self._emit(f"int {new_thread_id} = {new_block_id} * blockDim.x + threadIdx.x;")
                self._emit(f"if ({new_block_id} >= 0 && {new_block_id} < {case_value}) {{")
            elif level_name == "thread":
                self._emit(f"int {new_block_id} = {saved_ctx.block_id};")
                if cumulative_offset > 0:
                    self._emit(f"int {new_thread_id} = {saved_ctx.thread_id} - {cumulative_offset};")
                else:
                    self._emit(f"int {new_thread_id} = {saved_ctx.thread_id};")
                self._emit(f"if ({new_thread_id} >= 0 && {new_thread_id} < {case_value}) {{")
            elif level_name == "grid":
                self._emit(f"int {new_block_id} = {saved_ctx.block_id};")
                self._emit(f"int {new_thread_id} = {saved_ctx.thread_id};")
                self._emit(f"if (true) {{")

            self._indent += 1
            self._ctx = _Context(block_id=new_block_id, thread_id=new_thread_id, depth=new_depth)

            for stmt in case.body:
                self.visit(stmt)

            self._ctx = saved_ctx
            self._indent -= 1
            self._emit("}")

            self._indent -= 1
            self._emit("}")

            cumulative_offset += case_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(func_name: str, auto_sync: bool = False) -> str:
    """Generate CUDA C code for a registered prism function.

    If auto_sync=True, run sync analysis and insert sync nodes automatically.
    Existing kernels with manual unsafe("__syncthreads();") are unaffected.
    """
    from prism.registry import registry
    from prism.type_checker import check_function

    result = registry.get(func_name)
    if result is None:
        raise ValueError(f"Function '{func_name}' not found in registry")

    func_type, ast_node, _ = result
    signature = check_function(func_type, ast_node)

    if auto_sync:
        from prism.sync_analysis import analyze_sync
        from prism.sync_transform import transform_sync
        plan = analyze_sync(func_type, ast_node, signature)
        if plan.analyzed:
            ast_node = transform_sync(ast_node, plan)

    gen = CodeGenerator(func_type, signature)
    gen.visit(ast_node)
    return gen.get_code()
