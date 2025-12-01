"""
Automatic __syncthreads() insertion pass.

Analyzes shared memory access patterns in prism functions and determines
where __syncthreads() barriers are needed. Uses a phase-based conflict
detection algorithm: track the current access phase per shared memory root,
and insert a sync whenever a new access conflicts with the current phase.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from prism.signature import PerspectiveSignature


class AccessKind(Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"

def _has_read(kind: AccessKind) -> bool:
    return kind in (AccessKind.READ, AccessKind.READ_WRITE)

def _has_write(kind: AccessKind) -> bool:
    return kind in (AccessKind.WRITE, AccessKind.READ_WRITE)


@dataclass
class AccessEvent:
    """A single shared memory access observed during analysis."""
    lineno: int
    source: str       # short description of the statement
    root: str         # shared memory root name
    kind: AccessKind  # READ, WRITE, or READ_WRITE
    via: str          # the name actually used (e.g. "As_t" when root is "As")


@dataclass
class SyncReason:
    """Why a sync was inserted at a particular point."""
    conflicts: list[tuple[str, AccessKind, AccessKind]]  # (root, phase_kind, access_kind)
    phase: dict[str, AccessKind]                          # phase snapshot at conflict time
    trail: list[AccessEvent]                              # accesses that built the phase
    trigger_lineno: int
    trigger_source: str
    trigger_accesses: list[tuple[str, AccessKind]]


@dataclass
class SplitBarrierSync:
    """A cross-branch shared memory conflict requiring split-barrier (mbarrier) sync."""
    shared_root: str               # which shared root is involved
    writer_branch: int             # case value of the writing branch (-1 = prior code)
    reader_branch: int             # case value of the reading branch
    split_level: str               # e.g. "thread[384]"
    writer_access_level: str       # e.g. "thread[1]" (claim perspective)
    reader_access_level: str       # e.g. "thread[256]" (claim perspective)


@dataclass
class SyncPlan:
    sync_before: set[int] = field(default_factory=set)       # id(ast_node) → emit sync before this stmt
    sync_end_of_loop: set[int] = field(default_factory=set)   # id(for/while_node) → emit sync at end of loop body
    sync_levels: dict[int, str] = field(default_factory=dict) # id(node) → sync level (e.g. "block[1]")
    reasons: dict[int, SyncReason] = field(default_factory=dict)  # id(node) → why
    split_barriers: list[SplitBarrierSync] = field(default_factory=list)
    trace: list = field(default_factory=list)                 # full access trace for --trace mode
    analyzed: bool = True                                      # False if no shared memory in function


class SyncAnalyzer:
    """AST visitor that determines __syncthreads() insertion points."""

    def __init__(self):
        self._shared_roots: set[str] = set()
        self._shared_levels: dict[str, str] = {}  # name → its own perspective level
        self._parent_map: dict[str, str] = {}      # name → direct parent in partition chain
        self._phase: dict[str, AccessKind] = {}     # current access phase per shared root
        # Track which access names (not just roots) contributed to the current phase
        self._phase_access_names: dict[str, set[str]] = {}  # root → set of via names used
        self._sync_before: set[int] = set()
        self._sync_end_of_loop: set[int] = set()
        self._sync_levels: dict[int, str] = {}     # id(node) → sync level
        self._reasons: dict[int, SyncReason] = {}
        self._trail: list[AccessEvent] = []         # accesses since last sync
        self._trace: list = []                       # full trace for debug output
        self._split_barriers: list[SplitBarrierSync] = []
        self._phase_claim_levels: dict[str, str] = {}  # root → claim perspective level
        self._active_claim_levels: dict[str, str] = {}  # root → currently active claim level
        # Cumulative tracking for split-barrier analysis (never cleared by syncs)
        self._scope_accesses: dict[str, AccessKind] = {}
        self._scope_claim_levels: dict[str, str] = {}

    def analyze(self, node: ast.FunctionDef) -> SyncPlan:
        """Entry point: analyze a function and return a SyncPlan."""
        # First pass: collect shared memory roots
        self._collect_shared_params(node)
        self._collect_shared_roots(node.body)

        if not self._shared_roots:
            return SyncPlan(analyzed=False)

        # Second pass: walk the body and determine sync points
        self._walk_body(node.body)

        # Post-processing: remove __syncthreads covered by split-barriers.
        # Only remove when BOTH root AND direction match — a RAW split-barrier
        # (producer writes → consumer reads) does NOT cover a WAR conflict
        # (consumer reads → producer writes next iteration = "empty" barrier).
        if self._split_barriers:
            # Build (root, direction) coverage from split-barriers
            split_coverage = set()
            for sb in self._split_barriers:
                # Split barriers cover both directions:
                # full barrier = RAW (producer writes → consumer reads)
                # empty barrier = WAR (consumer reads → producer writes)
                split_coverage.add((sb.shared_root, 'RAW'))
                split_coverage.add((sb.shared_root, 'WAR'))
            to_remove = set()
            for node_id, reason in self._reasons.items():
                all_covered = True
                for root, phase_kind, access_kind in reason.conflicts:
                    if _has_write(phase_kind) and _has_read(access_kind):
                        direction = 'RAW'
                    elif _has_read(phase_kind) and _has_write(access_kind):
                        direction = 'WAR'
                    else:
                        direction = None
                    if (root, direction) not in split_coverage:
                        all_covered = False
                        break
                if all_covered:
                    to_remove.add(node_id)
            for node_id in to_remove:
                self._sync_before.discard(node_id)
                self._sync_end_of_loop.discard(node_id)

        return SyncPlan(
            sync_before=self._sync_before,
            sync_end_of_loop=self._sync_end_of_loop,
            sync_levels=self._sync_levels,
            reasons=self._reasons,
            split_barriers=self._split_barriers,
            trace=self._trace,
            analyzed=True,
        )

    def _collect_shared_params(self, node: ast.FunctionDef) -> None:
        """Check function parameters for shared() type annotations (e.g. ptr(shared(uint32_t)))."""
        for arg in node.args.args:
            if arg.arg == "template":
                break
            if arg.annotation and self._annotation_contains_shared(arg.annotation):
                self._shared_roots.add(arg.arg)
                self._shared_levels[arg.arg] = self._extract_perspective(arg.annotation)

    @staticmethod
    def _annotation_contains_shared(node: ast.AST) -> bool:
        """Check if an annotation contains shared() anywhere in the type nesting."""
        # Strip perspective: type @ level
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            node = node.left
        return SyncAnalyzer._type_contains_shared(node)

    @staticmethod
    def _extract_perspective(annotation: ast.AST) -> str:
        """Extract the perspective string (e.g. 'block[1]') from a type @ perspective annotation."""
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
            persp = annotation.right
            if isinstance(persp, ast.Subscript) and isinstance(persp.value, ast.Name):
                level = persp.value.id
                index = persp.slice.value if isinstance(persp.slice, ast.Constant) else "?"
                return f"{level}[{index}]"
        return "block[1]"  # fallback

    @staticmethod
    def _extract_p_kwarg(context_expr: ast.Call) -> Optional[str]:
        """Extract the p=level[N] perspective from a partition/claim call."""
        for kw in context_expr.keywords:
            if kw.arg == "p":
                persp = kw.value
                if isinstance(persp, ast.Subscript) and isinstance(persp.value, ast.Name):
                    level = persp.value.id
                    index = persp.slice.value if isinstance(persp.slice, ast.Constant) else "?"
                    return f"{level}[{index}]"
        return None

    def _parent_level_for_name(self, name: str) -> str:
        """Get the sync level for conflicts through `name`.

        The sync level is the PARENT's level in the partition chain.
        If name is a root (no parent), use its own declaration level.
        """
        if name in self._parent_map:
            parent = self._parent_map[name]
            return self._shared_levels.get(parent, "block[1]")
        # Root or unknown — use own level
        return self._shared_levels.get(name, "block[1]")

    def _sync_level_for_conflicts(self, conflicts: list[tuple[str, AccessKind, AccessKind]]) -> str:
        """Determine the sync level from the conflicting shared roots.

        Uses the access names tracked in _phase_access_names to find
        which partition nodes were used, then derives the level from
        the partition hierarchy (sync at parent's level).
        """
        levels = set()
        for root, _, _ in conflicts:
            # Check which names were used to access this root
            access_names = self._phase_access_names.get(root, set())
            for name in access_names:
                levels.add(self._parent_level_for_name(name))
            if not access_names:
                # Fallback: use root's own level
                levels.add(self._shared_levels.get(root, "block[1]"))
        if not levels:
            return "block[1]"
        # Return the broadest level (block > thread)
        return self._broadest_level(levels)

    @staticmethod
    def _broadest_level(levels: set[str]) -> str:
        """Return the broadest sync level. block > thread > grid."""
        # Simple ordering: block is broader than thread
        for level in sorted(levels):
            if level.startswith("block"):
                return level
        for level in sorted(levels):
            if level.startswith("thread"):
                return level
        return sorted(levels)[0] if levels else "block[1]"

    @staticmethod
    def _type_contains_shared(node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "shared":
                return True
            for arg in node.args:
                if SyncAnalyzer._type_contains_shared(arg):
                    return True
        return False

    def _collect_shared_roots(self, body: list[ast.stmt]) -> None:
        """Recursively find all shared(...) declarations in the function."""
        for stmt in body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                annotation = stmt.annotation
                if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
                    type_node = annotation.left
                else:
                    type_node = annotation
                if (isinstance(type_node, ast.Call)
                        and isinstance(type_node.func, ast.Name)
                        and type_node.func.id == "shared"):
                    self._shared_roots.add(stmt.target.id)
                    self._shared_levels[stmt.target.id] = self._extract_perspective(stmt.annotation)

            # Recurse into nested structures
            if isinstance(stmt, ast.For):
                self._collect_shared_roots(stmt.body)
            elif isinstance(stmt, ast.While):
                self._collect_shared_roots(stmt.body)
            elif isinstance(stmt, ast.If):
                self._collect_shared_roots(stmt.body)
                self._collect_shared_roots(stmt.orelse)
            elif isinstance(stmt, ast.With):
                self._collect_shared_roots(stmt.body)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    self._collect_shared_roots(case.body)

    def _maybe_sync_before_compound(self, stmt: ast.stmt) -> None:
        """Check if entering a compound statement would conflict with the current phase.

        If so, insert sync_before on the compound statement itself (not inside it).
        """
        first = self._collect_first_accesses([stmt])
        if first and self._check_conflict(first):
            lineno = getattr(stmt, 'lineno', 0)
            conflicts = self._find_conflicts(first)
            self._reasons[id(stmt)] = SyncReason(
                conflicts=conflicts,
                phase=dict(self._phase),
                trail=list(self._trail),
                trigger_lineno=lineno,
                trigger_source=self._stmt_summary(stmt),
                trigger_accesses=first,
            )
            self._sync_before.add(id(stmt))
            self._sync_levels[id(stmt)] = self._sync_level_for_conflicts(conflicts)
            self._trace.append(('sync', lineno, 'SYNC BEFORE', None, None, None, None))
            self._phase.clear()
            self._phase_access_names.clear()
            self._phase_claim_levels.clear()
            self._trail.clear()

    def _walk_body(self, body: list[ast.stmt]) -> None:
        """Walk a list of statements, inserting syncs as needed."""
        for stmt in body:
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                self._maybe_sync_before_compound(stmt)
            elif isinstance(stmt, ast.With):
                # Skip sync_before for claim() — cross-branch conflicts
                # are handled by split-barrier analysis inside match split
                if not self._is_claim_with(stmt):
                    self._maybe_sync_before_compound(stmt)

            if isinstance(stmt, ast.For):
                self._handle_for(stmt)
                continue

            if isinstance(stmt, ast.While):
                self._handle_while(stmt)
                continue

            if isinstance(stmt, ast.If):
                self._handle_if(stmt)
                continue

            if isinstance(stmt, ast.With):
                self._handle_with(stmt)
                continue

            if isinstance(stmt, ast.Match):
                # For non-split matches, check sync_before; split matches handle their own
                subject = stmt.subject
                is_split = (isinstance(subject, ast.Call) and isinstance(subject.func, ast.Name)
                            and subject.func.id == "split")
                if not is_split:
                    self._maybe_sync_before_compound(stmt)
                self._handle_match(stmt)
                continue

            # Regular statement: classify accesses and check for conflicts
            accesses, detailed = self._classify_accesses_detailed(stmt)
            lineno = getattr(stmt, 'lineno', 0)
            summary = self._stmt_summary(stmt)

            if accesses:
                # Record trace
                for root, kind, via in detailed:
                    self._trace.append(('access', lineno, summary, root, kind, via, dict(self._phase)))

                if self._check_conflict(accesses):
                    # Record why this sync is needed
                    conflicts = self._find_conflicts(accesses)
                    self._reasons[id(stmt)] = SyncReason(
                        conflicts=conflicts,
                        phase=dict(self._phase),
                        trail=list(self._trail),
                        trigger_lineno=lineno,
                        trigger_source=summary,
                        trigger_accesses=accesses,
                    )
                    self._sync_before.add(id(stmt))
                    self._sync_levels[id(stmt)] = self._sync_level_for_conflicts(conflicts)
                    self._trace.append(('sync', lineno, 'SYNC BEFORE', None, None, None, None))
                    self._phase.clear()
                    self._phase_access_names.clear()
                    self._phase_claim_levels.clear()
                    self._trail.clear()

                self._update_phase(accesses, detailed)
                # Record trail events
                for root, kind, via in detailed:
                    self._trail.append(AccessEvent(lineno, summary, root, kind, via))

    def _handle_for(self, node: ast.For) -> None:
        """Handle for-loop: walk body, then check back-edge conflict."""
        self._walk_body(node.body)

        # Back-edge check: does the end-of-body phase conflict with
        # what the start of the body would access?
        start_accesses = self._collect_first_accesses(node.body)

        if start_accesses and self._check_conflict_against(start_accesses, self._phase):
            lineno = getattr(node, 'lineno', 0)
            conflicts = self._find_conflicts(start_accesses)
            self._reasons[id(node)] = SyncReason(
                conflicts=conflicts,
                phase=dict(self._phase),
                trail=list(self._trail),
                trigger_lineno=lineno,
                trigger_source=f"back-edge → {self._stmt_summary(node)}",
                trigger_accesses=start_accesses,
            )
            self._sync_end_of_loop.add(id(node))
            self._sync_levels[id(node)] = self._sync_level_for_conflicts(conflicts)
            self._trace.append(('sync', lineno, 'SYNC END-OF-LOOP', None, None, None, None))
            self._phase.clear()
            self._phase_access_names.clear()
            self._phase_claim_levels.clear()
            self._trail.clear()

    def _handle_while(self, node: ast.While) -> None:
        """Handle while-loop: same as for-loop."""
        self._walk_body(node.body)

        start_accesses = self._collect_first_accesses(node.body)

        if start_accesses and self._check_conflict_against(start_accesses, self._phase):
            lineno = getattr(node, 'lineno', 0)
            conflicts = self._find_conflicts(start_accesses)
            self._reasons[id(node)] = SyncReason(
                conflicts=conflicts,
                phase=dict(self._phase),
                trail=list(self._trail),
                trigger_lineno=lineno,
                trigger_source=f"back-edge → {self._stmt_summary(node)}",
                trigger_accesses=start_accesses,
            )
            self._sync_end_of_loop.add(id(node))
            self._sync_levels[id(node)] = self._sync_level_for_conflicts(conflicts)
            self._trace.append(('sync', lineno, 'SYNC END-OF-LOOP', None, None, None, None))
            self._phase.clear()
            self._phase_access_names.clear()
            self._phase_claim_levels.clear()
            self._trail.clear()

    def _handle_if(self, node: ast.If) -> None:
        """Handle if/else: analyze both branches, merge phases conservatively."""
        saved_phase = dict(self._phase)

        # Analyze then-branch
        self._phase = dict(saved_phase)
        self._walk_body(node.body)
        then_phase = dict(self._phase)

        # Analyze else-branch
        self._phase = dict(saved_phase)
        if node.orelse:
            self._walk_body(node.orelse)
        else_phase = dict(self._phase)

        # Merge: union of both branches (conservative)
        merged = dict(then_phase)
        for name, kind in else_phase.items():
            if name in merged:
                if merged[name] != kind:
                    merged[name] = AccessKind.READ_WRITE
            else:
                merged[name] = kind
        self._phase = merged

    def _handle_with(self, node: ast.With) -> None:
        """Handle with-statements: detect partition/claim (register alias), detect group (recurse)."""
        item = node.items[0]
        context_expr = item.context_expr

        if not isinstance(context_expr, ast.Call) or not isinstance(context_expr.func, ast.Name):
            self._walk_body(node.body)
            return

        func_name = context_expr.func.id

        if func_name == "group":
            # Recurse into group body
            self._walk_body(node.body)
        elif func_name in ("partition", "claim"):
            # Register child in the data flow graph
            ptr_arg = context_expr.args[0]
            registered_root = None
            old_claim_level = None
            if isinstance(ptr_arg, ast.Name) and item.optional_vars and isinstance(item.optional_vars, ast.Name):
                alias = item.optional_vars.id
                parent_name = ptr_arg.id
                root = self._resolve_to_root(parent_name)
                if root is not None:
                    self._parent_map[alias] = parent_name
                    p_level = self._extract_p_kwarg(context_expr)
                    if p_level:
                        self._shared_levels[alias] = p_level
                        # Track claim level for split-barrier analysis
                        registered_root = root
                        old_claim_level = self._active_claim_levels.get(root)
                        self._active_claim_levels[root] = p_level
            # Recurse into partition body
            self._walk_body(node.body)
            # Clean up (scoped to the with block)
            if isinstance(item.optional_vars, ast.Name):
                self._parent_map.pop(item.optional_vars.id, None)
                self._shared_levels.pop(item.optional_vars.id, None)
            if registered_root is not None:
                if old_claim_level is not None:
                    self._active_claim_levels[registered_root] = old_claim_level
                else:
                    self._active_claim_levels.pop(registered_root, None)
        elif func_name == "wgmma_async":
            self._walk_body(node.body)
        else:
            self._walk_body(node.body)

    def _handle_match(self, node: ast.Match) -> None:
        """Handle match statement: detect cross-branch shared memory conflicts for split patterns."""
        is_split, split_level = self._is_match_split(node)

        if not is_split:
            self._handle_match_generic(node)
            return

        # match split(thread) — split-barrier analysis
        saved_phase = dict(self._phase)
        saved_access_names = {k: set(v) for k, v in self._phase_access_names.items()}
        saved_trail = list(self._trail)
        saved_claim_levels = dict(self._phase_claim_levels)
        # Save cumulative scope accesses (for incoming-phase check, before branches pollute it)
        saved_scope_accesses = dict(self._scope_accesses)
        saved_scope_claims = dict(self._scope_claim_levels)

        # (case_val, end_phase, phase_claims, branch_scope, branch_scope_claims)
        branch_info = []

        for case in node.cases:
            case_val = self._extract_case_value(case.pattern)

            # Start each split branch with empty phase
            # (different thread groups don't conflict via __syncthreads)
            self._phase = {}
            self._phase_access_names = {}
            self._phase_claim_levels = {}
            self._trail = []
            # Reset scope accesses for this branch (branch-local accumulation)
            self._scope_accesses = dict(saved_scope_accesses)
            self._scope_claim_levels = dict(saved_scope_claims)

            self._walk_body(case.body)

            branch_info.append((
                case_val, dict(self._phase), dict(self._phase_claim_levels),
                dict(self._scope_accesses), dict(self._scope_claim_levels),
            ))

        # Cross-branch conflict detection (between branches of THIS match)
        for i in range(len(branch_info)):
            for j in range(i + 1, len(branch_info)):
                case_i, phase_i, claims_i, _, _ = branch_info[i]
                case_j, phase_j, claims_j, _, _ = branch_info[j]

                for root in set(phase_i) | set(phase_j):
                    kind_i = phase_i.get(root)
                    kind_j = phase_j.get(root)
                    if kind_i and kind_j:
                        if _has_write(kind_i) and _has_read(kind_j):
                            self._split_barriers.append(SplitBarrierSync(
                                shared_root=root, writer_branch=case_i, reader_branch=case_j,
                                split_level=split_level,
                                writer_access_level=claims_i.get(root, "?"),
                                reader_access_level=claims_j.get(root, "?"),
                            ))
                        elif _has_read(kind_i) and _has_write(kind_j):
                            self._split_barriers.append(SplitBarrierSync(
                                shared_root=root, writer_branch=case_j, reader_branch=case_i,
                                split_level=split_level,
                                writer_access_level=claims_j.get(root, "?"),
                                reader_access_level=claims_i.get(root, "?"),
                            ))

        # Incoming phase vs branch conflicts (cross-match-statement detection)
        # Use scope-level cumulative accesses (not phase, which is cleared by syncs)
        scope_acc = saved_scope_accesses
        scope_claims = saved_scope_claims
        for case_val, branch_phase, branch_claims, _, _ in branch_info:
            for root, kind in branch_phase.items():
                if root in scope_acc:
                    existing = scope_acc[root]
                    if _has_write(existing) and _has_read(kind):
                        self._split_barriers.append(SplitBarrierSync(
                            shared_root=root, writer_branch=-1, reader_branch=case_val,
                            split_level=split_level,
                            writer_access_level=scope_claims.get(root, "?"),
                            reader_access_level=branch_claims.get(root, "?"),
                        ))
                    elif _has_read(existing) and _has_write(kind):
                        self._split_barriers.append(SplitBarrierSync(
                            shared_root=root, writer_branch=case_val, reader_branch=-1,
                            split_level=split_level,
                            writer_access_level=scope_claims.get(root, "?"),
                            reader_access_level=branch_claims.get(root, "?"),
                        ))

        # Merge phases: saved + all branches
        merged = dict(saved_phase)
        merged_claims = dict(saved_claim_levels)
        for _, end_phase, end_claims, _, _ in branch_info:
            for name, kind in end_phase.items():
                if name in merged:
                    if merged[name] != kind:
                        merged[name] = AccessKind.READ_WRITE
                else:
                    merged[name] = kind
            merged_claims.update(end_claims)

        self._phase = merged
        self._phase_access_names = saved_access_names
        self._phase_claim_levels = merged_claims
        self._trail = saved_trail
        # Restore scope accesses using branch scope (survives syncs, unlike phase)
        self._scope_accesses = dict(saved_scope_accesses)
        self._scope_claim_levels = dict(saved_scope_claims)
        for _, _, _, branch_scope, branch_scope_claims in branch_info:
            for name, kind in branch_scope.items():
                if name in self._scope_accesses:
                    if self._scope_accesses[name] != kind:
                        self._scope_accesses[name] = AccessKind.READ_WRITE
                else:
                    self._scope_accesses[name] = kind
            self._scope_claim_levels.update(branch_scope_claims)

    def _handle_match_generic(self, node: ast.Match) -> None:
        """Handle non-split match: treat like cascading if/else."""
        saved_phase = dict(self._phase)
        branch_phases = []

        for case in node.cases:
            self._phase = dict(saved_phase)
            self._walk_body(case.body)
            branch_phases.append(dict(self._phase))

        # Merge all branches
        merged = dict(branch_phases[0]) if branch_phases else dict(saved_phase)
        for bp in branch_phases[1:]:
            for name, kind in bp.items():
                if name in merged:
                    if merged[name] != kind:
                        merged[name] = AccessKind.READ_WRITE
                else:
                    merged[name] = kind
        self._phase = merged

    @staticmethod
    def _is_match_split(node: ast.Match) -> tuple[bool, str]:
        """Check if match subject is split(...) and compute split level from case values."""
        subject = node.subject
        if isinstance(subject, ast.Call) and isinstance(subject.func, ast.Name) and subject.func.id == "split":
            total = sum(SyncAnalyzer._extract_case_value(c.pattern) for c in node.cases)
            return (True, f"thread[{total}]") if total > 0 else (True, "thread[?]")
        return False, ""

    @staticmethod
    def _extract_case_value(pattern) -> int:
        """Extract the integer value from a match case pattern."""
        if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
            val = pattern.value.value
            return val if isinstance(val, int) else 0
        return 0

    @staticmethod
    def _is_claim_with(stmt: ast.With) -> bool:
        """Check if a with-statement is a claim() context manager."""
        item = stmt.items[0]
        ctx = item.context_expr
        return (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name)
                and ctx.func.id == "claim")

    @staticmethod
    def _stmt_summary(stmt: ast.stmt) -> str:
        """One-line summary of a statement."""
        try:
            return ast.unparse(stmt).split('\n')[0]
        except Exception:
            return type(stmt).__name__

    def _classify_accesses(self, stmt: ast.stmt) -> list[tuple[str, AccessKind]]:
        """Determine which shared roots are read/written by a statement."""
        classifier = _AccessClassifier(self._shared_roots, self._parent_map)
        classifier.visit(stmt)
        return classifier.accesses

    def _classify_accesses_detailed(self, stmt: ast.stmt) -> tuple[list[tuple[str, AccessKind]], list[tuple[str, AccessKind, str]]]:
        """Like _classify_accesses but also returns (root, kind, via_name) triples."""
        classifier = _AccessClassifier(self._shared_roots, self._parent_map)
        classifier.visit(stmt)
        return classifier.accesses, classifier.detailed

    def _check_conflict(self, accesses: list[tuple[str, AccessKind]]) -> bool:
        """Check if any access conflicts with the current phase."""
        return self._check_conflict_against(accesses, self._phase)

    def _find_conflicts(self, accesses: list[tuple[str, AccessKind]], phase: dict[str, AccessKind] | None = None) -> list[tuple[str, AccessKind, AccessKind]]:
        """Return list of (root, phase_kind, access_kind) for each conflicting root."""
        if phase is None:
            phase = self._phase
        result = []
        for root, kind in accesses:
            if root in phase:
                existing = phase[root]
                if (_has_write(existing) and _has_read(kind)) or \
                   (_has_read(existing) and _has_write(kind)):
                    result.append((root, existing, kind))
        return result

    def _check_conflict_against(self, accesses: list[tuple[str, AccessKind]], phase: dict[str, AccessKind]) -> bool:
        """Check if accesses conflict with a given phase.

        Conflict occurs when the combined phase+access spans both READ and WRITE
        on the same root (RAW or WAR hazard). Pure WRITE→WRITE is not a conflict.
        """
        for root, kind in accesses:
            if root in phase:
                existing = phase[root]
                if (_has_write(existing) and _has_read(kind)) or \
                   (_has_read(existing) and _has_write(kind)):
                    return True
        return False

    def _update_phase(self, accesses: list[tuple[str, AccessKind]],
                      detailed: list[tuple[str, AccessKind, str]] | None = None) -> None:
        """Update the current phase with new accesses."""
        for root, kind in accesses:
            if root in self._phase:
                existing = self._phase[root]
                if existing != kind:
                    self._phase[root] = AccessKind.READ_WRITE
            else:
                self._phase[root] = kind
            # Track claim level for split-barrier analysis
            if root in self._active_claim_levels:
                self._phase_claim_levels[root] = self._active_claim_levels[root]
                self._scope_claim_levels[root] = self._active_claim_levels[root]
            # Cumulative scope tracking (never cleared by syncs)
            if root in self._scope_accesses:
                existing = self._scope_accesses[root]
                if existing != kind:
                    self._scope_accesses[root] = AccessKind.READ_WRITE
            else:
                self._scope_accesses[root] = kind
        # Track which access names contributed to this phase
        if detailed:
            for root, kind, via in detailed:
                if root not in self._phase_access_names:
                    self._phase_access_names[root] = set()
                self._phase_access_names[root].add(via)

    def _collect_first_accesses(self, body: list[ast.stmt]) -> list[tuple[str, AccessKind]]:
        """Collect shared memory accesses from the first effective statement(s) in a body.

        Recurses into all compound statements (With, For, While, If) to find
        the first actual shared memory access, regardless of nesting depth.
        """
        for stmt in body:
            if isinstance(stmt, ast.With):
                result = self._collect_first_accesses_from_with(stmt)
                if result:
                    return result
                continue

            if isinstance(stmt, (ast.For, ast.While)):
                result = self._collect_first_accesses(stmt.body)
                if result:
                    return result
                continue

            if isinstance(stmt, ast.If):
                result = self._collect_first_accesses(stmt.body)
                if result:
                    return result
                if stmt.orelse:
                    result = self._collect_first_accesses(stmt.orelse)
                    if result:
                        return result
                continue

            if isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    result = self._collect_first_accesses(case.body)
                    if result:
                        return result
                continue

            accesses = self._classify_accesses(stmt)
            if accesses:
                return accesses
        return []

    def _collect_first_accesses_from_with(self, node: ast.With) -> list[tuple[str, AccessKind]]:
        """Collect first accesses from a with-statement body, temporarily registering parent mappings."""
        item = node.items[0]
        context_expr = item.context_expr

        saved_parents = dict(self._parent_map)

        if isinstance(context_expr, ast.Call) and isinstance(context_expr.func, ast.Name):
            func_name = context_expr.func.id
            if func_name in ("partition", "claim"):
                ptr_arg = context_expr.args[0]
                if isinstance(ptr_arg, ast.Name) and item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    alias = item.optional_vars.id
                    root = self._resolve_to_root(ptr_arg.id)
                    if root is not None:
                        self._parent_map[alias] = ptr_arg.id

        result = self._collect_first_accesses(node.body)
        self._parent_map = saved_parents
        return result

    def _resolve_to_root(self, name: str) -> Optional[str]:
        """Follow the parent chain to find the shared root, or return None if not shared."""
        if name in self._shared_roots:
            return name
        if name in self._parent_map:
            return self._resolve_to_root(self._parent_map[name])
        return None


class _AccessClassifier(ast.NodeVisitor):
    """Helper visitor that classifies shared memory accesses in a single statement."""

    def __init__(self, shared_roots: set[str], parent_map: dict[str, str]):
        self._shared_roots = shared_roots
        self._parent_map = parent_map
        self.accesses: list[tuple[str, AccessKind]] = []
        self.detailed: list[tuple[str, AccessKind, str]] = []  # (root, kind, via_name)
        self._seen: set[tuple[str, AccessKind]] = set()

    def _resolve(self, name: str) -> Optional[str]:
        """Follow the parent chain to find the shared root."""
        if name in self._shared_roots:
            return name
        if name in self._parent_map:
            return self._resolve(self._parent_map[name])
        return None

    def _add(self, root: str, kind: AccessKind, via: str | None = None) -> None:
        key = (root, kind)
        if key not in self._seen:
            self._seen.add(key)
            self.accesses.append(key)
            self.detailed.append((root, kind, via or root))

    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0]
        # LHS subscript → WRITE
        if isinstance(target, ast.Subscript):
            base_name = self._subscript_base(target)
            if base_name:
                root = self._resolve(base_name)
                if root:
                    self._add(root, AccessKind.WRITE, base_name)
                elif base_name == "float4_cast":
                    # float4_cast[X] = ... → actual write target is X
                    self._scan_writes_in_expr(target.slice)
        # RHS: scan for shared reads
        self._scan_reads(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # Target → READ + WRITE
        if isinstance(node.target, ast.Subscript):
            base_name = self._subscript_base(node.target)
            if base_name:
                root = self._resolve(base_name)
                if root:
                    self._add(root, AccessKind.READ, base_name)
                    self._add(root, AccessKind.WRITE, base_name)
        # Value → READ
        self._scan_reads(node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Function call at statement level (not unsafe, which is filtered earlier)
        if isinstance(node.value, ast.Call):
            self._handle_call(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Annotated assignment: check if RHS reads from shared
        if node.value is not None:
            self._scan_reads(node.value)

    def _handle_call(self, call: ast.Call) -> None:
        """Handle a function call — use callee's const annotations to determine access kind.

        If a parameter is ptr(const(...)), the callee only reads → READ.
        If a parameter is ptr(...) without const, the callee may read and write → READ+WRITE.
        If the callee is unknown, fall back to conservative READ+WRITE.
        """
        if not isinstance(call.func, ast.Name):
            # Attribute call (e.g. intrinsic.wgmma256): scan args as reads conservatively
            if isinstance(call.func, ast.Attribute):
                for arg in call.args:
                    self._scan_reads(arg)
            return
        func_name = call.func.id

        param_const = self._get_param_const_info(func_name)

        for i, arg in enumerate(call.args):
            names = self._extract_names(arg)
            for name in names:
                root = self._resolve(name)
                if root:
                    via = f"{name} (via {func_name}())"
                    if param_const is not None and i < len(param_const):
                        if param_const[i]:
                            self._add(root, AccessKind.READ, via)
                        else:
                            self._add(root, AccessKind.READ, via)
                            self._add(root, AccessKind.WRITE, via)
                    else:
                        # Unknown function — conservative
                        self._add(root, AccessKind.READ, via)
                        self._add(root, AccessKind.WRITE, via)

    @staticmethod
    def _is_const_ptr_param(annotation: ast.AST | None) -> bool:
        """Check if a parameter annotation is ptr(const(...))."""
        if annotation is None:
            return False
        # Strip perspective: type @ level
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
            type_node = annotation.left
        else:
            type_node = annotation
        if isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name):
            if type_node.func.id == "ptr" and type_node.args:
                inner = type_node.args[0]
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    return inner.func.id == "const"
        return False

    @classmethod
    def _get_param_const_info(cls, func_name: str) -> list[bool] | None:
        """Look up a prism function and return per-parameter const info."""
        try:
            from prism.registry import registry
            result = registry.get(func_name)
            if result is None:
                return None
            _, callee_ast, _ = result
            info = []
            for param in callee_ast.args.args:
                if param.arg == "template":
                    break
                info.append(cls._is_const_ptr_param(param.annotation))
            return info
        except Exception:
            return None

    def _scan_writes_in_expr(self, node: ast.AST) -> None:
        """Scan an expression for shared memory write targets (e.g. inside float4_cast[X])."""
        if isinstance(node, ast.Subscript):
            base_name = self._subscript_base(node)
            if base_name:
                root = self._resolve(base_name)
                if root:
                    self._add(root, AccessKind.WRITE, base_name)

    def _scan_reads(self, node: ast.AST) -> None:
        """Scan an expression tree for shared memory reads."""
        if isinstance(node, ast.Subscript):
            base_name = self._subscript_base(node)
            if base_name:
                root = self._resolve(base_name)
                if root:
                    self._add(root, AccessKind.READ, base_name)
            # Also scan index for reads
            self._scan_reads(node.slice)
            return

        if isinstance(node, ast.Name):
            # A bare name reference to shared memory (e.g., passing to a function)
            root = self._resolve(node.id)
            if root:
                self._add(root, AccessKind.READ, node.id)
            return

        if isinstance(node, ast.BinOp):
            self._scan_reads(node.left)
            self._scan_reads(node.right)
            return

        if isinstance(node, ast.UnaryOp):
            self._scan_reads(node.operand)
            return

        if isinstance(node, ast.Call):
            self._handle_call(node)
            return

        if isinstance(node, ast.IfExp):
            self._scan_reads(node.test)
            self._scan_reads(node.body)
            self._scan_reads(node.orelse)
            return

        if isinstance(node, ast.Compare):
            self._scan_reads(node.left)
            for comparator in node.comparators:
                self._scan_reads(comparator)
            return

        if isinstance(node, ast.BoolOp):
            for v in node.values:
                self._scan_reads(v)
            return

    def _subscript_base(self, node: ast.Subscript) -> Optional[str]:
        """Get the base name of a subscript expression (e.g., As[i] → 'As')."""
        if isinstance(node.value, ast.Name):
            return node.value.id
        # Nested subscript: As[i][j] → 'As'
        if isinstance(node.value, ast.Subscript):
            return self._subscript_base(node.value)
        return None

    def _extract_names(self, node: ast.AST) -> list[str]:
        """Extract all Name references from an AST node."""
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        for child in ast.iter_child_nodes(node):
            names.extend(self._extract_names(child))
        return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sync(func_type: str, ast_node: ast.FunctionDef,
                 signature: PerspectiveSignature) -> SyncPlan:
    """Analyze a prism function and return a SyncPlan for __syncthreads() insertion."""
    analyzer = SyncAnalyzer()
    return analyzer.analyze(ast_node)
