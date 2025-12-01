"""Transform pass: insert SyncOp nodes into the AST based on a SyncPlan.

Takes a type-checked AST and a SyncPlan from sync_analysis, deep-copies the AST,
and inserts concrete sync operation nodes (BlockSync, SplitWait, etc.) at the
appropriate positions. These nodes carry enough metadata for future optimization
passes (wait motion, arrive motion, etc.).
"""

from __future__ import annotations

import ast
import copy
from typing import Optional

from prism.sync_analysis import (
    SyncPlan, SyncReason, SplitBarrierSync, AccessKind,
    _AccessClassifier,
)
from prism.sync_nodes import (
    BlockSync, SplitWait, SplitArrive, SplitExpect,
    SplitProducerWait, PhaseFlip, BarrierSetup,
)


def _direction_from_reason(reason: SyncReason) -> str:
    """Derive RAW/WAR direction from a SyncReason's conflicts."""
    for _root, phase_kind, access_kind in reason.conflicts:
        if phase_kind == AccessKind.WRITE or phase_kind == AccessKind.READ_WRITE:
            if access_kind == AccessKind.READ or access_kind == AccessKind.READ_WRITE:
                return "RAW"
        if phase_kind == AccessKind.READ or phase_kind == AccessKind.READ_WRITE:
            if access_kind == AccessKind.WRITE or access_kind == AccessKind.READ_WRITE:
                return "WAR"
    return "RAW"


def _roots_from_reason(reason: SyncReason) -> list[str]:
    """Extract the list of shared roots from a SyncReason."""
    seen = set()
    roots = []
    for root, _, _ in reason.conflicts:
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return roots


def _build_id_map(original: ast.AST, copied: ast.AST) -> dict[int, ast.AST]:
    """Build a mapping from id(original_node) -> copied_node by parallel walk."""
    mapping: dict[int, ast.AST] = {}

    def walk_pair(orig, cp):
        mapping[id(orig)] = cp
        orig_children = list(ast.iter_child_nodes(orig))
        cp_children = list(ast.iter_child_nodes(cp))
        for o, c in zip(orig_children, cp_children):
            walk_pair(o, c)

    walk_pair(original, copied)
    return mapping


def _insert_block_syncs(body: list[ast.stmt], plan: SyncPlan,
                        id_map: dict[int, ast.AST]) -> list[ast.stmt]:
    """Walk a body list and insert BlockSync nodes. Returns new body list."""
    new_body: list[ast.stmt] = []

    for stmt in body:
        # Check if this stmt needs a sync_before
        # We need to find its original id — search id_map values
        orig_id = None
        for oid, mapped in id_map.items():
            if mapped is stmt:
                orig_id = oid
                break

        if orig_id is not None and orig_id in plan.sync_before:
            reason = plan.reasons.get(orig_id)
            level = plan.sync_levels.get(orig_id, "block[1]")
            if reason:
                direction = _direction_from_reason(reason)
                roots = _roots_from_reason(reason)
            else:
                direction = "RAW"
                roots = []
            new_body.append(BlockSync(roots=roots, direction=direction,
                                      level=level, position="before"))

        new_body.append(stmt)

        # Recurse into compound statements
        if isinstance(stmt, ast.For):
            stmt.body = _insert_block_syncs(stmt.body, plan, id_map)
            # Check end-of-loop sync
            if orig_id is not None and orig_id in plan.sync_end_of_loop:
                reason = plan.reasons.get(orig_id)
                level = plan.sync_levels.get(orig_id, "block[1]")
                if reason:
                    direction = _direction_from_reason(reason)
                    roots = _roots_from_reason(reason)
                else:
                    direction = "WAR"
                    roots = []
                stmt.body.append(BlockSync(roots=roots, direction=direction,
                                           level=level, position="end_of_loop"))
        elif isinstance(stmt, ast.While):
            stmt.body = _insert_block_syncs(stmt.body, plan, id_map)
            if orig_id is not None and orig_id in plan.sync_end_of_loop:
                reason = plan.reasons.get(orig_id)
                level = plan.sync_levels.get(orig_id, "block[1]")
                if reason:
                    direction = _direction_from_reason(reason)
                    roots = _roots_from_reason(reason)
                else:
                    direction = "WAR"
                    roots = []
                stmt.body.append(BlockSync(roots=roots, direction=direction,
                                           level=level, position="end_of_loop"))
        elif isinstance(stmt, ast.If):
            stmt.body = _insert_block_syncs(stmt.body, plan, id_map)
            if stmt.orelse:
                stmt.orelse = _insert_block_syncs(stmt.orelse, plan, id_map)
        elif isinstance(stmt, ast.With):
            stmt.body = _insert_block_syncs(stmt.body, plan, id_map)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = _insert_block_syncs(case.body, plan, id_map)

    return new_body


# ---------------------------------------------------------------------------
# Split-barrier slot assignment
# ---------------------------------------------------------------------------

def _assign_slots(split_barriers: list[SplitBarrierSync],
                  consumer_body: list[ast.stmt],
                  shared_roots: set[str],
                  parent_map: dict[str, str]) -> list[list[str]]:
    """Merge split barriers into slots based on consumer access groups.

    Walk the consumer branch statements. For each statement, classify which
    shared roots it accesses. Consecutive statements accessing the same set of
    roots form one access group. SplitBarrierSync entries whose roots fall in
    the same group get merged into one slot.

    Returns list of root groups (each inner list = one slot's roots).
    """
    # Collect all split-barrier roots
    sb_roots = {sb.shared_root for sb in split_barriers}

    # Walk consumer body and build ordered access groups
    groups: list[set[str]] = []

    def _classify_stmt_roots(stmt: ast.stmt) -> set[str]:
        classifier = _AccessClassifier(shared_roots, parent_map)
        classifier.visit(stmt)
        return {root for root, _kind in classifier.accesses if root in sb_roots}

    def _walk_for_groups(body: list[ast.stmt]) -> None:
        for stmt in body:
            # Recurse into compound statements to find actual accesses
            if isinstance(stmt, ast.With):
                _walk_for_groups(stmt.body)
                continue
            if isinstance(stmt, ast.For):
                _walk_for_groups(stmt.body)
                continue
            if isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    _walk_for_groups(case.body)
                continue
            if isinstance(stmt, ast.If):
                _walk_for_groups(stmt.body)
                if stmt.orelse:
                    _walk_for_groups(stmt.orelse)
                continue

            roots = _classify_stmt_roots(stmt)
            if not roots:
                continue
            # Merge into last group if same roots, else start new group
            if groups and groups[-1] == roots:
                pass  # same group
            else:
                # Check if any root in this set already appears in the last group
                # (partial overlap → merge)
                if groups and (groups[-1] & roots):
                    groups[-1] |= roots
                else:
                    groups.append(set(roots))

    _walk_for_groups(consumer_body)

    # Now assign each split barrier root to a group/slot
    # Roots not found in any group get their own slot
    slot_groups: list[list[str]] = []
    assigned: set[str] = set()

    for group in groups:
        slot_roots = sorted(group)
        if slot_roots:
            slot_groups.append(slot_roots)
            assigned.update(group)

    # Any remaining roots not in consumer groups
    for sb in split_barriers:
        if sb.shared_root not in assigned:
            slot_groups.append([sb.shared_root])
            assigned.add(sb.shared_root)

    return slot_groups


def _find_match_split_with_level(body: list[ast.stmt], target_level: str) -> Optional[ast.Match]:
    """Find the match split(thread) node whose case values sum to the target level."""
    for stmt in body:
        if isinstance(stmt, ast.Match):
            subject = stmt.subject
            if (isinstance(subject, ast.Call) and isinstance(subject.func, ast.Name)
                    and subject.func.id == "split"):
                total = sum(_extract_case_value(c.pattern) for c in stmt.cases)
                level = f"thread[{total}]"
                if level == target_level:
                    return stmt
        # Recurse
        if isinstance(stmt, ast.With):
            result = _find_match_split_with_level(stmt.body, target_level)
            if result:
                return result
        elif isinstance(stmt, ast.For):
            result = _find_match_split_with_level(stmt.body, target_level)
            if result:
                return result
        elif isinstance(stmt, ast.While):
            result = _find_match_split_with_level(stmt.body, target_level)
            if result:
                return result
        elif isinstance(stmt, ast.If):
            result = _find_match_split_with_level(stmt.body, target_level)
            if result:
                return result
            if stmt.orelse:
                result = _find_match_split_with_level(stmt.orelse, target_level)
                if result:
                    return result
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                result = _find_match_split_with_level(case.body, target_level)
                if result:
                    return result
    return None


def _extract_case_value(pattern) -> int:
    if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
        val = pattern.value.value
        return val if isinstance(val, int) else 0
    return 0


def _find_consumer_case(match_node: ast.Match, reader_branch: int) -> Optional[ast.match_case]:
    """Find the case with the given value."""
    for case in match_node.cases:
        if _extract_case_value(case.pattern) == reader_branch:
            return case
    return None


def _find_producer_branch(match_node: ast.Match, reader_branch: int) -> Optional[ast.match_case]:
    """Find the non-reader case (producer case)."""
    for case in match_node.cases:
        val = _extract_case_value(case.pattern)
        if val != reader_branch and val > 0:
            return case
    return None


def _collect_claim_aliases(body: list[ast.stmt]) -> dict[str, str]:
    """Walk nested with-claim statements and build alias→parent map."""
    aliases: dict[str, str] = {}
    for stmt in body:
        if isinstance(stmt, ast.With):
            item = stmt.items[0]
            ctx = item.context_expr
            if (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name)
                    and ctx.func.id in ("claim", "partition")
                    and item.optional_vars and isinstance(item.optional_vars, ast.Name)):
                if ctx.args and isinstance(ctx.args[0], ast.Name):
                    aliases[item.optional_vars.id] = ctx.args[0].id
            aliases.update(_collect_claim_aliases(stmt.body))
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                aliases.update(_collect_claim_aliases(case.body))
        elif isinstance(stmt, ast.For):
            aliases.update(_collect_claim_aliases(stmt.body))
        elif isinstance(stmt, ast.While):
            aliases.update(_collect_claim_aliases(stmt.body))
        elif isinstance(stmt, ast.If):
            aliases.update(_collect_claim_aliases(stmt.body))
            if stmt.orelse:
                aliases.update(_collect_claim_aliases(stmt.orelse))
    return aliases


def _resolve_alias_to_root(name: str, aliases: dict[str, str], shared_roots: set[str]) -> Optional[str]:
    """Follow alias chain to shared root."""
    visited = set()
    while name not in shared_roots:
        if name in visited:
            return None
        visited.add(name)
        if name in aliases:
            name = aliases[name]
        else:
            return None
    return name


def _find_innermost_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Drill through nested with-claim/partition statements to the innermost body."""
    for stmt in body:
        if isinstance(stmt, ast.With):
            item = stmt.items[0]
            ctx = item.context_expr
            if (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name)
                    and ctx.func.id in ("claim", "partition")):
                return _find_innermost_body(stmt.body)
    return body


def _find_load_tma_calls(body: list[ast.stmt], aliases: dict[str, str],
                          shared_roots: set[str]) -> list[tuple[str, ast.stmt]]:
    """Find load_tma calls and return (shared_root, stmt) pairs."""
    results = []
    for stmt in body:
        if isinstance(stmt, ast.With):
            results.extend(_find_load_tma_calls(stmt.body, aliases, shared_roots))
            continue
        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                results.extend(_find_load_tma_calls(case.body, aliases, shared_roots))
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "load_tma":
                # First arg is the destination (shared memory alias)
                if call.args and isinstance(call.args[0], ast.Name):
                    dst_name = call.args[0].id
                    root = _resolve_alias_to_root(dst_name, aliases, shared_roots)
                    if root:
                        results.append((root, stmt))
    return results


def _compute_slot_bytes(slot_roots: list[str], func_body: list[ast.stmt]) -> int:
    """Compute total bytes for a slot's shared memory roots.

    Finds the shared() declarations and sums element_count * element_bytes.
    """
    total = 0
    for stmt in _walk_all_stmts(func_body):
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id in slot_roots:
                annotation = stmt.annotation
                if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
                    type_node = annotation.left
                else:
                    type_node = annotation
                if (isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name)
                        and type_node.func.id == "shared"):
                    arg = type_node.args[0]  # type[size]
                    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                        type_name = arg.value.id
                        elem_bytes = _type_bytes(type_name)
                        size_val = _eval_const(arg.slice)
                        total += size_val * elem_bytes
    return total


def _walk_all_stmts(body: list[ast.stmt]):
    """Yield all statements recursively."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.For):
            yield from _walk_all_stmts(stmt.body)
        elif isinstance(stmt, ast.While):
            yield from _walk_all_stmts(stmt.body)
        elif isinstance(stmt, ast.If):
            yield from _walk_all_stmts(stmt.body)
            yield from _walk_all_stmts(stmt.orelse)
        elif isinstance(stmt, ast.With):
            yield from _walk_all_stmts(stmt.body)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                yield from _walk_all_stmts(case.body)


def _type_bytes(type_name: str) -> int:
    """Return byte size for a prism type name."""
    sizes = {
        "bf16": 2, "float": 4, "double": 8,
        "int": 4, "uint": 4, "uint32_t": 4,
        "uint64_t": 8, "bool": 1,
    }
    return sizes.get(type_name, 4)


def _eval_const(node: ast.AST) -> int:
    """Evaluate a constant expression."""
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_const(node.left)
        right = _eval_const(node.right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Div: lambda a, b: a // b,
        }
        fn = ops.get(type(node.op))
        return fn(left, right) if fn else 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_const(node.operand)
    return 0


def _find_for_loop_containing_match(body: list[ast.stmt], target_match: ast.Match) -> Optional[ast.For]:
    """Find the for-loop that contains the target match node."""
    for stmt in body:
        if isinstance(stmt, ast.For):
            if _contains_node(stmt.body, target_match):
                return stmt
            result = _find_for_loop_containing_match(stmt.body, target_match)
            if result:
                return result
        elif isinstance(stmt, ast.While):
            result = _find_for_loop_containing_match(stmt.body, target_match)
            if result:
                return result
        elif isinstance(stmt, ast.With):
            result = _find_for_loop_containing_match(stmt.body, target_match)
            if result:
                return result
        elif isinstance(stmt, ast.If):
            result = _find_for_loop_containing_match(stmt.body, target_match)
            if result:
                return result
            if stmt.orelse:
                result = _find_for_loop_containing_match(stmt.orelse, target_match)
                if result:
                    return result
        elif isinstance(stmt, ast.Match):
            if stmt is target_match:
                return None
            for case in stmt.cases:
                result = _find_for_loop_containing_match(case.body, target_match)
                if result:
                    return result
    return None


def _find_outermost_loop_containing(body: list[ast.stmt], target: ast.AST) -> ast.AST | None:
    """Find the outermost For/While loop in body that contains target."""
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.While)) and _contains_node(stmt.body, target):
            return stmt
        # Recurse into non-loop compounds to find a loop deeper
        if isinstance(stmt, ast.With):
            result = _find_outermost_loop_containing(stmt.body, target)
            if result:
                return result
        elif isinstance(stmt, ast.If):
            result = _find_outermost_loop_containing(stmt.body, target)
            if result:
                return result
            if stmt.orelse:
                result = _find_outermost_loop_containing(stmt.orelse, target)
                if result:
                    return result
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                result = _find_outermost_loop_containing(case.body, target)
                if result:
                    return result
    return None


def _contains_node(body: list[ast.stmt], target: ast.AST) -> bool:
    """Check if target node is anywhere in body."""
    for stmt in body:
        if stmt is target:
            return True
        if isinstance(stmt, ast.For) and _contains_node(stmt.body, target):
            return True
        if isinstance(stmt, ast.While) and _contains_node(stmt.body, target):
            return True
        if isinstance(stmt, ast.If):
            if _contains_node(stmt.body, target):
                return True
            if stmt.orelse and _contains_node(stmt.orelse, target):
                return True
        if isinstance(stmt, ast.With) and _contains_node(stmt.body, target):
            return True
        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                if _contains_node(case.body, target):
                    return True
    return False


# ---------------------------------------------------------------------------
# Split-barrier insertion
# ---------------------------------------------------------------------------

def _insert_split_barriers(func_body: list[ast.stmt], plan: SyncPlan,
                            shared_roots: set[str]) -> None:
    """Insert SplitWait, SplitArrive, SplitExpect, SplitProducerWait, PhaseFlip nodes.

    Modifies the AST in-place.
    """
    if not plan.split_barriers:
        return

    # All split barriers should share the same split_level
    split_level = plan.split_barriers[0].split_level
    reader_branch = plan.split_barriers[0].reader_branch
    writer_access_level = plan.split_barriers[0].writer_access_level
    reader_access_level = plan.split_barriers[0].reader_access_level

    # Step 5b: Find the match split(thread) with the right level
    # There may be multiple match split nodes — we need the one with cross-branch conflicts.
    # The consumer match is the one at split_level (e.g. thread[384]) with the reader_branch case.
    consumer_match = _find_match_split_with_level(func_body, split_level)
    if not consumer_match:
        return

    consumer_case = _find_consumer_case(consumer_match, reader_branch)
    if not consumer_case:
        return

    # Build parent_map from claim aliases in the consumer context
    # We need to walk up from the match to find claim contexts
    all_aliases = _collect_claim_aliases(func_body)
    parent_map = dict(all_aliases)

    # Step 5a: Assign slots from consumer access groups
    slot_groups = _assign_slots(plan.split_barriers, consumer_case.body,
                                shared_roots, parent_map)
    num_slots = len(slot_groups)

    # Step 5c: Insert consumer barriers — interleaved with access groups
    # For each top-level statement in the consumer case body, determine which
    # slot's roots it accesses. Insert SplitWait before the first statement
    # for each slot, and SplitArrive after the last statement for each slot.
    sb_roots = {sb.shared_root for sb in plan.split_barriers}

    def _stmt_accesses_roots(stmt: ast.stmt) -> set[str]:
        """Classify which split-barrier roots a statement (recursively) accesses."""
        classifier = _AccessClassifier(shared_roots, parent_map)
        classifier.visit(stmt)
        return {root for root, _kind in classifier.accesses if root in sb_roots}

    def _deep_classify(stmt: ast.stmt) -> set[str]:
        """Recursively classify including nested with/for/match bodies."""
        roots = _stmt_accesses_roots(stmt)
        if isinstance(stmt, ast.With):
            for s in stmt.body:
                roots |= _deep_classify(s)
        elif isinstance(stmt, ast.For):
            for s in stmt.body:
                roots |= _deep_classify(s)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                for s in case.body:
                    roots |= _deep_classify(s)
        elif isinstance(stmt, ast.If):
            for s in stmt.body:
                roots |= _deep_classify(s)
            for s in stmt.orelse:
                roots |= _deep_classify(s)
        return roots

    # Map each top-level consumer stmt to its slot (or None)
    stmt_slots: list[tuple[ast.stmt, int | None]] = []
    for stmt in consumer_case.body:
        roots = _deep_classify(stmt)
        matched_slot = None
        for slot_idx, slot_roots in enumerate(slot_groups):
            if roots & set(slot_roots):
                matched_slot = slot_idx
                break
        stmt_slots.append((stmt, matched_slot))

    # Propagate slot assignment: a stmt with no slot that immediately
    # precedes a stmt with a slot is part of the same semantic group.
    # This handles warpgroup_arrive() (no shared access) before wgmma_async
    # (shared access) — they should be treated as one unit.
    for i in range(len(stmt_slots) - 1, 0, -1):
        _, slot_cur = stmt_slots[i]
        _, slot_prev = stmt_slots[i - 1]
        if slot_cur is not None and slot_prev is None:
            stmt_slots[i - 1] = (stmt_slots[i - 1][0], slot_cur)

    # Find first and last occurrence of each slot
    first_for_slot: dict[int, int] = {}  # slot → index in stmt_slots
    last_for_slot: dict[int, int] = {}
    for i, (stmt, slot) in enumerate(stmt_slots):
        if slot is not None:
            if slot not in first_for_slot:
                first_for_slot[slot] = i
            last_for_slot[slot] = i

    # Build new body with interleaved wait/arrive nodes
    new_consumer_body: list = []
    for i, (stmt, slot) in enumerate(stmt_slots):
        # Insert SplitWait before the first statement of each slot
        for slot_idx in range(num_slots):
            if first_for_slot.get(slot_idx) == i:
                barrier_full = f"full_{slot_idx}"
                phase_var = f"p{slot_idx}"
                new_consumer_body.append(SplitWait(
                    slot=slot_idx, roots=slot_groups[slot_idx],
                    barrier_name=barrier_full, phase_var=phase_var,
                    split_level=split_level, consumer_level=reader_access_level,
                ))

        new_consumer_body.append(stmt)

        # Insert SplitArrive after the last statement of each slot
        for slot_idx in range(num_slots):
            if last_for_slot.get(slot_idx) == i:
                barrier_empty = f"empty_{slot_idx}"
                new_consumer_body.append(SplitArrive(
                    slot=slot_idx, roots=slot_groups[slot_idx],
                    barrier_name=barrier_empty,
                    split_level=split_level, consumer_level=reader_access_level,
                ))

    consumer_case.body = new_consumer_body

    # Step 5d: Insert producer barriers
    # Find the producer branch — the match case that's NOT the reader
    producer_case = _find_producer_branch(consumer_match, reader_branch)

    # The producer writes happen in a nested match inside the producer path.
    # Walk the entire for-loop body's claim context to find load_tma calls.
    # First, find the for-loop that contains this match.
    for_loop = _find_for_loop_containing_match(func_body, consumer_match)

    if for_loop:
        # Collect claim aliases from the for-loop body (producer claims)
        producer_aliases = _collect_claim_aliases(for_loop.body)
        producer_parent_map = dict(producer_aliases)

        # Find all load_tma calls in the for-loop body
        load_tma_calls = _find_load_tma_calls(for_loop.body, producer_parent_map, shared_roots)

        # Group load_tma calls by slot
        slot_loads: dict[int, list[ast.stmt]] = {i: [] for i in range(num_slots)}
        for root, stmt in load_tma_calls:
            for slot_idx, slot_roots in enumerate(slot_groups):
                if root in slot_roots:
                    slot_loads[slot_idx].append(stmt)
                    break

        # For each slot, insert SplitProducerWait + SplitExpect before the first load
        for slot_idx, slot_roots in enumerate(slot_groups):
            loads = slot_loads.get(slot_idx, [])
            if not loads:
                continue

            first_load = loads[0]
            barrier_full = f"full_{slot_idx}"
            barrier_empty = f"empty_{slot_idx}"
            phase_var = f"p{slot_idx}"

            # Compute bytes for this slot
            total_bytes = _compute_slot_bytes(slot_roots, func_body)

            # Find the body list containing the first load and insert before it
            _insert_before_stmt(for_loop.body, first_load, [
                SplitProducerWait(
                    slot=slot_idx, roots=slot_roots,
                    barrier_name=barrier_empty, phase_var=phase_var,
                    split_level=split_level, producer_level=writer_access_level,
                ),
                SplitExpect(
                    slot=slot_idx, roots=slot_roots,
                    barrier_name=barrier_full, total_bytes=total_bytes,
                    split_level=split_level, producer_level=writer_access_level,
                ),
            ])

    # Step 5e: Insert PhaseFlip at end of for-loop iteration body
    if for_loop:
        for slot_idx in range(num_slots):
            phase_var = f"p{slot_idx}"
            for_loop.body.append(PhaseFlip(slot=slot_idx, phase_var=phase_var))

    # Insert BarrierSetup node in the function body (before the for-loop)
    # We need num_consumers from the split barriers
    # num_consumers = how many consumer warp groups
    # For thread[384] split → case 128 (producer) + case 256 (consumer)
    # consumer_level = thread[256], so 256/128 = 2 consumer warp groups
    import re
    consumer_match_val = re.search(r'\[(\d+)\]', reader_access_level)
    consumer_threads = int(consumer_match_val.group(1)) if consumer_match_val else 256
    num_consumers = consumer_threads // 128
    if num_consumers < 1:
        num_consumers = 1

    setup = BarrierSetup(
        num_slots=num_slots, num_consumers=num_consumers,
        split_level=split_level,
        consumer_level=reader_access_level,
        producer_level=writer_access_level,
    )

    # Place BarrierSetup before the outermost loop containing the match.
    # Barriers persist across tile iterations, so they must be declared
    # outside the while(true) tile loop, not inside it.
    if for_loop:
        outer_loop = _find_outermost_loop_containing(func_body, for_loop)
        if outer_loop is not None:
            _insert_before_stmt(func_body, outer_loop, [setup])
        else:
            _insert_before_stmt(func_body, for_loop, [setup])


def _insert_before_stmt(body: list[ast.stmt], target: ast.stmt,
                         nodes: list) -> bool:
    """Insert nodes before target in body. Returns True if found. Recurses into compounds."""
    for i, stmt in enumerate(body):
        if stmt is target:
            for j, node in enumerate(nodes):
                body.insert(i + j, node)
            return True
        # Recurse
        if isinstance(stmt, ast.For):
            if _insert_before_stmt(stmt.body, target, nodes):
                return True
        elif isinstance(stmt, ast.While):
            if _insert_before_stmt(stmt.body, target, nodes):
                return True
        elif isinstance(stmt, ast.If):
            if _insert_before_stmt(stmt.body, target, nodes):
                return True
            if stmt.orelse and _insert_before_stmt(stmt.orelse, target, nodes):
                return True
        elif isinstance(stmt, ast.With):
            if _insert_before_stmt(stmt.body, target, nodes):
                return True
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                if _insert_before_stmt(case.body, target, nodes):
                    return True
    return False


# ---------------------------------------------------------------------------
# Collect shared roots from AST
# ---------------------------------------------------------------------------

def _collect_shared_roots_from_ast(body: list[ast.stmt]) -> set[str]:
    """Find all shared() declaration names in the AST."""
    roots: set[str] = set()
    for stmt in _walk_all_stmts(body):
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            annotation = stmt.annotation
            if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
                type_node = annotation.left
            else:
                type_node = annotation
            if (isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Name)
                    and type_node.func.id == "shared"):
                roots.add(stmt.target.id)
    return roots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_sync(ast_node: ast.FunctionDef, plan: SyncPlan) -> ast.FunctionDef:
    """Insert SyncOp nodes into a deep copy of the AST based on the SyncPlan."""
    copied = copy.deepcopy(ast_node)
    id_map = _build_id_map(ast_node, copied)

    # Step 1: BlockSync insertion
    if plan.sync_before or plan.sync_end_of_loop:
        copied.body = _insert_block_syncs(copied.body, plan, id_map)

    # Step 2: Split-barrier insertion
    if plan.split_barriers:
        shared_roots = _collect_shared_roots_from_ast(copied.body)
        _insert_split_barriers(copied.body, plan, shared_roots)

    return copied
