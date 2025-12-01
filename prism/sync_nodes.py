"""Sync AST node types inserted by the sync transform pass.

All inherit from ast.AST so they sit in body lists and work with ast.NodeVisitor.
Each carries optimization-relevant metadata (roots, levels, directions) so future
passes can move them closer to actual conflicting accesses.
"""

from __future__ import annotations

import ast


class SyncOp(ast.AST):
    """Base class for all sync operation nodes."""
    _fields = ()


class BlockSync(SyncOp):
    """__syncthreads() — same thread group R/W conflict."""
    _fields = ('roots', 'direction', 'level', 'position')

    def __init__(self, roots: list[str], direction: str, level: str, position: str):
        super().__init__()
        self.roots = roots
        self.direction = direction  # "RAW" or "WAR"
        self.level = level          # e.g. "block[1]"
        self.position = position    # "before" or "end_of_loop"


class SplitWait(SyncOp):
    """Consumer waits for producer to finish writing (wait on full barrier)."""
    _fields = ('slot', 'roots', 'barrier_name', 'phase_var', 'split_level', 'consumer_level')

    def __init__(self, slot: int, roots: list[str], barrier_name: str,
                 phase_var: str, split_level: str, consumer_level: str):
        super().__init__()
        self.slot = slot
        self.roots = roots
        self.barrier_name = barrier_name
        self.phase_var = phase_var
        self.split_level = split_level
        self.consumer_level = consumer_level


class SplitArrive(SyncOp):
    """Consumer signals it's done reading (arrive on empty barrier)."""
    _fields = ('slot', 'roots', 'barrier_name', 'split_level', 'consumer_level')

    def __init__(self, slot: int, roots: list[str], barrier_name: str,
                 split_level: str, consumer_level: str):
        super().__init__()
        self.slot = slot
        self.roots = roots
        self.barrier_name = barrier_name
        self.split_level = split_level
        self.consumer_level = consumer_level


class SplitExpect(SyncOp):
    """Producer sets expected bytes on full barrier before TMA load."""
    _fields = ('slot', 'roots', 'barrier_name', 'total_bytes', 'split_level', 'producer_level')

    def __init__(self, slot: int, roots: list[str], barrier_name: str,
                 total_bytes: int | None, split_level: str, producer_level: str):
        super().__init__()
        self.slot = slot
        self.roots = roots
        self.barrier_name = barrier_name
        self.total_bytes = total_bytes
        self.split_level = split_level
        self.producer_level = producer_level


class SplitProducerWait(SyncOp):
    """Producer waits for consumer to finish reading (wait on empty barrier)."""
    _fields = ('slot', 'roots', 'barrier_name', 'phase_var', 'split_level', 'producer_level')

    def __init__(self, slot: int, roots: list[str], barrier_name: str,
                 phase_var: str, split_level: str, producer_level: str):
        super().__init__()
        self.slot = slot
        self.roots = roots
        self.barrier_name = barrier_name
        self.phase_var = phase_var
        self.split_level = split_level
        self.producer_level = producer_level


class PhaseFlip(SyncOp):
    """Flip phase variable for double-buffering."""
    _fields = ('slot', 'phase_var')

    def __init__(self, slot: int, phase_var: str):
        super().__init__()
        self.slot = slot
        self.phase_var = phase_var


class BarrierSetup(SyncOp):
    """Barrier declarations, phase variables, init calls, initial consumer arrive, __syncthreads."""
    _fields = ('num_slots', 'num_consumers', 'split_level', 'consumer_level', 'producer_level')

    def __init__(self, num_slots: int, num_consumers: int,
                 split_level: str, consumer_level: str, producer_level: str):
        super().__init__()
        self.num_slots = num_slots
        self.num_consumers = num_consumers
        self.split_level = split_level
        self.consumer_level = consumer_level
        self.producer_level = producer_level
