"""
Checkpointing module for in-memory checkpoint management

This is the core of Gemini's innovation - storing checkpoints in RAM
instead of on disk for near-instant recovery.
"""

from .in_memory_checkpoint import (
    InMemoryCheckpoint,
    CheckpointEntry,
    compare_checkpoint_times
)

__all__ = [
    'InMemoryCheckpoint',
    'CheckpointEntry',
    'compare_checkpoint_times'
]

