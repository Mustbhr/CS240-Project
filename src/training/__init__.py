"""
Training module for distributed training loops
"""

from .baseline_trainer import (
    BaselineTrainer,
    TrainingConfig,
    TrainingMetrics,
    SimpleLanguageModel,
    SyntheticDataset
)

from .gemini_trainer import (
    GeminiTrainer,
    GeminiConfig,
    CheckpointReplicator
)

__all__ = [
    # Baseline
    'BaselineTrainer',
    'TrainingConfig',
    'TrainingMetrics',
    'SimpleLanguageModel',
    'SyntheticDataset',
    # Gemini
    'GeminiTrainer',
    'GeminiConfig',
    'CheckpointReplicator'
]

