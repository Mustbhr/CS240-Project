# Architecture Documentation

## System Overview

The Gemini reproduction system implements fast failure recovery for distributed training through in-memory checkpointing and cross-node replication.

## Implementation Status

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| Baseline Trainer | `src/training/baseline_trainer.py` | ✅ Complete | 773 |
| In-Memory Checkpoint | `src/checkpointing/in_memory_checkpoint.py` | ✅ Complete | 464 |
| Data Loader | `src/utils/data_loader.py` | ✅ Complete | 310 |
| Experiment Logger | `src/utils/experiment_logger.py` | ✅ Complete | 521 |
| Worker Agent | `src/agents/worker_agent.py` | 🔨 Skeleton | 170 |
| Root Agent | `src/agents/root_agent.py` | 🔨 Skeleton | 205 |
| Replication Manager | `src/checkpointing/replication.py` | ⏳ Pending | - |

---

## Component Architecture

### 1. Baseline Trainer (✅ Complete)
**Location**: `src/training/baseline_trainer.py`

The baseline trainer implements traditional distributed training with **disk-based checkpointing**. This serves as the comparison point for Gemini.

**Key Classes**:
```python
class TrainingConfig:
    # Configuration dataclass for all training parameters
    # - Model settings (hidden_size, num_layers, etc.)
    # - Training settings (batch_size, learning_rate, etc.)
    # - Checkpoint settings (frequency, directory)

class SimpleLanguageModel(nn.Module):
    # Transformer-based language model for testing
    # - Token + position embeddings
    # - N transformer blocks with attention
    # - Output projection for next-token prediction

class BaselineTrainer:
    # Main trainer class
    # Methods:
    #   - setup_distributed(): Initialize PyTorch DDP
    #   - create_model(): Create model wrapped in DDP
    #   - save_checkpoint(): Save to DISK (baseline)
    #   - load_checkpoint(): Load from DISK (recovery)
    #   - train(): Main training loop with metrics
```

**Checkpoint Format** (disk):
```python
checkpoint = {
    'iteration': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'config': dict,
    'timestamp': float
}
# Saved with: torch.save(checkpoint, path)
```

---

### 2. In-Memory Checkpoint (✅ Complete)
**Location**: `src/checkpointing/in_memory_checkpoint.py`

This is the **core Gemini innovation**. Instead of saving to disk, checkpoints are stored in RAM for near-instant access.

**Key Classes**:
```python
@dataclass
class CheckpointEntry:
    iteration: int
    data: bytes          # Serialized checkpoint
    size_bytes: int
    timestamp: float
    node_id: str
    is_replicated: bool
    replica_nodes: List[str]

class InMemoryCheckpoint:
    # RAM-based checkpoint storage
    # Methods:
    #   - save(): Serialize model state to RAM
    #   - load(): Deserialize from RAM
    #   - get_checkpoint_bytes(): Get raw bytes for replication
    #   - store_replica(): Store checkpoint from peer node
    #   - get_stats(): Memory usage statistics
```

**Performance Characteristics**:
- Save time: ~10-50ms (vs. 100-500ms for disk)
- Load time: ~5-20ms (vs. 50-200ms for disk)
- Memory overhead: ~2x model size per checkpoint
- Configurable max checkpoints (default: 3)

**Thread Safety**: Uses `threading.Lock` for concurrent access.

---

### 3. Data Loader (✅ Complete)
**Location**: `src/utils/data_loader.py`

**Classes**:
```python
class SyntheticDataset:
    # Random tokens for testing (no download needed)
    # Useful for infrastructure testing

class WikipediaDataset:
    # Real Wikipedia data from Hugging Face
    # Requires: pip install datasets transformers
```

**Factory Function**:
```python
dataset = create_dataset(
    dataset_type="synthetic",  # or "wikipedia"
    num_samples=10000,
    seq_length=512
)
```

---

### 4. Experiment Logger (✅ Complete)
**Location**: `src/utils/experiment_logger.py`

Unified logging with optional Weights & Biases (wandb) integration.

**Key Classes**:
```python
class ExperimentLogger:
    # Methods:
    #   - log(): Log any metrics
    #   - log_checkpoint(): Track checkpoint performance
    #   - log_recovery(): Track failure recovery events
    #   - log_comparison(): Compare disk vs memory
    #   - log_system_metrics(): GPU/CPU/memory usage
    #   - finish(): Save and close
```

**Tracked Metrics**:
- `train/loss`, `train/throughput`
- `checkpoint/disk_save_time_ms`, `checkpoint/memory_save_time_ms`
- `comparison/save_speedup`, `comparison/load_speedup`
- `recovery/time_ms`, `recovery/wasted_iterations`

**Local Backup**: All metrics also saved as JSON in `./logs/`

---

### 5. Worker Agent (🔨 Skeleton)
**Location**: `src/agents/worker_agent.py`

**Current Implementation**:
- Basic checkpoint capture
- Local storage management
- Placeholder for network communication

**TODO**:
- [ ] Network transfer for checkpoint shards
- [ ] Peer communication protocol
- [ ] Heartbeat mechanism

---

### 6. Root Agent (🔨 Skeleton)
**Location**: `src/agents/root_agent.py`

**Current Implementation**:
- Node registration
- Basic checkpoint tracking
- Recovery coordination logic

**TODO**:
- [ ] Heartbeat monitoring
- [ ] Failure detection
- [ ] Replication orchestration

---

## Data Flow

### Normal Training Flow (Current Implementation)
```
1. Training iteration completes
2. Every N iterations:
   a. BaselineTrainer.save_checkpoint() → Disk
   b. InMemoryCheckpoint.save() → RAM
3. Metrics logged to wandb/local
4. Training continues
```

### Failure Recovery Flow (Planned)
```
1. Root Agent detects node failure (heartbeat timeout)
2. Root Agent identifies which shards were lost
3. Root Agent requests checkpoint from replica nodes
4. Replacement node receives checkpoint via network
5. Training resumes from last checkpoint
```

---

## Performance Comparison

### Checkpoint Operations

| Operation | Disk (Baseline) | Memory (Gemini) | Speedup |
|-----------|-----------------|-----------------|---------|
| Save | 100-500ms | 10-50ms | ~10x |
| Load | 50-200ms | 5-20ms | ~10x |
| Total Recovery | 150-700ms | 15-70ms | ~10x |

*Actual speedup depends on model size and storage system.*

### Memory Overhead

```
Model size: 100MB
Checkpoint size: ~200MB (model + optimizer)
With 3 checkpoints: ~600MB RAM
Memory overhead: ~6x model size
```

---

## Network Communication (Planned)

### For Replication
- **Protocol**: TCP sockets or gRPC
- **Data**: Serialized checkpoint bytes
- **Compression**: Optional zlib (trades CPU for bandwidth)

### For Coordination
- **Heartbeat**: Every 10 seconds
- **Timeout**: 30 seconds to detect failure
- **Coordination**: Direct TCP or etcd

---

## Monitoring and Metrics

### Key Metrics (Implemented)
- Checkpoint capture time
- Checkpoint size
- Memory utilization
- Training throughput

### Logging Options
1. **wandb**: Interactive dashboard, charts
2. **Local JSON**: Backup in `./logs/`
3. **Console**: Real-time progress

### Usage
```python
logger = ExperimentLogger(
    project="gemini-cs240",
    use_wandb=True
)
logger.log({"loss": 0.5, "throughput": 100})
logger.log_checkpoint(iteration=100, save_time_ms=15.0, ...)
logger.finish()
```

---

## Testing Infrastructure

### Quick Test (`scripts/quick_test.py`)
- Environment verification
- Model forward/backward pass
- Checkpoint save/load
- ~30 seconds to run

### Full Test (`scripts/run_baseline_test.py`)
- Full training loop (50 iterations)
- Disk vs memory comparison
- Optional wandb logging
- ~2-5 minutes to run

### Run Tests
```bash
# Quick verification
python scripts/quick_test.py

# Full test without wandb
python scripts/run_baseline_test.py

# Full test with wandb
python scripts/run_baseline_test.py --wandb
```
