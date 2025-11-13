# Architecture Documentation

## System Overview

The Gemini reproduction system implements fast failure recovery for distributed training through in-memory checkpointing and cross-node replication.

## Component Architecture

### 1. Worker Agent
**Location**: `src/agents/worker_agent.py`

**Responsibilities**:
- Captures local model state during training
- Manages in-memory checkpoint storage
- Handles checkpoint shard replication to peer nodes
- Responds to recovery requests from Root Agent

**Key Methods**:
- `capture_checkpoint()`: Store model state in RAM
- `replicate_shard()`: Send checkpoint shard to peer node
- `serve_checkpoint()`: Provide checkpoint data during recovery

### 2. Root Agent
**Location**: `src/agents/root_agent.py`

**Responsibilities**:
- Monitors cluster health and detects node failures
- Coordinates checkpoint recovery process
- Manages group-placement strategy (m=2 replication)
- Orchestrates training resumption

**Key Methods**:
- `detect_failure()`: Monitor node heartbeats
- `initiate_recovery()`: Coordinate checkpoint restoration
- `rebalance_shards()`: Redistribute checkpoints after recovery

### 3. Checkpoint Management
**Location**: `src/checkpointing/`

**Components**:
- **InMemoryCheckpoint**: Efficient in-RAM checkpoint storage
- **ReplicationManager**: Handles shard distribution across nodes

**Design Considerations**:
- Memory-efficient serialization
- Network-optimized transfer protocols
- Consistency guarantees during concurrent access

### 4. Training Infrastructure
**Location**: `src/training/`

**Components**:
- **DistributedTrainer**: Main training loop with Gemini integration
- **BaselineTrainer**: Traditional NFS-based checkpointing for comparison

## Data Flow

### Normal Training Flow
```
1. Training iteration completes
2. Worker Agent captures checkpoint in RAM
3. Checkpoint shard replicated to peer node(s)
4. Training continues
```

### Failure Recovery Flow
```
1. Root Agent detects node failure
2. Root Agent identifies which shards were lost
3. Root Agent requests checkpoint from replica nodes
4. Failed node's checkpoint restored from replicas
5. Training resumes from last checkpoint
```

## Network Communication

- **NCCL**: GPU-to-GPU communication for training
- **etcd**: Distributed coordination and configuration
- **TCP/RDMA**: Checkpoint replication between nodes

## Replication Strategy

**Group Placement (m=2)**:
- Each checkpoint shard replicated to 2 nodes
- Strategic placement to minimize network hops
- Balance between redundancy and memory overhead

## Performance Considerations

### Memory Management
- Checkpoint size proportional to model size
- Trade-off: Checkpoint frequency vs. memory usage
- Target: <10% memory overhead

### Network Overhead
- Asynchronous replication to minimize training impact
- Compression for large model states
- Target: <5% throughput degradation

## Monitoring and Metrics

**Key Metrics**:
- Checkpoint capture time
- Replication latency
- Recovery time (failure to resumed training)
- Training throughput (samples/second)
- Memory utilization

**Logging**:
- Structured logging for all checkpoint operations
- Performance counters for bottleneck identification
- Recovery event traces

