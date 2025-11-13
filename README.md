# Fast Failure Recovery in Distributed Training with In-Memory Checkpoints

**Gemini Reproduction Project - CS240 Fall 2025**

## Team Members
- Mustafa Albahrani (mustafa.albahrani@kaust.edu.sa)
- Mohammed Alkhalifa (mohammed.alkhalifah@kaust.edu.sa)

## Project Overview

This project reproduces the core Gemini system ([SOSP 2023](https://dl.acm.org/doi/10.1145/3600006.3613145)) that achieves fast, fault-tolerant recovery in distributed deep learning by storing and replicating model checkpoints in RAM across training nodes.

### Key Goals
- Reproduce Gemini's key result: **≥15× faster failure recovery** with minimal overhead
- Demonstrate the feasibility of in-memory checkpointing in a scaled-down academic cluster environment
- Compare recovery latency vs. traditional NFS-based checkpointing

### Targeted Traits
- **Reliability**: Fast recovery from hardware/network failures
- **Scalability**: Efficient checkpoint replication across nodes

## Architecture

### Components
1. **Worker Agents**: Handle local checkpoint capture and replication on each node
2. **Root Agent**: Coordinates recovery and failure detection
3. **Checkpoint Shards**: Replicated between nodes using a group-placement strategy (m=2)
4. **Failure Injection**: Measure recovery latency vs. baseline NFS-based checkpointing

### Technologies
- **Framework**: PyTorch with DeepSpeed (ZeRO-3)
- **Communication**: NCCL, etcd
- **Language**: Python
- **Hardware**: 4 multi-GPU nodes (KAUST IBEX or AWS)
- **Dataset**: Scaled-down Wikipedia-en corpus from Hugging Face

## Project Structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   ├── architecture.md
│   └── milestones.md
├── src/
│   ├── agents/
│   │   ├── worker_agent.py
│   │   └── root_agent.py
│   ├── checkpointing/
│   │   ├── in_memory_checkpoint.py
│   │   └── replication.py
│   ├── training/
│   │   ├── distributed_trainer.py
│   │   └── baseline_trainer.py
│   └── utils/
│       ├── failure_injection.py
│       └── metrics.py
├── configs/
│   ├── training_config.yaml
│   └── cluster_config.yaml
├── scripts/
│   ├── setup_cluster.sh
│   └── run_experiment.sh
├── tests/
│   └── test_checkpointing.py
└── results/
    └── .gitkeep
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPUs
- Access to KAUST IBEX cluster or AWS instances

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd CS240-Project

# Install dependencies
pip install -r requirements.txt

# Configure cluster settings
cp configs/cluster_config.yaml.template configs/cluster_config.yaml
# Edit cluster_config.yaml with your node information
```

## Usage

### Running Baseline Training
```bash
# Traditional NFS-based checkpointing
python scripts/run_experiment.sh --mode baseline --nodes 4
```

### Running Gemini In-Memory Checkpointing
```bash
# In-memory checkpointing with replication
python scripts/run_experiment.sh --mode gemini --nodes 4 --replication-factor 2
```

### Injecting Failures
```bash
# Test recovery latency
python src/utils/failure_injection.py --target-node 2 --failure-time 300
```

## Milestones (6 weeks)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Environment setup | Working distributed training job; baseline metrics |
| 2-3 | Implement in-memory checkpointing | Checkpoint creation/loading from RAM; replication between nodes |
| 4-5 | Failure injection + recovery | Measured "wasted time" comparison vs. NFS baseline |
| 6 | Final integration + report | Demonstration, slides, and final report |

## Expected Outcomes

We will measure:
- **Recovery latency**: Time from failure to resumed training
- **Training throughput**: With vs. without checkpointing
- **Wasted time reduction**: Compared to persistent storage baseline

Success criteria: Achieve 10-13× faster recovery with minimal throughput loss.

## References

1. Wang et al., *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints*, SOSP 2023.
   - Paper: https://dl.acm.org/doi/10.1145/3600006.3613145
   - Artifact Repository: https://github.com/Gemini-artifacts/gemini

## License

This is an academic reproduction project for CS240 at KAUST.

## Acknowledgments

This project reproduces the Gemini system developed by Wang et al. (SOSP 2023). We acknowledge the original authors for their groundbreaking work in distributed training reliability.

