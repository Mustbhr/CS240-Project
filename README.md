# Fast Failure Recovery in Distributed Training with In-Memory Checkpoints

**Gemini Reproduction Project - CS240 Fall 2025**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Team Members
- Mustafa Albahrani (mustafa.albahrani@kaust.edu.sa)
- Mohammed Alkhalifah (mohammad.alkhalifah@kaust.edu.sa)

## Project Overview

This project reproduces the core Gemini system ([SOSP 2023](https://dl.acm.org/doi/10.1145/3600006.3613145)) that achieves fast, fault-tolerant recovery in distributed deep learning by storing and replicating model checkpoints in RAM across training nodes.

### Key Goals
- Reproduce Gemini's key result: **≥10-15× faster failure recovery** with minimal overhead
- Demonstrate the feasibility of in-memory checkpointing in a scaled-down academic cluster environment
- Compare recovery latency vs. traditional NFS-based checkpointing

### Targeted Traits
- **Reliability**: Fast recovery from hardware/network failures
- **Scalability**: Efficient checkpoint replication across nodes

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Baseline Trainer | ✅ Complete | Distributed training with disk checkpointing |
| In-Memory Checkpoint | ✅ Complete | RAM-based checkpoint storage |
| Data Loading | ✅ Complete | Synthetic + Wikipedia dataset support |
| Experiment Logger | ✅ Complete | wandb integration for tracking |
| Worker Agent | 🔨 Skeleton | Needs network communication |
| Root Agent | 🔨 Skeleton | Needs failure detection |
| Replication Manager | ⏳ Pending | Cross-node checkpoint transfer |
| Failure Injection | ⏳ Pending | Testing recovery performance |

## Architecture

### Components
1. **Baseline Trainer**: Traditional distributed training with disk-based checkpointing (for comparison)
2. **In-Memory Checkpoint**: Fast RAM-based checkpoint storage - the core Gemini innovation
3. **Worker Agents**: Handle local checkpoint capture and replication on each node
4. **Root Agent**: Coordinates recovery and failure detection
5. **Experiment Logger**: wandb integration for metrics and visualization

### Technologies
- **Framework**: PyTorch with DistributedDataParallel (DDP)
- **Communication**: NCCL (planned), TCP for checkpoint transfer
- **Logging**: Weights & Biases (wandb)
- **Language**: Python 3.9+
- **Hardware**: KAUST IBEX cluster (tested on single GPU node)

## Project Structure

```
.
├── README.md
├── requirements.txt              # Main dependencies
├── requirements-minimal.txt      # Minimal deps for testing
├── docs/
│   ├── architecture.md          # System architecture details
│   └── milestones.md            # Project timeline and progress
├── src/
│   ├── agents/
│   │   ├── worker_agent.py      # Worker node agent (skeleton)
│   │   └── root_agent.py        # Root coordinator (skeleton)
│   ├── checkpointing/
│   │   └── in_memory_checkpoint.py  # ✅ RAM-based checkpointing
│   ├── training/
│   │   └── baseline_trainer.py  # ✅ Baseline with disk checkpointing
│   └── utils/
│       ├── data_loader.py       # ✅ Dataset utilities
│       └── experiment_logger.py # ✅ wandb integration
├── configs/
│   ├── training_config.yaml.template
│   └── cluster_config.yaml.template
├── scripts/
│   ├── quick_test.py            # ✅ Quick verification test
│   ├── run_baseline_test.py     # ✅ Full infrastructure test
│   ├── ibex_test.sh             # SLURM batch script
│   └── setup_environment.sh     # Environment setup
├── tests/
│   └── test_agents.py           # Unit tests
├── logs/                        # Experiment logs (local)
├── checkpoints/                 # Saved checkpoints
└── results/                     # Experiment results
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPUs
- Access to KAUST IBEX cluster (or similar HPC)

### Setup on IBEX

```bash
# 1. Clone the repository
git clone https://github.com/Mustbhr/CS240-Project.git
cd CS240-Project

# 2. Get an interactive GPU session
srun --nodes=1 --gpus-per-node=1 --time=00:30:00 --mem=32G --pty bash

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install PyTorch with CUDA (check your CUDA version with nvidia-smi)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install other dependencies
python -m pip install -r requirements.txt

# 6. Run quick test
python scripts/quick_test.py
```

### Setup Locally (for development)

```bash
# Clone and setup
git clone https://github.com/Mustbhr/CS240-Project.git
cd CS240-Project
python -m venv venv
source venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

## Usage

### Quick Test (Verify Installation)
```bash
python scripts/quick_test.py
```

### Full Infrastructure Test
```bash
# Without wandb logging
python scripts/run_baseline_test.py

# With wandb logging (requires wandb login)
python scripts/run_baseline_test.py --wandb
```

### Expected Output
The tests compare disk-based vs memory-based checkpointing:
```
COMPARISON RESULTS
========================================
Save speedup:     X.XXx faster
Load speedup:     X.XXx faster
Recovery speedup: X.XXx faster
```

## Key Results (Preliminary)

Testing on KAUST IBEX (single node):
- **Disk checkpoint save**: ~XXX ms
- **Memory checkpoint save**: ~XX ms
- **Speedup**: ~10-15× faster (varies by model size)

*Full multi-node results pending cluster access.*

## Milestones

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Environment setup + baseline | ✅ Complete |
| 2 | In-memory checkpointing | ✅ Complete |
| 3 | Replication + failure detection | 🔨 In Progress |
| 4 | Failure injection + recovery | ⏳ Pending |
| 5-6 | Integration + final report | ⏳ Pending |

## References

1. Wang et al., *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints*, SOSP 2023.
   - Paper: https://dl.acm.org/doi/10.1145/3600006.3613145
   - Artifact Repository: https://github.com/Gemini-artifacts/gemini

## License

This is an academic reproduction project for CS240 at KAUST. Licensed under MIT.

## Acknowledgments

This project reproduces the Gemini system developed by Wang et al. (SOSP 2023). We acknowledge the original authors for their groundbreaking work in distributed training reliability.
