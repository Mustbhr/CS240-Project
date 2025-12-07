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

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Baseline Trainer | Complete | Distributed training with disk checkpointing |
| In-Memory Checkpoint | Complete | RAM-based checkpoint storage |
| Data Loading | Complete | Synthetic dataset support |
| Experiment Logger | Complete | Metrics and visualization |
| Replication Manager | Complete | Ring-topology checkpoint replication |
| Failure Simulation | Complete | Application-level failure injection |

## Architecture

### Components
1. **Baseline Trainer**: Traditional distributed training with disk-based checkpointing (for comparison)
2. **In-Memory Checkpoint**: Fast RAM-based checkpoint storage - the core Gemini innovation
3. **Worker Agents**: Handle local checkpoint capture and replication on each node
4. **Root Agent**: Coordinates recovery and failure detection
5. **Experiment Logger**: wandb integration for metrics and visualization

### Technologies
- **Framework**: PyTorch 2.8 with DistributedDataParallel (DDP)
- **Communication**: NCCL for gradients, TCP for checkpoint transfer
- **Language**: Python 3.9+, CUDA 12.9
- **Hardware**: 4x NVIDIA A100 GPUs (Vast.ai cloud)

## Project Structure

```
.
├── README.md
├── Project_Report.tex            # LaTeX report source
├── Presentation.tex              # LaTeX presentation source
├── CS240_Project_Report.pdf      # Final report
├── CS240_Project_Presentation.pdf # Final presentation
├── requirements.txt              # Dependencies
├── docs/
│   ├── architecture.md           # System architecture
│   └── milestones.md             # Project progress
├── src/
│   ├── checkpointing/
│   │   └── in_memory_checkpoint.py  # RAM-based checkpointing (core)
│   ├── training/
│   │   ├── baseline_trainer.py   # Disk-based checkpointing
│   │   └── gemini_trainer.py     # Gemini with RAM checkpointing
│   └── utils/
│       ├── data_loader.py        # Dataset utilities
│       └── experiment_logger.py  # Metrics logging
├── scripts/
│   ├── run_multi_gpu_experiment.py  # Main experiment script
│   ├── generate_figures.py       # Results visualization
│   ├── quick_test.py             # Quick verification
│   └── run_baseline_test.py      # Infrastructure test
├── figures/                      # Generated plots
└── results/                      # Experiment data (JSON)
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPUs (tested on A100)
- PyTorch 2.0+

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Mustbhr/CS240-Project.git
cd CS240-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Run quick test
python scripts/quick_test.py
```

## Usage

### Quick Test (Verify Installation)
```bash
python scripts/quick_test.py
```

### Full Infrastructure Test
```bash
python scripts/run_baseline_test.py
```

### Run Multi-GPU Experiment
```bash
# Run experiments with multiple GPUs
python scripts/run_multi_gpu_experiment.py --num-runs 3
```

### Generate Figures
```bash
python scripts/generate_figures.py
```


## Key Results

Tested on 4x NVIDIA A100 GPUs (Vast.ai):
- **Disk checkpoint save**: 7,012 ms
- **Memory checkpoint save**: 512 ms  
- **Checkpoint Speedup**: 13.7x faster
- **Recovery Speedup**: 6.1x faster
- **Throughput Improvement**: +100%

## Milestones

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Environment setup + baseline | Complete |
| 2 | In-memory checkpointing | Complete |
| 3 | Replication + failure detection | Complete |
| 4 | Failure injection + recovery | Complete |
| 5-6 | Integration + final report | Complete |

## References

1. Wang et al., *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints*, SOSP 2023.
   - Paper: https://dl.acm.org/doi/10.1145/3600006.3613145
   - Artifact Repository: https://github.com/Gemini-artifacts/gemini

## License

This is an academic reproduction project for CS240 at KAUST. Licensed under MIT.

## Acknowledgments

This project reproduces the Gemini system developed by Wang et al. (SOSP 2023). We acknowledge the original authors for their groundbreaking work in distributed training reliability.
