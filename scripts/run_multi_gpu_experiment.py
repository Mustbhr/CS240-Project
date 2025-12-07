#!/usr/bin/env python3
"""
Multi-GPU Experiment Runner

This script runs comprehensive experiments comparing:
1. Baseline (disk) checkpointing
2. Gemini (in-memory) checkpointing
3. Failure simulation and recovery

Run on your 4-GPU setup:
    python scripts/run_multi_gpu_experiment.py

With wandb logging:
    python scripts/run_multi_gpu_experiment.py --wandb

This will:
- Test single-GPU baseline and Gemini checkpointing
- Test multi-GPU distributed training
- Simulate failures and measure recovery
- Compare checkpoint performance
- Generate results for your report

Author: Mustafa Albahrani, Mohammed Alkhalifah
Course: CS240 - Distributed Systems
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

CHECK_POINT_FRQ = 25

HIDDEN_SIZE = 1024
NUM_LAYERS = 12
NUM_HEADS = 16

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global wandb logger
exp_logger = None


def init_wandb(use_wandb: bool, config: dict):
    """Initialize wandb if requested."""
    global exp_logger

    if use_wandb:
        try:
            from src.utils import ExperimentLogger

            exp_logger = ExperimentLogger(
                project="gemini-cs240",
                run_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                tags=["multi-gpu", "comparison"],
                use_wandb=True,
            )
            print("✓ wandb initialized")
            return True
        except Exception as e:
            print(f"⚠️ wandb init failed: {e}")
            return False
    return False


def log_to_wandb(metrics: dict):
    """Log metrics to wandb if initialized."""
    global exp_logger
    if exp_logger:
        exp_logger.log(metrics)


def check_environment():
    """Verify the environment is set up correctly."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available: {num_gpus} GPUs")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Check our modules
    try:
        from src.training import BaselineTrainer, TrainingConfig
        from src.checkpointing import InMemoryCheckpoint

        print("✓ Project modules loaded")
    except ImportError as e:
        print(f"✗ Failed to load modules: {e}")
        return False

    print()
    return True


def run_single_gpu_baseline(iterations=100, checkpoint_freq=CHECK_POINT_FRQ):
    """Run baseline (disk) checkpointing on single GPU."""
    print("=" * 60)
    print("EXPERIMENT 1: Single-GPU Baseline (Disk Checkpointing)")
    print("=" * 60)

    from src.training import BaselineTrainer, TrainingConfig
    from src.utils import SyntheticDataset

    config = TrainingConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/baseline_single",
        log_interval=25,
    )

    trainer = BaselineTrainer(config=config, rank=0, world_size=1, local_rank=0)

    dataset = SyntheticDataset(
        num_samples=5000, seq_length=256, vocab_size=config.vocab_size
    )

    start_time = time.time()
    results = trainer.train(dataset)
    total_time = time.time() - start_time

    # Extract checkpoint times
    checkpoint_times = [
        m.checkpoint_time * 1000
        for m in trainer.metrics_history
        if m.checkpoint_time > 0
    ]

    # MEASURE ACTUAL DISK RECOVERY TIME (not estimated!)
    disk_recovery_ms = 0
    checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_latest.pt"
    if checkpoint_path.exists():
        # Measure time to load from disk
        recovery_start = time.time()
        _ = torch.load(checkpoint_path, map_location='cuda:0')
        disk_recovery_ms = (time.time() - recovery_start) * 1000
        logger.info(f"Actual disk recovery time: {disk_recovery_ms:.2f}ms")

    print(f"\n📊 Baseline Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Iterations: {results['total_iterations']}")
    print(f"   Avg throughput: {results['average_throughput']:.1f} samples/s")
    print(f"   Checkpoint SAVE times: {checkpoint_times}")
    if checkpoint_times:
        print(f"   Avg checkpoint SAVE: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
    print(f"   Disk RECOVERY time: {disk_recovery_ms:.2f}ms")

    return {
        "type": "baseline_single",
        "total_time": total_time,
        "iterations": results["total_iterations"],
        "throughput": results["average_throughput"],
        "checkpoint_times_ms": checkpoint_times,
        "avg_checkpoint_ms": (
            sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
        ),
        "disk_recovery_ms": disk_recovery_ms,
    }


def run_single_gpu_gemini(iterations=100, checkpoint_freq=CHECK_POINT_FRQ):
    """Run Gemini (in-memory) checkpointing on single GPU."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Single-GPU Gemini (In-Memory Checkpointing)")
    print("=" * 60)

    from src.training import BaselineTrainer, TrainingConfig
    from src.checkpointing import InMemoryCheckpoint
    from src.utils import SyntheticDataset

    config = TrainingConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/gemini_single",
        log_interval=25,
    )

    # Create trainer (we'll override checkpoint behavior)
    device = torch.device("cuda:0")

    # Create model
    from src.training.baseline_trainer import SimpleLanguageModel

    model = SimpleLanguageModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_length=config.max_seq_length,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # In-memory checkpoint manager
    ckpt_manager = InMemoryCheckpoint(
        node_id="gpu-0", max_checkpoints=5, max_memory_gb=2.0
    )

    dataset = SyntheticDataset(
        num_samples=5000, seq_length=256, vocab_size=config.vocab_size
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Training loop
    checkpoint_times = []
    losses = []
    start_time = time.time()
    iteration = 0

    for epoch in range(100):  # Max epochs
        for batch in dataloader:
            if iteration >= iterations:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Gemini checkpoint (in-memory)
            if iteration > 0 and iteration % checkpoint_freq == 0:
                ckpt_time = ckpt_manager.save(model, optimizer, iteration)
                checkpoint_times.append(ckpt_time * 1000)  # Convert to ms
                logger.info(
                    f"Iter {iteration}: Gemini checkpoint in {ckpt_time*1000:.2f}ms"
                )

            if iteration % 25 == 0:
                logger.info(f"Iter {iteration}/{iterations} | Loss: {loss.item():.4f}")

            iteration += 1

        if iteration >= iterations:
            break

    total_time = time.time() - start_time
    total_samples = iteration * config.batch_size

    # MEASURE ACTUAL RAM RECOVERY TIME
    ram_recovery_ms = 0
    latest_iter = ckpt_manager.get_latest_iteration()
    if latest_iter is not None:
        recovery_start = time.time()
        ckpt_manager.load(model, optimizer, latest_iter, device)
        ram_recovery_ms = (time.time() - recovery_start) * 1000
        logger.info(f"Actual RAM recovery time: {ram_recovery_ms:.2f}ms")

    print(f"\n📊 Gemini Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Iterations: {iteration}")
    print(f"   Avg throughput: {total_samples/total_time:.1f} samples/s")
    print(f"   Checkpoint SAVE times: {checkpoint_times}")
    if checkpoint_times:
        print(f"   Avg checkpoint SAVE: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
    print(f"   RAM RECOVERY time: {ram_recovery_ms:.2f}ms ⚡")

    # Get memory stats
    stats = ckpt_manager.get_stats()
    print(f"   Memory used for checkpoints: {stats['current_memory_mb']:.2f}MB")

    return {
        "type": "gemini_single",
        "total_time": total_time,
        "iterations": iteration,
        "throughput": total_samples / total_time,
        "checkpoint_times_ms": checkpoint_times,
        "avg_checkpoint_ms": (
            sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
        ),
        "ram_recovery_ms": ram_recovery_ms,
        "memory_mb": stats["current_memory_mb"],
    }


def run_multi_gpu_disk_baseline(num_gpus=4, iterations=100, checkpoint_freq=CHECK_POINT_FRQ):
    """Run multi-GPU training with DISK checkpointing (baseline for comparison)."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3a: Multi-GPU DISK Baseline ({num_gpus} GPUs)")
    print("=" * 60)
    
    import torch.multiprocessing as mp
    from src.training.gemini_trainer import GeminiConfig, run_gemini_worker
    
    # Use Gemini trainer but with disk checkpointing flag
    config = GeminiConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/multi_disk",
        log_interval=25,
        replication_factor=1,  # No replication for disk baseline
        use_disk_checkpoint=True,  # Use disk instead of RAM
    )
    
    start_time = time.time()
    mp.spawn(run_gemini_worker, args=(num_gpus, config), nprocs=num_gpus, join=True)
    total_time = time.time() - start_time
    
    results_path = Path("./checkpoints/multi_disk/gemini/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        checkpoint_times = [c["total_ms"] for c in results.get("checkpoint_times", [])]
        measured_recovery_ms = results.get("measured_recovery_ms", 0)
        
        print(f"\n📊 Multi-GPU DISK Baseline Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   GPUs used: {num_gpus}")
        print(f"   Checkpoint SAVE times: {checkpoint_times}")
        if checkpoint_times:
            print(f"   Avg checkpoint SAVE: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
        print(f"   Disk RECOVERY time: {measured_recovery_ms:.2f}ms")
        
        return {
            "type": "multi_gpu_disk",
            "total_time": total_time,
            "num_gpus": num_gpus,
            "checkpoint_times_ms": checkpoint_times,
            "avg_checkpoint_ms": sum(checkpoint_times)/len(checkpoint_times) if checkpoint_times else 0,
            "measured_recovery_ms": measured_recovery_ms,
        }
    else:
        print("⚠️ Results file not found")
        return {"type": "multi_gpu_disk", "total_time": total_time, "note": "results not found"}


def run_multi_gpu_gemini(num_gpus=4, iterations=100, checkpoint_freq=CHECK_POINT_FRQ):
    """Run multi-GPU Gemini training with IN-MEMORY checkpointing + replication."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3b: Multi-GPU Gemini IN-MEMORY ({num_gpus} GPUs)")
    print("=" * 60)

    import torch.multiprocessing as mp
    import torch.distributed as dist

    from src.training.gemini_trainer import GeminiConfig, run_gemini_worker

    config = GeminiConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/gemini_multi",
        log_interval=25,
        replication_factor=2,
    )

    # Run multi-GPU training
    start_time = time.time()
    mp.spawn(run_gemini_worker, args=(num_gpus, config), nprocs=num_gpus, join=True)
    total_time = time.time() - start_time

    # Load results
    results_path = Path("./checkpoints/gemini_multi/gemini/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        checkpoint_times = [c["total_ms"] for c in results.get("checkpoint_times", [])]
        measured_recovery_ms = results.get("measured_recovery_ms", 0)

        print(f"\n📊 Multi-GPU Gemini (RAM) Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   GPUs used: {num_gpus}")
        print(f"   Checkpoint SAVE times: {checkpoint_times}")
        if checkpoint_times:
            print(f"   Avg checkpoint SAVE: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
        print(f"   RAM RECOVERY time: {measured_recovery_ms:.2f}ms ⚡")

        # Log to wandb (will be updated later with recovery time)
        log_to_wandb(
            {
                "multi_gpu_ram/total_time": total_time,
                "multi_gpu_ram/num_gpus": num_gpus,
                "multi_gpu_ram/avg_checkpoint_save_ms": (
                    sum(checkpoint_times) / len(checkpoint_times)
                    if checkpoint_times
                    else 0
                ),
            }
        )

        return {
            "type": "gemini_multi",
            "total_time": total_time,
            "num_gpus": num_gpus,
            "checkpoint_times_ms": checkpoint_times,
            "avg_checkpoint_ms": (
                sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
            ),
            "measured_recovery_ms": measured_recovery_ms,
            "raw_results": results,
        }
    else:
        print("⚠️ Results file not found")
        return {"type": "gemini_multi", "error": "results not found"}


def run_failure_simulation(
    num_gpus=4, iterations=150, checkpoint_freq=CHECK_POINT_FRQ, failure_iteration=75
):
    """
    Simulate a GPU failure and test recovery.

    This demonstrates Gemini's key benefit: fast recovery from replicas.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 4: Failure Simulation & Recovery")
    print("=" * 60)
    print(f"   Simulating GPU 1 failure at iteration {failure_iteration}")

    import torch.multiprocessing as mp
    from src.training.gemini_trainer import GeminiConfig, run_gemini_worker

    config = GeminiConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/gemini_failure",
        log_interval=25,
        replication_factor=2,
        simulate_failure=True,
        failure_gpu=1,  # GPU 1 will "fail"
        failure_iteration=failure_iteration,
    )

    start_time = time.time()

    try:
        mp.spawn(run_gemini_worker, args=(num_gpus, config), nprocs=num_gpus, join=True)
    except Exception as e:
        logger.info(f"Training completed with simulated failure: {e}")

    total_time = time.time() - start_time

    # Load results
    results_path = Path("./checkpoints/gemini_failure/gemini/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        recovery_events = results.get("recovery_events", [])

        print(f"\n📊 Failure Simulation Results:")
        print(f"   Total time with failure: {total_time:.2f}s")
        print(f"   Failure injected at iteration: {failure_iteration}")
        print(f"   Recovery events: {len(recovery_events)}")

        if recovery_events:
            for event in recovery_events:
                print(
                    f"   - GPU {event.get('failed_rank')} recovered by GPU {event.get('recovered_by')}"
                )
                print(f"     Recovery time: {event.get('recovery_time_ms', 0):.2f}ms")

        # Log to wandb
        log_to_wandb(
            {
                "failure/total_time": total_time,
                "failure/failure_iteration": failure_iteration,
                "failure/num_recovery_events": len(recovery_events),
                "failure/recovery_time_ms": (
                    recovery_events[0].get("recovery_time_ms", 0)
                    if recovery_events
                    else 0
                ),
            }
        )

        return {
            "type": "failure_simulation",
            "total_time": total_time,
            "failure_iteration": failure_iteration,
            "recovery_events": recovery_events,
        }
    else:
        print(
            "⚠️ Results file not found (may be expected if failure crashed the process)"
        )
        return {
            "type": "failure_simulation",
            "total_time": total_time,
            "note": "process may have crashed",
        }


def compare_results(baseline, gemini_single, gemini_multi=None, multi_disk=None, failure_results=None):
    """Compare and display results."""
    global exp_logger

    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Gemini")
    print("=" * 60)

    comparison = {}

    if baseline["avg_checkpoint_ms"] > 0 and gemini_single["avg_checkpoint_ms"] > 0:
        speedup = baseline["avg_checkpoint_ms"] / gemini_single["avg_checkpoint_ms"]

        print(f"\n📈 Single-GPU Checkpoint Performance:")
        print(f"   Disk:               {baseline['avg_checkpoint_ms']:.2f} ms")
        print(f"   RAM (Gemini):       {gemini_single['avg_checkpoint_ms']:.2f} ms")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   SPEEDUP:            {speedup:.1f}× FASTER! 🚀")

        comparison = {
            "single_gpu_speedup": speedup,
            "single_disk_ms": baseline["avg_checkpoint_ms"],
            "single_ram_ms": gemini_single["avg_checkpoint_ms"],
        }

        # Multi-GPU comparison with breakdown
        if (
            multi_disk
            and gemini_multi
            and multi_disk.get("avg_checkpoint_ms", 0) > 0
            and gemini_multi.get("avg_checkpoint_ms", 0) > 0
        ):
            multi_disk_ms = multi_disk["avg_checkpoint_ms"]
            multi_ram_total_ms = gemini_multi["avg_checkpoint_ms"]
            
            # Breakdown (RAM save is similar to single-GPU)
            ram_save_ms = gemini_single["avg_checkpoint_ms"]
            replication_ms = max(0, multi_ram_total_ms - ram_save_ms)
            
            print(f"\n🔄 Multi-GPU Checkpoint Breakdown:")
            print(f"   Multi-GPU DISK:        {multi_disk_ms:.2f} ms (save only)")
            print(f"   Multi-GPU RAM:")
            print(f"     - Save to RAM:       {ram_save_ms:.2f} ms")
            print(f"     - P2P Replication:   {replication_ms:.2f} ms (adds redundancy)")
            print(f"     - Total:             {multi_ram_total_ms:.2f} ms")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            if multi_disk_ms > multi_ram_total_ms:
                speedup = multi_disk_ms / multi_ram_total_ms
                print(f"   Checkpoint: RAM+Replication is {speedup:.1f}× faster than Disk")
            else:
                overhead = multi_ram_total_ms / multi_disk_ms
                print(f"   ⚠️ RAM+Replication is {overhead:.1f}× SLOWER than Disk (due to replication)")
            
            # MULTI-GPU RECOVERY COMPARISON (THE KEY METRIC!)
            multi_disk_recovery = multi_disk.get("measured_recovery_ms", 0)
            multi_ram_recovery = gemini_multi.get("measured_recovery_ms", 0)
            
            if multi_disk_recovery > 0 and multi_ram_recovery > 0:
                recovery_speedup = multi_disk_recovery / multi_ram_recovery
                print(f"\n   📦 Multi-GPU RECOVERY (Key Comparison!):")
                print(f"      Disk Recovery:   {multi_disk_recovery:.2f} ms")
                print(f"      RAM Recovery:    {multi_ram_recovery:.2f} ms")
                print(f"      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"      RECOVERY SPEEDUP: {recovery_speedup:.1f}× FASTER! ⚡")
                comparison["multi_gpu_disk_recovery_ms"] = multi_disk_recovery
                comparison["multi_gpu_ram_recovery_ms"] = multi_ram_recovery
                comparison["multi_gpu_recovery_speedup"] = recovery_speedup
            
            comparison["multi_gpu_disk_ms"] = multi_disk_ms
            comparison["multi_gpu_ram_total_ms"] = multi_ram_total_ms
            comparison["multi_gpu_ram_save_ms"] = ram_save_ms
            comparison["multi_gpu_replication_ms"] = replication_ms

        print(f"\n📊 Training Throughput:")
        print(f"   Baseline:           {baseline['throughput']:.1f} samples/s")
        print(f"   Gemini:             {gemini_single['throughput']:.1f} samples/s")

        # Recovery comparison using ACTUAL MEASURED times
        disk_recovery = baseline.get("disk_recovery_ms", 0)
        ram_recovery = gemini_single.get("ram_recovery_ms", 0)
        
        # Also get from failure simulation if available
        failure_ram_recovery = 0
        if failure_results and "recovery_events" in failure_results:
            events = failure_results.get("recovery_events", [])
            if events:
                failure_ram_recovery = events[0].get("recovery_time_ms", 0)
        
        # Use whichever RAM recovery we have
        if ram_recovery == 0 and failure_ram_recovery > 0:
            ram_recovery = failure_ram_recovery
        
        print(f"\n⚡ RECOVERY Performance (Key Gemini Benefit!):")
        if disk_recovery > 0:
            print(f"   Disk Recovery:      {disk_recovery:.2f} ms (measured)")
        else:
            # Estimate disk recovery as similar to save time
            disk_recovery = baseline["avg_checkpoint_ms"]
            print(f"   Disk Recovery:      ~{disk_recovery:.2f} ms (estimated ≈ save time)")
        
        if ram_recovery > 0:
            print(f"   RAM Recovery:       {ram_recovery:.2f} ms (measured)")
        else:
            print(f"   RAM Recovery:       (not measured)")
        
        if disk_recovery > 0 and ram_recovery > 0:
            recovery_speedup = disk_recovery / ram_recovery
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   RECOVERY SPEEDUP:   {recovery_speedup:.1f}× FASTER! ⚡")
            comparison["disk_recovery_ms"] = disk_recovery
            comparison["ram_recovery_ms"] = ram_recovery
            comparison["recovery_speedup"] = recovery_speedup

        # Log comprehensive comparison to wandb
        if exp_logger:
            # Single-GPU comparison
            log_to_wandb({
                "comparison/single_gpu_checkpoint_speedup": comparison.get("single_gpu_speedup", 0),
                "comparison/single_gpu_recovery_speedup": comparison.get("recovery_speedup", 0),
                "comparison/single_disk_save_ms": comparison.get("single_disk_ms", 0),
                "comparison/single_ram_save_ms": comparison.get("single_ram_ms", 0),
                "comparison/single_disk_recovery_ms": comparison.get("disk_recovery_ms", 0),
                "comparison/single_ram_recovery_ms": comparison.get("ram_recovery_ms", 0),
            })
            
            # Multi-GPU comparison
            if "multi_gpu_recovery_speedup" in comparison:
                log_to_wandb({
                    "comparison/multi_gpu_checkpoint_disk_ms": comparison.get("multi_gpu_disk_ms", 0),
                    "comparison/multi_gpu_checkpoint_ram_ms": comparison.get("multi_gpu_ram_total_ms", 0),
                    "comparison/multi_gpu_recovery_speedup": comparison.get("multi_gpu_recovery_speedup", 0),
                    "comparison/multi_gpu_disk_recovery_ms": comparison.get("multi_gpu_disk_recovery_ms", 0),
                    "comparison/multi_gpu_ram_recovery_ms": comparison.get("multi_gpu_ram_recovery_ms", 0),
                    "comparison/multi_gpu_replication_ms": comparison.get("multi_gpu_replication_ms", 0),
                })
            
            # Failure simulation
            if "failure_recovery_ms" in comparison:
                log_to_wandb({
                    "comparison/failure_recovery_ms": comparison.get("failure_recovery_ms", 0),
                })

        return comparison
    else:
        print("⚠️ Not enough data for comparison")
        return None


def save_final_results(
    baseline, gemini_single, gemini_multi, multi_disk, failure_results, comparison
):
    """Save all results for the report."""
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment_results_{timestamp}.json"

    all_results = {
        "timestamp": timestamp,
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ],
        },
        "experiments": {
            "baseline_single_disk": baseline,
            "gemini_single_ram": gemini_single,
            "multi_gpu_disk": multi_disk,
            "multi_gpu_ram": gemini_multi,
            "failure_simulation": failure_results,
        },
        "comparison": comparison,
    }

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")
    return results_file


def aggregate_results(all_runs: list) -> dict:
    """
    Aggregate results from multiple runs to compute mean ± std.
    
    Args:
        all_runs: List of result dictionaries from multiple runs
        
    Returns:
        Dictionary with aggregated statistics (mean and std for each metric)
    """
    if not all_runs:
        return {}
    
    if len(all_runs) == 1:
        return all_runs[0]
    
    aggregated = {"num_runs": len(all_runs), "individual_runs": all_runs}
    
    # Collect all numeric keys from experiments
    sample = all_runs[0]
    
    def aggregate_dict(key_prefix: str, dicts: list):
        """Aggregate numeric values from a list of dicts."""
        if not dicts or dicts[0] is None:
            return {}
        
        result = {}
        sample_dict = dicts[0]
        
        for key, value in sample_dict.items():
            if key in ["type", "note", "raw_results", "recovery_events", "checkpoint_times_ms"]:
                # Keep first value for non-numeric fields
                result[key] = value
                continue
                
            values = [d.get(key) for d in dicts if d and d.get(key) is not None]
            
            if not values:
                continue
                
            if isinstance(value, (int, float)):
                arr = np.array(values, dtype=float)
                result[f"{key}_mean"] = float(np.mean(arr))
                result[f"{key}_std"] = float(np.std(arr))
                result[f"{key}_min"] = float(np.min(arr))
                result[f"{key}_max"] = float(np.max(arr))
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                # For lists of numbers (like checkpoint_times_ms), compute overall stats
                all_values = []
                for d in dicts:
                    if d and key in d and d[key]:
                        all_values.extend(d[key])
                if all_values:
                    arr = np.array(all_values, dtype=float)
                    result[f"{key}_mean"] = float(np.mean(arr))
                    result[f"{key}_std"] = float(np.std(arr))
                    
        return result
    
    # Aggregate each experiment type
    for exp_key in ["baseline", "gemini_single", "gemini_multi", "multi_disk", "failure_results", "comparison"]:
        exp_dicts = [run.get(exp_key) for run in all_runs if run.get(exp_key)]
        if exp_dicts:
            aggregated[exp_key] = aggregate_dict(exp_key, exp_dicts)
    
    return aggregated


def run_single_experiment(args):
    """
    Run a single set of all experiments.
    
    Returns:
        Dictionary containing all experiment results
    """
    import gc
    
    # Experiment 1: Single-GPU Baseline (Disk)
    baseline = run_single_gpu_baseline(args.iterations, args.checkpoint_freq)

    # Log baseline to wandb
    log_to_wandb(
        {
            "baseline/avg_checkpoint_save_ms": baseline["avg_checkpoint_ms"],
            "baseline/disk_recovery_ms": baseline.get("disk_recovery_ms", 0),
            "baseline/throughput": baseline["throughput"],
            "baseline/total_time": baseline.get("total_time", 0),
        }
    )

    # Clear GPU memory after baseline
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment 2: Single-GPU Gemini (Memory)
    gemini_single = run_single_gpu_gemini(args.iterations, args.checkpoint_freq)

    # Log gemini to wandb
    log_to_wandb(
        {
            "gemini_single/avg_checkpoint_save_ms": gemini_single["avg_checkpoint_ms"],
            "gemini_single/ram_recovery_ms": gemini_single.get("ram_recovery_ms", 0),
            "gemini_single/throughput": gemini_single["throughput"],
            "gemini_single/memory_mb": gemini_single.get("memory_mb", 0),
            "gemini_single/total_time": gemini_single.get("total_time", 0),
        }
    )

    # CRITICAL: Clear GPU memory before multi-GPU experiments
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    print("\n🧹 Cleared GPU memory before multi-GPU experiments")

    # Experiment 3a: Multi-GPU DISK Baseline (for fair comparison)
    multi_disk = None
    if not args.skip_multi and torch.cuda.device_count() >= 2:
        num_gpus = min(args.gpus, torch.cuda.device_count())
        multi_disk = run_multi_gpu_disk_baseline(
            num_gpus, args.iterations, args.checkpoint_freq
        )
        
        # Log multi-GPU disk to wandb
        if multi_disk:
            log_to_wandb(
                {
                    "multi_gpu_disk/avg_checkpoint_save_ms": multi_disk.get("avg_checkpoint_ms", 0),
                    "multi_gpu_disk/disk_recovery_ms": multi_disk.get("measured_recovery_ms", 0),
                    "multi_gpu_disk/num_gpus": multi_disk.get("num_gpus", 0),
                    "multi_gpu_disk/total_time": multi_disk.get("total_time", 0),
                }
            )
        
        # Clear memory between experiments
        gc.collect()
        torch.cuda.empty_cache()

    # Experiment 3b: Multi-GPU Gemini IN-MEMORY with replication
    gemini_multi = None
    if not args.skip_multi and torch.cuda.device_count() >= 2:
        num_gpus = min(args.gpus, torch.cuda.device_count())
        gemini_multi = run_multi_gpu_gemini(
            num_gpus, args.iterations, args.checkpoint_freq
        )
        
        # Log multi-GPU RAM to wandb
        if gemini_multi:
            log_to_wandb(
                {
                    "multi_gpu_ram/avg_checkpoint_save_ms": gemini_multi.get("avg_checkpoint_ms", 0),
                    "multi_gpu_ram/ram_recovery_ms": gemini_multi.get("measured_recovery_ms", 0),
                    "multi_gpu_ram/num_gpus": gemini_multi.get("num_gpus", 0),
                    "multi_gpu_ram/total_time": gemini_multi.get("total_time", 0),
                }
            )

    # Experiment 4: Failure simulation
    failure_results = None
    if not args.skip_failure and torch.cuda.device_count() >= 2:
        num_gpus = min(args.gpus, torch.cuda.device_count())
        failure_results = run_failure_simulation(
            num_gpus=num_gpus,
            iterations=args.iterations + 50,
            checkpoint_freq=args.checkpoint_freq,
            failure_iteration=args.iterations // 2,
        )

    # Compare results
    comparison = compare_results(
        baseline, gemini_single, gemini_multi, multi_disk, failure_results
    )

    return {
        "baseline": baseline,
        "gemini_single": gemini_single,
        "gemini_multi": gemini_multi,
        "multi_disk": multi_disk,
        "failure_results": failure_results,
        "comparison": comparison,
    }


def main():
    global exp_logger

    parser = argparse.ArgumentParser(description="Multi-GPU Experiment Runner")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Training iterations"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=CHECK_POINT_FRQ,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--skip-multi", action="store_true", help="Skip multi-GPU experiment"
    )
    parser.add_argument(
        "--skip-failure", action="store_true", help="Skip failure simulation"
    )
    parser.add_argument(
        "--gpus", type=int, default=4, help="Number of GPUs for multi-GPU test"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of experiment runs to average"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GEMINI PROJECT - COMPREHENSIVE EXPERIMENTS")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    print(f"Number of runs: {args.num_runs}")
    print(f"wandb logging: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    # Check environment
    if not check_environment():
        print("Environment check failed!")
        return 1

    # Initialize wandb if requested
    if args.wandb:
        init_wandb(
            True,
            {
                "iterations": args.iterations,
                "checkpoint_freq": args.checkpoint_freq,
                "num_gpus": torch.cuda.device_count(),
            },
        )

    try:
        all_runs = []
        
        for run_idx in range(args.num_runs):
            if args.num_runs > 1:
                print("\n" + "=" * 60)
                print(f"🔄 RUN {run_idx + 1} OF {args.num_runs}")
                print("=" * 60)
            
            run_results = run_single_experiment(args)
            all_runs.append(run_results)
            
            # Log run number to wandb
            log_to_wandb({"run_number": run_idx + 1})
        
        # Aggregate results if multiple runs
        if args.num_runs > 1:
            print("\n" + "=" * 60)
            print("📊 AGGREGATING RESULTS FROM ALL RUNS")
            print("=" * 60)
            
            aggregated = aggregate_results(all_runs)
            
            # Print aggregated statistics
            print(f"\n📈 Aggregated Results ({args.num_runs} runs):")
            
            if "baseline" in aggregated:
                b = aggregated["baseline"]
                print(f"\n   Baseline (Disk):")
                if "avg_checkpoint_ms_mean" in b:
                    print(f"      Avg checkpoint: {b['avg_checkpoint_ms_mean']:.2f} ± {b['avg_checkpoint_ms_std']:.2f} ms")
                if "disk_recovery_ms_mean" in b:
                    print(f"      Disk recovery:  {b['disk_recovery_ms_mean']:.2f} ± {b['disk_recovery_ms_std']:.2f} ms")
                if "throughput_mean" in b:
                    print(f"      Throughput:     {b['throughput_mean']:.1f} ± {b['throughput_std']:.1f} samples/s")
            
            if "gemini_single" in aggregated:
                g = aggregated["gemini_single"]
                print(f"\n   Gemini (RAM):")
                if "avg_checkpoint_ms_mean" in g:
                    print(f"      Avg checkpoint: {g['avg_checkpoint_ms_mean']:.2f} ± {g['avg_checkpoint_ms_std']:.2f} ms")
                if "ram_recovery_ms_mean" in g:
                    print(f"      RAM recovery:   {g['ram_recovery_ms_mean']:.2f} ± {g['ram_recovery_ms_std']:.2f} ms")
                if "throughput_mean" in g:
                    print(f"      Throughput:     {g['throughput_mean']:.1f} ± {g['throughput_std']:.1f} samples/s")
            
            if "comparison" in aggregated:
                c = aggregated["comparison"]
                print(f"\n   📊 Comparison:")
                if "single_gpu_speedup_mean" in c:
                    print(f"      Checkpoint Speedup: {c['single_gpu_speedup_mean']:.1f}× ± {c['single_gpu_speedup_std']:.2f}")
                if "recovery_speedup_mean" in c:
                    print(f"      Recovery Speedup:   {c['recovery_speedup_mean']:.1f}× ± {c['recovery_speedup_std']:.2f}")
            
            # Log aggregated stats to wandb
            if exp_logger and "comparison" in aggregated:
                log_to_wandb({
                    "aggregated/num_runs": args.num_runs,
                    "aggregated/checkpoint_speedup_mean": aggregated["comparison"].get("single_gpu_speedup_mean", 0),
                    "aggregated/checkpoint_speedup_std": aggregated["comparison"].get("single_gpu_speedup_std", 0),
                    "aggregated/recovery_speedup_mean": aggregated["comparison"].get("recovery_speedup_mean", 0),
                    "aggregated/recovery_speedup_std": aggregated["comparison"].get("recovery_speedup_std", 0),
                })
            
            # Save aggregated results
            results_dir = Path("./results")
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            aggregated_file = results_dir / f"aggregated_results_{args.num_runs}runs_{timestamp}.json"
            with open(aggregated_file, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "num_runs": args.num_runs,
                    "config": {
                        "iterations": args.iterations,
                        "checkpoint_freq": args.checkpoint_freq,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                    },
                    "aggregated": aggregated,
                }, f, indent=2, default=str)
            
            print(f"\n💾 Aggregated results saved to: {aggregated_file}")
        else:
            # Single run - use existing save logic
            run = all_runs[0]
            save_final_results(
                run["baseline"], 
                run["gemini_single"], 
                run["gemini_multi"], 
                run["multi_disk"], 
                run["failure_results"], 
                run["comparison"]
            )

        # Finish wandb
        if exp_logger:
            exp_logger.finish()

        print("\n" + "=" * 60)
        print("✅ ALL EXPERIMENTS COMPLETE!")
        print("=" * 60)

        if args.wandb:
            print("\n📊 View results at: https://wandb.ai/your-username/gemini-cs240")

        return 0

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        if exp_logger:
            exp_logger.finish()
        return 1


if __name__ == "__main__":
    sys.exit(main())
