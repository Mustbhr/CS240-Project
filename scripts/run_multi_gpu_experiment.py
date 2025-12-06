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

    print(f"\n📊 Baseline Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Iterations: {results['total_iterations']}")
    print(f"   Avg throughput: {results['average_throughput']:.1f} samples/s")
    print(f"   Checkpoint times: {checkpoint_times}")
    if checkpoint_times:
        print(
            f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms"
        )

    return {
        "type": "baseline_single",
        "total_time": total_time,
        "iterations": results["total_iterations"],
        "throughput": results["average_throughput"],
        "checkpoint_times_ms": checkpoint_times,
        "avg_checkpoint_ms": (
            sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
        ),
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

    print(f"\n📊 Gemini Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Iterations: {iteration}")
    print(f"   Avg throughput: {total_samples/total_time:.1f} samples/s")
    print(f"   Checkpoint times: {checkpoint_times}")
    if checkpoint_times:
        print(
            f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms"
        )

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
        "memory_mb": stats["current_memory_mb"],
    }


def run_multi_gpu_gemini(num_gpus=4, iterations=100, checkpoint_freq=CHECK_POINT_FRQ):
    """Run multi-GPU Gemini training with replication."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3: Multi-GPU Gemini ({num_gpus} GPUs)")
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

        print(f"\n📊 Multi-GPU Gemini Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   GPUs used: {num_gpus}")
        print(f"   Checkpoint times: {checkpoint_times}")
        if checkpoint_times:
            print(
                f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms"
            )

        # Log to wandb
        log_to_wandb(
            {
                "multi_gpu/total_time": total_time,
                "multi_gpu/num_gpus": num_gpus,
                "multi_gpu/avg_checkpoint_ms": (
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


def compare_results(baseline, gemini_single, gemini_multi=None, failure_results=None):
    """Compare and display results."""
    global exp_logger

    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Gemini")
    print("=" * 60)

    comparison = {}

    if baseline["avg_checkpoint_ms"] > 0 and gemini_single["avg_checkpoint_ms"] > 0:
        speedup = baseline["avg_checkpoint_ms"] / gemini_single["avg_checkpoint_ms"]

        print(f"\n📈 Checkpoint Performance:")
        print(f"   Baseline (Disk):    {baseline['avg_checkpoint_ms']:.2f} ms")
        print(f"   Gemini (Memory):    {gemini_single['avg_checkpoint_ms']:.2f} ms")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   SPEEDUP:            {speedup:.1f}× FASTER! 🚀")

        print(f"\n📊 Training Throughput:")
        print(f"   Baseline:           {baseline['throughput']:.1f} samples/s")
        print(f"   Gemini:             {gemini_single['throughput']:.1f} samples/s")

        comparison = {
            "speedup": speedup,
            "baseline_ms": baseline["avg_checkpoint_ms"],
            "gemini_ms": gemini_single["avg_checkpoint_ms"],
            "baseline_throughput": baseline["throughput"],
            "gemini_throughput": gemini_single["throughput"],
        }

        if (
            gemini_multi
            and "avg_checkpoint_ms" in gemini_multi
            and gemini_multi["avg_checkpoint_ms"] > 0
        ):
            multi_speedup = (
                baseline["avg_checkpoint_ms"] / gemini_multi["avg_checkpoint_ms"]
            )
            print(f"\n🔄 Multi-GPU Replication:")
            print(
                f"   Checkpoint + Replication: {gemini_multi['avg_checkpoint_ms']:.2f} ms"
            )
            print(f"   Still {multi_speedup:.1f}× faster than disk!")
            comparison["multi_gpu_ms"] = gemini_multi["avg_checkpoint_ms"]
            comparison["multi_gpu_speedup"] = multi_speedup

        if failure_results and "recovery_events" in failure_results:
            events = failure_results.get("recovery_events", [])
            if events:
                recovery_time = events[0].get("recovery_time_ms", 0)
                print(f"\n🔧 Failure Recovery:")
                print(f"   Recovery from replica: {recovery_time:.2f} ms")
                print(
                    f"   vs Disk recovery: ~{baseline['avg_checkpoint_ms'] * 2:.0f}+ ms (load + network)"
                )
                comparison["recovery_time_ms"] = recovery_time

        # Log comparison to wandb
        if exp_logger:
            exp_logger.log_comparison(
                disk_save_time_ms=baseline["avg_checkpoint_ms"],
                disk_load_time_ms=baseline["avg_checkpoint_ms"] * 0.7,  # Estimate
                memory_save_time_ms=gemini_single["avg_checkpoint_ms"],
                memory_load_time_ms=gemini_single["avg_checkpoint_ms"]
                * 0.5,  # Estimate
            )

        return comparison
    else:
        print("⚠️ Not enough data for comparison")
        return None


def save_final_results(
    baseline, gemini_single, gemini_multi, failure_results, comparison
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
            "baseline": baseline,
            "gemini_single": gemini_single,
            "gemini_multi": gemini_multi,
            "failure_simulation": failure_results,
        },
        "comparison": comparison,
    }

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")
    return results_file


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
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GEMINI PROJECT - COMPREHENSIVE EXPERIMENTS")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
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
        # Experiment 1: Single-GPU Baseline (Disk)
        baseline = run_single_gpu_baseline(args.iterations, args.checkpoint_freq)

        # Log baseline to wandb
        log_to_wandb(
            {
                "baseline/avg_checkpoint_ms": baseline["avg_checkpoint_ms"],
                "baseline/throughput": baseline["throughput"],
            }
        )

        # Clear GPU memory after baseline
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Experiment 2: Single-GPU Gemini (Memory)
        gemini_single = run_single_gpu_gemini(args.iterations, args.checkpoint_freq)

        # Log gemini to wandb
        log_to_wandb(
            {
                "gemini/avg_checkpoint_ms": gemini_single["avg_checkpoint_ms"],
                "gemini/throughput": gemini_single["throughput"],
                "gemini/memory_mb": gemini_single.get("memory_mb", 0),
            }
        )

        # CRITICAL: Clear GPU memory before multi-GPU experiments
        # The single-GPU tests may have left memory allocated
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        print("\n🧹 Cleared GPU memory before multi-GPU experiments")

        # Experiment 3: Multi-GPU Gemini with replication
        gemini_multi = None
        if not args.skip_multi and torch.cuda.device_count() >= 2:
            num_gpus = min(args.gpus, torch.cuda.device_count())
            gemini_multi = run_multi_gpu_gemini(
                num_gpus, args.iterations, args.checkpoint_freq
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
            baseline, gemini_single, gemini_multi, failure_results
        )

        # Save everything
        save_final_results(
            baseline, gemini_single, gemini_multi, failure_results, comparison
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
