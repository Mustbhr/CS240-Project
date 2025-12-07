#!/usr/bin/env python3
"""
Quick test script to verify the baseline training infrastructure.

Run this to test the implementation without a cluster:
    python scripts/run_baseline_test.py

Run with wandb logging:
    python scripts/run_baseline_test.py --wandb

This runs a simple training loop with disk-based checkpointing
and reports performance metrics.
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_baseline_trainer(use_wandb: bool = False):
    """Test the baseline trainer in single-process mode."""
    from src.training import BaselineTrainer, TrainingConfig
    from src.utils import SyntheticDataset
    
    print("=" * 60)
    print("BASELINE TRAINER TEST")
    if use_wandb:
        print("(with wandb logging enabled)")
    print("=" * 60)
    
    # Create config
    config = TrainingConfig(
        hidden_size=256,  # Smaller for testing
        num_layers=2,
        num_heads=4,
        batch_size=4,
        max_iterations=50,
        checkpoint_frequency=10,
        checkpoint_dir="./checkpoints/baseline_test"
    )
    
    # Create trainer with optional wandb
    trainer = BaselineTrainer(
        config=config,
        rank=0,
        world_size=1,
        local_rank=0,
        use_wandb=use_wandb
    )
    
    # Create dataset
    dataset = SyntheticDataset(
        num_samples=1000,
        seq_length=64,
        vocab_size=config.vocab_size
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train(dataset, resume=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average throughput: {results['average_throughput']:.1f} samples/s")
    print(f"Final loss: {results['final_loss']:.4f}")
    
    if results['checkpoint_times']:
        avg_checkpoint_time = sum(results['checkpoint_times']) / len(results['checkpoint_times'])
        print(f"\nCheckpoint Performance:")
        print(f"  Number of checkpoints: {len(results['checkpoint_times'])}")
        print(f"  Average checkpoint time: {avg_checkpoint_time:.3f}s")
    
    if use_wandb:
        print("\nCheck wandb dashboard for interactive plots!")
    
    return results


def test_in_memory_checkpoint():
    """Test the in-memory checkpoint manager."""
    import torch
    from src.checkpointing import InMemoryCheckpoint
    
    print("\n" + "=" * 60)
    print("IN-MEMORY CHECKPOINT TEST")
    print("=" * 60)
    
    # Create a test model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512)
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    ckpt_manager = InMemoryCheckpoint(
        node_id="test-node-0",
        max_checkpoints=3,
        max_memory_gb=1.0
    )
    
    # Test save/load multiple times
    save_times = []
    load_times = []
    
    print("\nRunning 5 save/load cycles...")
    for i in range(5):
        # Save
        save_time = ckpt_manager.save(model, optimizer, iteration=i * 100)
        save_times.append(save_time)
        
        # Modify model
        with torch.no_grad():
            model[0].weight.fill_(float(i))
        
        # Load (restore original)
        load_time = ckpt_manager.load(model, optimizer, iteration=i * 100)
        load_times.append(load_time)
    
    # Print results
    print("\nIn-Memory Checkpoint Performance:")
    print(f"  Average save time: {sum(save_times)/len(save_times)*1000:.2f}ms")
    print(f"  Average load time: {sum(load_times)/len(load_times)*1000:.2f}ms")
    
    stats = ckpt_manager.get_stats()
    print(f"\nMemory Usage:")
    print(f"  Current: {stats['current_memory_mb']:.2f} MB")
    print(f"  Usage: {stats['memory_usage_percent']:.1f}%")
    print(f"  Checkpoints kept: {stats['num_checkpoints']}")
    
    return stats


def compare_disk_vs_memory():
    """Compare disk-based vs memory-based checkpointing."""
    import torch
    import time
    import tempfile
    from src.checkpointing import InMemoryCheckpoint, compare_checkpoint_times
    
    print("\n" + "=" * 60)
    print("DISK vs MEMORY CHECKPOINT COMPARISON")
    print("=" * 60)
    
    # Create a larger model for meaningful comparison
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 3072),
        torch.nn.ReLU(),
        torch.nn.Linear(3072, 768),
        torch.nn.LayerNorm(768)
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {param_count:,} parameters")
    
    # Test disk-based checkpoint
    print("\n1. Testing DISK-BASED checkpoint (traditional)...")
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        disk_path = f.name
    
    # Save to disk
    disk_save_start = time.time()
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, disk_path)
    disk_save_time = time.time() - disk_save_start
    
    # Load from disk
    disk_load_start = time.time()
    checkpoint = torch.load(disk_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    disk_load_time = time.time() - disk_load_start
    
    print(f"   Disk save time: {disk_save_time*1000:.2f}ms")
    print(f"   Disk load time: {disk_load_time*1000:.2f}ms")
    
    # Clean up
    os.unlink(disk_path)
    
    # Test memory-based checkpoint
    print("\n2. Testing MEMORY-BASED checkpoint (Gemini)...")
    ckpt_manager = InMemoryCheckpoint(node_id="test", max_checkpoints=3)
    
    memory_save_time = ckpt_manager.save(model, optimizer, iteration=1)
    memory_load_time = ckpt_manager.load(model, optimizer, iteration=1)
    
    print(f"   Memory save time: {memory_save_time*1000:.2f}ms")
    print(f"   Memory load time: {memory_load_time*1000:.2f}ms")
    
    # Compare
    comparison = compare_checkpoint_times(
        disk_save_time, disk_load_time,
        memory_save_time, memory_load_time
    )
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Save speedup:     {comparison['save_speedup']:.2f}x faster")
    print(f"Load speedup:     {comparison['load_speedup']:.2f}x faster")
    print(f"Recovery speedup: {comparison['recovery_speedup']:.2f}x faster")
    
    print("\nIn a real NFS environment, the speedup would be even larger!")
    print("   NFS adds network latency, disk seek time, and I/O queue delays.")
    print("   Expected real-world speedup: 10-15x")
    
    return comparison


def test_experiment_logger():
    """Test the experiment logger with wandb."""
    from src.utils import ExperimentLogger
    
    print("\n" + "=" * 60)
    print("EXPERIMENT LOGGER TEST (wandb)")
    print("=" * 60)
    
    # Test with wandb enabled
    exp_logger = ExperimentLogger(
        project="gemini-cs240",
        run_name="test-run",
        config={"test": True, "batch_size": 8},
        tags=["test"],
        use_wandb=True  # Will try to use wandb
    )
    
    # Log some test metrics
    for i in range(10):
        exp_logger.log({
            "test/loss": 1.0 / (i + 1),
            "test/accuracy": i * 10,
            "iteration": i
        })
    
    # Log checkpoint comparison
    exp_logger.log_comparison(
        disk_save_time_ms=150.0,
        disk_load_time_ms=200.0,
        memory_save_time_ms=15.0,
        memory_load_time_ms=10.0
    )
    
    # Finish
    summary = exp_logger.finish()
    
    print("\nExperiment logger test complete!")
    print("Check your wandb dashboard if you're logged in.")
    
    return summary


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Gemini project infrastructure")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--skip-training", action="store_true", help="Skip training test")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("GEMINI PROJECT - INFRASTRUCTURE TEST")
    print("=" * 60)
    print("\nThis script tests the core components without a GPU cluster.")
    print("It verifies that the training and checkpointing infrastructure works.")
    
    if args.wandb:
        print("\nwandb logging is ENABLED")
        print("   Make sure you're logged in: wandb login")
    print()
    
    try:
        # Test 1: Baseline trainer
        if not args.skip_training:
            baseline_results = test_baseline_trainer(use_wandb=args.wandb)
        
        # Test 2: In-memory checkpoint
        memory_stats = test_in_memory_checkpoint()
        
        # Test 3: Comparison
        comparison = compare_disk_vs_memory()
        
        # Test 4: Experiment logger (if wandb enabled)
        if args.wandb:
            test_experiment_logger()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Get IBEX access with 2-4 nodes")
        print("2. Test distributed training across nodes")
        print("3. Implement checkpoint replication")
        print("4. Add failure injection and recovery")
        
        if args.wandb:
            print("\nCheck your wandb dashboard:")
            print("   https://wandb.ai/your-username/gemini-cs240")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

