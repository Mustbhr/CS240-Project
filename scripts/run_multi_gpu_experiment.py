#!/usr/bin/env python3
"""
Multi-GPU Experiment Runner

This script runs comprehensive experiments comparing:
1. Baseline (disk) checkpointing
2. Gemini (in-memory) checkpointing

Run on your 4-GPU setup:
    python scripts/run_multi_gpu_experiment.py

This will:
- Test single-GPU baseline and Gemini checkpointing
- Test multi-GPU distributed training
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def run_single_gpu_baseline(iterations=100, checkpoint_freq=25):
    """Run baseline (disk) checkpointing on single GPU."""
    print("=" * 60)
    print("EXPERIMENT 1: Single-GPU Baseline (Disk Checkpointing)")
    print("=" * 60)
    
    from src.training import BaselineTrainer, TrainingConfig
    from src.utils import SyntheticDataset
    
    config = TrainingConfig(
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/baseline_single",
        log_interval=25
    )
    
    trainer = BaselineTrainer(
        config=config,
        rank=0,
        world_size=1,
        local_rank=0
    )
    
    dataset = SyntheticDataset(
        num_samples=5000,
        seq_length=256,
        vocab_size=config.vocab_size
    )
    
    start_time = time.time()
    results = trainer.train(dataset)
    total_time = time.time() - start_time
    
    # Extract checkpoint times
    checkpoint_times = [m.checkpoint_time * 1000 for m in trainer.metrics_history if m.checkpoint_time > 0]
    
    print(f"\n📊 Baseline Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Iterations: {results['total_iterations']}")
    print(f"   Avg throughput: {results['average_throughput']:.1f} samples/s")
    print(f"   Checkpoint times: {checkpoint_times}")
    if checkpoint_times:
        print(f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
    
    return {
        'type': 'baseline_single',
        'total_time': total_time,
        'iterations': results['total_iterations'],
        'throughput': results['average_throughput'],
        'checkpoint_times_ms': checkpoint_times,
        'avg_checkpoint_ms': sum(checkpoint_times)/len(checkpoint_times) if checkpoint_times else 0
    }


def run_single_gpu_gemini(iterations=100, checkpoint_freq=25):
    """Run Gemini (in-memory) checkpointing on single GPU."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Single-GPU Gemini (In-Memory Checkpointing)")
    print("=" * 60)
    
    from src.training import BaselineTrainer, TrainingConfig
    from src.checkpointing import InMemoryCheckpoint
    from src.utils import SyntheticDataset
    
    config = TrainingConfig(
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/gemini_single",
        log_interval=25
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
        max_seq_length=config.max_seq_length
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # In-memory checkpoint manager
    ckpt_manager = InMemoryCheckpoint(
        node_id="gpu-0",
        max_checkpoints=5,
        max_memory_gb=2.0
    )
    
    dataset = SyntheticDataset(
        num_samples=5000,
        seq_length=256,
        vocab_size=config.vocab_size
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
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
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Gemini checkpoint (in-memory)
            if iteration > 0 and iteration % checkpoint_freq == 0:
                ckpt_time = ckpt_manager.save(model, optimizer, iteration)
                checkpoint_times.append(ckpt_time * 1000)  # Convert to ms
                logger.info(f"Iter {iteration}: Gemini checkpoint in {ckpt_time*1000:.2f}ms")
            
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
        print(f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
    
    # Get memory stats
    stats = ckpt_manager.get_stats()
    print(f"   Memory used for checkpoints: {stats['current_memory_mb']:.2f}MB")
    
    return {
        'type': 'gemini_single',
        'total_time': total_time,
        'iterations': iteration,
        'throughput': total_samples / total_time,
        'checkpoint_times_ms': checkpoint_times,
        'avg_checkpoint_ms': sum(checkpoint_times)/len(checkpoint_times) if checkpoint_times else 0,
        'memory_mb': stats['current_memory_mb']
    }


def run_multi_gpu_gemini(num_gpus=4, iterations=100, checkpoint_freq=25):
    """Run multi-GPU Gemini training with replication."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 3: Multi-GPU Gemini ({num_gpus} GPUs)")
    print("=" * 60)
    
    import torch.multiprocessing as mp
    import torch.distributed as dist
    
    from src.training.gemini_trainer import GeminiConfig, run_gemini_worker
    
    config = GeminiConfig(
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        batch_size=8,
        max_iterations=iterations,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_dir="./checkpoints/gemini_multi",
        log_interval=25,
        replication_factor=2
    )
    
    # Run multi-GPU training
    start_time = time.time()
    mp.spawn(
        run_gemini_worker,
        args=(num_gpus, config),
        nprocs=num_gpus,
        join=True
    )
    total_time = time.time() - start_time
    
    # Load results
    results_path = Path("./checkpoints/gemini_multi/gemini/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        checkpoint_times = [c['total_ms'] for c in results.get('checkpoint_times', [])]
        
        print(f"\n📊 Multi-GPU Gemini Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   GPUs used: {num_gpus}")
        print(f"   Checkpoint times: {checkpoint_times}")
        if checkpoint_times:
            print(f"   Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.2f}ms")
        
        return {
            'type': 'gemini_multi',
            'total_time': total_time,
            'num_gpus': num_gpus,
            'checkpoint_times_ms': checkpoint_times,
            'avg_checkpoint_ms': sum(checkpoint_times)/len(checkpoint_times) if checkpoint_times else 0,
            'raw_results': results
        }
    else:
        print("⚠️ Results file not found")
        return {'type': 'gemini_multi', 'error': 'results not found'}


def compare_results(baseline, gemini_single, gemini_multi=None):
    """Compare and display results."""
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Gemini")
    print("=" * 60)
    
    if baseline['avg_checkpoint_ms'] > 0 and gemini_single['avg_checkpoint_ms'] > 0:
        speedup = baseline['avg_checkpoint_ms'] / gemini_single['avg_checkpoint_ms']
        
        print(f"\n📈 Checkpoint Performance:")
        print(f"   Baseline (Disk):    {baseline['avg_checkpoint_ms']:.2f} ms")
        print(f"   Gemini (Memory):    {gemini_single['avg_checkpoint_ms']:.2f} ms")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   SPEEDUP:            {speedup:.1f}× FASTER! 🚀")
        
        print(f"\n📊 Training Throughput:")
        print(f"   Baseline:           {baseline['throughput']:.1f} samples/s")
        print(f"   Gemini:             {gemini_single['throughput']:.1f} samples/s")
        
        if gemini_multi and 'avg_checkpoint_ms' in gemini_multi:
            print(f"\n🔄 Multi-GPU Replication:")
            print(f"   Checkpoint + Replication: {gemini_multi['avg_checkpoint_ms']:.2f} ms")
            print(f"   Still {baseline['avg_checkpoint_ms'] / gemini_multi['avg_checkpoint_ms']:.1f}× faster than disk!")
        
        return {
            'speedup': speedup,
            'baseline_ms': baseline['avg_checkpoint_ms'],
            'gemini_ms': gemini_single['avg_checkpoint_ms']
        }
    else:
        print("⚠️ Not enough data for comparison")
        return None


def save_final_results(baseline, gemini_single, gemini_multi, comparison):
    """Save all results for the report."""
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment_results_{timestamp}.json"
    
    all_results = {
        'timestamp': timestamp,
        'environment': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'num_gpus': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_properties(i).name 
                         for i in range(torch.cuda.device_count())]
        },
        'baseline': baseline,
        'gemini_single': gemini_single,
        'gemini_multi': gemini_multi,
        'comparison': comparison
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Experiment Runner')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--checkpoint-freq', type=int, default=25, help='Checkpoint frequency')
    parser.add_argument('--skip-multi', action='store_true', help='Skip multi-GPU experiment')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs for multi-GPU test')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("GEMINI PROJECT - COMPREHENSIVE EXPERIMENTS")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    print()
    
    # Check environment
    if not check_environment():
        print("Environment check failed!")
        return 1
    
    try:
        # Experiment 1: Single-GPU Baseline
        baseline = run_single_gpu_baseline(args.iterations, args.checkpoint_freq)
        
        # Experiment 2: Single-GPU Gemini
        gemini_single = run_single_gpu_gemini(args.iterations, args.checkpoint_freq)
        
        # Experiment 3: Multi-GPU Gemini (optional)
        gemini_multi = None
        if not args.skip_multi and torch.cuda.device_count() >= 2:
            num_gpus = min(args.gpus, torch.cuda.device_count())
            gemini_multi = run_multi_gpu_gemini(num_gpus, args.iterations, args.checkpoint_freq)
        
        # Compare results
        comparison = compare_results(baseline, gemini_single, gemini_multi)
        
        # Save everything
        save_final_results(baseline, gemini_single, gemini_multi, comparison)
        
        print("\n" + "=" * 60)
        print("✅ ALL EXPERIMENTS COMPLETE!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

