"""
Gemini Distributed Trainer with In-Memory Checkpointing

This trainer implements the Gemini approach:
1. Distributed training across multiple GPUs (each GPU acts as a "node")
2. In-memory checkpointing (fast save/load)
3. Checkpoint replication between GPUs for redundancy
4. Fast failure recovery from replicas

Key difference from Baseline:
- Baseline: Checkpoints to DISK (slow)
- Gemini: Checkpoints to RAM + replicas (fast!)

Author: Mustafa Albahrani, Mohammed Alkhalifah
Course: CS240 - Distributed Systems
"""

import os
import time
import json
import logging
import signal
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, asdict
import threading
import queue

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Import our components
from src.checkpointing.in_memory_checkpoint import InMemoryCheckpoint
from src.training.baseline_trainer import (
    SimpleLanguageModel, 
    TrainingConfig, 
    TrainingMetrics,
    SyntheticDataset
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig(TrainingConfig):
    """Extended configuration for Gemini trainer"""
    # Replication settings
    replication_factor: int = 2  # Number of replicas (m=2)
    async_replication: bool = True  # Replicate asynchronously
    
    # Failure simulation
    simulate_failure: bool = False
    failure_gpu: int = -1  # Which GPU to "fail" (-1 = random)
    failure_iteration: int = -1  # When to fail (-1 = random)


class CheckpointReplicator:
    """
    Manages checkpoint replication between GPUs.
    
    Each GPU keeps replicas of checkpoints from other GPUs.
    This enables fast recovery when a GPU "fails".
    """
    
    def __init__(self, rank: int, world_size: int, replication_factor: int = 2):
        self.rank = rank
        self.world_size = world_size
        self.replication_factor = replication_factor
        
        # Store replicas from other GPUs
        # Key: (source_rank, iteration), Value: checkpoint bytes
        self.replicas: Dict[Tuple[int, int], bytes] = {}
        
        # Track what we've replicated
        self.replication_log: List[Dict] = []
        
        logger.info(f"[GPU {rank}] CheckpointReplicator initialized with m={replication_factor}")
    
    def get_replica_targets(self, source_rank: int) -> List[int]:
        """
        Determine which GPUs should store replicas using ring strategy.
        
        For m=2 (2 copies total):
        - GPU 0's checkpoint → also stored on GPU 1
        - GPU 1's checkpoint → also stored on GPU 2
        - GPU 2's checkpoint → also stored on GPU 3
        - GPU 3's checkpoint → also stored on GPU 0
        """
        targets = []
        for i in range(1, self.replication_factor):
            target = (source_rank + i) % self.world_size
            targets.append(target)
        return targets
    
    def store_replica(self, source_rank: int, iteration: int, checkpoint_bytes: bytes):
        """Store a replica checkpoint from another GPU."""
        key = (source_rank, iteration)
        self.replicas[key] = checkpoint_bytes
        
        self.replication_log.append({
            'action': 'store_replica',
            'source': source_rank,
            'target': self.rank,
            'iteration': iteration,
            'size_bytes': len(checkpoint_bytes),
            'timestamp': time.time()
        })
        
        logger.debug(f"[GPU {self.rank}] Stored replica from GPU {source_rank} iter {iteration}")
    
    def get_replica(self, source_rank: int, iteration: int) -> Optional[bytes]:
        """Get a replica checkpoint for recovery."""
        key = (source_rank, iteration)
        return self.replicas.get(key)
    
    def has_replica(self, source_rank: int, iteration: int) -> bool:
        """Check if we have a replica."""
        return (source_rank, iteration) in self.replicas
    
    def cleanup_old_replicas(self, keep_iterations: List[int]):
        """Remove old replicas to free memory."""
        keys_to_remove = [
            key for key in self.replicas 
            if key[1] not in keep_iterations
        ]
        for key in keys_to_remove:
            del self.replicas[key]


class GeminiTrainer:
    """
    Gemini-style distributed trainer with in-memory checkpointing.
    
    Key Features:
    1. Each GPU acts as a "node" in a distributed system
    2. Checkpoints stored in RAM (not disk)
    3. Checkpoints replicated to peer GPUs for redundancy
    4. Fast recovery from peer when a GPU "fails"
    """
    
    def __init__(
        self,
        config: GeminiConfig,
        rank: int,
        world_size: int,
        local_rank: int
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        
        # In-memory checkpoint manager
        self.checkpoint_manager = InMemoryCheckpoint(
            node_id=f"gpu-{rank}",
            max_checkpoints=3,
            max_memory_gb=2.0  # Each GPU uses up to 2GB for checkpoints
        )
        
        # Checkpoint replicator
        self.replicator = CheckpointReplicator(
            rank=rank,
            world_size=world_size,
            replication_factor=config.replication_factor
        )
        
        # Metrics
        self.metrics_history: List[TrainingMetrics] = []
        self.checkpoint_times: List[Dict] = []
        self.recovery_events: List[Dict] = []
        
        # For failure simulation
        self.is_failed = False
        
        # Results directory
        self.results_dir = Path(config.checkpoint_dir) / "gemini"
        if rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[GPU {rank}] GeminiTrainer initialized on {self.device}")
    
    def setup_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size
            )
        logger.info(f"[GPU {self.rank}] Distributed setup complete")
    
    def create_model(self) -> nn.Module:
        """Create model wrapped in DDP."""
        model = SimpleLanguageModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_length=self.config.max_seq_length
        ).to(self.device)
        
        model = DDP(model, device_ids=[self.local_rank])
        logger.info(f"[GPU {self.rank}] Model created with DDP")
        return model
    
    def create_dataloader(self, dataset) -> DataLoader:
        """Create distributed data loader."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
    
    def save_checkpoint_gemini(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int
    ) -> Dict[str, float]:
        """
        Gemini-style checkpoint: Save to RAM + replicate to peers.
        
        Returns timing information for analysis.
        """
        timings = {}
        
        # Step 1: Save to local RAM (fast!)
        save_start = time.time()
        self.checkpoint_manager.save(model, optimizer, iteration)
        timings['local_save_ms'] = (time.time() - save_start) * 1000
        
        # Step 2: Get checkpoint bytes for replication
        checkpoint_bytes = self.checkpoint_manager.get_checkpoint_bytes(iteration)
        timings['checkpoint_size_mb'] = len(checkpoint_bytes) / (1024 * 1024)
        
        # Step 3: Replicate to peer GPUs
        replication_start = time.time()
        self._replicate_checkpoint(iteration, checkpoint_bytes)
        timings['replication_ms'] = (time.time() - replication_start) * 1000
        
        timings['total_ms'] = timings['local_save_ms'] + timings['replication_ms']
        
        # Log
        self.checkpoint_times.append({
            'iteration': iteration,
            'type': 'gemini',
            **timings,
            'timestamp': time.time()
        })
        
        logger.info(
            f"[GPU {self.rank}] Gemini checkpoint at iter {iteration}: "
            f"save={timings['local_save_ms']:.1f}ms, "
            f"replicate={timings['replication_ms']:.1f}ms, "
            f"size={timings['checkpoint_size_mb']:.2f}MB"
        )
        
        return timings
    
    def _replicate_checkpoint(self, iteration: int, checkpoint_bytes: bytes):
        """
        Replicate checkpoint to peer GPUs.
        
        Uses torch.distributed for communication between GPUs.
        """
        targets = self.replicator.get_replica_targets(self.rank)
        
        # For each target, we need to send our checkpoint
        # In a real implementation, this would use actual network transfer
        # Here we simulate by using distributed broadcast/gather
        
        # Synchronize all GPUs
        dist.barrier()
        
        # Each GPU broadcasts its checkpoint to others
        # Using all_gather to collect checkpoints from all GPUs
        checkpoint_tensor = torch.ByteTensor(list(checkpoint_bytes)).to(self.device)
        size_tensor = torch.tensor([len(checkpoint_bytes)], device=self.device)
        
        # Gather all sizes first
        all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, size_tensor)
        
        # Gather all checkpoints (padded to max size)
        max_size = max(s.item() for s in all_sizes)
        padded_checkpoint = torch.zeros(max_size, dtype=torch.uint8, device=self.device)
        padded_checkpoint[:len(checkpoint_bytes)] = checkpoint_tensor
        
        all_checkpoints = [torch.zeros(max_size, dtype=torch.uint8, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_checkpoints, padded_checkpoint)
        
        # Store replicas from GPUs we're responsible for backing up
        for source_rank in range(self.world_size):
            if source_rank == self.rank:
                continue  # Don't store our own
            
            # Check if we should store this GPU's replica
            their_targets = self.replicator.get_replica_targets(source_rank)
            if self.rank in their_targets:
                # Extract their checkpoint (up to their actual size)
                their_size = int(all_sizes[source_rank].item())
                their_bytes = bytes(all_checkpoints[source_rank][:their_size].cpu().tolist())
                self.replicator.store_replica(source_rank, iteration, their_bytes)
        
        dist.barrier()
    
    def recover_from_failure(
        self,
        failed_rank: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int
    ) -> Tuple[bool, float]:
        """
        Recover a failed GPU's state from replicas.
        
        Returns: (success, recovery_time_ms)
        """
        recovery_start = time.time()
        
        # Find who has the replica
        replica_holders = self.replicator.get_replica_targets(failed_rank)
        
        # Check if we have the replica
        if self.replicator.has_replica(failed_rank, iteration):
            # We have it! Use it.
            checkpoint_bytes = self.replicator.get_replica(failed_rank, iteration)
            
            # Load from bytes
            import io
            buffer = io.BytesIO(checkpoint_bytes)
            checkpoint = torch.load(buffer, map_location=self.device)
            
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            recovery_time = (time.time() - recovery_start) * 1000
            
            self.recovery_events.append({
                'failed_rank': failed_rank,
                'recovered_by': self.rank,
                'iteration': iteration,
                'recovery_time_ms': recovery_time,
                'timestamp': time.time()
            })
            
            logger.info(
                f"[GPU {self.rank}] Recovered GPU {failed_rank}'s state "
                f"from replica in {recovery_time:.1f}ms"
            )
            
            return True, recovery_time
        
        return False, 0.0
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step."""
        model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def train(self, dataset, resume: bool = False) -> Dict[str, Any]:
        """
        Main training loop with Gemini checkpointing.
        """
        model = self.create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        dataloader = self.create_dataloader(dataset)
        
        iteration = 0
        epoch = 0
        total_samples = 0
        start_time = time.time()
        
        logger.info(f"[GPU {self.rank}] Starting Gemini training")
        
        while iteration < self.config.max_iterations:
            epoch += 1
            dataloader.sampler.set_epoch(epoch)
            
            for batch in dataloader:
                if iteration >= self.config.max_iterations:
                    break
                
                # Check for simulated failure
                if (self.config.simulate_failure and 
                    self.rank == self.config.failure_gpu and
                    iteration == self.config.failure_iteration):
                    logger.warning(f"[GPU {self.rank}] SIMULATED FAILURE at iteration {iteration}!")
                    self.is_failed = True
                    # In real failure, process would die. Here we skip training.
                
                if self.is_failed:
                    # Skip training if "failed"
                    dist.barrier()  # Still sync with others
                    iteration += 1
                    continue
                
                step_start = time.time()
                loss = self.train_step(model, batch, optimizer)
                step_time = time.time() - step_start
                
                samples_per_sec = self.config.batch_size / step_time
                total_samples += self.config.batch_size
                
                # Logging
                if iteration % self.config.log_interval == 0 and self.rank == 0:
                    logger.info(
                        f"Iter {iteration}/{self.config.max_iterations} | "
                        f"Loss: {loss:.4f} | "
                        f"Throughput: {samples_per_sec:.1f} samples/s"
                    )
                
                # Gemini checkpoint (RAM + replication)
                checkpoint_timings = None
                if iteration > 0 and iteration % self.config.checkpoint_frequency == 0:
                    checkpoint_timings = self.save_checkpoint_gemini(model, optimizer, iteration)
                
                # Record metrics
                self.metrics_history.append(TrainingMetrics(
                    iteration=iteration,
                    loss=loss,
                    throughput=samples_per_sec,
                    checkpoint_time=checkpoint_timings['total_ms'] / 1000 if checkpoint_timings else 0,
                    recovery_time=0,
                    timestamp=time.time()
                ))
                
                iteration += 1
        
        total_time = time.time() - start_time
        
        # Gather results
        results = {
            'rank': self.rank,
            'total_iterations': iteration,
            'total_time': total_time,
            'total_samples': total_samples,
            'average_throughput': total_samples / total_time if total_time > 0 else 0,
            'checkpoint_times': self.checkpoint_times,
            'recovery_events': self.recovery_events,
            'final_loss': loss if not self.is_failed else None
        }
        
        # Save results
        if self.rank == 0:
            results_path = self.results_dir / "training_results.json"
            
            # Aggregate checkpoint times from all GPUs
            all_checkpoint_times = [None] * self.world_size
            dist.barrier()
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
        
        return results
    
    def cleanup(self):
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


def run_gemini_worker(rank: int, world_size: int, config: GeminiConfig):
    """Worker function for each GPU."""
    # Set environment for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Set the device for this process (rank maps to GPU index)
    torch.cuda.set_device(rank)
    
    trainer = GeminiTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=rank  # Each process uses its corresponding GPU
    )
    
    try:
        trainer.setup_distributed()
        
        dataset = SyntheticDataset(
            num_samples=10000,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        
        results = trainer.train(dataset)
        return results
        
    finally:
        trainer.cleanup()


def main():
    """Main entry point for multi-GPU Gemini training."""
    parser = argparse.ArgumentParser(description='Gemini Distributed Trainer')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--max-iterations', type=int, default=200, help='Max iterations')
    parser.add_argument('--checkpoint-freq', type=int, default=50, help='Checkpoint frequency')
    parser.add_argument('--hidden-size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--simulate-failure', action='store_true', help='Simulate GPU failure')
    parser.add_argument('--failure-gpu', type=int, default=1, help='Which GPU to fail')
    parser.add_argument('--failure-iteration', type=int, default=100, help='When to fail')
    
    args = parser.parse_args()
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < args.gpus:
        logger.warning(f"Requested {args.gpus} GPUs but only {num_gpus} available")
        args.gpus = num_gpus
    
    logger.info(f"Starting Gemini training on {args.gpus} GPUs")
    
    config = GeminiConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=8,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        checkpoint_frequency=args.checkpoint_freq,
        checkpoint_dir="./checkpoints",
        log_interval=10,
        replication_factor=2,
        simulate_failure=args.simulate_failure,
        failure_gpu=args.failure_gpu,
        failure_iteration=args.failure_iteration
    )
    
    # Launch multi-GPU training
    mp.spawn(
        run_gemini_worker,
        args=(args.gpus, config),
        nprocs=args.gpus,
        join=True
    )


if __name__ == "__main__":
    main()

