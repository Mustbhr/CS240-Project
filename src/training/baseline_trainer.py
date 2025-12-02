"""
Baseline Distributed Trainer with NFS Checkpointing

This trainer implements traditional distributed training with disk-based
checkpointing. It serves as a baseline to compare against Gemini's 
in-memory checkpointing approach.

Key Components:
1. Distributed training using PyTorch DDP (DistributedDataParallel)
2. Traditional checkpointing to disk (simulating NFS)
3. Recovery from disk-based checkpoints
4. Metrics collection for comparison

Author: Mustafa Albahrani, Mohammed Alkhalifa
Course: CS240 - Distributed Systems
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Stores training metrics for analysis"""
    iteration: int
    loss: float
    throughput: float  # samples per second
    checkpoint_time: float  # time to save checkpoint
    recovery_time: float  # time to load checkpoint (if applicable)
    timestamp: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model settings
    model_name: str = "simple_transformer"
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_seq_length: int = 512
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_iterations: int = 1000
    warmup_steps: int = 100
    
    # Checkpointing settings
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 100  # Save every N iterations
    
    # Distributed settings
    world_size: int = 4
    
    # Logging
    log_interval: int = 10


class SimpleTransformerBlock(nn.Module):
    """
    A simplified transformer block for training.
    
    This is a minimal implementation - in production you'd use
    a full model like GPT-2 from Hugging Face transformers.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimpleLanguageModel(nn.Module):
    """
    A simple transformer-based language model for testing.
    
    Architecture:
    - Token embedding
    - Position embedding
    - N transformer blocks
    - Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying (common practice in language models)
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Created model with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device),
            diagonal=1
        ).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        # Output projection
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for initial testing.
    
    Generates random token sequences. This allows us to test the training
    infrastructure without needing to download and process real data.
    
    Later, this will be replaced with real Wikipedia data.
    """
    
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


class BaselineTrainer:
    """
    Baseline distributed trainer with disk-based checkpointing.
    
    This class demonstrates traditional distributed training where:
    1. Training runs across multiple GPUs/nodes using DDP
    2. Checkpoints are saved to disk (simulating NFS)
    3. Recovery loads checkpoint from disk
    
    This serves as our baseline to compare against Gemini.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        local_rank: int = 0
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            rank: Global rank of this process (0 to world_size-1)
            world_size: Total number of processes
            local_rank: GPU index on this node (for multi-GPU nodes)
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Metrics storage
        self.metrics_history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[Rank {rank}] Initialized trainer on device: {self.device}")
    
    def setup_distributed(self):
        """
        Initialize distributed training environment.
        
        This sets up the process group for communication between nodes.
        In production, this would use NCCL backend for GPU communication.
        """
        if not dist.is_initialized():
            # For local testing, use gloo backend (works without NCCL)
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size
            )
            logger.info(f"[Rank {self.rank}] Distributed setup complete with {backend} backend")
    
    def create_model(self) -> nn.Module:
        """
        Create and wrap the model for distributed training.
        
        Returns:
            Model wrapped in DistributedDataParallel
        """
        model = SimpleLanguageModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_length=self.config.max_seq_length
        ).to(self.device)
        
        # Wrap with DDP for distributed training
        if dist.is_initialized():
            model = DDP(model, device_ids=[self.local_rank])
            logger.info(f"[Rank {self.rank}] Model wrapped with DDP")
        
        return model
    
    def create_dataloader(self, dataset: torch.utils.data.Dataset) -> DataLoader:
        """
        Create a distributed data loader.
        
        The DistributedSampler ensures each process gets different data.
        """
        sampler = None
        if dist.is_initialized():
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
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int
    ) -> float:
        """
        Save checkpoint to disk (simulating NFS).
        
        This is the BASELINE approach - saves everything to disk.
        We measure how long this takes for comparison.
        
        Args:
            model: The model to checkpoint
            optimizer: The optimizer state
            iteration: Current training iteration
            
        Returns:
            Time taken to save checkpoint (seconds)
        """
        # Only rank 0 saves to avoid conflicts
        if self.rank != 0:
            # Barrier to ensure all ranks wait for rank 0
            if dist.is_initialized():
                dist.barrier()
            return 0.0
        
        start_time = time.time()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pt"
        
        # Get the actual model (unwrap from DDP if needed)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(self.config),
            'timestamp': time.time()
        }
        
        # Save to disk (this is what we're measuring!)
        torch.save(checkpoint, checkpoint_path)
        
        # Also keep a "latest" pointer
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        elapsed = time.time() - start_time
        
        # Get checkpoint size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        logger.info(
            f"[Rank {self.rank}] Saved checkpoint at iteration {iteration} "
            f"({size_mb:.2f} MB) in {elapsed:.3f}s"
        )
        
        # Barrier to ensure all ranks wait
        if dist.is_initialized():
            dist.barrier()
        
        return elapsed
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Load checkpoint from disk (simulating NFS recovery).
        
        This is the BASELINE recovery approach - loads from disk.
        We measure how long this takes for comparison with Gemini.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Tuple of (iteration to resume from, time taken to load)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return 0, 0.0
        
        start_time = time.time()
        
        # Load from disk (this is what we're measuring!)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get the actual model (unwrap from DDP if needed)
        model_to_load = model.module if hasattr(model, 'module') else model
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        iteration = checkpoint['iteration']
        elapsed = time.time() - start_time
        
        logger.info(
            f"[Rank {self.rank}] Loaded checkpoint from iteration {iteration} "
            f"in {elapsed:.3f}s"
        )
        
        return iteration, elapsed
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform a single training step.
        
        Args:
            model: The model
            batch: Batch of data
            optimizer: The optimizer
            
        Returns:
            Loss value
        """
        model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss (cross-entropy for language modeling)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (common practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        This runs the training with periodic checkpointing to disk.
        
        Args:
            dataset: Training dataset
            resume: Whether to resume from latest checkpoint
            
        Returns:
            Training results and metrics
        """
        # Create model and optimizer
        model = self.create_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Create data loader
        dataloader = self.create_dataloader(dataset)
        
        # Resume from checkpoint if requested
        start_iteration = 0
        total_recovery_time = 0.0
        if resume:
            start_iteration, recovery_time = self.load_checkpoint(model, optimizer)
            total_recovery_time = recovery_time
        
        # Training loop
        iteration = start_iteration
        total_samples = 0
        epoch = 0
        training_start_time = time.time()
        
        logger.info(f"[Rank {self.rank}] Starting training from iteration {iteration}")
        
        while iteration < self.config.max_iterations:
            epoch += 1
            
            # Update sampler for new epoch (important for proper shuffling)
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            
            for batch in dataloader:
                if iteration >= self.config.max_iterations:
                    break
                
                step_start_time = time.time()
                
                # Training step
                loss = self.train_step(model, batch, optimizer)
                
                step_time = time.time() - step_start_time
                samples_per_sec = self.config.batch_size / step_time
                total_samples += self.config.batch_size
                
                # Logging
                if iteration % self.config.log_interval == 0 and self.rank == 0:
                    logger.info(
                        f"Iteration {iteration}/{self.config.max_iterations} | "
                        f"Loss: {loss:.4f} | "
                        f"Throughput: {samples_per_sec:.1f} samples/s"
                    )
                
                # Checkpointing (to disk - this is our baseline!)
                checkpoint_time = 0.0
                if iteration > 0 and iteration % self.config.checkpoint_frequency == 0:
                    checkpoint_time = self.save_checkpoint(model, optimizer, iteration)
                
                # Record metrics
                metric = TrainingMetrics(
                    iteration=iteration,
                    loss=loss,
                    throughput=samples_per_sec,
                    checkpoint_time=checkpoint_time,
                    recovery_time=total_recovery_time if iteration == start_iteration else 0.0,
                    timestamp=time.time()
                )
                self.metrics_history.append(metric)
                
                iteration += 1
        
        total_time = time.time() - training_start_time
        
        # Final results
        results = {
            'total_iterations': iteration,
            'total_time': total_time,
            'total_samples': total_samples * self.world_size,  # Account for all ranks
            'average_throughput': (total_samples * self.world_size) / total_time,
            'final_loss': loss,
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'checkpoint_times': [m.checkpoint_time for m in self.metrics_history if m.checkpoint_time > 0],
            'recovery_time': total_recovery_time
        }
        
        if self.rank == 0:
            logger.info(f"Training complete! Results: {results}")
            
            # Save results
            results_path = self.checkpoint_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"[Rank {self.rank}] Cleaned up distributed resources")


def run_training(rank: int, world_size: int, config: TrainingConfig):
    """
    Function to run training on each process.
    
    This is called by torch.multiprocessing.spawn for each rank.
    """
    # Set environment for distributed training
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    trainer = BaselineTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    
    try:
        trainer.setup_distributed()
        
        # Create synthetic dataset for testing
        dataset = SyntheticDataset(
            num_samples=10000,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        
        # Train
        results = trainer.train(dataset, resume=False)
        
        return results
        
    finally:
        trainer.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Baseline Distributed Trainer')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=1, help='GPUs per node')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--checkpoint-freq', type=int, default=100, help='Checkpoint frequency')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/baseline', help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        checkpoint_frequency=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        world_size=args.nodes * args.gpus_per_node
    )
    
    world_size = args.nodes * args.gpus_per_node
    
    if world_size > 1 and torch.cuda.is_available():
        # Multi-GPU training
        import torch.multiprocessing as mp
        mp.spawn(
            run_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process training (for testing)
        logger.info("Running in single-process mode (no distributed)")
        
        trainer = BaselineTrainer(
            config=config,
            rank=0,
            world_size=1,
            local_rank=0
        )
        
        dataset = SyntheticDataset(
            num_samples=10000,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        
        results = trainer.train(dataset, resume=args.resume)
        print(f"\nTraining Results:\n{json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()

