"""
Worker Agent - Handles local checkpoint capture and replication

Responsibilities:
- Capture model state during training
- Manage in-memory checkpoint storage
- Replicate checkpoint shards to peer nodes
- Serve checkpoints during recovery
"""

import torch
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import time


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    iteration: int
    timestamp: float
    model_size_bytes: int
    node_id: str
    replica_nodes: List[str]


class WorkerAgent:
    """
    Worker agent running on each training node.
    Manages local checkpointing and replication.
    """
    
    def __init__(self, node_id: str, memory_limit_gb: float = 10.0):
        """
        Initialize worker agent.
        
        Args:
            node_id: Unique identifier for this node
            memory_limit_gb: Maximum memory for checkpoints (GB)
        """
        self.node_id = node_id
        self.memory_limit_gb = memory_limit_gb
        self.checkpoints = {}  # iteration -> checkpoint data
        self.metadata = {}  # iteration -> CheckpointMetadata
        self.logger = logging.getLogger(f"WorkerAgent-{node_id}")
        
        self.logger.info(f"Initialized WorkerAgent {node_id} with {memory_limit_gb}GB limit")
    
    def capture_checkpoint(self, model: torch.nn.Module, optimizer, iteration: int) -> bool:
        """
        Capture model and optimizer state in RAM.
        
        Args:
            model: PyTorch model to checkpoint
            optimizer: Optimizer to checkpoint
            iteration: Current training iteration
            
        Returns:
            True if checkpoint captured successfully
        """
        try:
            start_time = time.time()
            
            # TODO: Implement efficient in-memory checkpoint capture
            # For now, just store state_dict
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration
            }
            
            # Calculate size
            # TODO: Implement accurate size calculation
            size_bytes = 0
            
            # Store checkpoint
            self.checkpoints[iteration] = checkpoint
            self.metadata[iteration] = CheckpointMetadata(
                iteration=iteration,
                timestamp=time.time(),
                model_size_bytes=size_bytes,
                node_id=self.node_id,
                replica_nodes=[]
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Captured checkpoint at iteration {iteration} in {elapsed:.3f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to capture checkpoint: {e}")
            return False
    
    def replicate_shard(self, iteration: int, target_node: str) -> bool:
        """
        Replicate checkpoint shard to target node.
        
        Args:
            iteration: Checkpoint iteration to replicate
            target_node: Target node ID
            
        Returns:
            True if replication successful
        """
        # TODO: Implement network transfer to peer node
        self.logger.info(f"Replicating checkpoint {iteration} to {target_node}")
        return True
    
    def serve_checkpoint(self, iteration: int) -> Optional[Dict]:
        """
        Serve checkpoint data for recovery.
        
        Args:
            iteration: Checkpoint iteration to serve
            
        Returns:
            Checkpoint data or None if not available
        """
        return self.checkpoints.get(iteration)
    
    def load_checkpoint(self, iteration: int, model: torch.nn.Module, optimizer) -> bool:
        """
        Load checkpoint into model and optimizer.
        
        Args:
            iteration: Checkpoint iteration to load
            model: Model to load state into
            optimizer: Optimizer to load state into
            
        Returns:
            True if load successful
        """
        try:
            checkpoint = self.checkpoints.get(iteration)
            if checkpoint is None:
                self.logger.error(f"Checkpoint {iteration} not found")
                return False
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info(f"Loaded checkpoint from iteration {iteration}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints to free memory.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        if len(self.checkpoints) <= keep_last_n:
            return
        
        iterations = sorted(self.checkpoints.keys())
        to_remove = iterations[:-keep_last_n]
        
        for iteration in to_remove:
            del self.checkpoints[iteration]
            del self.metadata[iteration]
        
        self.logger.info(f"Cleaned up {len(to_remove)} old checkpoints")

