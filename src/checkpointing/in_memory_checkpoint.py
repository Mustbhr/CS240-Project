"""
In-Memory Checkpoint Manager

This is the CORE of the Gemini system. Instead of saving checkpoints to disk,
we store them in RAM for near-instant recovery.

Key Features:
1. Store model state in RAM (not disk)
2. Track checkpoint metadata
3. Memory-efficient storage with optional compression
4. Fast serialization/deserialization

How it differs from traditional checkpointing:
- Traditional: torch.save(model, "/nfs/checkpoint.pt")  → Disk I/O (slow)
- Gemini:      checkpoint.save(model)                   → RAM (fast!)

Author: Mustafa Albahrani, Mohammed Alkhalifa
Course: CS240 - Distributed Systems
"""

import io
import time
import logging
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointEntry:
    """
    Represents a single checkpoint stored in memory.
    
    Stores the serialized checkpoint data and metadata for tracking.
    """
    iteration: int
    data: bytes  # Serialized checkpoint (model + optimizer state)
    size_bytes: int
    timestamp: float
    node_id: str
    is_replicated: bool = False
    replica_nodes: List[str] = field(default_factory=list)
    
    def get_size_mb(self) -> float:
        """Return size in megabytes"""
        return self.size_bytes / (1024 * 1024)


class InMemoryCheckpoint:
    """
    In-Memory Checkpoint Manager - Core Gemini Component
    
    This class manages checkpoints stored entirely in RAM instead of on disk.
    This enables near-instant checkpoint save/load operations.
    
    Key Design Decisions:
    1. Uses Python bytes for storage (serialized with torch.save to BytesIO)
    2. Keeps limited history to avoid OOM (configurable max_checkpoints)
    3. Thread-safe for concurrent access
    4. Tracks memory usage and provides stats
    
    Memory Overhead Analysis:
    - Checkpoint size ≈ Model size × 2 (model + optimizer states)
    - With 3 checkpoints: ~6× model size
    - For a 124M param model (~500MB): ~3GB RAM needed
    - Target: <10% of total RAM
    
    Performance Targets:
    - Save time: <1 second (vs. 10+ seconds for disk)
    - Load time: <0.5 seconds (vs. 30+ seconds for disk)
    """
    
    def __init__(
        self,
        node_id: str,
        max_checkpoints: int = 3,
        max_memory_gb: float = 10.0,
        compression: bool = False
    ):
        """
        Initialize the in-memory checkpoint manager.
        
        Args:
            node_id: Unique identifier for this node
            max_checkpoints: Maximum number of checkpoints to keep
            max_memory_gb: Maximum memory to use for checkpoints (GB)
            compression: Whether to compress checkpoints (trades CPU for memory)
        """
        self.node_id = node_id
        self.max_checkpoints = max_checkpoints
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.compression = compression
        
        # Checkpoint storage: OrderedDict to maintain insertion order
        self._checkpoints: OrderedDict[int, CheckpointEntry] = OrderedDict()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics tracking
        self._stats = {
            'total_saves': 0,
            'total_loads': 0,
            'total_save_time': 0.0,
            'total_load_time': 0.0,
            'current_memory_bytes': 0
        }
        
        logger.info(
            f"InMemoryCheckpoint initialized on {node_id}: "
            f"max_checkpoints={max_checkpoints}, max_memory={max_memory_gb}GB"
        )
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Save a checkpoint to RAM.
        
        This is the FAST path - instead of writing to disk, we serialize
        the state to bytes and keep it in memory.
        
        Args:
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint
            iteration: Current training iteration
            extra_data: Additional data to include (optional)
            
        Returns:
            Time taken to save (seconds)
        """
        start_time = time.time()
        
        # Handle DDP-wrapped models
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Build checkpoint dict
        checkpoint_data = {
            'iteration': iteration,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': time.time(),
            'node_id': self.node_id
        }
        if extra_data:
            checkpoint_data['extra'] = extra_data
        
        # Serialize to bytes (key difference from disk-based!)
        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        serialized_data = buffer.getvalue()
        
        # Optional compression
        if self.compression:
            import zlib
            serialized_data = zlib.compress(serialized_data, level=1)  # Fast compression
        
        size_bytes = len(serialized_data)
        
        # Create checkpoint entry
        entry = CheckpointEntry(
            iteration=iteration,
            data=serialized_data,
            size_bytes=size_bytes,
            timestamp=time.time(),
            node_id=self.node_id
        )
        
        # Thread-safe storage
        with self._lock:
            # Check memory limit
            total_memory = self._stats['current_memory_bytes'] + size_bytes
            if total_memory > self.max_memory_bytes:
                self._evict_oldest()
            
            # Store the checkpoint
            self._checkpoints[iteration] = entry
            self._stats['current_memory_bytes'] += size_bytes
            
            # Enforce max checkpoints limit
            while len(self._checkpoints) > self.max_checkpoints:
                self._evict_oldest()
        
        elapsed = time.time() - start_time
        
        # Update statistics
        self._stats['total_saves'] += 1
        self._stats['total_save_time'] += elapsed
        
        logger.info(
            f"[{self.node_id}] Saved checkpoint {iteration} to RAM: "
            f"{entry.get_size_mb():.2f} MB in {elapsed:.3f}s"
        )
        
        return elapsed
    
    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> float:
        """
        Load a checkpoint from RAM.
        
        This is where Gemini shines - loading from RAM is MUCH faster
        than loading from disk/NFS.
        
        Args:
            model: The model to load state into
            optimizer: The optimizer to load state into
            iteration: Specific iteration to load (latest if None)
            device: Device to map tensors to
            
        Returns:
            Time taken to load (seconds)
        """
        start_time = time.time()
        
        with self._lock:
            if not self._checkpoints:
                raise ValueError("No checkpoints available in memory")
            
            # Get the checkpoint
            if iteration is None:
                # Get latest
                iteration = max(self._checkpoints.keys())
            
            if iteration not in self._checkpoints:
                available = list(self._checkpoints.keys())
                raise KeyError(
                    f"Checkpoint {iteration} not found. Available: {available}"
                )
            
            entry = self._checkpoints[iteration]
            serialized_data = entry.data
        
        # Decompress if needed
        if self.compression:
            import zlib
            serialized_data = zlib.decompress(serialized_data)
        
        # Deserialize from bytes
        buffer = io.BytesIO(serialized_data)
        if device:
            checkpoint_data = torch.load(buffer, map_location=device)
        else:
            checkpoint_data = torch.load(buffer)
        
        # Handle DDP-wrapped models
        model_to_load = model.module if hasattr(model, 'module') else model
        
        # Load states
        model_to_load.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        elapsed = time.time() - start_time
        
        # Update statistics
        self._stats['total_loads'] += 1
        self._stats['total_load_time'] += elapsed
        
        logger.info(
            f"[{self.node_id}] Loaded checkpoint {iteration} from RAM in {elapsed:.3f}s"
        )
        
        return elapsed
    
    def get_checkpoint_bytes(self, iteration: int) -> Optional[bytes]:
        """
        Get raw checkpoint bytes for replication to another node.
        
        This is used by the Worker Agent to send checkpoints to peers.
        
        Args:
            iteration: Checkpoint iteration to retrieve
            
        Returns:
            Serialized checkpoint bytes, or None if not found
        """
        with self._lock:
            if iteration not in self._checkpoints:
                return None
            return self._checkpoints[iteration].data
    
    def store_replica(
        self,
        iteration: int,
        data: bytes,
        source_node: str
    ):
        """
        Store a replica checkpoint received from another node.
        
        This is called when receiving a checkpoint from a peer for redundancy.
        
        Args:
            iteration: Checkpoint iteration
            data: Serialized checkpoint bytes
            source_node: ID of the node that sent this checkpoint
        """
        with self._lock:
            # Check if we already have this checkpoint
            if iteration in self._checkpoints:
                logger.debug(f"Checkpoint {iteration} already exists, updating replica info")
                return
            
            # Check memory limits
            size_bytes = len(data)
            if self._stats['current_memory_bytes'] + size_bytes > self.max_memory_bytes:
                self._evict_oldest()
            
            entry = CheckpointEntry(
                iteration=iteration,
                data=data,
                size_bytes=size_bytes,
                timestamp=time.time(),
                node_id=source_node,  # Original owner
                is_replicated=True
            )
            
            self._checkpoints[iteration] = entry
            self._stats['current_memory_bytes'] += size_bytes
            
            logger.info(
                f"[{self.node_id}] Stored replica checkpoint {iteration} "
                f"from {source_node}: {entry.get_size_mb():.2f} MB"
            )
    
    def mark_as_replicated(self, iteration: int, replica_nodes: List[str]):
        """
        Mark a checkpoint as having been replicated to other nodes.
        
        Args:
            iteration: Checkpoint iteration
            replica_nodes: List of node IDs that have the replica
        """
        with self._lock:
            if iteration in self._checkpoints:
                self._checkpoints[iteration].is_replicated = True
                self._checkpoints[iteration].replica_nodes = replica_nodes
    
    def _evict_oldest(self):
        """
        Remove the oldest checkpoint to free memory.
        
        Called internally when we exceed memory or checkpoint limits.
        Must be called with lock held!
        """
        if not self._checkpoints:
            return
        
        # Remove oldest (first in OrderedDict)
        oldest_iter = next(iter(self._checkpoints))
        evicted = self._checkpoints.pop(oldest_iter)
        self._stats['current_memory_bytes'] -= evicted.size_bytes
        
        logger.debug(
            f"[{self.node_id}] Evicted checkpoint {oldest_iter} "
            f"({evicted.get_size_mb():.2f} MB)"
        )
    
    def get_latest_iteration(self) -> Optional[int]:
        """Get the most recent checkpoint iteration, or None if empty."""
        with self._lock:
            if not self._checkpoints:
                return None
            return max(self._checkpoints.keys())
    
    def get_all_iterations(self) -> List[int]:
        """Get all available checkpoint iterations."""
        with self._lock:
            return list(self._checkpoints.keys())
    
    def has_checkpoint(self, iteration: int) -> bool:
        """Check if a specific checkpoint exists."""
        with self._lock:
            return iteration in self._checkpoints
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get checkpoint manager statistics.
        
        Returns:
            Dictionary with performance and usage statistics
        """
        with self._lock:
            num_checkpoints = len(self._checkpoints)
            
            avg_save_time = (
                self._stats['total_save_time'] / self._stats['total_saves']
                if self._stats['total_saves'] > 0 else 0
            )
            avg_load_time = (
                self._stats['total_load_time'] / self._stats['total_loads']
                if self._stats['total_loads'] > 0 else 0
            )
            
            return {
                'node_id': self.node_id,
                'num_checkpoints': num_checkpoints,
                'iterations': list(self._checkpoints.keys()),
                'current_memory_mb': self._stats['current_memory_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_usage_percent': (
                    self._stats['current_memory_bytes'] / self.max_memory_bytes * 100
                ),
                'total_saves': self._stats['total_saves'],
                'total_loads': self._stats['total_loads'],
                'avg_save_time_s': avg_save_time,
                'avg_load_time_s': avg_load_time
            }
    
    def clear(self):
        """Clear all checkpoints from memory."""
        with self._lock:
            self._checkpoints.clear()
            self._stats['current_memory_bytes'] = 0
            logger.info(f"[{self.node_id}] Cleared all checkpoints")


# Convenience function for comparison
def compare_checkpoint_times(
    disk_save_time: float,
    disk_load_time: float,
    memory_save_time: float,
    memory_load_time: float
) -> Dict[str, float]:
    """
    Compare disk vs memory checkpoint performance.
    
    Returns:
        Dictionary with speedup factors
    """
    return {
        'save_speedup': disk_save_time / memory_save_time if memory_save_time > 0 else float('inf'),
        'load_speedup': disk_load_time / memory_load_time if memory_load_time > 0 else float('inf'),
        'disk_save_time': disk_save_time,
        'disk_load_time': disk_load_time,
        'memory_save_time': memory_save_time,
        'memory_load_time': memory_load_time,
        'total_disk_time': disk_save_time + disk_load_time,
        'total_memory_time': memory_save_time + memory_load_time,
        'recovery_speedup': (disk_save_time + disk_load_time) / (memory_save_time + memory_load_time)
    }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing InMemoryCheckpoint...")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100)
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    ckpt_manager = InMemoryCheckpoint(
        node_id="test-node",
        max_checkpoints=3,
        max_memory_gb=1.0
    )
    
    # Test saving
    print("\n1. Testing checkpoint save...")
    save_time = ckpt_manager.save(model, optimizer, iteration=100)
    print(f"   Save time: {save_time:.4f}s")
    
    # Test stats
    print("\n2. Current stats:")
    stats = ckpt_manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Modify model
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    
    # Test loading
    print("\n3. Testing checkpoint load...")
    load_time = ckpt_manager.load(model, optimizer, iteration=100)
    print(f"   Load time: {load_time:.4f}s")
    
    # Test multiple checkpoints
    print("\n4. Testing multiple checkpoints...")
    for i in range(5):
        ckpt_manager.save(model, optimizer, iteration=100 + i)
    
    print(f"   Available iterations: {ckpt_manager.get_all_iterations()}")
    print(f"   (max_checkpoints=3, so only last 3 are kept)")
    
    # Final stats
    print("\n5. Final stats:")
    stats = ckpt_manager.get_stats()
    print(f"   Memory used: {stats['current_memory_mb']:.2f} MB")
    print(f"   Avg save time: {stats['avg_save_time_s']:.4f}s")
    print(f"   Avg load time: {stats['avg_load_time_s']:.4f}s")
    
    print("\n✓ InMemoryCheckpoint working!")

