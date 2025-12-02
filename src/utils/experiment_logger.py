"""
Experiment Logger with Weights & Biases Integration

This module provides unified logging for experiments with optional wandb support.
It gracefully falls back to local logging if wandb is not available or disabled.

Features:
1. Automatic wandb initialization with project/run configuration
2. Metric logging with automatic chart generation
3. System metrics (GPU, memory) tracking
4. Checkpoint performance comparison
5. Failure event logging
6. Offline mode support for restricted networks

Usage:
    logger = ExperimentLogger(
        project="gemini-cs240",
        run_name="baseline-experiment",
        config={"batch_size": 8, "nodes": 4}
    )
    logger.log({"loss": 0.5, "throughput": 100})
    logger.finish()

Author: Mustafa Albahrani, Mohammed Alkhalifah
Course: CS240 - Distributed Systems
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint operation"""
    iteration: int
    save_time_ms: float
    load_time_ms: float
    size_mb: float
    checkpoint_type: str  # "disk" or "memory"
    timestamp: float


@dataclass 
class RecoveryMetrics:
    """Metrics for a recovery event"""
    iteration: int
    failed_node: str
    recovery_time_ms: float
    wasted_iterations: int
    checkpoint_type: str
    timestamp: float


class ExperimentLogger:
    """
    Unified experiment logger with optional wandb integration.
    
    Provides consistent logging interface whether wandb is available or not.
    All metrics are also saved locally as JSON for backup.
    """
    
    def __init__(
        self,
        project: str = "gemini-cs240",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        use_wandb: bool = True,
        offline: bool = False,
        log_dir: str = "./logs"
    ):
        """
        Initialize the experiment logger.
        
        Args:
            project: wandb project name
            run_name: Name for this run (auto-generated if None)
            config: Configuration dictionary to log
            tags: Tags for filtering runs (e.g., ["baseline", "4-nodes"])
            group: Group name for distributed runs
            job_type: Job type within group (e.g., "node-0", "coordinator")
            use_wandb: Whether to use wandb (set False to disable)
            offline: Run wandb in offline mode (sync later)
            log_dir: Directory for local log files
        """
        self.project = project
        self.config = config or {}
        self.use_wandb = use_wandb
        self.wandb_run = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        self.run_name = run_name
        
        # Local metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.checkpoint_metrics: List[CheckpointMetrics] = []
        self.recovery_metrics: List[RecoveryMetrics] = []
        
        # Initialize wandb if requested
        if use_wandb:
            self._init_wandb(
                project=project,
                run_name=run_name,
                config=config,
                tags=tags,
                group=group,
                job_type=job_type,
                offline=offline
            )
        
        logger.info(f"ExperimentLogger initialized: {run_name}")
    
    def _init_wandb(
        self,
        project: str,
        run_name: str,
        config: Optional[Dict],
        tags: Optional[List[str]],
        group: Optional[str],
        job_type: Optional[str],
        offline: bool
    ):
        """Initialize wandb with error handling."""
        try:
            import wandb
            
            # Set offline mode if requested
            if offline:
                os.environ["WANDB_MODE"] = "offline"
            
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                tags=tags,
                group=group,
                job_type=job_type,
                reinit=True  # Allow multiple inits in same process
            )
            
            # Define custom charts for our metrics
            wandb.define_metric("iteration")
            wandb.define_metric("loss", step_metric="iteration")
            wandb.define_metric("throughput", step_metric="iteration")
            wandb.define_metric("checkpoint/*", step_metric="iteration")
            wandb.define_metric("recovery/*", step_metric="iteration")
            
            logger.info(f"wandb initialized: {wandb.run.url}")
            
        except ImportError:
            logger.warning(
                "wandb not installed. Install with: pip install wandb\n"
                "Falling back to local logging only."
            )
            self.use_wandb = False
            self.wandb_run = None
            
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}\nFalling back to local logging.")
            self.use_wandb = False
            self.wandb_run = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb and local storage.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (uses internal counter if None)
        """
        # Add timestamp
        metrics_with_time = {
            **metrics,
            "_timestamp": time.time()
        }
        
        # Store locally
        self.metrics_history.append(metrics_with_time)
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
    
    def log_checkpoint(
        self,
        iteration: int,
        save_time_ms: float,
        size_mb: float,
        checkpoint_type: str,
        load_time_ms: float = 0.0
    ):
        """
        Log checkpoint performance metrics.
        
        Args:
            iteration: Training iteration
            save_time_ms: Time to save checkpoint (milliseconds)
            size_mb: Checkpoint size (megabytes)
            checkpoint_type: "disk" or "memory"
            load_time_ms: Time to load checkpoint (milliseconds)
        """
        metric = CheckpointMetrics(
            iteration=iteration,
            save_time_ms=save_time_ms,
            load_time_ms=load_time_ms,
            size_mb=size_mb,
            checkpoint_type=checkpoint_type,
            timestamp=time.time()
        )
        self.checkpoint_metrics.append(metric)
        
        # Log to wandb with prefixed names for grouping
        self.log({
            f"checkpoint/{checkpoint_type}_save_time_ms": save_time_ms,
            f"checkpoint/{checkpoint_type}_load_time_ms": load_time_ms,
            f"checkpoint/{checkpoint_type}_size_mb": size_mb,
            "iteration": iteration
        })
    
    def log_recovery(
        self,
        iteration: int,
        failed_node: str,
        recovery_time_ms: float,
        wasted_iterations: int,
        checkpoint_type: str
    ):
        """
        Log recovery event metrics.
        
        This is crucial for comparing Gemini vs NFS recovery!
        
        Args:
            iteration: Iteration when failure occurred
            failed_node: ID of the failed node
            recovery_time_ms: Time to recover (milliseconds)
            wasted_iterations: Iterations lost due to failure
            checkpoint_type: "disk" or "memory"
        """
        metric = RecoveryMetrics(
            iteration=iteration,
            failed_node=failed_node,
            recovery_time_ms=recovery_time_ms,
            wasted_iterations=wasted_iterations,
            checkpoint_type=checkpoint_type,
            timestamp=time.time()
        )
        self.recovery_metrics.append(metric)
        
        # Log to wandb
        self.log({
            f"recovery/{checkpoint_type}_time_ms": recovery_time_ms,
            f"recovery/{checkpoint_type}_wasted_iterations": wasted_iterations,
            "recovery/failed_node": failed_node,
            "iteration": iteration
        })
        
        logger.info(
            f"Recovery logged: {checkpoint_type} took {recovery_time_ms:.2f}ms, "
            f"wasted {wasted_iterations} iterations"
        )
    
    def log_comparison(
        self,
        disk_save_time_ms: float,
        disk_load_time_ms: float,
        memory_save_time_ms: float,
        memory_load_time_ms: float
    ):
        """
        Log a direct comparison between disk and memory checkpointing.
        
        This creates the key comparison chart for the project.
        """
        save_speedup = disk_save_time_ms / memory_save_time_ms if memory_save_time_ms > 0 else 0
        load_speedup = disk_load_time_ms / memory_load_time_ms if memory_load_time_ms > 0 else 0
        total_speedup = (disk_save_time_ms + disk_load_time_ms) / (memory_save_time_ms + memory_load_time_ms)
        
        comparison = {
            "comparison/disk_save_time_ms": disk_save_time_ms,
            "comparison/disk_load_time_ms": disk_load_time_ms,
            "comparison/memory_save_time_ms": memory_save_time_ms,
            "comparison/memory_load_time_ms": memory_load_time_ms,
            "comparison/save_speedup": save_speedup,
            "comparison/load_speedup": load_speedup,
            "comparison/total_speedup": total_speedup
        }
        
        self.log(comparison)
        
        logger.info(
            f"Comparison: Memory is {total_speedup:.1f}x faster overall "
            f"(save: {save_speedup:.1f}x, load: {load_speedup:.1f}x)"
        )
        
        return comparison
    
    def log_system_metrics(self):
        """Log system metrics (GPU memory, CPU usage, etc.)"""
        try:
            import psutil
            
            metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
                "system/memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
            
            # GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        metrics[f"system/gpu{i}_memory_allocated_gb"] = mem_allocated
                        metrics[f"system/gpu{i}_memory_reserved_gb"] = mem_reserved
            except:
                pass
            
            self.log(metrics)
            
        except ImportError:
            pass  # psutil not available
    
    def create_summary_table(self) -> Dict[str, Any]:
        """Create a summary of checkpoint performance for the report."""
        if not self.checkpoint_metrics:
            return {}
        
        disk_metrics = [m for m in self.checkpoint_metrics if m.checkpoint_type == "disk"]
        memory_metrics = [m for m in self.checkpoint_metrics if m.checkpoint_type == "memory"]
        
        summary = {}
        
        if disk_metrics:
            summary["disk"] = {
                "avg_save_time_ms": sum(m.save_time_ms for m in disk_metrics) / len(disk_metrics),
                "avg_load_time_ms": sum(m.load_time_ms for m in disk_metrics) / len(disk_metrics),
                "avg_size_mb": sum(m.size_mb for m in disk_metrics) / len(disk_metrics),
                "count": len(disk_metrics)
            }
        
        if memory_metrics:
            summary["memory"] = {
                "avg_save_time_ms": sum(m.save_time_ms for m in memory_metrics) / len(memory_metrics),
                "avg_load_time_ms": sum(m.load_time_ms for m in memory_metrics) / len(memory_metrics),
                "avg_size_mb": sum(m.size_mb for m in memory_metrics) / len(memory_metrics),
                "count": len(memory_metrics)
            }
        
        if disk_metrics and memory_metrics:
            summary["speedup"] = {
                "save": summary["disk"]["avg_save_time_ms"] / summary["memory"]["avg_save_time_ms"],
                "load": summary["disk"]["avg_load_time_ms"] / summary["memory"]["avg_load_time_ms"],
            }
        
        # Log summary to wandb
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.run.summary.update(summary)
            except:
                pass
        
        return summary
    
    def save_local(self):
        """Save all metrics to local JSON files."""
        # Save metrics history
        metrics_file = self.log_dir / f"{self.run_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        # Save checkpoint metrics
        checkpoint_file = self.log_dir / f"{self.run_name}_checkpoints.json"
        with open(checkpoint_file, 'w') as f:
            json.dump([asdict(m) for m in self.checkpoint_metrics], f, indent=2)
        
        # Save recovery metrics
        recovery_file = self.log_dir / f"{self.run_name}_recovery.json"
        with open(recovery_file, 'w') as f:
            json.dump([asdict(m) for m in self.recovery_metrics], f, indent=2)
        
        logger.info(f"Saved local logs to {self.log_dir}")
    
    def finish(self):
        """Finish the experiment and save all data."""
        # Create summary
        summary = self.create_summary_table()
        
        # Save locally
        self.save_local()
        
        # Finish wandb run
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.finish()
                logger.info("wandb run finished successfully")
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")
        
        return summary


def create_logger_for_distributed(
    rank: int,
    world_size: int,
    experiment_name: str,
    config: Optional[Dict] = None,
    use_wandb: bool = True
) -> ExperimentLogger:
    """
    Create a logger configured for distributed training.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        experiment_name: Name for this experiment
        config: Configuration to log
        use_wandb: Whether to use wandb
        
    Returns:
        Configured ExperimentLogger
    """
    # Only rank 0 logs to wandb to avoid duplicate data
    # Other ranks log locally only
    use_wandb_for_rank = use_wandb and (rank == 0)
    
    return ExperimentLogger(
        project="gemini-cs240",
        run_name=f"{experiment_name}_rank{rank}",
        config={
            **(config or {}),
            "rank": rank,
            "world_size": world_size
        },
        tags=[f"{world_size}-nodes", experiment_name],
        group=experiment_name,
        job_type=f"node-{rank}",
        use_wandb=use_wandb_for_rank
    )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ExperimentLogger...")
    
    # Test without wandb (local only)
    logger_instance = ExperimentLogger(
        project="test-project",
        run_name="test-run",
        config={"batch_size": 8, "nodes": 4},
        use_wandb=False  # Disable for testing
    )
    
    # Log some metrics
    for i in range(10):
        logger_instance.log({
            "loss": 1.0 / (i + 1),
            "throughput": 100 + i * 10,
            "iteration": i
        })
    
    # Log checkpoint metrics
    logger_instance.log_checkpoint(
        iteration=100,
        save_time_ms=150.0,
        size_mb=50.0,
        checkpoint_type="disk"
    )
    
    logger_instance.log_checkpoint(
        iteration=100,
        save_time_ms=15.0,
        size_mb=50.0,
        checkpoint_type="memory"
    )
    
    # Log comparison
    logger_instance.log_comparison(
        disk_save_time_ms=150.0,
        disk_load_time_ms=200.0,
        memory_save_time_ms=15.0,
        memory_load_time_ms=10.0
    )
    
    # Finish and save
    summary = logger_instance.finish()
    
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    
    print("\n✓ ExperimentLogger working!")

