"""
Root Agent - Coordinates recovery and failure detection

Responsibilities:
- Monitor cluster health
- Detect node failures
- Coordinate checkpoint recovery
- Manage checkpoint replication strategy
"""

import logging
import time
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum


class NodeStatus(Enum):
    """Node health status"""
    HEALTHY = "healthy"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class NodeInfo:
    """Information about a node in the cluster"""
    node_id: str
    status: NodeStatus
    last_heartbeat: float
    checkpoints: Set[int]  # Checkpoint iterations stored on this node


class RootAgent:
    """
    Root coordinator for the distributed training cluster.
    Manages failure detection and recovery orchestration.
    """
    
    def __init__(self, num_nodes: int, replication_factor: int = 2, heartbeat_timeout: float = 30.0):
        """
        Initialize root agent.
        
        Args:
            num_nodes: Total number of worker nodes
            replication_factor: Number of replicas per checkpoint (m)
            heartbeat_timeout: Seconds before considering node failed
        """
        self.num_nodes = num_nodes
        self.replication_factor = replication_factor
        self.heartbeat_timeout = heartbeat_timeout
        
        self.nodes: Dict[str, NodeInfo] = {}
        self.checkpoint_map: Dict[int, List[str]] = {}  # iteration -> list of node_ids
        self.logger = logging.getLogger("RootAgent")
        
        self.logger.info(f"Initialized RootAgent: {num_nodes} nodes, replication factor {replication_factor}")
    
    def register_node(self, node_id: str):
        """
        Register a new worker node.
        
        Args:
            node_id: Unique identifier for the node
        """
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            status=NodeStatus.HEALTHY,
            last_heartbeat=time.time(),
            checkpoints=set()
        )
        self.logger.info(f"Registered node {node_id}")
    
    def update_heartbeat(self, node_id: str):
        """
        Update heartbeat for a node.
        
        Args:
            node_id: Node sending heartbeat
        """
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
    
    def detect_failures(self) -> List[str]:
        """
        Check for failed nodes based on heartbeat timeout.
        
        Returns:
            List of failed node IDs
        """
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node_info in self.nodes.items():
            if node_info.status == NodeStatus.HEALTHY:
                if current_time - node_info.last_heartbeat > self.heartbeat_timeout:
                    self.logger.warning(f"Node {node_id} failed (heartbeat timeout)")
                    node_info.status = NodeStatus.FAILED
                    failed_nodes.append(node_id)
        
        return failed_nodes
    
    def get_replication_targets(self, source_node: str, iteration: int) -> List[str]:
        """
        Determine which nodes should store replicas using group-placement strategy.
        
        Args:
            source_node: Node that owns the checkpoint
            iteration: Checkpoint iteration
            
        Returns:
            List of target node IDs for replication
        """
        # TODO: Implement intelligent group-placement strategy
        # For now, use simple round-robin
        available_nodes = [
            node_id for node_id, info in self.nodes.items()
            if info.status == NodeStatus.HEALTHY and node_id != source_node
        ]
        
        # Select replication_factor - 1 nodes (source already has one copy)
        num_replicas = min(self.replication_factor - 1, len(available_nodes))
        return available_nodes[:num_replicas]
    
    def register_checkpoint(self, node_id: str, iteration: int, replica_nodes: List[str]):
        """
        Register a new checkpoint in the system.
        
        Args:
            node_id: Node that created the checkpoint
            iteration: Checkpoint iteration
            replica_nodes: Nodes storing replicas
        """
        all_nodes = [node_id] + replica_nodes
        self.checkpoint_map[iteration] = all_nodes
        
        for nid in all_nodes:
            if nid in self.nodes:
                self.nodes[nid].checkpoints.add(iteration)
        
        self.logger.debug(f"Registered checkpoint {iteration} on nodes: {all_nodes}")
    
    def initiate_recovery(self, failed_node: str, target_iteration: int) -> Optional[str]:
        """
        Coordinate recovery of checkpoints from a failed node.
        
        Args:
            failed_node: ID of the failed node
            target_iteration: Checkpoint iteration to recover
            
        Returns:
            Node ID that has the checkpoint replica, or None
        """
        self.logger.info(f"Initiating recovery for node {failed_node} at iteration {target_iteration}")
        
        # Find nodes with the checkpoint
        nodes_with_checkpoint = self.checkpoint_map.get(target_iteration, [])
        
        # Filter out failed node
        available_replicas = [
            nid for nid in nodes_with_checkpoint
            if nid != failed_node and self.nodes[nid].status == NodeStatus.HEALTHY
        ]
        
        if not available_replicas:
            self.logger.error(f"No replicas available for checkpoint {target_iteration}")
            return None
        
        # Select first available replica
        replica_node = available_replicas[0]
        self.logger.info(f"Recovering checkpoint {target_iteration} from node {replica_node}")
        
        return replica_node
    
    def rebalance_shards(self, recovered_node: str):
        """
        Rebalance checkpoint distribution after node recovery.
        
        Args:
            recovered_node: Node that has recovered
        """
        # TODO: Implement rebalancing logic
        self.logger.info(f"Rebalancing shards after {recovered_node} recovery")
        
        if recovered_node in self.nodes:
            self.nodes[recovered_node].status = NodeStatus.HEALTHY
    
    def get_cluster_status(self) -> Dict:
        """
        Get current cluster status.
        
        Returns:
            Dictionary with cluster health information
        """
        healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
        failed_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)
        
        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'failed_nodes': failed_nodes,
            'total_checkpoints': len(self.checkpoint_map)
        }

