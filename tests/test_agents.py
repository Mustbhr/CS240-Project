"""
Unit tests for Worker and Root agents
"""

import pytest
import torch
import torch.nn as nn
from src.agents import WorkerAgent, RootAgent


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class TestWorkerAgent:
    """Tests for WorkerAgent"""
    
    def test_initialization(self):
        """Test worker agent initialization"""
        agent = WorkerAgent(node_id="node-0", memory_limit_gb=5.0)
        assert agent.node_id == "node-0"
        assert agent.memory_limit_gb == 5.0
        assert len(agent.checkpoints) == 0
    
    def test_capture_checkpoint(self):
        """Test checkpoint capture"""
        agent = WorkerAgent(node_id="node-0")
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        success = agent.capture_checkpoint(model, optimizer, iteration=100)
        assert success
        assert 100 in agent.checkpoints
        assert 100 in agent.metadata
    
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        agent = WorkerAgent(node_id="node-0")
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Capture checkpoint
        agent.capture_checkpoint(model, optimizer, iteration=100)
        
        # Modify model
        with torch.no_grad():
            model.linear.weight.fill_(1.0)
        
        # Load checkpoint
        success = agent.load_checkpoint(100, model, optimizer)
        assert success
    
    def test_cleanup_old_checkpoints(self):
        """Test checkpoint cleanup"""
        agent = WorkerAgent(node_id="node-0")
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create 5 checkpoints
        for i in range(5):
            agent.capture_checkpoint(model, optimizer, iteration=i)
        
        # Keep last 2
        agent.cleanup_old_checkpoints(keep_last_n=2)
        
        assert len(agent.checkpoints) == 2
        assert 3 in agent.checkpoints
        assert 4 in agent.checkpoints


class TestRootAgent:
    """Tests for RootAgent"""
    
    def test_initialization(self):
        """Test root agent initialization"""
        agent = RootAgent(num_nodes=4, replication_factor=2)
        assert agent.num_nodes == 4
        assert agent.replication_factor == 2
        assert len(agent.nodes) == 0
    
    def test_node_registration(self):
        """Test node registration"""
        agent = RootAgent(num_nodes=4)
        agent.register_node("node-0")
        agent.register_node("node-1")
        
        assert len(agent.nodes) == 2
        assert "node-0" in agent.nodes
        assert "node-1" in agent.nodes
    
    def test_checkpoint_registration(self):
        """Test checkpoint registration"""
        agent = RootAgent(num_nodes=4)
        agent.register_node("node-0")
        agent.register_node("node-1")
        
        agent.register_checkpoint("node-0", iteration=100, replica_nodes=["node-1"])
        
        assert 100 in agent.checkpoint_map
        assert "node-0" in agent.checkpoint_map[100]
        assert "node-1" in agent.checkpoint_map[100]
    
    def test_recovery(self):
        """Test checkpoint recovery"""
        agent = RootAgent(num_nodes=4)
        agent.register_node("node-0")
        agent.register_node("node-1")
        
        agent.register_checkpoint("node-0", iteration=100, replica_nodes=["node-1"])
        
        # Simulate node-0 failure
        replica_node = agent.initiate_recovery("node-0", 100)
        
        assert replica_node == "node-1"
    
    def test_cluster_status(self):
        """Test cluster status reporting"""
        agent = RootAgent(num_nodes=4)
        agent.register_node("node-0")
        agent.register_node("node-1")
        
        status = agent.get_cluster_status()
        
        assert status['total_nodes'] == 2
        assert status['healthy_nodes'] == 2
        assert status['failed_nodes'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

