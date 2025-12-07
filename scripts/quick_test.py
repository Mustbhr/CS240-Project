#!/usr/bin/env python3
"""
Quick Test Script - Minimal test to verify everything works

Run this on IBEX with:
    python scripts/quick_test.py

This is a simpler test than run_baseline_test.py - good for first verification.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_environment():
    """Check that the environment is set up correctly."""
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)
    
    # Python version
    print(f"\n[OK] Python: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False
    
    # Our modules
    try:
        from src.training import BaselineTrainer, TrainingConfig
        print("[OK] BaselineTrainer module loaded")
    except ImportError as e:
        print(f"[FAIL] Failed to load BaselineTrainer: {e}")
        return False
    
    try:
        from src.checkpointing import InMemoryCheckpoint
        print("[OK] InMemoryCheckpoint module loaded")
    except ImportError as e:
        print(f"[FAIL] Failed to load InMemoryCheckpoint: {e}")
        return False
    
    try:
        from src.utils import SyntheticDataset, ExperimentLogger
        print("[OK] Utils modules loaded")
    except ImportError as e:
        print(f"[FAIL] Failed to load utils: {e}")
        return False
    
    # wandb (optional)
    try:
        import wandb
        print(f"[OK] wandb: {wandb.__version__} (optional)")
    except ImportError:
        print("○ wandb not installed (optional)")
    
    print("\n[OK] All required modules loaded!")
    return True


def test_model():
    """Test that we can create and run the model."""
    print("\n" + "=" * 50)
    print("MODEL TEST")
    print("=" * 50)
    
    import torch
    from src.training.baseline_trainer import SimpleLanguageModel
    
    # Create a small model
    model = SimpleLanguageModel(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=64
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"[OK] Model created on {device}")
    print(f"[OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch = torch.randint(0, 1000, (4, 63)).to(device)  # batch_size=4, seq_len=63
    output = model(batch)
    print(f"[OK] Forward pass: input {batch.shape} → output {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("[OK] Backward pass completed")
    
    return True


def test_checkpoint():
    """Test in-memory checkpointing."""
    print("\n" + "=" * 50)
    print("CHECKPOINT TEST")
    print("=" * 50)
    
    import torch
    from src.checkpointing import InMemoryCheckpoint
    
    # Create a simple model
    model = torch.nn.Linear(100, 100)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    ckpt = InMemoryCheckpoint(
        node_id="test-node",
        max_checkpoints=3,
        max_memory_gb=1.0
    )
    
    # Save checkpoint
    save_time = ckpt.save(model, optimizer, iteration=100)
    print(f"[OK] Checkpoint saved in {save_time*1000:.2f} ms")
    
    # Modify model
    with torch.no_grad():
        model.weight.fill_(0.0)
    
    # Load checkpoint
    load_time = ckpt.load(model, optimizer, iteration=100)
    print(f"[OK] Checkpoint loaded in {load_time*1000:.2f} ms")
    
    # Stats
    stats = ckpt.get_stats()
    print(f"[OK] Memory used: {stats['current_memory_mb']:.2f} MB")
    
    return True


def test_training_step():
    """Test a few training steps."""
    print("\n" + "=" * 50)
    print("TRAINING TEST (5 iterations)")
    print("=" * 50)
    
    import torch
    from src.training import BaselineTrainer, TrainingConfig
    from src.utils import SyntheticDataset
    
    # Minimal config
    config = TrainingConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        batch_size=4,
        max_iterations=5,  # Just 5 steps
        checkpoint_frequency=3,
        checkpoint_dir="./checkpoints/quick_test",
        log_interval=1
    )
    
    # Create trainer
    trainer = BaselineTrainer(
        config=config,
        rank=0,
        world_size=1,
        local_rank=0,
        use_wandb=False  # Disable wandb for quick test
    )
    
    # Create tiny dataset
    dataset = SyntheticDataset(
        num_samples=100,
        seq_length=64,
        vocab_size=config.vocab_size
    )
    
    # Train
    print("Running 5 training iterations...")
    results = trainer.train(dataset, resume=False)
    
    print(f"[OK] Completed {results['total_iterations']} iterations")
    print(f"[OK] Final loss: {results['final_loss']:.4f}")
    print(f"[OK] Throughput: {results['average_throughput']:.1f} samples/sec")
    
    return True


def main():
    """Run all quick tests."""
    print("\n" + "=" * 50)
    print("GEMINI PROJECT - QUICK TEST")
    print("=" * 50)
    print("This verifies the basic infrastructure works.")
    print("")
    
    all_passed = True
    
    # Test 1: Environment
    if not check_environment():
        print("\n[FAILED] Environment check failed!")
        return 1
    
    # Test 2: Model
    try:
        if not test_model():
            all_passed = False
    except Exception as e:
        print(f"\n[FAILED] Model test failed: {e}")
        all_passed = False
    
    # Test 3: Checkpoint
    try:
        if not test_checkpoint():
            all_passed = False
    except Exception as e:
        print(f"\n[FAILED] Checkpoint test failed: {e}")
        all_passed = False
    
    # Test 4: Training
    try:
        if not test_training_step():
            all_passed = False
    except Exception as e:
        print(f"\n[FAILED] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("[PASSED] ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYour environment is ready. Next steps:")
        print("1. Run full test: python scripts/run_baseline_test.py")
        print("2. Try with wandb: python scripts/run_baseline_test.py --wandb")
        print("3. Test with multiple nodes when ready")
        return 0
    else:
        print("[FAILED] SOME TESTS FAILED")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())

