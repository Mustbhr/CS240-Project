"""
Data Loading Utilities

This module provides utilities for loading and preprocessing datasets
for distributed training. It supports both:
1. Synthetic data (for testing without downloads)
2. Real Wikipedia data from Hugging Face

Author: Mustafa Albahrani, Mohammed Alkhalifa
Course: CS240 - Distributed Systems
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing infrastructure.
    
    Generates random token sequences without needing to download real data.
    Useful for:
    - Testing the training pipeline
    - Debugging distributed training
    - Benchmarking without I/O bottlenecks
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 512,
        vocab_size: int = 50257
    ):
        """
        Args:
            num_samples: Number of samples in the dataset
            seq_length: Length of each sequence
            vocab_size: Vocabulary size (GPT-2 default is 50257)
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        logger.info(
            f"Created SyntheticDataset with {num_samples} samples, "
            f"seq_length={seq_length}, vocab_size={vocab_size}"
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random tokens (deterministic based on idx for reproducibility)
        generator = torch.Generator().manual_seed(idx)
        tokens = torch.randint(
            0, self.vocab_size, (self.seq_length,), generator=generator
        )
        
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


class WikipediaDataset(Dataset):
    """
    Wikipedia dataset for language model training.
    
    Uses Hugging Face datasets library to load and tokenize Wikipedia.
    The dataset is processed for causal language modeling (next token prediction).
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        split: str = "train",
        subset_size: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer for text processing
            max_length: Maximum sequence length
            split: Dataset split ("train" or other)
            subset_size: If set, only use this many samples (for testing)
            cache_dir: Directory to cache the dataset
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Try to import and load dataset
        try:
            from datasets import load_dataset
            
            logger.info("Loading Wikipedia dataset from Hugging Face...")
            
            # Load Wikipedia dataset
            # Using a subset for faster loading during development
            if subset_size:
                # Load only a portion
                dataset = load_dataset(
                    "wikipedia",
                    "20220301.en",
                    split=f"{split}[:{subset_size}]",
                    cache_dir=cache_dir
                )
            else:
                dataset = load_dataset(
                    "wikipedia",
                    "20220301.en",
                    split=split,
                    cache_dir=cache_dir
                )
            
            logger.info(f"Loaded {len(dataset)} Wikipedia articles")
            
            # Tokenize and prepare
            self.examples = self._prepare_dataset(dataset)
            
            logger.info(f"Prepared {len(self.examples)} training examples")
            
        except ImportError:
            logger.warning(
                "datasets library not available. "
                "Install with: pip install datasets"
            )
            self.examples = []
        except Exception as e:
            logger.error(f"Failed to load Wikipedia dataset: {e}")
            self.examples = []
    
    def _prepare_dataset(self, dataset) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize and prepare examples for training.
        
        Concatenates articles and splits into fixed-length chunks.
        """
        examples = []
        current_tokens = []
        
        for article in dataset:
            # Tokenize the article text
            text = article['text']
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            current_tokens.extend(tokens)
            
            # Create examples when we have enough tokens
            while len(current_tokens) >= self.max_length:
                chunk = current_tokens[:self.max_length]
                current_tokens = current_tokens[self.max_length:]
                
                examples.append({
                    'input_ids': torch.tensor(chunk[:-1]),
                    'labels': torch.tensor(chunk[1:])
                })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a data loader with optional distributed sampling.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size per process
        world_size: Total number of distributed processes
        rank: Rank of current process
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader configured for distributed or single-process training
    """
    sampler = None
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for consistent batch size
    )
    
    return dataloader


def get_tokenizer(model_name: str = "gpt2"):
    """
    Get a tokenizer for text processing.
    
    Args:
        model_name: Name of the model (default: gpt2)
        
    Returns:
        Tokenizer object or None if not available
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # GPT-2 doesn't have a pad token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loaded tokenizer: {model_name}")
        return tokenizer
        
    except ImportError:
        logger.warning(
            "transformers library not available. "
            "Install with: pip install transformers"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None


def create_dataset(
    dataset_type: str = "synthetic",
    **kwargs
) -> Dataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_type: Type of dataset ("synthetic" or "wikipedia")
        **kwargs: Additional arguments for the specific dataset
        
    Returns:
        Dataset object
    """
    if dataset_type == "synthetic":
        return SyntheticDataset(
            num_samples=kwargs.get('num_samples', 10000),
            seq_length=kwargs.get('seq_length', 512),
            vocab_size=kwargs.get('vocab_size', 50257)
        )
    
    elif dataset_type == "wikipedia":
        tokenizer = get_tokenizer(kwargs.get('model_name', 'gpt2'))
        if tokenizer is None:
            logger.warning("Falling back to synthetic dataset")
            return SyntheticDataset()
        
        return WikipediaDataset(
            tokenizer=tokenizer,
            max_length=kwargs.get('max_length', 512),
            split=kwargs.get('split', 'train'),
            subset_size=kwargs.get('subset_size', 1000),
            cache_dir=kwargs.get('cache_dir', None)
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test synthetic dataset
    print("Testing SyntheticDataset...")
    dataset = SyntheticDataset(num_samples=100, seq_length=64, vocab_size=1000)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample label shape: {sample['labels'].shape}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(dataset, batch_size=8, world_size=1, rank=0)
    for batch in dataloader:
        print(f"Batch input shape: {batch['input_ids'].shape}")
        print(f"Batch label shape: {batch['labels'].shape}")
        break
    
    print("\n✓ Data loading utilities working!")

