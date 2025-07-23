import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from typing import Tuple, Dict, List, Optional
import random
import yaml


class PreprocessedCellStateDataset(Dataset):
    """Dataset for cell state VAE training using preprocessed data."""
    
    def __init__(self, preprocessed_dir: str, split: str = "train", config=None):
        """
        Args:
            preprocessed_dir: Directory containing preprocessed data
            split: Data split ("train", "val", or "test")
            config: Configuration object (optional, for compatibility)
        """
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.config = config
        
        # Load preprocessed data
        self._load_preprocessed_data()
        
        # Current epoch for bidirectional training
        self.current_epoch = 0
        self.bidirectional_training = (
            config.training.bidirectional_training if config else True
        )
    
    def _load_preprocessed_data(self):
        """Load preprocessed data from HDF5 files."""
        dataset_path = os.path.join(self.preprocessed_dir, "preprocessed_dataset.h5")
        pairs_path = os.path.join(self.preprocessed_dir, "training_pairs.npy")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Preprocessed dataset not found: {dataset_path}")
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Training pairs not found: {pairs_path}")
        
        # Load main dataset
        with h5py.File(dataset_path, 'r') as f:
            # Load cell embeddings
            self.cell_embeddings = torch.FloatTensor(f['cell_embeddings'][:])
            
            # Load KNN averages for all K values
            self.knn_averages = {}
            knn_group = f['knn_averages']
            for k_key in knn_group.keys():
                k = int(k_key.split('_')[1])  # Extract K from 'k_5', 'k_10', etc.
                self.knn_averages[k] = torch.FloatTensor(knn_group[k_key][:])
            
            # Load data splits
            splits_group = f['data_splits']
            self.split_indices = {
                'train': splits_group['train'][:],
                'val': splits_group['val'][:],
                'test': splits_group['test'][:]
            }
            
            # Load metadata
            meta_group = f['metadata']
            self.n_cells = meta_group.attrs['n_cells']
            self.n_features = meta_group.attrs['n_features']
            self.k_values = list(meta_group.attrs['k_values'])
        
        # Load training pairs
        all_training_pairs = np.load(pairs_path)
        
        # Filter training pairs for current split
        split_cell_indices = set(self.split_indices[self.split])
        self.training_pairs = [
            (cell_idx, k) for cell_idx, k in all_training_pairs
            if cell_idx in split_cell_indices
        ]
        
        print(f"Loaded {self.split} dataset:")
        print(f"  Cells: {len(self.split_indices[self.split])}")
        print(f"  Training pairs: {len(self.training_pairs)}")
        print(f"  K values: {self.k_values}")
        print(f"  Features: {self.n_features}")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for bidirectional training."""
        self.current_epoch = epoch
    
    def __len__(self) -> int:
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample."""
        cell_idx, k = self.training_pairs[idx]
        
        # Get individual cell embedding
        cell_embedding = self.cell_embeddings[cell_idx]
        
        # Get KNN average for this K value
        knn_embedding = self.knn_averages[k][cell_idx]
        
        # Determine training mode based on epoch
        if self.bidirectional_training and self.current_epoch % 2 == 1:
            training_mode = "reverse"
        else:
            training_mode = "forward"
        
        return {
            'cell_embedding': cell_embedding,
            'knn_embedding': knn_embedding,
            'training_mode': training_mode,
            'cell_idx': cell_idx,
            'k_value': k
        }


class PreprocessedCellStateDataModule:
    """Data module for handling preprocessed cell state data."""
    
    def __init__(self, preprocessed_dir: str, config):
        """
        Args:
            preprocessed_dir: Directory containing preprocessed data
            config: Configuration object
        """
        self.preprocessed_dir = preprocessed_dir
        self.config = config
        
        # Verify preprocessed data exists
        if not self._verify_preprocessed_data():
            raise FileNotFoundError(
                f"Preprocessed data not found in {preprocessed_dir}. "
                "Please run the preprocessing script first."
            )
        
        # Create datasets
        self.train_dataset = PreprocessedCellStateDataset(
            preprocessed_dir, split="train", config=config
        )
        self.val_dataset = PreprocessedCellStateDataset(
            preprocessed_dir, split="val", config=config
        )
        self.test_dataset = PreprocessedCellStateDataset(
            preprocessed_dir, split="test", config=config
        )
    
    def _verify_preprocessed_data(self) -> bool:
        """Verify that all required preprocessed files exist."""
        required_files = [
            "preprocessed_dataset.h5",
            "training_pairs.npy",
            "preprocessing_config.yaml"
        ]
        
        for filename in required_files:
            filepath = os.path.join(self.preprocessed_dir, filename)
            if not os.path.exists(filepath):
                print(f"Missing preprocessed file: {filepath}")
                return False
        
        return True
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.config.device == "cuda" else False,
            drop_last=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.config.device == "cuda" else False,
            drop_last=False,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.config.device == "cuda" else False,
            drop_last=False,
            collate_fn=collate_fn
        )
    
    def set_epoch(self, epoch: int):
        """Set epoch for all datasets (for bidirectional training)."""
        self.train_dataset.set_epoch(epoch)
        self.val_dataset.set_epoch(epoch)
        self.test_dataset.set_epoch(epoch)


# Keep the old classes for backward compatibility but mark as deprecated
class KNNGraph:
    """DEPRECATED: Use preprocessed data instead."""
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "KNNGraph is deprecated. Use the preprocessing script to create "
            "preprocessed data and use PreprocessedCellStateDataset instead."
        )


class CellStateDataset:
    """DEPRECATED: Use PreprocessedCellStateDataset instead."""
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "CellStateDataset is deprecated. Use the preprocessing script to create "
            "preprocessed data and use PreprocessedCellStateDataset instead."
        )


class CellStateDataModule:
    """DEPRECATED: Use PreprocessedCellStateDataModule instead."""
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "CellStateDataModule is deprecated. Use the preprocessing script to create "
            "preprocessed data and use PreprocessedCellStateDataModule instead."
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack tensors
    cell_embeddings = torch.stack([item['cell_embedding'] for item in batch])
    knn_embeddings = torch.stack([item['knn_embedding'] for item in batch])
    
    # Get training modes (should be the same within a batch)
    training_modes = [item['training_mode'] for item in batch]
    training_mode = training_modes[0]  # Use first mode (all should be same)
    
    # Stack indices and k values
    cell_indices = torch.tensor([item['cell_idx'] for item in batch])
    k_values = torch.tensor([item['k_value'] for item in batch])
    
    return {
        'cell_embedding': cell_embeddings,
        'knn_embedding': knn_embeddings,
        'training_mode': training_mode,
        'cell_idx': cell_indices,
        'k_value': k_values
    }


def create_datamodule(preprocessed_dir: str, config) -> PreprocessedCellStateDataModule:
    """Factory function to create data module from preprocessed data."""
    return PreprocessedCellStateDataModule(preprocessed_dir, config)


def load_preprocessing_config(preprocessed_dir: str) -> Dict:
    """
    Load preprocessing configuration from preprocessed data directory.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
    
    Returns:
        Preprocessing configuration dictionary
    """
    config_path = os.path.join(preprocessed_dir, "preprocessing_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Preprocessing config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_preprocessed_data(preprocessed_dir: str) -> Dict:
    """
    Verify preprocessed data and return metadata.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Dictionary with data statistics and metadata
    """
    dataset_path = os.path.join(preprocessed_dir, "preprocessed_dataset.h5")
    pairs_path = os.path.join(preprocessed_dir, "training_pairs.npy")
    config_path = os.path.join(preprocessed_dir, "preprocessing_config.yaml")
    
    if not all(os.path.exists(p) for p in [dataset_path, pairs_path, config_path]):
        raise FileNotFoundError("Missing required preprocessed files")
    
    # Load metadata
    with h5py.File(dataset_path, 'r') as f:
        meta = f['metadata']
        n_cells = meta.attrs['n_cells']
        n_features = meta.attrs['n_features']
        k_values = list(meta.attrs['k_values'])
        
        # Get split sizes
        splits = f['data_splits']
        train_size = len(splits['train'])
        val_size = len(splits['val'])
        test_size = len(splits['test'])
    
    # Load training pairs count
    training_pairs = np.load(pairs_path)
    n_pairs = len(training_pairs)
    
    # Load preprocessing config
    with open(config_path, 'r') as f:
        preprocessing_config = yaml.safe_load(f)
    
    return {
        'n_cells': n_cells,
        'n_features': n_features,
        'k_values': k_values,
        'n_training_pairs': n_pairs,
        'data_splits': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        },
        'preprocessing_config': preprocessing_config
    }


# Utility functions for evaluation (kept for compatibility)
def compute_reconstruction_metrics(reconstructed: torch.Tensor, 
                                 target: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction quality metrics."""
    mse = F.mse_loss(reconstructed, target).item()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(reconstructed, target, dim=1).mean().item()
    
    return {
        'mse': mse,
        'cosine_similarity': cos_sim
    }


def compute_diversity_metrics(samples: torch.Tensor) -> Dict[str, float]:
    """Compute diversity metrics for generated samples."""
    # Pairwise distances
    samples_norm = F.normalize(samples, p=2, dim=1)
    pairwise_similarities = torch.mm(samples_norm, samples_norm.t())
    
    # Mean pairwise distance (1 - cosine similarity)
    n_samples = len(samples)
    mask = torch.triu(torch.ones(n_samples, n_samples), diagonal=1).bool()
    mean_distance = (1 - pairwise_similarities[mask]).mean().item()
    
    # Coverage (variance along each dimension)
    coverage = samples.var(dim=0).mean().item()
    
    return {
        'mean_pairwise_distance': mean_distance,
        'coverage': coverage
    }
