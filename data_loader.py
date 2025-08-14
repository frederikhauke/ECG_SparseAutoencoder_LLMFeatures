import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tools.timing_extractor import get_timing_extractor


class PTBXLDataset(Dataset):
    """PTB-XL ECG Dataset loader that uses preprocessed cached data.

    """
    
    def __init__(self, preprocessed_path: str, original_data_path: Optional[str] = None, 
                 max_samples: Optional[int] = None):
        """
        Initialize PTB-XL dataset from preprocessed data.
        
        Args:
            preprocessed_path: Path to preprocessed data directory
            original_data_path: Path to original PTB-XL dataset (for timing extractor)
            max_samples: Maximum number of samples to load (for testing)
        """
        self.preprocessed_path = Path(preprocessed_path)
        self.original_data_path = Path(original_data_path) if original_data_path else None
        
        # Load preprocessing configuration
        with open(self.preprocessed_path / 'preprocessing_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.sampling_rate = self.config['sampling_rate']
        self.normalize = self.config['normalize']
        
        # Initialize timing extractor if original data path is provided
        if self.original_data_path:
            self.timing_extractor = get_timing_extractor(cache_path="times.csv", use_cache=True)
        else:
            self.timing_extractor = None
        
        # Load cached data
        self._load_cached_data(max_samples)
        
    def _load_cached_data(self, max_samples: Optional[int] = None):
        """Load preprocessed data from cached files."""
        print(f"Loading preprocessed ECG data from {self.preprocessed_path}...")
        
        # Load signals
        self.signals = np.load(self.preprocessed_path / 'signals.npy')
        
        # Load heartbeats
        self.heartbeats = np.load(self.preprocessed_path / 'heartbeats.npy')
        
        # Load metadata
        self.metadata = pd.read_csv(self.preprocessed_path / 'metadata.csv')
        
        # Set ecg_id as index for compatibility
        if 'ecg_id' in self.metadata.columns:
            self.metadata = self.metadata.set_index('ecg_id')
        
        # Limit samples if specified
        if max_samples and max_samples < len(self.signals):
            self.signals = self.signals[:max_samples]
            self.heartbeats = self.heartbeats[:max_samples]
            self.metadata = self.metadata.head(max_samples)
        
        print(f"Loaded {len(self.signals)} ECG signals")
        print(f"Signal shape: {self.signals.shape}")
        print(f"Heartbeats shape: {self.heartbeats.shape}")
        print(f"Sampling rate: {self.sampling_rate}Hz")
        print(f"Normalized: {self.normalize}")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx])
        
        # Load precomputed heartbeats
        heartbeats_tensor = torch.FloatTensor(self.heartbeats[idx])
        
        # Keep original signal for backward compatibility
        signal_flat = signal.flatten()
        
        return {
            'signal': signal,
            'signal_flat': signal_flat,
            'heartbeats': heartbeats_tensor,  # Precomputed simplified input
            'report': self.metadata.iloc[idx]['report'],
            'ecg_id': self.metadata.index[idx],
            'idx': idx
        }
    
    def _extract_timing_features(self, idx: int) -> np.ndarray:
        """Extract timing features using unified extractor."""
        if self.timing_extractor is None:
            raise ValueError("Timing extractor not available. Provide original_data_path when initializing dataset.")
        
        ecg_id = self.metadata.index[idx]
        signal_2d = self.signals[idx] if hasattr(self, 'signals') else None
        
        return self.timing_extractor.get_timing_features(
            ecg_id=ecg_id,
            signal_2d=signal_2d,
            sampling_rate=self.sampling_rate
        )
    
    def get_signal_shape(self):
        """Get the shape of ECG signals."""
        return self.signals[0].shape
    
    def get_flat_signal_dim(self):
        """Get the dimension of flattened ECG signals."""
        return np.prod(self.signals[0].shape)
    
    def get_heartbeat_dim(self):
        """Get the dimension of simplified heartbeat input."""
        # Calculate expected heartbeat dimension: num_leads * num_beats * beat_duration_samples
        num_leads = self.signals.shape[2] if len(self.signals.shape) == 3 else 12  # Default to 12 leads
        num_beats = 3
        beat_duration_samples = int(750 * self.sampling_rate / 1000)  # 750ms in samples
        
        # Return actual heartbeat dimension from loaded data
        return self.heartbeats.shape[1]


def create_data_loaders(preprocessed_path: str, batch_size: int = 32, 
                       test_fold: int = 10, original_data_path: Optional[str] = None,
                       max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders from preprocessed data.
    
    Args:
        preprocessed_path: Path to preprocessed data directory
        batch_size: Batch size for data loaders
        test_fold: Fold to use for testing (1-10)
        original_data_path: Path to original PTB-XL dataset (for timing features)
        max_samples: Maximum number of samples to load
        
    Returns:
        train_loader, test_loader, dataset
    """
    # Load full dataset from preprocessed data
    dataset = PTBXLDataset(preprocessed_path, original_data_path, max_samples=max_samples)
    
    # Split indices based on stratified folds
    train_indices = []
    test_indices = []
    
    for pos_idx, (idx, row) in enumerate(dataset.metadata.iterrows()):
        if row['strat_fold'] == test_fold:
            test_indices.append(pos_idx)
        else:
            train_indices.append(pos_idx)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, dataset


if __name__ == "__main__":
    # Test the dataset loader with preprocessed data
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the data loader with preprocessed data')
    parser.add_argument('--preprocessed_path', type=str, required=True,
                        help='Path to preprocessed data directory')
    parser.add_argument('--original_data_path', type=str, default=None,
                        help='Path to original PTB-XL dataset')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum samples for testing')
    
    args = parser.parse_args()
    
    # Create a small test dataset
    dataset = PTBXLDataset(args.preprocessed_path, args.original_data_path, max_samples=args.max_samples)
    
    # Print some statistics
    print(f"Dataset size: {len(dataset)}")
    print(f"Signal shape: {dataset.get_signal_shape()}")
    print(f"Flat signal dimension: {dataset.get_flat_signal_dim()}")
    print(f"Heartbeat dimension: {dataset.get_heartbeat_dim()}")
    
    # Test data loader
    train_loader, test_loader, _ = create_data_loaders(
        args.preprocessed_path, batch_size=4, original_data_path=args.original_data_path, max_samples=args.max_samples
    )
    
    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"Batch signal shape: {batch['signal'].shape}")
    print(f"Batch flat signal shape: {batch['signal_flat'].shape}")
    print(f"Batch heartbeats shape: {batch['heartbeats'].shape}")
    print(f"Sample report: {batch['report'][0]}")
