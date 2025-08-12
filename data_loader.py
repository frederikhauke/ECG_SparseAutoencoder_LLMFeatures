import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wfdb
import ast
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tools.timing_extractor import get_timing_extractor


class PTBXLDataset(Dataset):
    """PTB-XL ECG Dataset loader."""
    
    def __init__(self, data_path: str, sampling_rate: int = 100, 
                 normalize: bool = True, max_samples: int = None):
        """
        Initialize PTB-XL dataset.
        
        Args:
            data_path: Path to PTB-XL dataset
            sampling_rate: Sampling rate (100 or 500 Hz)
            normalize: Whether to normalize the ECG signals
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        
        # Initialize timing extractor
        self.timing_extractor = get_timing_extractor(cache_path="times.csv", use_cache=True)
        
        # Load metadata
        self.metadata = pd.read_csv(
            self.data_path / 'ptbxl_database.csv', 
            index_col='ecg_id'
        )
        
        # Parse SCP codes
        self.metadata.scp_codes = self.metadata.scp_codes.apply(
            lambda x: ast.literal_eval(x)
        )
        
        # Limit samples if specified
        if max_samples:
            self.metadata = self.metadata.head(max_samples)
        
        # Load ECG signals
        self._load_signals()
        
    def _load_signals(self):
        """Load ECG signals from the dataset."""
        print(f"Loading ECG signals at {self.sampling_rate}Hz...")
        
        if self.sampling_rate == 100:
            filenames = self.metadata.filename_lr
        else:
            filenames = self.metadata.filename_hr
            
        signals = []
        valid_indices = []
        
        for idx, filename in enumerate(filenames):
            try:
                signal_path = self.data_path / filename
                signal, _ = wfdb.rdsamp(str(signal_path))
                signals.append(signal)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
        self.signals = np.array(signals)
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        
        if self.normalize:
            self._normalize_signals()
            
        print(f"Loaded {len(self.signals)} ECG signals")
        print(f"Signal shape: {self.signals.shape}")
        
    def _normalize_signals(self):
        """Normalize ECG signals to zero mean and unit variance."""
        # Normalize per signal across all leads and time points
        for i in range(len(self.signals)):
            signal = self.signals[i]
            self.signals[i] = (signal - signal.mean()) / (signal.std() + 1e-8)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx])
        
        # Flatten the signal for the autoencoder
        signal_flat = signal.flatten()
        
        # Extract timing features using unified extractor
        timing_features = self._extract_timing_features(idx)
        
        # Combine ECG and timing features
        combined_features = torch.cat([signal_flat, torch.FloatTensor(timing_features)])
        
        return {
            'signal': signal,
            'signal_flat': signal_flat,
            'timing_features': torch.FloatTensor(timing_features),
            'combined_features': combined_features,
            'report': self.metadata.iloc[idx]['report'],
            'ecg_id': self.metadata.index[idx],
            'idx': idx
        }
    
    def _extract_timing_features(self, idx: int) -> np.ndarray:
        """Extract timing features using unified extractor."""
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


def create_data_loaders(data_path: str, batch_size: int = 32, 
                       test_fold: int = 10, sampling_rate: int = 100,
                       max_samples: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        data_path: Path to PTB-XL dataset
        batch_size: Batch size for data loaders
        test_fold: Fold to use for testing (1-10)
        sampling_rate: Sampling rate (100 or 500 Hz)
        max_samples: Maximum number of samples to load
        
    Returns:
        train_loader, test_loader
    """
    # Load full dataset
    dataset = PTBXLDataset(data_path, sampling_rate, max_samples=max_samples)
    
    # Split indices based on stratified folds
    train_indices = []
    test_indices = []
    
    for idx, row in dataset.metadata.iterrows():
        if row['strat_fold'] == test_fold:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, dataset


if __name__ == "__main__":
    # Test the dataset loader
    data_path = "physionet.org/files/ptb-xl/1.0.3/"
    
    # Create a small test dataset
    dataset = PTBXLDataset(data_path, max_samples=100)
    
    # Print some statistics
    print(f"Dataset size: {len(dataset)}")
    print(f"Signal shape: {dataset.get_signal_shape()}")
    print(f"Flat signal dimension: {dataset.get_flat_signal_dim()}")
    
    # Test data loader
    train_loader, test_loader, _ = create_data_loaders(
        data_path, batch_size=4, max_samples=100
    )
    
    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"Batch signal shape: {batch['signal'].shape}")
    print(f"Batch flat signal shape: {batch['signal_flat'].shape}")
    print(f"Sample report: {batch['report'][0]}")
