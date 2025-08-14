#!/usr/bin/env python3
"""
Simple test script to load and plot examples from the preprocessed dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import PTBXLDataset
import argparse


def plot_ecg_sample(sample_data, sample_idx, save_path=None):
    """Plot a single ECG sample with both full signal and heartbeats."""
    
    # Extract data
    signal = sample_data['signal'].numpy()  # Shape: (time_samples, leads)
    heartbeats = sample_data['heartbeats'].numpy()  # Shape: (225,) for 3 beats
    report = sample_data['report']
    ecg_id = sample_data['ecg_id']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'ECG Sample {sample_idx} (ID: {ecg_id})', fontsize=16)
    
    # Plot 1: Full ECG signal (Lead II - index 1)
    lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
    time_full = np.arange(len(lead_ii)) / 100  # 100 Hz sampling rate
    
    axes[0, 0].plot(time_full, lead_ii, 'b-', linewidth=0.8)
    axes[0, 0].set_title('Full ECG Signal (Lead II)', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Extracted heartbeats (3 beats concatenated)
    beat_samples = len(heartbeats) // 3  # 225 / 3 = 75 samples per beat
    time_beats = np.arange(len(heartbeats)) / 100  # Time for all beats
    
    axes[0, 1].plot(time_beats, heartbeats, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Extracted Heartbeats (3 beats, 750ms each)', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add vertical lines to separate beats
    for i in range(1, 3):
        beat_time = i * beat_samples / 100
        axes[0, 1].axvline(beat_time, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 3: Individual beats separated
    colors = ['red', 'green', 'blue']
    for i in range(3):
        start_idx = i * beat_samples
        end_idx = (i + 1) * beat_samples
        beat = heartbeats[start_idx:end_idx]
        beat_time = np.arange(len(beat)) / 100  # Time for single beat
        
        axes[1, 0].plot(beat_time, beat, color=colors[i], linewidth=1.5, 
                       label=f'Beat {i+1}', alpha=0.8)
    
    axes[1, 0].set_title('Individual Heartbeats Overlay', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Signal statistics and report
    axes[1, 1].axis('off')
    
    # Display statistics
    stats_text = f"""
ECG Statistics:
• Signal shape: {signal.shape}
• Heartbeats shape: {heartbeats.shape}
• Signal range: [{signal.min():.3f}, {signal.max():.3f}]
• Heartbeats range: [{heartbeats.min():.3f}, {heartbeats.max():.3f}]
• Mean amplitude: {signal.mean():.3f}
• Std amplitude: {signal.std():.3f}

Medical Report:
{report[:200]}{'...' if len(report) > 200 else ''}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Always save to default location if no save_path specified
        default_path = f"plots/ecg_sample_{sample_idx}_{ecg_id}.png"
        import os
        os.makedirs("plots", exist_ok=True)
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {default_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test preprocessed dataset loading and visualization')
    parser.add_argument('--preprocessed_path', type=str, default='preprocessed_data',
                        help='Path to preprocessed data directory')
    parser.add_argument('--original_data_path', type=str, default='physionet.org/files/ptb-xl/1.0.3/',
                        help='Path to original PTB-XL dataset (for timing features if needed)')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of samples to load and plot')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to load from dataset')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files instead of just displaying')
    
    args = parser.parse_args()
    
    print("Loading preprocessed dataset...")
    
    try:
        # Load dataset
        dataset = PTBXLDataset(
            preprocessed_path=args.preprocessed_path,
            original_data_path=args.original_data_path,
            max_samples=args.max_samples
        )
        
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        print(f"Signal shape: {dataset.get_signal_shape()}")
        print(f"Heartbeat dimension: {dataset.get_heartbeat_dim()}")
        print(f"Flat signal dimension: {dataset.get_flat_signal_dim()}")
        
        # Get random samples
        sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
        
        print(f"\nPlotting {len(sample_indices)} random samples:")
        
        for i, sample_idx in enumerate(sample_indices):
            print(f"\nSample {i+1}/{len(sample_indices)} (Dataset index: {sample_idx})")
            
            # Get sample data
            sample_data = dataset[sample_idx]
            
            # Print basic info
            print(f"  ECG ID: {sample_data['ecg_id']}")
            print(f"  Report: {sample_data['report'][:100]}...")
            
            # Plot
            if args.save_plots:
                import os
                os.makedirs("plots", exist_ok=True)
                save_path = f"plots/ecg_sample_{sample_idx}_id{sample_data['ecg_id']}.png"
            else:
                save_path = None
            plot_ecg_sample(sample_data, sample_idx, save_path)
            
            # Wait for user input to continue (except for saved plots)
            if not args.save_plots and i < len(sample_indices) - 1:
                input("Press Enter to continue to next sample...")
        
        print(f"\nTesting complete! Plotted {len(sample_indices)} samples.")
        
        # Test data loader as well
        print("\nTesting data loader...")
        from data_loader import create_data_loaders
        
        train_loader, test_loader, full_dataset = create_data_loaders(
            preprocessed_path=args.preprocessed_path,
            batch_size=4,
            test_fold=10,
            original_data_path=args.original_data_path,
            max_samples=args.max_samples
        )
        
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        
        print(f"Data loader test successful:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Sample batch shapes:")
        print(f"    - signals: {sample_batch['signal'].shape}")
        print(f"    - heartbeats: {sample_batch['heartbeats'].shape}")
        print(f"    - signal_flat: {sample_batch['signal_flat'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())