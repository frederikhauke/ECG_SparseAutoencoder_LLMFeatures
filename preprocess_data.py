#!/usr/bin/env python3
"""
ECG Data Preprocessing Script

This script preprocesses PTB-XL ECG data and saves the results to cached files
that can be loaded quickly by the data loader.
"""

import os
import torch
import numpy as np
import pandas as pd
import wfdb
import ast
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def extract_representative_heartbeats(signal_2d: np.ndarray, sampling_rate: int = 100, 
                                    detection_lead_idx: int = 1, num_beats: int = 3, 
                                    beat_duration_ms: int = 750, 
                                    qrs_offset_ms: int = 100) -> np.ndarray:
    """
    Extract representative heartbeats from ALL ECG leads.
    
    Args:
        signal_2d: ECG signal matrix (samples x leads)
        sampling_rate: Sampling rate in Hz
        detection_lead_idx: Lead index to use for R-peak detection (1 for lead II)
        num_beats: Number of heartbeats to extract per lead
        beat_duration_ms: Duration of each heartbeat in milliseconds
        qrs_offset_ms: Offset from QRS peak to center the beat (ms)
        
    Returns:
        Array of shape (num_leads * num_beats * beat_duration_samples,) containing concatenated heartbeats from all leads
    """
    try:
        import neurokit2 as nk
        
        # Use specified lead for R-peak detection (default Lead II)
        detection_lead = detection_lead_idx if signal_2d.shape[1] > detection_lead_idx else 0
        ecg_detection = signal_2d[:, detection_lead]
        
        # Process ECG to find R peaks using the detection lead
        signals, info = nk.ecg_process(ecg_detection, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
        
        if len(r_peaks) < num_beats:
            # If not enough peaks, repeat the available ones
            if len(r_peaks) == 0:
                # No peaks found, return zeros
                beat_samples = int(beat_duration_ms * sampling_rate / 1000)
                return np.zeros(signal_2d.shape[1] * num_beats * beat_samples)
            else:
                # Repeat available peaks to get required number
                r_peaks = np.tile(r_peaks, (num_beats // len(r_peaks) + 1))[:num_beats]
        
        # Calculate beat parameters
        beat_samples = int(beat_duration_ms * sampling_rate / 1000)  # 750ms in samples
        qrs_offset_samples = int(qrs_offset_ms * sampling_rate / 1000)  # 100ms offset
        
        # Extract beats from ALL leads
        all_heartbeats = []
        
        for lead_idx in range(signal_2d.shape[1]):  # Process each lead
            lead_signal = signal_2d[:, lead_idx]
            lead_heartbeats = []
            
            for i in range(num_beats):
                if i < len(r_peaks):
                    r_peak = r_peaks[i]
                    # Center the beat 100ms after R peak
                    start_idx = r_peak + qrs_offset_samples - beat_samples // 2
                    end_idx = start_idx + beat_samples
                    
                    # Handle edge cases
                    if start_idx < 0:
                        # Pad with zeros at the beginning
                        beat = np.zeros(beat_samples)
                        valid_start = 0
                        beat[abs(start_idx):] = lead_signal[0:end_idx]
                    elif end_idx > len(lead_signal):
                        # Pad with zeros at the end
                        beat = np.zeros(beat_samples)
                        valid_end = len(lead_signal) - start_idx
                        beat[:valid_end] = lead_signal[start_idx:len(lead_signal)]
                    else:
                        # Normal case
                        beat = lead_signal[start_idx:end_idx]
                    
                    lead_heartbeats.append(beat)
                else:
                    # If we run out of peaks, duplicate the last one
                    lead_heartbeats.append(lead_heartbeats[-1].copy())
            
            # Add this lead's heartbeats to the collection
            all_heartbeats.extend(lead_heartbeats)
        
        # Concatenate all heartbeats from all leads
        return np.concatenate(all_heartbeats)
        
    except Exception as e:
        print(f"Error extracting heartbeats: {e}")
        # Return zeros as fallback
        beat_samples = int(beat_duration_ms * sampling_rate / 1000)
        return np.zeros(signal_2d.shape[1] * num_beats * beat_samples)


def process_single_ecg(args):
    """
    Process a single ECG file. This function is designed to be called by ThreadPoolExecutor.
    
    Args:
        args: Tuple containing (ecg_id, row, data_path, sampling_rate)
        
    Returns:
        Tuple of (ecg_id, signal, heartbeats, success_flag)
    """
    ecg_id, row, data_path, sampling_rate = args
    
    try:
        # Get filename based on sampling rate
        filename = row['filename_lr'] if sampling_rate == 100 else row['filename_hr']
        signal_path = data_path / filename
        
        # Load ECG signal
        signal, _ = wfdb.rdsamp(str(signal_path))
        
        # Extract heartbeats from ALL leads
        heartbeats = extract_representative_heartbeats(
            signal, 
            sampling_rate=sampling_rate,
            detection_lead_idx=1,  # Use Lead II for R-peak detection
            num_beats=3,
            beat_duration_ms=750,
            qrs_offset_ms=100
        )
        
        return ecg_id, signal, heartbeats, True
        
    except Exception as e:
        # Return error info for logging
        return ecg_id, None, None, False


def normalize_signals(signals: np.ndarray, scale_min: float = -1.0, scale_max: float = 2.0) -> np.ndarray:
    """Apply fixed linear rescaling from [scale_min, scale_max] to [0,1]."""
    rng = (scale_max - scale_min)
    if rng <= 0:
        raise ValueError("scale_max must be greater than scale_min")
    
    normalized_signals = []
    for signal in signals:
        # Linear mapping
        scaled = (signal - scale_min) / rng
        # Clip to [0,1] to avoid extreme outliers
        scaled = np.clip(scaled, 0.0, 1.0)
        normalized_signals.append(scaled)
    
    return np.array(normalized_signals)


def preprocess_ptbxl_data(data_path: str, output_path: str, sampling_rate: int = 100, 
                         normalize: bool = True, max_samples: Optional[int] = None,
                         scale_min: float = -1.0, scale_max: float = 2.0, 
                         max_workers: int = 16):
    """
    Preprocess PTB-XL dataset and save cached results using multithreading.
    
    Args:
        data_path: Path to PTB-XL dataset
        output_path: Path to save preprocessed data
        sampling_rate: Sampling rate (100 or 500 Hz)
        normalize: Whether to apply fixed linear rescaling to [0,1]
        max_samples: Maximum number of samples to process (for testing)
        scale_min: Assumed lower bound of ECG amplitude (mV)
        scale_max: Assumed upper bound of ECG amplitude (mV)
        max_workers: Maximum number of threads (default: 16)
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    print(f"Preprocessing PTB-XL data from {data_path}")
    print(f"Output will be saved to {output_path}")
    print(f"Sampling rate: {sampling_rate}Hz")
    print(f"Normalize: {normalize}")
    if max_samples:
        print(f"Processing only first {max_samples} samples")
    
    # Load metadata
    print("Loading metadata...")
    metadata = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
    
    # Parse SCP codes
    metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Limit samples if specified
    if max_samples:
        metadata = metadata.head(max_samples)
    
    # Load ECG signals using multithreading
    print(f"Loading ECG signals at {sampling_rate}Hz using multithreading...")
    
    # Use the specified number of workers (default is 16)
    
    print(f"Using {max_workers} worker threads")
    
    # Prepare arguments for parallel processing
    process_args = []
    for idx, (ecg_id, row) in enumerate(metadata.iterrows()):
        process_args.append((ecg_id, row, data_path, sampling_rate))
    
    signals = []
    heartbeats_list = []
    valid_indices = []
    failed_files = []
    
    # Process files in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_ecg, args): idx 
            for idx, args in enumerate(process_args)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(process_args), desc="Processing signals") as pbar:
            for future in as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                ecg_id, signal, heartbeats, success = future.result()
                
                if success:
                    signals.append(signal)
                    heartbeats_list.append(heartbeats)
                    valid_indices.append(original_idx)
                else:
                    failed_files.append(ecg_id)
                
                pbar.update(1)
    
    # Log failed files
    if failed_files:
        print(f"Failed to process {len(failed_files)} files: {failed_files[:10]}{'...' if len(failed_files) > 10 else ''}")
    
    # Convert to arrays
    signals = np.array(signals)
    heartbeats_array = np.array(heartbeats_list)
    metadata = metadata.iloc[valid_indices].reset_index()
    
    print(f"Successfully loaded {len(signals)} ECG signals")
    print(f"Signal shape: {signals.shape}")
    print(f"Heartbeats shape: {heartbeats_array.shape}")
    
    # Apply normalization if requested
    if normalize:
        print("Normalizing signals...")
        signals = normalize_signals(signals, scale_min=scale_min, scale_max=scale_max)
        # Also normalize heartbeats
        heartbeats_array = normalize_signals(heartbeats_array.reshape(-1, 1), scale_min=scale_min, scale_max=scale_max).flatten().reshape(heartbeats_array.shape)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save signals
    np.save(output_path / 'signals.npy', signals)
    
    # Save heartbeats
    np.save(output_path / 'heartbeats.npy', heartbeats_array)
    
    # Save metadata
    metadata.to_csv(output_path / 'metadata.csv', index=False)
    
    # Save preprocessing config
    config = {
        'sampling_rate': sampling_rate,
        'normalize': normalize,
        'scale_min': scale_min,
        'scale_max': scale_max,
        'num_samples': len(signals),
        'signal_shape': signals.shape,
        'heartbeats_shape': heartbeats_array.shape
    }
    
    with open(output_path / 'preprocessing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Preprocessing complete! Data saved to {output_path}")
    print(f"Files created:")
    print(f"  - signals.npy: {signals.shape}")
    print(f"  - heartbeats.npy: {heartbeats_array.shape}")
    print(f"  - metadata.csv: {len(metadata)} rows")
    print(f"  - preprocessing_config.json")


def main():
    parser = argparse.ArgumentParser(description='Preprocess PTB-XL ECG dataset')
    parser.add_argument('--data_path', type=str, default="physionet.org/files/ptb-xl/1.0.3/",
                        help='Path to PTB-XL dataset')
    parser.add_argument('--output_path', type=str, default='preprocessed_data',
                        help='Path to save preprocessed data')
    parser.add_argument('--sampling_rate', type=int, default=100, choices=[100, 500],
                        help='Sampling rate (100 or 500 Hz)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip normalization')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--scale_min', type=float, default=-1.0,
                        help='Minimum value for normalization range')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='Maximum value for normalization range')
    parser.add_argument('--config', type=str, default="config.json",
                        help='JSON config file to load parameters from')
    parser.add_argument('--max_workers', type=int, default=32,
                        help='Maximum number of worker threads (default: 16)')
    
    args = parser.parse_args()
    
    # Load config if provided and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override args with config values
        if 'data' in config:
            data_config = config['data']
            args.data_path = data_config.get('data_path', args.data_path)
            args.sampling_rate = data_config.get('sampling_rate', args.sampling_rate)
            args.max_samples = data_config.get('max_samples', args.max_samples)
            args.no_normalize = not data_config.get('normalize', not args.no_normalize)
    
    preprocess_ptbxl_data(
        data_path=args.data_path,
        output_path=args.output_path,
        sampling_rate=args.sampling_rate,
        normalize=not args.no_normalize,
        max_samples=args.max_samples,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()