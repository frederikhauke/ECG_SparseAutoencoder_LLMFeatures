#!/usr/bin/env python3
"""
Extract ECG time intervals (PR, QRS, QT) for the entire PTB-XL dataset using NeuroKit2.

This script processes all ECG records in the PTB-XL dataset and extracts:
- PR intervals (P onset to QRS onset)
- QRS duration (QRS onset to QRS offset)
- QT intervals (QRS onset to T offset)
- Heart rate and other derived metrics

The results are saved to a comprehensive CSV file with detailed metadata.

Usage:
    python extract_all_times.py [--data_path PATH] [--output times.csv] [--sr 100] [--lead II] [--batch_size 100]

Requirements:
    - neurokit2, scipy, wfdb, numpy, pandas
    - Sufficient disk space for output file (~50-100MB for full dataset)
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

try:
    import neurokit2 as nk
except ImportError:
    raise SystemExit("NeuroKit2 is required. Install with: pip install neurokit2 scipy")

from data_loader import PTBXLDataset
from tools.timing_extractor import get_timing_extractor


def load_lead_from_signal(signal_2d: np.ndarray, lead_name: str, 
                         header_info: Optional[Dict] = None) -> Tuple[np.ndarray, str]:
    """
    Extract a single lead from multi-lead ECG signal.
    
    Args:
        signal_2d: ECG signal matrix (time_points, leads)
        lead_name: Preferred lead name (e.g., 'II', 'V2')
        header_info: Optional header information with lead names
        
    Returns:
        Tuple of (lead_signal, actual_lead_name)
    """
    if header_info and 'sig_name' in header_info:
        sig_names = [name.upper() for name in header_info['sig_name']]
        target_lead = lead_name.upper()
        
        # Try exact match first
        if target_lead in sig_names:
            idx = sig_names.index(target_lead)
            return signal_2d[:, idx], target_lead
        
        # Fallback to preferred leads
        for preferred in ['II', 'I', 'V1', 'V2']:
            if preferred in sig_names:
                idx = sig_names.index(preferred)
                return signal_2d[:, idx], preferred
    
    # Default fallback: use column 1 (usually Lead II) or 0
    if signal_2d.shape[1] > 1:
        return signal_2d[:, 1], f"{lead_name}?"
    return signal_2d[:, 0], f"{lead_name}?"


def compute_ecg_intervals(ecg_signal: np.ndarray, sampling_rate: int, 
                         min_hr: float = 40, max_hr: float = 200) -> Dict:
    """
    Compute ECG intervals using NeuroKit2 delineation.
    
    Args:
        ecg_signal: Single-lead ECG signal
        sampling_rate: Sampling rate in Hz
        min_hr: Minimum acceptable heart rate (bpm)
        max_hr: Maximum acceptable heart rate (bpm)
        
    Returns:
        Dictionary with interval measurements and metadata
    """
    try:
        # Process ECG: clean and detect R-peaks
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        
        # Quality check: reasonable heart rate
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            return _create_empty_result("insufficient_beats")
        
        # Calculate heart rate
        rr_intervals = np.diff(r_peaks) / sampling_rate  # seconds
        heart_rate = 60.0 / np.mean(rr_intervals)  # bpm
        
        if heart_rate < min_hr or heart_rate > max_hr:
            return _create_empty_result("unrealistic_hr")
        
        # Wave delineation
        try:
            delineate_result = nk.ecg_delineate(
                signals["ECG_Clean"], 
                r_peaks, 
                sampling_rate=sampling_rate, 
                method="dwt"
            )
            
            # Handle different API versions
            if isinstance(delineate_result, tuple) and len(delineate_result) == 2:
                _, waves_dict = delineate_result
            elif isinstance(delineate_result, dict):
                waves_dict = delineate_result
            else:
                return _create_empty_result("delineation_failed")
                
        except Exception:
            return _create_empty_result("delineation_error")
        
        # Extract fiducial points
        def extract_points(key: str) -> np.ndarray:
            points = waves_dict.get(key, [])
            if points is None:
                return np.array([], dtype=int)
            points_array = np.array(points, dtype=float)
            return points_array[~np.isnan(points_array)].astype(int)
        
        p_onsets = extract_points("ECG_P_Onsets")
        q_peaks = extract_points("ECG_Q_Peaks")  # QRS onset approximation
        s_peaks = extract_points("ECG_S_Peaks")  # QRS offset approximation
        t_offsets = extract_points("ECG_T_Offsets")
        
        # Compute beat-wise intervals
        pr_intervals = []
        qrs_durations = []
        qt_intervals = []
        
        for r_peak in r_peaks:
            # Find closest fiducial points
            p_onset = _find_closest_before(p_onsets, r_peak)
            q_peak = _find_closest_around(q_peaks, r_peak, window=int(0.1 * sampling_rate))
            s_peak = _find_closest_around(s_peaks, r_peak, window=int(0.1 * sampling_rate))
            t_offset = _find_closest_after(t_offsets, r_peak)
            
            # Use R-peak as fallback for QRS if Q not detected
            qrs_onset = q_peak if q_peak is not None else r_peak
            qrs_offset = s_peak if s_peak is not None else r_peak
            
            # Calculate intervals in milliseconds
            if p_onset is not None and qrs_onset > p_onset:
                pr_ms = (qrs_onset - p_onset) * 1000.0 / sampling_rate
                if 80 <= pr_ms <= 300:  # Physiological range
                    pr_intervals.append(pr_ms)
            
            if qrs_offset > qrs_onset:
                qrs_ms = (qrs_offset - qrs_onset) * 1000.0 / sampling_rate
                if 60 <= qrs_ms <= 200:  # Physiological range
                    qrs_durations.append(qrs_ms)
            
            if t_offset is not None and t_offset > qrs_onset:
                qt_ms = (t_offset - qrs_onset) * 1000.0 / sampling_rate
                if 200 <= qt_ms <= 600:  # Physiological range
                    qt_intervals.append(qt_ms)
        
        # Calculate statistics
        def calc_stats(intervals: List[float]) -> Tuple[float, float, int]:
            if not intervals:
                return np.nan, np.nan, 0
            arr = np.array(intervals)
            return float(np.mean(arr)), float(np.std(arr)), len(arr)
        
        pr_mean, pr_std, pr_count = calc_stats(pr_intervals)
        qrs_mean, qrs_std, qrs_count = calc_stats(qrs_durations)
        qt_mean, qt_std, qt_count = calc_stats(qt_intervals)
        
        # QTc calculation (Bazett's formula: QTc = QT / sqrt(RR))
        qtc_mean = np.nan
        if not np.isnan(qt_mean) and len(rr_intervals) > 0:
            mean_rr = np.mean(rr_intervals)
            qtc_mean = qt_mean / np.sqrt(mean_rr)
        
        return {
            'status': 'success',
            'heart_rate_bpm': heart_rate,
            'rr_mean_ms': np.mean(rr_intervals) * 1000,
            'rr_std_ms': np.std(rr_intervals) * 1000,
            'total_beats': len(r_peaks),
            'pr_mean_ms': pr_mean,
            'pr_std_ms': pr_std,
            'pr_count': pr_count,
            'qrs_mean_ms': qrs_mean,
            'qrs_std_ms': qrs_std,
            'qrs_count': qrs_count,
            'qt_mean_ms': qt_mean,
            'qt_std_ms': qt_std,
            'qt_count': qt_count,
            'qtc_mean_ms': qtc_mean,
            'signal_length_s': len(ecg_signal) / sampling_rate,
            'sampling_rate': sampling_rate
        }
        
    except Exception as e:
        return _create_empty_result(f"processing_error: {str(e)[:100]}")


def _create_empty_result(status: str) -> Dict:
    """Create empty result dictionary for failed processing."""
    return {
        'status': status,
        'heart_rate_bpm': np.nan,
        'rr_mean_ms': np.nan,
        'rr_std_ms': np.nan,
        'total_beats': 0,
        'pr_mean_ms': np.nan,
        'pr_std_ms': np.nan,
        'pr_count': 0,
        'qrs_mean_ms': np.nan,
        'qrs_std_ms': np.nan,
        'qrs_count': 0,
        'qt_mean_ms': np.nan,
        'qt_std_ms': np.nan,
        'qt_count': 0,
        'qtc_mean_ms': np.nan,
        'signal_length_s': np.nan,
        'sampling_rate': np.nan
    }


def _find_closest_before(points: np.ndarray, target: int) -> Optional[int]:
    """Find closest point before target."""
    before_points = points[points < target]
    return int(before_points[-1]) if len(before_points) > 0 else None


def _find_closest_after(points: np.ndarray, target: int) -> Optional[int]:
    """Find closest point after target."""
    after_points = points[points > target]
    return int(after_points[0]) if len(after_points) > 0 else None


def _find_closest_around(points: np.ndarray, target: int, window: int) -> Optional[int]:
    """Find closest point within window around target."""
    nearby_points = points[np.abs(points - target) <= window]
    if len(nearby_points) == 0:
        return None
    distances = np.abs(nearby_points - target)
    closest_idx = np.argmin(distances)
    return int(nearby_points[closest_idx])


def process_dataset_batch(dataset: PTBXLDataset, start_idx: int, end_idx: int, 
                         lead_name: str, data_path: Path) -> List[Dict]:
    """
    Process a batch of ECG records.
    
    Args:
        dataset: PTB-XL dataset instance
        start_idx: Starting index in dataset
        end_idx: Ending index in dataset
        lead_name: Preferred ECG lead name
        data_path: Path to dataset files
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for idx in range(start_idx, min(end_idx, len(dataset))):
        try:
            # Get sample
            sample = dataset[idx]
            row_meta = dataset.metadata.iloc[sample['idx']]
            
            # Load header for lead names
            sampling_rate = dataset.sampling_rate
            fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
            filename = row_meta[fname_col]
            
            header_info = None
            try:
                header_path = data_path / filename
                header = wfdb.rdheader(str(header_path))
                header_info = {'sig_name': header.sig_name}
            except Exception:
                header_info = None
            
            # Extract lead
            signal_2d = sample['signal'].numpy()
            ecg_1d, actual_lead = load_lead_from_signal(signal_2d, lead_name, header_info)
            
            # Compute intervals
            intervals = compute_ecg_intervals(ecg_1d, sampling_rate)
            
            # Compile result
            result = {
                'ecg_id': int(row_meta.name) if hasattr(row_meta.name, 'dtype') else row_meta.name,
                'filename': filename,
                'lead_used': actual_lead,
                'age': row_meta.get('age', np.nan),
                'sex': row_meta.get('sex', ''),
                'report': str(row_meta.get('report', '')).strip(),
                'strat_fold': row_meta.get('strat_fold', np.nan),
                **intervals
            }
            
            results.append(result)
            
        except Exception as e:
            # Create error record
            error_result = {
                'ecg_id': idx,
                'filename': 'unknown',
                'lead_used': lead_name,
                'age': np.nan,
                'sex': '',
                'report': '',
                'strat_fold': np.nan,
                **_create_empty_result(f"sample_error: {str(e)[:50]}")
            }
            results.append(error_result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract ECG intervals for entire PTB-XL dataset"
    )
    parser.add_argument("--data_path", 
                       default="physionet.org/files/ptb-xl/1.0.3/",
                       help="Path to PTB-XL dataset")
    parser.add_argument("--output", 
                       default="times.csv",
                       help="Output CSV filename")
    parser.add_argument("--sr", type=int, default=100, choices=[100, 500],
                       help="Sampling rate (100 or 500 Hz)")
    parser.add_argument("--lead", default="II",
                       help="Preferred ECG lead")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = Path(args.data_path)
    output_path = Path(args.output)
    
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"ECG Interval Extraction for PTB-XL Dataset")
    print(f"Data path: {data_path}")
    print(f"Output: {output_path}")
    print(f"Sampling rate: {args.sr} Hz")
    print(f"Preferred lead: {args.lead}")
    print(f"Batch size: {args.batch_size}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    start_time = time.time()
    
    dataset = PTBXLDataset(
        str(data_path), 
        sampling_rate=args.sr, 
        normalize=False,  # Keep raw signals for better delineation
        max_samples=args.max_samples
    )
    
    total_samples = len(dataset)
    print(f"Loaded {total_samples:,} ECG records in {time.time() - start_time:.1f}s")
    
    # Process in batches
    print(f"\nProcessing ECG intervals...")
    all_results = []
    
    with tqdm(total=total_samples, desc="Processing ECGs") as pbar:
        for batch_start in range(0, total_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_samples)
            
            batch_results = process_dataset_batch(
                dataset, batch_start, batch_end, args.lead, data_path
            )
            all_results.extend(batch_results)
            
            pbar.update(len(batch_results))
    
    # Create DataFrame and save
    print(f"\nSaving results to {output_path}")
    df = pd.DataFrame(all_results)
    
    # Add processing metadata
    df['processing_timestamp'] = pd.Timestamp.now()
    df['neurokit_version'] = nk.__version__
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print(f"="*60)
    print(f"Total records processed: {len(df):,}")
    print(f"Successful analyses: {(df['status'] == 'success').sum():,}")
    print(f"Failed analyses: {(df['status'] != 'success').sum():,}")
    
    if (df['status'] == 'success').any():
        success_df = df[df['status'] == 'success']
        print(f"\nInterval Statistics (successful analyses):")
        print(f"Heart Rate: {success_df['heart_rate_bpm'].mean():.1f}±{success_df['heart_rate_bpm'].std():.1f} bpm")
        print(f"PR interval: {success_df['pr_mean_ms'].mean():.0f}±{success_df['pr_mean_ms'].std():.0f} ms")
        print(f"QRS duration: {success_df['qrs_mean_ms'].mean():.0f}±{success_df['qrs_mean_ms'].std():.0f} ms")
        print(f"QT interval: {success_df['qt_mean_ms'].mean():.0f}±{success_df['qt_mean_ms'].std():.0f} ms")
        print(f"QTc (Bazett): {success_df['qtc_mean_ms'].mean():.0f}±{success_df['qtc_mean_ms'].std():.0f} ms")
    
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show failure reasons if any
    if (df['status'] != 'success').any():
        print(f"\nFailure reasons:")
        failure_counts = df[df['status'] != 'success']['status'].value_counts()
        for reason, count in failure_counts.head(10).items():
            print(f"  {reason}: {count:,} records")


if __name__ == "__main__":
    main()
