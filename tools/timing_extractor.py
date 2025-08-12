"""
Unified ECG timing extraction module with caching support.

This module provides a unified interface for extracting ECG timing features
across all scripts, with support for caching previously computed results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGTimingExtractor:
    """Unified ECG timing feature extractor with caching support."""
    
    def __init__(self, cache_path: str = "times.csv", use_cache: bool = True):
        """
        Initialize timing extractor.
        
        Args:
            cache_path: Path to cached timing results CSV file
            use_cache: Whether to use cached results when available
        """
        self.cache_path = Path(cache_path)
        self.use_cache = use_cache
        self._cache_df = None
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load timing cache from CSV file."""
        if self.use_cache and self.cache_path.exists():
            try:
                self._cache_df = pd.read_csv(self.cache_path)
                logger.info(f"Loaded timing cache with {len(self._cache_df)} records from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
                self._cache_df = None
        else:
            logger.info(f"Cache disabled or not found at {self.cache_path}")
            self._cache_df = None
    
    def get_timing_features(self, ecg_id: Union[int, str], 
                           signal_2d: Optional[np.ndarray] = None, 
                           sampling_rate: int = 100) -> np.ndarray:
        """
        Extract timing features for an ECG record.
        
        Args:
            ecg_id: ECG record identifier
            signal_2d: ECG signal matrix (optional, for fallback computation)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Array of [PR_ms, QRS_ms, QT_ms, HR_bpm]
        """
        # Try to get from cache first
        if self._cache_df is not None:
            cached_result = self._get_from_cache(ecg_id)
            if cached_result is not None:
                return cached_result
        
        # Fallback to real-time computation if signal provided
        if signal_2d is not None:
            logger.debug(f"Computing timing features for ECG {ecg_id}")
            return self._extract_timing_features_realtime(signal_2d, sampling_rate)
        
        # Return default values if no cache and no signal
        logger.warning(f"No cached timing data found for ECG {ecg_id} and no signal provided")
        return self._get_default_features()
    
    def _get_from_cache(self, ecg_id: Union[int, str]) -> Optional[np.ndarray]:
        """Get timing features from cache."""
        try:
            ecg_id = int(ecg_id)
            row = self._cache_df[self._cache_df['ecg_id'] == ecg_id]
            
            if len(row) == 0:
                return None
            
            row = row.iloc[0]
            
            # Only use successful extractions
            if row.get('status', '') != 'success':
                return None
            
            # Extract features, use defaults for NaN values
            pr_ms = row.get('pr_mean_ms', 150.0)
            qrs_ms = row.get('qrs_mean_ms', 80.0)
            qt_ms = row.get('qt_mean_ms', 400.0)
            hr_bpm = row.get('heart_rate_bpm', 70.0)
            
            # Handle NaN values
            if pd.isna(pr_ms):
                pr_ms = 150.0
            if pd.isna(qrs_ms):
                qrs_ms = 80.0
            if pd.isna(qt_ms):
                qt_ms = 400.0
            if pd.isna(hr_bpm):
                hr_bpm = 70.0
            
            return np.array([pr_ms, qrs_ms, qt_ms, hr_bpm])
            
        except Exception as e:
            logger.debug(f"Failed to get cached timing for ECG {ecg_id}: {e}")
            return None
    
    def _extract_timing_features_realtime(self, signal_2d: np.ndarray, 
                                         sampling_rate: int) -> np.ndarray:
        """Extract timing features using NeuroKit2 in real-time."""
        try:
            import neurokit2 as nk
            
            # Use lead II (column 1) or first available lead
            ecg_1d = signal_2d[:, 1] if signal_2d.shape[1] > 1 else signal_2d[:, 0]
            
            # Process ECG
            signals, info = nk.ecg_process(ecg_1d, sampling_rate=sampling_rate)
            r_peaks = info["ECG_R_Peaks"]
            
            if len(r_peaks) < 2:
                return self._get_default_features()
            
            # Delineate waves
            _, waves = nk.ecg_delineate(signals["ECG_Clean"], r_peaks, 
                                       sampling_rate=sampling_rate)
            
            # Extract timing features
            def get_points(key):
                pts = waves.get(key, [])
                return np.array(pts)[~np.isnan(pts)].astype(int) if pts is not None else np.array([])
            
            p_onsets = get_points("ECG_P_Onsets")
            q_peaks = get_points("ECG_Q_Peaks") 
            s_peaks = get_points("ECG_S_Peaks")
            t_offsets = get_points("ECG_T_Offsets")
            
            # Calculate intervals for first beat
            r_peak = r_peaks[0]
            
            # Find closest fiducials
            p_onset = p_onsets[p_onsets < r_peak][-1] if len(p_onsets[p_onsets < r_peak]) > 0 else None
            q_peak = q_peaks[np.abs(q_peaks - r_peak) <= 25][0] if len(q_peaks[np.abs(q_peaks - r_peak) <= 25]) > 0 else r_peak
            s_peak = s_peaks[np.abs(s_peaks - r_peak) <= 50][0] if len(s_peaks[np.abs(s_peaks - r_peak) <= 50]) > 0 else r_peak
            t_offset = t_offsets[t_offsets > r_peak][0] if len(t_offsets[t_offsets > r_peak]) > 0 else None
            
            # Calculate intervals in ms
            pr_ms = (q_peak - p_onset) * 1000 / sampling_rate if p_onset is not None else 150.0
            qrs_ms = (s_peak - q_peak) * 1000 / sampling_rate if s_peak != q_peak else 80.0
            qt_ms = (t_offset - q_peak) * 1000 / sampling_rate if t_offset is not None else 400.0
            hr_bpm = 60 / np.mean(np.diff(r_peaks) / sampling_rate)
            
            return np.array([pr_ms, qrs_ms, qt_ms, hr_bpm])
            
        except Exception as e:
            logger.debug(f"Real-time timing extraction failed: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> np.ndarray:
        """Return default timing features."""
        return np.array([150.0, 80.0, 400.0, 70.0])  # PR, QRS, QT, HR
    
    def get_batch_timing_features(self, ecg_ids: list, 
                                 signals: Optional[list] = None,
                                 sampling_rate: int = 100) -> np.ndarray:
        """
        Get timing features for a batch of ECG records.
        
        Args:
            ecg_ids: List of ECG identifiers
            signals: Optional list of ECG signal matrices
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Array of shape (batch_size, 4) with timing features
        """
        batch_features = []
        
        for i, ecg_id in enumerate(ecg_ids):
            signal = signals[i] if signals is not None else None
            features = self.get_timing_features(ecg_id, signal, sampling_rate)
            batch_features.append(features)
        
        return np.array(batch_features)
    
    def is_cache_available(self) -> bool:
        """Check if timing cache is available and loaded."""
        return self._cache_df is not None
    
    def get_cache_info(self) -> Dict:
        """Get information about the loaded cache."""
        if self._cache_df is None:
            return {"available": False}
        
        return {
            "available": True,
            "total_records": len(self._cache_df),
            "successful_records": (self._cache_df['status'] == 'success').sum(),
            "cache_path": str(self.cache_path),
            "columns": list(self._cache_df.columns)
        }


# Global extractor instance for easy import
_global_extractor = None


def get_timing_extractor(cache_path: str = "times.csv", 
                        use_cache: bool = True) -> ECGTimingExtractor:
    """Get or create global timing extractor instance."""
    global _global_extractor
    
    if _global_extractor is None or _global_extractor.cache_path != Path(cache_path):
        _global_extractor = ECGTimingExtractor(cache_path, use_cache)
    
    return _global_extractor


def extract_timing_features(ecg_id: Union[int, str], 
                           signal_2d: Optional[np.ndarray] = None,
                           sampling_rate: int = 100,
                           cache_path: str = "times.csv") -> np.ndarray:
    """
    Convenience function to extract timing features.
    
    Args:
        ecg_id: ECG record identifier
        signal_2d: Optional ECG signal matrix
        sampling_rate: Sampling rate in Hz
        cache_path: Path to cached timing results
        
    Returns:
        Array of [PR_ms, QRS_ms, QT_ms, HR_bpm]
    """
    extractor = get_timing_extractor(cache_path, use_cache=True)
    return extractor.get_timing_features(ecg_id, signal_2d, sampling_rate)
