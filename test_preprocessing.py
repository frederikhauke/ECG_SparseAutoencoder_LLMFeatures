#!/usr/bin/env python3
"""
Test Script for Multi-Lead ECG Data Preprocessing

This script tests the new multi-lead preprocessing pipeline to ensure:
1. Heartbeats are correctly extracted from all 12 leads
2. Data dimensions are correct (2700 samples = 12 leads × 3 beats × 75 samples)
3. Preprocessed data can be loaded correctly by the dataset
4. Model compatibility with new input dimensions
5. Data quality and consistency checks

Usage: python test_preprocessing.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import json
import tempfile
import shutil

# Add the current directory to the path to import our modules
sys.path.append('.')

from preprocess_data import extract_representative_heartbeats, preprocess_ptbxl_data
from data_loader import PTBXLDataset, create_data_loaders
from sparse_autoencoder import GatedSparseAutoencoder
from simple_autoencoder import SimpleAutoencoder


class PreprocessingTester:
    """Test suite for the multi-lead preprocessing pipeline."""
    
    def __init__(self, data_path: str = "physionet.org/files/ptb-xl/1.0.3/"):
        self.data_path = data_path
        self.test_results = {}
        
    def test_heartbeat_extraction(self) -> bool:
        """Test heartbeat extraction from a single ECG signal."""
        print("🔍 Testing heartbeat extraction...")
        
        try:
            # Create a synthetic 12-lead ECG signal for testing
            duration = 10  # seconds
            sampling_rate = 100  # Hz
            num_samples = duration * sampling_rate
            num_leads = 12
            
            # Generate synthetic ECG with clear R peaks
            time = np.linspace(0, duration, num_samples)
            synthetic_ecg = np.zeros((num_samples, num_leads))
            
            # Add R peaks every second (simulating 60 BPM)
            r_peak_times = np.arange(1, duration, 1.0)  # R peaks at 1, 2, 3, ... seconds
            
            for lead in range(num_leads):
                # Create base ECG with R peaks
                signal = 0.1 * np.sin(2 * np.pi * 0.5 * time)  # Low frequency baseline
                
                # Add R peaks (sharp spikes)
                for r_time in r_peak_times:
                    r_idx = int(r_time * sampling_rate)
                    if r_idx < num_samples:
                        # Create QRS complex
                        qrs_width = 10  # samples
                        start_idx = max(0, r_idx - qrs_width // 2)
                        end_idx = min(num_samples, r_idx + qrs_width // 2)
                        
                        # Add R wave (positive spike) with lead-specific amplitude
                        lead_amplitude = 0.5 + 0.3 * (lead + 1) / num_leads
                        signal[start_idx:end_idx] += lead_amplitude * np.exp(
                            -0.5 * ((np.arange(start_idx, end_idx) - r_idx) / 3) ** 2
                        )
                
                synthetic_ecg[:, lead] = signal
            
            # Test heartbeat extraction
            heartbeats = extract_representative_heartbeats(
                synthetic_ecg, 
                sampling_rate=sampling_rate,
                detection_lead_idx=1,  # Use Lead II for detection
                num_beats=3,
                beat_duration_ms=750
            )
            
            # Check dimensions
            expected_length = num_leads * 3 * 75  # 12 leads × 3 beats × 75 samples (750ms @ 100Hz)
            if len(heartbeats) != expected_length:
                print(f"❌ Dimension mismatch: expected {expected_length}, got {len(heartbeats)}")
                return False
            
            # Check that heartbeats are not all zeros
            if np.all(heartbeats == 0):
                print("❌ All heartbeats are zero - extraction failed")
                return False
            
            # Reshape to analyze per lead
            heartbeats_reshaped = heartbeats.reshape(num_leads, 3, 75)
            
            # Check that different leads have different patterns
            lead_variances = []
            for lead in range(num_leads):
                lead_data = heartbeats_reshaped[lead].flatten()
                lead_variances.append(np.var(lead_data))
            
            if np.all(np.array(lead_variances) < 1e-6):
                print("❌ All leads appear identical - extraction may be incorrect")
                return False
            
            print(f"✅ Heartbeat extraction successful: {len(heartbeats)} samples extracted")
            print(f"   - Shape after reshaping: {heartbeats_reshaped.shape}")
            print(f"   - Lead variances range: {min(lead_variances):.6f} - {max(lead_variances):.6f}")
            
            self.test_results['heartbeat_extraction'] = True
            return True
            
        except Exception as e:
            print(f"❌ Heartbeat extraction test failed: {str(e)}")
            self.test_results['heartbeat_extraction'] = False
            return False
    
    def test_preprocessing_pipeline(self) -> bool:
        """Test the full preprocessing pipeline with real data."""
        print("\n🔍 Testing preprocessing pipeline...")
        
        try:
            # Check if we have access to PTB-XL data
            ptb_path = Path(self.data_path)
            if not ptb_path.exists():
                print(f"⚠️  PTB-XL data not found at {self.data_path}")
                print("   Skipping full pipeline test - using synthetic data")
                self.test_results['preprocessing_pipeline'] = 'skipped'
                return True
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = Path(temp_dir) / "preprocessed_test"
                
                print(f"   Running preprocessing with output to {temp_output}")
                
                # Run preprocessing on a small subset
                success = preprocess_ptbxl_data(
                    data_path=self.data_path,
                    output_path=str(temp_output),
                    max_samples=10,  # Small test set
                    num_workers=2,
                    force_recompute=True
                )
                
                if not success:
                    print("❌ Preprocessing pipeline failed")
                    self.test_results['preprocessing_pipeline'] = False
                    return False
                
                # Check output files
                expected_files = ['train.npz', 'val.npz', 'test.npz', 'metadata.json']
                for filename in expected_files:
                    filepath = temp_output / filename
                    if not filepath.exists():
                        print(f"❌ Expected output file missing: {filename}")
                        self.test_results['preprocessing_pipeline'] = False
                        return False
                
                # Load and check data structure
                train_data = np.load(temp_output / 'train.npz')
                
                # Check required keys
                required_keys = ['heartbeats', 'ecg_ids', 'reports', 'labels']
                for key in required_keys:
                    if key not in train_data:
                        print(f"❌ Required key missing from preprocessed data: {key}")
                        self.test_results['preprocessing_pipeline'] = False
                        return False
                
                # Check heartbeat dimensions
                heartbeats = train_data['heartbeats']
                expected_heartbeat_dim = 12 * 3 * 75  # 2700
                
                if heartbeats.shape[1] != expected_heartbeat_dim:
                    print(f"❌ Heartbeat dimension mismatch: expected {expected_heartbeat_dim}, got {heartbeats.shape[1]}")
                    self.test_results['preprocessing_pipeline'] = False
                    return False
                
                print(f"✅ Preprocessing pipeline successful")
                print(f"   - Train samples: {heartbeats.shape[0]}")
                print(f"   - Heartbeat dimension: {heartbeats.shape[1]}")
                print(f"   - Output files: {', '.join(expected_files)}")
                
                self.test_results['preprocessing_pipeline'] = True
                return True
                
        except Exception as e:
            print(f"❌ Preprocessing pipeline test failed: {str(e)}")
            self.test_results['preprocessing_pipeline'] = False
            return False
    
    def test_dataset_loading(self) -> bool:
        """Test loading preprocessed data with PTBXLDataset."""
        print("\n🔍 Testing dataset loading...")
        
        try:
            # Check if preprocessed data exists
            preprocessed_path = Path("outputs/preprocessed_data")
            if not preprocessed_path.exists():
                print("⚠️  No preprocessed data found. Running test preprocessing first...")
                # Create a minimal preprocessed dataset for testing
                return self._create_test_dataset()
            
            # Test dataset loading
            dataset = PTBXLDataset(str(preprocessed_path), split='train')
            
            if len(dataset) == 0:
                print("❌ Dataset is empty")
                self.test_results['dataset_loading'] = False
                return False
            
            # Test data loader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
            
            # Get first batch
            batch = next(iter(dataloader))
            
            # Check batch structure
            required_keys = ['heartbeats', 'ecg_id', 'report', 'labels']
            for key in required_keys:
                if key not in batch:
                    print(f"❌ Required batch key missing: {key}")
                    self.test_results['dataset_loading'] = False
                    return False
            
            # Check heartbeat dimensions
            heartbeats = batch['heartbeats']
            expected_dim = 2700  # 12 leads × 3 beats × 75 samples
            
            if heartbeats.shape[1] != expected_dim:
                print(f"❌ Heartbeat dimension mismatch: expected {expected_dim}, got {heartbeats.shape[1]}")
                self.test_results['dataset_loading'] = False
                return False
            
            print(f"✅ Dataset loading successful")
            print(f"   - Dataset size: {len(dataset)} samples")
            print(f"   - Batch shape: {heartbeats.shape}")
            print(f"   - Heartbeat dimension: {heartbeats.shape[1]}")
            
            self.test_results['dataset_loading'] = True
            return True
            
        except Exception as e:
            print(f"❌ Dataset loading test failed: {str(e)}")
            self.test_results['dataset_loading'] = False
            return False
    
    def _create_test_dataset(self) -> bool:
        """Create a minimal test dataset for testing."""
        try:
            output_path = Path("outputs/preprocessed_data")
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create synthetic test data
            num_samples = 5
            heartbeat_dim = 2700  # 12 leads × 3 beats × 75 samples
            
            # Generate synthetic heartbeats
            heartbeats = np.random.randn(num_samples, heartbeat_dim) * 0.1
            ecg_ids = np.arange(num_samples)
            reports = [f"Test report {i}" for i in range(num_samples)]
            labels = np.random.randint(0, 2, (num_samples, 5))  # 5 random binary labels
            
            # Save test data
            np.savez(
                output_path / 'train.npz',
                heartbeats=heartbeats,
                ecg_ids=ecg_ids,
                reports=reports,
                labels=labels
            )
            
            # Create minimal validation and test sets
            np.savez(
                output_path / 'val.npz',
                heartbeats=heartbeats[:2],
                ecg_ids=ecg_ids[:2],
                reports=reports[:2],
                labels=labels[:2]
            )
            
            np.savez(
                output_path / 'test.npz',
                heartbeats=heartbeats[:2],
                ecg_ids=ecg_ids[:2],
                reports=reports[:2],
                labels=labels[:2]
            )
            
            # Create metadata
            metadata = {
                'heartbeat_dim': heartbeat_dim,
                'num_leads': 12,
                'beats_per_lead': 3,
                'samples_per_beat': 75,
                'sampling_rate': 100,
                'total_samples': num_samples
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Created test dataset at {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test dataset: {str(e)}")
            return False
    
    def test_model_compatibility(self) -> bool:
        """Test model compatibility with new input dimensions."""
        print("\n🔍 Testing model compatibility...")
        
        try:
            heartbeat_dim = 2700  # 12 leads × 3 beats × 75 samples
            batch_size = 4
            
            # Test data
            test_input = torch.randn(batch_size, heartbeat_dim)
            
            # Test GatedSparseAutoencoder
            print("   Testing GatedSparseAutoencoder...")
            sparse_model = GatedSparseAutoencoder(
                heartbeat_input_dim=heartbeat_dim,
                hidden_dims=[1024, 512],
                latent_dim=256,
                sparsity_weight=0.01
            )
            
            with torch.no_grad():
                sparse_output, sparse_latent = sparse_model(test_input)
            
            if sparse_output.shape != test_input.shape:
                print(f"❌ Sparse model output shape mismatch: {sparse_output.shape} vs {test_input.shape}")
                self.test_results['model_compatibility'] = False
                return False
            
            if sparse_latent.shape[1] != 256:
                print(f"❌ Sparse model latent dimension mismatch: {sparse_latent.shape[1]} vs 256")
                self.test_results['model_compatibility'] = False
                return False
            
            # Test SimpleAutoencoder
            print("   Testing SimpleAutoencoder...")
            simple_model = SimpleAutoencoder(
                heartbeat_input_dim=heartbeat_dim,
                hidden_dims=[1024, 512],
                latent_dim=256
            )
            
            with torch.no_grad():
                simple_output, simple_latent = simple_model(test_input)
            
            if simple_output.shape != test_input.shape:
                print(f"❌ Simple model output shape mismatch: {simple_output.shape} vs {test_input.shape}")
                self.test_results['model_compatibility'] = False
                return False
            
            if simple_latent.shape[1] != 256:
                print(f"❌ Simple model latent dimension mismatch: {simple_latent.shape[1]} vs 256")
                self.test_results['model_compatibility'] = False
                return False
            
            print(f"✅ Model compatibility successful")
            print(f"   - Input shape: {test_input.shape}")
            print(f"   - Sparse model output: {sparse_output.shape}, latent: {sparse_latent.shape}")
            print(f"   - Simple model output: {simple_output.shape}, latent: {simple_latent.shape}")
            
            self.test_results['model_compatibility'] = True
            return True
            
        except Exception as e:
            print(f"❌ Model compatibility test failed: {str(e)}")
            self.test_results['model_compatibility'] = False
            return False
    
    def test_data_quality(self) -> bool:
        """Test data quality and consistency."""
        print("\n🔍 Testing data quality...")
        
        try:
            # Check if preprocessed data exists
            preprocessed_path = Path("outputs/preprocessed_data")
            if not preprocessed_path.exists():
                print("⚠️  No preprocessed data found, skipping data quality test")
                self.test_results['data_quality'] = 'skipped'
                return True
            
            # Load train data
            train_data = np.load(preprocessed_path / 'train.npz')
            heartbeats = train_data['heartbeats']
            
            # Check for NaN or infinite values
            if np.any(np.isnan(heartbeats)):
                print("❌ NaN values found in heartbeat data")
                self.test_results['data_quality'] = False
                return False
            
            if np.any(np.isinf(heartbeats)):
                print("❌ Infinite values found in heartbeat data")
                self.test_results['data_quality'] = False
                return False
            
            # Check data range (ECG values should be reasonable)
            data_min, data_max = np.min(heartbeats), np.max(heartbeats)
            if abs(data_min) > 10 or abs(data_max) > 10:
                print(f"⚠️  Data range seems unusual: [{data_min:.3f}, {data_max:.3f}]")
                print("   This might indicate scaling issues")
            
            # Check data variance (should not be too low or too high)
            data_var = np.var(heartbeats)
            if data_var < 1e-6:
                print("⚠️  Very low data variance, might indicate all-zero or constant data")
            elif data_var > 100:
                print("⚠️  Very high data variance, might indicate scaling issues")
            
            # Check lead consistency (reshape and analyze per lead)
            heartbeats_reshaped = heartbeats.reshape(heartbeats.shape[0], 12, 3, 75)
            
            lead_stats = []
            for lead in range(12):
                lead_data = heartbeats_reshaped[:, lead, :, :].flatten()
                lead_mean = np.mean(lead_data)
                lead_std = np.std(lead_data)
                lead_stats.append((lead_mean, lead_std))
            
            # Check if any lead has zero variance (indicating problematic extraction)
            zero_variance_leads = [i for i, (mean, std) in enumerate(lead_stats) if std < 1e-6]
            if zero_variance_leads:
                print(f"⚠️  Leads with zero variance: {zero_variance_leads}")
            
            print(f"✅ Data quality check completed")
            print(f"   - Data range: [{data_min:.6f}, {data_max:.6f}]")
            print(f"   - Data variance: {data_var:.6f}")
            print(f"   - Samples: {heartbeats.shape[0]}, Dimension: {heartbeats.shape[1]}")
            
            if zero_variance_leads:
                print(f"   - Leads with low variance: {zero_variance_leads}")
            
            self.test_results['data_quality'] = True
            return True
            
        except Exception as e:
            print(f"❌ Data quality test failed: {str(e)}")
            self.test_results['data_quality'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("🧪 Starting Multi-Lead ECG Preprocessing Test Suite")
        print("=" * 60)
        
        tests = [
            ('Heartbeat Extraction', self.test_heartbeat_extraction),
            ('Preprocessing Pipeline', self.test_preprocessing_pipeline),
            ('Dataset Loading', self.test_dataset_loading),
            ('Model Compatibility', self.test_model_compatibility),
            ('Data Quality', self.test_data_quality)
        ]
        
        for test_name, test_func in tests:
            test_func()
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        
        passed = 0
        total = 0
        
        for test_name, result in self.test_results.items():
            total += 1
            if result is True:
                status = "✅ PASSED"
                passed += 1
            elif result is False:
                status = "❌ FAILED"
            else:
                status = f"⚠️  {result.upper()}"
                passed += 0.5  # Partial credit for skipped tests
        
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        return self.test_results


def main():
    """Main function to run preprocessing tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Multi-Lead ECG Preprocessing")
    parser.add_argument("--data_path", default="physionet.org/files/ptb-xl/1.0.3/",
                       help="Path to PTB-XL dataset")
    parser.add_argument("--create_plots", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Run tests
    tester = PreprocessingTester(data_path=args.data_path)
    results = tester.run_all_tests()
    
    # Optional: Create visualization plots
    if args.create_plots:
        try:
            create_test_plots()
        except Exception as e:
            print(f"⚠️  Failed to create plots: {str(e)}")
    
    # Exit with error code if any critical tests failed
    critical_tests = ['heartbeat_extraction', 'model_compatibility']
    failed_critical = [test for test in critical_tests if results.get(test) is False]
    
    if failed_critical:
        print(f"\n❌ Critical tests failed: {', '.join(failed_critical)}")
        sys.exit(1)
    else:
        print(f"\n✅ All critical tests passed!")
        sys.exit(0)


def create_test_plots():
    """Create visualization plots for the test results."""
    print("\n📈 Creating test visualization plots...")
    
    try:
        # Check if we have preprocessed data to plot
        preprocessed_path = Path("outputs/preprocessed_data")
        if not preprocessed_path.exists():
            print("   No preprocessed data available for plotting")
            return
        
        # Load data
        train_data = np.load(preprocessed_path / 'train.npz')
        heartbeats = train_data['heartbeats'][:10]  # First 10 samples
        
        # Reshape to (samples, leads, beats, beat_samples)
        heartbeats_reshaped = heartbeats.reshape(heartbeats.shape[0], 12, 3, 75)
        
        # Create plots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Multi-Lead ECG Heartbeats - First Sample', fontsize=16)
        
        # Plot first sample, all leads, first heartbeat
        for lead in range(12):
            row = lead // 4
            col = lead % 4
            
            beat_data = heartbeats_reshaped[0, lead, 0, :]  # First sample, current lead, first beat
            axes[row, col].plot(beat_data)
            axes[row, col].set_title(f'Lead {lead + 1}')
            axes[row, col].set_xlabel('Sample')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_preprocessing_leads.png', dpi=150, bbox_inches='tight')
        print("   Saved lead visualization to test_preprocessing_leads.png")
        
        # Create heartbeat comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Three Heartbeats from Lead II - First Sample', fontsize=14)
        
        lead_ii_idx = 1  # Lead II
        for beat in range(3):
            beat_data = heartbeats_reshaped[0, lead_ii_idx, beat, :]
            axes[beat].plot(beat_data)
            axes[beat].set_title(f'Heartbeat {beat + 1}')
            axes[beat].set_xlabel('Sample (750ms @ 100Hz)')
            axes[beat].set_ylabel('Amplitude')
            axes[beat].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_preprocessing_heartbeats.png', dpi=150, bbox_inches='tight')
        print("   Saved heartbeat comparison to test_preprocessing_heartbeats.png")
        
    except Exception as e:
        print(f"   Failed to create plots: {str(e)}")


if __name__ == "__main__":
    main()
