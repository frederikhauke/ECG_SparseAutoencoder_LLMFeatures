#!/usr/bin/env python3
"""
Feature Interpretation Script for ECG Sparse Autoencoder

This script:
1. Loads a trained sparse autoencoder
2. For each latent feature, finds ECGs that activate it most strongly
3. Uses Azure OpenAI to analyze the clinical reports of these ECGs
4. Generates human-readable interpretations of what each feature represents

Usage: python interpret_features.py --model_path checkpoints/best_model.pth
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import json
import os
from collections import Counter

# Azure OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

from data_loader import create_data_loaders, PTBXLDataset
from sparse_autoencoder import GatedSparseAutoencoder
from simple_autoencoder import SimpleAutoencoder

# Load environment variables
load_dotenv()


class FeatureInterpreter:
    """Interprets sparse autoencoder features using Azure OpenAI."""
    
    def __init__(self, model_path: str, preprocessed_path: str, config_path: str = None, 
                 original_data_path: str = None, device: str = 'cpu'):
        """Initialize the feature interpreter."""
        self.model_path = model_path
        self.preprocessed_path = preprocessed_path
        self.config_path = config_path
        self.original_data_path = original_data_path
        self.device = device
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model, self.model_type = self._load_model()
        
        # Setup Azure OpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        print(f"Feature interpreter initialized")
        print(f"Model type: {self.model_type}")
        print(f"Model latent dimension: {self.model.latent_dim}")
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded config from {self.config_path}")
            return config
        else:
            print("No config file provided or found, using defaults")
            return {
                "model": {
                    "hidden_dims": [1024, 512],
                    "latent_dim": 256,
                    "sparsity_weight": 0.001,
                    "alpha_aux": 0.0,
                    "target_sparsity": 0.0,
                    "use_frozen_decoder_for_aux": False
                }
            }
    
    def _load_model(self) -> Tuple[torch.nn.Module, str]:
        """Load the trained autoencoder model (sparse or simple)."""
        # Force CPU mapping and use safe globals for PyTorch 2.6 compatibility
        with torch.serialization.safe_globals([
            np._core.multiarray.scalar, 
            np.dtype, 
            np.ndarray,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.dtypes.Int64DType,
            np.dtypes.Float64DType,
            np.dtypes.Int32DType,
            np.dtypes.Float32DType
        ]):
            checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract model configuration from checkpoint and external config
        checkpoint_config = checkpoint.get('model_config', {})
        external_model_config = self.config.get('model', {})
        
        # External config takes priority over checkpoint config
        combined_config = {**checkpoint_config, **external_model_config}
        
        # Get heartbeat input dimension from preprocessed data
        test_dataset = PTBXLDataset(self.preprocessed_path, self.original_data_path, max_samples=1)
        heartbeat_input_dim = test_dataset.get_heartbeat_dim()
        
        print(f"Detected heartbeat input dimension: {heartbeat_input_dim}")
        
        # Detect model type from checkpoint or config
        model_type = external_model_config.get('type') or self._detect_model_type(checkpoint)
        
        # Create appropriate model using combined configuration
        model_args = {
            'heartbeat_input_dim': heartbeat_input_dim,
            'hidden_dims': combined_config.get('hidden_dims', [1024, 512]),
            'latent_dim': combined_config.get('latent_dim', 256),
            'sparsity_weight': combined_config.get('sparsity_weight', 0.01),
            'alpha_aux': combined_config.get('alpha_aux', 0.02),
            'target_sparsity': combined_config.get('target_sparsity', 0.1),
            'use_frozen_decoder_for_aux': combined_config.get('use_frozen_decoder_for_aux', True)
        }
        
        if model_type == 'simple':
            model = SimpleAutoencoder(**model_args)
        else:
            model = GatedSparseAutoencoder(**model_args)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model, model_type
    
    def _detect_model_type(self, checkpoint) -> str:
        """Detect whether checkpoint is from sparse or simple autoencoder."""
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Check for GatedSparseAutoencoder specific parameters
        sparse_keys = ['W_gate', 'b_gate', 'r_mag', 'b_mag']
        if any(key in state_dict for key in sparse_keys):
            return 'sparse'
        
        # Check for SimpleAutoencoder specific parameters (CNN layers)
        simple_keys = ['encoder.0.weight', 'decoder_conv.0.weight']
        if any(key in state_dict for key in simple_keys):
            return 'simple'
        
        # Default to sparse
        return 'sparse'
    
    def find_max_activating_samples(self, feature_idx: int, dataloader, 
                                   top_k: int = 20) -> List[Tuple[int, float, str]]:
        """Find samples that maximally activate a specific feature."""
        self.model.eval()
        activations = []
        sample_data = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['heartbeats'].to(self.device)  # Use heartbeats from preprocessed data
                x_hat, latent = self.model(x)  # Both sparse and simple models return (reconstruction, latent)
                
                # Get activations for specific feature
                feature_acts = latent[:, feature_idx].cpu().numpy()
                
                # Store activations and corresponding reports
                for i, activation in enumerate(feature_acts):
                    idx_in_batch = i
                    report = batch['report'][idx_in_batch]
                    ecg_id = batch['ecg_id'][idx_in_batch].item() if hasattr(batch['ecg_id'][idx_in_batch], 'item') else batch['ecg_id'][idx_in_batch]
                    
                    activations.append(activation)
                    sample_data.append((ecg_id, report))
        
        # Sort by activation strength and get top k
        sorted_indices = np.argsort(activations)[::-1][:top_k]
        
        top_samples = []
        for idx in sorted_indices:
            ecg_id, report = sample_data[idx]
            activation_val = activations[idx]
            top_samples.append((ecg_id, activation_val, report))
        
        return top_samples
    
    def analyze_reports_with_llm(self, reports: List[str], feature_idx: int) -> Dict[str, str]:
        """Use Azure OpenAI to analyze reports and extract only a summary (max 2 sentences) and a single key word/expression."""
        reports_text = "\n".join([f"- {report}" for report in reports[:10]])
        prompt = f"""
You are a clinical AI assistant analyzing ECG reports to understand what a machine learning feature has learned.

I have a sparse autoencoder trained on ECG data. Feature #{feature_idx} activates most strongly on the following ECG reports:

{reports_text}

Based on these reports, please provide ONLY the following in your response:

1. Summary: A concise summary of what this feature represents, in at most 2 sentences.
2. Key Word: A single clinical key word or short expression (1-3 words) that best describes what this feature detects.

Format your response as:
Summary: <your summary>
Key Word: <your key word or expression>
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert cardiologist and ECG interpreter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            interpretation = response.choices[0].message.content
            # Extract summary and key word
            summary = ""
            key_word = ""
            for line in interpretation.split('\n'):
                l = line.strip()
                if l.lower().startswith("summary:"):
                    summary = l.split(":",1)[-1].strip()
                elif l.lower().startswith("key word:"):
                    key_word = l.split(":",1)[-1].strip()
            # Fallback: try to find lines if not found
            if not summary or not key_word:
                lines = [l.strip() for l in interpretation.split('\n') if l.strip()]
                for l in lines:
                    if not summary and "summary" in l.lower():
                        summary = l.split(":",1)[-1].strip()
                    if not key_word and "key word" in l.lower():
                        key_word = l.split(":",1)[-1].strip()
            return {
                'feature_idx': feature_idx,
                'summary': summary,
                'key_word': key_word,
                'num_reports_analyzed': len(reports)
            }
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            return {
                'feature_idx': feature_idx,
                'summary': 'Analysis failed',
                'key_word': 'Unknown',
                'num_reports_analyzed': len(reports)
            }
    
    def interpret_all_features(self, top_k_features: int = 20, samples_per_feature: int = 20) -> List[Dict]:
        """Interpret the top k most active features."""
        
        # Create dataset and data loader using preprocessed data
        dataset = PTBXLDataset(self.preprocessed_path, split='train')
        train_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False
        )
        
        # First, find which features are most active overall
        print("Finding most active features...")
        all_activations = []
        
        with torch.no_grad():
            for batch in train_loader:
                x = batch['heartbeats'].to(self.device)  # Use heartbeats from preprocessed data
                x_hat, latent = self.model(x)  # Both sparse and simple models return (reconstruction, latent)
                all_activations.append(latent.cpu().numpy())
        
        # Combine all activations
        all_activations = np.vstack(all_activations)
        feature_activity = np.mean(np.abs(all_activations), axis=0)
        most_active_features = np.argsort(feature_activity)[-top_k_features:][::-1]
        
        print(f"Analyzing top {top_k_features} most active features...")
        
        interpretations = []
        
        for i, feature_idx in enumerate(most_active_features):
            print(f"\n[{i+1}/{top_k_features}] Analyzing feature {feature_idx}...")
            
            # Find samples that activate this feature most
            top_samples = self.find_max_activating_samples(
                feature_idx, train_loader, samples_per_feature
            )
            
            # Extract reports
            reports = [sample[2] for sample in top_samples if isinstance(sample[2], str) and len(sample[2].strip()) > 10]
            
            if len(reports) < 3:
                print(f"Skipping feature {feature_idx} - too few valid reports")
                continue
            
            print(f"Found {len(reports)} reports for analysis")
            
            # Analyze with LLM
            interpretation = self.analyze_reports_with_llm(reports, feature_idx)
            
            # Add additional metadata
            interpretation.update({
                'activity_rank': i + 1,
                'mean_activity': float(feature_activity[feature_idx]),
                'top_activations': [float(sample[1]) for sample in top_samples[:5]],
                'sample_reports': reports[:3]  # Store first 3 as examples
            })
            
            interpretations.append(interpretation)
            
            print(f"✓ Feature {feature_idx}: {interpretation.get('summary', 'Analysis completed')}")
        
        return interpretations
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_interpretations(self, interpretations: List[Dict], filename: str = 'feature_interpretations.json'):
        """Save interpretations to JSON file."""
        # Convert numpy types to JSON serializable types
        serializable_interpretations = self._convert_to_serializable(interpretations)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_interpretations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Feature interpretations saved to {filename}")
        
        # Also create a summary report
        self._create_summary_report(interpretations)
    
    def _create_summary_report(self, interpretations: List[Dict]):
        """Create a human-readable summary report."""
        
        report_lines = [
            "# ECG Sparse Autoencoder Feature Interpretations",
            f"Generated from model: {self.model_path}",
            f"Total features analyzed: {len(interpretations)}",
            f"Model latent dimension: {self.model.latent_dim}",
            "",
            "## Feature Interpretations",
            ""
        ]
        
        for interp in interpretations:
            feature_idx = interp['feature_idx']
            summary = interp.get('summary', 'No summary available')
            key_word = interp.get('key_word', 'Unknown')
            report_lines.extend([
                f"### Feature {feature_idx} (Rank #{interp['activity_rank']})",
                f"**Key Word:** {key_word}",
                f"**Summary:** {summary}",
                f"**Activity Level:** {interp['mean_activity']:.4f}",
                ""
            ])
        
        # Save report
        with open('feature_interpretations_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("✓ Summary report saved to feature_interpretations_report.md")


def main():
    parser = argparse.ArgumentParser(description="Interpret ECG Sparse Autoencoder Features")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--config_path", default="config_simple.json",
                       help="Path to model configuration file")
    parser.add_argument("--preprocessed_path", default="outputs/preprocessed_data",
                       help="Preprocessed data path")
    parser.add_argument("--data_path", default="physionet.org/files/ptb-xl/1.0.3/",
                       help="PTB-XL dataset path (for metadata)")
    parser.add_argument("--top_features", type=int, default=15,
                       help="Number of top features to interpret")
    parser.add_argument("--samples_per_feature", type=int, default=20,
                       help="Number of top-activating samples per feature")
    
    args = parser.parse_args()
    
    # Check if Azure OpenAI is configured
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: Azure OpenAI credentials not found!")
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your .env file")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create interpreter
    interpreter = FeatureInterpreter(args.model_path, args.preprocessed_path, args.config_path, args.data_path, device)
    
    # Interpret features
    interpretations = interpreter.interpret_all_features(
        args.top_features, 
        args.samples_per_feature
    )
    
    # Save results
    interpreter.save_interpretations(interpretations)
    
    print(f"\n🎉 Analysis complete! Interpreted {len(interpretations)} features.")
    print("Check feature_interpretations.json and feature_interpretations_report.md for results.")


if __name__ == "__main__":
    main()
