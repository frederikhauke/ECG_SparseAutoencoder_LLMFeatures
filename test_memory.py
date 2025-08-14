#!/usr/bin/env python3
"""
Quick memory test script to check if model fits in GPU memory
"""

import torch
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from simple_autoencoder import SimpleAutoencoder
from sparse_autoencoder import GatedSparseAutoencoder

def test_model_memory(config_path: str = 'config_simple.json', device: str = 'cuda'):
    """Test if model can be created and loaded on device."""
    
    print(f"Testing model memory usage on {device}")
    print(f"Using config: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['model']
    
    # Calculate input dimension for multi-lead data
    # 12 leads * 3 beats * 75 samples = 2700
    heartbeat_input_dim = 2700
    
    print(f"Input dimension: {heartbeat_input_dim}")
    print(f"Model type: {model_config['type']}")
    print(f"Hidden dims: {model_config['hidden_dims']}")
    print(f"Latent dim: {model_config['latent_dim']}")
    
    try:
        # Create model
        if model_config['type'] == 'simple':
            model = SimpleAutoencoder(
                heartbeat_input_dim=heartbeat_input_dim,
                hidden_dims=model_config['hidden_dims'],
                latent_dim=model_config['latent_dim'],
                sparsity_weight=model_config['sparsity_weight'],
                alpha_aux=model_config['alpha_aux'],
                target_sparsity=model_config['target_sparsity'],
                use_frozen_decoder_for_aux=model_config['use_frozen_decoder_for_aux']
            )
        else:
            model = GatedSparseAutoencoder(
                heartbeat_input_dim=heartbeat_input_dim,
                hidden_dims=model_config['hidden_dims'],
                latent_dim=model_config['latent_dim'],
                sparsity_weight=model_config['sparsity_weight'],
                alpha_aux=model_config['alpha_aux'],
                target_sparsity=model_config['target_sparsity'],
                use_frozen_decoder_for_aux=model_config['use_frozen_decoder_for_aux']
            )
        
        print("✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Try to move to device
        print(f"\\nAttempting to move model to {device}...")
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to(device)
            print("✓ Model moved to GPU successfully")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"GPU memory allocated: {memory_allocated:.2f} GB")
                print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        else:
            model = model.to('cpu')
            print("✓ Model on CPU")
        
        # Test forward pass with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            try:
                print(f"\\nTesting batch size {batch_size}...")
                
                # Create dummy input
                x = torch.randn(batch_size, heartbeat_input_dim).to(device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(x)
                    if isinstance(output, tuple):
                        x_hat, h = output
                        print(f"  ✓ Forward pass successful - Output shape: {x_hat.shape}, Latent: {h.shape}")
                    else:
                        print(f"  ✓ Forward pass successful - Output shape: {output.shape}")
                
                # Check memory after forward pass
                if device == 'cuda' and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"  GPU memory after forward pass: {memory_allocated:.2f} GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ✗ OOM at batch size {batch_size}")
                    break
                else:
                    print(f"  ✗ Error at batch size {batch_size}: {e}")
                    break
        
        print("\\n✓ Memory test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during memory test: {e}")
        return False
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model memory usage')
    parser.add_argument('--config', type=str, default='config_simple.json',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to test (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Test CUDA availability
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = 'cpu'
        else:
            print(f"CUDA available - Device: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    success = test_model_memory(args.config, args.device)
    
    if success:
        print("\\n🎉 Model should fit in memory for training!")
        print("You can now run the memory-efficient training script:")
        print(f"python train_memory_efficient.py --config {args.config}")
    else:
        print("\\n❌ Model may not fit in memory. Consider:")
        print("1. Further reducing batch size")
        print("2. Reducing model dimensions in config")
        print("3. Using CPU for training")

if __name__ == "__main__":
    main()
