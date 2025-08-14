#!/usr/bin/env python3
"""
Quick test to verify SimpleAutoencoder output shape is correct.
"""

import torch
import sys
sys.path.append('/home/homesOnMaster/fhauke/ECG')

from simple_autoencoder import SimpleAutoencoder

def test_autoencoder_shape():
    """Test that SimpleAutoencoder produces correct output shape."""
    
    # Create model with same parameters as config
    model = SimpleAutoencoder(
        heartbeat_input_dim=2700,
        hidden_dims=[1024, 512],
        latent_dim=256,
        sparsity_weight=0.0001
    )
    
    # Create test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 2700)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    output, latent = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Check shapes
    if output.shape == input_tensor.shape:
        print("✅ Output shape matches input shape!")
        return True
    else:
        print(f"❌ Shape mismatch: output {output.shape} vs input {input_tensor.shape}")
        return False

if __name__ == "__main__":
    success = test_autoencoder_shape()
    if success:
        print("\n🎉 SimpleAutoencoder shape test PASSED!")
    else:
        print("\n❌ SimpleAutoencoder shape test FAILED!")
        sys.exit(1)
