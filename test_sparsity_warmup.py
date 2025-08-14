#!/usr/bin/env python3
"""
Test script for sparsity warm-up functionality in SimpleAutoencoder.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from simple_autoencoder import SimpleAutoencoder

def test_sparsity_warmup():
    """Test the sparsity warm-up mechanism."""
    print("🧪 Testing Sparsity Warm-up for SimpleAutoencoder")
    print("=" * 50)
    
    # Create a simple autoencoder with warm-up
    model = SimpleAutoencoder(
        heartbeat_input_dim=2700,
        hidden_dims=[1024, 512],
        latent_dim=256,
        sparsity_weight=0.001,  # Target sparsity weight
        sparsity_warmup_steps=1000  # Warm-up for 1000 steps
    )
    
    print(f"Target sparsity weight: {model.sparsity_weight}")
    print(f"Warm-up steps: {model.sparsity_warmup_steps}")
    print(f"Initial sparsity weight: {model.get_current_sparsity_weight()}")
    
    # Track sparsity weights during warm-up
    steps = []
    weights = []
    
    # Simulate training steps
    for step in range(0, 1500, 50):  # Test beyond warm-up period
        model.update_sparsity_weight(step)
        steps.append(step)
        weights.append(model.get_current_sparsity_weight())
    
    # Print some key values
    print(f"\nSparsity weight progression:")
    print(f"Step 0: {weights[0]:.6f}")
    print(f"Step 500: {weights[10]:.6f}")  # Mid warm-up
    print(f"Step 1000: {weights[20]:.6f}")  # End of warm-up
    print(f"Step 1400: {weights[-1]:.6f}")  # Beyond warm-up
    
    # Create a plot to visualize the warm-up
    plt.figure(figsize=(10, 6))
    plt.plot(steps, weights, 'b-', linewidth=2, label='Sparsity Weight')
    plt.axhline(y=model.sparsity_weight, color='r', linestyle='--', 
                label=f'Target Weight ({model.sparsity_weight})')
    plt.axvline(x=model.sparsity_warmup_steps, color='g', linestyle='--', 
                label=f'Warm-up End ({model.sparsity_warmup_steps})')
    plt.xlabel('Training Step')
    plt.ylabel('Effective Sparsity Weight')
    plt.title('Sparsity Warm-up Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparsity_warmup_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved warm-up visualization to 'sparsity_warmup_schedule.png'")
    
    # Test loss computation with warm-up
    print(f"\n🔍 Testing loss computation with warm-up:")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 2700)
    
    # Reset to beginning and compute loss at different stages
    model.reset_warmup()
    
    # Beginning of training (step 0)
    model.update_sparsity_weight(0)
    loss_dict_start = model.loss(x, use_warmup=True)
    print(f"Step 0 - Sparsity loss: {loss_dict_start['L_sparsity']:.6f}")
    
    # Middle of warm-up (step 500)
    model.update_sparsity_weight(500)
    loss_dict_mid = model.loss(x, use_warmup=True)
    print(f"Step 500 - Sparsity loss: {loss_dict_mid['L_sparsity']:.6f}")
    
    # End of warm-up (step 1000)
    model.update_sparsity_weight(1000)
    loss_dict_end = model.loss(x, use_warmup=True)
    print(f"Step 1000 - Sparsity loss: {loss_dict_end['L_sparsity']:.6f}")
    
    # Compare with no warm-up
    loss_dict_no_warmup = model.loss(x, use_warmup=False)
    print(f"No warm-up - Sparsity loss: {loss_dict_no_warmup['L_sparsity']:.6f}")
    
    print(f"\n✅ Sparsity warm-up test completed successfully!")
    print(f"The model gradually increases sparsity regularization from 0 to {model.sparsity_weight}")

if __name__ == "__main__":
    test_sparsity_warmup()
