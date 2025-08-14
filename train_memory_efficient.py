#!/usr/bin/env python3
"""
Memory-Efficient Training Script for ECG Autoencoders

This script implements several memory optimization techniques:
- Gradient accumulation
- Mixed precision training (if available)
- Smaller batch sizes
- CPU offloading of certain operations
- Explicit memory management
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import time
from typing import Dict, List
import argparse
import gc
import os

from data_loader import create_data_loaders
from sparse_autoencoder import GatedSparseAutoencoder
from simple_autoencoder import SimpleAutoencoder

# Try to use mixed precision if available
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
    print("Mixed precision training available")
except ImportError:
    AMP_AVAILABLE = False
    print("Mixed precision training not available")


class MemoryEfficientTrainer:
    """Memory-efficient trainer for ECG Autoencoders."""
    
    def __init__(self, model, device: str = 'cuda', use_amp: bool = True, accumulation_steps: int = 4):
        """
        Initialize the trainer.
        
        Args:
            model: Autoencoder model (GatedSparseAutoencoder or SimpleAutoencoder)
            device: Device for training ('cuda' or 'cpu')
            use_amp: Use automatic mixed precision (if available)
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Try to move model to device
        try:
            self.model.to(device)
            print(f"Model moved to {device}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Error moving model to {device}: {e}")
                print("Trying CPU instead...")
                self.device = 'cpu'
                self.model.to('cpu')
            else:
                raise e
        
        # Setup mixed precision
        self.use_amp = use_amp and AMP_AVAILABLE and device == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")
        
        self.train_losses = []
        self.val_losses = []
        self.reconstruction_losses = []
        self.sparsity_losses = []
        self.aux_losses = []
    
    def clear_cache(self):
        """Clear GPU cache and collect garbage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_epoch(self, train_loader, optimizer, epoch: int, k_frozen_update: int = None) -> Dict[str, float]:
        """Train for one epoch with memory-efficient techniques."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_aux_loss = 0.0
        
        # Clear cache at start of epoch
        self.clear_cache()
        
        # Zero gradients at the start
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                           desc=f'Epoch {epoch}')
        
        for batch_idx, batch in progress_bar:
            try:
                # Move batch to device efficiently
                x = batch['heartbeats'].to(self.device, non_blocking=True)
                
                # Forward pass with optional mixed precision
                if self.use_amp:
                    with autocast():
                        loss_dict = self._compute_loss(x, epoch, k_frozen_update)
                        # Scale loss by accumulation steps for gradient accumulation
                        loss = loss_dict['total_loss'] / self.accumulation_steps
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                else:
                    loss_dict = self._compute_loss(x, epoch, k_frozen_update)
                    # Scale loss by accumulation steps for gradient accumulation
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                    loss.backward()
                
                # Accumulate losses
                epoch_loss += loss_dict['total_loss'].item()
                epoch_recon_loss += loss_dict['reconstruction_loss'].item()
                epoch_sparsity_loss += loss_dict['sparsity_loss'].item()
                epoch_aux_loss += loss_dict['aux_loss'].item()
                
                # Update parameters every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss_dict["total_loss"].item():.4f}',
                    'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
                    'Sparsity': f'{loss_dict["sparsity_loss"].item():.4f}'
                })
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\\nOOM error at batch {batch_idx}, clearing cache and continuing...")
                    self.clear_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Handle final accumulated gradients
        if len(train_loader) % self.accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_losses = {
            'total_loss': epoch_loss / num_batches,
            'reconstruction_loss': epoch_recon_loss / num_batches,
            'sparsity_loss': epoch_sparsity_loss / num_batches,
            'aux_loss': epoch_aux_loss / num_batches
        }
        
        return avg_losses
    
    def _compute_loss(self, x, epoch: int, k_frozen_update: int = None):
        """Compute loss with memory-efficient operations."""
        # Determine model type and compute accordingly
        if isinstance(self.model, GatedSparseAutoencoder):
            return self._compute_sparse_loss(x, epoch, k_frozen_update)
        else:
            return self._compute_simple_loss(x)
    
    def _compute_sparse_loss(self, x, epoch: int, k_frozen_update: int = None):
        """Compute loss for sparse autoencoder."""
        # Forward pass
        x_hat, h = self.model(x)
        
        # Reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
        
        # Sparsity loss (L1 regularization)
        sparsity_loss = torch.mean(torch.abs(h)) * self.model.sparsity_weight
        
        # Auxiliary loss (if using frozen decoder)
        aux_loss = torch.tensor(0.0, device=self.device)
        if (self.model.use_frozen_decoder_for_aux and 
            k_frozen_update and epoch < k_frozen_update):
            with torch.no_grad():
                x_hat_aux = self.model.decode(h, use_frozen=True)
            aux_loss = torch.nn.functional.mse_loss(x_hat_aux, x) * self.model.alpha_aux
        
        total_loss = reconstruction_loss + sparsity_loss + aux_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'aux_loss': aux_loss
        }
    
    def _compute_simple_loss(self, x):
        """Compute loss for simple autoencoder."""
        # Forward pass
        x_hat, _ = self.model(x)
        
        # Only reconstruction loss for simple autoencoder
        reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
        
        return {
            'total_loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': torch.tensor(0.0, device=self.device),
            'aux_loss': torch.tensor(0.0, device=self.device)
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                try:
                    x = batch['heartbeats'].to(self.device, non_blocking=True)
                    
                    if self.use_amp:
                        with autocast():
                            loss_dict = self._compute_loss(x, 0)  # epoch=0 for validation
                    else:
                        loss_dict = self._compute_loss(x, 0)
                    
                    val_loss += loss_dict['total_loss'].item()
                    val_recon_loss += loss_dict['reconstruction_loss'].item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("\\nOOM during validation, clearing cache...")
                        self.clear_cache()
                        continue
                    else:
                        raise e
        
        return {
            'val_loss': val_loss / len(val_loader),
            'val_reconstruction_loss': val_recon_loss / len(val_loader)
        }
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              learning_rate: float = 1e-3, weight_decay: float = 1e-4,
              patience: int = 15, save_dir: str = 'checkpoints',
              k_frozen_update: int = None):
        """Train the model with memory-efficient techniques."""
        
        # Create save directory
        Path(save_dir).mkdir(exist_ok=True)
        
        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=learning_rate, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader, optimizer, epoch, k_frozen_update)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Store losses
            self.train_losses.append(train_losses['total_loss'])
            self.val_losses.append(val_losses['val_loss'])
            self.reconstruction_losses.append(train_losses['reconstruction_loss'])
            self.sparsity_losses.append(train_losses['sparsity_loss'])
            self.aux_losses.append(train_losses['aux_loss'])
            
            epoch_time = time.time() - start_time
            
            print(f"\\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_losses['total_loss']:.6f}")
            print(f"Val Loss: {val_losses['val_loss']:.6f}")
            print(f"Reconstruction: {train_losses['reconstruction_loss']:.6f}")
            if train_losses['sparsity_loss'] > 0:
                print(f"Sparsity: {train_losses['sparsity_loss']:.6f}")
            if train_losses['aux_loss'] > 0:
                print(f"Auxiliary: {train_losses['aux_loss']:.6f}")
            
            # Save best model
            if val_losses['val_loss'] < best_val_loss:
                best_val_loss = val_losses['val_loss']
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['val_loss'],
                    'model_config': {
                        'input_dim': self.model.input_dim,
                        'latent_dim': self.model.latent_dim,
                        'hidden_dims': getattr(self.model, 'hidden_dims', []),
                        'sparsity_weight': getattr(self.model, 'sparsity_weight', 0.0),
                        'alpha_aux': getattr(self.model, 'alpha_aux', 0.0),
                        'target_sparsity': getattr(self.model, 'target_sparsity', 0.0),
                        'use_frozen_decoder_for_aux': getattr(self.model, 'use_frozen_decoder_for_aux', False)
                    }
                }
                
                torch.save(checkpoint, f'{save_dir}/best_model.pth')
                print(f"✓ New best model saved! Val loss: {best_val_loss:.6f}")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\\nEarly stopping triggered after {patience} epochs without improvement")
                    break
            
            # Clear cache at end of epoch
            self.clear_cache()
        
        print("\\nTraining completed!")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def plot_training_curves(self, save_dir: str):
        """Plot training curves."""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Reconstruction loss
        plt.subplot(1, 3, 2)
        plt.plot(self.reconstruction_losses, label='Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Reconstruction Loss')
        
        # Sparsity and aux losses
        plt.subplot(1, 3, 3)
        if max(self.sparsity_losses) > 0:
            plt.plot(self.sparsity_losses, label='Sparsity Loss')
        if max(self.aux_losses) > 0:
            plt.plot(self.aux_losses, label='Auxiliary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Regularization Losses')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_dir}/training_curves.png")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Memory-Efficient ECG Autoencoder Training')
    parser.add_argument('--config', type=str, default='config_simple.json', 
                       help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detected if not specified')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Auto-detect device if not specified
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Configuration: {args.config}")
    
    # Create data loaders with reduced batch size for memory efficiency
    train_loader, val_loader, test_loader = create_data_loaders(
        config['data']['data_path'],
        config['data']['preprocessed_path'],
        batch_size=config['training']['batch_size'],
        test_fold=config['data']['test_fold'],
        max_samples=config['data']['max_samples']
    )
    
    print(f"Data loaded - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Get input dimension
    sample_batch = next(iter(train_loader))
    heartbeat_input_dim = sample_batch['heartbeats'].shape[1]
    print(f"Heartbeat input dimension: {heartbeat_input_dim}")
    
    # Create model
    model_config = config['model']
    
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
        print("Created SimpleAutoencoder")
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
        print("Created GatedSparseAutoencoder")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create memory-efficient trainer
    trainer = MemoryEfficientTrainer(model, device, accumulation_steps=args.accumulation_steps)
    
    # Training configuration
    train_config = config['training']
    
    # Train model
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_config['num_epochs'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        patience=train_config['patience'],
        save_dir=train_config['save_dir'],
        k_frozen_update=model_config.get('k_frozen_update')
    )
    
    print(f"\\nBest validation loss: {training_results['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
