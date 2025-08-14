import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import time
from typing import Dict, List

from data_loader import create_data_loaders
from sparse_autoencoder import GatedSparseAutoencoder
from simple_autoencoder import SimpleAutoencoder


class ECGAutoencoderTrainer:
    """Trainer for ECG Autoencoders (both sparse and simple)."""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: Autoencoder model (GatedSparseAutoencoder or SimpleAutoencoder)
            device: Device for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader, optimizer, epoch: int, k_frozen_update: int | None = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0
        total_kl_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for step, batch in enumerate(pbar, start=1):
            x = batch['heartbeats'].to(self.device)  # Use simplified heartbeats
            
            optimizer.zero_grad()
            
            # Update sparsity weight for warm-up (if using SimpleAutoencoder)
            if isinstance(self.model, SimpleAutoencoder):
                self.model.update_sparsity_weight()
            
            # Forward pass and loss (use GatedSparseAutoencoder interface)
            losses = self.model.loss(x, lambda_sparsity=self.model.sparsity_weight)
            total_loss_batch = losses['loss']
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if k_frozen_update and step % k_frozen_update == 0:
                # Refresh frozen decoder snapshot
                self.model.refresh_frozen_decoder()
            total_loss += total_loss_batch.item()
            total_recon_loss += losses['L_recon'].item()
            total_l1_loss += losses['L_sparsity'].item()
            total_kl_loss += losses['L_aux'].item()
            num_batches += 1
            
            # Update progress bar with current sparsity weight for SimpleAutoencoder
            postfix = {
                'loss': f'{total_loss_batch.item():.4f}',
                'recon': f'{losses["L_recon"].item():.4f}',
                'l1': f'{losses["L_sparsity"].item():.4f}',
                'aux': f'{losses["L_aux"].item():.4f}'
            }
            if isinstance(self.model, SimpleAutoencoder):
                postfix['sparsity_w'] = f'{self.model.get_current_sparsity_weight():.6f}'
            pbar.set_postfix(postfix)
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['heartbeats'].to(self.device)  # Use simplified heartbeats
                losses = self.model.loss(x, lambda_sparsity=self.model.sparsity_weight)
                total_loss += losses['loss'].item()
                total_recon_loss += losses['L_recon'].item()
                total_l1_loss += losses['L_sparsity'].item()
                total_kl_loss += losses['L_aux'].item()
                num_batches += 1
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              learning_rate: float = 0.001, save_dir: str = 'checkpoints',
              k_frozen_update: int | None = 200):
        """
        Train the sparse autoencoder.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save checkpoints
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, epoch, k_frozen_update=k_frozen_update)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            scheduler.step(val_metrics['total_loss'])
            
            # Save metrics
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Recon: {train_metrics['reconstruction_loss']:.4f}, "
                  f"L1: {train_metrics['l1_loss']:.4f})")
            print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Recon: {val_metrics['reconstruction_loss']:.4f}, "
                  f"L1: {val_metrics['l1_loss']:.4f})")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Clean state dict to ensure PyTorch 2.6 compatibility
                model_state = {}
                for key, value in self.model.state_dict().items():
                    if torch.is_tensor(value):
                        model_state[key] = value
                    else:
                        # Convert any non-tensor values to tensors if possible
                        try:
                            model_state[key] = torch.tensor(value)
                        except:
                            # Skip non-serializable items
                            continue
                
                # Save model (exclude optimizer to avoid numpy serialization issues)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'train_loss': float(train_metrics['total_loss']),
                    'val_loss': float(val_metrics['total_loss']),
                    'model_config': {
                        'heartbeat_input_dim': int(self.model.input_dim),
                        'latent_dim': int(self.model.latent_dim),
                        'sparsity_weight': float(self.model.sparsity_weight),
                        'alpha_aux': float(self.model.alpha_aux),
                        'target_sparsity': float(self.model.target_sparsity),
                        'use_frozen_decoder_for_aux': bool(self.model.use_frozen_decoder_for_aux),
                        'hidden_dims': self.model.hidden_dims
                    }
                }, save_path / 'best_model.pth')
                
                print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\\nEarly stopping after {epoch} epochs (patience: {max_patience})")
                break
                
            # Save checkpoint every 10 epochs (exclude optimizer to avoid numpy issues)
            if epoch % 10 == 0:
                # Clean state dict for checkpoint
                checkpoint_state = {}
                for key, value in self.model.state_dict().items():
                    if torch.is_tensor(value):
                        checkpoint_state[key] = value
                    else:
                        # Convert any non-tensor values to tensors if possible
                        try:
                            checkpoint_state[key] = torch.tensor(value)
                        except:
                            # Skip non-serializable items
                            continue
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': checkpoint_state,
                    'train_loss': float(train_metrics['total_loss']),
                    'val_loss': float(val_metrics['total_loss'])
                }, save_path / f'checkpoint_epoch_{epoch}.pth')
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Validation', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        train_recon = [m['reconstruction_loss'] for m in self.train_metrics]
        val_recon = [m['reconstruction_loss'] for m in self.val_metrics]
        axes[0, 1].plot(train_recon, label='Train', alpha=0.8)
        axes[0, 1].plot(val_recon, label='Validation', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # L1 sparsity loss
        train_l1 = [m['l1_loss'] for m in self.train_metrics]
        val_l1 = [m['l1_loss'] for m in self.val_metrics]
        axes[1, 0].plot(train_l1, label='Train', alpha=0.8)
        axes[1, 0].plot(val_l1, label='Validation', alpha=0.8)
        axes[1, 0].set_title('L1 Sparsity Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Training Metrics\\nCompleted', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14)
        axes[1, 1].set_title('Training Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def load_config(config_path: str = 'config.json') -> dict:
    """Load configuration from JSON file."""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Flatten nested config for easier access
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    return flat_config


def main():
    """Main training function."""
    # Load configuration from config.json
    config = load_config('config_simple.json')
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, dataset = create_data_loaders(
        preprocessed_path=config.get('preprocessed_path', 'preprocessed_data'),
        batch_size=config['batch_size'],
        test_fold=config['test_fold'],
        original_data_path=config.get('data_path'),
        max_samples=config['max_samples']
    )
    
    # Get input dimensions from dataset
    heartbeat_input_dim = dataset.get_heartbeat_dim()  # 3 beats * 750ms * sampling_rate/1000
    print(f"Heartbeat input dimension: {heartbeat_input_dim}")
    
    # Create model based on type
    model_type = config.get('type', 'sparse').lower()
    print(f"Creating {model_type} autoencoder...")
    
    model_args = {
        'heartbeat_input_dim': heartbeat_input_dim,
        'hidden_dims': config['hidden_dims'],
        'latent_dim': config['latent_dim'],
        'sparsity_weight': config.get('sparsity_weight', 0.01),
        'alpha_aux': config.get('alpha_aux', 0.02),
        'target_sparsity': config.get('target_sparsity', 0.1),
        'use_frozen_decoder_for_aux': config.get('use_frozen_decoder_for_aux', True)
    }
    
    if model_type == 'simple':
        model_args['sparsity_warmup_steps'] = config.get('sparsity_warmup_steps', 1000)
        model = SimpleAutoencoder(**model_args)
    elif model_type == 'sparse':
        model = GatedSparseAutoencoder(**model_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'simple' or 'sparse'.")
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = ECGAutoencoderTrainer(model, device)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        k_frozen_update=config.get('k_frozen_update')
    )
    
    # Plot training curves
    trainer.plot_training_curves('checkpoints/training_curves.png')
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
