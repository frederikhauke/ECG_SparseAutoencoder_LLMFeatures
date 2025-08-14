import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional, Dict, Tuple
from tools.timing_extractor import extract_timing_features


class SimpleAutoencoder(nn.Module):
    """Simple Autoencoder for ECG heartbeats - drop-in replacement for GatedSparseAutoencoder.
    
    This is a standard autoencoder without sparsity constraints or gating mechanisms.
    It maintains the same interface as GatedSparseAutoencoder for compatibility.
    """
    
    def __init__(self, heartbeat_input_dim: int, hidden_dims: list, latent_dim: int,
                 sparsity_weight: float = 0.0, use_frozen_decoder_for_aux: bool = False,
                 alpha_aux: float = 0.0, target_sparsity: float = 0.0):
        super().__init__()
        self.input_dim = heartbeat_input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Keep these for compatibility but don't use them
        self.sparsity_weight = sparsity_weight
        self.use_frozen_decoder_for_aux = use_frozen_decoder_for_aux
        self.alpha_aux = alpha_aux
        self.target_sparsity = target_sparsity
        
        # Build encoder
        encoder_layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())  # Keep latent activations positive for compatibility
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse hidden layers
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Final decoder layer back to input dimension
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)
    
    def encode(self, x: Tensor):
        """Encode input to latent space.
        
        Returns tuple for compatibility with GatedSparseAutoencoder interface:
        (pre_gate, RA, mag, h) where for simple autoencoder all are the same latent representation.
        """
        h = self.encoder(x)
        # Return the same latent representation for all outputs to maintain interface compatibility
        return h, h, h, h
    
    def decode(self, h: Tensor, use_frozen: bool = False):
        """Decode latent representation back to input space.
        
        Args:
            h: Latent representation
            use_frozen: Ignored in simple autoencoder (for compatibility)
        """
        return self.decoder(h)
    
    def forward(self, x: Tensor):
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        _, _, _, h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    def loss(self, x: Tensor, lambda_sparsity: float = 0.0, sae_rad_style: bool = False, 
             alpha_aux: float = None, target_sparsity: float = None):
        """Compute loss - simplified to only reconstruction loss.
        
        Args:
            x: Input tensor
            lambda_sparsity: Ignored in simple autoencoder
            sae_rad_style: Ignored in simple autoencoder
            alpha_aux: Ignored in simple autoencoder
            target_sparsity: Ignored in simple autoencoder
            
        Returns:
            Dictionary with loss components (for compatibility)
        """
        x_hat, h = self.forward(x)
        
        # Only reconstruction loss for simple autoencoder
        L_recon = F.mse_loss(x_hat, x, reduction="mean")
        
        # Set other losses to zero for compatibility
        L_sparsity = torch.tensor(0.0, device=x.device)
        L_aux = torch.tensor(0.0, device=x.device)
        alpha_aux_val = alpha_aux if alpha_aux is not None else self.alpha_aux
        
        total_loss = L_recon
        
        return {
            "loss": total_loss,
            "L_recon": L_recon,
            "L_sparsity": L_sparsity,
            "L_aux": L_aux,
            "alpha_aux": alpha_aux_val
        }
    
    @torch.no_grad()
    def refresh_frozen_decoder(self):
        """Placeholder for compatibility with GatedSparseAutoencoder interface."""
        pass  # No frozen decoder in simple autoencoder
    
    def extract_timing_features(self, ecg_signal: np.ndarray, sampling_rate: int = 100, 
                               ecg_id: Optional[int] = None) -> np.ndarray:
        """Extract timing features (for compatibility)."""
        return extract_timing_features(
            ecg_id=ecg_id or 0,
            signal_2d=ecg_signal if ecg_signal.ndim == 2 else ecg_signal.reshape(-1, 1),
            sampling_rate=sampling_rate
        )
    
    def get_feature_activations(self, dataloader, device: str = 'cpu') -> np.ndarray:
        """Get feature activations for analysis (for compatibility)."""
        self.eval()
        activations = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['heartbeats'].to(device)
                _, h = self.forward(x)
                activations.append(h.cpu().numpy())
        return np.vstack(activations)


class ECGFeatureAnalyzer:
    """Analyzer for simple autoencoder features on ECG data (for compatibility)."""
    
    def __init__(self, model: SimpleAutoencoder, dataset, device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.to(device)
    
    def find_max_activating_samples(self, feature_idx: int, dataloader, top_k: int = 10) -> list:
        """Find samples that maximally activate a specific feature."""
        self.model.eval()
        activations = []
        sample_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['heartbeats'].to(self.device)
                _, h = self.model(x)
                feature_acts = h[:, feature_idx].cpu().numpy()
                activations.extend(feature_acts)
                sample_indices.extend(batch['idx'].numpy())
        
        sorted_indices = np.argsort(activations)[::-1][:top_k]
        results = []
        
        for idx in sorted_indices:
            sample_idx = sample_indices[idx]
            activation_val = activations[idx]
            sample_data = self.dataset[sample_idx]
            results.append((sample_idx, activation_val, sample_data))
        
        return results
    
    def analyze_feature_patterns(self, feature_idx: int, dataloader, top_k: int = 20) -> Dict:
        """Analyze patterns in feature activations."""
        max_activating = self.find_max_activating_samples(feature_idx, dataloader, top_k)
        
        reports = []
        activations = []
        scp_codes = []
        
        for sample_idx, activation_val, sample_data in max_activating:
            reports.append(sample_data['report'])
            activations.append(activation_val)
            if 'scp_codes' in sample_data:
                scp_codes.append(sample_data['scp_codes'])
        
        return {
            'feature_idx': feature_idx,
            'reports': reports,
            'activations': activations,
            'scp_codes': scp_codes,
            'max_activating_samples': max_activating
        }


if __name__ == "__main__":
    # Test the simple autoencoder with same parameters as sparse autoencoder
    heartbeat_input_dim = 3 * 75  # 3 beats * 750ms * 0.1 samples/ms (100Hz)
    hidden_dims = [512, 256]
    latent_dim = 128
    
    model = SimpleAutoencoder(
        heartbeat_input_dim=heartbeat_input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    )
    
    batch_size = 4
    heartbeat_input = torch.randn(batch_size, heartbeat_input_dim)
    
    # Test forward pass
    x_hat, h = model(heartbeat_input)
    
    # Test loss
    losses = model.loss(heartbeat_input, lambda_sparsity=0.01)
    
    print(f"Simple Autoencoder created successfully!")
    print(f"Heartbeat input dim: {heartbeat_input_dim}")
    print(f"Input shape: {heartbeat_input.shape}")
    print(f"Latent shape: {h.shape}")
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"Reconstruction loss: {losses['L_recon'].item():.4f}")
    print(f"Sparsity loss: {losses['L_sparsity'].item():.4f}")
    print(f"Auxiliary loss: {losses['L_aux'].item():.4f}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")