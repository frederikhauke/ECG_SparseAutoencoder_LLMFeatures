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
                 alpha_aux: float = 0.0, target_sparsity: float = 0.3,
                 sparsity_warmup_steps: int = 1000):
        super().__init__()
        self.input_dim = heartbeat_input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Sparsity warm-up parameters
        self.initial_sparsity_weight = sparsity_weight
        self.sparsity_weight = 0.0  # Start at 0 and warm up
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.current_step = 0
        
        # Keep these for compatibility
        self.use_frozen_decoder_for_aux = use_frozen_decoder_for_aux
        self.alpha_aux = alpha_aux
        self.target_sparsity = target_sparsity
        
        # Reconstruction weight to prioritize reconstruction over sparsity
        self.reconstruction_weight = 10.0  # Weight reconstruction 10x more than sparsity
        
        # CNN Encoder - designed for multi-lead heartbeats (12 leads * 3 beats * 75 samples = 2700)
        # More memory-efficient with smaller channel dimensions
        self.encoder = nn.Sequential(
            # First conv block: capture local beat patterns
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15),  # 2700 -> 675
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # Second conv block: beat-to-beat patterns
            nn.Conv1d(32, 64, kernel_size=15, stride=3, padding=7),  # 675 -> 225
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Third conv block: lead integration patterns  
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4),  # 225 -> 75
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # Fourth conv block: high-level features
            nn.Conv1d(128, 256, kernel_size=5, stride=3, padding=2),  # 75 -> 25
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            # Flatten and final linear layers
            nn.Flatten(),  # 256 * 25 = 6400
            nn.Linear(256 * 25, hidden_dims[0] if hidden_dims else 512),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] if hidden_dims else 512, latent_dim),
            nn.ReLU()  # Keep latent activations positive for compatibility
        )
        
        # CNN Decoder (transpose convolutions) - reverse of encoder
        self.decoder = nn.Sequential(
            # Linear layers to expand back to conv feature map
            nn.Linear(latent_dim, hidden_dims[0] if hidden_dims else 512),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] if hidden_dims else 512, 256 * 25),
            nn.ReLU(),
            
            # Reshape for transpose convolutions
            # This will be handled in forward pass: view(-1, 256, 25)
            
        )
        
        # Transpose convolution layers (defined separately for easier reshaping)
        self.decoder_conv = nn.Sequential(
            # First transpose conv block: 25 -> 75
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=3, padding=2, output_padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # Second transpose conv block: 75 -> 225
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, output_padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Third transpose conv block: 225 -> 675
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=3, padding=7, output_padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # Final transpose conv block: 675 -> 2700
            nn.ConvTranspose1d(32, 1, kernel_size=31, stride=4, padding=15, output_padding=3),
            nn.Sigmoid()  # Normalize output to [0, 1] to match input scaling
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using appropriate initialization for CNN and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x: Tensor):
        """Encode input to latent space.
        
        Returns tuple for compatibility with GatedSparseAutoencoder interface:
        (pre_gate, RA, mag, h) where for simple autoencoder all are the same latent representation.
        """
        # Reshape input for CNN: (batch_size, 1, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        h = self.encoder(x)
        # Return the same latent representation for all outputs to maintain interface compatibility
        return h, h, h, h
    
    def decode(self, h: Tensor, use_frozen: bool = False):
        """Decode latent representation back to input space.
        
        Args:
            h: Latent representation
            use_frozen: Ignored in simple autoencoder (for compatibility)
        """
        # Pass through linear decoder layers
        x = self.decoder(h)
        
        # Reshape for transpose convolutions: (batch_size, 256, 25)
        x = x.view(-1, 256, 25)
        
        # Pass through transpose convolutions
        x = self.decoder_conv(x)
        
        # Flatten back to original shape: (batch_size, sequence_length)
        x = x.squeeze(1)  # Remove channel dimension
        
        return x
    
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
        """Compute loss with reconstruction and sparsity components.
        
        Args:
            x: Input tensor
            lambda_sparsity: Weight for sparsity loss (L1 penalty on latent activations)
            sae_rad_style: Ignored in simple autoencoder
            alpha_aux: Ignored in simple autoencoder  
            target_sparsity: Ignored in simple autoencoder
            
        Returns:
            Dictionary with loss components
        """
        x_hat, h = self.forward(x)
        
        # Reconstruction loss
        L_recon = F.mse_loss(x_hat, x, reduction="mean")
        
        # Sparsity loss (L1 penalty on latent activations)
        if lambda_sparsity > 0:
            L_sparsity = lambda_sparsity * h.abs().mean()
        else:
            L_sparsity = torch.tensor(0.0, device=x.device)
        
        # Target sparsity loss - encourage activations to match target sparsity level
        current_sparsity = (h > 0.1).float().mean()  # Fraction of activations > 0.1
        target_sparsity_val = target_sparsity if target_sparsity is not None else self.target_sparsity
        L_target_sparsity = F.mse_loss(current_sparsity, torch.tensor(target_sparsity_val, device=x.device))
        
        # Auxiliary loss combines sparsity losses for compatibility
        L_aux = L_target_sparsity
        alpha_aux_val = alpha_aux if alpha_aux is not None else self.alpha_aux
        
        # Weight reconstruction much more heavily than sparsity
        total_loss = self.reconstruction_weight * L_recon + L_sparsity + alpha_aux_val * L_aux
        
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
    
    def update_sparsity_weight(self):
        """Update sparsity weight with linear warm-up schedule."""
        if self.current_step < self.sparsity_warmup_steps:
            # Linear warm-up from 0 to initial_sparsity_weight
            progress = self.current_step / self.sparsity_warmup_steps
            self.sparsity_weight = progress * self.initial_sparsity_weight
        else:
            # After warm-up, use full sparsity weight
            self.sparsity_weight = self.initial_sparsity_weight
        
        self.current_step += 1
    
    def get_current_sparsity_weight(self):
        """Get current sparsity weight for monitoring."""
        return self.sparsity_weight
    
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