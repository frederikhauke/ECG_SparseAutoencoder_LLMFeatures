import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional, Dict, Tuple
from tools.timing_extractor import extract_timing_features

# Utility function for decoder column norms
def col_norms(matrix: Tensor, eps=1e-8) -> Tensor:
    return torch.sqrt((matrix ** 2).sum(dim=0) + eps)

class GatedSparseAutoencoder(nn.Module):
    """
    Gated Sparse Autoencoder for ECG + timing features, following the provided template.
    """
    def __init__(self, ecg_input_dim: int, timing_features_dim: int, hidden_dims: list, latent_dim: int, sparsity_weight: float = 0.01, use_frozen_decoder_for_aux: bool = True):
        super().__init__()
        self.ecg_input_dim = ecg_input_dim
        self.timing_features_dim = timing_features_dim
        self.input_dim = ecg_input_dim + timing_features_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.sparsity_weight = sparsity_weight  # For config and trainer access
        self.use_frozen_decoder_for_aux = use_frozen_decoder_for_aux

        # Encoder: stack hidden layers, then use gating/mag/decoder as in template
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        encoder_out_dim = prev_dim

        # Gate weights (W_gate) and bias
        self.W_gate = nn.Parameter(torch.randn(latent_dim, encoder_out_dim) * (1.0 / (encoder_out_dim ** 0.5)))
        self.b_gate = nn.Parameter(torch.zeros(latent_dim))

        # Magnitude weights (W_mag) share direction with W_gate, scaled by exp(r_mag)
        self.r_mag = nn.Parameter(torch.zeros(latent_dim))
        self.b_mag = nn.Parameter(torch.zeros(latent_dim))

        # Decoder
        self.W_dec = nn.Parameter(torch.randn(self.input_dim, latent_dim) * (1.0 / (latent_dim ** 0.5)))
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))

        if use_frozen_decoder_for_aux:
            self.register_buffer("_W_dec_frozen", self.W_dec.detach().clone())
            self.register_buffer("_b_dec_frozen", self.b_dec.detach().clone())

    def compute_W_mag(self) -> Tensor:
        scales = torch.exp(self.r_mag).unsqueeze(1)
        return self.W_gate * scales

    def encode(self, x: Tensor):
        x_proj = self.encoder(x)
        pre_gate = F.linear(x_proj, self.W_gate, self.b_gate)
        gate_mask = (pre_gate > 0).float()
        W_mag = self.compute_W_mag()
        mag = F.linear(x_proj, W_mag, self.b_mag)
        mag = F.relu(mag)
        RA = F.relu(pre_gate)
        h = gate_mask * mag
        return pre_gate, RA, mag, h

    def decode(self, h: Tensor, use_frozen: bool = False):
        if use_frozen and self.use_frozen_decoder_for_aux:
            W = self._W_dec_frozen
            b = self._b_dec_frozen
            return F.linear(h, W, b)
        else:
            return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x: Tensor):
        pre_gate, RA, mag, h = self.encode(x)
        x_hat = self.decode(h, use_frozen=False)
        return x_hat, h

    def loss(self, x: Tensor, lambda_sparsity: float, sae_rad_style: bool = False):
        pre_gate, RA, mag, h = self.encode(x)
        x_hat = self.decode(h, use_frozen=False)
        # Reconstruction MSE
        L_recon = F.mse_loss(x_hat, x, reduction="mean")
        # Sparsity term
        if sae_rad_style:
            dec_col_norms = col_norms(self.W_dec)
            weighted = RA * dec_col_norms.unsqueeze(0)
            L_sparsity = lambda_sparsity * weighted.mean()
        else:
            L_sparsity = lambda_sparsity * RA.abs().mean()
        # Auxiliary loss
        if self.use_frozen_decoder_for_aux and (not sae_rad_style):
            x_hat_aux = self.decode(RA, use_frozen=True)
        else:
            x_hat_aux = self.decode(RA, use_frozen=False)
        L_aux = F.mse_loss(x_hat_aux, x, reduction="mean")
        total = L_recon + L_sparsity + L_aux
        return {
            "loss": total,
            "L_recon": L_recon,
            "L_sparsity": L_sparsity,
            "L_aux": L_aux
        }

    def extract_timing_features(self, ecg_signal: np.ndarray, sampling_rate: int = 100, ecg_id: Optional[int] = None) -> np.ndarray:
        return extract_timing_features(
            ecg_id=ecg_id or 0,
            signal_2d=ecg_signal if ecg_signal.ndim == 2 else ecg_signal.reshape(-1, 1),
            sampling_rate=sampling_rate
        )

    def get_feature_activations(self, dataloader, device: str = 'cpu') -> np.ndarray:
        self.eval()
        activations = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['signal_flat'].to(device)
                _, h = self.forward(x)
                activations.append(h.cpu().numpy())
        return np.vstack(activations)



# Updated ECGFeatureAnalyzer for GatedSparseAutoencoder
class ECGFeatureAnalyzer:
    """Analyzer for gated sparse autoencoder features on ECG data."""
    def __init__(self, model: GatedSparseAutoencoder, dataset, device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.to(device)
    def find_max_activating_samples(self, feature_idx: int, dataloader, top_k: int = 10) -> list:
        self.model.eval()
        activations = []
        sample_indices = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['signal_flat'].to(self.device)
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
        max_activating = self.find_max_activating_samples(feature_idx, dataloader, top_k)
        reports = []
        activations = []
        scp_codes = []
        for sample_idx, activation_val, sample_data in max_activating:
            reports.append(sample_data['report'])
            activations.append(activation_val)
            scp_codes.append(sample_data['scp_codes'])
        return {
            'feature_idx': feature_idx,
            'reports': reports,
            'activations': activations,
            'scp_codes': scp_codes,
            'max_activating_samples': max_activating
        }

if __name__ == "__main__":
    # Test the gated sparse autoencoder with ECG + timing features
    ecg_input_dim = 12 * 1000  # 12 leads, 1000 time points
    timing_features_dim = 4    # PR, QRS, QT, HR
    hidden_dims = [2048, 1024, 512]
    latent_dim = 256
    model = GatedSparseAutoencoder(
        ecg_input_dim=ecg_input_dim,
        timing_features_dim=timing_features_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    )
    batch_size = 4
    combined_input = torch.randn(batch_size, ecg_input_dim + timing_features_dim)
    x_hat, h = model(combined_input)
    losses = model.loss(combined_input, lambda_sparsity=0.01)
    print(f"Model created successfully!")
    print(f"ECG input dim: {ecg_input_dim}, Timing features dim: {timing_features_dim}")
    print(f"Combined input shape: {combined_input.shape}")
    print(f"Latent shape: {h.shape}")
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
