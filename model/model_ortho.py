"""
AES Method 1: Orthogonal Disentanglement Model.
Splits latent space into Quality (z_q) and Domain (z_d) subspaces.
Includes a Decoder for reconstruction.
"""

from typing import Sequence

import gin
import torch
import torch.nn as nn


# --- Helper ---
def _get_conv1d_layer(in_channels, out_channels, ln_shape, kernel_size, pool_size, use_pooling, use_normalization, dropout_rate):
    # Padding set to kernel_size // 2 to maintain dimension before pooling
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)]
    if use_normalization:
        # LayerNorm in 1D usually requires specific shape or transposing, 
        # using InstanceNorm1d is safer for variable length signals in this context,
        # but sticking to LayerNorm if shape is provided, or GroupNorm/InstanceNorm.
        # Here we follow the logic: if ln_shape is strictly handled, keep LayerNorm.
        # For simplicity in this adaptation, we stick to the original logic or use InstanceNorm if shape is dynamic.
        layers.append(nn.LayerNorm(ln_shape)) 
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    if use_pooling:
        layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
    return nn.Sequential(*layers)

@gin.configurable
class DANNProjectionHead(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int],
        conv_channels: Sequence[int] = (32, 32),
        dense_neurons: Sequence[int] = (64, 32, 14), 
        
        # --- Domain Classifier Params ---
        domain_classifier_neurons: Sequence[int] = (64, 32),
        num_domains: int = 16, # Default for Single Domain Experiment
        
        # --- General Params ---
        use_poolings: Sequence[bool] = (True, True),
        use_normalizations: Sequence[bool] = (True, True),
        kernel_size: int = 5,
        pool_size: int = 5,
        dropout_rate: float = 0.3,
        apply_linear_transform: bool = True,

        # --- AES Scaling (1-10) ---
        output_scale: float = 4.5,
        output_bias: float = 5.5,
    ):
        super().__init__()
        self._output_scale = output_scale
        self._output_bias = output_bias
        self._apply_linear_transform = apply_linear_transform

        # 1. Shared Encoder
        self._encoder = self._build_encoder(in_shape, conv_channels, use_poolings, use_normalizations, kernel_size, pool_size, dropout_rate)
        
        # Calculate Flatten Size dynamically
        self._flatten = nn.Flatten()
        with torch.no_grad():
            # Create a dummy input to trace shapes
            # in_shape is (Channels, Time), e.g., (768, 500)
            dummy_input = torch.zeros((1,) + in_shape)
            enc_out = self._encoder(dummy_input)
            self._enc_out_shape = enc_out.shape # (B, C_out, T_out)
            self._flat_dim = self._flatten(enc_out).shape[-1]

        # 2. Split Dimensions
        # We split the FLATTENED vector. 
        # z_q gets first half, z_d gets second half.
        self.latent_dim = self._flat_dim
        self.split_dim = self.latent_dim // 2
        
        # Ensure split is valid
        if self.latent_dim % 2 != 0:
            raise ValueError(f"Latent dimension {self.latent_dim} is not divisible by 2. Adjust channels or pooling.")

        # 3. Quality Head (Input: z_q)
        self._quality_head = self._build_dense_head(self.split_dim, dense_neurons)

        # 4. Domain Head (Input: z_d) - No GRL needed here, just aux task
        self._domain_classifier = self._build_dense_head(self.split_dim, list(domain_classifier_neurons) + [num_domains])

        # 5. Decoder (Input: z_q + z_d) -> Reconstruct input features
        # We try to reconstruct back to 'in_shape'. 
        self._decoder = self._build_decoder(self._enc_out_shape, in_shape, conv_channels, kernel_size, pool_size)

        # Output logic
        self._softplus = nn.Softplus()
        self._output_dim = dense_neurons[-1]
        
        # Determine targets
        if self._output_dim == 14: self._n_targets = 4
        elif self._output_dim == 20: self._n_targets = 5
        elif self._output_dim == 4: self._n_targets = 4
        elif self._output_dim == 5: self._n_targets = 5
        else: self._n_targets = 4

        self._covariance_indices = self._get_covariance_indices(self._output_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        # Encoder
        # x: (B, C, T)
        enc_features = self._encoder(x) # (B, C_out, T_out)
        flat_features = self._flatten(enc_features) # (B, Dim)
        
        # Split
        z_q = flat_features[:, :self.split_dim]
        z_d = flat_features[:, self.split_dim:]

        # --- Path A: Quality Prediction (using z_q) ---
        quality_raw_preds = self._quality_head(z_q)

        # --- Path B: Domain Classification (using z_d) ---
        domain_logits = self._domain_classifier(z_d)

        # --- Path C: Reconstruction (using z_q + z_d) ---
        # Concatenate back to confirm we have full info
        combined = torch.cat([z_q, z_d], dim=1) 
        # Reshape for decoder: (B, Dim) -> (B, C_out, T_out)
        combined_reshaped = combined.view(-1, *self._enc_out_shape[1:])
        recon_output = self._decoder(combined_reshaped)
        
        # Resize recon to match input x length (handling pooling rounding differences)
        if recon_output.shape[-1] != x.shape[-1]:
            recon_output = torch.nn.functional.interpolate(recon_output, size=x.shape[-1], mode='linear', align_corners=False)

        # --- Output Processing ---
        # Probabilistic case
        if self._output_dim > self._n_targets: 
            mean_predictions = quality_raw_preds[:, :self._n_targets]
            cov_matrix = self._get_covariance_matrix(quality_raw_preds)
            
            if self._apply_linear_transform:
                mean_predictions, cov_matrix = self._linear_transform(mean_predictions, cov_matrix)
            
            main_output = (mean_predictions, cov_matrix)
        
        # Non-probabilistic case
        else:
            if self._apply_linear_transform:
                quality_preds = self._output_scale * quality_raw_preds + self._output_bias
            else:
                quality_preds = quality_raw_preds
            main_output = quality_preds

        return main_output, domain_logits, z_q, z_d, recon_output

    # --- Helpers ---
    def _build_encoder(self, in_shape, conv_channels, use_poolings, use_normalizations, kernel_size, pool_size, dropout_rate):
        layers = []
        in_channels, time_dim = in_shape[0], in_shape[1]
        for i, out_channels in enumerate(conv_channels):
            # Calculate output time dim logic
            ln_time_dim = (time_dim + 2 * (kernel_size//2) - 1 * (kernel_size - 1) - 1) // 1 + 1
            layers.append(_get_conv1d_layer(in_channels, out_channels, (out_channels, ln_time_dim), kernel_size, pool_size, use_poolings[i], use_normalizations[i], dropout_rate))
            in_channels = out_channels
            time_dim = ln_time_dim // pool_size if use_poolings[i] else ln_time_dim
        return nn.Sequential(*layers)
        
    def _build_decoder(self, enc_shape, input_shape, conv_channels, kernel_size, pool_size):
        # enc_shape: (B, C_last, T_last)
        layers = []
        in_ch = enc_shape[1]
        
        # Reverse the channel list: e.g. [32, 32] -> [32, 32]
        # Final target is input_shape[0] (e.g. 768)
        reversed_channels = list(conv_channels[:-1])[::-1] + [input_shape[0]] 
        
        for i, out_ch in enumerate(reversed_channels):
            # Upsample (Inverse Pooling)
            layers.append(nn.Upsample(scale_factor=pool_size, mode='nearest'))
            # Conv (Inverse Conv / Processing)
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2))
            
            # Activation (except last layer)
            if i < len(reversed_channels) - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch
            
        return nn.Sequential(*layers)

    def _build_dense_head(self, in_features, dense_neurons):
        layers = []
        curr = in_features
        for neurons in dense_neurons[:-1]:
            layers.append(nn.Linear(curr, neurons))
            layers.append(nn.ReLU())
            curr = neurons
        layers.append(nn.Linear(curr, dense_neurons[-1]))
        return nn.Sequential(*layers)

    def _get_covariance_indices(self, output_dim):
        if output_dim == 14: return [(i, j) for i in range(4) for j in range(i+1, 4)]
        if output_dim == 20: return [(i, j) for i in range(5) for j in range(i+1, 5)]
        return []
    
    def _get_covariance_matrix(self, predictions):
        batch_size, device = predictions.shape[0], predictions.device
        n = self._n_targets
        L = torch.zeros((batch_size, n, n), device=device)
        diag_elements = self._softplus(predictions[:, n:2*n])
        L[:, range(n), range(n)] = diag_elements
        cov_preds = predictions[:, 2*n:]
        for idx, (i, j) in enumerate(self._covariance_indices):
            L[:, i, j] = cov_preds[:, idx]
        return torch.matmul(L, L.transpose(1, 2))

    def _linear_transform(self, mean, covariance):
        batch_size, device = mean.shape[0], mean.device
        n = self._n_targets
        A = torch.diag_embed(torch.full((batch_size, n), self._output_scale, device=device))
        b = torch.full((batch_size, n), self._output_bias, device=device)
        transformed_mean = torch.matmul(A, mean.unsqueeze(-1)).squeeze(-1) + b
        transformed_cov = torch.matmul(torch.matmul(A, covariance), A.transpose(1, 2))
        return transformed_mean, transformed_cov