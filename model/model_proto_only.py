"""
Prototypical Network for Quality Assessment (No Domain Adversarial Branch).
Uses learnable prototypes to regress mean scores for interpretability and stability.
"""

from typing import Sequence

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper function (unchanged)
def _get_conv1d_layer(
    in_channels: int, out_channels: int, ln_shape: tuple[int], kernel_size: int,
    pool_size: int, use_pooling: bool, use_normalization: bool,
    dropout_rate: float,
) -> nn.Module:
    """Returns a 1D conv layer with optional normalization and pooling."""
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
    if use_normalization:
        layers.append(nn.LayerNorm(ln_shape))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    if use_pooling:
        layers.append(nn.MaxPool1d(kernel_size=pool_size))
    return nn.Sequential(*layers)


@gin.configurable
class PrototypicalHead(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int],
        conv_channels: Sequence[int] = (32, 32),
        dense_neurons: Sequence[int] = (64, 32, 14), # AES Probabilistic (4 mean + 4 var + 6 cov)
        use_poolings: Sequence[bool] = (True, True),
        use_normalizations: Sequence[bool] = (True, True),
        kernel_size: int = 5,
        pool_size: int = 5,
        dropout_rate: float = 0.3,
        apply_linear_transform: bool = True,
        
        # --- Prototype Parameters ---
        proto_embed_dim: int = 64,
        num_prototypes: int = 10,
        num_dims: int = 4, # AES has 4 dimensions
        
        # --- Output Scaling (AES 1-10) ---
        output_scale: float = 4.5,
        output_bias: float = 5.5,
    ):
        super().__init__()
        
        self._output_scale = output_scale
        self._output_bias = output_bias
        self._apply_linear_transform = apply_linear_transform

        # --- Shared Encoder ---
        self._encoder = self._build_encoder(
            in_shape, conv_channels, use_poolings, use_normalizations,
            kernel_size, pool_size, dropout_rate
        )
        self._flatten = nn.Flatten()
        encoder_out_dim = self._calculate_dense_input_size(in_shape)

        # --- 1. Quality Prediction Head (Main Regression) ---
        self._quality_head = self._build_dense_head(encoder_out_dim, dense_neurons)

        # --- 2. Prototypical Mean Path ---
        self._num_dims = num_dims
        
        # (2.1) Proto Embedder: Project features to prototype space
        self._proto_embedder = nn.Sequential(
            nn.Linear(encoder_out_dim, proto_embed_dim),
            nn.ReLU()
        )
        
        # (2.2) Learnable Prototypes: (Num_Dims, K, Embed_Dim)
        self._prototypes = nn.Parameter(
            torch.randn(self._num_dims, num_prototypes, proto_embed_dim)
        )
        
        # (2.3) Mean Regressors: Map similarity scores to a mean score
        self._mean_regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_prototypes, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(self._num_dims)
        ])
        
        # (2.4) Combiner: Combine Proto-based Mean and Direct Regression Mean
        # Input: [Proto_Mean (4) + Direct_Mean (4)] = 8 -> Output: 4
        self._mean_combiner = nn.Linear(self._num_dims * 2, self._num_dims)

        # --- Helper logic for Probabilistic Output ---
        self._softplus = nn.Softplus()
        self._output_dim = dense_neurons[-1]
        
        # Determine targets (AES: 14 out -> 4 targets)
        if self._output_dim == 14: self._n_targets = 4
        elif self._output_dim == 20: self._n_targets = 5
        else: self._n_targets = self._output_dim # Fallback
        
        self._covariance_indices = self._get_covariance_indices(self._output_dim)


    def forward(self, x: torch.Tensor) -> tuple:
        """
        Performs a forward pass using Prototypes and Direct Regression.
        """
        # Shared Encoder
        encoder_features = self._encoder(x)
        flat_features = self._flatten(encoder_features)
        
        # --- Path A: Prototypical Mean ---
        h_embed = self._proto_embedder(flat_features) # (B, proto_embed_dim)
        
        # Calculate Cosine Similarity with Prototypes
        # h_expanded: (B, 1, 1, D)
        # protos_expanded: (1, Num_Dims, K, D)
        h_expanded = h_embed.unsqueeze(1).unsqueeze(1)
        protos_expanded = self._prototypes.unsqueeze(0)
        similarities = F.cosine_similarity(h_expanded, protos_expanded, dim=-1) # (B, Num_Dims, K)
        
        mean_proto_list = []
        for i in range(self._num_dims):
            sim_dim_i = similarities[:, i, :]  # (B, K)
            mean_dim_i = self._mean_regressors[i](sim_dim_i)  # (B, 1)
            mean_proto_list.append(mean_dim_i)
        mean_proto = torch.cat(mean_proto_list, dim=1)  # (B, Num_Dims)
        
        # --- Path B: Direct Quality Head ---
        quality_raw_preds = self._quality_head(flat_features)
        
        # --- Combine Paths ---
        # Probabilistic case (AES 14 dims: 0-3 Mean, 4-7 Var, 8-13 Cov)
        if self._output_dim > self._n_targets:
            direct_mean = quality_raw_preds[:, :self._n_targets]
            
            # Combine Proto Mean and Direct Mean
            combined_input = torch.cat([mean_proto, direct_mean], dim=1) # (B, 2*Num_Dims)
            final_mean = self._mean_combiner(combined_input)      # (B, Num_Dims)
            
            # Get Covariance
            cov_matrix = self._get_covariance_matrix(quality_raw_preds)
            
            # Linear Transform (Scale to 1-10)
            if self._apply_linear_transform:
                final_mean, cov_matrix = self._linear_transform(final_mean, cov_matrix)
            
            return final_mean, cov_matrix
            
        else: # Non-probabilistic
            combined_input = torch.cat([mean_proto, quality_raw_preds], dim=1)
            final_mean = self._mean_combiner(combined_input)
            
            if self._apply_linear_transform:
                return self._output_scale * final_mean + self._output_bias
            return final_mean

    def get_prototype_diversity_loss(self) -> torch.Tensor:
        """
        Calculates Orthogonality Loss for prototypes to encourage diversity.
        """
        loss_diversity = 0.0
        for i in range(self._num_dims):
            protos = self._prototypes[i] # (K, D)
            protos_norm = F.normalize(protos, p=2, dim=1)
            sim_matrix = torch.matmul(protos_norm, protos_norm.t()) # (K, K)
            identity = torch.eye(protos.size(0), device=protos.device)
            off_diagonal_sim = sim_matrix - identity
            loss_diversity += (off_diagonal_sim ** 2).mean()

        return loss_diversity / self._num_dims

    # --- Helper methods (Covariance, Encoder, etc.) ---
    def _get_covariance_indices(self, output_dim):
        if output_dim == 14: # AES
             return [(i, j) for i in range(4) for j in range(i+1, 4)]
        elif output_dim == 20: # NISQA
             return [(i, j) for i in range(5) for j in range(i+1, 5)]
        return []

    def _get_covariance_matrix(self, predictions: torch.Tensor) -> torch.Tensor:
        batch_size, device = predictions.shape[0], predictions.device
        n = self._n_targets
        L = torch.zeros((batch_size, n, n), device=device)
        
        # Diagonal (Variance)
        diag_elements = self._softplus(predictions[:, n:2*n])
        L[:, range(n), range(n)] = diag_elements
        
        # Off-diagonal (Covariance)
        cov_preds = predictions[:, 2*n:]
        for idx, (i, j) in enumerate(self._covariance_indices):
            L[:, i, j] = cov_preds[:, idx]
            
        return torch.matmul(L, L.transpose(1, 2))

    def _linear_transform(self, mean: torch.Tensor, covariance: torch.Tensor) -> tuple:
        batch_size, device = mean.shape[0], mean.device
        n = self._n_targets
        A = torch.diag_embed(torch.full((batch_size, n), self._output_scale, device=device))
        b = torch.full((batch_size, n), self._output_bias, device=device)
        transformed_mean = torch.matmul(A, mean.unsqueeze(-1)).squeeze(-1) + b
        transformed_cov = torch.matmul(torch.matmul(A, covariance), A.transpose(1, 2))
        return transformed_mean, transformed_cov
    
    def _build_encoder(self, in_shape, conv_channels, use_poolings, use_normalizations, kernel_size, pool_size, dropout_rate):
        layers = []
        in_channels, time_dim = in_shape[0], in_shape[1]
        for i, out_channels in enumerate(conv_channels):
            ln_time_dim = (time_dim + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
            layers.append(_get_conv1d_layer(
                in_channels, out_channels, ln_shape=(out_channels, ln_time_dim),
                kernel_size=kernel_size, pool_size=pool_size,
                use_pooling=use_poolings[i], use_normalization=use_normalizations[i],
                dropout_rate=dropout_rate,
            ))
            in_channels = out_channels
            time_dim = ln_time_dim // pool_size if use_poolings[i] else ln_time_dim
        return nn.Sequential(*layers)

    def _calculate_dense_input_size(self, in_shape):
        with torch.no_grad():
            x = torch.zeros((1,) + in_shape)
            x = self._encoder(x)
            return self._flatten(x).shape[-1]

    def _build_dense_head(self, in_features, dense_neurons):
        layers = []
        curr = in_features
        for neurons in dense_neurons[:-1]:
            layers.append(nn.Linear(curr, neurons))
            layers.append(nn.ReLU())
            curr = neurons
        layers.append(nn.Linear(curr, dense_neurons[-1]))
        return nn.Sequential(*layers)