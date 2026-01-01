"""
AES Combined Model: Prototypical Network + Domain Adversarial Training (DANN).
Adapts features to be domain-invariant while learning interpretable prototypes for quality prediction.
"""

from typing import Sequence

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# --- GRL Component ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

# --- Helper ---
def _get_conv1d_layer(in_channels, out_channels, ln_shape, kernel_size, pool_size, use_pooling, use_normalization, dropout_rate):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
    if use_normalization: layers.append(nn.LayerNorm(ln_shape))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    if use_pooling: layers.append(nn.MaxPool1d(kernel_size=pool_size))
    return nn.Sequential(*layers)

@gin.configurable
class PrototypicalDANNHead(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int],
        conv_channels: Sequence[int] = (32, 32),
        dense_neurons: Sequence[int] = (64, 32, 14), # AES Probabilistic
        
        # --- Domain Classifier Params ---
        domain_classifier_neurons: Sequence[int] = (64, 32),
        num_domains: int = 2,
        
        # --- Prototype Params ---
        proto_embed_dim: int = 64,
        num_prototypes: int = 10,
        num_dims: int = 4, # AES has 4 dimensions
        
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
        self._flatten = nn.Flatten()
        encoder_out_dim = self._calculate_dense_input_size(in_shape)

        # 2. Quality Head (Direct Regression)
        self._quality_head = self._build_dense_head(encoder_out_dim, dense_neurons)

        # 3. Domain Classifier Head
        self._domain_classifier = self._build_dense_head(encoder_out_dim, list(domain_classifier_neurons) + [num_domains])
        self.grl = GradientReversalLayer()

        # 4. Prototypical Path
        self._num_dims = num_dims
        self._proto_embedder = nn.Sequential(nn.Linear(encoder_out_dim, proto_embed_dim), nn.ReLU())
        self._prototypes = nn.Parameter(torch.randn(self._num_dims, num_prototypes, proto_embed_dim))
        self._mean_regressors = nn.ModuleList([
            nn.Sequential(nn.Linear(num_prototypes, 16), nn.ReLU(), nn.Linear(16, 1)) for _ in range(self._num_dims)
        ])
        
        # Combiner: [ProtoMean(4) + DirectMean(4)] -> FinalMean(4)
        self._mean_combiner = nn.Linear(self._num_dims * 2, self._num_dims)

        # Output logic
        self._softplus = nn.Softplus()
        self._output_dim = dense_neurons[-1]
        
        # Determine targets (AES: 14 out -> 4 targets)
        if self._output_dim == 14: self._n_targets = 4
        elif self._output_dim == 20: self._n_targets = 5
        else: self._n_targets = self._output_dim

        self._covariance_indices = self._get_covariance_indices(self._output_dim)

    def forward(self, x: torch.Tensor, domain_weight: float = 1.0) -> tuple:
        # Encoder
        encoder_features = self._encoder(x)
        flat_features = self._flatten(encoder_features)
        
        # --- Path A: Prototypical Mean ---
        h_embed = self._proto_embedder(flat_features) 
        h_expanded = h_embed.unsqueeze(1).unsqueeze(1) # (B, 1, 1, D)
        protos_expanded = self._prototypes.unsqueeze(0) # (1, 4, K, D)
        similarities = F.cosine_similarity(h_expanded, protos_expanded, dim=-1) # (B, 4, K)
        
        mean_proto_list = []
        for i in range(self._num_dims):
            mean_proto_list.append(self._mean_regressors[i](similarities[:, i, :]))
        mean_proto = torch.cat(mean_proto_list, dim=1) # (B, 4)

        # --- Path B: Direct Quality ---
        quality_raw_preds = self._quality_head(flat_features)

        # --- Path C: Domain Classifier (Adversarial) ---
        reversed_features = self.grl(flat_features, domain_weight)
        domain_logits = self._domain_classifier(reversed_features)

        # --- Combine & Output ---
        if self._output_dim > self._n_targets: # Probabilistic
            direct_mean = quality_raw_preds[:, :self._n_targets]
            combined_input = torch.cat([mean_proto, direct_mean], dim=1)
            final_mean = self._mean_combiner(combined_input)
            
            cov_matrix = self._get_covariance_matrix(quality_raw_preds)
            
            if self._apply_linear_transform:
                final_mean, cov_matrix = self._linear_transform(final_mean, cov_matrix)
            
            return (final_mean, cov_matrix), domain_logits
        else:
            # Non-probabilistic implementation skipped for brevity
            return quality_raw_preds, domain_logits

    def get_prototype_diversity_loss(self) -> torch.Tensor:
        loss_diversity = 0.0
        for i in range(self._num_dims):
            protos = self._prototypes[i]
            protos_norm = F.normalize(protos, p=2, dim=1)
            sim_matrix = torch.matmul(protos_norm, protos_norm.t())
            identity = torch.eye(protos.size(0), device=protos.device)
            loss_diversity += ((sim_matrix - identity) ** 2).mean()
        return loss_diversity / self._num_dims

    # --- Helpers ---
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

    def _build_encoder(self, in_shape, conv_channels, use_poolings, use_normalizations, kernel_size, pool_size, dropout_rate):
        layers = []
        in_channels, time_dim = in_shape[0], in_shape[1]
        for i, out_channels in enumerate(conv_channels):
            ln_time_dim = (time_dim + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
            layers.append(_get_conv1d_layer(in_channels, out_channels, (out_channels, ln_time_dim), kernel_size, pool_size, use_poolings[i], use_normalizations[i], dropout_rate))
            in_channels = out_channels
            time_dim = ln_time_dim // pool_size if use_poolings[i] else ln_time_dim
        return nn.Sequential(*layers)

    def _calculate_dense_input_size(self, in_shape):
        with torch.no_grad(): return self._flatten(self._encoder(torch.zeros((1,) + in_shape))).shape[-1]

    def _build_dense_head(self, in_features, dense_neurons):
        layers = []
        curr = in_features
        for neurons in dense_neurons[:-1]:
            layers.append(nn.Linear(curr, neurons)); layers.append(nn.ReLU()); curr = neurons
        layers.append(nn.Linear(curr, dense_neurons[-1]))
        return nn.Sequential(*layers)