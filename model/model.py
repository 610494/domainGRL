"""Multivariate Gaussian SSL-based projection head."""

from typing import Sequence

import gin
import torch
import torch.nn as nn


def _get_conv1d_layer(
    in_channels: int,
    out_channels: int,
    ln_shape: tuple[int],
    kernel_size: int,
    pool_size: int,
    use_pooling: bool,
    use_normalization: bool,
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
class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int],
        conv_channels: Sequence[int] = (32, 32),
        dense_neurons: Sequence[int] = (64, 32, 14), # 預設改為 14 (AES Probabilistic)
        use_poolings: Sequence[bool] = (True, True),
        use_normalizations: Sequence[bool] = (True, True),
        kernel_size: int = 5,
        pool_size: int = 5,
        dropout_rate: float = 0.3,
        apply_linear_transform: bool = True,
        # [新增] 用於控制輸出範圍的參數
        # NISQA (1-5): scale=2.0, bias=3.0
        # AES (1-10): scale=4.5, bias=5.5
        output_scale: float = 4.5, 
        output_bias: float = 5.5,
    ):
        super().__init__()

        if len(conv_channels) != len(use_poolings):
            raise ValueError(f"{conv_channels=} and {use_poolings=} must have same length.")
        if len(conv_channels) != len(use_normalizations):
            raise ValueError(f"{conv_channels=} and {use_normalizations=} must have same length.")
        
        # 儲存縮放參數
        self._output_scale = output_scale
        self._output_bias = output_bias
        self._apply_linear_transform = apply_linear_transform
        
        # Build encoder.
        self._encoder = self._build_encoder(
            in_shape, conv_channels, use_poolings, use_normalizations,
            kernel_size, pool_size, dropout_rate
        )
        
        # Build dense head.
        self._flatten = nn.Flatten()
        in_dense = self._calculate_dense_input_size(in_shape)
        self._head = self._build_dense_head(in_dense, dense_neurons)

        self._softplus = nn.Softplus()
        
        # [修改] 自動判斷目標維度 (n_targets)
        # 20 -> 5 (NISQA), 14 -> 4 (AES)
        out_dim = dense_neurons[-1]
        if out_dim == 20: self._n_targets = 5
        elif out_dim == 14: self._n_targets = 4
        elif out_dim == 5: self._n_targets = 5
        elif out_dim == 4: self._n_targets = 4
        else:
            # Fallback (預設當作 AES 4維)
            self._n_targets = 4 

        self._covariance_indices = self._get_covariance_indices(out_dim)

    def _build_encoder(self, in_shape, conv_channels, use_poolings, use_normalizations, kernel_size, pool_size, dropout_rate):
        layers = []
        current_ln_time_dim = self._calculate_conv_output_size(
            in_shape[1], kernel_size, pool_size=1) # Normalization before pooling
        
        layers.append(_get_conv1d_layer(
            in_shape[0], conv_channels[0], kernel_size=kernel_size, pool_size=pool_size,
            dropout_rate=dropout_rate, use_pooling=use_poolings[0],
            use_normalization=use_normalizations[0], ln_shape=(conv_channels[0], current_ln_time_dim),
        ))
        
        for i in range(1, len(conv_channels)):
            prev_ln_time_dim = current_ln_time_dim
            current_ln_time_dim = self._calculate_conv_output_size(
                prev_ln_time_dim, kernel_size, pool_size if use_poolings[i-1] else 1
            )
            layers.append(_get_conv1d_layer(
                conv_channels[i-1], conv_channels[i], kernel_size=kernel_size, pool_size=pool_size,
                dropout_rate=dropout_rate, use_pooling=use_poolings[i],
                use_normalization=use_normalizations[i], ln_shape=(conv_channels[i], current_ln_time_dim)
            ))
        return nn.Sequential(*layers)

    def _calculate_conv_output_size(self, input_size, kernel_size, pool_size, stride=1, padding=0, dilation=1):
        conv_time_output = (input_size // pool_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return conv_time_output

    def _calculate_dense_input_size(self, in_shape):
        with torch.no_grad():
            x = torch.zeros((1,) + in_shape)
            x = self._encoder(x)
            return self._flatten(x).shape[-1]

    def _build_dense_head(self, in_features, dense_neurons):
        layers = []
        layers.extend([nn.Linear(in_features, dense_neurons[0]), nn.ReLU()])
        for i in range(len(dense_neurons) - 1):
            layers.append(nn.Linear(dense_neurons[i], dense_neurons[i + 1]))
            if i < len(dense_neurons) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _get_covariance_indices(self, output_dim: int) -> list[tuple[int, int]]:
        """Get indices for covariance matrix construction."""
        if output_dim == 5: return []
        elif output_dim == 4: return []
        elif output_dim == 10: return [(-1, -1)] # NISQA Diagonal
        elif output_dim == 8: return [(-1, -1)]  # AES Diagonal
        elif output_dim == 20:
            # NISQA Full (5 dims)
            return [(i, j) for i in range(5) for j in range(i+1, 5)]
        elif output_dim == 14:
            # [新增] AES Full (4 dims)
            return [(i, j) for i in range(4) for j in range(i+1, 4)]
        else:
            raise ValueError(f"Unsupported output dimension: {output_dim=}.")

    def _get_covariance_matrix(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix using Cholesky decomposition approach."""
        batch_size, _ = predictions.shape
        n_targets = self._n_targets
        
        cov_matrix = torch.zeros((batch_size, n_targets, n_targets), device=predictions.device)
        
        # Indices: [0 ~ n-1] Mean
        #          [n ~ 2n-1] Variance (Diagonal)
        #          [2n ~ end] Covariance (Lower Triangular)
        var_predictions = self._softplus(predictions[:, n_targets:2*n_targets])
        cov_predictions = predictions[:, 2*n_targets:]
        
        for i in range(n_targets):
            cov_matrix[:, i, i] = var_predictions[:, i]
        
        for idx, (i, j) in enumerate(self._covariance_indices):
            if (i, j) == (-1, -1): break
            cov_matrix[:, i, j] = cov_predictions[:, idx]
        
        return torch.matmul(cov_matrix, cov_matrix.transpose(1, 2))

    def _linear_transform(self, mean: torch.Tensor, covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply linear transformation: y = Ax + b for unbiased estimation."""
        batch_size = mean.shape[0]
        device = mean.device
        n_targets = self._n_targets
        
        # [修改] 使用 Config 傳入的 scale 和 bias
        A = torch.diag_embed(torch.full((batch_size, n_targets), self._output_scale, device=device))
        b = torch.full((batch_size, n_targets), self._output_bias, device=device)
        
        transformed_mean = torch.matmul(A, mean.unsqueeze(-1)).squeeze(-1) + b
        transformed_cov = torch.matmul(torch.matmul(A, covariance), A.transpose(1, 2))
        return transformed_mean, transformed_cov

    def forward(self, x: torch.Tensor):
        x = self._encoder(x)
        x = self._flatten(x)
        predictions = self._head(x)
        
        # Non-probabilistic case
        if predictions.shape[-1] == self._n_targets:
            if self._apply_linear_transform:
                return self._output_scale * predictions + self._output_bias
            return predictions
        
        # Probabilistic case
        mean_predictions = predictions[:, :self._n_targets]
        cov_predictions = self._get_covariance_matrix(predictions)
        if self._apply_linear_transform:
            mean_predictions, cov_predictions = self._linear_transform(mean_predictions, cov_predictions)
            
        return mean_predictions, cov_predictions