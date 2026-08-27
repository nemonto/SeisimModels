"""Model registry — inference-side port of GMP_X5 `py_models/models.py`.

Architectures for every checkpoint this package ships: DLinear, LSTM, GRU,
PatchTST, ModernTCN, and the persistence baseline. Blocks come from
`._blocks`, so the package runs standalone on `torch` alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._blocks import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FlattenHead,
    FullAttention,
    PatchEmbedding,
    Transpose,
    _ModernTCNBlock,
)


# -------------------------------------------------------------------- DLinear

class DLinear(nn.Module):
    """Decomposition-Linear: avg-pool trend + residual seasonal, two linear heads."""
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, kernel_size: int = 25, individual: bool = False):
        super().__init__()
        self.output_window = output_window
        self.out_channels = out_channels
        self.input_window = input_window
        self.individual = individual
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        init = (1.0 / input_window) * torch.ones(output_window, input_window)
        if individual:
            self.lin_t = nn.ModuleList([nn.Linear(input_window, output_window) for _ in range(in_channels)])
            self.lin_s = nn.ModuleList([nn.Linear(input_window, output_window) for _ in range(in_channels)])
            for layer in self.lin_t:
                layer.weight = nn.Parameter(init.clone())
            for layer in self.lin_s:
                layer.weight = nn.Parameter(init.clone())
        else:
            self.lin_t = nn.Linear(input_window, output_window)
            self.lin_s = nn.Linear(input_window, output_window)
            self.lin_t.weight = nn.Parameter(init.clone())
            self.lin_s.weight = nn.Parameter(init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 2, 1)
        trend = self.avg(x_t)[:, :, :self.input_window]
        seas = x_t - trend
        if self.individual:
            ot = torch.stack([self.lin_t[i](trend[:, i, :]) for i in range(trend.size(1))], dim=1)
            os_ = torch.stack([self.lin_s[i](seas[:, i, :]) for i in range(seas.size(1))], dim=1)
        else:
            ot, os_ = self.lin_t(trend), self.lin_s(seas)
        return (ot + os_).permute(0, 2, 1)


# ----------------------------------------------------------------- LSTM / GRU

class LSTM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.output_window, self.out_channels = output_window, out_channels
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, out_channels * output_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), self.output_window, self.out_channels)


class GRU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.output_window, self.out_channels = output_window, out_channels
        self.gru = nn.GRU(
            input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, out_channels * output_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), self.output_window, self.out_channels)


# ------------------------------------------------------------------- PatchTST

class PatchTST(nn.Module):
    """Channel-independent PatchTST (ICLR 2023), TSL-faithful. Single-channel only."""
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 factor: int = 1, activation: str = "gelu"):
        super().__init__()
        assert in_channels == 1, "PatchTST in GMP_X5 currently single-channel only"
        self.input_window = input_window
        self.output_window = output_window
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_len = patch_len
        self.stride = stride

        padding = stride
        self.patch_embedding = PatchEmbedding(
            d_model=d_model, patch_len=patch_len, stride=stride,
            padding=padding, dropout=dropout,
        )

        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(mask_flag=False, factor=factor,
                                  attention_dropout=dropout,
                                  output_attention=False),
                    d_model, nhead,
                ),
                d_model=d_model, d_ff=dim_feedforward,
                dropout=dropout, activation=activation,
            )
            for _ in range(num_layers)
        ]
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2),
            ),
        )

        n_patches = int((input_window - patch_len) / stride + 2)
        self.head_nf = d_model * n_patches
        self.head = FlattenHead(
            n_vars=in_channels, nf=self.head_nf,
            target_window=output_window, head_dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars,
                                  enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        return dec_out.permute(0, 2, 1)


# ------------------------------------------------------------------ ModernTCN

class ModernTCN(nn.Module):
    """ModernTCN (Luo & Wang, ICLR 2024) — compact single-stage variant."""
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, patch_len: int = 8, patch_stride: int = 4,
                 dim: int = 64, n_blocks: int = 3, kernel_size: int = 17,
                 ffn_ratio: int = 1, dropout: float = 0.1):
        super().__init__()
        self.output_window = output_window
        self.out_channels  = out_channels
        self.patch_embed = nn.Conv1d(in_channels, dim, patch_len,
                                     stride=patch_stride, padding=patch_len // 2)
        n_patches = (input_window + 2 * (patch_len // 2) - patch_len) // patch_stride + 1
        self.blocks = nn.Sequential(*[
            _ModernTCNBlock(dim, kernel_size, ffn_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(dim * n_patches, out_channels * output_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.view(x.size(0), self.output_window, self.out_channels)


# ---------------------------------------------------------------------- Naive

class Naive(nn.Module):
    """Persistence baseline: repeat the last input timestep H times."""
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, **kwargs):
        super().__init__()
        self.output_window, self.out_channels = output_window, out_channels
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :self.out_channels]
        return last.expand(-1, self.output_window, -1)


# ------------------------------------------------------------------- registry

_REGISTRY = {
    "dlinear":   DLinear,
    "lstm":      LSTM,
    "gru":       GRU,
    "moderntcn": ModernTCN,
    "patch-tst": PatchTST,
    "naive":     Naive,
}

BASELINE_MODELS = frozenset({"naive"})


def build_model(model_type: str, model_cfg: dict) -> nn.Module:
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**model_cfg)


def model_keys() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = ["build_model", "model_keys", "BASELINE_MODELS"]
