"""Legacy GMP_X3 checkpoint support.

The five legacy families (LSTM, GRU, PatchTST, DLinear, CNN-LSTM) shipped
under `checkpoints/legacy/` were trained at 16 Hz on the GMP_X3 dataset with
0.5 s output. LSTM and GRU are architecture-identical to the current classes
and reuse them; the other three are verbatim ports from GMP_X3
`py_models/models.py` (class bodies and attribute names unchanged so the
legacy state dicts load as-is).

`LegacyGMP` adapts the X3 unified-checkpoint schema (meta.version "1.0",
window under `data.window_size`, no target_mode) to the `GMP` interface;
prediction, scaling, and seamed/overlap-add stitching are all inherited, so a
legacy checkpoint behaves exactly like a current one. Legacy targets are
absolute by construction.

`load_gmp(path)` sniffs the schema and returns the right class — use it for
anything path-driven.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ._models   import GRU, LSTM
from .predictor import GMP


# ---- verbatim X3 architectures (GMP_X3 py_models/models.py) -----------------

class CNNLSTMX3(nn.Module):
    def __init__(self, in_channels, out_channels, input_window, output_window,
                 hidden_size=64, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.output_window = output_window
        self.out_channels = out_channels

        # CNN feature extractor
        self.conv1 = nn.Conv1d(in_channels, hidden_size, kernel_size,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                               padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # LSTM on CNN features
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, out_channels * output_window)

    def forward(self, x):
        # CNN: (batch, time, channels) -> (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        # Back to (batch, time, features) for LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(out.size(0), self.output_window, self.out_channels)


class PatchTSTX3(nn.Module):
    """X3's PatchTST (plain nn.TransformerEncoder, learned pos. embedding) —
    a different architecture from the TSL-faithful X5 PatchTST."""
    def __init__(self, in_channels, out_channels, input_window, output_window,
                 patch_len=16, stride=8, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.output_window = output_window
        self.out_channels = out_channels
        self.patch_len = patch_len
        self.stride = stride

        self.num_patches = (input_window - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.dropout_layer = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.head = nn.Linear(self.num_patches * d_model,
                              out_channels * output_window)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T)         # (B*C, T)
        x = x.unfold(1, self.patch_len, self.stride)     # (B*C, P, patch_len)
        x = self.patch_proj(x) + self.pos_emb            # (B*C, P, d_model)
        x = self.dropout_layer(x)
        x = self.encoder(x)                              # (B*C, P, d_model)
        x = x.reshape(B * C, -1)
        x = self.head(x)                                 # (B*C, out_ch * H)
        x = x.reshape(B, C, self.out_channels, self.output_window)
        # sum over input channels (C) to produce out_channels output
        x = x.sum(dim=1)                                 # (B, out_ch, H)
        return x.permute(0, 2, 1)                        # (B, H, out_ch)


class DLinearX3(nn.Module):
    """X3's DLinear — identical math to the X5 class but different parameter
    names (`linear_trend`/`linear_seasonal`) and no 1/K initialisation."""
    def __init__(self, in_channels, out_channels, input_window, output_window,
                 kernel_size=25, individual=False):
        super().__init__()
        self.output_window = output_window
        self.out_channels = out_channels
        self.input_window = input_window
        self.individual = individual

        # Ensure kernel_size is odd so symmetric padding keeps length exact
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2)

        if individual:
            self.linear_trend    = nn.ModuleList(
                [nn.Linear(input_window, output_window) for _ in range(in_channels)])
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(input_window, output_window) for _ in range(in_channels)])
        else:
            self.linear_trend    = nn.Linear(input_window, output_window)
            self.linear_seasonal = nn.Linear(input_window, output_window)

    def forward(self, x):
        # x: (B, T, C)
        x_t = x.permute(0, 2, 1)
        trend = self.avg(x_t)[:, :, :self.input_window]
        seasonal = x_t - trend

        if self.individual:
            out_trend    = torch.stack(
                [self.linear_trend[i](trend[:, i, :]) for i in range(trend.size(1))], dim=1)
            out_seasonal = torch.stack(
                [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(seasonal.size(1))], dim=1)
        else:
            out_trend    = self.linear_trend(trend)
            out_seasonal = self.linear_seasonal(seasonal)

        out = out_trend + out_seasonal
        return out.permute(0, 2, 1)


_LEGACY_REGISTRY = {
    "lstm":      LSTM,
    "gru":       GRU,
    "dlinear":   DLinearX3,
    "patch-tst": PatchTSTX3,
    "cnn-lstm":  CNNLSTMX3,
}


# ---- loading ----------------------------------------------------------------

def is_legacy_checkpoint(ckpt: dict) -> bool:
    """X5 unified checkpoints stamp meta.version "X5-..."; X3's is "1.0"."""
    return not str(ckpt.get("meta", {}).get("version", "")).startswith("X5")


class LegacyGMP(GMP):
    """`predictor.GMP` over an X3 unified checkpoint."""
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device,
                          weights_only=False)
        if not is_legacy_checkpoint(ckpt):
            raise ValueError(f"{checkpoint_path} is an X5 checkpoint — "
                             "load it with predictor.GMP")

        self.model_type    = ckpt["meta"]["model_type"]
        d                  = ckpt["data"]
        self.input_window  = int(d["window_size"]["input"])
        self.output_window = int(d["window_size"]["output"])
        self.sampling_rate = int(d["sampling_rate"])
        self.target_mode   = "absolute"
        self.scaler_mean = np.asarray(d["scaler"]["mean"], dtype=np.float32)
        self.scaler_std  = np.asarray(d["scaler"]["std"],  dtype=np.float32)

        cls = _LEGACY_REGISTRY.get(self.model_type)
        if cls is None:
            raise ValueError(f"no legacy architecture for {self.model_type!r} "
                             f"(have {list(_LEGACY_REGISTRY)})")
        self.model = cls(**ckpt["model"]["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()


def load_gmp(checkpoint_path: str | Path, device: str | None = None) -> GMP:
    """Open a unified checkpoint with the loader matching its schema."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                      weights_only=False)
    cls = LegacyGMP if is_legacy_checkpoint(ckpt) else GMP
    return cls(checkpoint_path, device=device)
