"""GMP_X6 model registry — predictor-side port.

Mirrors `py_models/models.py` but imports its blocks from `predictor/_blocks.py`
so the predictor package runs standalone with `torch` only — no `py_models`
import. Class bodies are identical, so X6 unified checkpoints (the proposed
family included: its Wiener weights live in the state dict) load as-is.

Keep this file in sync with `py_models/models.py` whenever model definitions
change. (`predictor/_blocks.py` likewise mirrors `py_models/model_blocks.py`.)
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
    """Decomposition-Linear: avg-pool trend + residual seasonal, two linear heads.

    The linear layers are initialised to `1/input_window * ones`, matching the
    Time-Series-Library reference. This biases the model toward predicting the
    lookback mean — useful for the persistence-trap regime.

    An implementation variant (docs/review.md C2): the moving average
    zero-pads its window (`AvgPool1d(padding=k//2)`), where the LTSF-Linear
    reference repeats the end values — a material difference on short
    inputs. The two heads can express any linear map of the window, so a
    gap to the closed-form FIR reflects optimisation/conditioning, not
    expressiveness.
    """
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
    """Channel-independent PatchTST (ICLR 2023) — this repo's implementation
    in the Time-Series-Library layout, not the original supervised backbone
    (docs/review.md C2): LayerNorm inside the encoder blocks plus an outer
    BatchNorm, sinusoidal patch positions, no residual attention, and no
    input normalisation (no RevIN / per-window standardisation: inputs are
    globally scaled; the proposed family's σ wrapper supplies it there).
    Results are this implementation's, not a ranking of the original.

    Single-channel only.
    """
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 factor: int = 1, activation: str = "gelu"):
        super().__init__()
        assert in_channels == 1, "PatchTST in GMP_X6 currently single-channel only"
        self.input_window = input_window
        self.output_window = output_window
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_len = patch_len
        self.stride = stride

        # PatchEmbedding pads `stride` samples on the right via replicate
        # padding before unfolding, so the patch count picks up the trailing
        # samples of the lookback that a plain unfold would drop.
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
        # x: (B, T, C)  -> (B, C, T)
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)            # (B*C, P, D)
        enc_out, _ = self.encoder(enc_out)                   # (B*C, P, D)
        enc_out = enc_out.reshape(-1, n_vars,
                                  enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)                # (B, C, D, P)
        dec_out = self.head(enc_out)                         # (B, C, H)
        return dec_out.permute(0, 2, 1)                      # (B, H, C)


# ------------------------------------------------------------------ ModernTCN

class ModernTCN(nn.Module):
    """ModernTCN (Luo & Wang, ICLR 2024) — compact single-stage variant.

    Adapted for narrow-band (0.05-0.5 Hz) 8 Hz seismic signals: no RevIN
    (windows are bandpass zero-mean), no structural reparam (kernel here
    already covers the band), no multi-stage (sequence is short after
    patching), no series_decomp (bandpass strips the trend).
    """
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
        # (B, T, C) -> (B, C, T) -> (B, dim, n_patches) -> (B, H, C)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.view(x.size(0), self.output_window, self.out_channels)


# ------------------------------------------------------------------- Proposed

# Backbones the proposed model accepts, each with the output layer that is
# zero-initialised so the residual branch starts at exactly 0.
_BACKBONE_HEAD = {
    "gru":       lambda m: m.fc,
    "lstm":      lambda m: m.fc,
    "patch-tst": lambda m: m.head.linear,
    "moderntcn": lambda m: m.head,
}

WIENER_INITS = ("wls", "closed_form", "random")
LINEAR_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class Proposed(nn.Module):
    """Classical-residual, scale-equivariant forecaster (docs/plan.md §3.4):

        ŷ = W·x + σ(x) · g_θ(x / σ(x))

    W·x    linear branch: one (H, K) FIR map shared across channels, no bias,
           in `linear_dtype` arithmetic (docs/review.md R2). "float64" keeps
           the canonical fit exact: float64 weights on float64 windows, so
           step 0 is the classical `wls` forecast itself (its weights reach
           ~1e8 at 2.5 s and ~1e12 at 20 s, docs/findings.md 84, and a
           float32 window rounds them into the forecast's error). The model
           then takes float64 windows (`input_dtype`; the datamodule and
           predictor.GMP serve them) and returns float64; g_θ still runs in
           float32 on x/σ. "float32" (the default) is the arithmetic of every
           checkpoint trained before 2026-10-02: float32 windows, W rounded
           to float32 — a rounded-input model, not the float64 baseline.
           The runner sets W before training on the training split
           (`init_linear`): `wiener_init="wls"` (default) is the
           canonical weighted least-squares fit exp2_classical persists
           (py_models.classical.fit_ls: float64, QR, val-selected ridge;
           py_utils.runner.init_proposed), the `wls` baseline's own W;
           "closed_form" is the Toeplitz Wiener fit (`fit_wiener`, far from
           the LS fit on this input, docs/findings.md 14); "random" keeps
           nn.Linear's init.
           `freeze_linear=True` keeps W fixed at that fit (the design:
           the residual doesn't train from the Toeplitz W, and fine-tuning
           the wls W drifts; findings 18, 20).
    σ(x)   window RMS per channel (RevIN-style, without mean removal), so
           f(αx) = α f(x) for α > 0 wherever the RMS floor is inactive
           (σ(x) and σ(αx) both above `eps`): W·x is homogeneous and g_θ only
           ever sees the scale-free x / σ(x). Below the floor σ is clamped
           and homogeneity breaks; an all-zero window returns g_θ's output
           at zero input scaled by eps (≈ 0, not exactly 0). Equivariance
           is to pure rescaling only — not to spectral change, additive
           noise or seasonal transfer (E5 tests that empirically).
    g_θ    backbone (GRU by default; LSTM, PatchTST or ModernTCN swap in) on
           the normalised window: the nonlinear remainder. Its output layer
           is zero-initialised, so the model starts exactly at W·x (the
           step-0 validation is W alone).

    E3 ablations are constructor flags: `use_linear=False` drops W·x,
    `scale_equivariant=False` feeds g_θ the raw window (ŷ = W·x + g_θ(x)),
    `wiener_init` picks W's start (read by the runner; `wiener_ridge` only
    for "closed_form"), `freeze_linear` fixes or fine-tunes W, `backbone`
    swaps g_θ, `linear_dtype` sets the linear branch's arithmetic (above).
    Every other keyword is a backbone hyperparameter.
    """
    def __init__(self, in_channels: int, out_channels: int, input_window: int,
                 output_window: int, backbone: str = "gru",
                 use_linear: bool = True, scale_equivariant: bool = True,
                 wiener_init: str = "wls", wiener_ridge: float = 1e-6,
                 freeze_linear: bool = False, eps: float = 1e-6,
                 linear_dtype: str = "float32", **backbone_cfg):
        super().__init__()
        if backbone not in _BACKBONE_HEAD:
            raise ValueError(f"backbone must be one of {list(_BACKBONE_HEAD)}; "
                             f"got {backbone!r}")
        if wiener_init not in WIENER_INITS:
            raise ValueError(f"wiener_init must be one of {WIENER_INITS}; "
                             f"got {wiener_init!r}")
        if linear_dtype not in LINEAR_DTYPES:
            raise ValueError(f"linear_dtype must be one of {list(LINEAR_DTYPES)}; "
                             f"got {linear_dtype!r}")
        if in_channels != out_channels:
            raise ValueError("the proposed model forecasts each input channel")
        self.output_window, self.out_channels = output_window, out_channels
        self.scale_equivariant = bool(scale_equivariant)
        self.wiener_init = wiener_init
        self.wiener_ridge = float(wiener_ridge)
        self.eps = float(eps)
        self.linear_dtype = linear_dtype
        # The window dtype the model takes (and returns); g_θ runs in float32.
        self.input_dtype = LINEAR_DTYPES[linear_dtype]
        self.linear = (nn.Linear(input_window, output_window, bias=False,
                                 dtype=self.input_dtype)
                       if use_linear else None)
        if self.linear is not None and freeze_linear:
            self.linear.weight.requires_grad_(False)
        self.backbone = _REGISTRY[backbone](
            in_channels=in_channels, out_channels=out_channels,
            input_window=input_window, output_window=output_window,
            **backbone_cfg)
        head = _BACKBONE_HEAD[backbone](self.backbone)
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def init_linear(self, weight) -> None:
        """Set W, shape (H, K): the linear fit named by `wiener_init`, kept
        in the branch's dtype (exact for linear_dtype float64)."""
        if self.linear is None:
            raise RuntimeError("no linear branch to initialise (use_linear=False)")
        w = torch.as_tensor(weight, dtype=self.linear.weight.dtype,
                            device=self.linear.weight.device)
        if w.shape != self.linear.weight.shape:
            raise ValueError(f"Wiener weight {tuple(w.shape)} vs linear branch "
                             f"{tuple(self.linear.weight.shape)}")
        self.linear.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, C) -> (B, H, C), both in input_dtype
        if x.dtype != self.input_dtype:
            raise TypeError(f"the proposed model (linear_dtype={self.linear_dtype}) "
                            f"takes {self.input_dtype} windows; got {x.dtype}")
        if self.scale_equivariant:
            sigma = x.pow(2).mean(dim=1, keepdim=True).sqrt() \
                     .clamp_min(self.eps).detach()               # (B, 1, C)
            out = sigma * self.backbone((x / sigma).float()).to(x.dtype)
        else:
            out = self.backbone(x.float()).to(x.dtype)
        if self.linear is not None:
            out = out + self.linear(x.transpose(1, 2)).transpose(1, 2)
        return out


# ------------------------------------------------------------------- registry

_REGISTRY = {
    "dlinear":   DLinear,
    "lstm":      LSTM,
    "gru":       GRU,
    "moderntcn": ModernTCN,
    "patch-tst": PatchTST,
    "proposed":  Proposed,
}


def build_model(model_type: str, model_cfg: dict) -> nn.Module:
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**model_cfg)


def model_keys() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = ["build_model", "model_keys"]
