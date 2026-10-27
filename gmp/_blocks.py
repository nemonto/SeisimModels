"""Building blocks for `predictor/_models.py` — predictor-side port.

Verbatim mirror of `py_models/model_blocks.py` so the predictor package runs
standalone with `torch` only. Keep the two files in sync.

Contents
--------
Shared helpers
    Transpose, TriangularCausalMask

PatchTST blocks (ported from Time-Series-Library)
    PositionalEmbedding, PatchEmbedding,
    FullAttention, AttentionLayer,
    EncoderLayer, Encoder,
    FlattenHead

ModernTCN block (compact single-stage variant of Luo & Wang, ICLR 2024)
    _ModernTCNBlock
"""
from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------- helpers

class Transpose(nn.Module):
    """Module wrapper for `torch.Tensor.transpose`. Used so a Sequential
    norm_layer can transpose around BatchNorm1d.
    """
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class TriangularCausalMask:
    """Boolean upper-triangular mask for causal attention. Kept for API parity
    with the ported FullAttention; PatchTST uses `mask_flag=False` so this is
    only constructed when a caller explicitly enables masking.
    """
    def __init__(self, B: int, L: int, device: str | torch.device = "cpu"):
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device),
                diagonal=1,
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


# ============================================================ PatchTST blocks

class PositionalEmbedding(nn.Module):
    """Standard sine/cosine positional encoding, registered as a buffer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Patch + linear projection + sine/cosine position, with right replicate-pad.

    Input  x: (B, C, T)
    Output (z, n_vars) where z: (B*C, n_patches, d_model),
                       n_patches = (T - patch_len) / stride + 2.
    """
    def __init__(self, d_model: int, patch_len: int, stride: int,
                 padding: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FullAttention(nn.Module):
    """Standard scaled dot-product attention over `(B, L, H, E)` queries."""
    def __init__(self, mask_flag: bool = True, factor: int = 1,
                 scale: float | None = None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float("-inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), (A if self.output_attention else None)


class AttentionLayer(nn.Module):
    """Multi-head wrapper: projects Q/K/V then calls inner attention."""
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys: int | None = None, d_values: int | None = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """Pre-attention residual + Conv1d FFN + LayerNorm.

    PatchTST overrides the encoder-level `norm_layer` to swap LayerNorm for
    BatchNorm via Transpose; this class still applies its own pre/post
    LayerNorm, which is the upstream behaviour.
    """
    def __init__(self, attention: nn.Module, d_model: int,
                 d_ff: int | None = None, dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Stack of EncoderLayer modules with an optional outer norm."""
    def __init__(self, attn_layers: list[nn.Module], norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class FlattenHead(nn.Module):
    """Channel-independent head: flatten patches per variate then linear.

    Input  x: (B, n_vars, d_model, n_patches)
    Output    (B, n_vars, target_window)
    """
    def __init__(self, n_vars: int, nf: int, target_window: int,
                 head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


# =========================================================== ModernTCN blocks

class _ModernTCNBlock(nn.Module):
    """Depthwise temporal mixing + pointwise ConvFFN with BatchNorm and residuals.

    Input/output shape: (B, dim, L).
    """
    def __init__(self, dim: int, kernel_size: int, ffn_ratio: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.dw    = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.norm1 = nn.BatchNorm1d(dim)
        hidden = dim * ffn_ratio
        self.ffn1  = nn.Conv1d(dim, hidden, 1)
        self.act   = nn.GELU()
        self.ffn2  = nn.Conv1d(hidden, dim, 1)
        self.norm2 = nn.BatchNorm1d(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(self.dw(x))
        x = x + self.drop(y)
        y = self.norm2(self.ffn2(self.act(self.ffn1(x))))
        return x + self.drop(y)
