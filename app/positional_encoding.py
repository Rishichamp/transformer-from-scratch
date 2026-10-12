"""
Positional encoding implementations.

Self-attention is permutation-invariant -- without positional information,
"the cat sat on the mat" and "mat the on sat cat the" produce identical
attention patterns. Each scheme below breaks this symmetry differently.

Four schemes are implemented, selectable via ModelConfig.pos_encoding:

  - "learned":    nn.Embedding(max_seq_len, d_model), added to token embeddings.
                  Simple, but hard limit at max_seq_len (GPT-2 style).

  - "sinusoidal": Fixed sin/cos functions of position, added to token embeddings.
                  No parameters, works for any length seen at inference
                  (original "Attention Is All You Need").

  - "rope":       Rotary Position Embedding. Rotates query/key vectors by an
                  angle proportional to position, applied INSIDE attention
                  (not added to embeddings). The dot product Q.K after rotation
                  depends only on the RELATIVE position (m-n), not absolute
                  positions. Used in LLaMA, Mistral, Qwen.

  - "alibi":      Attention with Linear Biases. Adds a bias term to attention
                  scores proportional to -|m-n|, scaled per-head. No rotation,
                  no embedding addition -- just a penalty for distance.
                  Used in BLOOM, MPT.

  - "none":       No positional information at all (ablation baseline --
                  demonstrates why positional encoding is necessary).
"""

import math
import torch
import torch.nn as nn


# Learned absolute positional encoding (GPT-2 style)
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) token embeddings. Returns x + positional embeddings."""
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.embedding(positions)


# Sinusoidal positional encoding (original Transformer paper)
class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Precomputed as a buffer (not a parameter) -- no gradient, works for any
    sequence length up to the precomputed max.
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# RoPE -- Rotary Position Embedding
def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin caches for RoPE.

    Returns:
      cos, sin: each of shape (max_seq_len, d_head)

    Each pair of dimensions (2i, 2i+1) is rotated by angle theta_i * position,
    where theta_i = base^(-2i/d_head). Lower-frequency pairs (small i) encode
    coarse/long-range position; higher-frequency pairs (large i) encode
    fine-grained/local position -- analogous to a Fourier basis over positions.
    """
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head))  # (d_head/2,)
    positions = torch.arange(max_seq_len, dtype=torch.float)                             # (T,)
    freqs = torch.outer(positions, inv_freq)                                             # (T, d_head/2)
    # Duplicate each frequency for the (x, y) pair it rotates: (T, d_head)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: (x1, x2) -> (-x2, x1). Standard RoPE helper."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    q, k:    (B, n_heads, T, d_head)
    cos, sin: (T, d_head) -- sliced from the precomputed cache for the current sequence length

    The rotation is: q_rotated = q*cos + rotate_half(q)*sin
    This is equivalent to treating each (2i,2i+1) pair as a 2D vector and
    rotating it by angle theta_i * position -- a position-dependent rotation
    matrix applied to each frequency pair.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (T, d_head) -> (1, 1, T, d_head)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ALiBi -- Attention with Linear Biases
def build_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    Build the ALiBi bias matrix added to attention scores BEFORE softmax.

    bias[h, i, j] = -slope_h * |i - j|   for j <= i (causal)

    Slopes are geometrically spaced per head: slope_h = 2^(-8h/n_heads) for h=1..n_heads
    (the original paper's formula for n_heads that are powers of 2).

    Heads with LARGE slopes penalise distant tokens HEAVILY -> attend mostly
    locally. Heads with SMALL slopes penalise distant tokens MILDLY -> attend
    more globally. This gives the model both local and global heads "for free",
    without any learned parameters, and generalises to sequence lengths longer
    than seen during training.

    Returns: (n_heads, seq_len, seq_len)
    """
    slopes = torch.tensor([2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)])  # (n_heads,)

    positions = torch.arange(seq_len)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (T, T) = |i-j|

    bias = -slopes.view(n_heads, 1, 1) * distance.unsqueeze(0)  # (n_heads, T, T)
    return bias


# Factory
def build_positional_encoding(pos_encoding: str, max_seq_len: int, d_model: int):
    """
    Returns an nn.Module for "learned"/"sinusoidal"/"none" (applied to embeddings),
    or None for "rope"/"alibi" (applied INSIDE attention -- see model.py).
    """
    if pos_encoding == "learned":
        return LearnedPositionalEncoding(max_seq_len, d_model)
    elif pos_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(max_seq_len, d_model)
    elif pos_encoding in ("rope", "alibi", "none"):
        return None
    else:
        raise ValueError(f"Unknown pos_encoding: {pos_encoding}")
