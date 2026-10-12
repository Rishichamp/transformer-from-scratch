"""
From-scratch decoder-only transformer (GPT architecture).

Design goals:
  - Every component is implemented from first principles (no nn.MultiheadAttention,
    no nn.TransformerEncoder) so the architecture is fully inspectable.
  - Positional encoding is PLUGGABLE via ModelConfig.pos_encoding:
    "learned" / "sinusoidal" (added to embeddings) or "rope" / "alibi"
    (applied inside attention) or "none" (ablation).
  - Activation is PLUGGABLE: "gelu" (standard FFN) or "swiglu" (LLaMA-style gated FFN).
  - Attention weights are returned from forward() so Project 2's
    interpretability toolkit can analyse them directly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import ModelConfig
from app.positional_encoding import (
    build_positional_encoding, build_rope_cache, build_alibi_bias, apply_rope,
)


# Feed-forward variants
class GELUFeedForward(nn.Module):
    """Standard position-wise FFN: Linear -> GELU -> Linear. (BERT/GPT-2 style)"""
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        d_ff = ffn_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    """
    Gated FFN: (SiLU(xW1) * xW3) @ W2.  (LLaMA / Mistral / Gemma style)

    Uses 2/3 of the "equivalent" hidden dim per matrix so the total parameter
    count is comparable to a standard 4x GELU FFN despite having 3 matrices.
    """
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        d_ff = int(ffn_mult * d_model * 2 / 3)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def build_feedforward(config: ModelConfig) -> nn.Module:
    if config.activation == "gelu":
        return GELUFeedForward(config.d_model, config.ffn_mult, config.dropout)
    elif config.activation == "swiglu":
        return SwiGLUFeedForward(config.d_model, config.ffn_mult, config.dropout)
    else:
        raise ValueError(f"Unknown activation: {config.activation}")


# Multi-head causal self-attention
class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with pluggable positional encoding.

      - "rope":  rotates Q and K before the dot product (relative positions)
      - "alibi": adds a distance-based bias to attention scores before softmax
      - other:   plain scaled dot-product attention (positional info, if any,
                 was already added to the input embeddings)

    Returns (output, attention_weights) -- weights are post-softmax,
    shape (B, n_heads, T, T), used by Project 2 for interpretability analysis.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.pos_encoding = config.pos_encoding

        self.W_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.W_out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop  = nn.Dropout(config.dropout)

        if config.pos_encoding == "rope":
            cos, sin = build_rope_cache(config.max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        if config.pos_encoding == "alibi":
            alibi = build_alibi_bias(config.n_heads, config.max_seq_len)
            self.register_buffer("alibi_bias", alibi, persistent=False)

        # Causal mask: (1, 1, max_seq_len, max_seq_len), lower-triangular = allowed
        causal = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, config.max_seq_len, config.max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        qkv = self.W_qkv(x)
        q, k, v = qkv.split(C, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.pos_encoding == "rope":
            cos = self.rope_cos[:T].to(x.dtype)
            sin = self.rope_sin[:T].to(x.dtype)
            q, k = apply_rope(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        if self.pos_encoding == "alibi":
            scores = scores + self.alibi_bias[:, :T, :T].to(scores.dtype).unsqueeze(0)

        causal = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~causal, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights_dropped = self.drop(weights)

        out = torch.matmul(weights_dropped, v)            # (B, nh, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out), weights


# Transformer block (Pre-LN)
class TransformerBlock(nn.Module):
    """
    One transformer layer: attention + FFN, each with residual connection
    and Pre-LayerNorm (normalise BEFORE the sublayer -- more stable training
    than the original Post-LN, used by GPT-2 onward).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = build_feedforward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


# Full model
class GPTLanguageModel(nn.Module):
    """
    Decoder-only transformer language model.

    forward() returns (logits, loss) by default, or (logits, loss, attention_weights)
    when return_attention=True -- attention_weights is a list of (B, n_heads, T, T)
    tensors, one per layer. This is what Project 2's interpretability toolkit consumes.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = build_positional_encoding(
            config.pos_encoding, config.max_seq_len, config.d_model
        )
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: input embedding and output projection share weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style initialisation: N(0, 0.02) for Linear and Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        # Subtract tied weights once (lm_head.weight IS token_embedding.weight)
        return sum(p.numel() for p in self.parameters()) - self.token_embedding.weight.numel()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None, return_attention: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.token_embedding(idx)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.drop(x)

        attentions = []
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_attention:
                attentions.append(attn_weights)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )

        if return_attention:
            return logits, loss, attentions
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, and/or top-p (nucleus) sampling.
        idx: (B, T) starting token ids. Returns (B, T + max_new_tokens).
        """
        from app.generation import sample_next_token

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]  # (B, vocab_size)
            next_token = sample_next_token(next_logits, temperature, top_k, top_p)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
