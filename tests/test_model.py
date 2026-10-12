"""
Tests for the from-scratch transformer implementation.

These tests verify the MATHEMATICAL PROPERTIES of the architecture,
not just that it "runs" -- this is the difference between code that
executes and code that is correct.
"""
import math
import torch
import pytest

from app.config import ModelConfig
from app.model import GPTLanguageModel, MultiHeadAttention, GELUFeedForward, SwiGLUFeedForward
from app.positional_encoding import (
    build_rope_cache, apply_rope, rotate_half, build_alibi_bias,
    LearnedPositionalEncoding, SinusoidalPositionalEncoding,
)


def make_config(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=32, d_model=32, n_heads=4, n_layers=2,
        max_seq_len=16, dropout=0.0, pos_encoding="rope", activation="gelu",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ── Shape and forward-pass tests ──────────────────────────────────

@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope", "alibi", "none"])
def test_forward_pass_shapes(pos_encoding):
    config = make_config(pos_encoding=pos_encoding)
    model = GPTLanguageModel(config)
    B, T = 2, 10
    idx = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None  # no targets provided

    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0
    assert loss.dim() == 0  # scalar


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_feedforward_variants(activation):
    config = make_config(activation=activation)
    model = GPTLanguageModel(config)
    B, T = 2, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, _ = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert not torch.isnan(logits).any()


def test_attention_returns_weights_summing_to_one():
    """Each row of the attention matrix is a probability distribution (sums to 1)."""
    config = make_config()
    model = GPTLanguageModel(config)
    B, T = 1, 8
    idx = torch.randint(0, config.vocab_size, (B, T))

    _, _, attentions = model(idx, return_attention=True)
    assert len(attentions) == config.n_layers

    for layer_attn in attentions:
        assert layer_attn.shape == (B, config.n_heads, T, T)
        row_sums = layer_attn.sum(dim=-1)  # sum over key positions
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ── Causal masking ──────────────────────────────────────────────────

@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope", "alibi", "none"])
def test_causal_mask_prevents_attending_to_future(pos_encoding):
    """Attention weight at [query=i, key=j] must be 0 for all j > i, for every pos encoding."""
    config = make_config(pos_encoding=pos_encoding)
    model = GPTLanguageModel(config)
    B, T = 1, 8
    idx = torch.randint(0, config.vocab_size, (B, T))

    _, _, attentions = model(idx, return_attention=True)

    for layer_attn in attentions:
        # layer_attn: (B, n_heads, T, T)
        for i in range(T):
            for j in range(i + 1, T):
                assert torch.allclose(
                    layer_attn[:, :, i, j], torch.zeros_like(layer_attn[:, :, i, j]), atol=1e-6
                ), f"pos_encoding={pos_encoding}: query {i} attends to future key {j}"


def test_causal_mask_diagonal_and_below_nonzero():
    """Each token attends to itself and prior tokens with positive weight."""
    config = make_config()
    model = GPTLanguageModel(config)
    B, T = 1, 6
    idx = torch.randint(0, config.vocab_size, (B, T))
    _, _, attentions = model(idx, return_attention=True)

    for layer_attn in attentions:
        for i in range(T):
            for j in range(i + 1):
                assert (layer_attn[:, :, i, j] > 0).all()


# ── Weight tying ───────────────────────────────────────────────────

def test_weight_tying():
    """lm_head.weight and token_embedding.weight must be the SAME tensor object."""
    config = make_config()
    model = GPTLanguageModel(config)
    assert model.lm_head.weight is model.token_embedding.weight


def test_num_parameters_accounts_for_tying():
    """num_parameters should not double-count the tied embedding/lm_head weight."""
    config = make_config()
    model = GPTLanguageModel(config)
    raw_total = sum(p.numel() for p in model.parameters())
    # raw_total counts the tied weight TWICE (once via embedding, once via lm_head
    # parameter list iteration is per-MODULE, but since they're the same Parameter
    # object, sum() over model.parameters() may or may not double count depending
    # on de-duplication -- num_parameters explicitly subtracts one copy).
    assert model.num_parameters == raw_total - model.token_embedding.weight.numel()


# ── Gradient flow ────────────────────────────────────────────────────

@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope", "alibi", "none"])
def test_gradients_flow_to_all_parameters(pos_encoding):
    """After backward(), every parameter that should be trainable has a gradient."""
    config = make_config(pos_encoding=pos_encoding)
    model = GPTLanguageModel(config)
    B, T = 2, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, loss = model(idx, targets)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


# ── Positional encoding mathematical properties ──────────────────────

def test_sinusoidal_encoding_shape_and_range():
    pe = SinusoidalPositionalEncoding(max_seq_len=20, d_model=16)
    x = torch.zeros(1, 10, 16)
    out = pe(x)
    assert out.shape == (1, 10, 16)
    # sin/cos values are bounded in [-1, 1]
    assert out.abs().max() <= 1.0 + 1e-6


def test_learned_encoding_is_trainable():
    pe = LearnedPositionalEncoding(max_seq_len=20, d_model=16)
    assert pe.embedding.weight.requires_grad


def test_rope_preserves_vector_norm():
    """
    RoPE applies a ROTATION -- rotations are norm-preserving (orthogonal
    transformations). ||rotate(q)|| == ||q|| for any position.
    """
    d_head = 8
    cos, sin = build_rope_cache(max_seq_len=10, d_head=d_head)

    q = torch.randn(1, 1, 5, d_head)  # (B, n_heads, T, d_head)
    q_rot, _ = apply_rope(q, q.clone(), cos[:5], sin[:5])

    norm_before = q.norm(dim=-1)
    norm_after = q_rot.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_rope_relative_position_property():
    """
    THE key RoPE property: the dot product between a rotated query at
    position m and a rotated key at position n depends ONLY on (m - n),
    not on the absolute values of m and n.

    Verify: dot(rotate(q, m), rotate(k, n)) is the same for any (m, n)
    pair with the same difference m - n.
    """
    d_head = 8
    max_len = 20
    cos, sin = build_rope_cache(max_seq_len=max_len, d_head=d_head)

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    def rotated_dot(m: int, n: int) -> float:
        cos_m, sin_m = cos[m:m+1], sin[m:m+1]
        cos_n, sin_n = cos[n:n+1], sin[n:n+1]
        q_rot, _ = apply_rope(q, q.clone(), cos_m, sin_m)
        _, k_rot = apply_rope(k.clone(), k, cos_n, sin_n)
        return (q_rot[0, 0, 0] @ k_rot[0, 0, 0]).item()

    # Pairs (m, n) with the same relative distance m-n=3
    pairs = [(3, 0), (7, 4), (10, 7), (15, 12)]
    dots = [rotated_dot(m, n) for m, n in pairs]

    for d in dots[1:]:
        assert math.isclose(d, dots[0], rel_tol=1e-4, abs_tol=1e-4), (
            f"RoPE relative-position property violated: {dots}"
        )

    # Sanity: a DIFFERENT relative distance should generally give a different dot product
    different_distance_dot = rotated_dot(5, 0)  # distance 5, not 3
    assert not math.isclose(different_distance_dot, dots[0], rel_tol=1e-4, abs_tol=1e-4)


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    # First half becomes -second_half, second half becomes first_half
    assert torch.allclose(rotated, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_alibi_bias_shape_and_decay():
    """ALiBi bias must be 0 on the diagonal (distance 0) and increasingly
    negative for tokens further in the past (monotonic decay with distance)."""
    n_heads, seq_len = 4, 10
    bias = build_alibi_bias(n_heads, seq_len)
    assert bias.shape == (n_heads, seq_len, seq_len)

    for h in range(n_heads):
        # Diagonal is exactly 0 (distance 0 -> no penalty)
        diag = torch.diagonal(bias[h])
        assert torch.allclose(diag, torch.zeros_like(diag))

        # For a fixed query position, bias should be monotonically
            # As j increases, key moves closer to query -> bias increases (less negative).
        query_pos = seq_len - 1
        row = bias[h, query_pos, :query_pos + 1]  # bias[query, 0..query]
        for j in range(1, len(row)):
            assert row[j] >= row[j - 1] - 1e-6, "ALiBi: closer keys have less-negative bias"


def test_alibi_slopes_decrease_across_heads():
    """Head 0 should have the STEEPEST slope (most local), later heads gentler (more global)."""
    n_heads, seq_len = 4, 10
    bias = build_alibi_bias(n_heads, seq_len)

    # Compare bias at distance=seq_len-1 (furthest) across heads -- head 0
    # should have the most negative (largest magnitude) bias
    far_biases = [bias[h, seq_len - 1, 0].item() for h in range(n_heads)]
    for i in range(len(far_biases) - 1):
        assert far_biases[i] <= far_biases[i + 1], (
            "ALiBi slopes should decrease (penalty should weaken) across heads"
        )


# ── Generation ────────────────────────────────────────────────────

def test_generate_produces_correct_length():
    config = make_config()
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 14)


def test_generate_respects_max_seq_len():
    """Generation should not crash even when total length exceeds max_seq_len
    (the model truncates context via idx_cond)."""
    config = make_config(max_seq_len=8)
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=20, temperature=1.0)
    assert out.shape == (1, 24)


def test_generate_temperature_zero_is_deterministic():
    """temperature<=1e-6 should be greedy -- running twice gives identical output."""
    config = make_config()
    torch.manual_seed(0)
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))

    out1 = model.generate(idx.clone(), max_new_tokens=10, temperature=0.0)
    out2 = model.generate(idx.clone(), max_new_tokens=10, temperature=0.0)
    assert torch.equal(out1, out2)


# ── Sequence length boundary ───────────────────────────────────────

def test_forward_raises_on_excessive_sequence_length():
    config = make_config(max_seq_len=8)
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (1, 9))  # 9 > max_seq_len=8
    with pytest.raises(AssertionError):
        model(idx)
