"""
Sampling strategies for autoregressive generation.

Three knobs, composable:
  - temperature: rescales logits before softmax. T=1.0 leaves the distribution
                  unchanged; T<1.0 sharpens it (more greedy); T>1.0 flattens it
                  (more random). T->0 approaches argmax (greedy decoding).
  - top_k:        truncate to the k highest-probability tokens before sampling.
                  Prevents sampling from the long low-probability tail.
  - top_p:        nucleus sampling -- truncate to the smallest set of tokens
                  whose cumulative probability exceeds p. Adapts the truncation
                  size to the shape of the distribution (peaked vs flat).

top_k and top_p can be combined: top_k is applied first, then top_p further
restricts within that set.
"""

import torch
import torch.nn.functional as F


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set all but the top-k logits to -inf. logits: (B, vocab_size)."""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability >= p, set the rest to -inf. logits: (B, vocab_size).
    """
    if p is None or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Tokens to remove: cumulative prob EXCEEDS p, but always keep at least the first token
    sorted_remove = cumulative_probs > p
    sorted_remove[..., 0] = False
    # Shift right so we keep the token that CROSSES the threshold
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> torch.Tensor:
    """
    Sample the next token from logits. logits: (B, vocab_size).
    Returns: (B, 1) token ids.

    temperature <= 1e-6 is treated as greedy (argmax) regardless of top_k/top_p.
    """
    if temperature <= 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
