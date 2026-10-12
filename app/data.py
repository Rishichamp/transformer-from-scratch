"""
Dataset loading and batching for character-level language modelling.

The dataset is a single long sequence of token ids. Training batches are
constructed by sampling random starting positions and slicing fixed-length
windows -- the standard approach for autoregressive LM pretraining on a
single corpus (used in nanoGPT and similar minimal implementations).
"""

import torch
from pathlib import Path

from app.tokenizer import CharTokenizer


def load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def prepare_data(
    text: str, tokenizer: CharTokenizer, train_split: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the full corpus and split into train/val tensors of token ids."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(len(data) * train_split)
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) sequences.

    For each sequence, target is the input shifted by one position --
    target[i] = input[i+1], i.e. "predict the next character".

    Returns:
      x: (batch_size, block_size) input token ids
      y: (batch_size, block_size) target token ids (x shifted by 1)
    """
    assert len(data) > block_size, (
        f"Dataset has {len(data)} tokens, need at least {block_size + 1} for block_size={block_size}"
    )
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, batch_size: int, block_size: int, eval_iters: int = 20, device: str = "cpu") -> float:
    """Average loss over eval_iters random batches -- a low-variance loss estimate."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()
