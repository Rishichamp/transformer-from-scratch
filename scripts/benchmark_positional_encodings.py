"""
Benchmark script: compare training curves across positional encoding schemes.

Trains small models with identical architecture (same d_model, n_heads,
n_layers, seed) but different pos_encoding settings, and reports final
train/val loss for each. This directly operationalises the Module 5
discussion of learned vs sinusoidal vs RoPE vs ALiBi vs no positional
encoding.

Usage:
    python scripts/benchmark_positional_encodings.py
    python scripts/benchmark_positional_encodings.py --epochs 3 --steps-per-epoch 50

Expected qualitative result on a short corpus:
  - "none" should be noticeably WORSE than all others (no positional info)
  - "learned", "sinusoidal", "rope", "alibi" should be fairly close for
    short sequences, with differences becoming more pronounced on longer
    sequences or out-of-distribution lengths (discussed in the study guide)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ModelConfig
from app.tokenizer import CharTokenizer
from app.data import load_corpus, prepare_data, get_batch, estimate_loss
from app.model import GPTLanguageModel
from scripts.train import make_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

POS_ENCODINGS = ["none", "learned", "sinusoidal", "rope", "alibi"]


def train_one(
    pos_encoding: str,
    tokenizer: CharTokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    d_model: int,
    n_heads: int,
    n_layers: int,
    max_seq_len: int,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len, dropout=0.1,
        pos_encoding=pos_encoding, activation="gelu",
    )
    model = GPTLanguageModel(config)

    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = make_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)

    start = time.time()
    loss = None
    for _ in range(total_steps):
        x, y = get_batch(train_data, batch_size, max_seq_len)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    final_train_loss = loss.item()
    final_val_loss = estimate_loss(model, val_data, batch_size, max_seq_len, eval_iters=10)
    elapsed = time.time() - start

    return {
        "pos_encoding": pos_encoding,
        "params": model.num_parameters,
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default="./data/sample_corpus.txt")
    args = parser.parse_args()

    text = load_corpus(args.data_path)
    tokenizer = CharTokenizer.from_text(text)
    train_data, val_data = prepare_data(text, tokenizer)

    print("\n" + "=" * 70)
    print("POSITIONAL ENCODING BENCHMARK")
    print("=" * 70)
    print(f"d_model={args.d_model} n_heads={args.n_heads} n_layers={args.n_layers} "
          f"max_seq_len={args.max_seq_len}")
    print(f"epochs={args.epochs} steps_per_epoch={args.steps_per_epoch} "
          f"(total_steps={args.epochs * args.steps_per_epoch})")
    print(f"vocab_size={tokenizer.vocab_size}")
    print("=" * 70 + "\n")

    results = []
    for pos_enc in POS_ENCODINGS:
        logger.info(f"Training with pos_encoding='{pos_enc}'...")
        result = train_one(
            pos_enc, tokenizer, train_data, val_data,
            args.d_model, args.n_heads, args.n_layers, args.max_seq_len,
            args.batch_size, args.epochs, args.steps_per_epoch, args.seed,
        )
        results.append(result)
        logger.info(f"  -> val_loss={result['final_val_loss']:.4f} ({result['time_s']:.1f}s)")

    print("\n" + "=" * 70)
    print("RESULTS (sorted by validation loss, lower = better)")
    print("=" * 70)
    print(f"{'pos_encoding':<14} {'params':>10} {'train_loss':>11} {'val_loss':>9} {'time(s)':>8}")
    for r in sorted(results, key=lambda x: x["final_val_loss"]):
        print(f"{r['pos_encoding']:<14} {r['params']:>10,} {r['final_train_loss']:>11.4f} "
              f"{r['final_val_loss']:>9.4f} {r['time_s']:>8.1f}")
    print("=" * 70)
    print("\nNote: with so few steps this is a SANITY CHECK, not a publishable")
    print("benchmark. 'none' (no positional encoding) should generally be the")
    print("worst, demonstrating why positional information is necessary.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
