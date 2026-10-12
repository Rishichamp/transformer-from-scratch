"""
Training script for the Transformer Architecture Lab.

Usage:
    python scripts/train.py

Implements the full training loop from Module 5's notes:
  - AdamW with decoupled weight decay (no decay on bias/LayerNorm params)
  - Linear warmup -> cosine decay learning rate schedule
  - Gradient clipping
  - Periodic train/val loss estimation

Saves a checkpoint containing model weights, config, and tokenizer vocab
to CHECKPOINT_DIR, which app/main.py loads for serving.
"""

import logging
import sys
import time
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.tokenizer import CharTokenizer
from app.data import load_corpus, prepare_data, get_batch, estimate_loss
from app.model import GPTLanguageModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for warmup_steps, then cosine decay to 0 over the remaining steps."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    torch.manual_seed(settings.training_config().seed)

    print("\n" + "=" * 60)
    print("TRANSFORMER ARCHITECTURE LAB -- TRAINING")
    print("=" * 60)
    print(f"Positional encoding : {settings.POS_ENCODING}")
    print(f"Activation          : {settings.ACTIVATION}")
    print(f"d_model / heads / layers : {settings.D_MODEL} / {settings.N_HEADS} / {settings.N_LAYERS}")
    print(f"max_seq_len         : {settings.MAX_SEQ_LEN}")
    print(f"batch_size          : {settings.BATCH_SIZE}")
    print(f"learning_rate       : {settings.LEARNING_RATE}")
    print(f"epochs              : {settings.NUM_EPOCHS}")
    print("=" * 60 + "\n")

    # Data
    text = load_corpus(settings.DATA_PATH)
    tokenizer = CharTokenizer.from_text(text)
    train_data, val_data = prepare_data(text, tokenizer)
    logger.info(
        f"Corpus: {len(text):,} chars | vocab_size={tokenizer.vocab_size} | "
        f"train={len(train_data):,} tokens | val={len(val_data):,} tokens"
    )

    # Model
    model_config = settings.model_config(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(model_config)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Device: {device}")

    # Optimiser + schedule
    train_config = settings.training_config()

    # AdamW: weight decay on matrices (dim>=2) only, not on biases/LayerNorm (dim<2)
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_config.learning_rate, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_data) // (train_config.batch_size * model_config.max_seq_len))
    total_steps = steps_per_epoch * train_config.num_epochs
    warmup_steps = max(1, int(total_steps * train_config.warmup_ratio))
    scheduler = make_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Total steps: {total_steps} ({steps_per_epoch}/epoch), warmup_steps={warmup_steps}")

    # Training loop
    step = 0
    start_time = time.time()
    for epoch in range(train_config.num_epochs):
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, train_config.batch_size, model_config.max_seq_len, device)

            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % train_config.eval_every_steps == 0 or step == total_steps - 1:
                val_loss = estimate_loss(model, val_data, train_config.batch_size, model_config.max_seq_len, device=device)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                logger.info(
                    f"step {step:5d}/{total_steps} | train_loss {loss.item():.4f} | "
                    f"val_loss {val_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                )

            step += 1

    # Save checkpoint
    ckpt_dir = Path(settings.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config.to_dict(),
    }, ckpt_dir / settings.CHECKPOINT_NAME)

    tokenizer.save(str(ckpt_dir / "tokenizer.json"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoint saved to: {ckpt_dir / settings.CHECKPOINT_NAME}")
    print(f"Tokenizer saved to:  {ckpt_dir / 'tokenizer.json'}")
    print(f"\nNext steps:")
    print(f"  python run.py")
    print(f'  curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \\')
    print(f'       -d \'{{"prompt":"Attention is","max_new_tokens":200,"temperature":0.8}}\'')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
