# Transformer From Scratch

[![tests](https://github.com/Rishichamp/transformer-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/transformer-from-scratch/actions/workflows/tests.yml)

A GPT-style, decoder-only transformer built from first principles in PyTorch: no `nn.MultiheadAttention`, no `nn.TransformerEncoder`. It's a lab for comparing architecture choices, especially **positional encodings**, with a training loop and a small API for generation and attention inspection.

## Features

- **Hand-written attention:** multi-head causal self-attention with an explicit causal mask, plus weight tying between the embedding and output layers.
- **Pluggable positional encodings:** `learned`, `sinusoidal`, `rope` (rotary) and `alibi` (linear attention biases), switched with one setting.
- **Pluggable feed-forward:** GELU or SwiGLU.
- **Training script:** AdamW with decoupled weight decay (none on biases or LayerNorm), linear warmup then cosine decay, gradient clipping, periodic train/val loss.
- **Benchmark script:** trains identical models that differ only in positional encoding and compares their loss curves.
- **Sampling:** temperature, top-k and top-p (nucleus), which can be combined.
- **Character-level tokenizer**, so the focus stays on the architecture rather than BPE.

## Tech stack

Python 3.11 · PyTorch · FastAPI · pytest

## Project structure

```
app/
  model.py                 GPTLanguageModel, attention, transformer blocks, feed-forward variants
  positional_encoding.py   learned, sinusoidal, RoPE and ALiBi
  generation.py            temperature / top-k / top-p sampling
  tokenizer.py             character-level tokenizer
  data.py                  batching for language-model training
  config.py                ModelConfig, TrainingConfig, Settings
  main.py                  FastAPI server
scripts/
  train.py                             train and save a checkpoint
  benchmark_positional_encodings.py    compare encoding schemes
data/sample_corpus.txt     small training corpus
tests/                     19 tests (32 cases with parametrisation)
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # architecture knobs: POS_ENCODING, ACTIVATION, D_MODEL, N_HEADS, N_LAYERS
python scripts/train.py        # trains on data/sample_corpus.txt and writes checkpoints/
python run.py                  # http://127.0.0.1:8000/docs
```

Without a checkpoint the server uses an untrained model: `/generate` returns random characters (and says `"trained": false`), but `/architecture` and `/attention` work straight away.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/generate` | Generate a continuation with configurable sampling |
| GET | `/architecture` | Model configuration and parameter count |
| POST | `/attention` | Raw attention weights for an input (used by the interpretability toolkit) |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/ -q
```

Covers tensor shapes, attention rows summing to 1, the causal mask, weight tying, gradient flow, each positional encoding's mathematical properties (RoPE preserves norms and depends only on relative position; ALiBi slopes decay across heads) and generation behaviour.

## Related

[attention-interpretability-toolkit](https://github.com/Rishichamp/attention-interpretability-toolkit) analyses this model's attention patterns.

Built as part of Module 5 of a Generative AI engineering programme.

## Author

**Rishi Singh** — [GitHub](https://github.com/Rishichamp)
