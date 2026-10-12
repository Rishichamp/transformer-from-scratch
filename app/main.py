"""
FastAPI server for the Transformer Architecture Lab.

Endpoints:
  POST /generate     -- generate text continuation with configurable sampling
  GET  /architecture -- returns model config and parameter count
  POST /attention    -- returns raw attention weights for input text
                         (consumed by Project 2's interpretability toolkit)
  GET  /health       -- health check

If no checkpoint is found, a freshly-initialised (untrained) model is served
with a small built-in vocabulary -- /generate will produce random text, but
/architecture and /attention work immediately for architecture inspection.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings, ModelConfig
from app.tokenizer import CharTokenizer
from app.model import GPTLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state = {"model": None, "tokenizer": None, "trained": False}


def _fallback_tokenizer() -> CharTokenizer:
    """Small built-in vocabulary for serving an untrained model when no checkpoint exists."""
    vocab = sorted(set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"\n-()"
    ))
    return CharTokenizer(vocab)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path = Path(settings.CHECKPOINT_DIR) / settings.CHECKPOINT_NAME
    tok_path  = Path(settings.CHECKPOINT_DIR) / "tokenizer.json"

    if ckpt_path.exists() and tok_path.exists():
        logger.info(f"Loading trained checkpoint from {ckpt_path}")
        tokenizer = CharTokenizer.load(str(tok_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ModelConfig(**checkpoint["config"])
        model = GPTLanguageModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _state["trained"] = True
    else:
        logger.warning(
            "No trained checkpoint found. Serving an UNTRAINED model with a "
            "fallback vocabulary. /generate will produce random text. "
            "Run 'python scripts/train.py' to train a model."
        )
        tokenizer = _fallback_tokenizer()
        config = settings.model_config(vocab_size=tokenizer.vocab_size)
        model = GPTLanguageModel(config)
        model.eval()
        _state["trained"] = False

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    yield
    _state["model"] = None
    _state["tokenizer"] = None


app = FastAPI(
    title="Transformer Architecture Lab",
    description=(
        "From-scratch decoder-only transformer with pluggable positional "
        "encoding (learned/sinusoidal/RoPE/ALiBi) and feed-forward variants "
        "(GELU/SwiGLU). Serves text generation and attention introspection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field("", description="Text prompt to continue (can be empty)")
    max_new_tokens: int = Field(100, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    full_text: str
    trained: bool


class ArchitectureResponse(BaseModel):
    config: dict
    num_parameters: int
    trained: bool
    vocab_size: int


class AttentionRequest(BaseModel):
    text: str = Field(..., description="Text to analyse (will be truncated to max_seq_len)")


class AttentionResponse(BaseModel):
    text: str
    tokens: list[str]
    n_layers: int
    n_heads: int
    attention_weights: list[list[list[list[float]]]]  # [layer][head][query_pos][key_pos]


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    cfg = _state["model"].config if _state["model"] else None
    return f"""<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🔬 Transformer Architecture Lab</h1>
    <p>Positional encoding: <b>{cfg.pos_encoding if cfg else 'N/A'}</b> |
       Activation: <b>{cfg.activation if cfg else 'N/A'}</b></p>
    <p>Trained: <b>{_state['trained']}</b></p>
    <p><a href="/docs">📖 API Docs</a></p>
    <h3>Endpoints</h3>
    <ul>
      <li>POST /generate - text generation with temperature/top-k/top-p</li>
      <li>GET /architecture - model config and parameter count</li>
      <li>POST /attention - raw attention weights for interpretability</li>
    </ul>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] is not None else "no_model",
        "trained": _state["trained"],
    }


@app.get("/architecture", response_model=ArchitectureResponse)
async def architecture():
    model = _state["model"]
    return ArchitectureResponse(
        config=model.config.to_dict(),
        num_parameters=model.num_parameters,
        trained=_state["trained"],
        vocab_size=_state["tokenizer"].vocab_size,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model = _state["model"]
    tokenizer = _state["tokenizer"]

    if req.prompt:
        ids = tokenizer.encode(req.prompt)
        if not ids:
            raise HTTPException(400, "Prompt contains no characters in the model's vocabulary.")
    else:
        # Start from a single space (or first vocab token) if no prompt given
        ids = tokenizer.encode(" ") or [0]

    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    full_text = tokenizer.decode(out[0].tolist())
    generated_only = tokenizer.decode(out[0, len(ids):].tolist())

    return GenerateResponse(
        prompt=req.prompt,
        generated=generated_only,
        full_text=full_text,
        trained=_state["trained"],
    )


@app.post("/attention", response_model=AttentionResponse)
async def attention(req: AttentionRequest):
    model = _state["model"]
    tokenizer = _state["tokenizer"]

    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")

    ids = tokenizer.encode(req.text)[:model.config.max_seq_len]
    if len(ids) < 2:
        raise HTTPException(400, "Text must contain at least 2 known characters.")

    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, _, attentions = model(idx, return_attention=True)

    # attentions: list of (1, n_heads, T, T) -> nested lists [layer][head][q][k]
    weights = [layer_attn[0].tolist() for layer_attn in attentions]

    return AttentionResponse(
        text=req.text,
        tokens=[tokenizer.itos[i] for i in ids],
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        attention_weights=weights,
    )
