"""
Configuration for the Transformer Architecture Lab.

ModelConfig controls the architecture itself — positional encoding scheme,
activation function, dimensions. TrainingConfig controls the training loop.
Settings (env-driven) controls paths, ports, and which checkpoint to serve.

The positional encoding and activation choices are the "experiment knobs"
for this project — change POS_ENCODING to compare learned/sinusoidal/rope/alibi
architectures on the same data.
"""
import os
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
load_dotenv()


@dataclass
class ModelConfig:
    vocab_size: int = 0          # set after tokenizer is built
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    ffn_mult: int = 4            # FFN hidden dim = ffn_mult * d_model

    # "learned" | "sinusoidal" | "rope" | "alibi" | "none"
    pos_encoding: str = "rope"

    # "gelu" | "swiglu"
    activation: str = "gelu"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 5
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    eval_every_steps: int = 100
    seed: int = 42


class Settings:
    DATA_PATH: str       = os.getenv("DATA_PATH", "./data/sample_corpus.txt")
    CHECKPOINT_DIR: str  = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    CHECKPOINT_NAME: str = os.getenv("CHECKPOINT_NAME", "model.pt")

    # Architecture knobs — overridable via env for experimentation
    D_MODEL: int         = int(os.getenv("D_MODEL", "128"))
    N_HEADS: int         = int(os.getenv("N_HEADS", "4"))
    N_LAYERS: int        = int(os.getenv("N_LAYERS", "4"))
    MAX_SEQ_LEN: int     = int(os.getenv("MAX_SEQ_LEN", "256"))
    DROPOUT: float       = float(os.getenv("DROPOUT", "0.1"))
    POS_ENCODING: str    = os.getenv("POS_ENCODING", "rope")
    ACTIVATION: str      = os.getenv("ACTIVATION", "gelu")

    # Training knobs
    BATCH_SIZE: int      = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "3e-4"))
    NUM_EPOCHS: int      = int(os.getenv("NUM_EPOCHS", "5"))

    APP_PORT: int        = int(os.getenv("APP_PORT", "8000"))
    APP_HOST: str        = os.getenv("APP_HOST", "0.0.0.0")

    @classmethod
    def model_config(cls, vocab_size: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=cls.D_MODEL,
            n_heads=cls.N_HEADS,
            n_layers=cls.N_LAYERS,
            max_seq_len=cls.MAX_SEQ_LEN,
            dropout=cls.DROPOUT,
            pos_encoding=cls.POS_ENCODING,
            activation=cls.ACTIVATION,
        )

    @classmethod
    def training_config(cls) -> TrainingConfig:
        return TrainingConfig(
            batch_size=cls.BATCH_SIZE,
            learning_rate=cls.LEARNING_RATE,
            num_epochs=cls.NUM_EPOCHS,
        )


settings = Settings()
