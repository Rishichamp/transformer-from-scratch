"""
Character-level tokenizer.

For a from-scratch transformer lab, a character-level vocabulary keeps
the implementation focused on the ARCHITECTURE (attention, positional
encoding, training dynamics) rather than on tokenisation/BPE complexity.

Vocabulary is built from the training corpus and saved alongside the
model checkpoint so inference can reconstruct the exact same mapping.
"""

import json
from pathlib import Path


class CharTokenizer:
    """Simple character-level tokenizer with a fixed vocabulary."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from all unique characters in the text, sorted for determinism."""
        vocab = sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids. Unknown characters are skipped."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"])
