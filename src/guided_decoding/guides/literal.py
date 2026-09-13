"""Forces an exact byte string, then releases the model."""

import numpy as np

from ..vocab import Vocab


class LiteralGuide:
    """Forces an exact byte string, then releases the model."""

    def __init__(self, vocab: Vocab, target: str):
        self._vocab = vocab
        self._target = target.encode()
        self._pos = 0

        vocab_size = len(vocab.token_bytes)
        self._masks = np.full((len(self._target) + 1, vocab_size), -np.inf, dtype=np.float32)
        for n in range(len(self._target) + 1):
            remaining = self._target[n:]
            allowed = vocab.tokens_prefixing(remaining)
            self._masks[n, allowed] = 0.0

    def bias(self) -> np.ndarray:
        return self._masks[self._pos]

    def advance(self, token: int) -> None:
        raw = self._vocab.token_bytes[token]
        assert raw is not None
        self._pos += len(raw)

    def finished(self) -> bool:
        return self._pos == len(self._target)

    def __repr__(self) -> str:
        return f"LiteralGuide({self._target.decode()!r})"
