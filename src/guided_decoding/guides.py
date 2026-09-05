"""The seam: a Guide steers decoding by biasing logits at each step."""

from typing import Protocol, TypeVar

import numpy as np

from .vocab import Vocab

S = TypeVar("S")


class Guide(Protocol[S]):
    def init(self) -> S:
        """Initial state, called once before decoding begins."""
        ...

    def bias(self, state: S) -> np.ndarray:
        """Additive logit deltas, shape (vocab_size,), float32.
        0.0 = no opinion. -inf = banned. Finite non-zero = soft preference."""
        ...

    def advance(self, state: S, token: int) -> S:
        """State after the given token is emitted. Must not mutate `state`."""
        ...

    def finished(self, state: S) -> bool:
        """True when the guide has nothing further to say."""
        ...


class LiteralGuide:
    """Forces an exact byte string, then releases the model."""

    def __init__(self, vocab: Vocab, target: str):
        self._vocab = vocab
        self._target = target.encode()

        vocab_size = len(vocab.token_bytes)
        self._masks = np.full((len(self._target) + 1, vocab_size), -np.inf, dtype=np.float32)
        for n in range(len(self._target) + 1):
            remaining = self._target[n:]
            allowed = vocab.tokens_prefixing(remaining)
            self._masks[n, allowed] = 0.0

    def init(self) -> int:
        return 0

    def bias(self, state: int) -> np.ndarray:
        return self._masks[state]

    def advance(self, state: int, token: int) -> int:
        raw = self._vocab.token_bytes[token]
        assert raw is not None
        return state + len(raw)

    def finished(self, state: int) -> bool:
        return state == len(self._target)
