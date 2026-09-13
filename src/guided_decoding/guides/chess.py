"""Forces one legal SAN move for a fixed position, then the stop token."""

import chess
import numpy as np

from ..vocab import Vocab


class ChessGuide:
    """Forces one legal SAN move for a fixed position, then the stop token."""

    def __init__(self, vocab: Vocab, board: chess.Board):
        self._vocab = vocab
        self._vocab_size = len(vocab.token_bytes)
        self._targets = [board.san(m).encode() for m in board.legal_moves]
        self._state = b""

    def bias(self) -> np.ndarray:
        mask = np.full(self._vocab_size, -np.inf, dtype=np.float32)
        for target in self._targets:
            if target.startswith(self._state):
                remaining = target[len(self._state):]
                if remaining:
                    mask[self._vocab.tokens_prefixing(remaining)] = 0.0
                else:
                    mask[self._vocab.eos_id] = 0.0  # move complete -> stop token
        return mask

    def advance(self, token: int) -> None:
        raw = self._vocab.token_bytes[token]
        assert raw is not None
        self._state += raw

    def finished(self) -> bool:
        return False  # the stop token ends it, via the loop's eos break

    def __repr__(self) -> str:
        return f"ChessGuide({len(self._targets)} legal moves)"
