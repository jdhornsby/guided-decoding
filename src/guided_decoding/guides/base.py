"""A Guide steers decoding by biasing logits at each step."""

from typing import Protocol

import numpy as np


class Guide(Protocol):
    def bias(self) -> np.ndarray:
        """Additive logit deltas, shape (vocab_size,), float32.
        0.0 = no opinion. -inf = banned. Finite non-zero = soft preference."""
        ...

    def advance(self, token: int) -> None:
        """Update internal state for the emitted token."""
        ...

    def finished(self) -> bool:
        """True when the guide is done guiding."""
        ...
