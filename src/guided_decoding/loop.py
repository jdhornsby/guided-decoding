"""The decode loop, plus the samplers that turn logits into a token."""

import numpy as np


class DeadEnd(Exception):
    """Raised when a guide's bias bans every token in the vocabulary."""

    def __init__(self, guide, step: int):
        super().__init__(f"{type(guide).__name__} left no allowed tokens at step {step}")


def generate(model, tokenizer, guide, prompt_ids, max_tokens, sampler, tracer):
    """Runs the guided decode loop; `model` is ids -> (vocab,) float32 logits, real or fake."""
    ids = list(prompt_ids)
    state = guide.init()
    guiding = True

    for step in range(max_tokens):
        raw = model(ids)
        logits = raw.copy()

        if guiding:
            logits = logits + guide.bias(state)
            if not np.isfinite(logits).any():
                raise DeadEnd(guide, step)

        token = sampler(logits)
        tracer.record(step, raw, logits, token, guiding)

        ids.append(token)
        if token == tokenizer.eos_token_id:
            break

        if guiding:
            state = guide.advance(state, token)
            guiding = not guide.finished(state)

    return ids


def greedy(logits: np.ndarray) -> int:
    """Always take the highest-probability token."""
    return int(np.argmax(logits))


def temperature_top_p(temperature: float, top_p: float, rng: np.random.Generator):
    """Sampler factory: softmax(logits / temperature), truncated to top_p cumulative mass."""

    def sample(logits: np.ndarray) -> int:
        scaled = logits / temperature
        order = np.argsort(scaled)[::-1]
        probs = _softmax(scaled[order])

        cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
        probs[cutoff:] = 0.0
        probs /= probs.sum()

        choice = rng.choice(len(probs), p=probs)
        return int(order[choice])

    return sample


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()
