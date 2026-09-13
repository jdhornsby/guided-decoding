"""The decode loop, plus the sampling that turns logits into a token."""

import numpy as np


class DeadEnd(Exception):
    """Raised when a guide's bias bans every token in the vocabulary."""

    def __init__(self, guide, step: int):
        super().__init__(f"{type(guide).__name__} left no allowed tokens at step {step}")


def generate(model, tokenizer, guide, prompt_ids, max_tokens, temperature, top_p, rng, tracer,
             think_end=None, max_think=None):
    """Runs the decode loop; `model` is ids -> (vocab,) float32 logits, real or fake.
    `guide` may be None to decode without one. Sampling is temperature/top_p; temperature
    0.0 takes the top token. While the model is thinking (before it emits `think_end`) the
    guide is disengaged and those steps are untraced; if thinking runs past `max_think`
    tokens it is cut off by forcing the `think_end` token."""
    ids = list(prompt_ids)
    guiding = guide is not None
    thinking = think_end is not None
    thought = 0

    for step in range(max_tokens):
        raw = model(ids)
        logits = raw.copy()

        delta = guide.bias() if (guiding and not thinking) else None
        if delta is not None:
            logits = logits + delta
            if not np.isfinite(logits).any():
                raise DeadEnd(guide, step)

        token = _sample(logits, temperature, top_p, rng)
        if not thinking:  # skip thinking steps
            tracer.record(step, raw, logits, token, guiding)

        ids.append(token)
        if token == tokenizer.eos_token_id:
            break

        if thinking:
            thought += 1
            if token == think_end:
                thinking = False
            elif max_think is not None and thought > max_think:
                ids.append(think_end)  # out of budget: force the model to stop thinking
                thinking = False
        elif guiding:
            guide.advance(token)
            guiding = not guide.finished()

    return ids


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    """temperature 0.0 takes the top token; otherwise softmax(logits / temperature) truncated to top_p."""
    if temperature == 0.0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    order = np.argsort(scaled)[::-1]
    probs = _softmax(scaled[order])

    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    probs[cutoff:] = 0.0
    probs /= probs.sum()

    return int(order[rng.choice(len(probs), p=probs)])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()
