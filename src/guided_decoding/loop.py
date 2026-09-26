"""The decode loop, plus the sampling that turns logits into a token."""

from dataclasses import dataclass, field

import numpy as np


class DeadEnd(Exception):
    """Raised when a guide's bias bans every token in the vocabulary."""

    def __init__(self, guide, step: int):
        super().__init__(f"{type(guide).__name__} left no allowed tokens at step {step}")


@dataclass
class ThinkConfig:
    """Controls the thinking phase, if the model has one."""
    end_token: int
    max_tokens: int | None = None
    timeout_nudge_ids: list[int] = field(default_factory=list)  # said before forcing </think> on timeout
    resume_ids: list[int] = field(default_factory=list)  # said right after </think>, natural or forced


def generate(model, tokenizer, guide, prompt_ids, max_tokens, temperature, top_p, rng, tracer,
             top_k=0, think: ThinkConfig | None = None):
    """Runs the decode loop."""
    ids = list(prompt_ids)
    guiding = guide is not None
    thinking = think is not None
    thought = 0
    think_ids: list[int] = []

    for step in range(max_tokens):
        raw = model(ids)
        logits = raw.copy()

        delta = guide.bias() if (guiding and not thinking) else None
        if delta is not None:
            logits = logits + delta
            if not np.isfinite(logits).any():
                raise DeadEnd(guide, step)

        token = _sample(logits, temperature, top_p, rng, top_k)
        if not thinking:
            tracer.record(step, raw, logits, token, guiding)

        ids.append(token)
        if token == tokenizer.eos_token_id:
            break

        if thinking:
            thought += 1
            if token == think.end_token:
                _end_thinking(ids, think_ids, think, tracer, truncated=False)
                thinking = False
            else:
                think_ids.append(token)
                if think.max_tokens is not None and thought > think.max_tokens:
                    _end_thinking(ids, think_ids, think, tracer, truncated=True)
                    thinking = False
        elif guiding:
            guide.advance(token)
            guiding = not guide.finished()

    return ids


def _end_thinking(ids: list[int], think_ids: list[int], think: ThinkConfig, tracer, truncated: bool) -> None:
    if truncated:
        ids.extend(think.timeout_nudge_ids)
        think_ids.extend(think.timeout_nudge_ids)
        ids.append(think.end_token)
    ids.extend(think.resume_ids)
    think_ids.extend(think.resume_ids)
    tracer.think(think_ids, truncated=truncated)


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator,
            top_k: int = 0) -> int:
    """temperature 0.0 takes the top token; otherwise softmax(logits / temperature), first
    truncated to top_k candidates (0 disables), then to top_p."""
    if temperature == 0.0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    order = np.argsort(scaled)[::-1]
    if top_k > 0:
        order = order[:top_k]
    probs = _softmax(scaled[order])

    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    probs[cutoff:] = 0.0
    probs /= probs.sum()

    return int(order[rng.choice(len(probs), p=probs)])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()
