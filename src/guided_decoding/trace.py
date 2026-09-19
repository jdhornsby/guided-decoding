"""Writes one JSONL line per decode step, for `jq`-driven inspection."""

import json
from pathlib import Path

import numpy as np


class Tracer:
    """Records each step's raw and biased logits to traces/<run_id>.jsonl."""

    def __init__(self, tokenizer, run_id: str, path: str | None = None, out_dir: str = "traces"):
        path = Path(path) if path else Path(out_dir) / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer = tokenizer
        self._file = open(path, "w")

    def meta(self, record: dict) -> None:
        """Writes the {"kind": "meta", ...} record; must be called before any record()."""
        self._file.write(json.dumps({"kind": "meta", **record}) + "\n")
        self._file.flush()

    def record(self, step: int, raw: np.ndarray, biased: np.ndarray, token: int, guided: bool):
        allowed = int(np.isfinite(biased).sum())
        line = {
            "step": step,
            "token": token,
            "token_str": self._tokenizer.decode([token]),
            "guided": guided,
            "chosen_rank_raw": self._rank(raw, token),
            "chosen_logprob_raw": self._logprob(raw, token),
            "allowed_count": allowed,
            "top_k_raw": self._top_k(raw, k=5),
            "top_k_biased": self._top_k(biased, k=min(allowed, 10)),
            "entropy_raw": self._entropy(raw),
            "entropy_biased": self._entropy(biased),
        }
        self._file.write(json.dumps(line) + "\n")
        self._file.flush()

    def think(self, step: int, token_ids: list[int], truncated: bool) -> None:
        """Records one buffered line for a whole thinking phase, not per token."""
        line = {
            "kind": "think",
            "step": step,
            "tokens": len(token_ids),
            "truncated": truncated,
            "text": self._tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        }
        self._file.write(json.dumps(line) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def _top_k(self, logits: np.ndarray, k: int) -> list[list]:
        finite = np.flatnonzero(np.isfinite(logits))
        k = min(k, len(finite))
        top = finite[np.argpartition(logits[finite], -k)[-k:]] if k else finite[:0]
        top = top[np.argsort(logits[top])[::-1]]
        return [[self._tokenizer.decode([i]), float(logits[i])] for i in top]

    def _entropy(self, logits: np.ndarray) -> float:
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        nonzero = probs[probs > 0]
        return float(-(nonzero * np.log(nonzero)).sum())

    def _rank(self, logits: np.ndarray, token: int) -> int:
        return int((logits > logits[token]).sum())

    def _logprob(self, logits: np.ndarray, token: int) -> float:
        shifted = logits - np.max(logits)
        log_probs = shifted - np.log(np.exp(shifted).sum())
        return float(log_probs[token])
