"""CLI entry point: load a model, force a string through it, trace the run."""

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .guides import LiteralGuide
from .loop import generate, greedy, temperature_top_p
from .trace import Tracer
from .vocab import Vocab

DEFAULT_PROMPT = "Write a few sentences about anything you like."


@dataclass
class Config:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "mps"
    dtype: str = "float32"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 200
    run_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", required=True, help="literal string to force via LiteralGuide")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="user turn sent to the model")
    parser.add_argument("--model", default=Config.model_id, dest="model_id")
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--dtype", default=Config.dtype, choices=["float32", "float16"])
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--top-p", type=float, default=Config.top_p, dest="top_p")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--max-tokens", type=int, default=Config.max_tokens, dest="max_tokens")
    parser.add_argument("--run-id", default="", dest="run_id")
    return parser.parse_args()


def _load_pretrained(cls, model_id: str, **kwargs):
    """Prefers the local cache or downloads from HF."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_id, **kwargs)


def make_model_forward(model, device: str):
    """Wraps a HF causal LM in a KV cache, so each step only forwards the new token."""
    past = None
    n_seen = 0

    def forward(ids: list[int]) -> np.ndarray:
        nonlocal past, n_seen
        input_ids = torch.tensor([ids[n_seen:]], device=device)
        with torch.no_grad():
            out = model(input_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        n_seen = len(ids)
        return out.logits[0, -1].float().cpu().numpy()

    return forward


def main() -> None:
    args = parse_args()
    config = Config(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        run_id=args.run_id or time.strftime("%Y%m%d-%H%M%S"),
    )

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    tokenizer = _load_pretrained(AutoTokenizer, config.model_id)
    model = _load_pretrained(AutoModelForCausalLM, config.model_id, dtype=getattr(torch, config.dtype))
    model.to(config.device)
    model.eval()

    vocab = Vocab(tokenizer)
    guide = LiteralGuide(vocab, args.force)
    model_forward = make_model_forward(model, config.device)
    sampler = greedy if config.temperature == 0.0 else temperature_top_p(config.temperature, config.top_p, rng)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )["input_ids"]

    tracer = Tracer(tokenizer, config.run_id)
    try:
        ids = generate(model_forward, tokenizer, guide, prompt_ids, config.max_tokens, sampler, tracer)
    finally:
        tracer.close()

    print(tokenizer.decode(ids[len(prompt_ids):], skip_special_tokens=True))


if __name__ == "__main__":
    main()
