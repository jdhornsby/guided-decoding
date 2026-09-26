"""Loading and running HF causal LMs, shared by the CLI entry points."""

import numpy as np
import torch


def load_pretrained(cls, model_id: str, **kwargs):
    """Prefers the local cache or downloads from HF."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_id, **kwargs)


def build_quantization_config(quant: str):
    """Builds a transformers quant config."""
    if quant == "none":
        return None
    if quant in ("fp4", "nf4"):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quant == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig
        from transformers import TorchAoConfig
        return TorchAoConfig(quant_type=Float8WeightOnlyConfig())
    raise ValueError(f"unknown --quant {quant!r}")


def load_model(cls, model_id: str, device: str, dtype: torch.dtype, quant: str = "none"):
    """Loads a model."""
    quant_config = build_quantization_config(quant)
    if quant_config is None:
        model = load_pretrained(cls, model_id, dtype=dtype)
        model.to(device)
    else:
        model = load_pretrained(cls, model_id, quantization_config=quant_config, device_map={"": device})
    model.eval()
    return model


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
