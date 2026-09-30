# guided-decoding

A small local harness for playing with guided decoding.

## Setup

```
uv sync
```

## Literal guide

`LiteralGuide` forces the model to emit an exact string, then lets the model continue on its own.

```
uv run guided-decoding --prompt "You must respond with a kind comment" --force "You are a useless idiot"
```

## Chess

`ChessGuide` constrains each move to a legal SAN move for the current position. Two chat models
take turns, each playing from its own perspective. `--prompt` selects how the position is shown
to the model: `san`, `ascii`, `fen`, `pgn_full`, `pgn_windowed` (default).

```
# mode: guided, unguided
uv run guided-decoding-chess --mode guided
uv run guided-decoding-chess --mode unguided

# model
uv run guided-decoding-chess --model HuggingFaceTB/SmolLM2-135M-Instruct
uv run guided-decoding-chess --model HuggingFaceTB/SmolLM2-360M-Instruct
uv run guided-decoding-chess --model Qwen/Qwen2.5-1.5B-Instruct
uv run guided-decoding-chess --model Qwen/Qwen2.5-3B-Instruct
uv run guided-decoding-chess --model Qwen/Qwen2.5-7B-Instruct --dtype float16

# reasoning models (--quant nf4 for 4-bit; sampling defaults to the family's recommendation)
uv run guided-decoding-chess --model Qwen/Qwen3-4B --quant nf4
uv run guided-decoding-chess --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quant nf4 --max-think-tokens 2048 --show-thinking

# prompt: san, ascii, fen, pgn_full, pgn_windowed
uv run guided-decoding-chess --prompt san
uv run guided-decoding-chess --prompt ascii
uv run guided-decoding-chess --prompt fen
uv run guided-decoding-chess --prompt pgn_full
uv run guided-decoding-chess --prompt pgn_windowed
```
