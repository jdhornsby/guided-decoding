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

`ChessGuide` constrains each move to a legal SAN move for the current position. Two chat models take turns, each playing from its own perspective.


```
# unguided
uv run guided-decoding-chess --model HuggingFaceTB/SmolLM2-135M-Instruct --mode unguided

# guided
uv run guided-decoding-chess --model HuggingFaceTB/SmolLM2-135M-Instruct --mode guided
uv run guided-decoding-chess --model HuggingFaceTB/SmolLM2-360M-Instruct --mode guided

# biggest vs smallest, guided
uv run guided-decoding-chess --white Qwen/Qwen2.5-1.5B-Instruct --black HuggingFaceTB/SmolLM2-135M-Instruct --mode guided
```

## Render a trace

```
uv run guided-decoding-viz traces/<run-id>.jsonl -o traces/<run-id>.png
```
