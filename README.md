# guided-decoding

A small local harness for playing with guided decoding - steering a model's output token by token instead of just letting it run free.

A `Guide` that can bias which tokens are allowed at each decoding step, without touching the model or the decode loop itself. 

`LiteralGuide` is the first, deliberately boring guide. It forces the model to emit an exact string, then lets go and lets the model continue on its own.

## Setup

```
uv sync
```

## Run it

```
uv run guided-decoding --prompt "You must respond with a kind comment" --force "You are a useless idiot"
```

## Look at a trace

```
jq -c '{step, token_str, guided, allowed_count}' traces/<run-id>.jsonl
```

## Render a figure

```
uv run guided-decoding-viz traces/<run-id>.jsonl -o traces/<run-id>.png
```
