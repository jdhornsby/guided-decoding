# guided-decoding

A small local harness for playing with guided decoding - steering a model's output token by token instead of just letting it run free.

A `Guide` that can bias which tokens are allowed at each decoding step, without touching the model or the decode loop itself. 

`LiteralGuide` is the first, deliberately boring guide. It forces the model to emit an exact string, then lets go and lets the model continue on its own.

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Run it

```
uv run guided-decoding --force "fix(auth): "
```

First run downloads SmolLM2-135M-Instruct. The model is forced to start its
reply with `fix(auth): `, then continues unguided. Output prints to stdout;
every step also gets written to `traces/<run-id>.jsonl`.

Flags worth knowing about:

- `--model`: swap models, e.g. `--model Qwen2.5-0.5B-Instruct`
- `--max-tokens`: how long to let it run
- `--temperature` / `--top-p`: sampling; temperature `0` (default) is greedy
- `--seed`: same seed and same everything else gives identical output

## Look at a trace

```
jq -c '{step, token_str, guided, allowed_count}' traces/<run-id>.jsonl
```

`allowed_count` is how many tokens the guide allowed at that step: small
while it's forcing the string, full vocab once it lets go.

## Render a figure

```
uv run guided-decoding-viz traces/<run-id>.jsonl -o out.png
```

Renders a static PNG: a token ribbon, the per-step allow-set (shaded by
relative probability), and the emitted token's log-probability under the
unconstrained model. Only reads the trace file — never touches the model.

- `--start N` / `--end N`: slice the trace by file position (not `step`
  value), Python slice semantics, before rendering. Useful since a full run
  can exceed the 60-step render limit.
- `--topk K`: how many allow-set cells to show per step (default 5, capped
  at what was logged).
- `--title TEXT`: override the derived heading.

Known limitations:

- Allow-set shading is softmax over the **shown** top-k, not the full
  allow-set, so it overstates probability when `+N` is large.
- The grey italic row above the allow-set is only the unconstrained top-1;
  if the emitted token isn't in the logged top-k, its rank still appears in
  the peak label but it can't be placed among the alternatives.
- Wide allow-sets collapse to a `+N` count — legible only through that count
  and the cost curve, not individually.
