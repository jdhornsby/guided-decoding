"""Render a slice of a guided-decoding trace as a static PNG.

Every mark is derived from the trace. All steps get identical treatment;
nothing special-cases a forced prefix. An optional leading
{"kind":"meta",...} record supplies the title and subtitle.
"""
import json, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

ACCENT      = "#C2410C"
ACCENT_MID  = "#EA8C3E"
ACCENT_PALE = "#FDEBD6"
GREY        = "#9CA3AF"
GREY_PALE   = "#E8EAED"
INK         = "#1F2328"
MUTED       = "#6B7280"

MAX_STEPS = 60
REQUIRED_STEP_FIELDS = (
    "step", "token_str", "chosen_logprob_raw", "chosen_rank_raw",
    "allowed_count", "top_k_raw", "top_k_biased",
)

RAMP = LinearSegmentedColormap.from_list("a", ["#FFFFFF", ACCENT_PALE, ACCENT_MID])


def show(tok, width=11):
    t = tok.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")
    return t if len(t) <= width else t[: width - 1] + "…"


def load(path):
    meta, steps = {}, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            (meta.update(rec) if rec.get("kind") == "meta" else steps.append(rec))
    return meta, steps


def check_steps(steps):
    for i, s in enumerate(steps):
        for field in REQUIRED_STEP_FIELDS:
            if field not in s:
                raise SystemExit(f"step at index {i} is missing required field {field!r}")


def softmax(xs):
    a = np.array(xs, dtype=float)
    a -= a.max()
    e = np.exp(a)
    return e / e.sum()


def build(steps, out, meta=None, title=None, topk=5):
    meta = meta or {}
    n = len(steps)
    lp = np.array([s["chosen_logprob_raw"] for s in steps])
    x = np.arange(n)

    shown = [[(t, v) for t, v in s["top_k_biased"] if v is not None][:topk]
             for s in steps]
    extra = [s["allowed_count"] - len(c) for s, c in zip(steps, shown)]
    rows = max(len(c) + (1 if e > 0 else 0) for c, e in zip(shown, extra))

    fw = max(9.0, 0.85 * n + 2.2)
    fig = plt.figure(figsize=(fw, 8.8), dpi=200)
    L = 1.30 / fw          # constant physical margins, not fractional,
    R = 1 - 0.22 / fw      # so labels survive narrow ranges
    gs = fig.add_gridspec(
        3, 1, height_ratios=[0.45, 1.95, 2.4], hspace=0.14,
        left=L, right=R, top=0.855, bottom=0.075,
    )
    ax_rib, ax_set, ax_cost = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- ribbon ---------------------------------------------------------
    ax_rib.set_xlim(-0.6, n - 0.4)
    ax_rib.set_ylim(0, 1)
    ax_rib.axis("off")
    for i, s in enumerate(steps):
        ax_rib.add_patch(FancyBboxPatch(
            (i - 0.42, 0.24), 0.84, 0.54,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            linewidth=1.1, edgecolor=ACCENT, facecolor=ACCENT_PALE))
        ax_rib.text(i, 0.51, show(s["token_str"], 9), ha="center",
                    va="center", fontsize=8.5, color=INK)

    # ---- allow-set ------------------------------------------------------
    CH, GAP = 1.0, 0.9
    for i, s in enumerate(steps):
        ax_set.text(i, rows + GAP + 0.5, show(s["top_k_raw"][0][0], 12),
                    ha="center", va="center", fontsize=7.5, color=GREY,
                    style="italic")
        cands = shown[i]
        probs = softmax([v for _, v in cands])
        for slot, idx in enumerate(np.argsort(-probs)):
            tok = cands[idx][0]
            y = (rows - 1 - slot) * CH
            chosen = tok == s["token_str"]
            ax_set.add_patch(Rectangle(
                (i - 0.41, y + 0.06), 0.82, CH - 0.12,
                facecolor=RAMP(0.15 + 0.85 * probs[idx]),
                edgecolor=ACCENT if chosen else GREY_PALE,
                linewidth=1.9 if chosen else 0.9, zorder=3))
            ax_set.text(i, y + CH / 2, show(tok, 10), ha="center", va="center",
                        fontsize=7.4, zorder=4, color=INK if chosen else MUTED,
                        weight="bold" if chosen else "normal")
        if extra[i] > 0:
            y = (rows - 1 - len(cands)) * CH
            ax_set.text(i, y + CH / 2, f"+{extra[i]:,}", ha="center",
                        va="center", fontsize=6.8, color=GREY, style="italic")

    ax_set.axhline(rows + GAP * 0.35, color=GREY_PALE, lw=0.9, ls=(0, (4, 3)))
    ax_set.text(-0.55, rows + GAP + 0.5, "unconstrained top-1", ha="right",
                va="center", fontsize=7.5, color=GREY, style="italic")
    ax_set.set_xlim(-0.6, n - 0.4)
    ax_set.set_ylim(-0.15, rows + GAP + 1.15)
    ax_set.set_ylabel(f"top {topk} tokens the guide allowed\n"
                      "(shaded by relative probability)",
                      fontsize=8.5, color=MUTED)
    ax_set.set_xticks([]); ax_set.set_yticks([])
    for sp in ax_set.spines.values():
        sp.set_visible(False)

    # ---- log-probability -------------------------------------------------
    lo = lp.min() * 1.16
    ax_cost.fill_between(x, 0, lp, color=ACCENT_PALE, zorder=2)
    ax_cost.plot(x, lp, color=ACCENT, lw=1.7, zorder=4)
    ax_cost.scatter(x, lp, s=24, color=ACCENT, zorder=5)
    ax_cost.axhline(0, color=GREY_PALE, lw=1.0, zorder=3)

    mean = lp.mean()
    ax_cost.axhline(mean, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax_cost.text(n - 0.5, mean, f"mean {mean:.1f} ", va="bottom", ha="right",
                 fontsize=7, color=MUTED)

    pk = int(np.argmin(lp))
    p = math.exp(lp[pk])
    odds = f"1 in {float(f'{1/p:.2g}'):,.0f}" if p < 1e-3 else f"{p*100:.1f}%"
    side = 0.32 if pk < n * 0.75 else -0.32
    ax_cost.annotate(
        f"{lp[pk]:.2f} nats · rank {steps[pk]['chosen_rank_raw']:,} · {odds}",
        xy=(pk, lp[pk]), xytext=(pk + side, lp[pk]), fontsize=8.5,
        color=ACCENT, va="center", ha="left" if side > 0 else "right")

    ticks = [i for i in range(n) if i % max(1, n // 12) == 0]
    ax_cost.set_xticks(ticks)
    ax_cost.set_xticklabels([str(steps[i]["step"]) for i in ticks], fontsize=7)
    ax_cost.set_xlim(-0.6, n - 0.4)
    ax_cost.set_ylim(lo, -lo * 0.06)
    ax_cost.set_xlabel("step", fontsize=8.5, color=MUTED)
    ax_cost.set_ylabel("log-probability of emitted token\n"
                       "under the unconstrained model (nats)",
                       fontsize=8.5, color=MUTED)
    ax_cost.tick_params(labelsize=7.5, colors=MUTED)
    ax_cost.grid(axis="y", color=GREY_PALE, lw=0.8)
    ax_cost.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax_cost.spines[sp].set_visible(False)
    ax_cost.spines["left"].set_color(GREY_PALE)

    ax_cost.text(-0.4, lo * 0.965,
                 f"{n} steps · {lp.sum():.0f} nats total",
                 ha="left", va="bottom", fontsize=8, color=MUTED)

    span = f"steps {steps[0]['step']}–{steps[-1]['step']}"
    head = title or meta.get("guide") or "guided decoding trace"
    sub = " · ".join([str(meta[k]) for k in ("model", "prompt", "seed")
                           if k in meta] + [span])
    fig.text(L, 0.945, head, fontsize=13, color=INK, ha="left")
    fig.text(L, 0.905, sub, fontsize=9, color=MUTED, ha="left")

    fig.savefig(out, facecolor="white", metadata={"Software": None})
    print(f"wrote {out}  ({span}, {n} steps)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "--start/--end are positions in the trace file (Python slice\n"
            "semantics, [start:end), end exclusive), not `step` values.\n"
            "They coincide unless the trace is concatenated or filtered."
        ),
    )
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", default="trace.png")
    ap.add_argument("--start", type=int, default=0, help="file position to start at (default 0)")
    ap.add_argument("--end", type=int, default=None, help="file position to end before (default: end of trace)")
    ap.add_argument("--topk", type=int, default=5, help="allow-set cells to show per step (default 5)")
    ap.add_argument("-t", "--title", default=None, help="override the derived heading")
    a = ap.parse_args()

    m, s = load(a.trace)
    end = a.end if a.end is not None else len(s)
    sl = s[a.start:end]
    if not sl:
        raise SystemExit(f"empty range {a.start}:{end} (trace has {len(s)} steps)")
    if len(sl) > MAX_STEPS:
        raise SystemExit(
            f"range {a.start}:{end} has {len(sl)} steps, exceeding {MAX_STEPS}; "
            "narrow it with --start/--end"
        )
    check_steps(sl)
    build(sl, a.out, m, a.title, a.topk)


if __name__ == "__main__":
    main()
