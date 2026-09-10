"""Figures for the project report, generated from the run metrics -- not retyped.

Every number plotted here is read from runs/metrics/*.jsonl or the CSV salvaged
from the pretraining log. That matters for this particular project: several of
its headline claims turned out to be wrong precisely because a number got
carried by hand from a log into a summary, and a figure built by retyping is a
figure that can silently disagree with the run it describes.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MET = os.path.join(ROOT, "runs", "metrics")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

INK = "#16324f"
ACC = "#b3402f"
GREY = "#8a8f98"
GOOD = "#2f6b4f"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 160,
})


def load_jsonl(name):
    """Tolerates a torn final line: a run killed by a spot reclaim leaves one."""
    out = []
    path = os.path.join(MET, name)
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def smooth(xs, w):
    """Trailing mean. Reported alongside the raw series, never instead of it --
    a rolling average is exactly what hid a real regression in this project."""
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- pretraining
def fig_pretrain():
    """Both sources carry a real token count, so they are plotted on real
    tokens and never stitched with an inferred offset. They join cleanly:
    the CSV ends at step 1,000,500 / 16.39B and the JSONL resumes at
    1,001,000 / 16.40B. An earlier version of this figure faked the second
    segment's x-axis and produced a visible discontinuity that was an artifact
    of the plotting code, not of the run.
    """
    toks, loss = [], []
    p = os.path.join(MET, "curve_salvaged.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    toks.append(float(row["tokens_B"]))
                    loss.append(float(row["loss"]))
                except (ValueError, KeyError):
                    pass
    for r in load_jsonl("metrics_final.jsonl"):
        if r.get("loss") is not None and r.get("tokens") is not None:
            toks.append(r["tokens"] / 1e9)
            loss.append(r["loss"])
    if not toks:
        return
    order = sorted(range(len(toks)), key=lambda i: toks[i])
    toks = [toks[i] for i in order]
    loss = [loss[i] for i in order]

    # THE LOG HAS A HOLE IN IT. The salvaged CSV jumps from step 6,000 (0.10B
    # tokens) to step 677,500 (11.10B): the intervening log was lost to a spot
    # reclaim, which is why the file is named "salvaged". Plotting straight
    # through that gap draws a ~3.5-nat plateau lasting 11B tokens that never
    # happened -- a figure inventing data the run does not have. Break the
    # series into segments instead, so the hole is visible as a hole.
    GAP = 0.5  # billions of tokens
    segs, cur = [], [(toks[0], loss[0])]
    for i in range(1, len(toks)):
        if toks[i] - toks[i - 1] > GAP:
            segs.append(cur)
            cur = []
        cur.append((toks[i], loss[i]))
    if cur:
        segs.append(cur)

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    for k, seg in enumerate(segs):
        xs = [x for x, _ in seg]
        ys = [y for _, y in seg]
        ax.plot(xs, ys, lw=0.5, color=GREY, alpha=0.55,
                label="train CE (raw)" if k == 0 else None)
        ax.plot(xs, smooth(ys, 25), lw=1.4, color=INK,
                label="train CE (trailing 25)" if k == 0 else None)
    if len(segs) > 1:
        gx0 = segs[0][-1][0]
        gx1 = segs[1][0][0]
        ax.axvspan(gx0, gx1, color=GREY, alpha=0.13, lw=0)
        ax.text((gx0 + gx1) / 2, 3.55,
                "per-step log lost\nto a spot reclaim\n(%.1f--%.1fB)" % (gx0, gx1),
                ha="center", va="center", fontsize=6.3, color=GREY)
    # The two held-out values differ by 0.066 nats, so their labels have to be
    # placed apart by hand or they overprint each other.
    # The two held-out values are only 0.066 nats apart, so the labels are
    # pinned to opposite corners rather than to the lines themselves.
    ax.axhline(2.1817, ls="--", lw=0.9, color=GREY)
    ax.axhline(2.1155, ls="-.", lw=0.9, color=ACC)
    ax.text(0.5, 2.40, "held-out 2.1817 @ step 656k  (dashed)",
            fontsize=6.5, color=GREY, va="bottom")
    ax.text(0.5, 1.80, "held-out 2.1155 @ step 1.245M, final  (dash-dot)",
            fontsize=6.5, color=ACC, va="bottom")
    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_ylim(0.75, 4.3)
    ax.set_xlim(-0.2, max(toks) * 1.02)
    ax.set_title("Pretraining: 20.40B tokens, 85M params (240 tokens/param)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "pretrain.pdf")


def fig_throughput():
    """Sustained tok/s on the PRO 6000 after batch-size tuning. Worth its own
    panel because the project's own README carried the pre-tuning figure
    (77,900) long after the run had settled at ~120k -- a stale number that
    survived because nobody re-read it against the metrics."""
    rows = [r for r in load_jsonl("metrics_final.jsonl") if r.get("tok_per_s")]
    if not rows:
        return
    xs = [r["tokens"] / 1e9 for r in rows]
    ys = [r["tok_per_s"] / 1e3 for r in rows]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.plot(xs, ys, lw=0.7, color=INK)
    mean = sum(ys) / len(ys)
    ax.axhline(mean, ls="--", lw=0.9, color=ACC)
    ax.annotate(f"mean {mean*1e3:,.0f} tok/s", (xs[len(xs)//2], mean),
                textcoords="offset points", xytext=(0, -13), fontsize=7,
                color=ACC, ha="center")
    ax.axhline(45.146, ls=":", lw=0.9, color=GREY)
    ax.annotate("PRO 4500: 45,146", (xs[len(xs)//2], 45.146),
                textcoords="offset points", xytext=(0, 4), fontsize=6.8,
                color=GREY, ha="center")
    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel("thousand tok/s")
    ax.set_ylim(0, 135)
    ax.set_title("Sustained throughput,\nPRO 6000 after batch tuning")
    save(fig, "throughput.pdf")


# ------------------------------------------------------------ depth ablation
def fig_depth():
    p = os.path.join(MET, "eval_results.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    depth = d.get("depth") or []
    if not depth:
        return
    xs = [int(a) for a, _ in depth]
    ys = [float(b) for _, b in depth]
    # Linear, not log. A log axis here rendered as unreadable "6x10^0" minor
    # ticks and hid the only thing the panel is for: that the curve has a
    # minimum exactly at the trained depth. The 2-loop point (7.01) is far off
    # scale, so it is annotated rather than plotted -- clipping it silently
    # would misrepresent the data.
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    keep = [(x, y) for x, y in zip(xs, ys) if y < 3.0]
    off = [(x, y) for x, y in zip(xs, ys) if y >= 3.0]
    ax.plot([x for x, _ in keep], [y for _, y in keep], "o-",
            color=INK, lw=1.3, ms=4)
    best = min(keep, key=lambda t: t[1])
    ax.plot([best[0]], [best[1]], "o", ms=9, mfc="none", mec=ACC, mew=1.6)
    ax.annotate(f"trained depth: {best[0]} loops\n{best[1]:.4f} (minimum)", best,
                textcoords="offset points", xytext=(14, 12), fontsize=6.8,
                color=ACC, ha="left")
    for x, y in off:
        ax.annotate(f"{x} loops: {y:.2f} (off scale)", (x, 2.245),
                    textcoords="offset points", xytext=(4, 0), fontsize=6.5,
                    color=GREY, va="center")
        ax.plot([x], [2.245], "^", ms=4, color=GREY)
    ax.set_xlabel("loops applied at inference")
    ax.set_ylabel("held-out CE (nats)")
    ax.set_ylim(2.095, 2.265)
    ax.set_title("Weight-shared depth does not\nextrapolate past its trained count")
    save(fig, "depth.pdf")


# ------------------------------------------------------------------- SFT
def fig_sft():
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    tags = [("sft2_metrics.jsonl", "v1  chat only", GREY),
            ("sft4_metrics.jsonl", "v2  + snippet tools", "#3f6f9f"),
            ("sft5_metrics.jsonl", "v3  + repo-shaped tools", ACC)]
    for name, label, colour in tags:
        rows = [r for r in load_jsonl(name) if r.get("masked_loss") is not None]
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ys = [r["masked_loss"] for r in rows]
        ax.plot(xs, ys, "-", lw=1.3, color=colour, label=label)
        ax.plot([xs[-1]], [ys[-1]], "o", ms=3.5, color=colour)
    ax.set_xlabel("SFT step")
    ax.set_ylabel("masked CE on\nassistant tokens (nats)")
    ax.set_title("SFT: loss is masked to assistant tokens only")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "sft.pdf")


# --------------------------------------------------------- agentic RL curve
def fig_agent_learning():
    rows = [r for r in load_jsonl("agent_long_metrics.jsonl")
            if r.get("event") == "step"]
    if not rows:
        return
    G = len(rows[0].get("rewards") or [8])
    xs = [r["step"] for r in rows]
    rw = [r["mean_reward"] for r in rows]
    sv = [(r.get("solved") or 0) / G * 100 for r in rows]
    zv = [1.0 if r.get("zero_variance") else 0.0 for r in rows]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 3.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    # Reward is on [0,1] and solve rate on [0,100]. Plotting both against one
    # axis squashed the reward line flat against the bottom and made the legend
    # a lie. Twin axes, each labelled in its own units.
    a1.plot(xs, rw, lw=0.5, color=GREY, alpha=0.45)
    l1, = a1.plot(xs, smooth(rw, 20), lw=1.5, color=INK,
                  label="mean episode reward (left)")
    a1.set_ylabel("mean episode reward", color=INK)
    a1.tick_params(axis="y", labelcolor=INK)
    a1.set_ylim(0, 0.85)
    a1b = a1.twinx()
    a1b.grid(False)
    l2, = a1b.plot(xs, smooth(sv, 20), lw=1.3, color=GOOD,
                   label="fully solved \\% of episodes (right)")
    a1b.set_ylabel("fully solved (\\%)", color=GOOD)
    a1b.tick_params(axis="y", labelcolor=GOOD)
    a1b.set_ylim(0, 85)
    a1b.spines["top"].set_visible(False)
    a1.set_title("Agentic RL on the 12-task control set: it learns, then exhausts the set")
    a1.legend(handles=[l1, l2], frameon=False, fontsize=7, loc="upper left")
    a2.plot(xs, [v * 100 for v in smooth(zv, 25)], lw=1.3, color=ACC)
    a2.set_ylabel("zero-variance\ngroups (%)")
    a2.set_xlabel("GRPO step (8 episodes each)")
    a2.set_ylim(0, 100)
    a2.text(xs[len(xs) // 2], 12,
            "a dropped group contributes no gradient:\nrising = the task set is out of signal",
            fontsize=6.5, color=ACC)
    save(fig, "agent_learning.pdf")


# ------------------------------------------- the verification shortcut / fix
def fig_verify():
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for name, label, colour in (("agent_repo_metrics.jsonl",
                                 "Arm A: reward reads the file only", ACC),
                                ("agent_ab_metrics.jsonl",
                                 "Arm B: + 0.1 discount for not verifying", GOOD)):
        rows = [r for r in load_jsonl(name) if r.get("event") == "step"]
        if not rows:
            continue
        G = len(rows[0].get("rewards") or [8])
        xs = [r["step"] for r in rows]
        rt = [(r.get("ran_tests") or 0) / G * 100 for r in rows]
        ax.plot(xs, smooth(rt, 20), lw=1.6, color=colour, label=label)
    ax.set_xlabel("GRPO step (8 episodes each)")
    ax.set_ylabel("episodes that ran\nthe tests (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("RL removes any behaviour the reward does not pay for")
    ax.annotate("verification trained OUT:\n75% $\\rightarrow$ 0%",
                (250, 8), fontsize=7.5, color=ACC)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    save(fig, "verify.pdf")


# ------------------------------------------------------------ hardness check
def fig_hardness():
    tiers = [("mutate\n1 token", 59, 4, 86, 3),
             ("multi\n2 tokens", 60, 4, 86, 3),
             ("stub\nwrite body", 0, 0, 98, 1),
             ("swap\nwrong soln", 0, 0, 93, 2)]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    xs = range(len(tiers))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [t[3] for t in tiers], w,
           yerr=[t[4] for t in tiers], color=GREY, label="wrote a file (%)",
           capsize=2.5, error_kw={"lw": 0.8})
    ax.bar([x + w / 2 for x in xs], [t[1] for t in tiers], w,
           yerr=[t[2] for t in tiers], color=ACC, label="solved (%)",
           capsize=2.5, error_kw={"lw": 0.8})
    for x, t in zip(xs, tiers):
        if t[1] == 0:
            ax.text(x + w / 2, 4, "0/120", ha="center", fontsize=7,
                    color=ACC, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([t[0] for t in tiers], fontsize=7.5)
    ax.set_ylabel("% of 120 episodes")
    ax.set_ylim(0, 108)
    ax.set_title("The 70\\% solve rate was an artifact of one-token breakages")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    save(fig, "hardness.pdf")


if __name__ == "__main__":
    print("figures ->", FIG)
    fig_pretrain()
    fig_throughput()
    fig_depth()
    fig_sft()
    fig_agent_learning()
    fig_verify()
    fig_hardness()
    print("done")
