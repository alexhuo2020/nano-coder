"""Score a checkpoint on ALL FOUR breakage tiers, with standard errors.

One tier is not a result. This project's two worst errors both came from
judging a model by a single tier: `mutate` alone said 70% (flattering), and
`stub` alone said 0% (damning). Neither was the model's actual profile.

Usage:
    python scripts/scoreboard.py CKPT [--tasks 20] [--samples 6] [--tiers ...]

Reports per tier: pass@1 with a standard error, pass@k, the share of samples
earning ANY partial credit (the quantity that decides whether RL can run at
all -- zero partial credit means every GRPO group is zero-variance), and what
the agent actually left in the file.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import random
import shutil
import sys

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
sys.path.insert(0, "/home/ubuntu/bnano")

import torch

from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.repo_agent import run_repo_episode
from blackwell_lm.scenario import build_scenarios
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default="/home/ubuntu/bnano/tokenizer.json")
    ap.add_argument("--tiers", default="mutate,multi,stub,swap")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--task-set", default="mbpp")
    ap.add_argument("--holdout", type=int, default=0,
                    help="evaluate ONLY on the last N tasks, which must match "
                         "train_sft.py --tool-holdout. Anything else reports "
                         "train accuracy.")
    ap.add_argument("--n-loops", type=int, default=None,
                    help="override the loop count at inference (weight-shared "
                         "depth). The depth ablation minimised held-out LOSS "
                         "at the trained 8, but loss and task success are "
                         "different objectives and extra depth on a hard "
                         "prompt was advertised as free -- never tested "
                         "against a task metric.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 is the RL rollout setting, correct for "
                         "exploration and not necessarily for SOLVING. Lower "
                         "temperature usually raises pass@1 and lowers pass@k; "
                         "neither the 1.0 here nor the 0.7 the server used was "
                         "ever chosen by measurement.")
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    eos = tok.token_to_id(EOS)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    cfg.max_seq_len = max(cfg.max_seq_len, 2048)
    model = BlackwellLM(cfg, precision="bf16", device="cuda",
                        dtype=torch.bfloat16)
    load_stage_checkpoint(a.ckpt, model)
    model.eval()
    # A bigger loop count needs the RoPE table and mask built for it; the
    # buffer is non-persistent so this only costs its construction.
    n_loops = a.n_loops or cfg.n_loops

    tasks = get_tasks(a.task_set)
    if a.holdout:
        tasks = tasks[-a.holdout:]
        print(f"[scoreboard] HELD-OUT split: {len(tasks)} tasks "
              f"({tasks[0].name}..{tasks[-1].name}), none of them seen in "
              f"training", flush=True)
    name = os.path.basename(a.ckpt)
    n = a.tasks * a.samples
    print()
    print(f"=== {name}  {a.tasks} tasks x {a.samples} samples = {n} episodes "
          f"per tier  (T={a.temperature}, loops={n_loops}, "
          f"max_turns={a.max_turns}) ===")
    print(f"{'tier':8} {'pass@1':>12} {'pass@k':>8} {'partial':>9} "
          f"{'wrote':>7} {'verified':>9} {'turns':>6}")

    rows = []
    for tier in [t.strip() for t in a.tiers.split(",") if t.strip()]:
        scns = build_scenarios(tasks, seed=11, limit=a.tasks, difficulty=tier)
        random.Random(0).shuffle(scns)
        scns = scns[:a.tasks]

        solved = wrote = verified = tot = partial = any_task = 0
        turns_sum = 0
        kinds = collections.Counter()
        for sc in scns:
            hit = 0
            for _ in range(a.samples):
                ep = run_repo_episode(model, tok, sc, eos,
                                      max_turns=a.max_turns,
                                      max_new_tokens=256, n_loops=n_loops,
                                      temperature=a.temperature,
                                      keep_repo=True)
                tot += 1
                turns_sum += ep.turns
                if ep.reward >= 1.0:
                    solved += 1
                    hit += 1
                if ep.reward > 0:
                    partial += 1
                wrote += 1 if ep.wrote_file else 0
                verified += 1 if ep.verified_pass else 0
                sol = os.path.join(ep.repo or "", "solution.py")
                code = ""
                if ep.repo and os.path.isfile(sol):
                    code = open(sol, encoding="utf-8", errors="replace").read()
                if not code.strip():
                    kinds["empty"] += 1
                elif code.strip().endswith("pass") and "return" not in code:
                    kinds["still-a-stub"] += 1
                else:
                    try:
                        compile(code, "s.py", "exec")
                        kinds["valid-python"] += 1
                    except SyntaxError:
                        kinds["syntax-error"] += 1
                if ep.repo:
                    shutil.rmtree(ep.repo, ignore_errors=True)
            any_task += 1 if hit else 0

        p1 = solved / tot
        se = math.sqrt(max(p1 * (1 - p1), 1e-9) / tot) * 100
        print(f"{tier:8} {p1*100:7.1f}% +-{se:<3.1f} "
              f"{any_task/len(scns)*100:7.0f}% {partial/tot*100:8.1f}% "
              f"{wrote/tot*100:6.0f}% {verified/tot*100:8.0f}% "
              f"{turns_sum/tot:6.2f}   {dict(kinds)}")
        rows.append((tier, p1 * 100, any_task / len(scns) * 100))
        kinds.clear()

    print()
    print("mean pass@1 across tiers: "
          f"{sum(r[1] for r in rows)/max(1,len(rows)):.1f}%   "
          "(the number a single-tier run cannot tell you)")


if __name__ == "__main__":
    main()
