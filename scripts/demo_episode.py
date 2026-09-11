"""Print one full agent episode: the broken file, every turn, and the result.

This is the same harness that produces the scoreboard numbers, so what it shows
is exactly what is being measured -- no separate demo path that might flatter.

Usage:
    python scripts/demo_episode.py CKPT [--tier mutate] [--holdout 74]
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--tier", default="mutate")
    ap.add_argument("--holdout", type=int, default=74,
                    help="draw ONLY from tasks never seen in training")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-attempts", type=int, default=8,
                    help="keep sampling until one SOLVES, and report how many "
                         "attempts it took -- at ~57%% pass@1 a single success "
                         "shown on its own would misrepresent the model")
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

    tasks = get_tasks("mbpp")
    if a.holdout:
        tasks = tasks[-a.holdout:]
    scns = build_scenarios(tasks, seed=7, limit=6, difficulty=a.tier)
    random.Random(1).shuffle(scns)
    sc = scns[0]

    print("\n" + "=" * 72)
    print(f"TASK {sc.name}   (held-out: never seen in training)")
    print("=" * 72)
    print(sc.prompt.strip()[:500])
    print("\n--- solution.py AS GIVEN (verified to fail its tests) ---")
    print(sc.broken.rstrip())

    for attempt in range(1, a.max_attempts + 1):
        ep = run_repo_episode(model, tok, sc, eos, max_turns=6,
                              max_new_tokens=256, n_loops=cfg.n_loops,
                              temperature=a.temperature, keep_repo=True)
        solved = ep.reward >= 1.0
        if solved or attempt == a.max_attempts:
            print(f"\n--- TRANSCRIPT (attempt {attempt}) ---")
            for role, text in ep.transcript:
                body = text.strip()
                if len(body) > 700:
                    body = body[:700] + " ...[truncated]"
                print(f"\n[{role.upper()}]\n{body}")
            sol = os.path.join(ep.repo or "", "solution.py")
            final = ""
            if ep.repo and os.path.isfile(sol):
                final = open(sol, encoding="utf-8").read()
            print("\n--- solution.py AS LEFT BY THE AGENT ---")
            print(final.rstrip() or "(unchanged / empty)")
            print(f"\n--- GRADE (harness runs the hidden tests) ---")
            print(f"    {ep.passed}/{ep.total} tests passed   reward {ep.reward:.2f}"
                  f"   {'SOLVED' if solved else 'FAILED'}")
            print(f"    turns {ep.turns}, tool calls {ep.tool_calls}, "
                  f"errors {ep.tool_errors}")
            print(f"\nsucceeded on attempt {attempt} of {a.max_attempts}"
                  if solved else
                  f"\nno success in {a.max_attempts} attempts")
            if ep.repo:
                shutil.rmtree(ep.repo, ignore_errors=True)
            return 0 if solved else 1
        if ep.repo:
            shutil.rmtree(ep.repo, ignore_errors=True)
        print(f"attempt {attempt}: failed ({ep.passed}/{ep.total})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
