"""Single-turn MBPP: write the function from the prompt alone, no repo, no tools.

WHY RE-MEASURE. This project reported 9/300 (3%) and used it as the headline
evidence that the model "cannot write code". That figure was taken at
temperature 1.0, the RL rollout setting. Temperature turned out to be worth
+24pp on every agentic tier, so the 3% is a claim about a sampling
configuration as much as about the model.

This is the harder task than any repo tier: there is no existing file to read,
no tests to iterate against, and no second attempt. Whatever it scores here is
the floor of what the model can do unaided.

Usage:
    python scripts/mbpp_eval.py CKPT [--temperature 0.2] [--holdout 74]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
sys.path.insert(0, "/home/ubuntu/bnano")

import torch

from blackwell_lm import chat
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.generate import generate
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.reward import extract_code
from blackwell_lm.sandbox import run_tests
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default="/home/ubuntu/bnano/tokenizer.json")
    ap.add_argument("--tasks", type=int, default=74)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--holdout", type=int, default=0,
                    help="use ONLY the last N tasks (match --tool-holdout)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
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
    tasks = tasks[:a.tasks]

    solved = partial = tot = any_task = extracted = 0
    for t in tasks:
        hit = 0
        for _ in range(a.samples):
            msgs = chat.user_turn(t.prompt)
            ids = chat.tokenize_prompt(tok, msgs)
            toks, valid, _ = generate(model, ids,
                                      max_new_tokens=a.max_new_tokens,
                                      eos_id=eos, temperature=a.temperature,
                                      num_return_sequences=1,
                                      n_loops=cfg.n_loops)
            text = tok.decode([int(x) for x, v in
                               zip(toks[0].tolist(), valid[0].tolist()) if v])
            code = extract_code(text)
            tot += 1
            if not code:
                continue
            extracted += 1
            ok, total, _ = run_tests(code, t.tests, timeout=6.0,
                                     setup=getattr(t, "setup", ""))
            if total and ok == total:
                solved += 1
                hit += 1
            elif ok:
                partial += 1
        any_task += 1 if hit else 0

    p1 = solved / max(tot, 1)
    se = math.sqrt(max(p1 * (1 - p1), 1e-9) / max(tot, 1)) * 100
    print()
    print(f"=== {os.path.basename(a.ckpt)}  single-turn MBPP  "
          f"{len(tasks)} tasks x {a.samples} samples = {tot} samples  "
          f"(T={a.temperature}{', HELD-OUT' if a.holdout else ''}) ===")
    print(f"  emitted a code block : {extracted/max(tot,1)*100:5.1f}%")
    print(f"  pass@1               : {p1*100:5.1f}% +-{se:.1f}   ({solved}/{tot})")
    print(f"  pass@{a.samples:<2}              : "
          f"{any_task/max(len(tasks),1)*100:5.1f}%   "
          f"({any_task}/{len(tasks)} tasks)")
    print(f"  partial credit       : {partial/max(tot,1)*100:5.1f}%")


if __name__ == "__main__":
    main()
