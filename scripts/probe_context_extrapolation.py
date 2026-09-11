"""Does this model extrapolate past its 2,048-token training length?

THE PREDICTION, AND WHY IT IS NOT OBVIOUS. A globally-attending model trained
at 2,048 degrades sharply beyond it: query-key pairs at relative offsets it
never saw produce RoPE rotations it never learned to read. This model is
different in one architectural respect that should matter more than the
training length: `window=1024` with `global_every_loop=0`, so EVERY attention
application is a 1,024-token causal sliding window and no layer ever sees the
full context. RoPE's dot product depends on the relative offset i-j, and inside
a 1,024 window every offset is <= 1,024 -- in-distribution regardless of
whether the token sits at absolute position 500 or 50,000.

If that reasoning holds, held-out loss should be FLAT in sequence length, and
raising max_seq_len is free. If it does not hold, loss will climb past 2,048
and the context genuinely needs continued pretraining with position scaling.
Either way the answer is a measurement, and it costs one pass over held-out
text.

Reported per length: mean CE over the WHOLE sequence, and -- more informative --
CE over only the tokens BEYOND 2,048, which is where extrapolation actually
gets exercised. An aggregate can look fine simply because most tokens sit in
the region the model was trained on.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
sys.path.insert(0, "/home/ubuntu/bnano")

import torch

from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.data import DEFAULT_MIXTURE, packed_blocks, stream_mixture
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.tokenizer import EOS, load_tokenizer


def evaluate(model, blocks, cfg, trained_len: int):
    """Returns (all_tokens_CE, beyond_trained_len_CE)."""
    tot, n = 0.0, 0
    tot_far, n_far = 0.0, 0
    with torch.no_grad():
        for x in blocks:
            x = x.cuda(non_blocking=True)
            inp, tgt = x[:, :-1], x[:, 1:]
            logits = model(inp, n_loops=cfg.n_loops)
            ce = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                tgt.reshape(-1), reduction="none").view(tgt.shape)
            tot += ce.sum().item()
            n += ce.numel()
            if ce.shape[1] > trained_len:
                far = ce[:, trained_len:]
                tot_far += far.sum().item()
                n_far += far.numel()
    return (tot / max(n, 1), (tot_far / n_far) if n_far else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ubuntu/bnano/pretrain.pt")
    ap.add_argument("--tokenizer", default="/home/ubuntu/bnano/tokenizer.json")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    ap.add_argument("--blocks", type=int, default=12,
                    help="held-out sequences per length")
    ap.add_argument("--trained-len", type=int, default=2048)
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    eos = tok.token_to_id(EOS)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    base = dict(ck["cfg"])
    print(f"checkpoint {os.path.basename(a.ckpt)}: trained max_seq_len="
          f"{base.get('max_seq_len')}  window={base.get('window')}  "
          f"global_every_loop={base.get('global_every_loop')}  "
          f"n_loops={base.get('n_loops')}", flush=True)
    print()
    print(f"{'seq_len':>8} {'CE (all tok)':>14} {'CE (>2048 only)':>17} "
          f"{'ppl':>9}  {'vs 2048':>9}")

    ref = None
    for L in [int(s) for s in a.lengths.split(",")]:
        cfg_d = dict(base)
        cfg_d["max_seq_len"] = L          # rebuilds the RoPE table at this size
        cfg = ModelConfig(**cfg_d)
        model = BlackwellLM(cfg, precision="bf16", device="cuda",
                            dtype=torch.bfloat16)
        load_stage_checkpoint(a.ckpt, model)
        model.eval()

        # A FRESH stream per length, seeded identically, so every length is
        # scored on the same underlying text rather than on whatever happened
        # to come next in a shared iterator.
        # packed_blocks(seq_len=L) yields 1-D arrays of exactly L+1 tokens,
        # which is what gives input/target pairs without re-reading.
        stream = packed_blocks(stream_mixture(DEFAULT_MIXTURE, seed=1234), tok,
                               L, eos)
        blocks = []
        for i, b in enumerate(stream):
            if i >= a.blocks:
                break
            blocks.append(torch.from_numpy(b.astype("int64"))[None, :])

        ce, ce_far = evaluate(model, blocks, cfg, a.trained_len)
        if ref is None:
            ref = ce
        delta = ce - ref
        far = f"{ce_far:.4f}" if not math.isnan(ce_far) else "-"
        print(f"{L:>8} {ce:>14.4f} {far:>17} {math.exp(min(ce, 20)):>9.2f}  "
              f"{delta:>+9.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
