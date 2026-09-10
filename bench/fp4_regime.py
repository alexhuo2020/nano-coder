"""Find the regime where NVFP4 actually beats FP8, sweeping BOTH axes.

An earlier sweep varied only d_model, at a small token-batch, and concluded FP4
does not pay below d_model 3072. That conclusion is confounded: FP4's advantage
is tensor-core *compute* throughput, which only materialises when the GEMM is
compute-bound -- and the GEMM's M dimension (tokens per micro-batch) is what
drives arithmetic intensity just as much as its K/N (d_model/ffn) do.

So sweep both. If FP4's win is recoverable at small d_model simply by enlarging
the batch, then a 96GB card can have FP4 *and* a model small enough to train to
competence -- which the d_model-only reading said was impossible.

Requires a TE built with NVTE_CUDA_ARCHS=120a; the released wheel silently
falls back to biased FP4 rounding and floods stderr per-thread, which both
corrupts numerics and destroys the timing.
"""
import argparse
import itertools
import json

import torch

from blackwell_lm.model import ModelConfig, nvfp4_stochastic_rounding_ok
from bench.throughput import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--out", default="fp4_regime.json")
    a = p.parse_args()

    ok = nvfp4_stochastic_rounding_ok()
    print(f"NVFP4 stochastic-rounding PTX available: {ok}")
    if not ok:
        print("  !! TE was not built for sm_120a. NVFP4 numbers below are INVALID.")

    widths = [768, 1536]
    batches = [8, 16, 32, 48, 64]
    results = []
    for d_model, batch in itertools.product(widths, batches):
        tokens = batch * a.seq_len
        cfg = ModelConfig(d_model=d_model, n_layers=12,
                          n_heads=d_model // 128,
                          n_kv_heads=max(1, d_model // 128 // 3),
                          ffn_hidden=4 * d_model, max_seq_len=a.seq_len,
                          window=0, global_every=0)  # full causal: no mask, fastest path
        row = {"d_model": d_model, "batch": batch, "tokens_per_batch": tokens}
        for prec in ("fp8", "nvfp4"):
            try:
                tps, loss, peak = run(cfg, prec, batch, a.steps, a.warmup)
                row[prec] = round(tps)
                row[f"{prec}_peak_gib"] = round(peak, 1)
            except torch.OutOfMemoryError:
                row[prec] = None
                torch.cuda.empty_cache()
            except Exception as e:  # keep sweeping; record what broke
                row[prec] = None
                row[f"{prec}_error"] = type(e).__name__
                torch.cuda.empty_cache()
        if row.get("fp8") and row.get("nvfp4"):
            row["ratio"] = round(row["nvfp4"] / row["fp8"], 4)
        results.append(row)
        r = row.get("ratio")
        print(f"  d_model={d_model:5d} batch={batch:3d} ({tokens:>6,} tok) "
              f"fp8={row.get('fp8') or 'OOM':>9} nvfp4={row.get('nvfp4') or 'OOM':>9} "
              f"ratio={r if r else '-'}")
        if row.get("fp8") is None and row.get("nvfp4") is None:
            print(f"    (both OOM at batch {batch}; stopping this width)")
            batches = [b for b in batches if b < batch]

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    best = [r for r in results if r.get("ratio")]
    if best:
        top = max(best, key=lambda r: r["ratio"])
        print(f"best NVFP4/FP8 ratio: {top['ratio']:.3f}x at d_model={top['d_model']} "
              f"batch={top['batch']} ({top['tokens_per_batch']:,} tokens/batch)")


if __name__ == "__main__":
    main()
