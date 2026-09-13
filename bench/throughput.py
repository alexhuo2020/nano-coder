"""Measure real training tokens/s for nanocoder on the local GPU, and A/B
the precision choice the model's design rests on (FP8 vs BF16 vs, if TE was
built for the right arch, NVFP4).

Reports tokens/s over a timed window that EXCLUDES warmup, because the first
steps pay one-off allocation and autotune costs that would flatter a short run.
"""
import argparse
import time

import torch

from nanocoder.model import BlackwellLM, ModelConfig


def run(cfg, precision, batch, steps, warmup):
    model = BlackwellLM(cfg, precision=precision, device="cuda", dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(0, cfg.vocab_size, (batch, cfg.max_seq_len), device="cuda")

    def step():
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(), ids.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    toks = steps * batch * cfg.max_seq_len
    peak = torch.cuda.max_memory_allocated() / 2**30
    return toks / dt, loss.item(), peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", default="fp8", choices=["fp8", "bf16", "nvfp4"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    a = p.parse_args()

    cfg = ModelConfig(d_model=a.d_model, n_layers=a.n_layers,
                      n_heads=a.d_model // 128, n_kv_heads=max(1, a.d_model // 128 // 3),
                      ffn_hidden=4 * a.d_model, max_seq_len=a.seq_len)
    model_params = None
    tps, loss, peak = run(cfg, a.precision, a.batch, a.steps, a.warmup)
    print(f"precision={a.precision} d_model={cfg.d_model} L={cfg.n_layers} "
          f"heads={cfg.n_heads}/{cfg.n_kv_heads} batch={a.batch} seq={cfg.max_seq_len}")
    print(f"  {tps:,.0f} tok/s | loss {loss:.3f} | peak mem {peak:.1f} GiB")


if __name__ == "__main__":
    main()
