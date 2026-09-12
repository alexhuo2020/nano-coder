"""Pretraining for nano-coder.

Budgeted by TOKENS, not steps, because tokens/param is the quantity that decides
whether the result is coherent -- and the failure this project exists to avoid is
undertraining, not slow convergence.

Spot-safe by construction: checkpoints on a wall-clock interval, mirrors to S3,
resumes from wherever it left off, and traps SIGTERM so an interruption notice
still produces a usable checkpoint. A multi-day single-GPU run WILL be
interrupted; that is planned for rather than hoped against.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time

import torch

# Transformer Engine keeps its FP8/NVFP4 scaling metadata in a pickled "extra
# state" blob, and because unpickling can execute arbitrary code it REFUSES to
# load one unless this is set. Our checkpoints are self-produced and read back
# from our own private bucket, so they are a trusted source. This must be set
# before the model import below, which is what pulls TE in.
#
# This is load-bearing for spot training specifically: without it, a fresh run
# is fine but every RESUME dies inside load_state_dict -- so an interrupted run
# comes back, crashes, gets relaunched, and crashes again, burning the whole
# night in a loop while looking like it is being managed. Found by testing a
# resume rather than a cold start.
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

from blackwell_lm.data import (DEFAULT_MIXTURE, batches, packed_blocks, prefetch,
                               stream_mixture)
from blackwell_lm.model import BlackwellLM, ModelConfig, nvfp4_stochastic_rounding_ok
from blackwell_lm.tokenizer import EOS, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, load_tokenizer


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def lr_at(step: int, total: int, peak: float, warmup: int, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = peak * floor_frac
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


def append_metric(path: str, row: dict):
    """Append one JSON line of training metrics.

    Exists because the loss history was NOT durably recorded: it lived only in a
    log file on ephemeral spot storage, and two instance migrations destroyed
    about two thirds of the curve before anyone looked. The checkpoint carried
    model/opt/step/cfg and no loss at all, so a reclaim erased the only record
    of how training had gone. This file is mirrored to S3 with every checkpoint.
    """
    import json
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def s3_sync(local: str, uri: str | None):
    if not uri:
        return
    try:
        import boto3
        bucket, _, key = uri[len("s3://"):].partition("/")
        boto3.client("s3").upload_file(local, bucket, key)
        log(f"mirrored checkpoint -> {uri}")
    except Exception as e:
        log(f"WARNING: S3 mirror failed ({e!r}); local checkpoint is still valid")


def s3_fetch(uri: str | None, local: str) -> bool:
    if not uri:
        return False
    try:
        import boto3, botocore.exceptions
        bucket, _, key = uri[len("s3://"):].partition("/")
        boto3.client("s3").download_file(bucket, key, local)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--target-tokens", type=float, default=14e9, help="token budget (default: a 2-day PRO 6000 run)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--precision", default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint", default="pretrain.pt")
    p.add_argument("--s3-uri", default=os.environ.get("BLACKWELL_S3_URI"))
    p.add_argument("--checkpoint-every-s", type=float, default=1800)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--metrics", default="metrics.jsonl",
                   help="durable JSONL of step/loss/lr; mirrored to S3 with each checkpoint")
    p.add_argument("--fim-rate", type=float, default=0.5)
    p.add_argument("--prefetch", type=int, default=64, help="blocks buffered ahead of the GPU")
    a = p.parse_args()

    assert torch.cuda.is_available(), "pretraining requires a CUDA device"
    tok = load_tokenizer(a.tokenizer)
    eos_id = tok.token_to_id(EOS)
    fim_ids = tuple(tok.token_to_id(t) for t in (FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX))
    assert eos_id is not None and None not in fim_ids, "tokenizer is missing EOS/FIM sentinels"

    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), max_seq_len=a.seq_len)
    if a.precision == "nvfp4" and not nvfp4_stochastic_rounding_ok():
        log("WARNING: NVFP4 selected but stochastic-rounding PTX is unavailable. TE will "
            "silently fall back to BIASED 4-bit rounding and flood stderr per-thread, "
            "which also destroys throughput. Rebuild TE with NVTE_CUDA_ARCHS=120a, or "
            "use --precision fp8.")

    model = BlackwellLM(cfg, precision=a.precision, device="cuda", dtype=torch.bfloat16)

    # FP32 MASTER WEIGHTS. The model must hold bf16 parameters -- TE's NVFP4
    # quantizer rejects fp32 input outright ("RHT is only supported for bfloat16
    # input") -- but bf16 parameters cannot be the optimizer's state, because
    # bf16 cannot represent the updates.
    #
    # MEASURED, on the real checkpoint at step 657k: with bf16 parameters, all
    # five RMSNorm gains had moved 0.00% of their elements in 657,000 steps --
    # still bit-exactly 1.0 -- and the embedding was losing 44% of its updates.
    # The reason is resolution, not gradients: bf16 spacing at |w|~1.0 is 2^-7 =
    # 7.8e-3 while an AdamW step is ~lr = 1.6e-4, so `p = p - update` rounds
    # straight back to p. It gets worse as lr decays: at the schedule floor of
    # 3e-5 even the weight matrices (|w|~0.03, spacing 1.2e-4) fall to
    # update/spacing = 0.25, so the anneal phase would silently stop learning
    # altogether rather than converging.
    #
    # So the optimizer owns fp32 masters and the bf16 params are a compute-only
    # copy refreshed after each step. Costs ~3% throughput (measured) and about
    # 1GB of memory, on a card with 70GB idle.
    # 1-D PARAMETERS (the RMSNorm gains) STAY FROZEN AT 1.0 -- learned the hard
    # way. They had been frozen for the run's entire history by the bf16 rounding
    # bug, so the other 85M weights are converged AGAINST gains of exactly 1.0.
    # Simply unfreezing them was measurably harmful: their optimizer exp_avg had
    # accumulated 657k steps of gradients whose updates were being discarded, so
    # the moment updates could land that stale momentum discharged -- in 39k
    # steps the gains went 1.0 -> mean 0.67 (n1), 0.70 (n2), 2.17 (final norm),
    # and fixed-block loss on the real mixture rose 1.4555 -> 1.6659.
    #
    # Freezing them is also just a legitimate architecture choice (a fixed unit
    # gain), and they are 4,864 of 85M parameters -- 0.006% of capacity, not
    # worth a second transient in a run that is 53% complete. The fp32 masters
    # below still fix the part that matters: without them the weight matrices
    # and embedding lose their updates once lr anneals toward 3e-5.
    for _n, _p in model.named_parameters():
        if _p.ndim == 1:
            _p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    master = [p.detach().float().clone() for p in params]
    for q in master:
        q.requires_grad_(True)
        q.grad = torch.zeros_like(q)          # preallocated once, reused
    opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    log(f"model: d_model={cfg.d_model} unique_blocks={cfg.n_layers} n_loops={cfg.n_loops} "
        f"(effective depth {cfg.effective_depth}) | {model.n_params()/1e6:.1f}M params "
        f"| precision={model.precision}")

    tokens_per_step = a.batch_size * a.seq_len
    total_steps = int(a.target_tokens / tokens_per_step)
    log(f"budget: {a.target_tokens/1e9:.1f}B tokens = {total_steps:,} steps "
        f"({tokens_per_step:,} tokens/step) -> {a.target_tokens/model.n_params():.0f} tokens/param")

    start_step = 0
    prior_loss = []
    if not os.path.exists(a.checkpoint) and s3_fetch(a.s3_uri, a.checkpoint):
        log("recovered checkpoint from S3")
    if os.path.exists(a.checkpoint):
        ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        # Prefer saved fp32 masters. A checkpoint written before master weights
        # existed has none, in which case seeding them from the bf16 weights is
        # exact (bf16 -> fp32 is a lossless upcast) and loses no progress.
        if "master" in ck:
            with torch.no_grad():
                torch._foreach_copy_(master, [t.to("cuda") for t in ck["master"]])
            log("restored fp32 master weights from checkpoint")
        else:
            with torch.no_grad():
                torch._foreach_copy_(master, params)
            log("checkpoint predates fp32 masters; seeded them from the bf16 "
                "weights (lossless upcast)")
        # The saved optimizer state may cover MORE parameters than are now
        # trainable (checkpoints written before the 1-D gains were frozen hold
        # 11 entries; only 6 tensors train now). Its state is keyed by position
        # in the original parameter order, so remap those positions rather than
        # discarding the state -- throwing away Adam moments mid-run causes a
        # loss spike on the next few hundred steps.
        saved = ck["opt"]
        n_now = len(params)
        if len(saved["param_groups"][0]["params"]) != n_now:
            all_p = list(model.parameters())
            keep_idx = [i for i, q in enumerate(all_p) if q.requires_grad]
            remapped = {}
            for new_i, old_i in enumerate(keep_idx):
                if old_i in saved["state"]:
                    remapped[new_i] = saved["state"][old_i]
            saved = {"state": remapped,
                     "param_groups": [{**saved["param_groups"][0],
                                       "params": list(range(n_now))}]}
            log(f"remapped optimizer state: {len(ck['opt']['state'])} saved entries "
                f"-> {len(remapped)} for the {n_now} trainable tensors")
        opt.load_state_dict(saved)
        start_step = ck["step"]
        prior_loss = ck.get("loss_history") or []
        log(f"resumed at step {start_step:,} ({start_step*tokens_per_step/1e9:.2f}B tokens done)")

    stop = {"now": False}
    def _stop(signum, frame):
        log(f"signal {signum} received -- checkpointing before exit")
        stop["now"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # prefetch on a background thread so tokenization/streaming overlaps the GPU
    stream = prefetch(
        packed_blocks(stream_mixture(DEFAULT_MIXTURE), tok, a.seq_len, eos_id,
                      fim_ids=fim_ids, fim_rate=a.fim_rate),
        depth=a.prefetch,
    )
    data = batches(stream, a.batch_size)

    def save(step):
        tmp = a.checkpoint + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": cfg.__dict__,
                    # fp32 masters are the real weights; the bf16 state_dict is
                    # a rounded view of them kept for eval/SFT compatibility
                    "master": [q.detach().cpu() for q in master],
                    # loss history rides WITH the weights, so a reclaim can
                    # never again destroy the record of how training went
                    "loss": (sum(losses[-a.log_every:]) / min(len(losses), a.log_every)
                             if losses else None),
                    "loss_history": recent_loss[-2000:]}, tmp)
        os.replace(tmp, a.checkpoint)          # atomic: an interrupted save cannot corrupt the good one
        log(f"checkpoint saved at step {step:,}")
        s3_sync(a.checkpoint, a.s3_uri)
        if a.s3_uri:                            # metrics next to the checkpoint
            base = a.s3_uri.rsplit("/", 1)[0]
            s3_sync(a.metrics, f"{base}/metrics/metrics.jsonl")

    model.train()
    t_start = time.perf_counter()
    last_ck = t_start
    losses = []
    recent_loss = list(prior_loss)   # (step, loss) pairs, persisted in the checkpoint
    for step in range(start_step, total_steps):
        inp, tgt = next(data)
        logits = model(inp)                     # loop count is SAMPLED here (model.training)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(), tgt.reshape(-1))
        loss.backward()
        # bf16 grads -> fp32 master grads, one fused launch
        torch._foreach_copy_([q.grad for q in master], [p.grad for p in params])
        torch.nn.utils.clip_grad_norm_(master, a.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, a.lr, a.warmup)
        opt.step()
        with torch.no_grad():                 # fp32 masters -> bf16 compute copy
            torch._foreach_copy_(params, master)
        model.zero_grad(set_to_none=False)    # keep grad buffers allocated
        losses.append(loss.item())
        if (step + 1) % a.log_every == 0:
            recent_loss.append((step + 1, round(losses[-1], 5)))

        if (step + 1) % a.log_every == 0:
            done = (step + 1 - start_step) * tokens_per_step
            el = time.perf_counter() - t_start
            append_metric(a.metrics, {
                "t": time.time(), "step": step + 1,
                "loss": round(sum(losses[-a.log_every:]) / a.log_every, 5),
                "lr": lr_at(step, total_steps, a.lr, a.warmup),
                "tokens": (step + 1) * tokens_per_step,
                "tok_per_s": round(done / el),
            })
            log(f"step {step+1:,}/{total_steps:,} | loss {sum(losses[-a.log_every:])/a.log_every:.4f} "
                f"| {done/el:,.0f} tok/s | {(step+1)*tokens_per_step/1e9:.2f}B tokens "
                f"| lr {lr_at(step, total_steps, a.lr, a.warmup):.2e} "
                f"| eta {(total_steps-step-1)*tokens_per_step/max(done/el,1)/3600:.1f}h")

        if stop["now"] or time.perf_counter() - last_ck >= a.checkpoint_every_s:
            save(step + 1)
            last_ck = time.perf_counter()
            if stop["now"]:
                log("exiting cleanly after checkpoint")
                sys.exit(0)

    save(total_steps)
    log(f"pretraining complete: {total_steps*tokens_per_step/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
