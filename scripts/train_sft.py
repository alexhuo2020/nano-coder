"""Stage 2: supervised fine-tuning.

Turns the pretrained next-token model into something that answers a question in
a fixed conversation format. Cheap compared to pretraining (hours, not days) and
the prerequisite for both RL stages: GRPO needs a policy that at least attempts
the format, or every rollout scores zero and there is no gradient.

GATE TO CLEAR BEFORE MOVING ON (checked by --eval-every):
  * masked held-out loss below the base checkpoint's on the same data, and
  * the model actually emits a fenced code block when asked.
Both are printed; the second is the one that matters, because a good loss with
no format compliance still gives GRPO nothing to work with.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import signal
import sys
import time

# Before any TE import: see blackwell_lm/checkpoint.py.
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

import torch

from blackwell_lm import chat
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.generate import generate
from blackwell_lm import metrics
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.optim import MasterWeightOptimizer, frozen_report
from blackwell_lm.sft import (DEFAULT_SFT_MIXTURE, masked_cross_entropy,
                              sft_batches, stream_sft)
from blackwell_lm.tokenizer import EOS, load_tokenizer


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def lr_at(step, total, peak, warmup, floor_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = peak * floor_frac
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


def s3_sync(local, uri):
    if not uri:
        return
    try:
        import boto3
        bucket, _, key = uri[len("s3://"):].partition("/")
        boto3.client("s3").upload_file(local, bucket, key)
        log(f"mirrored -> {uri}")
    except Exception as e:
        log(f"WARNING: S3 mirror failed ({e!r}); local checkpoint is still valid")


@torch.no_grad()
def format_compliance(model, tok, eos_id, n=4, n_loops=None):
    """Does it actually produce a fenced code block? The loss cannot tell you."""
    hits = 0
    for q in ["Write a function add(a, b) that returns a + b.",
              "Write a function that reverses a string.",
              "Write a function to compute a factorial.",
              "Write a function that returns the maximum of a list."][:n]:
        ids = chat.tokenize_prompt(tok, chat.user_turn(q))
        toks, valid, _ = generate(model, ids, max_new_tokens=96, eos_id=eos_id,
                                  temperature=0.8, top_p=0.95, n_loops=n_loops)
        text = tok.decode([int(t) for t, v in zip(toks[0].tolist(), valid[0].tolist()) if v])
        if "```" in text or text.lstrip().startswith(("def ", "import ")):
            hits += 1
    return hits / max(1, n)


@torch.no_grad()
def tool_call_rate(model, tok, eos_id, n=4, n_loops=None):
    """Does the model emit a PARSEABLE tool call under the agent system prompt?

    This is the gate agentic RL actually depends on. With no tool call ever
    sampled there is no gradient toward producing one, so RL cannot bootstrap
    the convention by itself -- which is exactly how the first agentic run ended
    up as single-turn RL with 0 tool calls across 100 episodes.
    """
    from blackwell_lm.agent import AGENT_SYSTEM, parse_tool_call

    hits = 0
    qs = ["Write a function add(a, b) that returns a + b.",
          "Write a function that reverses a string.",
          "Write a function to compute a factorial.",
          "Write a function that returns the maximum of a list."][:n]
    for q in qs:
        msgs = [{"role": chat.SYSTEM, "content": AGENT_SYSTEM},
                {"role": chat.USER, "content": q}]
        ids = chat.tokenize_prompt(tok, msgs)
        toks, valid, _ = generate(model, ids, max_new_tokens=128, eos_id=eos_id,
                                  temperature=0.8, top_p=0.95, n_loops=n_loops)
        text = tok.decode([int(t) for t, v in zip(toks[0].tolist(), valid[0].tolist()) if v])
        if parse_tool_call(text) is not None:
            hits += 1
    return hits / max(1, len(qs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--init-from", required=True, help="pretraining checkpoint")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    # BF16 by default: SFT is a plain cross-entropy so low precision is safe,
    # but the RL stages that follow are not, and keeping one precision across
    # post-training removes a variable from any comparison between them.
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint", default="sft.pt")
    p.add_argument("--s3-uri", default=os.environ.get("BLACKWELL_SFT_S3_URI"))
    p.add_argument("--checkpoint-every-s", type=float, default=900)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--held-out", type=int, default=32, help="batches reserved for eval")
    p.add_argument("--tool-frac", type=float, default=0.25,
                   help="share of SFT examples that are synthetic TOOL-USE "
                        "trajectories. Without these the policy never emits a "
                        "tool call and agentic RL degenerates to single-turn RL "
                        "(measured: 0 tool calls in 100 episodes).")
    p.add_argument("--tool-task-set", default="mbpp", choices=["mbpp", "easy"])
    p.add_argument("--tool-difficulty", default="mutate",
                   help="comma-separated tiers to build repo demos from, e.g. "
                        "'stub,swap,mutate'. ONE TIER TEACHES ONE PRIOR: "
                        "trained only on mutate the policy learned 'write the "
                        "file back with one token changed' and scored 0/240 on "
                        "stub; trained only on stub it scored 0/240 on swap. "
                        "A mixture is the only way to get one model that "
                        "handles all of them.")
    p.add_argument("--tool-holdout", type=int, default=0,
                   help="reserve the LAST N tool tasks for evaluation and "
                        "never build demos from them. Without this, training "
                        "uses every MBPP task and the scoreboard draws its "
                        "tasks from the same pool, so the reported pass@1 is "
                        "TRAIN accuracy wearing the clothes of a benchmark.")
    p.add_argument("--tool-retry-frac", type=float, default=0.0,
                   help="share of repo demos that are RETRY trajectories "
                        "(wrong fix -> failing tests -> correct fix). Every "
                        "other demo is first-try-correct, so the policy has "
                        "never been shown what to do with a failing test "
                        "report -- driving a real CLI it reads, tests, then "
                        "stops. pass@12 is 40% vs pass@1 9.6%, so the ability "
                        "to land a second attempt exists and is unused.")

    p.add_argument("--tool-mode", default="snippet",
                   choices=["snippet", "repo", "both"],
                   help="snippet: run_tests(code) trajectories. repo: "
                        "read_file/write_file/run_tests over a real repo, which "
                        "is the schema the agentic repo environment actually "
                        "uses. both: half and half.")
    p.add_argument("--metrics", default="sft_metrics.jsonl",
                   help="durable JSONL of losses and gate results; mirrored to S3")
    a = p.parse_args()

    assert torch.cuda.is_available(), "SFT requires a CUDA device"
    tok = load_tokenizer(a.tokenizer)
    eos_id = tok.token_to_id(EOS)
    assert eos_id is not None, "tokenizer is missing EOS"

    ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    cfg.max_seq_len = max(cfg.max_seq_len, a.max_len + 1)
    log(f"base checkpoint: step {ck.get('step')} | d_model={cfg.d_model} "
        f"loops={cfg.n_loops} vocab={cfg.vocab_size}")
    assert cfg.vocab_size >= tok.get_vocab_size(), (
        f"tokenizer vocab {tok.get_vocab_size()} exceeds the checkpoint's {cfg.vocab_size}; "
        "this is a different tokenizer than the one pretraining used and every token id "
        "would mean something else"
    )

    model = BlackwellLM(cfg, precision=a.precision, device="cuda", dtype=torch.bfloat16)
    rep = load_stage_checkpoint(a.init_from, model)
    log(f"loaded pretrained weights (dropped {rep['dropped_quant_state']} quant-state entries) "
        f"| {model.n_params()/1e6:.1f}M params | precision={model.precision}")

    # Fresh optimizer -- pretraining Adam moments are meaningless under a
    # different objective. FP32 masters are MANDATORY here, not an optimisation:
    # at lr 1e-5 an AdamW update is 0.08x the bf16 spacing at |w|~0.03, so
    # updating bf16 params in place would discard essentially every update and
    # SFT would be a no-op that still printed a falling loss.
    opt = MasterWeightOptimizer(model, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                grad_clip=a.grad_clip)
    log(f"optimizer: {frozen_report(model)}")

    stream = stream_sft(DEFAULT_SFT_MIXTURE)
    if a.tool_frac > 0:
        import random as _random

        from blackwell_lm.tasks import get_tasks
        from blackwell_lm.tool_sft import stream_tool_sft

        base_tasks = get_tasks(a.tool_task_set)
        if a.tool_holdout:
            held = base_tasks[-a.tool_holdout:]
            base_tasks = base_tasks[:-a.tool_holdout]
            print(f"[sft] holdout: training on {len(base_tasks)} tool tasks, "
                  f"reserving {len(held)} ({held[0].name}..{held[-1].name}) "
                  f"for evaluation", flush=True)
        if a.tool_mode == "snippet":
            tool_stream = stream_tool_sft(base_tasks, seed=1)
        else:
            from blackwell_lm.scenario import build_scenarios
            from blackwell_lm.tool_sft import stream_repo_sft

            tiers = [t.strip() for t in a.tool_difficulty.split(",")
                     if t.strip()]
            valid = {"mutate", "multi", "stub", "swap"}
            bad = [t for t in tiers if t not in valid]
            if bad:
                raise SystemExit(f"unknown --tool-difficulty tier(s): {bad}; "
                                 f"choose from {sorted(valid)}")
            scns = []
            for ti, tier in enumerate(tiers):
                # A distinct seed per tier: the same seed would pick the same
                # tasks for every tier, so a "mixture" would be several views
                # of one small task subset instead of broader coverage.
                scns.extend(build_scenarios(base_tasks, seed=1 + ti,
                                            difficulty=tier))
            print(f"[sft] repo demo pool: {len(scns)} scenarios across "
                  f"tiers {tiers}", flush=True)
            repo_stream = stream_repo_sft(scns, seed=1,
                                          retry_frac=a.tool_retry_frac)
            if a.tool_mode == "repo":
                tool_stream = repo_stream
            else:
                snip = stream_tool_sft(base_tasks, seed=1)

                def _mix(x, y, seed=2):
                    r = _random.Random(seed)
                    while True:
                        yield next(x) if r.random() < 0.5 else next(y)

                tool_stream = _mix(repo_stream, snip)

        def blended(instr, tools, frac, seed=0):
            """Interleave instruction data with tool-use trajectories.

            Mixed at the EXAMPLE level rather than as separate phases: tuned on
            tools last, a model tends to emit tool calls for everything; tuned
            on them first, it forgets them.
            """
            rng = _random.Random(seed)
            while True:
                yield next(tools) if rng.random() < frac else next(instr)

        stream = blended(stream, tool_stream, a.tool_frac)
        log(f"SFT mixture: {1 - a.tool_frac:.0%} instruction + {a.tool_frac:.0%} "
            f"synthetic tool-use [mode={a.tool_mode}"
            f"{'/' + a.tool_difficulty if a.tool_mode != 'snippet' else ''}] "
            f"({a.tool_task_set} "
            f"reference solutions; tool output from REAL sandbox execution)")
    batches = sft_batches(stream, tok, eos_id, a.batch_size, a.max_len)
    held = list(itertools.islice(batches, a.held_out))
    log(f"reserved {len(held)} held-out batches (never trained on)")

    def evaluate():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for inp, tgt, msk in held:
                tot += masked_cross_entropy(model(inp, n_loops=cfg.n_loops), tgt, msk).item()
        model.train()
        return tot / max(1, len(held))

    base_loss = evaluate()
    base_fmt = format_compliance(model, tok, eos_id, n_loops=cfg.n_loops)
    base_tool = tool_call_rate(model, tok, eos_id, n_loops=cfg.n_loops)
    log(f"GATE BASELINE (pretrained, before SFT): held-out masked loss {base_loss:.4f} "
        f"| format compliance {base_fmt:.0%} | tool-call rate {base_tool:.0%}")
    # Written durably immediately. The first SFT run completed and then lost its
    # gate numbers to a spot reclaim, because they existed only in a log file on
    # ephemeral storage while the checkpoint sailed safely to S3.
    gates = {"baseline_masked_loss": base_loss, "baseline_format_compliance": base_fmt,
             "baseline_tool_call_rate": base_tool}
    metrics.append(a.metrics, {"event": "gate_baseline", **gates})
    metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "sft_metrics.jsonl"), log)

    stop = {"now": False}
    def _stop(signum, frame):
        log(f"signal {signum} -- checkpointing before exit")
        stop["now"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def save(step):
        tmp = a.checkpoint + ".tmp"
        torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__,
                    "stage": "sft", "gates": dict(gates),
                    "loss": (sum(run[-a.log_every:]) / min(len(run), a.log_every)
                             if run else None),
                    **opt.state_dict()}, tmp)
        os.replace(tmp, a.checkpoint)
        log(f"checkpoint saved at step {step:,}")
        s3_sync(a.checkpoint, a.s3_uri)
        metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "sft_metrics.jsonl"), log)

    model.train()
    t0 = time.perf_counter()
    last_ck = t0
    run = []
    for step in range(a.steps):
        inp, tgt, msk = next(batches)
        loss = masked_cross_entropy(model(inp, n_loops=cfg.n_loops), tgt, msk)
        loss.backward()
        opt.set_lr(lr_at(step, a.steps, a.lr, a.warmup))
        opt.step()                      # clips, steps, mirrors masters -> bf16
        opt.zero_grad()
        run.append(loss.item())

        if (step + 1) % a.log_every == 0:
            el = time.perf_counter() - t0
            metrics.append(a.metrics, {
                "event": "train", "step": step + 1,
                "loss": round(sum(run[-a.log_every:]) / a.log_every, 5),
                "lr": lr_at(step, a.steps, a.lr, a.warmup),
                "seq_per_s": round((step + 1) * a.batch_size / el, 2)})
            log(f"step {step+1:,}/{a.steps:,} | loss {sum(run[-a.log_every:])/a.log_every:.4f} "
                f"| {(step+1)*a.batch_size/el:.1f} seq/s "
                f"| lr {lr_at(step, a.steps, a.lr, a.warmup):.2e}")

        if (step + 1) % a.eval_every == 0:
            hl = evaluate()
            fmt = format_compliance(model, tok, eos_id, n_loops=cfg.n_loops)
            trate = tool_call_rate(model, tok, eos_id, n_loops=cfg.n_loops)
            verdict = "PASS" if hl < base_loss else "NOT YET"
            log(f"GATE @ step {step+1:,}: held-out {hl:.4f} (base {base_loss:.4f}) [{verdict}] "
                f"| format compliance {fmt:.0%} (base {base_fmt:.0%})")
            gates.update(step=step + 1, masked_loss=hl, format_compliance=fmt,
                         tool_call_rate=trate, verdict=verdict)
            log(f"       tool-call rate {trate:.0%} (base {base_tool:.0%})")
            metrics.append(a.metrics, {"event": "gate", "step": step + 1,
                                       "masked_loss": hl, "format_compliance": fmt,
                                       "verdict": verdict})

        if stop["now"] or time.perf_counter() - last_ck >= a.checkpoint_every_s:
            save(step + 1)
            last_ck = time.perf_counter()
            if stop["now"]:
                log("exiting cleanly after checkpoint")
                sys.exit(0)

    save(a.steps)
    hl = evaluate()
    fmt = format_compliance(model, tok, eos_id, n_loops=cfg.n_loops)
    trate = tool_call_rate(model, tok, eos_id, n_loops=cfg.n_loops)
    log(f"SFT complete. held-out masked loss {base_loss:.4f} -> {hl:.4f} "
        f"| format compliance {base_fmt:.0%} -> {fmt:.0%} "
        f"| tool-call rate {base_tool:.0%} -> {trate:.0%}")
    final_verdict = ("PASS" if (hl < base_loss and fmt > 0) else
                     "FAIL -- GRPO on this checkpoint will produce all-zero rewards")
    log("GATE TO GRPO: " + final_verdict)
    gates.update(final_masked_loss=hl, final_format_compliance=fmt,
                 final_tool_call_rate=trate, final_verdict=final_verdict)
    metrics.append(a.metrics, {"event": "gate_final", **gates})
    metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "sft_metrics.jsonl"), log)
    save(a.steps)     # re-save so the final gates are inside the checkpoint too


if __name__ == "__main__":
    main()
