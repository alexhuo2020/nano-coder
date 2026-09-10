"""Stage 3: GRPO with verifiable execution rewards (single-turn).

The policy writes code, the code is RUN against asserts, and the fraction of
asserts that pass is the reward. No reward model, no human labels.

WHAT THIS SCRIPT REPORTS AND WHY. The headline number is not the loss -- a
policy-gradient loss is not comparable across steps and reading it as progress
is a mistake. The numbers that matter are:

  * mean reward / pass rate -- the thing being optimised;
  * UPDATE STEPS vs ROLLOUT STEPS -- their ratio is the diagnostic that a
    previous run in this lineage needed. Every group had identical rewards, so
    every group was correctly dropped as zero-variance, so the model received
    ZERO updates for hours while the loop happily printed rollout statistics.
    If update_steps stays 0, no amount of tuning matters and the fix is a
    stronger base model or an easier task set (--task-set easy).
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import signal
import sys
import time

os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

import torch

from blackwell_lm import chat
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.generate import completion_logprobs, generate
from blackwell_lm.grpo import GRPOConfig, group_advantages, grpo_objective
from blackwell_lm import metrics
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.optim import MasterWeightOptimizer, frozen_report
from blackwell_lm.reward import code_reward
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, load_tokenizer


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def rollout(model, tok, task, eos_id, gcfg, n_loops):
    """G on-policy completions for one task, with rewards.

    Sampling is at temperature 1.0 / top_p 1.0 -- see generate(): a truncated
    sampler makes the behaviour policy differ from the model and biases the
    importance ratio.
    """
    prompt_ids = chat.tokenize_prompt(tok, chat.user_turn(task.prompt))
    toks, valid, old_lp = generate(
        model, prompt_ids, max_new_tokens=gcfg.max_new_tokens, eos_id=eos_id,
        num_return_sequences=gcfg.group_size, n_loops=n_loops, return_logprobs=True,
    )
    rewards, details = [], []
    for r in range(toks.shape[0]):
        ids = [int(t) for t, v in zip(toks[r].tolist(), valid[r].tolist()) if v]
        text = tok.decode(ids)
        rw, detail = code_reward(text, task.tests, setup=task.setup)
        rewards.append(rw)
        details.append(detail)
    return prompt_ids, toks, valid, old_lp, torch.tensor(rewards, device=toks.device), details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--init-from", required=True, help="SFT checkpoint")
    p.add_argument("--task-set", default="mbpp", choices=["mbpp", "easy"])
    p.add_argument("--task-limit", type=int, default=None)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.28)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # BF16, not FP4. A ratio-and-KL objective is far more precision-sensitive
    # than a cross-entropy: in bf16 the k3 estimator already went NEGATIVE in a
    # previous run, and FP4 for RL is a separate research question, not an
    # assumption to smuggle into a pipeline that is being validated.
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    p.add_argument("--checkpoint", default="grpo.pt")
    p.add_argument("--s3-uri", default=os.environ.get("BLACKWELL_GRPO_S3_URI"))
    p.add_argument("--checkpoint-every-s", type=float, default=900)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics", default="grpo_metrics.jsonl",
                   help="durable JSONL of rewards/updates; mirrored to S3. An RL stage whose results live only on a spot instance has no results.")
    a = p.parse_args()

    assert torch.cuda.is_available(), "GRPO requires a CUDA device"
    tok = load_tokenizer(a.tokenizer)
    eos_id = tok.token_to_id(EOS)
    gcfg = GRPOConfig(group_size=a.group_size, clip_low=a.clip_low, clip_high=a.clip_high,
                      kl_beta=a.kl_beta, update_epochs=a.update_epochs,
                      max_new_tokens=a.max_new_tokens)

    ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    model = BlackwellLM(cfg, precision=a.precision, device="cuda", dtype=torch.bfloat16)
    load_stage_checkpoint(a.init_from, model)
    log(f"policy loaded from {a.init_from} (stage={ck.get('stage')}, step={ck.get('step')}) "
        f"| {model.n_params()/1e6:.1f}M params | precision={model.precision}")

    # Frozen reference for the KL term. A deep copy rather than a second load so
    # it is provably the same starting weights as the policy.
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)

    # At lr 1e-6 an AdamW update is 0.008x the bf16 spacing, so without FP32
    # masters only ~2% of elements can move at all (measured; 15% with masters).
    # GRPO would run every rollout, score every reward and compute every
    # advantage while training on a sliver of the model -- easily mistaken for
    # the far more famous zero-variance-group cause of "no progress".
    opt = MasterWeightOptimizer(model, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                grad_clip=a.grad_clip)
    log(f"optimizer: {frozen_report(model)}")

    tasks = get_tasks(a.task_set, limit=a.task_limit)
    log(f"task set {a.task_set!r}: {len(tasks)} tasks | group size {gcfg.group_size} "
        f"| {gcfg.update_epochs} update epochs")
    rng = random.Random(a.seed)

    stop = {"now": False}
    def _stop(signum, frame):
        log(f"signal {signum} -- checkpointing before exit")
        stop["now"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def save(step, stats):
        tmp = a.checkpoint + ".tmp"
        torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__,
                    "stage": "grpo", "stats": stats, **opt.state_dict()}, tmp)
        os.replace(tmp, a.checkpoint)
        log(f"checkpoint saved at step {step:,}")
        s3_sync(a.checkpoint, a.s3_uri)
        metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "grpo_metrics.jsonl"), log)

    rollout_steps = update_steps = dropped = 0
    rw_hist, solved = [], 0
    t0 = time.perf_counter()
    last_ck = t0
    for step in range(a.steps):
        task = tasks[rng.randrange(len(tasks))]
        prompt_ids, toks, valid, old_lp, rewards, _ = rollout(
            model, tok, task, eos_id, gcfg, cfg.n_loops)
        rollout_steps += 1
        rw_hist.extend(rewards.tolist())
        solved += int((rewards >= 1.0).any().item())

        adv = group_advantages(rewards)
        if adv is None:
            dropped += 1
        else:
            plen = len(prompt_ids)
            seq = torch.cat([
                torch.tensor(prompt_ids, device=toks.device)[None].expand(toks.shape[0], -1),
                toks], dim=1)
            with torch.no_grad():
                ref_lp = completion_logprobs(ref, plen, seq, n_loops=cfg.n_loops)
            vf = valid.float()
            for _ in range(gcfg.update_epochs):
                new_lp = completion_logprobs(model, plen, seq, n_loops=cfg.n_loops)
                loss, met = grpo_objective(new_lp, old_lp, ref_lp, vf, adv, gcfg)
                loss.backward()
                opt.step()
                opt.zero_grad()
                update_steps += 1

        # Recorded EVERY step, not every log interval: the ratio of update_steps
        # to rollout_steps is the diagnostic that distinguishes "RL is broken"
        # from "the policy is too weak", and losing it to a reclaim would mean
        # re-running hours of rollouts to answer that question again.
        metrics.append(a.metrics, {
            "event": "rollout", "step": step + 1,
            "task": task.name, "rewards": [round(x, 4) for x in rewards.tolist()],
            "mean_reward": round(float(rewards.mean()), 4),
            "zero_variance": adv is None,
            "update_steps": update_steps, "dropped": dropped})

        if (step + 1) % a.log_every == 0:
            recent = rw_hist[-a.log_every * gcfg.group_size:]
            el = time.perf_counter() - t0
            log(f"step {step+1:,}/{a.steps:,} | mean reward {sum(recent)/max(1,len(recent)):.4f} "
                f"| fully solved {solved}/{rollout_steps} "
                f"| updates {update_steps} | zero-variance groups dropped {dropped} "
                f"| {(step+1)/el*60:.1f} rollouts/min")
            if update_steps == 0 and rollout_steps >= 20:
                log("DIAGNOSTIC: 0 update steps after 20 rollouts -- every group scored "
                    "identically, so there is no preference signal at all. This is NOT an "
                    "RL hyperparameter problem. Use --task-set easy to confirm the code "
                    "path works, then improve the base checkpoint.")

        if stop["now"] or time.perf_counter() - last_ck >= a.checkpoint_every_s:
            save(step + 1, {"mean_reward": sum(rw_hist) / max(1, len(rw_hist)),
                            "update_steps": update_steps, "dropped": dropped})
            last_ck = time.perf_counter()
            if stop["now"]:
                log("exiting cleanly after checkpoint")
                sys.exit(0)

    stats = {"mean_reward": sum(rw_hist) / max(1, len(rw_hist)),
             "update_steps": update_steps, "dropped": dropped,
             "fully_solved": solved, "rollouts": rollout_steps}
    save(a.steps, stats)
    metrics.append(a.metrics, {"event": "final", **stats})
    metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "grpo_metrics.jsonl"), log)
    log(f"GRPO complete: {stats}")
    log("GATE TO AGENTIC RL: " + ("PASS" if update_steps > 0 else
        "FAIL -- zero update steps means the policy never received a gradient"))


if __name__ == "__main__":
    main()
