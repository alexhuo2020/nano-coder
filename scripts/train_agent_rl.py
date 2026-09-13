"""Stage 4: agentic tool-use RL (multi-turn).

Same verifiable reward as GRPO, but the policy now runs a LOOP: it may call
run_python / run_tests, read the real output, and revise before answering. That
is what separates a coding agent from a code completer, and it is the last stage
of the pre-training -> post-training -> agentic-training pipeline this project
exists to demonstrate end to end.

CREDIT ASSIGNMENT, STATED PLAINLY. The reward is known only at the END of an
episode (did the final answer pass the tests), so this broadcasts the episode's
group-relative advantage to EVERY assistant turn in that episode and treats each
turn as its own GSPO sample. That is the simple, standard choice, and it is
biased: a turn that was irrelevant to the outcome gets the same credit as the one
that fixed the bug. Per-turn value estimation would sharpen it, and is
deliberately out of scope here rather than half-implemented -- with a model this
small, terminal-reward broadcasting is not the binding constraint.

The reward is also terminal-only ON PURPOSE (see agent.episode_reward): paying
for intermediate tool successes is trivially farmed by calling a passing tool in
a loop and never answering.
"""

# Run from anywhere: put the repo root on sys.path so this works without
# the caller having set PYTHONPATH. Aliased imports keep it independent of
# whatever the module imports below.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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

from nanocoder.agent import ToolBox, episode_reward, run_episode
from nanocoder.repo_agent import run_repo_episode
from nanocoder.scenario import build_scenarios
from nanocoder.checkpoint import load_stage_checkpoint
from nanocoder.generate import completion_logprobs
from nanocoder.grpo import GRPOConfig, group_advantages, grpo_objective
from nanocoder import metrics
from nanocoder.model import BlackwellLM, ModelConfig
from nanocoder.optim import MasterWeightOptimizer, frozen_report
from nanocoder.sandbox import posix_limits_active
from nanocoder.tasks import get_tasks
from nanocoder.tokenizer import EOS, load_tokenizer


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--init-from", required=True, help="GRPO (or SFT) checkpoint")
    p.add_argument("--task-set", default="mbpp",
                   choices=["mbpp", "easy", "repo-mbpp", "repo-easy"],
                   help="repo-* materialises each task as a scratch REPO with "
                        "verified-broken code, and the agent must read/edit/verify "
                        "with MCP-shaped tools. The reward is then the repo's file "
                        "state, not text extracted from the transcript.")
    p.add_argument("--task-limit", type=int, default=None)
    p.add_argument("--difficulty", default="mutate",
                   choices=["mutate", "multi", "stub", "swap"],
                   help="repo modes: how solution.py is broken. mutate is one "
                        "regex substitution and may be FLATTERING -- stub and "
                        "swap cannot be solved by spotting an odd token.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group-size", type=int, default=6, help="episodes per task")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    p.add_argument("--checkpoint", default="agent_rl.pt")
    p.add_argument("--s3-uri", default=os.environ.get("BLACKWELL_AGENT_S3_URI"))
    p.add_argument("--checkpoint-every-s", type=float, default=900)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--tool-timeout", type=float, default=6.0)
    p.add_argument("--verify-bonus", type=float, default=0.0,
                   help="repo mode: small bonus when the agent VERIFIED with "
                        "run_tests and the file genuinely passes. Without it, "
                        "verifying earns nothing and costs a turn, so RL learns "
                        "to skip it -- measured: ran_tests fell 79%% -> 1%%.")
    p.add_argument("--allow-weak-sandbox", action="store_true",
                   help="run even without POSIX resource limits (NOT for real runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics", default="agent_metrics.jsonl",
                   help="durable JSONL of rewards/updates; mirrored to S3. An RL stage whose results live only on a spot instance has no results.")
    a = p.parse_args()

    assert torch.cuda.is_available(), "agentic RL requires a CUDA device"
    if not posix_limits_active() and not a.allow_weak_sandbox:
        raise SystemExit(
            "refusing to run: POSIX resource limits are unavailable, so the sandbox "
            "degrades to timeout-only. This stage executes thousands of pieces of "
            "model-invented code; a single unbounded allocation takes the box down. "
            "Run on Linux, or pass --allow-weak-sandbox if you accept that."
        )

    tok = load_tokenizer(a.tokenizer)
    eos_id = tok.token_to_id(EOS)
    gcfg = GRPOConfig(group_size=a.group_size, kl_beta=a.kl_beta,
                      update_epochs=a.update_epochs, max_new_tokens=a.max_new_tokens)

    ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    model = BlackwellLM(cfg, precision=a.precision, device="cuda", dtype=torch.bfloat16)
    load_stage_checkpoint(a.init_from, model)
    log(f"policy from {a.init_from} (stage={ck.get('stage')}, step={ck.get('step')}) "
        f"| {model.n_params()/1e6:.1f}M params | precision={model.precision}")

    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    # lr 5e-7 is 0.004x the bf16 spacing -- the most extreme case in the
    # pipeline. FP32 masters are what make this stage able to learn anything.
    opt = MasterWeightOptimizer(model, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                grad_clip=a.grad_clip)
    log(f"optimizer: {frozen_report(model)}")

    repo_mode = a.task_set.startswith("repo-")
    base_set = a.task_set.replace("repo-", "")
    tasks = get_tasks(base_set, limit=a.task_limit)
    if repo_mode:
        # Built ONCE and reused: verifying a scenario costs two real subprocesses
        # (reference must pass, mutation must fail), and rebuilding every epoch
        # would make the data pipeline the bottleneck.
        tasks = build_scenarios(tasks, seed=a.seed, limit=a.task_limit,
                                timeout=a.tool_timeout,
                                difficulty=a.difficulty)
    toolbox = ToolBox(timeout=a.tool_timeout)
    log(f"task set {a.task_set!r}: {len(tasks)} "
        f"{'repo scenarios' if repo_mode else 'tasks'} | {gcfg.group_size} "
        f"episodes/task | up to {a.max_turns} turns | "
        f"sandbox limits={posix_limits_active()}")
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
                    "stage": "agent_rl", "stats": stats, **opt.state_dict()}, tmp)
        os.replace(tmp, a.checkpoint)
        log(f"checkpoint saved at step {step:,}")
        s3_sync(a.checkpoint, a.s3_uri)
        metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "agent_metrics.jsonl"), log)

    episodes = update_steps = dropped = 0
    tool_calls = tool_errors = 0
    rw_hist, turn_hist = [], []
    t0 = time.perf_counter()
    last_ck = t0
    for step in range(a.steps):
        task = tasks[rng.randrange(len(tasks))]
        if not repo_mode:
            toolbox.for_task(task)  # run_tests uses THIS task's hidden tests
        group, rewards = [], []
        for _ in range(gcfg.group_size):
            if repo_mode:
                # Fresh repo per episode -- a shared directory would let one
                # episode's edits leak into the next and silently correlate
                # rewards the group-relative advantage assumes are independent.
                ep = run_repo_episode(model, tok, task, eos_id,
                                      max_turns=a.max_turns,
                                      max_new_tokens=a.max_new_tokens,
                                      n_loops=cfg.n_loops, record_logprobs=True,
                                      tool_timeout=a.tool_timeout,
                                      verify_bonus=a.verify_bonus)
                rw = ep.reward
            else:
                ep = run_episode(model, tok, task.prompt, toolbox, eos_id,
                                 max_turns=a.max_turns,
                                 max_new_tokens=a.max_new_tokens,
                                 n_loops=cfg.n_loops, record_logprobs=True)
                rw, _ = episode_reward(ep, task.tests, timeout=a.tool_timeout,
                                       setup=task.setup)
            group.append(ep)
            rewards.append(rw)
            episodes += 1
            tool_calls += ep.tool_calls
            tool_errors += ep.tool_errors
            turn_hist.append(ep.turns)
        rw_hist.extend(rewards)

        adv = group_advantages(torch.tensor(rewards))

        # Recorded EVERY step. A long agentic run exists to detect a TREND, and
        # a trend cannot be recovered from a summary printed every 5 steps --
        # especially not after a spot reclaim. Earlier this script only
        # published a metrics file it never wrote a single row to.
        metrics.append(a.metrics, {
            "event": "step", "step": step + 1, "task": task.name,
            "rewards": [round(x, 4) for x in rewards],
            "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
            "solved": sum(1 for x in rewards if x >= 1.0),
            "turns": [e.turns for e in group],
            "tool_calls": sum(e.tool_calls for e in group),
            "tool_errors": sum(e.tool_errors for e in group),
            "from_tool_call": sum(1 for e in group
                                  if getattr(e, "final_from_tool_call", False)),
            # repo mode only: did the agent actually EDIT and VERIFY? An agent
            # that never writes a file cannot possibly earn reward, so these
            # separate "cannot" from "tried and failed".
            "wrote_file": sum(1 for e in group if getattr(e, "wrote_file", False)),
            "ran_tests": sum(1 for e in group if getattr(e, "ran_tests", False)),
            "verified_pass": sum(1 for e in group
                                 if getattr(e, "verified_pass", False)),
            "zero_variance": adv is None,
            "cum_updates": update_steps, "cum_dropped": dropped})

        if adv is None:
            dropped += 1
        else:
            # Broadcast each episode's advantage to all of its assistant turns.
            for ep, ep_adv in zip(group, adv.tolist()):
                for tr in ep.steps:
                    if tr.logprobs is None or tr.tokens.numel() == 0:
                        continue
                    plen = len(tr.prompt_ids)
                    seq = torch.cat([
                        torch.tensor(tr.prompt_ids, device=tr.tokens.device)[None],
                        tr.tokens], dim=1)
                    with torch.no_grad():
                        ref_lp = completion_logprobs(ref, plen, seq, n_loops=cfg.n_loops)
                    vf = tr.valid.float()
                    a_t = torch.tensor([ep_adv], device=tr.tokens.device)
                    for _ in range(gcfg.update_epochs):
                        new_lp = completion_logprobs(model, plen, seq, n_loops=cfg.n_loops)
                        loss, _m = grpo_objective(new_lp, tr.logprobs, ref_lp, vf, a_t, gcfg)
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                        update_steps += 1

        if (step + 1) % a.log_every == 0:
            recent = rw_hist[-a.log_every * gcfg.group_size:]
            el = time.perf_counter() - t0
            extra = ""
            if repo_mode:
                wrote = sum(1 for e in group if getattr(e, "wrote_file", False))
                ran = sum(1 for e in group if getattr(e, "ran_tests", False))
                extra = f"| wrote_file {wrote}/{len(group)} ran_tests {ran}/{len(group)} "
            log(f"step {step+1:,}/{a.steps:,} | mean reward "
                f"{sum(recent)/max(1,len(recent)):.4f} {extra}| episodes {episodes} "
                f"| mean turns {sum(turn_hist)/max(1,len(turn_hist)):.2f} "
                f"| tool calls {tool_calls} (errors {tool_errors}) "
                f"| updates {update_steps} | dropped {dropped} "
                f"| {episodes/el*60:.1f} episodes/min")
            if tool_calls == 0 and episodes >= 20:
                log("DIAGNOSTIC: the policy has never emitted a parseable tool call, so "
                    "this is single-turn RL with extra steps. Check that SFT taught the "
                    "```tool convention (see agent.AGENT_SYSTEM) before reading any "
                    "reward trend as agentic learning.")

        if stop["now"] or time.perf_counter() - last_ck >= a.checkpoint_every_s:
            save(step + 1, {"mean_reward": sum(rw_hist) / max(1, len(rw_hist)),
                            "update_steps": update_steps, "tool_calls": tool_calls})
            last_ck = time.perf_counter()
            if stop["now"]:
                log("exiting cleanly after checkpoint")
                sys.exit(0)

    stats = {"mean_reward": sum(rw_hist) / max(1, len(rw_hist)),
             "episodes": episodes, "update_steps": update_steps, "dropped": dropped,
             "tool_calls": tool_calls, "tool_errors": tool_errors,
             "mean_turns": sum(turn_hist) / max(1, len(turn_hist))}
    save(a.steps, stats)
    metrics.append(a.metrics, {"event": "final", **stats})
    metrics.publish(a.metrics, metrics.metrics_uri_for(a.s3_uri, "agent_metrics.jsonl"), log)
    log(f"agentic RL complete: {stats}")


if __name__ == "__main__":
    main()
