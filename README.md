# nano-coder

**A complete, honest, reproducible pipeline for training a coding agent — small
enough to read end to end.**

An 85M-parameter model trained from scratch on one consumer GPU
(pretrain → SFT → GRPO → agentic RL), then wired into **Claude Code** and
**Codex CLI** as a real backend. It repairs broken code in a repository at
**56.7% ±4.5** on tasks it has never seen, and runs on a laptop CPU at 4.5 tok/s.

It is also a record of **four times a number turned out to mean something other
than what it appeared to mean** — which is the part most worth reading.

---

## It works, and here is the proof

Held-out MBPP task 903. Claude Code as the harness; the 85M model as the brain.
No Anthropic or OpenAI model anywhere in the loop.

**Given** — verified to fail its tests:
```python
def count_Unset_Bits(n):
    cnt = 0
    for i in range(1, n + 1):
        temp = i
        while temp:
            if temp % 2 == 0:
                cnt += 1
            temp = temp // 2
    return not cnt      # ← the bug
```

**The model, unaided:**
```
read_file  → Read       reads the real file
write_file → Write      writes the fix
run_tests  → Bash       pytest actually runs
```

**Result:** `3/3 tests passed` — first attempt, on a task never seen in training.

Reproduce it: `python scripts/demo_episode.py sft_ctl.pt --tier mutate --holdout 74`

---

## What it can and cannot do

| task | held-out |
|---|---|
| repair a wrong operator in a repo | **56.7% ±4.5** |
| repair two wrong tokens | **55.8% ±4.5** |
| write a function body from scratch | **~0–7.5%** |
| single-turn code generation | **6.4% ±1.4** pass@1 |
| emit a well-formed tool call | ~100% |

It is a **narrow repair agent**, not a coding assistant. It will not build you a
feature. Judged against that narrow task it works; judged as a general model it
does not, and this repo says so throughout.

---

## Why it might be useful to you

- **The whole loop is here and it is small.** Every stage is one script you can
  read in an afternoon: `train_pretrain.py`, `train_sft.py`, `train_grpo.py`,
  `train_agent_rl.py`.
- **The agent environment is honest by construction.** Scenarios are *verified
  broken*, the reward reads the **file** the agent leaves behind rather than its
  transcript, and the **harness owns the tests** — the model was caught inventing
  its own and passing them.
- **Four difficulty tiers.** `mutate` (revert one token) flatters a model badly;
  `stub` and `swap` cannot be solved by spotting an odd character. Evaluating on
  one tier is how this project produced a 70% that was really 0%.
- **A working CLI integration.** `serve/` translates between a coding CLI's
  protocol and a small model's trained conventions — including the traps that
  make a correct model look broken.
- **The negative results are kept.** Including the ones that embarrass the author.

---

## Quickstart

### Run the trained model locally (no GPU)

```bash
python -m venv ~/bn                     # short path: see serve/README.md on Windows MAX_PATH
~/bn/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/bn/bin/pip install tokenizers numpy pytest

~/bn/bin/python serve/anthropic_shim.py \
  --ckpt local/sft_ctl.pt --tokenizer local/tokenizer.json \
  --device cpu --port 8799 --context 4096 \
  --claude-code --truncate --chat-passthrough \
  --temperature 0.2 --cwd /path/to/your/repo
```

Then point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8799
export ANTHROPIC_AUTH_TOKEN=dummy-local
export ANTHROPIC_MODEL=nano-coder-85m
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000     # must be LARGE — see serve/README.md
claude -p "The file solution.py in this repo is failing its tests. Read it, fix it, and verify with run_tests."
```

### Train your own

```bash
python scripts/train_sft.py --init-from BASE.pt --checkpoint sft.pt \
  --tool-mode repo --tool-difficulty mutate --tool-holdout 74 \
  --tool-frac 0.5 --steps 4000 --max-len 1536

python scripts/scoreboard.py sft.pt --holdout 74 --temperature 0.2
```

`--tool-holdout` and `--holdout` **must match**, or the scoreboard reports train
accuracy. That mistake inflated this project's headline from 23.5% to 78.5%.

### Run the tests

```bash
python scripts/test_repo_agent_cpu.py     # environment, reward, containment
python serve/test_cli_adapter.py          # Claude Code protocol
python serve/test_codex_adapter.py        # OpenAI Responses protocol
python serve/test_chat_routing.py         # chat vs agent prompt routing
```

All CPU-only; no GPU required.

---

## Documentation

| doc | what's in it |
|---|---|
| **[docs/LESSONS.md](docs/LESSONS.md)** | **start here** — what this project actually taught, and what it cost |
| [docs/PIPELINE.md](docs/PIPELINE.md) | every stage, the real commands, and the flags that matter |
| [docs/RESULTS.md](docs/RESULTS.md) | all measurements, including the retracted ones |
| [docs/HARDWARE.md](docs/HARDWARE.md) | NVFP4, weight sharing, Transformer Engine on `sm_120` |
| [serve/README.md](serve/README.md) | serving to Claude Code / Codex, and the traps |
| [report/report.pdf](report/report.pdf) | 11-page write-up, figures generated from `runs/metrics/` |

---

## The four defaults

The single most transferable thing here. Each was chosen because it was the path
of least resistance, and each silently became a *definition*:

| the default | it produced | the truth |
|---|---|---|
| benchmark tier = `mutate` | **70%** solve rate | measuring "revert the odd token" |
| training tier = `mutate` | **0/240**, "capability ceiling" | demos taught copying |
| sampling `T = 1.0` | **1.0%** on MBPP | 6.4% at T=0.2 — 1.0 is for *exploration* |
| no train/test split | **78.5%** mean | **23.5%** held out; the rest was memorisation |

Two made the model look better than it was; two made it look worse. An
unexamined default is not biased toward flattery — it is simply unmeasured.

**Before believing a number, ask what the evaluation could not have
distinguished.**

---

## Repository layout

```
nanocoder/     model, tokenizer, data, chat format, sandbox
                  optim.py      fp32 master weights (bf16 params lose updates)
                  scenario.py   verified-broken repo tasks, 4 difficulty tiers
                  repo_agent.py multi-turn episodes; reward reads the FILE
                  mcp_tools.py  4 tools; the system prompt is GENERATED from them
scripts/          train_* (pretrain, sft, grpo, agent_rl)
                  scoreboard.py, mbpp_eval.py, demo_episode.py
                  test_*_cpu.py
serve/            anthropic_shim.py  one server, two protocols
                  cli_adapter.py     Claude Code translation
                  codex_adapter.py   OpenAI Responses translation
runs/metrics/     per-step JSONL from every real run
runs/results/     probe logs and transcripts behind every number
report/           LaTeX source + PDF; figures generated from runs/metrics/
```

---

## Status and honest limits

- **The pipeline is complete and validated on real hardware.** It was built to be
  pointed at a larger base model; that is the recommended next step.
- **The model is small and weak at code generation** (6.4% pass@1). It is useful
  for the one task it was trained on.
- **RL added nothing over good demonstrations here.** At ~10% pass@1 with 8
  episodes per group, ~75% of groups are zero-variance. A curriculum sampling
  tasks with known non-zero pass@k would fix that; it is untested.
- **The sandbox is a robustness boundary, not a containment boundary** — no
  network, mount or PID namespace. Run the trainer in a container with no egress
  if that matters.
- **Codex CLI support is implemented and tested but blocked on Windows** where
  `codex doctor` reports `sandbox backend: disabled`.
- **One unexplained defect:** the server occasionally logs an implausible
  per-request duration. Treat per-request timings in the log as unreliable.

---

## Acknowledgements

Weight-shared depth follows Universal Transformers and ALBERT; the RL recipe
borrows GSPO's sequence-level ratio, DAPO's Clip-Higher, and Schulman's k3 KL
estimator. Tasks come from MBPP. Built with [Claude Code](https://claude.com/claude-code).
