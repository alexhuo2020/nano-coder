# The pipeline, stage by stage

Every stage ran on real hardware. Commands are the ones actually used, not
idealised versions.

```
tokenizer → pretrain → SFT → GRPO → agentic RL → evaluate → serve
```

---

## 0. Tokenizer

32,784-token BPE, trained on the same corpus mixture as pretraining.

```bash
python scripts/train_tokenizer.py --vocab-size 32784 --out tokenizer.json
```

> **The tokenizer is load-bearing.** A *different* tokenizer silently
> invalidates every checkpoint — weights load without error and produce
> nonsense. Publish the exact one a run used alongside its weights.

---

## 1. Pretraining

85M parameters, `d_model=1536`, **one transformer block applied 8 times**
(weight-shared depth), 2,048-token context, NVFP4/BF16 via Transformer Engine.

```bash
python scripts/train_pretrain.py --seq-len 2048 --batch-size 16 \
  --checkpoint pretrain.pt --metrics metrics.jsonl
```

20.40B tokens (240 tokens/param — deliberately past Chinchilla, because the
downstream stages needed the strongest base at a fixed small size).

| checkpoint | step | held-out CE |
|---|---|---|
| baseline | 656,711 | 2.1817 |
| mid | 910,397 | 2.1325 |
| **final** | **1,245,117** | **2.1155** |

**Two findings that changed everything downstream:** bf16 params cannot hold
optimizer state (→ `blackwell_lm/optim.py` keeps fp32 masters), and flat loss at
high LR is not saturation. See [LESSONS.md](LESSONS.md) §8–9.

---

## 2. Supervised fine-tuning

Loss is masked to **assistant tokens only** — training on the user's turns
teaches the model to write the questions, which shows up at generation time as
it inventing a user turn and answering itself.

```bash
python scripts/train_sft.py \
  --init-from pretrain.pt --checkpoint sft.pt \
  --tool-mode repo --tool-difficulty mutate \
  --tool-frac 0.5 --tool-task-set mbpp --tool-holdout 74 \
  --steps 4000 --batch-size 8 --max-len 1536 --lr 1e-5
```

| flag | why it matters |
|---|---|
| `--tool-difficulty` | **decides what the demos teach.** Accepts a mixture (`stub,swap,mutate`). The default caused a 0/240 misdiagnosed as a capability ceiling. |
| `--tool-holdout N` | reserves the last N tasks. **Without it the eval reports train accuracy.** |
| `--tool-retry-frac` | share of demos that are *wrong fix → failing tests → correct fix*. |
| `--max-len` | tool trajectories over this are **dropped, not truncated** — truncation keeps the head, so a retry demo would keep the wrong fix and lose the correction. |

Result: masked CE 1.8564 → 0.9830, chat-format compliance 0% → 100%,
tool-call rate 0% → 100%.

---

## 3. GRPO with execution-verified rewards

Rewards come from **running the code against hidden tests** — never a learned
reward model. Sequence-level GSPO ratios, Schulman k3 KL, DAPO Clip-Higher,
zero-variance group dropping.

```bash
python scripts/train_grpo.py --init-from sft.pt --task-set mbpp \
  --group-size 8 --kl-beta 0.02 --lr 5e-7
```

| task set | rollouts | update steps | fully solved |
|---|---|---|---|
| control (12 tasks) | 40 | 76 | 30 |
| MBPP (300 tasks) | 300 | 52 | 9/300 |

The control set is what separates *a pipeline defect* from *a capability limit*.
The same code takes 76 update steps on solvable tasks and 52 on MBPP because 91%
of MBPP groups are zero-variance — a model limit, and the control proves it is
not a bug.

---

## 4. Agentic RL over a real repository

Each task becomes a directory holding a **verified-broken** `solution.py`. Four
MCP-shaped tools (`list_dir`, `read_file`, `write_file`, `run_tests`) declared
with real JSON Schema, and **the system prompt is generated from those
declarations** so it cannot drift from the dispatcher.

```bash
python scripts/train_agent_rl.py \
  --init-from sft.pt --task-set repo-mbpp --difficulty stub \
  --group-size 8 --max-turns 5 --verify-bonus 0.1
```

Three properties that make the environment trustworthy rather than merely
functional:

1. **Breakage is verified both ways** — the reference must pass and the broken
   file must fail, in the sandbox, before a scenario is used. Without both, an
   unsolvable task and a broken policy look identical.
2. **The reward reads the FILE, not the transcript.** An agent that edits
   correctly and says nothing scores full marks; one that describes a perfect
   fix without writing it scores zero.
3. **The harness owns the grader.** The model was caught inventing its own tests
   and passing them.

> **Honest result:** on this model RL added nothing over good demonstrations.
> From an SFT checkpoint at 9.6% pass@1, 63 steps / 504 episodes / 782 update
> steps produced no improvement. At ~10% pass@1 with 8 episodes per group, ~75%
> of groups are zero-variance and contribute no gradient. A curriculum that
> samples tasks with known non-zero pass@k would recover most of that budget.

---

## 5. Evaluation

**Always all four tiers, always held out, always with standard errors.**

```bash
python scripts/scoreboard.py sft_ctl.pt \
  --holdout 74 --tasks 20 --samples 6 --temperature 0.2
```

```bash
python scripts/mbpp_eval.py sft_ctl.pt --holdout 74 --temperature 0.2
```

| tier | what is broken | can it be solved by spotting an odd token? |
|---|---|---|
| `mutate` | one regex substitution | yes — **flattering** |
| `multi` | two substitutions | yes |
| `stub` | body replaced by `pass` | **no** |
| `swap` | a different task's real solution | **no** |

`--holdout` must match `--tool-holdout`. Sweep `--temperature`: 1.0 is the RL
rollout default and understates solving roughly six-fold.

---

## 6. Serving

See [../serve/README.md](../serve/README.md). Runs on a GPU box or **locally on
CPU** (4.5 tok/s, ~340 MB RAM, no Transformer Engine needed).

---

## Reproducing on one GPU

Wall clock from the real runs on a single 32 GB consumer GPU:

| stage | wall clock | note |
|---|---|---|
| pretrain 20.4B tokens | ~3 days | by far the longest |
| SFT | ~10 min | 4,000 steps |
| GRPO | ~1 h | |
| agentic RL | ~2 h | 400 steps × 8 episodes |
| scoreboard (4 tiers) | ~25 min | 120 episodes/tier |

**Skip pretraining** and start from a released base model if you only want the
post-training stages — that is what the pipeline was built to be pointed at,
and it turns a three-day run into an afternoon.
