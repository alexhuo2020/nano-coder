# Hardware findings: what an RTX PRO Blackwell actually rewards

Measured on one GPU. Extracted from the original project README; the
NVFP4, weight-sharing and Transformer Engine results are here, and the
training/agent results are in [RESULTS.md](RESULTS.md).

## What the hardware actually rewards

### What actually governs NVFP4's win: the contraction dimension

Three axes swept independently on one GPU. Only one of them matters.

**Axis 1 - d_model** (ffn = 4x d_model, seq 2048):

| d_model | fp8 tok/s | nvfp4 tok/s | ratio |
|---------|-----------|-------------|-------|
| 768  | 76,070 | 76,117 | 1.001x |
| 1024 | 43,762 | 42,583 | **0.973x - FP4 is a LOSS** |
| 2048 | 18,076 | 19,912 | 1.102x |
| 3072 | 12,854 | 15,769 | **1.227x** |
| 4096 | 10,728 | 12,993 | 1.211x |
| *pure 4096- GEMM* | *214.5 TFLOP/s* | *320.1 TFLOP/s* | *1.49x* |

**Axis 2 - batch (tokens per GEMM).** The obvious hypothesis is that FP4 needs a
compute-bound GEMM, so a bigger batch should unlock it. It does not:

| d_model | tokens/batch | fp8 | nvfp4 | ratio |
|---------|--------------|-----|-------|-------|
| 768 | 16,384 | 83,244 | 83,338 | 1.0011x |
| 768 | 32,768 | 81,068 | 81,180 | 1.0014x |

Doubling M moved the ratio by 0.0003. **Batch size does not unlock FP4**, which
also means the 96GB card's memory is not the lever here.

**Axis 3 - FFN width, at fixed d_model.** Every sweep above had ffn = 4x d_model,
confounding "d_model matters" with "GEMM output width matters". Separating them:

| d_model | ffn | ffn/d_model | ratio |
|---------|-----|-------------|-------|
| 768 | 3072 | 4x | 1.010x |
| 768 | 6144 | 8x | 1.033x |
| 768 | 12288 | 16x | 1.035x |
| 768 | 24576 | 32x | **1.036x - saturated** |
| 1536 | 12288 | 8x | **1.126x** |

Widening the FFN **8x** bought 1.010 -> 1.036 and then flatlined. Raising
d_model from 768 to 1536 at the *same* ffn jumped 1.035 -> 1.126.

**Conclusion: NVFP4's benefit is governed by d_model - the GEMM's contraction
(K) dimension - not by the output width (N) and not by the batch (M).** That
matches how NVFP4 is built: its block scales live along K in 16-element groups,
so a small K carries more scale-factor overhead per unit of compute. The
practical rule is a single number: **NVFP4 needs d_model >= 3072.**

### Looping is how you afford d_model 3072 - and it is also faster

d_model 3072 costs ~134M parameters per block, so dense depth 8 is a 1.2B model:
untrainable to competence on one GPU. One *shared* block applied 8 times has
depth-8 compute for one block's parameters. At matched effective depth and
matched FLOPs, varying only whether blocks are shared:

| d_model | layout | params | fp8 | nvfp4 | ratio |
|---------|--------|--------|-----|-------|-------|
| 3072 | dense 8x1 | **1208M** | 16,968 | 20,299 | 1.196x |
| 3072 | **looped 1x8** | **239M** | 17,689 | **21,436** | **1.212x** |
| 1536 | dense 8x1 | 327M | 49,147 | 52,903 | 1.076x |
| 1536 | looped 1x8 | 85M | 50,774 | 55,047 | 1.084x |
| 768 | dense 8x1 | 94M | 112,991 | 115,171 | 1.019x |
| 768 | looped 1x8 | 34M | 114,370 | 116,553 | 1.019x |

**5.1x fewer parameters and 5.6% faster.** The speed is a bandwidth effect: one
weight set reused 8 times stays resident in cache, instead of streaming eight
distinct blocks from DRAM - the axis this hardware is weakest on. Looping also
lifts the FP4 ratio slightly (1.196 -> 1.212 at d=3072), consistent with FP4's
per-weight quantization cost being amortised across the 8 reuses, though that
effect is small (+1.6pp) next to the parameter and bandwidth wins.

Weight-shared depth is not new (Universal Transformers, ALBERT; recently
recurrent-depth latent reasoning and Mixture-of-Recursions). What is specific
here is *why* it is the right choice on this hardware: it is the only way to
reach the d_model where Blackwell's 4-bit path pays while staying small enough
to train, and it independently reduces the weight traffic that bounds this GPU.

It also buys **test-time compute scaling for free**: `forward(..., n_loops=N)`
overrides the loop count, so a hard prompt can be given more depth without
touching the weights.

### If you do use NVFP4, the released Transformer Engine wheel is silently broken

On sm_120 (RTX Blackwell), the stock `transformer_engine` wheel is built for
plain `sm_120`, so the FP4 stochastic-rounding cast has no valid PTX. Every cast
prints, **per CUDA thread**:

```
FP4 cvt PTX instructions are architecture-specific.
Try recompiling with sm_XXXa instead of sm_XXX.
```

Nothing raises. `autocast` succeeds, outputs are finite, gradients flow - and
you train with *biased* 4-bit rounding while believing otherwise. The per-thread
device printf also makes end-to-end training unmeasurably slow (75,215 log lines
in one partial run); redirecting stderr does not help, because the device-side
printf itself costs the time.

The fix is to build TE for the architecture-accelerated target:

```bash
NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120a NVTE_WITH_NCCL_EP=0 MAX_JOBS=8 \
  pip install --no-build-isolation \
  "transformer_engine[pytorch] @ git+https://github.com/NVIDIA/TransformerEngine.git@main"
```

Four traps in that one command, all hit and solved:
- `#egg=pkg[extra]` is invalid in modern pip - use the PEP 508 `pkg[extra] @ git+...` form.
- TE's bundled 3rd-party NCCL fails on CUDA 13 headers (`ncclDevCommRequirements has no member worldGinBarrierCount`) - `NVTE_WITH_NCCL_EP=0` skips it, and TE's own error text suggests it.
- cmake needs an unversioned `libcudnn.so`; the `nvidia-cudnn-cu13` wheel ships only `libcudnn.so.9`. Symlink it.
- `nvidia-cudnn-frontend` must be installed or cmake configuration fails.

### Hopper-tuned kernels do not fit

Blackwell workstation parts expose **99KB** of shared memory per block; Hopper
exposes ~227KB. That single number is why highly-tuned Hopper code does not
simply run here - porting `modded-nanogpt` to this GPU required fixing a
hardcoded `compute_capability="90"`, discovering FlashAttention-3 ships no
sm_120 kernels at all, and shrinking an MLP kernel's TMA tiles from
`BLOCK_N=256, num_stages=4` (196KB) to fit 99KB.

Once ported it reached 57,477 tok/s, later 62,603 with a newer toolchain - while
the clean model in this repo does **83,177 tok/s** on the same card. A simple
architecture that suits the hardware beat a far more sophisticated one tuned for
different hardware.

### Never hand SDPA an explicit mask

Sliding-window attention implemented as an SDPA boolean `attn_mask` measured
**58,653 tok/s**; the same model with `is_causal=True` measured **83,177**. A
**1.42x penalty**, because any explicit mask disables FlashAttention. Rewritten
with FlexAttention and a cached `BlockMask`, the windowed path recovers to
**81,132** - a fused kernel *and* skipped blocks.

This bug is easy to ship by accident: it looks like an optimization.

## Sizing: why 129M

Throughput falls roughly as 1/params, so a fixed compute budget trades size
against tokens. On one PRO 6000 for 30 days:

| params | tok/s | 30-day tokens | tok/param | verdict |
|--------|-------|---------------|-----------|---------|
| 0.13B | 143,564 | 372B | 2,880 | over-trained for the budget |
| 0.44B | 56,949 | 148B | 335 | good |
| 1.50B | 23,940 | 62B | 41 | badly undertrained |
| 3.00B | 14,669 | 38B | 13 | badly undertrained |

Undertraining is not a mild penalty. A 193M model trained at **20 tok/param**
produced fluent-looking prose that was not valid code in any language, and
scored **0.0000 on 1,400 MBPP reinforcement-learning rollout groups** - meaning
the RL stage could not even start, because zero reward variance yields zero
gradient. Reference: CodeGen-350M-mono writes working Python after ~620
tok/param.

129M at 289 tok/param (3 days) is the point where a single GPU can reach a
token budget that produces coherent output. **Tokens, not architecture, are the
binding constraint** - which is also why this repo does not use a
mixture-of-experts: MoE spends memory bandwidth per FLOP (Blackwell's weak
axis) and inflates parameters ~5x for the same active FLOPs, pushing you
straight back into the undertrained regime.

## Model

```
129M params | d_model 768 | 12 layers | 6 heads / 2 KV heads | head_dim 128
ffn 3072 (4x) | vocab 32768 tied | FP8 linears, BF16 attention core
```

- **head_dim exactly 128**, and every dimension a multiple of 16 - including the
  *fused QKV width*, which is a GEMM dimension too and which a legal-looking GQA
  split can silently make illegal. `ModelConfig.validate()` rejects these rather
  than letting them run slowly.
- **GQA 3:1.** The attention core stays BF16 with no low-precision path, so its
  K/V traffic lands on the weak axis. Fewer KV heads cut it ~3x.
- **Deliberately boring elementwise.** A profile of `modded-nanogpt` here put
  17.6% of the step in elementwise kernels (gating, MUDD skips, per-head
  lambdas) and 13.2% in a large embedding backward - 31% of runtime that three
  separate toolchain upgrades could not move, because it is bandwidth-bound. So:
  RMSNorm, RoPE, SwiGLU, tied embeddings, a modest vocab, nothing clever.
- **KV cache for generation.** Measured **36x** faster generation on this
  hardware. Agentic rollouts are generation-bound, so this is what makes the
  agentic stage affordable rather than theoretical.

## The full loop: pre-training -> post-training -> agentic training

The point of the project is not only a fast pretrain; it is the whole modern
pipeline, on one GPU, with each stage gated on evidence that the previous one
worked.

| Stage | Script | Objective | Reward / signal |
|---|---|---|---|
| 1. Pretrain | `scripts/train_pretrain.py` | next token, code-heavy mixture, FIM | cross-entropy |
| 2. SFT | `scripts/train_sft.py` | masked CE on assistant turns | held-out masked loss + format compliance |
| 3. GRPO | `scripts/train_grpo.py` | GSPO ratio + k3 KL, single turn | **executed** asserts (partial credit) |
| 4. Agentic RL | `scripts/train_agent_rl.py` | same objective, multi-turn tool use | terminal episode reward |

No learned reward model anywhere: for code the ground truth is "does it run and
pass the tests", which is cheaper and not hackable in the ways a learned RM is.

### The failures this pipeline is shaped around

Each of these cost real debugging time in an earlier model in this lineage, and
each is now either impossible by construction or caught by a test.

- **Prompt-format drift between SFT and RL.** SFT trained with role markers
  while the rollout path encoded the bare question, so the policy was asked to
  continue a format it had never seen. Nothing crashed; reward was simply always
  zero, and it read as an RL-tuning problem for days. `blackwell_lm/chat.py` is
  now the single source of truth, and a test asserts the rollout prompt is a
  token-for-token **prefix** of the trained sequence.
- **An unnormalised KL.** With a length-normalised ratio and an unnormalised
  KL, the KL term measured ~1748 against an objective of order 1 - it was
  69.94 of a 69.93 total loss, so the policy gradient was numerically absent.
  Both terms are now per-token, and a test proves doubling the completion
  length changes neither.
- **bf16 made the k3 KL negative** (about -1/1024), which is impossible for a
  KL. The objective is computed in fp32.
- **Zero-variance groups.** A weak policy scores 0.0 on everything, so every
  group has zero spread and is correctly dropped - and GRPO then runs for hours
  with **zero update steps** while printing healthy-looking rollout stats. The
  scripts report `updates` next to `dropped` and say so explicitly, and
  `--task-set easy` exists as the control that separates "RL is broken" from
  "the model is too weak yet".
- **Silent precision downgrade.** Requesting fp8/nvfp4 without a working
  Transformer Engine now raises instead of falling back to BF16. It once
  produced an "NVFP4 vs FP8" measurement in which *both arms were BF16* and
  came out identical to 0.1% - which looked exactly like a finding.
- **TE refuses to load its own checkpoint.** `_extra_state` is a pickle, and TE
  will not unpickle it unless `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`. A cold
  start never touches that path, so the run looks fine for hours and then every
  **resume** dies - i.e. it only breaks on preemptible hardware, in a
  relaunch-and-die loop that looks like it is being managed.

### Bugs found by writing the tests

- **Incremental decode ignored the sliding window.** The `T == 1` branch let the
  single query attend to the entire KV cache, so a layer trained at window 1024
  generated with unbounded context. Measured divergence from windowed
  full-recompute: **1.25 in logits**, now 1.0e-06. The pre-existing cache test
  could not catch it because it ran with `window=0`.
- **`global_every` is inert at the default config.** With `n_layers=1` the
  cadence keys off a layer index that is always 0, so *no* application is ever
  full-context despite the docstring. Left as the default (the live run trained
  that way) and fixed behind `global_every_loop`, which applies the cadence over
  the **loop** index - the only version that does anything for a looped model.
- **A JSON array tool call crashed the episode.** `json.loads` accepts a bare
  list, which then has no `.get`. Malformed tool calls must be data, not
  exceptions, or the only episodes that fail are the ones that never produce a
  gradient.

## Layout

```
blackwell_lm/model.py        the model (measured design decisions cited inline)
blackwell_lm/tokenizer.py    code-aware BPE (3.63 vs GPT-2's 2.60 chars/token)
blackwell_lm/data.py         streaming code-heavy mixture + FIM + packing
blackwell_lm/chat.py         conversation format: ONE source of truth
blackwell_lm/generate.py     KV-cache decode + log-prob scoring
blackwell_lm/sft.py          instruction data -> (ids, targets, loss_mask)
blackwell_lm/sandbox.py      executes untrusted model-written code
blackwell_lm/reward.py       execution reward, partial credit
blackwell_lm/grpo.py         GSPO ratio, k3 KL, Clip-Higher
blackwell_lm/agent.py        multi-turn tool-use episodes
blackwell_lm/tasks.py        MBPP + a deliberately trivial control set
blackwell_lm/checkpoint.py   stage/precision transitions
scripts/test_model_cpu.py       11 model correctness tests
scripts/test_data_cpu.py         5 data-pipeline tests
scripts/test_posttrain_cpu.py   18 format/generation/GRPO/SFT tests
scripts/test_sandbox_cpu.py     20 adversarial sandbox + agent tests
scripts/smoke_pipeline_cpu.py   all stages wired, end to end, on CPU
bench/throughput.py          tokens/s + precision A/B on the local GPU
```

Correctness tests cover the parts most likely to be silently wrong: that the
sliding window *actually* masks (a token outside the window provably cannot
influence a later one), that incremental KV-cache decoding matches
full-recompute to 1.0e-06 **including under a window** - a cache that drifts
would corrupt every multi-turn agentic rollout without ever raising - and that
the GRPO objective's ratio and KL are both length-normalised.

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "transformer_engine[pytorch]>=2.18"    # FP8 only; see above for FP4
# every test suite runs on CPU, no GPU and no network needed
PYTHONPATH=. python scripts/test_model_cpu.py
PYTHONPATH=. python scripts/test_data_cpu.py
PYTHONPATH=. python scripts/test_posttrain_cpu.py
PYTHONPATH=. python scripts/test_sandbox_cpu.py      # Linux: exercises the rlimits
PYTHONPATH=. python scripts/smoke_pipeline_cpu.py
PYTHONPATH=. python bench/throughput.py --precision fp8 --batch 8 --seq-len 2048
```

The full loop, on one GPU:

```bash
PYTHONPATH=. python scripts/train_tokenizer.py --docs 200000
PYTHONPATH=. python scripts/train_pretrain.py --precision nvfp4 --target-tokens 20.4e9
PYTHONPATH=. python scripts/train_sft.py   --init-from pretrain.pt
PYTHONPATH=. python scripts/train_grpo.py  --init-from sft.pt   --task-set mbpp
PYTHONPATH=. python scripts/train_agent_rl.py --init-from grpo.pt
```

## Results: the full loop, measured

Every number below is from a real run on real hardware, including the ones that
did not work.

### Stage 1 - pretraining

20.40B tokens, 1,245,117 steps, 85M params (d_model 1536, one block looped 8x),
NVFP4, on a single RTX PRO 6000 Blackwell. ~46h of GPU time at 120,470 tok/s,
spanning three machines and two interruptions.

| held-out (MBPP, never trained on) | step | loss |
|---|---|---|
| mid-run | 656,711 | 2.1817 |
| mid-run | 910,397 | 2.1325 |
| **final** | **1,245,117** | **2.1155** |

The anneal is where the last gains live: held-out loss was FLAT across ~1B
tokens at lr 1.4e-4, then moved 0.066 as lr decayed to 3e-5. Flat loss at high
lr is not saturation, and reading it as such nearly ended the run 9h early - see
the anneal probe below.

### Stage 2 - SFT

4,000 steps on opencoder-sft / evol-codealpaca / tulu3.

| gate | pretrained | after SFT |
|---|---|---|
| masked held-out loss | 2.3255 | **1.7522** |
| format compliance | **0%** | **100%** |

The pretrained model essentially never emitted a fenced code block; after SFT it
does so every time, and writes correct code for simple prompts:

```
Write a function add(a, b) that returns a + b.
-> Here is the code to solve this problem:
   ```python
   def add(a, b):
       return a + b
   ```
```

### Stage 3 - GRPO with execution rewards

Rewards come from RUNNING the code against asserts in a sandbox. No reward model.

| task set | rollouts | mean reward | fully solved | **update steps** | zero-variance dropped |
|---|---|---|---|---|---|
| easy (control) | 40 | 0.269 -> 0.400 | 30/40 | **76** | 2 (5%) |
| MBPP | 300 | 0.0082 | 9/300 | **52** | 274 (91%) |

**The control run is the point.** A previous model in this lineage produced ZERO
update steps across three GRPO attempts, because every group scored identically
and was correctly dropped. Here the easy set yields 76 updates and a rising
reward, which proves the RL path works; MBPP then yields only 52 updates from
300 rollouts because the policy is too weak, not because the code is wrong.
Being able to tell those two apart is exactly why the trivial task set exists.

MBPP reward distribution over 2,400 samples: 2,367 zeros, 17 at 0.33, 6 at 0.67,
10 at 1.0. Partial credit fired 23 times - under a binary reward those groups
would also have been zero-variance and dropped.

### Stage 4 - agentic tool-use RL

Three attempts. The first two failed in different ways, and both failures were
found by instrumentation rather than by inspection of the reward:

| attempt | tool calls | mean turns | tool errors | **updates** | mean reward |
|---|---|---|---|---|---|
| v1 - no tool data in SFT | 0 | 1.00 | 0 | 12 | 0.025 |
| v2 - tool SFT, model supplied the tests | 50 | 1.36 | 3 | **0** | 0.0000 |
| **v3 - harness owns the tests** | **196** | **2.18** | **0** | **69** | **0.058** |

**v1 was not agentic at all.** 0 tool calls in 100 episodes: none of the SFT
sources contain tool-use examples, and RL cannot teach a convention the policy
never emits -- with no tool call ever sampled there is no gradient toward making
one. The script said so itself rather than reporting a reward as if it meant
something:

    DIAGNOSTIC: the policy has never emitted a parseable tool call, so this is
    single-turn RL with extra steps.

**v2 taught the model to grade itself.** Adding synthetic tool trajectories
lifted the tool-call rate from 0/32 to 11/32 samples, and the policy started
using tools (50 calls, 1.36 turns). But the trajectories carried the TESTS
inside the tool call, so the policy learned to generate them -- and generated
broken ones. `def identity(x): return x`, which is correct, scored 0/3 against
its own hallucinated `assert identity(9) is x`. Every episode scored 0, every
group was zero-variance, and the stage produced **zero** update steps. The model
was being graded on its invented grader instead of its code.

**v3 fixes the contract.** `run_tests(code)` now takes only code; the harness
supplies the task's hidden tests and ignores any the model passes. Plus, an
episode that exhausts its turns mid-tool-call falls back to the last code it
submitted, instead of scoring a flat 0 for every such episode. Result: 196 tool
calls over 120 episodes, 2.18 turns, **zero malformed calls**, 69 update steps
and a non-zero reward.

**A 400-step run then produced the actual learning curve** (3,200 episodes,
group 8, up to 4 turns, lr 1e-6):

| steps | reward | solved/episode | turns | zero-variance | tool errors |
|---|---|---|---|---|---|
| 1-50 | 0.2288 | 20.0% | 2.96 | 54% | 3 |
| 51-100 | 0.3688 | 30.0% | 3.25 | 56% | 0 |
| 151-200 | 0.5871 | 48.0% | 3.77 | 60% | 0 |
| 251-300 | 0.5637 | 39.0% | 3.96 | 76% | 1 |
| **301-350** | **0.6800** | **51.2%** | 3.98 | 90% | 1 |
| 351-400 | 0.6400 | 49.5% | 3.97 | 90% | 3 |

First 40 steps to last 40: **reward 0.1828 -> 0.6562 (3.6x)**, and episodes that
fully solve their task went **15.6% -> 52.5%**. 3,413 update steps, 11,045 tool
calls with **8 malformed (0.07%)**, mean turns rising 2.96 -> 3.97.

**The zero-variance rate climbing 54% -> 90% is not degradation.** It is the
policy succeeding so consistently that groups stop having spread, so there is
nothing left to learn from: a 12-task set is exhausted as a signal once half the
episodes solve it outright. Rising zero-variance late in a run means "graduate to
harder tasks", not "training broke" -- the same statistic means the opposite
thing early (policy too weak) and late (task set too easy), which is why the
easy-set control and the update-step count have to be read together.

Two design rules this produced, both learned the expensive way:

  * **The harness owns the grader.** If the model can pass its own tests, it
    will learn to, and its tests will be wrong.
  * **Synthetic tool output must be REAL.** Every "N/M tests passed" line in the
    trajectories comes from executing the code in the sandbox, including the
    deliberately-mutated first attempts. Hand-written plausible output would
    teach a fictional environment.

### The repo environment (MCP-shaped tools)

The task above is still "emit a snippet, we run it", which teaches a convention
that transfers to no real coding agent. `blackwell_lm/mcp_tools.py` and
`blackwell_lm/scenario.py` replace it with a scratch REPO the agent must
navigate:

  * tools are declared MCP-style -- `name`, `description`, `input_schema`
    (real JSON Schema) -- for `list_dir`, `read_file`, `write_file`, `run_tests`,
    and the **system prompt is generated from those declarations**, so it cannot
    drift from the dispatcher;
  * each task becomes a directory holding a **verified-broken** `solution.py`
    (the reference is checked to pass and the mutation checked to fail, in the
    sandbox, before the scenario is used);
  * **the reward reads the FILE, not the transcript.** An agent that edits
    correctly and says nothing scores full marks; one that describes a perfect
    fix without writing it scores zero. That also makes the reward structurally
    immune to the answer-extraction bug that gave an earlier run 100 identical
    zeros.

Verified end to end: broken `return a - b` -> `1/3 tests` -> agent writes a fix
-> `3/3 tests`, with path escapes, unknown tools, missing and mistyped arguments
all returned as readable errors rather than exceptions. Available as
`--task-set repo-easy` / `repo-mbpp`.

#### The 70% solve rate on these repos was an artifact. Measured.

Trained on `repo-mbpp`, an A/B took the solve rate 50% -> 70% and self-
verification 8% -> 91%. That looked like the headline result of the whole
project, and it was wrong -- so before building anything on top of it, the same
checkpoint was re-probed against harder breakages of the *same* tasks:

| tier | how `solution.py` is broken | n | solved +-SE | wrote a file +-SE |
|------|------------------------------|---|-------------|-------------------|
| `mutate` | one regex substitution (`+`->`-`) | 120 | **59% +-4** | 86% +-3 |
| `multi`  | two substitutions at once | 120 | **60% +-4** | 86% +-3 |
| `stub`   | function body replaced by `pass` | 120 | **0% +-0** | 98% +-1 |
| `swap`   | a *different* task's real solution | 120 | **0% +-0** | 93% +-2 |

**Zero of 120, twice.** The 70% measured "revert the token that looks odd", not
"fix the code". `stub` and `swap` are the honest tiers precisely because neither
can be solved by spotting a suspicious character.

The tell was visible beforehand and nearly ignored: the same checkpoint solves
**9/300 (3%)** of single-turn MBPP. A 20x gap between writing a solution and
repairing one should have been suspicious on sight -- single-token reversal is a
keyhole skill that reads as debugging.

Note the direction of the `wrote a file` column: **93-98% on the hard tiers,
higher than on the easy ones.** The agentic workflow is genuinely learned -- it
reads, edits and verifies. It just writes wrong code with total confidence. Tool
use and coding are separable capabilities, and at 85M only the first is
reachable.

#### ...and then the diagnosis was wrong too. It was the TRAINING data.

The paragraph that used to sit here argued the 0/120 was an 85M capability
ceiling, and that teacher distillation therefore could not help. That was wrong,
and the cause was one defaulted argument in this repo:

```python
scns = build_scenarios(base_tasks, seed=1)   # difficulty defaults to "mutate"
```

Every repo demonstration the policy had ever seen was a single-token mutation,
where the correct fix is *write the file back with one token changed*. So that
is what it learned -- and the stub probe caught it exactly: **236 of 240
episodes wrote the unchanged stub back to disk.** It had mastered copying, not
coding.

Adding `--tool-difficulty` and rebuilding the demos on `stub` -- where the
demonstration writes the **whole reference body** -- and re-running SFT for
1,500 steps (11 minutes):

| stub tier, 240 episodes | mutate-trained demos | **stub-trained demos** |
|---|---|---|
| pass@1 | 0.0% (0/240) | **9.6% +-1.9** (23/240) |
| pass@12 | 0.0% (0/20 tasks) | **40.0%** (8/20 tasks) |
| any partial credit | 0.0% | **11.2%** |
| left the stub unchanged | **236/240** | 99/240 |
| wrote valid Python | 1/240 | **92/240** |

Correct code, written from a bare stub, reward 1.00:

```python
def max_of_nth(test_list, N):
  res = max([sub[N] for sub in test_list])
  return (res)
```

**The method lesson, learned twice in one day in opposite directions.** A
benchmark's difficulty and a training set's difficulty are the same hidden
parameter. The flattering *benchmark* made the model look better than it was
(70% -> 0%); the flattering *training set* made it look worse than it was
(0% -> 9.6%). Check what your demonstrations actually demonstrate -- print one
and read it -- before declaring any capability limit.

And pass@12 = 40% is the precondition RL needed: groups now have reward spread,
so GRPO takes real update steps instead of dropping every group as
zero-variance.

### Honest summary

All four stages are demonstrated end to end on real hardware. The resulting 85M
model writes correct simple functions, holds the chat format perfectly, and uses
tools multi-turn with real execution feedback -- the stated bar was "may make
errors but should look acceptable", and that is met for simple prompts.

What it does NOT do: solve MBPP (9/300 tasks; 1.4% of samples earn any credit),
or write a function body it was not shown -- 0/120 on the `stub` and `swap`
tiers, while still writing the file 93-98% of the time. Those are honest
limits of an 85M model at 240 tokens/param, not pipeline defects -- the easy-set
control run separates the two, and it passes.

## Honest limits

- Throughput: the **PRO 4500** (32GB) sustained **45,146 tok/s**. A **PRO 6000**
  (96GB) later ran the final 4.0B tokens and sustained a mean of **119,807
  tok/s** (median 119,848, max 121,020) after the batch size was raised to fill
  96GB - a measured **2.65x**, cross-checked against wall clock (9.27h for
  4.0B tokens implies 119,770 tok/s).
  <br>*This bullet previously read "PRO 6000 figures are scaled by 1.726x ...
  nothing here has run on a PRO 6000". Both halves were true when written and
  both were still here long after a PRO 6000 had run at 120k tok/s. The number
  was never re-read against the metrics that disproved it - the same failure as
  the 70% retraction above, in a cheaper place.*
- This does **not** beat an H100 on speed, and nothing on this hardware does:
  the memory-bandwidth gap is roughly 2x and was immovable across three
  toolchain upgrades. Blackwell's argument is tokens-per-unit-cost rather than
  raw speed: it stays within 1.81x on throughput at a materially lower hourly
  rate than a datacentre part.
- The trained model (85M) produces *acceptable-looking* output, not correct
  output - and the hardness audit above shows how far that gap goes: it cannot
  write a function body it was not shown (0/240 from a stub). Judge it against
  the stated goal, which was to validate the pipeline.
- **The agentic stage learns, on a task set it then exhausts.** 400 steps took
  reward 0.183 -> 0.656 and full-solve rate 15.6% -> 52.5%, but zero-variance
  groups rose to 90% because the 12-task easy set runs out of signal. The repo
  scenarios replaced it and trained (50% -> 70% solved, 8% -> 91% verified) --
  but that gain lives **entirely** on the one-token-mutation tier and is 0% on
  `stub`/`swap`. See the hardness table above; treat the 70% as void.
- **bf16 parameters cannot be the optimizer's state.** TE's NVFP4 path requires
  bf16 params, but an AdamW update below the bf16 spacing rounds away entirely:
  pretraining ran 657,000 steps with all five RMSNorm gains still bit-exactly
  1.0, and at the post-training learning rates only 8% (SFT) and 2% (GRPO) of
  elements could move at all. `blackwell_lm/optim.py` keeps fp32 masters for
  ~3% overhead. Naively unfreezing the frozen gains mid-run made things WORSE
  (stale momentum discharged; held-out loss +0.052), so 1-D gains stay frozen.
- **A rolling training loss cannot detect any of this.** It missed a real
  regression, then twice signalled a plateau that was not one. Every quality
  claim here comes from a fixed held-out set instead.

- **The sandbox is a robustness boundary, not a containment boundary.** It
  applies POSIX rlimits, an isolated interpreter, a throwaway directory and
  process-group kills, but creates no network, mount or PID namespace. Run the
  trainer in a container with no egress if that matters. It degrades to
  timeout-only on Windows, and the agentic script refuses to start there
  without `--allow-weak-sandbox`.
- **Agentic credit assignment is terminal-reward broadcasting.** The episode's
  advantage goes to every assistant turn in it, so a turn that was irrelevant
  gets the same credit as the one that fixed the bug. Per-turn value estimation
  would sharpen this; at this model size it is not the binding constraint.
