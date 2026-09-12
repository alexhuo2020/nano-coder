# Results

Every number here is measured, with the conditions stated. Where a number was
later retracted it is shown **with** its retraction, because the retractions are
the more useful half.

Unless stated otherwise: 120 episodes per tier, `T=0.2`, `max_turns=6`, tasks
**held out** from training (MBPP 901–974, none seen during SFT).

---

## Headline: what the model can do

| task | held-out result |
|---|---|
| repair a wrong operator in a repo (`mutate`) | **56.7% ±4.5** |
| repair two wrong tokens (`multi`) | **55.8% ±4.5** |
| write a function body from `pass` (`stub`) | **0%** (7.5% if trained for it, at a cost) |
| replace a wrong-but-working function (`swap`) | **0%** (5.0% likewise) |
| single-turn code generation (no repo, no tools) | **6.4% ±1.4** pass@1, 10.8% pass@4 |
| emits a well-formed tool call | ~100% |
| malformed tool calls | 0.07% |

**Read that as:** a working narrow agent for token-level repair, and a weak
code generator. Those are separable, and only the first is solid at this scale.

---

## Pretraining

| checkpoint | step | held-out CE | code split | text split |
|---|---|---|---|---|
| baseline | 656,711 | 2.1817 | 1.0598 | 3.5910 |
| mid | 910,397 | 2.1325 | 1.2693 | 3.4211 |
| **final** | **1,245,117** | **2.1155** | 1.1979 | 3.3643 |

20.40B tokens, 85M params (240 tokens/param).

**Depth does not extrapolate.** Varying loop count at inference, held-out CE is
minimised at exactly the trained depth:

| loops | 2 | 4 | **8** | 12 | 16 |
|---|---|---|---|---|---|
| CE | 7.0137 | 2.2154 | **2.1155** | 2.1305 | 2.1794 |

**Context extrapolates for free.** CE on tokens *past* the 2,048 training
length: 1.26 at 4,096 and 1.16 at 8,192 — no degradation, where broken
extrapolation would show CE in the tens. Cause: RoPE is a non-persistent buffer
and every attention application is a 1,024-token sliding window, so no relative
offset above 1,024 ever occurs. Serving at 32,768 costs +20 MiB.

---

## SFT

| run | data | masked CE | format compliance | tool-call rate |
|---|---|---|---|---|
| v1 | chat only | 2.0056 → 1.2648 | 50% → 100% | 0% → 0% |
| v2 | + snippet tools | 2.0128 → 1.2277 | 0% → 100% | 0% → 50% |
| **v3** | + repo-shaped tools | **1.8564 → 0.9830** | 0% → 100% | **0% → 100%** |

---

## Sampling temperature — the largest single win

Same checkpoint, same held-out tasks, no retraining. 296 single-turn MBPP
samples:

| T | pass@1 | pass@4 | emitted a code block |
|---|---|---|---|
| 1.0 | 1.0% ±0.6 | 2.7% | 97.6% |
| **0.2** | **6.4% ±1.4** | **10.8%** | 100% |

**6.4×.** On the agentic tiers the same change is worth **+24 points**:

| T | `mutate` | `stub` |
|---|---|---|
| 1.0 | 52.5% ±4.6 | 7.5% ±2.4 |
| 0.5 | 74.2% ±4.0 | 25.0% ±4.0 |
| 0.2 | 76.7% ±3.9 | 31.7% ±4.2 |

T=1.0 is correct for RL *exploration* (it is what gives GRPO reward variance)
and wrong for *solving*. Every earlier figure in this project was measured at
1.0 because that is what the rollouts use.

---

## Train vs held-out — the memorisation gap

Same model, same tiers, only the task split differs:

| tier | tasks it trained on | **held out** |
|---|---|---|
| `mutate` | 95.0% ±2.0 | 40.8% ±4.5 |
| `multi` | 95.0% ±2.0 | 40.8% ±4.5 |
| `stub` | 87.5% ±3.0 | 7.5% ±2.4 |
| `swap` | 36.7% ±4.4 | 5.0% ±2.0 |
| **mean** | **78.5%** | **23.5%** |

`stub` fell 87.5 → 7.5. The model had memorised 300 reference solutions.

---

## The intervention that failed

Four tiers + retry demos vs a control trained identically on one tier, both on
the same unseen tasks:

| tier | `mutate` only | four tiers + retry |
|---|---|---|
| `mutate` | **56.7% ±4.5** | 40.8% ±4.5 |
| `multi` | **55.8% ±4.5** | 40.8% ±4.5 |
| `stub` | 0.0% | **7.5% ±2.4** |
| `swap` | 0.0% | **5.0% ±2.0** |
| **mean** | **28.1%** | 23.5% |

Net **−4.6 points**. At 85M, task-shape priors compete for capacity.

---

## Retracted results

Kept deliberately — each shows a way to fool yourself.

**"70% solve rate on repository repair."** Re-probed against harder breakages of
the *same* tasks:

| tier | n | solved ±SE | wrote a file ±SE |
|---|---|---|---|
| `mutate` | 120 | 59% ±4 | 86% ±3 |
| `multi` | 120 | 60% ±4 | 86% ±3 |
| `stub` | 120 | **0% ±0** | 98% ±1 |
| `swap` | 120 | **0% ±0** | 93% ±2 |

Note the write rate is *higher* on the hard tiers. It wrote a file confidently
every time and earned nothing — 236 of 240 episodes wrote the unchanged stub
back to disk.

**"0/240 is an 85M capability ceiling."** No — the SFT demos defaulted to the
`mutate` tier and taught copying. Rebuilding them on `stub` appeared to give
0% → 9.6%.

**"Stub demos take pass@1 to 9.6%."** That was train accuracy (see the
memorisation gap above).

**"The control scores 73.3%, so the intervention hurt."** That control predated
the holdout and had trained on the "held-out" tasks. A control trained without
the same split is not a control.

---

## Agentic RL

On the 12-task control set it genuinely learns: reward 0.183 → 0.656,
fully-solved 15.6% → 52.5% over 400 steps (3,200 episodes, 3,413 update steps) —
with zero-variance groups rising to ~70% as a 12-task set runs out of signal.

On repo tasks from a good SFT checkpoint it added **nothing**: 63 steps, 504
episodes, 782 update steps, no improvement.

**RL removes any behaviour the reward does not pay for.** With reward reading
only the final file, self-verification was trained *out* — episodes running the
tests fell from 75.3% to 0.0%, and tool errors fell 433 → 58 in lockstep,
because the errors *were* the `run_tests` calls. Adding a 10% discount for
finishing unverified held verification at 86%.

---

## Hardware

| | PRO 4500 (32 GB) | PRO 6000 (96 GB) |
|---|---|---|
| sustained | 45,146 tok/s | **119,807 tok/s** |

2.65×, cross-checked against wall clock (9.27 h for 4.0B tokens implies 119,770
tok/s). **NVFP4's benefit is governed by `d_model`** — worthless at 768,
1.084× at 1536, 1.227× at 3072 — not by batch size or FFN width.

Local CPU inference: **4.5 tok/s**, ~340 MB RAM, no GPU and no Transformer
Engine required.

---

## Raw evidence

Every table above is reproducible from `runs/results/` (probe logs and
transcripts) and `runs/metrics/` (per-step JSONL). The report figures are
generated from those files by `report/make_figures.py` — no number is retyped.
