# What this project actually taught

Every item here cost real time and is backed by a measurement in this repo. They
are ordered by how much they changed the outcome, not by how interesting they
sound.

---

## 1. Four unexamined defaults produced four different "results"

This is the finding. Each default was chosen because it was the path of least
resistance, and each silently became a *definition* that the model then got
credit or blame for.

| # | the default | what it produced | the truth |
|---|-------------|------------------|-----------|
| 1 | benchmark tier = `mutate` | **70%** solve rate | measuring "revert the odd token", not repair |
| 2 | training tier = `mutate` | **0/240** "capability ceiling" | demos taught copying, not coding |
| 3 | sampling `T = 1.0` | **1.0%** on MBPP | correct for RL *exploration*, wrong for *solving* — 6.4% at T=0.2 |
| 4 | no train/test split | **78.5%** mean | 23.5% on held-out; the rest was memorisation |

Two of these made the model look **better** than it was, two made it look
**worse**. That symmetry is the point: an unexamined default is not biased
toward flattery, it is simply unmeasured.

**The rule:** before believing any number about a model, ask what the evaluation
*could not have distinguished*. Then split the tasks and sweep the sampling.

---

## 2. RL sharpens a distribution; it cannot create support in it

Observed three independent times. If the policy never emits a behaviour, there
is nothing for reward to reinforce:

- The model emitted `write_file` with no `content` **19 times out of 19**.
- A better error message — returning the exact expected call shape as data —
  fixed **0 of 18**.
- RL on it took **0 update steps**: every group had identical reward, so every
  group was dropped as zero-variance.
- Demonstrations of the missing call fixed it immediately (tool-call rate
  0% → 100%).

**Corollary learned later, and it matters:** "add demonstrations" is not enough.
They must demonstrate *the target behaviour*. Demos of an adjacent, easier task
actively train the wrong skill — which is exactly how default #2 happened.

---

## 3. At small scale, task-shape priors compete for capacity

Trained on all four breakage tiers plus retry demos, against a control trained
identically on one tier. Both evaluated on the same 74 unseen tasks:

| tier | one tier (`mutate`) | four tiers + retry |
|------|--------------------|--------------------|
| `mutate` | **56.7% ±4.5** | 40.8% ±4.5 |
| `multi` | **55.8% ±4.5** | 40.8% ±4.5 |
| `stub` | 0.0% | **7.5% ±2.4** |
| `swap` | 0.0% | **5.0% ±2.0** |
| mean | **28.1%** | 23.5% |

Teaching `stub` partly **evicted** `mutate`. The intervention was a net loss.

The train-set numbers hid this completely — there the model held all four tiers
at once (78.5% mean), because it had *memorised* the solutions, and memorisation
does not trade off the way a learned prior does.

---

## 4. A control trained without the same split is not a control

The first comparison used a checkpoint that predated the held-out split and had
therefore trained on the "held-out" tasks. It scored 73.3% where the properly
split model scored 40.8%, which looked like strong evidence the new training had
*hurt*. It was measuring memorisation.

It looked exactly like a control. That is what made it dangerous.

---

## 5. Tool *results* must match training, not just tool *calls*

The model's own `run_tests` returns `"3/3 tests passed"`. Wired to a real CLI it
received raw pytest output instead — `1 failed in 0.08s`, tracebacks, summary
lines — which it had never seen.

Measured consequence: across 10 trials it called `read → test → read → test` and
**never once** `write_file`, because it could not tell whether its edit had
worked. 0/10, while the same model solved 56.7% in its own harness.

Normalising the output back to the trained form fixed it.

---

## 6. When a model does worse through an integration than in its harness, suspect the integration

Three bugs in the CLI adapter each impersonated a plausible *model* failure, and
hours went into behavioural explanations for what were plumbing errors:

| bug | presented as | actually |
|-----|--------------|----------|
| cwd fell back to the **server's** path | "it echoes the file back" | editing a phantom file; the real one untouched |
| raw pytest output | "it loops without fixing" | it couldn't tell if the edit worked |
| `os.path.basename` doesn't split Windows paths on Linux | "it omits `content`" | the replayed transcript taught it 60-char absolute paths |

The tell was visible throughout and ignored: **the same model solved the same
task in the harness on the first try.**

---

## 7. Four measurement batches were invalid, not negative

A shim killed mid-run; a dead SSH tunnel; `Start-Process` handed a shell wrapper
instead of the `.exe`; and bug #5. Each produced clean-looking failure counts
that measured nothing.

**A measurement harness must verify its own plumbing** — probe the connection
before each trial, use absolute binary paths, timeout per attempt — or it
manufactures fake negatives that are indistinguishable from results.

---

## 8. bf16 parameters cannot hold optimizer state

BF16 spacing near `x` is `2^(floor(log2 x) − 7)`, so any AdamW update smaller
than that rounds away entirely. Measured, not theorised: after 657,000 steps all
five RMSNorm gain vectors were still **bit-exactly 1.0**, and at post-training
learning rates only **8%** (SFT) and **2%** (GRPO) of parameter elements could
move at all.

The fix — fp32 master weights — had its own trap: applying masters to the 1-D
gains that had been frozen for 657k steps made the model **worse** (held-out
+0.052) as stale momentum discharged into weights that had never moved.

---

## 9. A rolling training loss cannot detect any of this

It missed a real regression, and twice signalled a plateau that was not one. An
anneal probe at the schedule floor recovered 0.053 nats in 41M tokens after 1B
tokens of apparent flatness.

**"Saturated" is a claim about capacity and requires an anneal probe to
support.** Every quality claim in this project comes from a fixed held-out set
instead.

---

## 10. Let the harness own the grader

The model was observed inventing its own tests and passing them. `run_tests(code)`
now ignores model-supplied tests entirely.

Relatedly: reward reads the **file the agent leaves behind**, not its transcript.
An earlier design extracted code from the final message; an episode that ran out
of turns mid-tool-call had nothing to extract, scored a flat 0, made every group
zero-variance, and produced 100 episodes with 0 updates.

---

## 11. RL will remove any behaviour the reward does not pay for

Because reward read only the final file, calling `run_tests` earned nothing and
cost a turn. Over 84 steps, `ran_tests` collapsed from **79% of episodes to 1%**,
and tool errors fell 162 → 0 in lockstep (the errors *were* the `run_tests`
calls). Reward still rose throughout.

The fix is an incentive, not a patch to the policy — and the first attempt was a
no-op: an additive bonus clamped to `[0,1]` can only fire when the reward is
already 1.0. Worse, the unit test passed it, because it checked the clamp rather
than checking that the incentive changed any ordering.

---

## Operational lessons

- **Terminating a GPU instance does not shut it down if an ASG owns it.** A
  replacement appeared three minutes later. Scale every ASG to `0/0/0` *first*.
  An ASG with `desired > 0` and no instances is not idle — it is waiting.
- **A one-time spot request stays `active` against a dead instance** and holds
  GPU quota, faking `MaxSpotInstanceCountExceeded`.
- **`pgrep -f PATTERN` matches its own command line.** Hit four times in one
  session, including a watcher that reported a finished job as running forever.
  Use `pgrep -af "python3 script[.]py"` and confirm against GPU memory.
- **`NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`** must be set before importing
  Transformer Engine or every *resume* crashes — cold starts look healthy, so it
  surfaces only after a spot reclaim.
- **The sandbox is a robustness boundary, not a containment boundary.** POSIX
  rlimits, an isolated interpreter, a throwaway directory and process-group
  kills — but no network, mount or PID namespace.
