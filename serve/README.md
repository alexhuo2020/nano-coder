# Serving blackwell-nanogpt to a coding CLI

Drives **Claude Code** or **Codex CLI** with the local 85M checkpoint. No
Anthropic or OpenAI model is involved at any point — the only weights are the
ones in this repo's checkpoint.

## Which checkpoint, and what it scores

Use **`sft_ctl.pt`**. Measured on 74 tasks never seen in training, T=0.2,
120 episodes per tier:

| tier | what is broken | `sft_ctl` | `sft_ho` |
|------|----------------|-----------|----------|
| `mutate` | one token (`+` -> `-`) | **56.7% ±4.5** | 40.8% ±4.5 |
| `multi` | two tokens | **55.8% ±4.5** | 40.8% ±4.5 |
| `stub` | body replaced by `pass` | 0.0% | **7.5% ±2.4** |
| `swap` | a different function entirely | 0.0% | **5.0% ±2.0** |
| mean | | **28.1%** | 23.5% |

`sft_ho` was trained on all four tiers plus retry demos and is **worse
overall**: it buys 7.5% and 5.0% on the two hard tiers by giving up ~16 points
on each easy one. At 85M, task-shape priors compete for capacity — four of them
do not fit, and teaching `stub` partly evicts `mutate`. Since real breakage is
far more often a wrong operator than a deleted function body, `sft_ctl` is the
better default. Switch to `sft_ho` only if you specifically need non-zero
`stub`/`swap`.

Single-turn code generation (no repo, no tools, held-out): **6.4% ±1.4**
pass@1, 10.8% pass@4. A weak coder in absolute terms.

Every number above is on HELD-OUT tasks. On tasks it trained on the same model
scores 78.5% mean — that gap is memorisation, and it is why the split exists.

## Why a plain question made it read `solution.py`

In `--claude-code` mode the server injects the AGENT system prompt on every
request — it opens with *"You are a coding agent working in a repository"* and
its first example is a `read_file` call. Half the SFT was repo trajectories
that always begin with `read_file`. So "who are you" was answered by reading a
file: the model was following its prompt, not malfunctioning.

`--chat-passthrough` routes requests with no repo signal (no file, test, fix or
function mentioned, and no tool traffic yet) to the chat prompt instead.
Verified: "who are you" now returns text; "solution.py is failing its tests"
still returns a tool call.

**This does not make it conversational.** Asked who it is, it replies with an
unrelated Python function — its chat prompt says *"answer with runnable
Python"* and 100% of its SFT taught it to emit code. The routing is correct;
the model is a code model. Treat chat mode as a way to avoid a confusing
behaviour, not as a feature.

## What it is actually good at

One task shape, the one it was trained on:

> `solution.py` is failing its tests. Read it, fix it, and verify with `run_tests`.

It reads the file, edits it, runs the tests, and retries when they fail. That is
the whole job. It is an 85M-parameter model: it is **not** a general coding
assistant, it cannot do multi-file work, and it will not build you a feature.
Judge it against the narrow task, which it does measurably well, and not
against a frontier model, which it is not.

## Running it locally, with no GPU

Verified: claude-cli driving this model on a Windows CPU fixed a held-out task
end to end (pytest FAILED -> `1 passed`). Measured **4.5 tok/s**, ~4-18s per
agent turn, ~340 MB of RAM.

```bash
# 1. a venv at a SHORT path. The Windows Store Python's site-packages path is
#    long enough that torch's own deep files exceed MAX_PATH and pip dies
#    mid-install, which is how a machine ends up with an unimportable torch.
python -m venv C:\Users\<you>\bn
C:\Users\<you>\bn\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
C:\Users\<you>\bn\Scripts\python.exe -m pip install tokenizers numpy pytest

# 2. the weights: put the checkpoint and its tokenizer in local/
#    (a checkpoint is ~1.1 GB; the tokenizer ~2 MB)
#    THE TOKENIZER IS LOAD-BEARING: a different one loads without error and
#    produces nonsense, so use the one published with the weights.

# 3. serve on CPU
C:\Users\<you>\bn\Scripts\python.exe serve/anthropic_shim.py \
  --ckpt local/sft_ctl.pt --tokenizer local/tokenizer.json \
  --device cpu --port 8799 --context 4096 --claude-code --truncate \
  --chat-passthrough \
  --temperature 0.2 --cwd 'C:\path\to\your\repo'
```

Transformer Engine is not needed and is not installed: `model.py` makes its
import optional and the low-precision layers fall back to `nn.Linear`.

Known issue: the server log occasionally reports an implausible per-request
duration (seen: 10731s during a run that took minutes). The cause is not yet
identified, so treat per-request timings in the log as unreliable.

## Start the server

On the GPU box (the checkpoint and tokenizer live beside it):

```bash
cd /home/ubuntu/bnano
PYTHONPATH=. python3 anthropic_shim.py \
  --ckpt sft_ctl.pt \
  --cwd 'C:\path\to\your\repo' \
  --context 32768 \
  --claude-code \
  --truncate \
  --temperature 0.2
```

`--temperature 0.2` is measured, not guessed: at the RL rollout default of 1.0
the same checkpoint scores 24 points lower on every tier (see the table in
`anthropic_shim.py`). `--context 32768` is free on this architecture — RoPE is
a non-persistent buffer and every attention application is a 1,024-token
sliding window, so no relative offset beyond 1,024 ever occurs.

**`--cwd` is not optional when the CLI runs on another machine.** It is the
CLIENT's repository directory, and the server cannot infer it. Without it the
fallback is the *server's* working directory, so Claude Code is handed
`/home/ubuntu/bnano/solution.py` on a Windows client: every tool call reports
success against a file that is not yours, the real one is never touched, and
the tests keep failing. That failure mode reads exactly like "the model keeps
writing the file back unchanged", which cost hours to see through.

If the CLI runs on a different machine, tunnel the port:

```bash
ssh -i KEY -N -L 8788:127.0.0.1:8788 ubuntu@HOST
```

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
export ANTHROPIC_AUTH_TOKEN=dummy-local-shim      # not a real key; nothing leaves the machine
export ANTHROPIC_MODEL=blackwell-nanogpt-85m
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000      # see the warning below
claude -p "The file solution.py in this repo is failing its tests. Read it, fix it, and verify with run_tests." \
  --max-turns 8 --permission-mode bypassPermissions
```

**`CLAUDE_CODE_MAX_CONTEXT_TOKENS` must be large, not the model's real window.**
The CLI counts *its own* ~18.8k-token prompt against that number. Declaring the
true 32768 made it decide it was near the limit and **auto-compact** — which
asks the 85M model to summarise the conversation, produces word salad, and then
*replaces the conversation with that word salad*, destroying every later turn.
The server strips the prompt to ~220 tokens before the model sees it, so there
is no real overflow to protect against.

## Point Codex at it

Codex 0.150.1 speaks only the OpenAI Responses API; the same server handles it
on `/v1/responses`. Use an isolated `CODEX_HOME` so your real `~/.codex` (which
holds credentials) is never read or modified:

```bash
export CODEX_HOME=/tmp/bnano_codex_home
mkdir -p "$CODEX_HOME" && cp serve/codex_config.toml "$CODEX_HOME/config.toml"
export BNANO_API_KEY=dummy-local-shim
codex exec -s workspace-write --skip-git-repo-check "The file solution.py ... fix it"
```

Known limitation on Windows: `codex doctor` may report
`sandbox backend: disabled`, in which case Codex refuses to run **any** command
("blocked by policy") regardless of this server. That is a Codex installation
issue, not a model or adapter issue. `codex_adapter.py` enforces path
containment itself — absolute paths, drive letters, UNC paths and `..`
traversal are rejected before a command is ever emitted — so the adapter does
not depend on the client's sandbox for safety.

## Measuring it

```bash
# all four breakage tiers, on tasks never seen in training
PYTHONPATH=. python3 scripts/scoreboard.py sft_ho.pt \
  --holdout 74 --tasks 20 --samples 6 --temperature 0.2
```

`--holdout 74` must match `train_sft.py --tool-holdout 74`. Without it the
scoreboard draws tasks from the same pool training used, and reports **train
accuracy wearing the clothes of a benchmark** — a mistake made once here
already.

End-to-end through the CLI, both an easy and an honest breakage:

```powershell
powershell -File serve/run_cli_trials.ps1 -Trials 6 -Kind both
```

## Files

| file | what it does |
|---|---|
| `anthropic_shim.py` | the server; `/v1/messages` and `/v1/responses`, one loaded model |
| `cli_adapter.py` | Claude Code protocol translation (tool_use blocks, paths, line numbers) |
| `codex_adapter.py` | OpenAI Responses protocol, shell-command mapping, path containment |
| `probe_server.py` | records what a client actually sends; how the prompt sizes were measured |
| `test_cli_adapter.py` | 11 tests |
| `test_codex_adapter.py` | 8 tests |
| `run_cli_trials.ps1` | end-to-end solve rate over N trials |
