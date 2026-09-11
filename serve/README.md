# Serving blackwell-nanogpt to a coding CLI

Drives **Claude Code** or **Codex CLI** with the local 85M checkpoint. No
Anthropic or OpenAI model is involved at any point — the only weights are the
ones in this repo's checkpoint.

## What it is actually good at

One task shape, the one it was trained on:

> `solution.py` is failing its tests. Read it, fix it, and verify with `run_tests`.

It reads the file, edits it, runs the tests, and retries when they fail. That is
the whole job. It is an 85M-parameter model: it is **not** a general coding
assistant, it cannot do multi-file work, and it will not build you a feature.
Judge it against the narrow task, which it does measurably well, and not
against a frontier model, which it is not.

## Start the server

On the GPU box (the checkpoint and tokenizer live beside it):

```bash
cd /home/ubuntu/bnano
PYTHONPATH=. python3 anthropic_shim.py \
  --ckpt sft_ho.pt \
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
