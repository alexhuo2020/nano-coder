"""Translate between the OpenAI Responses API (Codex CLI) and this model.

WHY A SECOND ADAPTER. Codex CLI 0.150.1 dropped `wire_api = "chat"` outright
("no longer supported"), so it speaks only POST /v1/responses -- a different
wire format from Anthropic's /v1/messages, with `instructions` instead of
`system`, an `input` array instead of `messages`, and a streaming event
vocabulary of its own. None of the Anthropic adapter's wire handling carries
over; only the idea does.

THE HARDER DIFFERENCE IS THE TOOLS. Claude Code offers Read/Write/Bash, which
map almost one-to-one onto the model's read_file/write_file/run_tests. Codex
offers NO file tools at all -- just `exec_command`, a PTY shell (measured: 10
tools, of which exec_command is the only one that touches files). So every
tool the model knows has to become a shell command.

That makes quoting the whole problem. The model writes arbitrary Python source
into `content`: quotes, backslashes, newlines, dollar signs. Interpolating that
into a command line has to survive cmd.exe AND bash, and the failure mode is
not an error -- it is a file written with subtly mangled content, which looks
exactly like the model generating bad code. So file content is BASE64-ENCODED
and decoded by a Python one-liner on the far side. The command line then
contains nothing but `[A-Za-z0-9+/=]`, and the model's bytes arrive verbatim
or not at all.

Paths are passed as ARGV rather than interpolated into the Python source for
the same reason: `open('...')` breaks on a path containing a quote, and
Windows paths are full of backslashes that Python would read as escapes.
"""
from __future__ import annotations

import base64
import json
import re
import uuid

# `python` may not be on PATH as `python3` on Windows; Codex runs commands in a
# PTY with the user's PATH, and this project already established that `python`
# resolves there (pytest was run through it).
PY = "python"


_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


def _within(path: str, cwd: str) -> bool:
    """True if `path` stays inside `cwd`.

    Done on the STRING form for the client's platform, because this server
    runs on Linux and the client may be on Windows -- os.path.abspath here
    would resolve against the wrong root entirely. Absolute paths, drive
    letters, UNC paths and `..` traversal are all rejected; the model's normal
    output ("solution.py") passes untouched.
    """
    p = (path or "").strip().replace("\\", "/")
    if not p or p.startswith("/") or _WIN_ABS.match(path or ""):
        return False
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    depth = 0
    for seg in parts:
        if seg == "..":
            depth -= 1
            if depth < 0:
                return False
        else:
            depth += 1
    return True


def _psq(s: str) -> str:
    """PowerShell single-quoted literal: only ' needs escaping, by doubling."""
    return "'" + (s or "").replace("'", "''") + "'"


def _shq(s: str) -> str:
    """POSIX single-quoted literal."""
    return "'" + (s or "").replace("'", "'\\''") + "'"


def model_call_to_codex(name: str, args: dict, cwd: str):
    """(model tool, args) -> exec_command arguments, or None if unmappable.

    WHY THE COMMANDS LOOK LIKE THIS. Codex runs Windows commands as
        powershell.exe -Command "<cmd>"
    so any DOUBLE quote inside <cmd> has to survive a second round of
    escaping, and its Windows policy rejects string-built composed commands
    outright -- measured: a `python -c "..."` form came back as
    `rejected: blocked by policy` before it ever ran. The Windows commands
    below therefore use native cmdlets and SINGLE quotes exclusively, so
    nothing inside needs escaping at all.

    File content is still BASE64 on both platforms. The model writes arbitrary
    Python -- quotes, backslashes, newlines, `$` -- and interpolating that into
    a command line fails by writing subtly mangled content, which is
    indistinguishable from the model generating bad code.
    """
    args = args or {}
    win = bool(_WIN_ABS.match(cwd or ""))

    # CONTAINMENT IS ENFORCED HERE, NOT BY THE CLIENT. Codex's Windows sandbox
    # backend reports "disabled" on this machine, so every command is either
    # refused outright or (with --dangerously-bypass-approvals-and-sandbox)
    # run with no confinement at all. The path in a tool call is chosen by an
    # 85M model whose output is frequently garbage, so a write must never be
    # able to escape the working directory just because the client stopped
    # checking.
    if name in ("read_file", "write_file", "list_dir"):
        if not _within(args.get("path") or ".", cwd):
            return None

    if name == "read_file":
        path = args.get("path") or ""
        if win:
            return {"cmd": f"Get-Content -Raw -LiteralPath {_psq(path)}"}
        return {"cmd": f"cat {_shq(path)}"}

    if name == "write_file":
        # `content` missing is the failure this policy historically had; it is
        # reported, never invented. Same rule as cli_adapter.
        if "content" not in args:
            return None
        path = args.get("path") or ""
        blob = base64.b64encode(
            (args.get("content") or "").encode("utf-8")).decode("ascii")
        if win:
            return {"cmd": (
                f"[IO.File]::WriteAllBytes((Join-Path (Get-Location) "
                f"{_psq(path)}), [Convert]::FromBase64String({_psq(blob)}))")}
        return {"cmd": (
            f"{PY} -c 'import base64,sys;open(sys.argv[1],\"wb\")"
            f".write(base64.b64decode(sys.argv[2]))' "
            f"{_shq(path)} {_shq(blob)}")}

    if name == "run_tests":
        return {"cmd": f"{PY} -m pytest -q"}

    if name == "list_dir":
        path = args.get("path") or "."
        if win:
            return {"cmd": f"Get-ChildItem -Name -LiteralPath {_psq(path)}"}
        return {"cmd": f"ls -la {_shq(path)}"}

    return None


# Recognise our own generated commands coming back in the transcript, so the
# history replayed to the model looks like its training transcripts rather
# than like shell invocations it has never seen.
_RE_READ = re.compile(r"sys\.stdout\.write\(open\(sys\.argv\[1\]")
_RE_WRITE = re.compile(r"base64\.b64decode")
_RE_TESTS = re.compile(r"-m pytest")
_RE_LS = re.compile(r"os\.listdir")


def codex_call_to_model(cmd: str) -> dict:
    cmd = cmd or ""
    if _RE_WRITE.search(cmd):
        return {"name": "write_file", "args": {"path": "solution.py"}}
    if _RE_READ.search(cmd):
        return {"name": "read_file", "args": {"path": "solution.py"}}
    if _RE_TESTS.search(cmd):
        return {"name": "run_tests", "args": {}}
    if _RE_LS.search(cmd):
        return {"name": "list_dir", "args": {"path": "."}}
    return {"name": "run_tests", "args": {}}


# ------------------------------------------------------------ inbound parsing
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for blk in content or []:
        if isinstance(blk, dict) and blk.get("type") in (
                "input_text", "output_text", "text", "summary_text"):
            parts.append(blk.get("text", ""))
    return "\n".join(p for p in parts if p)


def input_to_turns(body) -> list:
    """Codex `input` items -> (role, text) turns in the MODEL's convention.

    `instructions` (21,026 chars, measured) and the tool schemas are dropped
    entirely: the model gets its own trained system prompt instead, for the
    same reason as the Anthropic path -- the measured failure was
    out-of-distribution prompting, not capability. The developer-role item is
    Codex's skills/scaffolding block and is dropped with them.
    """
    turns = []
    for item in body.get("input") or []:
        itype = item.get("type")
        if itype == "message":
            role = item.get("role")
            if role == "developer":
                continue                      # client scaffolding, not content
            txt = _text_of(item.get("content"))
            txt = re.sub(r"<environment_context>.*?</environment_context>", "",
                         txt, flags=re.DOTALL)
            txt = re.sub(r"<skills_instructions>.*?</skills_instructions>", "",
                         txt, flags=re.DOTALL).strip()
            if txt:
                turns.append(("user" if role != "assistant" else "assistant",
                              txt))
        elif itype == "function_call":
            try:
                a = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                a = {}
            payload = codex_call_to_model(a.get("cmd", ""))
            turns.append(("assistant",
                          "```tool\n" + json.dumps(payload) + "\n```"))
        elif itype == "function_call_output":
            out = item.get("output")
            if isinstance(out, dict):
                out = out.get("content") or json.dumps(out)
            turns.append(("tool", str(out or "")))
    return turns


# ----------------------------------------------------------- outbound events
def build_events(text: str, cwd: str, model_name: str, split_fn):
    """SSE events for one Responses-API turn.

    `split_fn` is cli_adapter.split_model_output -- shared so that BOTH CLI
    adapters decide what counts as a tool call using the harness that trained
    the model, rather than each growing its own notion of one.
    """
    prose, call = split_fn(text)
    mapped = model_call_to_codex(call[0], call[1], cwd) if call else None

    rid = "resp_" + uuid.uuid4().hex[:20]
    items = []
    if mapped is not None:
        if prose:
            items.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:16],
                          "role": "assistant", "status": "completed",
                          "content": [{"type": "output_text", "text": prose}]})
        items.append({"type": "function_call",
                      "id": "fc_" + uuid.uuid4().hex[:16],
                      "call_id": "call_" + uuid.uuid4().hex[:16],
                      "name": "exec_command",
                      "arguments": json.dumps(mapped),
                      "status": "completed"})
    else:
        items.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:16],
                      "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text",
                                   "text": prose or (text or "").strip()
                                   or "(empty response)"}]})

    resp = {"id": rid, "object": "response", "status": "in_progress",
            "model": model_name, "output": []}
    events = [("response.created", {"type": "response.created",
                                    "response": resp})]
    for idx, item in enumerate(items):
        # The "added" item must carry an empty `content` array for messages.
        # Omitting it made Codex log "OutputTextDelta without active item" and
        # drop the text: it had no open content part to attach the delta to.
        added = {k: v for k, v in item.items()
                 if k in ("type", "id", "call_id", "name", "role", "status")}
        if item["type"] == "message":
            added["content"] = []
            added["status"] = "in_progress"
        events.append(("response.output_item.added",
                       {"type": "response.output_item.added",
                        "output_index": idx, "item": added}))
        if item["type"] == "message":
            events.append(("response.content_part.added",
                           {"type": "response.content_part.added",
                            "output_index": idx, "content_index": 0,
                            "item_id": item["id"],
                            "part": {"type": "output_text", "text": "",
                                     "annotations": []}}))
            events.append(("response.output_text.delta",
                           {"type": "response.output_text.delta",
                            "output_index": idx, "content_index": 0,
                            "item_id": item["id"],
                            "delta": item["content"][0]["text"]}))
            events.append(("response.output_text.done",
                           {"type": "response.output_text.done",
                            "output_index": idx, "content_index": 0,
                            "item_id": item["id"],
                            "text": item["content"][0]["text"]}))
            events.append(("response.content_part.done",
                           {"type": "response.content_part.done",
                            "output_index": idx, "content_index": 0,
                            "item_id": item["id"],
                            "part": {"type": "output_text",
                                     "text": item["content"][0]["text"],
                                     "annotations": []}}))
        else:
            events.append(("response.function_call_arguments.delta",
                           {"type": "response.function_call_arguments.delta",
                            "output_index": idx, "item_id": item["id"],
                            "delta": item["arguments"]}))
            events.append(("response.function_call_arguments.done",
                           {"type": "response.function_call_arguments.done",
                            "output_index": idx, "item_id": item["id"],
                            "arguments": item["arguments"]}))
        events.append(("response.output_item.done",
                       {"type": "response.output_item.done",
                        "output_index": idx, "item": item}))
    done = dict(resp)
    done["status"] = "completed"
    done["output"] = items
    done["usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    events.append(("response.completed", {"type": "response.completed",
                                          "response": done}))
    tool_name = (call[0] if call else None)
    return events, ("exec_command" if mapped else None), tool_name
