"""Translate between Claude Code's protocol and this model's trained one.

THE PROBLEM THIS SOLVES. Two mismatches stop the 85M checkpoint from driving a
coding CLI, and neither is about capability:

  1. PROMPT DISTRIBUTION. Claude Code sends ~18,850 tokens (measured) with 27
     JSON-Schema tool declarations. The model was SFT'd on a ~500-token system
     prompt with exactly four tools. Given the CLI's prompt it produced prose
     about an invented "My User GPT" -- not a capability failure, an
     out-of-distribution one. Given its OWN prompt it writes correct code.

  2. TOOL CHANNEL. The model emits a fenced ```tool {"name":..,"args":..}```
     block, which is what it was trained on. Claude Code only executes
     `tool_use` CONTENT BLOCKS with `stop_reason="tool_use"`. Returning the
     model's text verbatim means the CLI renders a tool call as prose and runs
     nothing, so every episode looks like the model refused to act.

So the adapter shows the model its own training distribution inbound, and
speaks Claude Code's protocol outbound. The alternative -- retraining the model
on the CLI's format -- is strictly more expensive and would still leave the
tool-channel bug.

WHAT IS DELIBERATELY NOT HIDDEN: if the model emits no tool call, or emits one
naming a tool that does not exist, that is passed through as a plain text reply
rather than being repaired into something plausible. A silent repair here would
make the model look more capable than it is, which is the exact failure mode
this project has already published twice.
"""
from __future__ import annotations

import json
import os
import re
import uuid

# ---------------------------------------------------------------- tool mapping
#
# The model knows four tools. Claude Code offers ~27. Rather than teach the
# model a new vocabulary, map its four onto CLI tools that are always present.
# `Bash` is the target for list_dir and run_tests because it is unconditionally
# available, whereas LS/Glob availability varies by client version and settings.

_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


def _abs(path: str, cwd: str) -> str:
    """Resolve the model's relative path against the CLIENT's cwd.

    Claude Code's Read/Write require ABSOLUTE paths; the model emits relative
    ones ("solution.py") because that is what its repo scenarios look like.

    THE PLATFORM HERE IS THE CLIENT'S, NOT THIS PROCESS'S. The server runs on
    Linux while the CLI may run on Windows, and `os.path.isabs("C:\\\\x")` is
    False on Linux -- so os.path.join would have produced
    "/server/cwd/C:\\Users\\..." and every file operation would have failed
    with a path that looks absurd but is hard to trace back to here. Decide
    Windows-ness from the cwd string and join with that platform's separator.
    """
    if not path:
        return cwd
    win = bool(_WIN_ABS.match(cwd or ""))
    if win:
        if _WIN_ABS.match(path):
            return path
        sep = "\\" if "\\" in (cwd or "") else "/"
        return (cwd or "").rstrip("\\/") + sep + path.replace("/", sep)
    if os.path.isabs(path) or path.startswith("/"):
        return path
    return os.path.normpath(os.path.join(cwd, path)).replace("\\", "/")


def model_call_to_cli(name: str, args: dict, cwd: str):
    """(model tool, args) -> (CLI tool, input) or None if unmappable."""
    args = args or {}
    if name == "read_file":
        return "Read", {"file_path": _abs(args.get("path", ""), cwd)}
    if name == "write_file":
        # `content` is the field the policy historically omitted; if it is
        # missing we must NOT invent one -- see the module docstring.
        if "content" not in args:
            return None
        return "Write", {"file_path": _abs(args.get("path", ""), cwd),
                         "content": args.get("content")}
    if name == "list_dir":
        target = _abs(args.get("path", "."), cwd)
        return "Bash", {"command": f"ls -la {json.dumps(target)}",
                        "description": "List directory contents"}
    if name == "run_tests":
        return "Bash", {"command": "python -m pytest -q 2>&1 | tail -20",
                        "description": "Run the test suite"}
    return None


# Reverse direction: a transcript replayed back to the model must look like the
# transcript it was trained on, so CLI tool names are rewritten to the model's.
_CLI_TO_MODEL = {
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "write_file",
    "Bash": "run_tests",
    "Glob": "list_dir",
    "LS": "list_dir",
    "Grep": "list_dir",
}


def _basename(path: str) -> str:
    """Basename for EITHER platform's separator.

    os.path.basename runs on the SERVER. On Linux it does not treat `\\` as a
    separator, so a Windows client's "C:\\Users\\me\\proj\\solution.py" came
    back whole. That full path was then replayed into the model's transcript,
    where training had only ever shown "solution.py" -- and the model started
    emitting 60-character absolute paths and omitting `content` entirely.
    Every write_file became UNMAPPABLE and no fix was ever applied.
    """
    return re.split(r"[\\/]", (path or "").strip())[-1]


def cli_call_to_model(name: str, tool_input: dict) -> dict:
    """A CLI tool_use block -> the model's fenced-tool JSON payload."""
    ti = tool_input or {}
    mapped = _CLI_TO_MODEL.get(name, name)
    if mapped == "read_file":
        return {"name": "read_file",
                "args": {"path": _basename(ti.get("file_path", ""))}}
    if mapped == "write_file":
        return {"name": "write_file",
                "args": {"path": _basename(ti.get("file_path", "")),
                         "content": ti.get("content", ti.get("new_string", ""))}}
    if mapped == "list_dir":
        return {"name": "list_dir", "args": {"path": "."}}
    if mapped == "run_tests":
        return {"name": "run_tests", "args": {}}
    return {"name": mapped, "args": ti}


# --------------------------------------------------------------- tool results
_LINENO = re.compile(r"^\s*\d+\t", re.MULTILINE)


_PYTEST_MIXED = re.compile(r"(\d+)\s+failed,\s*(\d+)\s+passed")
_PYTEST_PASSED = re.compile(r"(?<!no )(\d+)\s+passed")
_PYTEST_FAILED = re.compile(r"(\d+)\s+failed")
_PYTEST_ERROR = re.compile(r"(\d+)\s+error")


def normalise_test_output(text: str) -> str | None:
    """Raw pytest output -> the "N/M tests passed" the model was trained on.

    THE TOOL RESULT FORMAT MATTERS AS MUCH AS THE TOOL CALL FORMAT, and this
    was missed for a layer longer. The model's own `run_tests` returns exactly
    "3/3 tests passed". Mapped onto Claude Code's `Bash` it instead receives
    pytest's real output -- "1 failed in 0.08s", assertion tracebacks, summary
    lines -- which it has never seen.

    Measured consequence: across 10 CLI trials the model called
    read_file -> run_tests -> read_file -> run_tests and NEVER write_file,
    because it could not tell from the output whether anything had happened.
    0/10 solved, while the same model in the harness solves 56.7%.

    Returns None when the text is not a pytest summary, so ordinary command
    output passes through untouched.
    """
    t = text or ""
    m = _PYTEST_MIXED.search(t)
    if m:
        failed, passed = int(m.group(1)), int(m.group(2))
        return f"{passed}/{passed + failed} tests passed"
    m = _PYTEST_FAILED.search(t)
    if m:
        failed = int(m.group(1))
        p = _PYTEST_PASSED.search(t)
        passed = int(p.group(1)) if p else 0
        return f"{passed}/{passed + failed} tests passed"
    m = _PYTEST_ERROR.search(t)
    if m:
        # A collection error is zero of an unknown total; say so plainly rather
        # than inventing a denominator.
        return "0 tests passed (the test run errored)"
    m = _PYTEST_PASSED.search(t)
    if m:
        passed = int(m.group(1))
        return f"{passed}/{passed} tests passed"
    return None


def strip_line_numbers(text: str) -> str:
    """Claude Code's Read returns cat -n style output ("   1\\tdef add(a, b):").

    The model's trained `read_file` returns the RAW file, so the numbered form
    is out of distribution in the one place it matters most -- the content it
    is about to rewrite. Observed directly: handed a numbered file, the model
    wrote back something unrelated to it.

    Only a leading "<digits><TAB>" is removed, which is the exact shape the
    client emits; a tab inside the line's own text is untouched.
    """
    return _LINENO.sub("", text or "")


# ------------------------------------------------------------- outbound blocks
#
# THE PARSER IS IMPORTED, NOT REIMPLEMENTED. The RL and SFT harnesses decide
# what counts as a tool call with agent.parse_tool_call and agent.TOOL_FENCE.
# A second, parallel parser here would drift from it, and the drift would be
# invisible: serving would accept calls training rejected (or the reverse), so
# the model's measured tool-call rate would stop predicting its served
# behaviour. This was written as a private regex first; reusing the trained
# definition is the only way the two stay the same thing.
from nanocoder.agent import TOOL_FENCE, parse_tool_call


def split_model_output(text: str):
    """Return (prose_before_call, (name, args) or None)."""
    text = text or ""
    call = parse_tool_call(text)
    m = TOOL_FENCE.search(text)
    prose = (text[:m.start()] if m else text).strip()
    if call is None:
        # Malformed or absent: hand back the whole text, unrepaired.
        return text.strip(), None
    return prose, call


def build_response(text: str, cwd: str, model_name: str, msg_id: str,
                   n_prompt: int):
    """Anthropic-shaped response, with a real tool_use block when the model
    made a call. `stop_reason` must be "tool_use" or the CLI will not execute
    it -- returning the right blocks with stop_reason "end_turn" silently does
    nothing, which cost an hour to notice the first time."""
    prose, call = split_model_output(text)
    content = []
    stop = "end_turn"
    mapped = None
    if call is not None:
        mapped = model_call_to_cli(call[0], call[1], cwd)
    if mapped is not None:
        if prose:
            content.append({"type": "text", "text": prose})
        content.append({"type": "tool_use",
                        "id": "toolu_" + uuid.uuid4().hex[:20],
                        "name": mapped[0], "input": mapped[1]})
        stop = "tool_use"
    else:
        # No call, or one we could not map: hand back the text as it is.
        content.append({"type": "text", "text": prose or (text or "").strip()
                        or "(empty response)"})
    return {
        "id": msg_id, "type": "message", "role": "assistant",
        "model": model_name, "content": content,
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": n_prompt,
                  "output_tokens": max(1, len(text or "") // 4)},
    }, (mapped[0] if mapped else None), (call[0] if call else None)


# -------------------------------------------------------------- inbound prompt
def extract_cwd(system_text: str, fallback: str) -> str:
    """Find the CLIENT's working directory in the client's own prompt.

    Getting this wrong is silent and total: the fallback is the SERVER's cwd,
    so every Read/Write is sent an absolute path that does not exist on the
    client, and the CLI answers "File does not exist" -- which looks exactly
    like the model asking for the wrong file. Measured once, at the cost of a
    full CLI round trip.

    Claude Code writes it as:
        - Primary working directory: C:\\Users\\me\\proj
    which none of the first patterns tried here matched. Note the patterns take
    the rest of the LINE rather than \\S+, because real paths contain spaces
    ("Program Files") and a \\S+ capture truncates them mid-path.
    """
    for pat in (r"Primary working directory:[ \t]*(.+)",
                r"Working directory:[ \t]*(.+)",
                r"working directory is[ \t]*(.+)",
                r"cwd:[ \t]*(.+)"):
        m = re.search(pat, system_text or "")
        if m:
            return m.group(1).strip().rstrip(".,;")
    return fallback
