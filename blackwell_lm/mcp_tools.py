"""MCP-shaped tool definitions over a real scratch repo.

WHY THIS REPLACES THE AD-HOC THREE TOOLS. The existing agent has run_python /
run_tests / read_file with hand-rolled argument handling. That is enough to prove
a loop runs, and it is NOT the shape real coding agents (Codex, Claude Code,
anything MCP-based) present: a declared tool list with JSON-Schema inputs, over a
filesystem the agent must navigate. Training on the toy shape teaches a
convention that transfers nowhere.

So tools are declared the way MCP declares them -- name, description,
input_schema -- and the system prompt is GENERATED from those declarations
rather than hand-written. That means the prompt can never drift from the actual
dispatcher, which is the same class of bug as the SFT/RL prompt-format drift
this project already paid for once.

WHAT IS DELIBERATELY NOT HERE: no network, no package installation, no git. The
sandbox is a robustness boundary, not a containment boundary (see sandbox.py),
and every extra verb is another thing a policy can be rewarded for abusing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from blackwell_lm.sandbox import posix_limits_active, run_python

MAX_OUTPUT_CHARS = 1200
ERR = "error: "

# MCP-style declarations. `input_schema` is real JSON Schema so the same list can
# drive an MCP server, the prompt text, and argument validation from one source.
TOOLS = [
    {
        "name": "list_dir",
        "description": "List files in a directory of the working repo.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "relative path, '.' for root"}},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the working repo.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite a file in the working repo with new contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": ("Run the repo's test suite and report how many tests passed. "
                        "You do not write the tests and cannot see them."),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def render_system_prompt(tools=TOOLS) -> str:
    """Build the agent system prompt FROM the tool declarations.

    Generated, not hand-written, so the prompt cannot describe a tool the
    dispatcher does not implement (or omit one it does).
    """
    lines = [
        "You are a coding agent working in a repository. Call a tool by emitting "
        "a fenced block:",
        "```tool",
        '{"name": "read_file", "args": {"path": "solution.py"}}',
        "```",
        "",
        "Available tools:",
    ]
    for t in tools:
        props = t["input_schema"].get("properties", {})
        args = ", ".join(props.keys())
        lines.append(f"  {t['name']}({args}) - {t['description']}")
    lines += [
        "",
        "Work by reading the code, fixing it with write_file, then run_tests to "
        "check. When the tests pass, reply with your final answer as a ```python "
        "block containing the fixed file.",
    ]
    return "\n".join(lines)


def _validate(tool: dict, args: dict):
    """Minimal JSON-Schema check: required keys present and of declared type.

    Returned as an error STRING rather than raised, because a malformed tool call
    must be data the policy can read and react to -- early in training almost
    every call is malformed, and raising would mean the episodes that fail are
    exactly the ones that never produce a gradient.
    """
    schema = tool["input_schema"]
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            # Show the EXPECTED CALL, not just the missing key. A bare "requires
            # 'content'" told the policy nothing it could act on: measured, it
            # emitted {"path": "solution.py"} and no content 19 times out of 19,
            # never once recovering. An error that demonstrates the shape is the
            # difference between a dead end and in-context self-correction --
            # which is itself the agentic skill worth training.
            example = {k: ("<the full file contents>" if k == "content"
                           else f"<{k}>") for k in props}
            shape = json.dumps({"name": tool["name"], "args": example})
            return f"{ERR}{tool['name']} requires '{key}'. Expected call: {shape}"
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "object": dict, "array": list}
    for key, spec in schema.get("properties", {}).items():
        if key in args and spec.get("type") in types:
            if not isinstance(args[key], types[spec["type"]]):
                return (f"{ERR}{tool['name']}.{key} must be "
                        f"{spec['type']}, got {type(args[key]).__name__}")
    return None


def _truncate(s: str) -> str:
    s = s or ""
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return s[:half] + f"\n...[{len(s) - MAX_OUTPUT_CHARS} chars elided]...\n" + s[-half:]


class RepoToolBox:
    """Tools scoped to one scratch repo directory.

    Every path is resolved with realpath BEFORE the containment check, so a
    symlink cannot point out of the repo -- os.path.normpath alone accepts that,
    which is the standard way this check is written wrong.
    """

    def __init__(self, repo: str, tests: list[str] | None = None,
                 setup: str = "", timeout: float = 10.0):
        self.repo = os.path.realpath(repo)
        self.tests = list(tests or [])
        self.setup = setup
        self.timeout = timeout
        self.by_name = {t["name"]: t for t in TOOLS}

    # ---------------------------------------------------------------- helpers

    def _resolve(self, rel: str):
        target = os.path.realpath(os.path.join(self.repo, rel or "."))
        if not (target == self.repo or target.startswith(self.repo + os.sep)):
            return None
        return target

    # ------------------------------------------------------------------ tools

    def list_dir(self, path: str = "."):
        t = self._resolve(path)
        if t is None:
            return f"{ERR}path escapes the repo"
        if not os.path.isdir(t):
            return f"{ERR}not a directory: {path}"
        names = sorted(os.listdir(t))
        return _truncate("\n".join(names) or "(empty)")

    def read_file(self, path: str = ""):
        t = self._resolve(path)
        if t is None:
            return f"{ERR}path escapes the repo"
        if not os.path.isfile(t):
            return f"{ERR}no such file: {path}"
        with open(t, encoding="utf-8", errors="replace") as fh:
            return _truncate(fh.read())

    def write_file(self, path: str = "", content: str = ""):
        t = self._resolve(path)
        if t is None:
            return f"{ERR}path escapes the repo"
        os.makedirs(os.path.dirname(t) or self.repo, exist_ok=True)
        with open(t, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"wrote {len(content)} chars to {path}"

    def run_tests(self, **kw):
        """Run the HARNESS-owned tests against the repo's current state.

        The tests are never supplied by the model. Letting it pass its own was a
        measured failure: the policy learned to generate them, generated broken
        ones, and correct code scored zero against a hallucinated assert.
        """
        if not self.tests:
            return f"{ERR}no tests configured for this task"
        sol = os.path.join(self.repo, "solution.py")
        if not os.path.isfile(sol):
            return f"{ERR}solution.py not found in the repo"
        with open(sol, encoding="utf-8", errors="replace") as fh:
            code = fh.read()
        from blackwell_lm.sandbox import run_tests as _rt
        passed, total, detail = _rt(code, self.tests, timeout=self.timeout,
                                    setup=self.setup)
        note = ("\n(note: the tests are fixed by the task; yours were ignored)"
                if "tests" in kw else "")
        return _truncate(f"{passed}/{total} tests passed"
                         + (f"\n{detail}" if detail else "") + note)

    # --------------------------------------------------------------- dispatch

    def dispatch(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output_text, ok). `ok` reports whether the call WORKED, not
        merely whether dispatch avoided an exception."""
        tool = self.by_name.get(name)
        if tool is None:
            return f"{ERR}unknown tool {name!r}; available: " \
                   f"{', '.join(self.by_name)}", False
        bad = _validate(tool, args if isinstance(args, dict) else {})
        if bad:
            return bad, False
        fn = getattr(self, name)
        try:
            out = fn(**{k: v for k, v in args.items()
                        if k in tool["input_schema"].get("properties", {})}
                     ) if name != "run_tests" else fn(**args)
        except TypeError as e:
            return f"{ERR}bad arguments for {name}: {e}", False
        except Exception as e:                 # a tool must never kill an episode
            return f"{ERR}{type(e).__name__}: {e}", False
        return out, not str(out).startswith(ERR)
