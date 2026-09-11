"""Tests for the OpenAI Responses / Codex CLI adapter.

Run: python serve/test_codex_adapter.py

A file, not an inline heredoc: the fixtures contain Windows paths and
JSON-escaped newlines, and a shell heredoc mangles "C:\\Users" into an invalid
\\U escape.
"""
import base64
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli_adapter as A
import codex_adapter as C

WIN = "C:\\Users\\me\\proj"
POSIX = "/repo"

# quotes, a backslash escape, a newline, $, backtick and %: everything that
# breaks when a command line is built by string interpolation
NASTY = "def add(a, b):\n    s = \"a'b\\\"c\"\n    return a + b  # $HOME `x` %PATH%\n"


def call(payload, cwd):
    text = "```tool\n" + json.dumps(payload) + "\n```"
    events, codex_tool, model_tool = C.build_events(
        text, cwd, "m", A.split_model_output)
    fcs = [e[1]["item"] for e in events
           if e[0] == "response.output_item.done"
           and e[1]["item"]["type"] == "function_call"]
    cmd = json.loads(fcs[0]["arguments"])["cmd"] if fcs else None
    return cmd, codex_tool, model_tool, events


def test_windows_commands_contain_no_double_quotes():
    """Codex runs Windows commands as powershell.exe -Command "<cmd>", so a
    double quote inside <cmd> needs a second round of escaping -- and its
    policy rejected a nested `python -c "..."` form outright
    ("blocked by policy") before it ever ran."""
    for payload in ({"name": "read_file", "args": {"path": "solution.py"}},
                    {"name": "write_file",
                     "args": {"path": "solution.py", "content": NASTY}},
                    {"name": "list_dir", "args": {"path": "."}},
                    {"name": "run_tests", "args": {}}):
        cmd, _, _, _ = call(payload, WIN)
        assert cmd is not None, payload
        assert '"' not in cmd, (payload["name"], cmd)
    print("windows commands use single quotes only: PASS")


def test_file_content_survives_base64_verbatim():
    cmd, _, _, _ = call({"name": "write_file",
                         "args": {"path": "solution.py", "content": NASTY}}, WIN)
    blob = re.findall(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", cmd)[0]
    assert base64.b64decode(blob).decode("utf-8") == NASTY

    cmd, _, _, _ = call({"name": "write_file",
                         "args": {"path": "s.py", "content": NASTY}}, POSIX)
    blob = re.findall(r"'([A-Za-z0-9+/=]{20,})'", cmd)[-1]
    assert base64.b64decode(blob).decode("utf-8") == NASTY
    print("content survives base64 verbatim on both platforms: PASS")


def test_single_quote_in_a_path_is_escaped():
    cmd, _, _, _ = call({"name": "read_file", "args": {"path": "it's.py"}}, WIN)
    assert "'it''s.py'" in cmd, cmd
    cmd, _, _, _ = call({"name": "read_file", "args": {"path": "it's.py"}}, POSIX)
    assert "'it'\\''s.py'" in cmd, cmd
    print("a quote in the path is escaped, not left to break the literal: PASS")


def test_write_without_content_is_not_invented():
    cmd, ctool, mtool, events = call(
        {"name": "write_file", "args": {"path": "s.py"}}, WIN)
    assert cmd is None and ctool is None
    assert mtool == "write_file", "the attempt must still be reported"
    kinds = [e[1]["item"]["type"] for e in events
             if e[0] == "response.output_item.done"]
    assert kinds == ["message"], kinds
    print("write_file without content reported, never executed: PASS")


def test_text_turn_emits_the_content_part_events():
    """Omitting content_part.added made Codex log
    'OutputTextDelta without active item' and drop the text entirely."""
    _, ctool, mtool, events = call({"name": "nope", "args": {}}, WIN)
    assert ctool is None and mtool == "nope"
    kinds = [k for k, _ in events]
    for needed in ("response.created", "response.output_item.added",
                   "response.content_part.added", "response.output_text.delta",
                   "response.output_text.done", "response.content_part.done",
                   "response.output_item.done", "response.completed"):
        assert needed in kinds, (needed, kinds)
    added = [e[1] for e in events
             if e[0] == "response.output_item.added"][0]["item"]
    assert added.get("content") == [], "message items open with empty content"
    print("text turn emits the full content_part event sequence: PASS")


def test_function_call_event_shape():
    _, ctool, mtool, events = call(
        {"name": "run_tests", "args": {}}, WIN)
    assert ctool == "exec_command" and mtool == "run_tests"
    done = [e[1]["item"] for e in events
            if e[0] == "response.output_item.done"][0]
    assert done["type"] == "function_call"
    assert done["name"] == "exec_command"
    assert done["call_id"].startswith("call_")
    json.loads(done["arguments"])          # must be a JSON *string*
    final = [e[1] for e in events if e[0] == "response.completed"][0]
    assert final["response"]["status"] == "completed"
    assert final["response"]["output"], "completed must carry the output items"
    print("function_call item and completed envelope well formed: PASS")


def test_paths_cannot_escape_the_working_directory():
    """Codex's Windows sandbox backend reports "disabled", so with the bypass
    flag there is NO client-side confinement. The path comes from an 85M model
    whose output is frequently garbage, so containment is enforced here."""
    for bad in ("/etc/passwd", "C:\\Windows\\System32\\drivers\\etc\\hosts",
                "..\\..\\secrets.txt", "../../../etc/shadow",
                "\\\\server\\share\\x", "sub/../../out.py"):
        cmd, ctool, mtool, _ = call(
            {"name": "write_file", "args": {"path": bad, "content": "x"}}, WIN)
        assert cmd is None, (bad, cmd)
        assert ctool is None, bad
        assert mtool == "write_file", "the attempt is still reported"
        cmd, _, _, _ = call({"name": "read_file", "args": {"path": bad}}, WIN)
        assert cmd is None, (bad, cmd)

    # ordinary relative paths, including into a subdirectory, still work
    for ok in ("solution.py", "./solution.py", "pkg/mod.py", "a/b/../c.py"):
        cmd, ctool, _, _ = call(
            {"name": "write_file", "args": {"path": ok, "content": "x"}}, WIN)
        assert cmd is not None and ctool == "exec_command", ok
    print("paths cannot escape the working directory: PASS")


def test_codex_input_is_parsed_back_into_model_turns():
    body = {
        "instructions": "you are codex",
        "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "<skills_instructions>x</skills_instructions>"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "fix solution.py"}]},
            {"type": "function_call", "name": "exec_command",
             "arguments": json.dumps({"cmd": "python -m pytest -q"})},
            {"type": "function_call_output", "output": "1 failed"},
        ],
    }
    turns = C.input_to_turns(body)
    roles = [r for r, _ in turns]
    assert roles == ["user", "assistant", "tool"], turns
    assert "fix solution.py" in turns[0][1]
    assert "run_tests" in turns[1][1], turns[1][1]
    assert turns[2][1] == "1 failed"
    print("developer scaffolding dropped; tool traffic in model format: PASS")


if __name__ == "__main__":
    test_windows_commands_contain_no_double_quotes()
    test_file_content_survives_base64_verbatim()
    test_single_quote_in_a_path_is_escaped()
    test_write_without_content_is_not_invented()
    test_text_turn_emits_the_content_part_events()
    test_function_call_event_shape()
    test_paths_cannot_escape_the_working_directory()
    test_codex_input_is_parsed_back_into_model_turns()
    print("\nAll codex_adapter tests passed.")
