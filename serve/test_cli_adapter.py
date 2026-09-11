"""Tests for the Claude Code protocol adapter. Run: python serve/test_cli_adapter.py

These are written as a file rather than typed inline because the fixtures
contain JSON-escaped newlines, and passing those through a shell heredoc
corrupts them into real newlines -- which silently turns a valid fixture into
an invalid one and makes a correct parser look broken.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli_adapter as A

CWD = "/repo"


def fence(payload: dict) -> str:
    """Build a call the way the SFT trajectories build one: json.dumps, so
    newlines inside `content` are properly escaped."""
    return "```tool\n" + json.dumps(payload) + "\n```"


def test_write_file_becomes_a_real_tool_use_block():
    body = "def f():\n    return 1\n"          # genuine newlines in the content
    text = "I will fix it.\n" + fence(
        {"name": "write_file", "args": {"path": "solution.py", "content": body}})
    resp, cli, model = A.build_response(text, CWD, "m", "msg_1", 100)
    assert model == "write_file" and cli == "Write", (model, cli)
    # stop_reason MUST be tool_use or the CLI renders the call and runs nothing
    assert resp["stop_reason"] == "tool_use", resp["stop_reason"]
    assert [b["type"] for b in resp["content"]] == ["text", "tool_use"]
    tu = resp["content"][-1]
    assert tu["input"]["file_path"] == "/repo/solution.py", tu["input"]
    assert tu["input"]["content"] == body, "content must survive verbatim"
    assert tu["id"].startswith("toolu_")
    print("write_file -> Write, absolute path, content intact: PASS")


def test_read_and_tests_map_to_cli_tools():
    resp, cli, _ = A.build_response(
        fence({"name": "read_file", "args": {"path": "solution.py"}}),
        CWD, "m", "i", 1)
    assert cli == "Read"
    assert resp["content"][0]["input"]["file_path"] == "/repo/solution.py"

    resp, cli, _ = A.build_response(
        fence({"name": "run_tests", "args": {}}), CWD, "m", "i", 1)
    assert cli == "Bash" and "pytest" in resp["content"][0]["input"]["command"]

    resp, cli, _ = A.build_response(
        fence({"name": "list_dir", "args": {"path": "."}}), CWD, "m", "i", 1)
    assert cli == "Bash" and "ls -la" in resp["content"][0]["input"]["command"]
    print("read_file/run_tests/list_dir map to Read/Bash/Bash: PASS")


def test_a_write_with_no_content_is_not_invented():
    """The policy historically emitted write_file with no `content` 19/19
    times. Filling one in here would execute a fabricated edit and make the
    model look capable; it must degrade to text instead."""
    resp, cli, model = A.build_response(
        fence({"name": "write_file", "args": {"path": "s.py"}}),
        CWD, "m", "i", 1)
    assert model == "write_file", "the call should still be REPORTED"
    assert cli is None, "but it must not be mapped to an executable tool"
    assert resp["stop_reason"] == "end_turn"
    assert resp["content"][0]["type"] == "text"
    print("write_file without content is reported, not executed: PASS")


def test_absent_and_malformed_calls_degrade_to_text():
    for text in ("Here is some prose.",
                 "```tool\n{\"name\": \"write_file\", \"args\": {oops\n```",
                 "```tool\n[1, 2, 3]\n```",
                 "```tool\n\"just a string\"\n```"):
        resp, cli, model = A.build_response(text, CWD, "m", "i", 1)
        assert cli is None and model is None, (text[:30], cli, model)
        assert resp["stop_reason"] == "end_turn"
        assert resp["content"][0]["type"] == "text"
    print("absent/malformed/non-object payloads degrade to text: PASS")


def test_unknown_tool_is_not_silently_dropped():
    resp, cli, model = A.build_response(
        fence({"name": "rm_rf", "args": {"path": "/"}}), CWD, "m", "i", 1)
    assert model == "rm_rf", "an unmapped call must still be reported"
    assert cli is None, "and must never be executed"
    assert resp["stop_reason"] == "end_turn"
    print("unknown tool reported but never executed: PASS")


def test_round_trip_back_into_the_models_format():
    """A CLI transcript replayed to the model must look like its training
    transcripts, or the history is as out-of-distribution as the prompt was."""
    p = A.cli_call_to_model("Write", {"file_path": "/repo/solution.py",
                                      "content": "x=1"})
    assert p == {"name": "write_file",
                 "args": {"path": "solution.py", "content": "x=1"}}, p
    assert A.cli_call_to_model("Bash", {"command": "pytest"}) == \
        {"name": "run_tests", "args": {}}
    assert A.cli_call_to_model("Read", {"file_path": "/a/b/s.py"}) == \
        {"name": "read_file", "args": {"path": "s.py"}}
    # Edit carries new_string rather than content
    p = A.cli_call_to_model("Edit", {"file_path": "/repo/s.py",
                                     "new_string": "y=2"})
    assert p["args"]["content"] == "y=2", p
    print("CLI tool_use round-trips into the model's fenced format: PASS")


def test_parser_is_the_trained_one():
    """Guards against someone reintroducing a private regex: the adapter must
    agree with the harness that trained the model."""
    from blackwell_lm.agent import parse_tool_call
    text = fence({"name": "read_file", "args": {"path": "x.py"}})
    assert A.split_model_output(text)[1] == parse_tool_call(text)
    print("adapter and training harness share one parser: PASS")


def test_line_numbers_are_stripped_from_read_results():
    """Claude Code's Read is cat -n style; the model's read_file is raw."""
    numbered = "     1\tdef add(a, b):\n     2\t    return a - b\n"
    assert A.strip_line_numbers(numbered) == "def add(a, b):\n    return a - b\n"
    # a tab inside the line's own text must survive
    assert A.strip_line_numbers("  3\tx = 'a\tb'") == "x = 'a\tb'"
    # unnumbered output passes through untouched
    plain = "3 passed in 0.10s"
    assert A.strip_line_numbers(plain) == plain
    assert A.strip_line_numbers("") == ""
    print("Read line numbers stripped, inner tabs preserved: PASS")


def test_cwd_extraction():
    assert A.extract_cwd("x\nWorking directory: /home/u/proj\ny",
                         "/fb") == "/home/u/proj"
    assert A.extract_cwd("nothing here", "/fb") == "/fb"

    # The phrasing Claude Code actually uses, verbatim from a captured
    # request. Missing this sent every Read the SERVER's cwd, and the CLI
    # replied "File does not exist" -- indistinguishable from the model
    # naming the wrong file.
    real = ("# Environment\nYou have been invoked in the following "
            "environment: \n - Primary working directory: "
            "C:\\Users\\E20263395\\apps\\blackwell-nanogpt\n"
            " - Is a git repository: true\n - Platform: win32\n")
    assert A.extract_cwd(real, "/fb") == \
        "C:\\Users\\E20263395\\apps\\blackwell-nanogpt", A.extract_cwd(real, "/fb")

    # a path containing spaces must not be truncated at the first one
    spaced = " - Primary working directory: C:\\Program Files\\my proj\n - x\n"
    assert A.extract_cwd(spaced, "/fb") == "C:\\Program Files\\my proj"
    # an already-absolute path from the model is left alone
    resp, cli, _ = A.build_response(
        fence({"name": "read_file", "args": {"path": "/etc/hosts"}}),
        CWD, "m", "i", 1)
    assert resp["content"][0]["input"]["file_path"] == "/etc/hosts"
    print("cwd extraction and absolute-path passthrough: PASS")


def test_windows_client_paths_are_not_mangled():
    """The server runs on Linux; the CLI may run on Windows. Joining a Windows
    cwd with posixpath produced "/linux/cwd/C:\\Users\\..." -- a path that
    fails in a way that is hard to trace back to the adapter."""
    # Hardcoded Windows-style regardless of the platform running the test:
    # the point is that the SERVER's os.sep must not influence the result.
    win = "C:\\Users\\me\\proj"
    resp, cli, _ = A.build_response(
        fence({"name": "read_file", "args": {"path": "solution.py"}}),
        win, "m", "i", 1)
    got = resp["content"][0]["input"]["file_path"]
    assert got == "C:\\Users\\me\\proj\\solution.py", got

    # an already-absolute Windows path is left alone
    resp, _, _ = A.build_response(
        fence({"name": "read_file", "args": {"path": "D:\\other\\x.py"}}),
        win, "m", "i", 1)
    assert resp["content"][0]["input"]["file_path"] == "D:\\other\\x.py"

    # and a posix cwd still behaves
    resp, _, _ = A.build_response(
        fence({"name": "read_file", "args": {"path": "s.py"}}),
        "/repo/sub", "m", "i", 1)
    assert resp["content"][0]["input"]["file_path"] == "/repo/sub/s.py"
    print("windows client paths preserved, posix unaffected: PASS")


if __name__ == "__main__":
    test_write_file_becomes_a_real_tool_use_block()
    test_read_and_tests_map_to_cli_tools()
    test_a_write_with_no_content_is_not_invented()
    test_absent_and_malformed_calls_degrade_to_text()
    test_unknown_tool_is_not_silently_dropped()
    test_round_trip_back_into_the_models_format()
    test_parser_is_the_trained_one()
    test_line_numbers_are_stripped_from_read_results()
    test_cwd_extraction()
    test_windows_client_paths_are_not_mangled()
    print("\nAll cli_adapter tests passed.")
