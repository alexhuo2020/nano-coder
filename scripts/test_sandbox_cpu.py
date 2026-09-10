"""Adversarial tests for the sandbox, the execution reward, and the agent loop.

Run these BEFORE ever pointing the sandbox at real model output. Everything the
policy emits gets executed, and reward hacking is not hypothetical: gradient
descent will find whatever this fails to forbid.

These spawn real subprocesses, so they need a real OS. On Linux they exercise
the POSIX resource limits; on Windows those do not exist and the affected tests
SKIP loudly rather than passing vacuously -- a green run on Windows must not be
mistaken for a validated sandbox.
"""
import os
import sys

from blackwell_lm.agent import (AGENT_SYSTEM, MAX_TOOL_CHARS, Episode, ToolBox,
                                episode_reward, parse_tool_call, run_episode)
from blackwell_lm.reward import code_reward, extract_code
from blackwell_lm.sandbox import posix_limits_active, run_python, run_tests

SKIPS = []


def _skip(name, why):
    SKIPS.append(name)
    print(f"{name}: SKIP ({why})")


# ---------------------------------------------------------------- the sandbox

def test_normal_code_runs():
    r = run_python("print('hello'); print(2 + 2)")
    assert r.ok and not r.timed_out, r
    assert "hello" in r.stdout and "4" in r.stdout, r.stdout
    print("test_normal_code_runs: PASS")


def test_infinite_loop_is_killed():
    """The single most likely thing a weak policy emits."""
    r = run_python("while True:\n    pass", timeout=3.0)
    assert r.timed_out, r
    assert not r.ok
    print("test_infinite_loop_is_killed: PASS")


def test_sleep_is_killed_by_wallclock_not_cpu():
    """RLIMIT_CPU never fires on a sleeping process, which is why the wall-clock
    timeout has to exist as well."""
    r = run_python("import time\ntime.sleep(30)", timeout=3.0)
    assert r.timed_out, r
    print("test_sleep_is_killed_by_wallclock_not_cpu: PASS")


def test_memory_bomb_is_stopped():
    if not posix_limits_active():
        return _skip("test_memory_bomb_is_stopped", "no POSIX rlimits on this OS")
    r = run_python("x = bytearray(4_000_000_000)", timeout=20.0, mem_mb=256)
    assert not r.ok, "a 4GB allocation succeeded under a 256MB limit"
    print("test_memory_bomb_is_stopped: PASS")


def test_child_processes_do_not_outlive_the_timeout():
    """Killing only the direct child leaves its children holding memory on the
    GPU box for the rest of the run, so the whole process GROUP is killed."""
    if not posix_limits_active():
        return _skip("test_child_processes_do_not_outlive_the_timeout", "POSIX only")
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(45)'])\n"
        "time.sleep(45)\n"
    )
    r = run_python(code, timeout=3.0)
    assert r.timed_out, r
    print("test_child_processes_do_not_outlive_the_timeout: PASS")


def test_extra_file_names_cannot_escape():
    try:
        run_python("pass", extra_files={"../escaped.py": "x = 1"})
    except ValueError as e:
        assert "escapes the sandbox" in str(e)
    else:
        raise AssertionError("a ../ filename was accepted")
    print("test_extra_file_names_cannot_escape: PASS")


def test_workdir_is_removed_afterwards():
    r = run_python("import os; print(os.getcwd())")
    assert r.ok, r
    workdir = r.stdout.strip().splitlines()[-1]
    assert not os.path.exists(workdir), f"sandbox dir survived: {workdir}"
    print("test_workdir_is_removed_afterwards: PASS")


def test_network_isolation_is_NOT_claimed():
    """Honest boundary test. This sandbox does not create a network namespace,
    so it must not be described as blocking egress. If this ever starts failing
    because egress IS blocked, the docstring in sandbox.py should be updated --
    but until then, do not claim a protection that is not there.
    """
    assert "not a containment boundary" in __import__(
        "blackwell_lm.sandbox", fromlist=["x"]).__doc__
    print("test_network_isolation_is_NOT_claimed: PASS (limitation documented)")


# ------------------------------------------------------------ execution reward

def test_run_tests_gives_partial_credit():
    """Partial credit is what keeps a gradient alive when the policy passes
    almost nothing -- a binary reward makes every group zero-variance."""
    code = "def add(a, b):\n    return a + b\n"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(1, 1) == 99"]
    passed, total, detail = run_tests(code, tests)
    assert (passed, total) == (2, 3), (passed, total, detail)
    print("test_run_tests_gives_partial_credit: PASS (2/3)")


def test_run_tests_reports_setup_failure_distinctly():
    passed, total, detail = run_tests("def broken(:\n", ["assert True"])
    assert passed == 0 and "setup failed" in detail, (passed, detail)
    print(f"test_run_tests_reports_setup_failure_distinctly: PASS ({detail[:40]}...)")


def test_one_failing_test_does_not_mask_the_others():
    code = "def f(x):\n    return x\n"
    tests = ["assert f(1) == 1", "raise RuntimeError('boom')", "assert f(2) == 2"]
    passed, total, _ = run_tests(code, tests)
    assert (passed, total) == (2, 3), (passed, total)
    print("test_one_failing_test_does_not_mask_the_others: PASS")


def test_test_setup_code_is_run_and_not_scored():
    """MBPP ships fixtures in `test_setup_code` that its asserts reference.
    Skipping them makes those problems unsolvable, and the lost reward is
    indistinguishable from the model simply being wrong."""
    code = "def total(xs):\n    return sum(xs)\n"
    tests = ["assert total(DATA) == 6", "assert total([]) == 0"]
    with_setup = run_tests(code, tests, setup="DATA = [1, 2, 3]")
    without = run_tests(code, tests)
    assert with_setup[:2] == (2, 2), with_setup
    # without the fixture the DATA-dependent assert fails, so the fix is
    # observable rather than a no-op
    assert without[:2] == (1, 2), without
    # the setup itself is not counted in the denominator
    assert with_setup[1] == len(tests)
    print(f"test_test_setup_code_is_run_and_not_scored: PASS "
          f"({with_setup[0]}/{with_setup[1]} with setup vs {without[0]}/{without[1]} without)")


def test_extract_code_handles_fence_primed_and_prose():
    assert extract_code("blah\n```python\ndef f():\n    pass\n```\nmore") == "def f():\n    pass"
    assert extract_code("```\nx = 1\n```") == "x = 1"
    # primed: the prompt already opened the fence, so there is no opening fence
    assert extract_code("def g():\n    return 1\n```\ntrailing", primed=True) == \
        "def g():\n    return 1"
    # unfenced but code-shaped is accepted
    assert extract_code("import os\nprint(os.sep)").startswith("import os")
    # prose is NOT fed to the interpreter (that would yield a meaningless
    # SyntaxError reward indistinguishable from a genuinely wrong answer)
    assert extract_code("Sure! Here is how you would do it.") == ""
    print("test_extract_code_handles_fence_primed_and_prose: PASS")


def test_code_reward_scores_correct_and_wrong():
    tests = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]
    good, _ = code_reward("```python\ndef add(a, b):\n    return a + b\n```", tests)
    bad, _ = code_reward("```python\ndef add(a, b):\n    return a * b\n```", tests)
    none, detail = code_reward("I am not sure.", tests)
    assert good == 1.0, good
    assert bad < 1.0, bad
    assert none == 0.0 and "no code" in detail
    print(f"test_code_reward_scores_correct_and_wrong: PASS (1.0 / {bad:.2f} / 0.0)")


# -------------------------------------------------------------- the agent loop

def test_parse_tool_call_accepts_valid_and_rejects_junk():
    ok = parse_tool_call('```tool\n{"name": "run_python", "args": {"code": "print(1)"}}\n```')
    assert ok == ("run_python", {"code": "print(1)"}), ok
    for junk in ["no tool here",
                 "```tool\nnot json\n```",
                 '```tool\n{"args": {}}\n```',            # missing name
                 '```tool\n{"name": 5, "args": {}}\n```',  # wrong type
                 '```tool\n["run_python"]\n```']:
        assert parse_tool_call(junk) is None, junk
    print("test_parse_tool_call_accepts_valid_and_rejects_junk: PASS")


def test_toolbox_read_file_is_scoped_and_symlink_safe():
    import tempfile
    scratch = tempfile.mkdtemp(prefix="bnano_scr_")
    with open(os.path.join(scratch, "ok.txt"), "w") as fh:
        fh.write("inside")
    tb = ToolBox(scratch_dir=scratch)
    assert tb.read_file(path="ok.txt") == "inside"
    for bad in ["../../etc/passwd", os.path.abspath(os.sep), "..%s.." % os.sep]:
        out = tb.read_file(path=bad)
        assert out.startswith("error:"), (bad, out[:60])
    if posix_limits_active():
        link = os.path.join(scratch, "sneaky")
        try:
            os.symlink("/etc", link)
            out = tb.read_file(path="sneaky/passwd")
            assert out.startswith("error:"), out[:60]
        except (OSError, NotImplementedError):
            pass
    print("test_toolbox_read_file_is_scoped_and_symlink_safe: PASS")


def test_toolbox_never_raises_on_bad_arguments():
    """A malformed tool call must become an error STRING the model can read and
    react to, not an exception that ends the episode. Early in training almost
    every call is malformed, so raising would mean training only on the
    behaviour the model already has."""
    tb = ToolBox()
    for name, args in [("run_python", {}),
                       ("run_python", {"code": ""}),
                       ("run_tests", {"code": "x=1"}),
                       ("nope", {"a": 1}),
                       ("run_python", {"unexpected": 1})]:
        out, ok = tb.dispatch(name, args)
        assert isinstance(out, str) and out.startswith("error:"), (name, args, out[:60])
        assert ok is False
    good, ok = tb.dispatch("run_python", {"code": "print(41 + 1)"})
    assert ok and "42" in good, good
    print("test_toolbox_never_raises_on_bad_arguments: PASS")


def test_tool_output_is_truncated():
    tb = ToolBox()
    out, ok = tb.dispatch("run_python", {"code": "print('x' * 100000)"})
    assert ok
    assert len(out) <= MAX_TOOL_CHARS + 80, len(out)
    assert "elided" in out
    print(f"test_tool_output_is_truncated: PASS ({len(out)} chars)")


def test_episode_reward_scores_only_the_final_answer():
    """Rewarding intermediate tool successes is a hacking surface: the cheapest
    way to farm per-turn credit is to call a passing tool forever."""
    ep = Episode(messages=[], final="```python\ndef add(a, b):\n    return a + b\n```")
    r, _ = episode_reward(ep, ["assert add(1, 1) == 2"])
    assert r == 1.0, r
    ep2 = Episode(messages=[], final="I called a tool successfully!")
    r2, _ = episode_reward(ep2, ["assert add(1, 1) == 2"])
    assert r2 == 0.0, r2
    print("test_episode_reward_scores_only_the_final_answer: PASS")


def test_run_episode_terminates_with_an_untrained_model():
    """An untrained policy emits noise. The loop must still terminate and return
    an Episode rather than raising."""
    import torch

    from blackwell_lm.model import BlackwellLM, ModelConfig
    from blackwell_lm.tokenizer import EOS, train_tokenizer

    tok = train_tokenizer(["def f():\n    return 1\n"] * 60, vocab_size=512,
                          out_path="/tmp/test_agent_tok.json")
    # A tiny corpus yields fewer merges than requested, so the tokenizer vocab
    # is rarely a multiple of 16 -- which the FP8 GEMM constraint requires. The
    # model vocab may legitimately be LARGER than the tokenizer's (the extra
    # rows are simply never emitted), so round up rather than relaxing validate().
    vocab = ((tok.get_vocab_size() + 15) // 16) * 16
    cfg = ModelConfig(vocab_size=vocab, d_model=256, n_layers=1,
                      n_heads=2, n_kv_heads=1, ffn_hidden=512, max_seq_len=512,
                      window=32, global_every=0, n_loops=1, loop_sample=None,
                      zero_init_residual=False)
    m = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.float32)
    ep = run_episode(m, tok, "add two numbers", ToolBox(), tok.token_to_id(EOS),
                     max_turns=2, max_new_tokens=8, n_loops=1)
    assert isinstance(ep, Episode)
    assert 1 <= ep.turns <= 2, ep.turns
    assert ep.messages[0]["content"] == AGENT_SYSTEM
    print(f"test_run_episode_terminates_with_an_untrained_model: PASS "
          f"({ep.turns} turns, {ep.tool_calls} tool calls)")


def test_toolbox_ignores_model_supplied_tests():
    """The harness owns the tests; the model must not be able to grade itself.

    Learned from a live failure: the tool-SFT trajectories carried the tests in
    the call, so the policy learned to GENERATE them and generated broken ones.
    Correct code then scored zero against a hallucinated assert, every episode
    scored 0, and the agentic stage produced zero update steps.
    """
    from blackwell_lm.tasks import EASY_TASKS

    task = next(t for t in EASY_TASKS if t.name == "identity")
    tb = ToolBox().for_task(task)

    good = tb.run_tests(code=task.solution)
    assert good.startswith("2/2") or good.startswith("3/3"), good

    # a model trying to supply its own (broken) tests must be ignored, not obeyed
    hijack = tb.run_tests(code=task.solution,
                          tests=["assert identity(9) is x"])
    assert not hijack.startswith("0/"), f"model-supplied tests were honoured: {hijack}"
    assert "ignored" in hijack, hijack
    print("test_toolbox_ignores_model_supplied_tests: PASS "
          f"({good.splitlines()[0]}; hijack attempt ignored)")


if __name__ == "__main__":
    test_normal_code_runs()
    test_infinite_loop_is_killed()
    test_sleep_is_killed_by_wallclock_not_cpu()
    test_memory_bomb_is_stopped()
    test_child_processes_do_not_outlive_the_timeout()
    test_extra_file_names_cannot_escape()
    test_workdir_is_removed_afterwards()
    test_network_isolation_is_NOT_claimed()
    test_run_tests_gives_partial_credit()
    test_run_tests_reports_setup_failure_distinctly()
    test_one_failing_test_does_not_mask_the_others()
    test_test_setup_code_is_run_and_not_scored()
    test_extract_code_handles_fence_primed_and_prose()
    test_code_reward_scores_correct_and_wrong()
    test_parse_tool_call_accepts_valid_and_rejects_junk()
    test_toolbox_read_file_is_scoped_and_symlink_safe()
    test_toolbox_never_raises_on_bad_arguments()
    test_toolbox_ignores_model_supplied_tests()
    test_tool_output_is_truncated()
    test_episode_reward_scores_only_the_final_answer()
    test_run_episode_terminates_with_an_untrained_model()
    if SKIPS:
        print(f"\nAll test_sandbox_cpu tests passed, with {len(SKIPS)} SKIPPED "
              f"(not a validated sandbox on this OS): {SKIPS}")
        sys.exit(0)
    print("\nAll test_sandbox_cpu tests passed (POSIX limits exercised).")
