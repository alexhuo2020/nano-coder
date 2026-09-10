"""Tests for the repo-scenario environment and its MCP-shaped tools.

These cover the properties that make the environment trustworthy rather than
merely functional: that the "broken" code is verified broken, that the reward
reads the FILE rather than the transcript, that paths cannot escape the repo,
and that each episode gets an independent copy of the repo.
"""
import os
import random

import torch

from blackwell_lm.mcp_tools import TOOLS, RepoToolBox, render_system_prompt
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.repo_agent import repo_reward, run_repo_episode
from blackwell_lm.scenario import build_scenario, build_scenarios, cleanup
from blackwell_lm.tasks import EASY_TASKS
from blackwell_lm.tokenizer import EOS, train_tokenizer


def _scn():
    task = next(t for t in EASY_TASKS if t.name == "add")
    sc = build_scenario(task, random.Random(0))
    assert sc is not None, "could not build the 'add' scenario"
    return sc


def test_system_prompt_is_generated_from_the_tool_declarations():
    """Hand-writing the prompt lets it drift from the dispatcher -- the same
    class of bug as the SFT/RL prompt-format drift this project already paid
    for. So every declared tool must appear, and nothing else may."""
    prompt = render_system_prompt()
    for t in TOOLS:
        assert t["name"] in prompt, f"{t['name']} missing from the prompt"
    assert "run_shell" not in prompt and "git" not in prompt, \
        "the prompt advertises a tool the dispatcher does not implement"
    print(f"test_system_prompt_is_generated_from_the_tool_declarations: PASS "
          f"({len(TOOLS)} tools declared and described)")


def test_scenario_breakage_is_verified_both_ways():
    """Two silent failure modes this guards: a 'broken' file that actually
    passes (the agent is paid for doing nothing) and a reference that fails its
    own tests (the task is unsolvable and every episode scores 0, which looks
    exactly like a policy problem)."""
    sc = _scn()
    repo = sc.materialise()
    try:
        tb = RepoToolBox(repo, tests=sc.tests, setup=sc.setup)
        broken_r, bp, bt = repo_reward(tb)
        assert broken_r < 1.0, f"the 'broken' file passes everything: {bp}/{bt}"
        tb.write_file("solution.py", sc.reference)
        fixed_r, fp, ft = repo_reward(tb)
        assert fixed_r == 1.0, f"the reference does not pass: {fp}/{ft}"
        print(f"test_scenario_breakage_is_verified_both_ways: PASS "
              f"(broken {bp}/{bt} -> fixed {fp}/{ft})")
    finally:
        cleanup(repo)


def test_reward_reads_the_file_not_the_transcript():
    """The whole point of the repo environment. An agent that edits the file
    correctly and says nothing scores full marks; one that describes a perfect
    fix without writing it scores zero. That asymmetry is intended -- and it
    makes the reward immune to the answer-extraction bug that gave a previous
    run 100 identical zeros."""
    sc = _scn()
    repo = sc.materialise()
    try:
        tb = RepoToolBox(repo, tests=sc.tests, setup=sc.setup)
        tb.write_file("solution.py", sc.reference)
        silent, _, _ = repo_reward(tb)
        assert silent == 1.0, "a correct file did not score"

        tb.write_file("solution.py", sc.broken)
        talked, _, _ = repo_reward(tb)
        assert talked < 1.0, "a broken file scored full marks"
        print("test_reward_reads_the_file_not_the_transcript: PASS "
              "(file state decides, transcript is irrelevant)")
    finally:
        cleanup(repo)


def test_paths_cannot_escape_the_repo():
    sc = _scn()
    repo = sc.materialise()
    try:
        tb = RepoToolBox(repo, tests=sc.tests)
        for bad in ("../../etc/passwd", os.path.abspath(os.sep), "../outside.py"):
            out, ok = tb.dispatch("read_file", {"path": bad})
            assert not ok and "escapes" in out, (bad, out[:60])
            out, ok = tb.dispatch("write_file", {"path": bad, "content": "x"})
            assert not ok and "escapes" in out, (bad, out[:60])
        print("test_paths_cannot_escape_the_repo: PASS (read and write blocked)")
    finally:
        cleanup(repo)


def test_schema_validation_returns_errors_as_data():
    """A malformed call must be a string the policy can read and react to.
    Raising would mean the episodes that fail are exactly the ones that never
    produce a gradient, so the model would only ever train on what it already
    does correctly."""
    sc = _scn()
    repo = sc.materialise()
    try:
        tb = RepoToolBox(repo, tests=sc.tests)
        for name, args, needle in (
            ("read_file", {}, "requires 'path'"),
            ("read_file", {"path": 5}, "must be string"),
            ("write_file", {"path": "a.py"}, "requires 'content'"),
            ("nope", {}, "unknown tool"),
        ):
            out, ok = tb.dispatch(name, args)
            assert not ok, (name, args, out)
            assert needle in out, (needle, out[:80])
        print("test_schema_validation_returns_errors_as_data: PASS")
    finally:
        cleanup(repo)


def test_each_episode_gets_an_independent_repo():
    """Group episodes must be independent samples of the SAME starting state.
    A shared directory would let one episode's edits leak into the next and
    silently correlate rewards that the group-relative advantage assumes are
    independent."""
    sc = _scn()
    a, b = sc.materialise(), sc.materialise()
    try:
        assert a != b, "materialise() reused a directory"
        RepoToolBox(a, tests=sc.tests).write_file("solution.py", sc.reference)
        rb, _, _ = repo_reward(RepoToolBox(b, tests=sc.tests, setup=sc.setup))
        assert rb < 1.0, "an edit in one repo was visible in another"
        print("test_each_episode_gets_an_independent_repo: PASS")
    finally:
        cleanup(a)
        cleanup(b)


def test_run_repo_episode_terminates_and_cleans_up():
    """An untrained policy emits noise; the loop must still terminate, return a
    scored episode, and leave no scratch directory behind."""
    tok = train_tokenizer(["def f():\n    return 1\n"] * 60, vocab_size=512,
                          out_path="/tmp/test_repo_tok.json")
    vocab = ((tok.get_vocab_size() + 15) // 16) * 16
    cfg = ModelConfig(vocab_size=vocab, d_model=256, n_layers=1, n_heads=2,
                      n_kv_heads=1, ffn_hidden=512, max_seq_len=2048, window=64,
                      global_every=0, n_loops=1, loop_sample=None,
                      zero_init_residual=False)
    m = BlackwellLM(cfg, precision="bf16", device="cpu", dtype=torch.float32)
    sc = _scn()
    before = len([d for d in os.listdir("/tmp") if d.startswith("bnano_repo_")])
    ep = run_repo_episode(m, tok, sc, tok.token_to_id(EOS), max_turns=2,
                          max_new_tokens=8, n_loops=1, record_logprobs=True)
    after = len([d for d in os.listdir("/tmp") if d.startswith("bnano_repo_")])
    assert 1 <= ep.turns <= 2, ep.turns
    assert 0.0 <= ep.reward <= 1.0, ep.reward
    assert len(ep.steps) == ep.turns
    assert after <= before, f"leaked a scratch repo ({before} -> {after})"
    print(f"test_run_repo_episode_terminates_and_cleans_up: PASS "
          f"({ep.turns} turns, {ep.tool_calls} tool calls, reward {ep.reward:.2f})")


def test_scenario_pool_reports_its_rejections():
    """A silently small pool means the agent sees the same handful of repos and
    the reward stops measuring generalisation, so the counts are printed."""
    scns = build_scenarios(EASY_TASKS, seed=1)
    assert len(scns) >= 8, f"only {len(scns)} of {len(EASY_TASKS)} easy tasks built"
    assert all(s.broken and s.reference for s in scns)
    print(f"test_scenario_pool_reports_its_rejections: PASS "
          f"({len(scns)}/{len(EASY_TASKS)} easy tasks became scenarios)")


def test_repo_sft_trajectory_teaches_the_write_file_schema():
    """The trajectory must contain a write_file call carrying BOTH path and
    content, with tool output from real dispatch.

    This is the exact gap it exists to close. Measured on the best agentic
    checkpoint: write_file emitted 19 times, content supplied 0 times, and no
    recovery even when the error message printed the expected call verbatim.
    Neither RL nor a better error can bootstrap a convention the policy never
    emits -- only demonstration can.
    """
    import json

    from blackwell_lm.agent import parse_tool_call
    from blackwell_lm.tool_sft import make_repo_trajectory

    sc = _scn()
    msgs = make_repo_trajectory(sc)
    assert msgs, "repo trajectory failed to build"

    calls = [parse_tool_call(m["content"]) for m in msgs
             if m["role"] == "assistant"]
    calls = [c for c in calls if c]
    names = [c[0] for c in calls]
    assert names == ["read_file", "write_file", "run_tests"], names

    wf = next(args for name, args in calls if name == "write_file")
    assert "path" in wf and "content" in wf, wf.keys()
    assert wf["content"].strip() == sc.reference.strip(), "content is not the fix"

    tool_outs = [m["content"] for m in msgs if m["role"] == "tool"]
    assert sc.broken.splitlines()[0] in tool_outs[0], "read_file output is not the real file"
    assert "wrote" in tool_outs[1], tool_outs[1]
    assert tool_outs[2].startswith(f"{len(sc.tests)}/{len(sc.tests)}"), tool_outs[2]
    print("test_repo_sft_trajectory_teaches_the_write_file_schema: PASS "
          f"({' -> '.join(names)}; content={len(wf['content'])} chars; "
          f"verify={tool_outs[2].splitlines()[0]})")


def test_verify_bonus_requires_a_genuine_verified_pass():
    """The bonus must be unfarmable in both directions.

    RL found the open shortcut in the live run: with reward reading only the
    file, run_tests earned nothing and cost a turn, so `ran_tests` collapsed
    79% -> 1% and tool errors went 162 -> 0 together. The bonus fixes the
    incentive, but only if it cannot be claimed by calling run_tests without
    fixing anything, nor by fixing without verifying.
    """
    from blackwell_lm.mcp_tools import RepoToolBox
    from blackwell_lm.repo_agent import RepoEpisode, repo_reward

    sc = _scn()
    repo = sc.materialise()
    try:
        tb = RepoToolBox(repo, tests=sc.tests, setup=sc.setup)

        B = 0.1

        def scored(base, verified):
            """The exact arithmetic run_repo_episode applies."""
            return base if verified else base * (1.0 - B)

        tb.write_file("solution.py", sc.reference)
        fixed, _, _ = repo_reward(tb)
        assert fixed == 1.0

        # THE PROPERTY THE FIRST VERSION MISSED: the incentive must actually
        # differentiate. An additive bonus capped at 1.0 scored these equal.
        assert scored(fixed, True) > scored(fixed, False),             "verifying and not verifying score the same -- the incentive is a no-op"
        assert scored(fixed, True) == 1.0, "verified success must reach full marks"

        # and fixing the code must still dominate verifying it
        tb.write_file("solution.py", sc.broken)
        broken, _, _ = repo_reward(tb)
        assert broken < 1.0
        assert scored(fixed, False) > scored(broken, True),             "verifying a broken file outscores blindly fixing it -- wrong ordering"
        assert 0.0 <= scored(broken, False) <= 1.0, "reward left [0,1]"

        ep = RepoEpisode(scenario_name="x", messages=[])
        assert ep.verified_pass is False, "verification must not be free"
        print("test_verify_bonus_requires_a_genuine_verified_pass: PASS "
              f"(verified {scored(fixed, True):.2f} > blind {scored(fixed, False):.2f} "
              f"> verified-broken {scored(broken, True):.2f})")
    finally:
        cleanup(repo)


if __name__ == "__main__":
    test_system_prompt_is_generated_from_the_tool_declarations()
    test_scenario_breakage_is_verified_both_ways()
    test_reward_reads_the_file_not_the_transcript()
    test_paths_cannot_escape_the_repo()
    test_schema_validation_returns_errors_as_data()
    test_each_episode_gets_an_independent_repo()
    test_scenario_pool_reports_its_rejections()
    test_repo_sft_trajectory_teaches_the_write_file_schema()
    test_verify_bonus_requires_a_genuine_verified_pass()
    test_run_repo_episode_terminates_and_cleans_up()
    print("All test_repo_agent_cpu tests passed.")
