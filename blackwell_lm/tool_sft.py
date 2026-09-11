"""Synthetic tool-use trajectories, to teach the ```tool convention.

WHY THIS EXISTS. Agentic RL ran 100 episodes and produced ZERO tool calls with
mean turns 1.00 -- the loop worked, the policy simply never called a tool,
because none of the SFT sources (opencoder-sft, evol-codealpaca, tulu3) contain
tool-call examples. RL cannot teach a convention the policy never emits: with no
tool call ever sampled, there is no gradient toward making one.

TOOL OUTPUT IS REAL, NOT INVENTED. Every "N/M tests passed" line in these
trajectories comes from actually executing the code in the sandbox. Writing
plausible-looking output by hand would teach the model to expect responses the
real tools never produce, which is worse than no training at all -- it would
learn a fictional environment and then be surprised by the true one.

THE FAILURE-THEN-FIX SHAPE IS THE POINT. A trajectory that calls a tool once and
succeeds teaches "emit a tool call", but not "read the result and revise", which
is the behaviour that makes an agent worth having. So a fraction of trajectories
deliberately start from a MUTATED solution, show the real failing output, and
then fix it. The mutation is verified to actually fail before use; a mutation
that happens to still pass would teach the model to "fix" working code.
"""

from __future__ import annotations

import json
import random
import re

from blackwell_lm.agent import AGENT_SYSTEM
from blackwell_lm.chat import ASSISTANT, SYSTEM, TOOL, USER
from blackwell_lm.sandbox import run_tests

# Small, targeted mutations that usually break a solution while keeping it
# syntactically valid -- a SyntaxError would teach a different (and less useful)
# lesson than a wrong answer.
_MUTATIONS = (
    (r"\+", "-"),
    (r"\breturn\b", "return not"),
    (r"\bmax\b", "min"),
    (r"\bsorted\(", "reversed("),
    (r"== ", "!= "),
    (r"\brange\(", "range(1, "),
)


def _mutate(code: str, rng: random.Random) -> str | None:
    """One plausible bug, or None if nothing applied."""
    opts = [(p, r) for p, r in _MUTATIONS if re.search(p, code)]
    if not opts:
        return None
    p, r = rng.choice(opts)
    return re.sub(p, r, code, count=1)


def _tool_call(code: str, tests: list[str] | None = None) -> str:
    """A call carrying ONLY code.

    The tests are deliberately absent. Including them taught the policy to
    generate its own tests, and it generated broken ones -- so correct code
    scored zero against a hallucinated grader. The harness owns the tests.
    """
    return ("```tool\n"
            + json.dumps({"name": "run_tests", "args": {"code": code}})
            + "\n```")


def _final(code: str) -> str:
    return f"Here is the code to solve this problem:\n```python\n{code}\n```"


def make_trajectory(task, rng: random.Random, timeout: float = 6.0,
                    fail_first_rate: float = 0.5):
    """One tool-use conversation for a task, or None if it cannot be built.

    Returns a message list in the project's chat format. The system prompt is
    the SAME AGENT_SYSTEM the agentic rollouts use -- if the two differed, SFT
    would teach a convention the RL stage never asks for, which is precisely the
    prompt-format drift this project already got burned by once.
    """
    code = (getattr(task, "solution", None) or "").strip()
    if not code or not task.tests:
        return None

    # verify the reference solution actually passes, with real execution
    p_ok, total, _ = run_tests(code, task.tests, timeout=timeout,
                               setup=getattr(task, "setup", ""))
    if p_ok != total or total == 0:
        return None                      # cannot teach from a broken reference

    msgs = [{"role": SYSTEM, "content": AGENT_SYSTEM},
            {"role": USER, "content": task.prompt}]

    if rng.random() < fail_first_rate:
        bad = _mutate(code, rng)
        if bad:
            p_bad, t_bad, detail = run_tests(bad, task.tests, timeout=timeout,
                                             setup=getattr(task, "setup", ""))
            if p_bad < t_bad:            # only use a mutation that REALLY fails
                msgs += [
                    {"role": ASSISTANT, "content": _tool_call(bad)},
                    {"role": TOOL, "content": f"{p_bad}/{t_bad} tests passed"
                                              + (f"\n{detail}" if detail else "")},
                ]

    msgs += [
        {"role": ASSISTANT, "content": _tool_call(code)},
        {"role": TOOL, "content": f"{p_ok}/{total} tests passed"},
        {"role": ASSISTANT, "content": _final(code)},
    ]
    return msgs


def stream_tool_sft(tasks, seed: int = 0, fail_first_rate: float = 0.5):
    """Endless stream of tool-use conversations, reshuffled each pass."""
    rng = random.Random(seed)
    pool = [t for t in tasks if (getattr(t, "solution", None) or "").strip()]
    if not pool:
        raise RuntimeError(
            "no tasks carry a reference solution; tool trajectories need one "
            "(MBPP's `code` field, or Task.solution)")
    # CACHE the built trajectories. Each one costs one or two real sandbox
    # executions (~0.3s of process spawn), and at 25% of a batch-8 SFT stream
    # that is ~17 subprocesses per second -- the data pipeline would become the
    # bottleneck and the GPU would sit idle. Building each task once and then
    # cycling keeps the tool output real while making the stream free.
    cache: list = []
    first_pass = True
    while True:
        rng.shuffle(pool)
        if first_pass:
            for t in pool:
                m = make_trajectory(t, rng, fail_first_rate=fail_first_rate)
                if m:
                    cache.append(m)
                    yield m
            first_pass = False
            if not cache:
                raise RuntimeError("every candidate trajectory failed to build")
            print(f"[tool_sft] cached {len(cache)} verified trajectories "
                  f"from {len(pool)} tasks", flush=True)
        else:
            rng.shuffle(cache)
            for m in cache:
                yield m

# ---------------------------------------------------------------- repo shape

def make_repo_trajectory(scenario, tool_timeout: float = 10.0):
    """A read -> write -> verify -> answer trajectory over a real repo.

    WHY THIS IS NEEDED SEPARATELY from the snippet trajectories above: the repo
    tools are a different schema. `write_file` carries the WHOLE FILE as
    `content`, and a policy trained only on `run_tests(code=...)` -- one short
    string -- does not generalise to it. Measured on the best agentic checkpoint:
    it emitted `write_file` 19 times and supplied `content` zero times, and it
    did not recover even when the error message demonstrated the exact expected
    call. RL cannot fix this either, because a group where no episode ever
    writes a file has no reward spread to learn from.

    Every tool output here is produced by ACTUALLY DISPATCHING against a
    materialised repo, so the trajectory teaches the real environment rather
    than a plausible-looking imitation of it.
    """
    from blackwell_lm.mcp_tools import RepoToolBox, render_system_prompt
    from blackwell_lm.scenario import cleanup

    repo = scenario.materialise()
    try:
        tb = RepoToolBox(repo, tests=scenario.tests, setup=scenario.setup,
                         timeout=tool_timeout)
        msgs = [{"role": SYSTEM, "content": render_system_prompt()},
                {"role": USER, "content": scenario.prompt}]

        def turn(name, args):
            msgs.append({"role": ASSISTANT,
                         "content": "```tool\n"
                                    + json.dumps({"name": name, "args": args})
                                    + "\n```"})
            out, ok = tb.dispatch(name, args)
            msgs.append({"role": TOOL, "content": out})
            return out, ok

        # 1. look at the broken file
        turn("read_file", {"path": "solution.py"})
        # 2. write the fix -- the step the policy cannot currently form
        _, ok = turn("write_file", {"path": "solution.py",
                                    "content": scenario.reference})
        if not ok:
            return None
        # 3. verify, and only keep the trajectory if the tests really pass
        out, _ = turn("run_tests", {})
        if not out.startswith(f"{len(scenario.tests)}/{len(scenario.tests)}"):
            return None
        # 4. state the answer
        msgs.append({"role": ASSISTANT,
                     "content": "The tests pass now.\n```python\n"
                                + scenario.reference + "\n```"})
        return msgs
    finally:
        cleanup(repo)


def make_repo_retry_trajectory(scenario, rng, tool_timeout: float = 10.0):
    """read -> write a WRONG fix -> tests FAIL -> write the right fix -> pass.

    WHY THIS SHAPE IS MISSING AND WHY IT MATTERS. Every trajectory above is
    first-try-correct, so the policy has never been shown what to do with a
    failing test report. Measured consequence, driving a real CLI: it reads the
    file, runs the tests, sees "1/3 tests passed", and then stops -- it has no
    demonstrated notion of using that output to try again.

    The capability is there and unused: pass@12 is 40% where pass@1 is 9.6%, so
    a second and third attempt would land far more often than the first. What
    is missing is the habit of taking one.

    The wrong first attempt is the scenario's OWN broken file (or another
    mutation of it), so the failure is real and the test output is real -- not
    a plausible-looking imitation of a failure, which would teach the model to
    expect error text that never occurs.
    """
    from blackwell_lm.mcp_tools import RepoToolBox, render_system_prompt
    from blackwell_lm.scenario import _breakages, cleanup

    repo = scenario.materialise()
    try:
        tb = RepoToolBox(repo, tests=scenario.tests, setup=scenario.setup,
                         timeout=tool_timeout)
        msgs = [{"role": SYSTEM, "content": render_system_prompt()},
                {"role": USER, "content": scenario.prompt}]

        def turn(name, args, prefix=""):
            """`prefix` rides in the SAME assistant message as the tool call.

            It must not be its own turn. run_repo_episode ends the episode the
            moment an assistant turn contains no parseable tool call:

                call = parse_tool_call(text)
                if call is None: break

            so a demo with a standalone "that did not pass, let me fix it"
            message teaches the policy to answer a failing test with a comment
            and stop -- terminating the episode at exactly the point this
            trajectory exists to teach it to continue. parse_tool_call scans
            for the fence anywhere in the text, so prose may precede it.
            """
            body = "```tool\n" + json.dumps({"name": name, "args": args}) + "\n```"
            msgs.append({"role": ASSISTANT,
                         "content": (prefix + "\n" + body) if prefix else body})
            out, ok = tb.dispatch(name, args)
            msgs.append({"role": TOOL, "content": out})
            return out, ok

        turn("read_file", {"path": "solution.py"})

        # A genuinely wrong first attempt: a fresh mutation of the reference,
        # falling back to the scenario's own broken file.
        wrong = next(_breakages(scenario.reference, rng, "mutate", None), None)
        if not wrong or wrong.strip() == scenario.reference.strip():
            wrong = scenario.broken
        if not wrong or wrong.strip() == scenario.reference.strip():
            return None

        _, ok = turn("write_file", {"path": "solution.py", "content": wrong})
        if not ok:
            return None
        out, _ = turn("run_tests", {})
        full = f"{len(scenario.tests)}/{len(scenario.tests)}"
        if out.startswith(full):
            return None          # the "wrong" attempt passed: no failure to learn from

        # The recovery, which is the behaviour being taught. The commentary
        # rides along with the call rather than forming a turn of its own.
        _, ok = turn("write_file",
                     {"path": "solution.py", "content": scenario.reference},
                     prefix="That did not pass. Let me correct it.")
        if not ok:
            return None
        out, _ = turn("run_tests", {})
        if not out.startswith(full):
            return None
        msgs.append({"role": ASSISTANT,
                     "content": "The tests pass now.\n```python\n"
                                + scenario.reference + "\n```"})
        return msgs
    finally:
        cleanup(repo)


def stream_repo_sft(scenarios, seed: int = 0, tool_timeout: float = 10.0,
                    retry_frac: float = 0.0):
    """Endless stream of repo trajectories, built once then cycled.

    Cached for the same reason as the snippet trajectories: each one costs
    several real subprocesses, and rebuilding every epoch would make the data
    pipeline the bottleneck instead of the GPU.
    """
    rng = random.Random(seed)
    cache, retries = [], []
    for sc in scenarios:
        m = make_repo_trajectory(sc, tool_timeout=tool_timeout)
        if m:
            cache.append(m)
        if retry_frac > 0:
            r = make_repo_retry_trajectory(sc, rng, tool_timeout=tool_timeout)
            if r:
                retries.append(r)
    if not cache:
        raise RuntimeError("no repo trajectories could be built")
    print(f"[tool_sft] cached {len(cache)} verified REPO trajectories "
          f"from {len(scenarios)} scenarios"
          + (f", plus {len(retries)} RETRY trajectories "
             f"(wrong fix -> failing tests -> correct fix)" if retries else ""),
          flush=True)
    if retries:
        # Mixed into one pool rather than trained as a phase: tuned on retries
        # last, the model learns to write a wrong answer first on purpose.
        n_retry = max(1, int(len(cache) * retry_frac / max(1e-9, 1 - retry_frac)))
        cache = cache + [retries[i % len(retries)] for i in range(n_retry)]
        print(f"[tool_sft] pool is {len(cache)} trajectories "
              f"({n_retry} of them retries, ~{n_retry / len(cache) * 100:.0f}%)",
              flush=True)
    while True:
        rng.shuffle(cache)
        for m in cache:
            yield m
