"""Multi-turn episodes over a real repo, with MCP-shaped tools.

THE REWARD IS THE REPO, NOT THE TRANSCRIPT. This is the substantive difference
from `agent.run_episode`. There, the reward came from extracting code out of the
model's final message, which turned out to be fragile in exactly the way that
matters: an episode that ran out of turns mid-tool-call had no extractable code,
scored a flat 0, and made every group zero-variance -- 100 episodes, 0 updates.

Here the agent's work product is the FILE it leaves behind. Reward = fraction of
the harness's hidden tests that pass against `solution.py` as it stands when the
episode ends. That is both more faithful to what a coding agent is for, and
structurally immune to answer-extraction bugs: there is nothing to extract.

A consequence worth stating: an agent that edits the file correctly and then
says nothing still scores full marks, and one that describes a perfect fix
without writing it scores zero. That asymmetry is intended.

THE SHORTCUT THIS ORIGINALLY LEFT OPEN, AND THE FIX. Because the reward reads
only the file, calling `run_tests` earned nothing and cost a turn -- so RL
correctly learned to stop verifying. Measured over 84 steps of repo-mbpp:
`ran_tests` collapsed from 79% of episodes to 1%, tool errors fell 162 -> 0 in
lockstep (the errors WERE the run_tests calls), and reward still rose. The
policy was maximising exactly what it was told to; the behaviour I actually
wanted -- read, fix, VERIFY -- had been trained out.

`verify_bonus` fixes the incentive rather than the policy: a small bonus, paid
only when the agent's own last `run_tests` reported a pass AND the file really
does pass. Both conditions matter. Paying for the call alone would be farmable
by spamming run_tests; paying without requiring the call would restore the
shortcut. The bonus is deliberately small, so verification is worth a turn
without ever outweighing actually fixing the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from blackwell_lm import chat
from blackwell_lm.agent import TurnRecord, parse_tool_call
from blackwell_lm.mcp_tools import RepoToolBox, render_system_prompt
from blackwell_lm.sandbox import run_tests
from blackwell_lm.scenario import cleanup


@dataclass
class RepoEpisode:
    scenario_name: str
    messages: list
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    wrote_file: bool = False
    ran_tests: bool = False
    verified_pass: bool = False    # the agent's OWN last run_tests reported a pass
    reward: float = 0.0
    passed: int = 0
    total: int = 0
    transcript: list = field(default_factory=list)
    steps: list = field(default_factory=list)


def repo_reward(toolbox: RepoToolBox) -> tuple[float, int, int]:
    """Score the repo's CURRENT solution.py against the hidden tests."""
    import os

    sol = os.path.join(toolbox.repo, "solution.py")
    if not os.path.isfile(sol):
        return 0.0, 0, len(toolbox.tests)
    with open(sol, encoding="utf-8", errors="replace") as fh:
        code = fh.read()
    if not code.strip():
        return 0.0, 0, len(toolbox.tests)
    passed, total, _ = run_tests(code, toolbox.tests, timeout=toolbox.timeout,
                                 setup=toolbox.setup)
    return (passed / total if total else 0.0), passed, total


def run_repo_episode(model, tok, scenario, eos_id: int, max_turns: int = 6,
                     max_new_tokens: int = 256, temperature: float = 1.0,
                     n_loops: int | None = None, record_logprobs: bool = False,
                     tool_timeout: float = 10.0, keep_repo: bool = False,
                     verify_bonus: float = 0.0):
    """One episode against a freshly materialised copy of the scenario.

    A FRESH repo per episode is essential: the group's episodes must be
    independent samples of the same starting state, and a shared directory would
    let one episode's edits leak into the next -- silently correlating rewards
    that the group-relative advantage assumes are independent.
    """
    from blackwell_lm.generate import generate

    repo = scenario.materialise()
    tb = RepoToolBox(repo, tests=scenario.tests, setup=scenario.setup,
                     timeout=tool_timeout)
    ep = RepoEpisode(
        scenario_name=scenario.name,
        messages=[{"role": chat.SYSTEM, "content": render_system_prompt()},
                  {"role": chat.USER, "content": scenario.prompt}],
    )
    try:
        for _ in range(max_turns):
            prompt_ids = chat.tokenize_prompt(tok, ep.messages)
            toks, valid, lp = generate(
                model, prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
                temperature=1.0 if record_logprobs else temperature,
                num_return_sequences=1, n_loops=n_loops,
                return_logprobs=record_logprobs,
            )
            ep.steps.append(TurnRecord(prompt_ids=prompt_ids, tokens=toks,
                                       valid=valid, logprobs=lp))
            text = tok.decode([int(t) for t, v in
                               zip(toks[0].tolist(), valid[0].tolist()) if v])
            ep.messages.append({"role": chat.ASSISTANT, "content": text})
            ep.turns += 1
            ep.transcript.append(("assistant", text))

            call = parse_tool_call(text)
            if call is None:
                break                     # the agent considers itself finished
            name, args = call
            out, ok = tb.dispatch(name, args)
            ep.tool_calls += 1
            ep.tool_errors += 0 if ok else 1
            if name == "write_file" and ok:
                ep.wrote_file = True
            if name == "run_tests":
                ep.ran_tests = True
                # Record what the agent was TOLD, not what is true. The bonus
                # below requires both, so a stale or lucky pass cannot earn it.
                ep.verified_pass = bool(
                    out and out.split("/")[0].isdigit()
                    and out.split("/")[0] != "0"
                    and out.split()[0] == f"{len(tb.tests)}/{len(tb.tests)}")
            ep.messages.append({"role": chat.TOOL, "content": out})
            ep.transcript.append(("tool", out))

        ep.reward, ep.passed, ep.total = repo_reward(tb)
        if verify_bonus and not ep.verified_pass:
            # Applied as a DISCOUNT for not verifying, not a bonus for
            # verifying. An additive bonus capped at 1.0 is a no-op -- it can
            # only fire when the reward is already 1.0, and then clamps straight
            # back to it. (I wrote exactly that first, and the test passed
            # because it checked the cap instead of checking that the incentive
            # changes anything.)
            #
            # A discount gives the ordering that was wanted -- verified success
            # strictly above blind success -- while keeping the reward inside
            # [0, 1] so the group-relative advantage is computed on an unchanged
            # scale. Fixing the code still dominates: at the default 0.1, a
            # blind full fix scores 0.9, far above any partial fix.
            ep.reward *= (1.0 - verify_bonus)
    finally:
        if not keep_repo:
            cleanup(repo)
    return ep
