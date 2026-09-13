"""Execution-based rewards.

Deliberately NO learned reward model. For code the ground truth is "does it run
and pass the tests", which is cheap, unhackable in the ways a learned RM is, and
does not need a second network trained on preferences this weak policy never
produced.

The one design choice with teeth is PARTIAL CREDIT: reward is the fraction of
asserts passed, not pass/fail. At the start of RL a small model passes almost
nothing, and a binary reward is then zero for every sample in every group --
which makes the group-relative advantage exactly zero and the update a no-op.
Partial credit is what keeps a gradient alive in the regime where the training
actually starts.
"""

from __future__ import annotations

import re

from nanocoder.sandbox import run_tests

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)
_CODEISH = re.compile(r"^\s*(def |class |import |from |@|#|if |for |while |return )")


def extract_code(text: str, primed: bool = False) -> str:
    """Pull runnable Python out of a model completion.

    `primed=True` means the prompt already ended inside an opened ```python
    fence, so the completion starts mid-code-block and has no opening fence to
    find. Getting this wrong yields empty code and therefore zero reward for
    every sample -- a failure that looks exactly like "the model can't code".
    """
    if primed:
        return text.split("```", 1)[0].strip()
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    # No fence at all: accept it only if it actually looks like code, so prose
    # is not fed to the interpreter to produce a meaningless SyntaxError reward.
    if _CODEISH.match(text):
        return text.strip()
    return ""


def code_reward(completion: str, tests: list[str], primed: bool = False,
                timeout: float = 6.0, setup: str = "") -> tuple[float, str]:
    """Fraction of `tests` that pass, plus a short diagnostic.

    Returns (reward in [0,1], detail). No format bonus: rewarding "produced a
    fenced block" is trivially hackable by emitting an empty one.
    """
    code = extract_code(completion, primed=primed)
    if not code:
        return 0.0, "no code found"
    if not tests:
        return 0.0, "no tests supplied"
    passed, total, detail = run_tests(code, tests, timeout=timeout, setup=setup)
    return passed / total, detail
