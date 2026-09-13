"""Verifiable coding tasks for the RL stages.

A task is a question plus asserts that decide, by execution, whether an answer
is right. MBPP is the default because its 974 problems are short, each ships
three asserts, and the function name is pinned by the tests -- so a correct
solution is unambiguous and the reward needs no judge.

TWO TASK SETS, AND THE SMALL ONE IS NOT A TOY. `easy` exists because of a
concrete failure: a weak policy scored 0.0 on every MBPP sample in every group,
every group therefore had zero variance, every group was dropped, and GRPO ran
for hours with ZERO update steps while looking like it was training. When that
happens you cannot tell "RL is broken" from "the model is too weak yet". The
easy set is the control: if a run cannot move on tasks this trivial, the bug is
in the RL code, not the policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    prompt: str
    tests: list[str]
    name: str = ""
    # Fixture code the asserts depend on (MBPP's test_setup_code). Rare -- 1 of
    # 374 train problems -- but ignoring it makes those problems unsolvable and
    # the lost reward is indistinguishable from the model being wrong.
    setup: str = ""
    # Reference solution, used ONLY to synthesise tool-use SFT trajectories
    # (see nanocoder/tool_sft.py). Never shown to the policy at RL time --
    # that would make the execution reward meaningless.
    solution: str = ""


# Deliberately trivial: one-liners a barely-trained model can sometimes emit.
# Each pins its function name in the prompt, because the tests call it by name
# and a right answer under a different name would score zero for a reason that
# has nothing to do with reasoning.
EASY_TASKS = [
    Task("Write a function `add(a, b)` that returns the sum of a and b.",
         ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"], "add", solution="def add(a, b):\n    return a + b"),
    Task("Write a function `identity(x)` that returns x unchanged.",
         ["assert identity(5) == 5", "assert identity('a') == 'a'"], "identity", solution="def identity(x):\n    return x"),
    Task("Write a function `square(x)` that returns x squared.",
         ["assert square(3) == 9", "assert square(0) == 0"], "square", solution="def square(x):\n    return x * x"),
    Task("Write a function `is_even(n)` that returns True if n is even.",
         ["assert is_even(2) is True", "assert is_even(3) is False"], "is_even", solution="def is_even(n):\n    return n % 2 == 0"),
    Task("Write a function `first(xs)` that returns the first element of a list.",
         ["assert first([1, 2]) == 1", "assert first(['a']) == 'a'"], "first", solution="def first(xs):\n    return xs[0]"),
    Task("Write a function `length(xs)` that returns the number of items in a list.",
         ["assert length([1, 2, 3]) == 3", "assert length([]) == 0"], "length", solution="def length(xs):\n    return len(xs)"),
    Task("Write a function `double(x)` that returns x multiplied by two.",
         ["assert double(4) == 8", "assert double(0) == 0"], "double", solution="def double(x):\n    return x * 2"),
    Task("Write a function `negate(x)` that returns the negation of x.",
         ["assert negate(3) == -3", "assert negate(0) == 0"], "negate", solution="def negate(x):\n    return -x"),
    Task("Write a function `maximum(a, b)` that returns the larger of a and b.",
         ["assert maximum(1, 2) == 2", "assert maximum(5, 3) == 5"], "maximum", solution="def maximum(a, b):\n    return a if a > b else b"),
    Task("Write a function `join_words(xs)` that joins a list of strings with a single space.",
         ["assert join_words(['a', 'b']) == 'a b'", "assert join_words(['x']) == 'x'"],
         "join_words", solution="def join_words(xs):\n    return ' '.join(xs)"),
    Task("Write a function `reverse_list(xs)` that returns the list reversed.",
         ["assert reverse_list([1, 2, 3]) == [3, 2, 1]", "assert reverse_list([]) == []"],
         "reverse_list", solution="def reverse_list(xs):\n    return list(reversed(xs))"),
    Task("Write a function `count_char(s, c)` that returns how many times c appears in s.",
         ["assert count_char('aab', 'a') == 2", "assert count_char('abc', 'z') == 0"],
         "count_char", solution="def count_char(s, c):\n    return s.count(c)"),
]


def load_mbpp(split: str = "train", limit: int | None = None) -> list[Task]:
    """MBPP, fully materialised.

    Materialising is a deliberate exception to this project's streaming rule:
    the whole set is under a megabyte, RL revisits prompts across epochs, and a
    streaming iterator would silently give a different task order to the policy
    and the reference model.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    out = []
    for rec in ds:
        tests = list(rec.get("test_list") or [])
        text = rec.get("text") or rec.get("prompt")
        if not tests or not text:
            continue
        # MBPP's asserts name the function, and the bare text often does not, so
        # the signature is shown to the model. Without it the policy has to guess
        # the name and loses every reward for a reason unrelated to its reasoning.
        out.append(Task(f"{text}\n\nYour solution must satisfy:\n{tests[0]}",
                        tests, str(rec.get("task_id", "")),
                        setup=(rec.get("test_setup_code") or ""),
                        solution=(rec.get("code") or "")))
        if limit and len(out) >= limit:
            break
    if not out:
        raise RuntimeError("MBPP loaded but produced no usable tasks")
    return out


# A KNOWN LIMIT OF MBPP, not of this code: some problems reference helper types
# the prompt never defines (e.g. asserts calling Pair(...) with no Pair given),
# so a correct solution has to guess a matching definition. Those items cap the
# achievable pass rate below 100%, and a plateau there should not be read as the
# policy having stopped learning.


def get_tasks(name: str, limit: int | None = None) -> list[Task]:
    if name == "easy":
        return list(EASY_TASKS)
    if name == "mbpp":
        return load_mbpp(limit=limit)
    raise ValueError(f"unknown task set {name!r} (expected 'easy' or 'mbpp')")
