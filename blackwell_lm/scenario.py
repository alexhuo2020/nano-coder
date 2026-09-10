"""Repo scenarios: a task becomes a real directory with broken code in it.

WHY. Until now the "agentic" task was: emit a snippet, we run it. The policy
never had to navigate anything, so nothing it learned would transfer to a real
coding agent. A scenario instead materialises a scratch repo containing a
DELIBERATELY BROKEN solution.py, and the agent has to read it, work out what is
wrong, write a fix, and run the tests -- which is the actual shape of the job.

THE BREAKAGE IS VERIFIED. Each scenario checks that the broken file really fails
and that the reference solution really passes, in the sandbox, before the
scenario is used. Without both checks you get two silent failure modes: a
"broken" file that happens to pass (the agent is rewarded for doing nothing) and
a reference that fails (the task is unsolvable and every episode scores zero,
which looks exactly like a policy problem).

Scenarios are CACHED after first construction: verifying one costs two real
subprocesses, and rebuilding them every epoch would make the data pipeline the
bottleneck -- the same mistake already made once with the tool-SFT trajectories.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from blackwell_lm.sandbox import run_tests

# Same mutation family as tool_sft: small, plausible, and syntactically valid, so
# the agent has to reason about behaviour rather than spot a SyntaxError.
_MUTATIONS = (
    (r"\+", "-"),
    (r"\breturn\b", "return not"),
    (r"\bmax\b", "min"),
    (r"\bsorted\(", "reversed("),
    (r"== ", "!= "),
    (r"\brange\(", "range(1, "),
    (r"\blen\(", "id("),
)

README = """# task

{prompt}

`solution.py` is failing its tests. Fix it.
"""


@dataclass
class Scenario:
    name: str
    prompt: str
    tests: list[str]
    setup: str = ""
    broken: str = ""
    reference: str = ""
    files: dict = field(default_factory=dict)

    def materialise(self, root: str | None = None) -> str:
        """Write this scenario into a fresh directory and return its path."""
        repo = root or tempfile.mkdtemp(prefix="bnano_repo_")
        os.makedirs(repo, exist_ok=True)
        for rel, content in self.files.items():
            path = os.path.join(repo, rel)
            os.makedirs(os.path.dirname(path) or repo, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        return repo


def _mutate(code: str, rng: random.Random):
    opts = [(p, r) for p, r in _MUTATIONS if re.search(p, code)]
    rng.shuffle(opts)
    for p, r in opts:
        yield re.sub(p, r, code, count=1)


def build_scenario(task, rng: random.Random, timeout: float = 6.0) -> Scenario | None:
    """One verified repo scenario, or None if it cannot be built honestly."""
    ref = (getattr(task, "solution", None) or "").strip()
    if not ref or not task.tests:
        return None

    setup = getattr(task, "setup", "")
    ok, total, _ = run_tests(ref, task.tests, timeout=timeout, setup=setup)
    if total == 0 or ok != total:
        return None                      # reference does not pass: unsolvable

    for cand in _mutate(ref, rng):
        p_bad, t_bad, _ = run_tests(cand, task.tests, timeout=timeout, setup=setup)
        if p_bad < t_bad:                # verified to actually fail
            prompt = (f"{task.prompt}\n\nThe file solution.py in this repo is "
                      f"failing its tests. Read it, fix it, and verify with "
                      f"run_tests.")
            return Scenario(
                name=task.name or "task",
                prompt=prompt,
                tests=list(task.tests),
                setup=setup,
                broken=cand,
                reference=ref,
                files={
                    "solution.py": cand,
                    "README.md": README.format(prompt=task.prompt),
                },
            )
    return None                          # no mutation broke it: skip


def build_scenarios(tasks, seed: int = 0, limit: int | None = None,
                    timeout: float = 6.0):
    """Verified scenarios, built once. Reports how many were rejected and why
    counts matter: a silently small pool means the agent sees the same handful
    of repos and the reward stops measuring generalisation."""
    rng = random.Random(seed)
    out, no_ref, ref_fails, unbreakable = [], 0, 0, 0
    for t in tasks:
        if not (getattr(t, "solution", None) or "").strip():
            no_ref += 1
            continue
        sc = build_scenario(t, rng, timeout=timeout)
        if sc is None:
            # distinguish the two rejection causes for the log
            ok, total, _ = run_tests(t.solution, t.tests, timeout=timeout,
                                     setup=getattr(t, "setup", ""))
            if total == 0 or ok != total:
                ref_fails += 1
            else:
                unbreakable += 1
            continue
        out.append(sc)
        if limit and len(out) >= limit:
            break
    print(f"[scenario] built {len(out)} verified repo scenarios "
          f"(rejected: {no_ref} no reference, {ref_fails} reference fails its "
          f"own tests, {unbreakable} no mutation broke it)", flush=True)
    if not out:
        raise RuntimeError("no scenarios could be built")
    return out


def cleanup(repo: str):
    shutil.rmtree(repo, ignore_errors=True)
