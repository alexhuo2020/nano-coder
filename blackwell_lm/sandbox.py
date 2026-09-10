"""Execution of untrusted, model-generated code.

The reward signal for a coding model has to come from RUNNING the code -- a
learned reward model on code is both weaker and reward-hackable -- which means
every RL step executes text the policy invented. Assume it is hostile, because
gradient descent will happily find whatever the sandbox fails to forbid.

WHAT THIS DOES AND DOES NOT PROTECT AGAINST. On Linux it applies POSIX resource
limits (CPU seconds, address space, open files, subprocesses), runs in a
throwaway directory, isolates the interpreter (`-I`), starts a new session and
kills the whole PROCESS GROUP on timeout. It does NOT create a network, mount,
or PID namespace, so it is a robustness boundary against runaway generated code
-- not a containment boundary against a determined attacker. For that, run the
trainer itself inside a container with no network egress.

On Windows the resource limits do not exist and this degrades to timeout-only.
That is a development convenience for writing tests, NOT the security boundary,
and `posix_limits_active()` reports which one you are getting so a run can
refuse to train against the weak one.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource            # POSIX only
except ImportError:            # pragma: no cover - Windows
    resource = None


def posix_limits_active() -> bool:
    """True when the real resource limits are available (i.e. on Linux)."""
    return resource is not None


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool
    returncode: int | None = None


def _limiter(cpu_s: int, mem_bytes: int, nofile: int, nproc: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        # blocks fork bombs; the limit is per-UID so keep it generous enough
        # that the trainer's own processes are unaffected
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_python(code: str, timeout: float = 6.0, mem_mb: int = 512,
               extra_files: dict[str, str] | None = None) -> RunResult:
    """Run `code` in a throwaway directory and return its result.

    The wall-clock `timeout` and RLIMIT_CPU are BOTH needed: RLIMIT_CPU alone
    never fires on code that sleeps or blocks on a socket, and a wall-clock
    timeout alone leaves a spinning process to be killed only after it has
    burned a core for the full duration.
    """
    workdir = tempfile.mkdtemp(prefix="bnano_sbx_")
    try:
        for name, content in (extra_files or {}).items():
            # keep writes inside the sandbox even if a filename tries to escape
            safe = os.path.normpath(os.path.join(workdir, name))
            if not safe.startswith(os.path.abspath(workdir) + os.sep):
                raise ValueError(f"extra_files name escapes the sandbox: {name!r}")
            os.makedirs(os.path.dirname(safe), exist_ok=True)
            with open(safe, "w", encoding="utf-8") as fh:
                fh.write(content)
        path = os.path.join(workdir, "_prog.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)

        kw = {}
        if resource is not None:
            kw["preexec_fn"] = _limiter(int(timeout) + 1, mem_mb * 1024 * 1024, 64, 256)
            kw["start_new_session"] = True      # gives us a killable process group
        p = subprocess.Popen(
            [sys.executable, "-I", path],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, **kw,
        )
        try:
            out, err = p.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            # kill the GROUP: killing only the direct child leaves anything it
            # spawned alive, holding the GPU box's memory for the rest of the run
            try:
                if resource is not None:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:                            # pragma: no cover - Windows
                    p.kill()
            except (ProcessLookupError, PermissionError):
                pass
            out, err = p.communicate()
            timed_out = True
        return RunResult(ok=(not timed_out and p.returncode == 0), stdout=out or "",
                         stderr=err or "", timed_out=timed_out, returncode=p.returncode)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_HARNESS = r'''
import json, sys
_results = []
try:
    exec(compile(open(%(prog)r, encoding="utf-8").read(), "solution", "exec"), globals())
    # Dataset-provided fixtures (MBPP's test_setup_code) run AFTER the solution
    # and are not scored: they build objects the asserts reference. Skipping them
    # makes those tasks unsolvable and silently depresses the reward.
    if %(setup)r.strip():
        exec(compile(%(setup)r, "setup", "exec"), globals())
    _setup_ok = True
    _setup_err = ""
except BaseException as e:
    _setup_ok, _setup_err = False, f"{type(e).__name__}: {e}"
if _setup_ok:
    for _t in %(tests)r:
        try:
            exec(_t, globals())
            _results.append(True)
        except BaseException:
            _results.append(False)
print("__BNANO__" + json.dumps({"setup_ok": _setup_ok, "setup_err": _setup_err,
                                "results": _results}))
'''


def run_tests(code: str, tests: list[str], timeout: float = 6.0,
              mem_mb: int = 512, setup: str = "") -> tuple[int, int, str]:
    """Run `code`, then each test string, in ONE subprocess.

    One process rather than one per assert: a group rollout scores G*|tests|
    snippets per prompt per step, and process startup would dominate the step
    time. Each test is still isolated by try/except, so one failure does not
    mask the others -- which is what makes PARTIAL credit possible.

    Returns (passed, total, detail).
    """
    workdir = tempfile.mkdtemp(prefix="bnano_h_")
    try:
        prog = os.path.join(workdir, "solution.py")
        with open(prog, "w", encoding="utf-8") as fh:
            fh.write(code)
        harness = _HARNESS % {"prog": prog, "tests": list(tests), "setup": setup or ""}
        r = run_python(harness, timeout=timeout, mem_mb=mem_mb)
        marker = "__BNANO__"
        if marker in r.stdout:
            payload = json.loads(r.stdout.split(marker, 1)[1].splitlines()[0])
            if not payload["setup_ok"]:
                return 0, len(tests), f"setup failed: {payload['setup_err']}"
            res = payload["results"]
            return sum(res), len(tests), ""
        if r.timed_out:
            return 0, len(tests), "timeout"
        return 0, len(tests), (r.stderr or "no harness output")[-400:]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
