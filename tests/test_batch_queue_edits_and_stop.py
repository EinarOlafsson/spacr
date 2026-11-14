"""Table-driven tests of the batch queue's editing, planning and Stop paths.

Covers :meth:`spacr.batch.Queue.remove` (only the jobs that referred to the
removed one are rewired), the dependency problems :func:`spacr.batch.validate_queue`
reports (a typo is a typo, a loop is a cycle), the per-job lines of
:func:`spacr.batch.plan`, how :func:`spacr.batch.subprocess_runner` stops a
child once Stop was pressed (an exited child is never signalled, a live one
gets SIGTERM, a deaf one SIGKILL after the grace period), and the thread-local
scopes of :func:`spacr.cancellation.installed_token` that carry the Stop.

The queue is standard-library only, so the real functions run; a stub child
stands in for a process only where the point is a race between the child
exiting and the Stop check, which a real process cannot produce reliably.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from spacr import batch
from spacr.batch import Job, Queue, plan, validate_queue
from spacr.cancellation import (
    CancellationToken,
    PipelineCancelled,
    current_token,
    installed_token,
)


def _queue(*jobs: Job, name: str = "overnight") -> Queue:
    """A queue built without the add()-time validation, as loading one does."""
    return Queue(jobs=list(jobs), name=name)


# ---------------------------------------------------------------------------
# Queue.remove
# ---------------------------------------------------------------------------

def _four_job_queue():
    """mask-1 <- measure-1 <- classify-1, and measure-2 after mask-1 and measure-1."""
    return _queue(Job(module="mask", id="mask-1"),
                  Job(module="measure", id="measure-1", depends_on=["mask-1"]),
                  Job(module="classify", id="classify-1", depends_on=["measure-1"]),
                  Job(module="measure", id="measure-2", depends_on=["mask-1", "measure-1"]))


# (case, ids removed in order, expected return values, expected {id: depends_on})
REMOVALS = [
    ("removing a root drops only the references to it", ["mask-1"], [True],
     {"measure-1": [], "classify-1": ["measure-1"], "measure-2": ["measure-1"]}),
    ("a second removal of the same id reports False", ["mask-1", "mask-1"], [True, False],
     {"measure-1": [], "classify-1": ["measure-1"], "measure-2": ["measure-1"]}),
    ("removing a middle job rewires both of its dependants", ["measure-1"], [True],
     {"mask-1": [], "classify-1": [], "measure-2": ["mask-1"]}),
    ("removing a leaf touches nothing else", ["classify-1"], [True],
     {"mask-1": [], "measure-1": ["mask-1"], "measure-2": ["mask-1", "measure-1"]}),
    ("an unknown id changes nothing", ["ghost"], [False],
     {"mask-1": [], "measure-1": ["mask-1"], "classify-1": ["measure-1"],
      "measure-2": ["mask-1", "measure-1"]}),
]


def test_queue_remove_rewires_only_the_jobs_that_referred_to_it(subtests):
    """``Queue.remove`` keeps order and every unrelated dependency verbatim."""
    for case_id, removed, returned, remaining in REMOVALS:
        with subtests.test(msg=case_id):
            queue = _four_job_queue()
            assert [queue.remove(job_id) for job_id in removed] == returned
            assert {job.id: job.depends_on for job in queue.jobs} == remaining
            assert queue.ids == list(remaining)


# ---------------------------------------------------------------------------
# validate_queue: dependency problems
# ---------------------------------------------------------------------------

def _dependency_messages(queue):
    """``(job, severity, message)`` of every dependency or cycle problem, in order."""
    return [(p.job_id, p.severity, p.message) for p in validate_queue(queue)
            if p.message.startswith(("depends_on", "dependency cycle"))
            or "depends on itself" in p.message]


DEPENDENCY_CASES = [
    ("a typo is reported as a typo, not a cycle",
     [Job(module="mask", id="mask-1"),
      Job(module="measure", id="measure-1", depends_on=["mesure-1"])],
     [("measure-1", "error", "depends_on names 'mesure-1', which is not a job in this queue.")]),
    ("a real loop is a cycle even beside an unknown name",
     [Job(module="mask", id="a", depends_on=["b", "ghost"]),
      Job(module="mask", id="b", depends_on=["a"])],
     [("a", "error", "dependency cycle: a -> b -> a"),
      ("a", "error", "depends_on names 'b', which comes *later* in the queue — "
                     "this job would always be skipped."),
      ("a", "error", "depends_on names 'ghost', which is not a job in this queue.")]),
    ("a self-reference is named as such",
     [Job(module="mask", id="a", depends_on=["a"])],
     [("a", "error", "dependency cycle: a -> a"),
      ("a", "error", "job 'a' depends on itself.")]),
    ("an earlier dependency is fine",
     [Job(module="mask", id="a"), Job(module="mask", id="b", depends_on=["a"])], []),
]


def test_validate_queue_dependency_problems(subtests):
    """Unknown, later, self and cyclic dependencies each get their own exact message."""
    for case_id, jobs, expected in DEPENDENCY_CASES:
        with subtests.test(msg=case_id):
            assert _dependency_messages(_queue(*jobs)) == expected


# ---------------------------------------------------------------------------
# plan: the per-job block
# ---------------------------------------------------------------------------

def _job_block(text):
    """The plan lines between the header and the pre-flight section."""
    lines = text.splitlines()
    start = lines.index(" nothing here ever runs two of them at once.") + 2
    end = next(i for i, line in enumerate(lines) if line.startswith(" pre-flight"))
    return lines[start:end]


PLAN_CASES = [
    ("a src line only for the job that names one",
     lambda d: [Job(module="mask", id="mask-1", label="mask plate 1",
                    settings={"src": str(d / "plate1")}),
                Job(module="measure", id="measure-1", label="measure nowhere",
                    settings={"src": ""})],
     lambda d: ["   1. mask-1     mask plate 1",
                "      spacr.core.preprocess_generate_masks()",
                f"      src {d / 'plate1'}",
                "",
                "   2. measure-1  measure nowhere",
                "      spacr.measure.measure_crop()",
                ""]),
    ("an unreadable settings file is named on the job's own line",
     lambda d: [Job(module="mask", id="mask-1", label="mask", settings=str(d / "gone.csv"))],
     lambda d: ["   1. mask-1  mask",
                "      spacr.core.preprocess_generate_masks()",
                f"      src (settings unreadable: settings file not found: {d / 'gone.csv'})",
                ""]),
]


def test_plan_job_lines(subtests, tmp_path):
    """Each planned job prints its id, label, entry point and only the src it really has."""
    for index, (case_id, jobs, expected) in enumerate(PLAN_CASES):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"plan{index}"
            (folder / "plate1").mkdir(parents=True)
            text = plan(_queue(*jobs(folder)))
            assert text.splitlines()[1] == (
                f" spaCR batch queue: overnight — {len(jobs(folder))} job(s)")
            assert _job_block(text) == expected(folder)


# ---------------------------------------------------------------------------
# subprocess_runner: Stop
# ---------------------------------------------------------------------------

class _StubChild:
    """A child process that exits after ``alive_polls`` liveness checks."""

    def __init__(self, alive_polls: int, returncode: int = 0):
        self._alive_polls = alive_polls
        self._returncode = returncode
        self.polls = 0
        self.signals = []
        self.returncode = None

    def poll(self):
        self.polls += 1
        if self.polls <= self._alive_polls:
            return None
        self.returncode = self._returncode
        return self._returncode

    def wait(self, timeout=None):
        if self.signals:
            self.returncode = self._returncode
            return self._returncode
        raise subprocess.TimeoutExpired("spacr-run", timeout or 0)

    def terminate(self):
        self.signals.append("SIGTERM")

    def kill(self):
        self.signals.append("SIGKILL")


def _deaf_child(ready_path):
    """A child that ignores SIGTERM and says so before it starts sleeping."""
    return [sys.executable, "-c", textwrap.dedent(f"""
        import pathlib, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(ready_path)!r}).write_text("deaf")
        time.sleep(120)
    """)]


def _stop_with_stub(monkeypatch, folder, child):
    """Run with Stop already pressed; return the signals the stub received."""
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **kw: child)
    token = CancellationToken()
    token.cancel("the user pressed Stop")
    with installed_token(token), pytest.raises(PipelineCancelled):
        batch.subprocess_runner(Job(module="mask", id="mask-1"), "", str(folder / "job.log"))
    assert child.polls >= 2, "the handler never re-checked the child"
    return child.signals, None


def _stop_a_deaf_child(monkeypatch, folder):
    """Run a real child that ignores SIGTERM and press Stop once it is deaf."""
    ready = folder / "child-is-deaf"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None: _deaf_child(ready))
    started = {}
    real_popen = subprocess.Popen

    def recording(*args, **kwargs):
        started["process"] = real_popen(*args, **kwargs)
        return started["process"]

    monkeypatch.setattr(batch.subprocess, "Popen", recording)
    token = CancellationToken()

    def stop_once_deaf():
        # Stop only after the ignore handler is installed: a child stopped
        # while the interpreter starts dies on SIGTERM and proves nothing.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.02)
        token.cancel("the user pressed Stop")

    watcher = threading.Thread(target=stop_once_deaf, daemon=True)
    watcher.start()
    began = time.monotonic()
    with installed_token(token), pytest.raises(PipelineCancelled):
        batch.subprocess_runner(Job(module="mask", id="mask-1"), "", str(folder / "job.log"))
    elapsed = time.monotonic() - began
    watcher.join(timeout=5.0)
    assert ready.exists(), "the child never installed its SIGTERM handler"
    process = started["process"]
    assert process.poll() == -9, f"the child ended with {process.returncode}, not by SIGKILL"
    return ["SIGKILL"], elapsed


# (case, run(monkeypatch, folder) -> (signals received, elapsed or None), expected signals)
STOP_CASES = [
    ("a child that already exited is not signalled",
     lambda mp, d: _stop_with_stub(mp, d, _StubChild(alive_polls=1)), []),
    ("a live child gets SIGTERM and no SIGKILL",
     lambda mp, d: _stop_with_stub(mp, d, _StubChild(alive_polls=99)), ["SIGTERM"]),
    ("a child that ignores SIGTERM is killed after the grace period",
     _stop_a_deaf_child, ["SIGKILL"]),
]


@pytest.mark.posix_only
def test_subprocess_runner_stops_a_cancelled_child(subtests, tmp_path, monkeypatch):
    """Stop re-raises, logs ``cancelled safely``, and signals only a child still running."""
    for index, (case_id, run, signals) in enumerate(STOP_CASES):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"stop{index}"
            folder.mkdir()
            with monkeypatch.context() as patch:
                received, elapsed = run(patch, folder)
            assert received == signals
            if elapsed is not None:
                # SIGTERM first, then the five-second grace period, then SIGKILL
                assert elapsed >= 5.0
            log = (folder / "job.log").read_text(encoding="utf-8")
            assert log.startswith("# spaCR queue job mask-1 (")
            assert "\n# cancelled safely " in log and log.endswith("; child process stopped\n")
            assert "# finished " not in log


# ---------------------------------------------------------------------------
# installed_token scopes
# ---------------------------------------------------------------------------

def test_installed_token_scopes_are_per_thread_and_nest(subtests):
    """A scope is visible only to its own thread and restores the outer token on exit."""
    outer, inner = CancellationToken(), CancellationToken()

    with subtests.test(msg="a worker thread does not inherit the starter's token"):
        seen = []
        with installed_token(outer):
            worker = threading.Thread(target=lambda: seen.append(current_token()))
            worker.start()
            worker.join()
            seen.append(current_token())
        assert seen == [None, outer]
        assert current_token() is None

    with subtests.test(msg="a worker's own scope never leaks into the starter"):
        seen = []

        def scoped():
            with installed_token(inner) as installed:
                seen.append(installed)

        with installed_token(outer):
            worker = threading.Thread(target=scoped)
            worker.start()
            worker.join()
            seen.append(current_token())
        assert seen == [inner, outer]

    with subtests.test(msg="a nested scope restores the outer token, even when it raised"):
        with installed_token(outer):
            with pytest.raises(PipelineCancelled), installed_token(inner):
                assert current_token() is inner
                inner.cancel("stop")
                inner.checkpoint()
            assert current_token() is outer
        assert current_token() is None
