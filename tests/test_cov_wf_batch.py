"""The batch queue's quiet paths -- the ones taken when nothing is wrong.

Every case here is a branch that only runs when the *ordinary* thing happens:
a duplicated job that was given its own id, a removed job that most of the
queue never referred to, a dependency typo that is a typo and not a cycle, a
job whose settings name no folder, a summary written for a queue that kept no
log directory, and a child that finished by itself in the instant Stop was
pressed.

None of it needs a GPU. The queue is standard-library-only by design, so the
whole file drives the real functions -- the one child process is stood in for
by a stub only where the point is a *race* between the child exiting and the
Stop check, which cannot be produced reliably with a real process.
"""
from __future__ import annotations

import subprocess

import pytest

from spacr import batch
from spacr.batch import (
    ERROR,
    STATUS_SUCCESS,
    Job,
    Queue,
    QueueResult,
    plan,
    validate_queue,
)
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token


def _queue(*jobs: Job, name: str = "overnight") -> Queue:
    """A queue built without the add()-time validation, as loading one does."""
    return Queue(jobs=list(jobs), name=name)


# ---------------------------------------------------------------------------
# Job.copy -- Duplicate, with and without a chosen id
# ---------------------------------------------------------------------------

def test_a_duplicate_keeps_the_id_you_chose_and_forgets_the_one_you_did_not():
    """Duplicate is how a twelve-job night is built from one job, and the id
    is what depends_on refers to.

    A clone that silently kept the original's id would make two jobs share it,
    so an unnamed clone must come back id-less for the queue to mint a fresh
    one. But a caller who *did* name the clone -- the GUI passes the id it
    just minted -- must get that name back, or its depends_on wiring points at
    a job that does not exist.
    """
    original = Job(module="mask", id="mask-1", label="mask plate 1",
                   settings={"src": "/data/p1"}, overrides=["diameter=30"])
    original.status = STATUS_SUCCESS
    original.exit_code = 0

    unnamed = original.copy()
    named = original.copy(id="mask-7", label="mask plate 7")

    assert unnamed.id == "", "an unnamed clone kept the original's id"
    assert named.id == "mask-7"
    assert named.label == "mask plate 7"
    # Both clones are fresh regardless of which branch produced them.
    assert (unnamed.status, named.status) == (batch.STATUS_PENDING,
                                              batch.STATUS_PENDING)
    assert unnamed.exit_code is None and named.exit_code is None
    assert named.override_args == ["diameter=30"]
    assert named.settings == {"src": "/data/p1"}
    assert original.id == "mask-1" and original.status == STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Queue.remove -- rewriting only the jobs that actually referred to it
# ---------------------------------------------------------------------------

def test_removing_a_job_rewrites_its_dependants_and_leaves_the_rest_alone():
    """Deleting job 1 of twelve must not quietly re-wire the other eleven.

    A dangling depends_on would skip every job that referred to the deleted
    one, so those references are dropped -- but a job that depended on
    something *else* has to keep that dependency verbatim, or Measure would
    stop waiting for its own Mask and run against a half-written database.
    """
    mask = Job(module="mask", id="mask-1")
    measure = Job(module="measure", id="measure-1", depends_on=["mask-1"])
    classify = Job(module="classify", id="classify-1", depends_on=["measure-1"])
    both = Job(module="measure", id="measure-2",
               depends_on=["mask-1", "measure-1"])
    queue = _queue(mask, measure, classify, both)

    assert queue.remove("mask-1") is True
    assert queue.ids == ["measure-1", "classify-1", "measure-2"]
    assert measure.depends_on == [], "the dangling reference survived"
    assert classify.depends_on == ["measure-1"], (
        "an unrelated dependency was rewritten by an unrelated removal")
    assert both.depends_on == ["measure-1"], (
        "removing one dependency dropped the other one too")
    assert queue.remove("mask-1") is False, "a second removal claimed success"


# ---------------------------------------------------------------------------
# cycle detection -- a name that is not a job is not a cycle
# ---------------------------------------------------------------------------

def test_a_dependency_typo_is_reported_as_a_typo_not_as_a_cycle():
    """"depends_on names 'mesure-1'" tells the user what to fix; "dependency
    cycle" sends them looking for a loop that is not there.

    The cycle walk has to step over a name that is not a job in this queue --
    if it followed it, a queue holding one typo would either crash on the
    lookup or invent a cycle -- while a queue that really does loop must still
    be caught, because no run order can satisfy it.
    """
    typo = _queue(Job(module="mask", id="mask-1"),
                  Job(module="measure", id="measure-1",
                      depends_on=["mesure-1"]))
    messages = [p.message for p in validate_queue(typo) if p.is_error]
    assert any("'mesure-1'" in m and "not a job in this queue" in m
               for m in messages), messages
    assert not [m for m in messages if "dependency cycle" in m], (
        "an unknown dependency was reported as a cycle")

    # Same walk, a real loop: it must not be silent just because it steps
    # over unknown names elsewhere.
    looped = _queue(Job(module="mask", id="a", depends_on=["b", "ghost"]),
                    Job(module="mask", id="b", depends_on=["a"]))
    cycles = [p.message for p in validate_queue(looped)
              if p.severity == ERROR and "dependency cycle" in p.message]
    assert cycles, [p.message for p in validate_queue(looped)]
    assert "a" in cycles[0] and "b" in cycles[0], cycles


# ---------------------------------------------------------------------------
# plan() -- the src line is only printed when there is a src
# ---------------------------------------------------------------------------

def test_the_plan_prints_a_src_line_only_for_the_jobs_that_have_one(tmp_path):
    """The plan is what a user reads before committing a night to the queue,
    and its ``src`` line is how they check each job points at the right plate.

    A job whose settings blank the src has nothing to show there. Printing an
    empty ``src`` line for it would read as "src is fine, it is just narrow"
    -- the exact misreading that sends a twelve-hour queue at the wrong
    folder. The line has to be absent, while the job beside it that *does*
    name a plate still shows it.
    """
    plate = tmp_path / "plate1"
    plate.mkdir()
    named = Job(module="mask", id="mask-1", label="mask plate 1",
                settings={"src": str(plate)})
    blank = Job(module="measure", id="measure-1", label="measure nowhere",
                settings={"src": ""})
    text = plan(_queue(named, blank))

    lines = [line.rstrip() for line in text.splitlines()]
    assert f"      src {plate}" in lines, text
    src_lines = [line for line in lines if line.startswith("      src")]
    assert src_lines == [f"      src {plate}"], (
        f"a job with no src still printed a src line: {src_lines}")
    # Both jobs are in the plan; only one of them contributes a src line.
    assert any("mask-1" in line and "mask plate 1" in line for line in lines)
    assert any("measure-1" in line and "measure nowhere" in line
               for line in lines)


def test_the_plan_says_which_settings_it_could_not_read(tmp_path):
    """A settings path that went missing between building the queue and
    planning it has to show up on the job's own line.

    Failing silently here is what produces a plan that looks complete and a
    queue that dies on job nine at 3 a.m.
    """
    missing = tmp_path / "gone.csv"
    text = plan(_queue(Job(module="mask", id="mask-1", label="mask",
                           settings=str(missing))))
    assert "settings unreadable" in text, text
    assert "mask-1" in text


# ---------------------------------------------------------------------------
# QueueResult.summary -- naming only the paths that exist
# ---------------------------------------------------------------------------

def test_the_summary_names_the_log_folder_and_the_queue_file_only_when_it_has_them():
    """The summary is read hours later, and its header is where the user goes
    to find the per-job logs and the queue file to resume.

    A queue run straight from the GUI with no queue file on disk, or one
    handed a runner that writes no logs, has no such path -- and a header line
    reading ``logs`` followed by nothing would send the user hunting for a
    folder that was never created.
    """
    queue = _queue(Job(module="mask", id="mask-1", label="mask"),
                   name="overnight")

    without_logs = QueueResult(queue=queue, log_dir="",
                               path="/srv/runs/overnight.queue.json",
                               started="2026-01-01T00:00:00+00:00",
                               finished="2026-01-01T00:30:00+00:00")
    header = without_logs.summary().splitlines()
    assert any(line.strip() == "queue     /srv/runs/overnight.queue.json"
               for line in header), header
    assert not [line for line in header if line.startswith(" logs")], (
        "a run that kept no log folder still advertised one")

    with_logs = QueueResult(queue=queue, log_dir="/srv/runs/logs",
                            path="/srv/runs/overnight.queue.json")
    lines = with_logs.summary().splitlines()
    assert any(line.strip() == "logs      /srv/runs/logs" for line in lines), lines
    assert any(line.strip() == "queue     /srv/runs/overnight.queue.json"
               for line in lines), lines
    assert "0 ok, 0 failed, 0 skipped, 1 not run" in lines[1], lines[1]
    assert without_logs.duration_s == 1800.0


# ---------------------------------------------------------------------------
# subprocess_runner -- Stop arriving as the child is already exiting
# ---------------------------------------------------------------------------

class _StubChild:
    """A child process that exits after ``alive_polls`` liveness checks."""

    def __init__(self, alive_polls: int, returncode: int = 0):
        self._alive_polls = alive_polls
        self._returncode = returncode
        self.polls = 0
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        self.polls += 1
        if self.polls <= self._alive_polls:
            return None
        self.returncode = self._returncode
        return self._returncode

    def wait(self, timeout=None):
        if self.terminated or self.killed:
            self.returncode = self._returncode
            return self._returncode
        raise subprocess.TimeoutExpired("spacr-run", timeout or 0)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _run_cancelled(monkeypatch, tmp_path, child, log_name):
    """Drive subprocess_runner with ``child`` and Stop already pressed."""
    monkeypatch.setattr(batch.subprocess, "Popen",
                        lambda *a, **kw: child)
    log = tmp_path / log_name
    token = CancellationToken()
    token.cancel("the user pressed Stop")
    with installed_token(token):
        with pytest.raises(PipelineCancelled):
            batch.subprocess_runner(Job(module="mask", id="mask-1"), "", str(log))
    return log


def test_a_child_that_finished_by_itself_is_not_signalled_on_the_way_out(
        tmp_path, monkeypatch):
    """Stop must not fire SIGTERM at a process id that is no longer the job's.

    A child that exited in the instant between the Stop check and the handler
    looking again has already been reaped; signalling then is at best noise in
    the log and at worst a signal delivered to whatever the OS next gave that
    pid. The cancellation still has to be recorded and re-raised, so the queue
    marks the job stopped rather than successful -- and a child that really
    *is* still running must still be told to stop.
    """
    finished = _StubChild(alive_polls=1)
    log = _run_cancelled(monkeypatch, tmp_path, finished, "01_finished.log")

    assert finished.polls >= 2, "the handler never re-checked the child"
    assert finished.terminated is False, (
        "an already-exited child was sent SIGTERM anyway")
    assert finished.killed is False
    assert "cancelled safely" in log.read_text(encoding="utf-8")

    # The same code path, with a child that is genuinely still alive: this one
    # does get terminated, which is what makes the absence above meaningful.
    running = _StubChild(alive_polls=99)
    other = _run_cancelled(monkeypatch, tmp_path, running, "02_running.log")
    assert running.terminated is True, "a live child was left running"
    assert running.killed is False, "SIGTERM was skipped in favour of SIGKILL"
    assert "cancelled safely" in other.read_text(encoding="utf-8")
