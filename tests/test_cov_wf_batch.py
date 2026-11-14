"""The batch queue's quiet paths -- the ones taken when nothing is wrong.

Every case here is a branch that only runs when the *ordinary* thing happens:
a duplicated job that was given its own id, and a summary written for a queue
that kept no log directory.  Removal, dependency validation, the plan and the
Stop paths are table-driven in ``tests/test_batch_queue_edits_and_stop.py``.

None of it needs a GPU: the queue is standard-library-only by design.
"""
from __future__ import annotations

from spacr import batch
from spacr.batch import (
    STATUS_SUCCESS,
    Job,
    Queue,
    QueueResult,
)


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


# ---------------------------------------------------------------------------
# cycle detection -- a name that is not a job is not a cycle
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# plan() -- the src line is only printed when there is a src
# ---------------------------------------------------------------------------


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
