"""How long a job has taken, and what a queue check prints about it.

``elapsed_s`` has three answers and they mean different things: a finished
duration, a clock still running, and None for a job that has not started. A
queue view showing 0 for the third would say a pending job took no time, which
is the same number a job that finished instantly shows.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _job(**changes):
    from spacr.batch import Job

    fields = dict(module="measure", overrides={}, depends_on=())
    fields.update(changes)
    return Job(**fields)


def _iso(seconds_ago):
    return (datetime.now(timezone.utc)
            - timedelta(seconds=seconds_ago)).isoformat()


def test_a_finished_job_reports_the_duration_it_recorded():
    """The first branch: a recorded duration wins over any clock.

    Recomputing from the timestamps would drift with the reader's clock and
    make a finished job's time change every time the view refreshed.
    """
    from spacr.batch import STATUS_SUCCESS

    # duration_s is derived from the two timestamps, not stored.
    job = _job(status=STATUS_SUCCESS, started=_iso(912.5),
               finished=_iso(900))

    assert job.duration_s == pytest.approx(12.5, abs=0.5)
    assert job.elapsed_s == job.duration_s


def test_a_running_job_counts_up_from_when_it_started():
    """The second: the time SO FAR, which is what a live view shows."""
    from spacr.batch import STATUS_RUNNING

    job = _job(status=STATUS_RUNNING, started=_iso(30))

    elapsed = job.elapsed_s

    assert elapsed is not None
    assert 25.0 <= elapsed <= 60.0


def test_a_running_job_never_reports_a_negative_time():
    """The ``max(0.0, ...)``, which a clock skew produces.

    A start stamped in the future -- a container whose clock was set after the
    job began, or a queue file copied between machines -- would otherwise show
    a job that has taken minus four seconds.
    """
    from spacr.batch import STATUS_RUNNING

    job = _job(status=STATUS_RUNNING, started=_iso(-120))

    assert job.elapsed_s == 0.0


def test_a_pending_job_reports_no_time_at_all():
    """The third: None, and deliberately not zero.

    Zero is what a job that finished instantly shows, and a queue view cannot
    tell the two apart if this returns it.
    """
    job = _job()

    assert job.elapsed_s is None


def test_a_job_with_an_unreadable_start_reports_no_time():
    """``_parse_iso`` returning None, which a hand-edited queue file produces."""
    from spacr.batch import STATUS_RUNNING

    job = _job(status=STATUS_RUNNING, started="not a timestamp")

    assert job.elapsed_s is None


# ---------------------------------------------------------------------------
# format_problems — errors, warnings, and neither
# ---------------------------------------------------------------------------

def _problem(severity, message="something", job_id="j1"):
    from spacr.batch import Problem

    return Problem(job_id=job_id, severity=severity, message=message,
                   fix="do the thing")


def test_a_clean_queue_says_every_job_is_runnable():
    """The early return, which is the answer a user wants to see."""
    from spacr.batch import format_problems

    assert "every job is runnable" in format_problems([])


def test_errors_are_headed_as_blocking():
    """The wording matters: the queue will not start until they are fixed."""
    from spacr.batch import format_problems

    text = format_problems([_problem("error", "no src")])

    assert "will not start" in text
    assert "no src" in text


def test_warnings_are_headed_as_non_blocking():
    """The other heading, which says the queue WILL run.

    Conflating the two would either stop a runnable queue or start an
    unrunnable one, depending on which way they were merged.
    """
    from spacr.batch import format_problems

    text = format_problems([_problem("warning", "no dst set")])

    assert "will run, but check" in text
    assert "will not start" not in text


def test_both_kinds_are_printed_under_their_own_headings():
    """Both sections at once, which is the ordinary result of a real check."""
    from spacr.batch import format_problems

    text = format_problems([_problem("error", "no src"),
                            _problem("warning", "no dst set")])

    assert "will not start" in text and "will run, but check" in text
    assert text.index("will not start") < text.index("will run, but check")


# ---------------------------------------------------------------------------
# Queue as a container
# ---------------------------------------------------------------------------

def test_a_queue_is_sized_and_iterated_like_its_job_list():
    """``__len__`` and ``__iter__``, which every caller relies on.

    They are one line each and untested, which is exactly how a container
    comes to iterate something other than what it counts.
    """
    from spacr.batch import Queue

    jobs = [_job(id="a"), _job(id="b"), _job(id="c")]
    queue = Queue(jobs=jobs, created=_iso(0))

    assert len(queue) == 3
    assert [job.id for job in queue] == ["a", "b", "c"]
    assert list(queue) == jobs
