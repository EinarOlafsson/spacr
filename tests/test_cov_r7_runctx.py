"""Round-7 ``spacr.runctx``: the diagnostics that must never cost the run.

Everything in this module's round-7 target list sits on one of two seams.

The first is :meth:`RunContext.register_worker` — the seam a sampled child
process uses to say *which* worker it is. Every one of its escapes returns
the empty string rather than raising, because a label is a diagnostic and a
run that dies for want of one is worse than a run whose process tree is
anonymous. Pinned here: no sampler at all, a caller that hands over a
complete stamp, a creation time supplied rather than looked up, a pid psutil
cannot read, and a sampler that refuses the stamp.

The second is resource accounting's three ``except`` arms —
:func:`_start_resource_accounting`, :func:`_register_resource_artifact` and
:func:`_stop_resource_accounting`. Each is driven here by breaking exactly
one collaborator, and each test also drives the working path in the same
place, so "no artifact was registered" is a fact about the failure and not
about the fixture. Plus the GUI-preference resolver, which must not pull Qt
into a headless run: three of its four ways of answering ``None``.

One target is unreachable and is proved rather than driven: the
``if exc is None`` guard in :meth:`ErrorPolicy._give_up`.

CPU-only and offline throughout.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import pytest

from spacr import artifacts, ports, runctx
from spacr.runctx import ON_ERROR_RETRY, ErrorPolicy, RunContext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _RecordingSampler:
    """Stands in for ``fit_resources._ResourceSampler``.

    Only ``_register_worker`` is exercised; the real sampler owns a thread
    and a file, neither of which this seam touches.
    """

    def __init__(self, answer="stamped", explode=False):
        self.stamps = []
        self._answer = answer
        self._explode = explode

    def _register_worker(self, stamp):
        self.stamps.append(stamp)
        if self._explode:
            raise RuntimeError("sampler is not accepting workers")
        return self._answer


def _context(sampler=None, **kwargs):
    return RunContext(run_id="r7", module="mask",
                      _resource_sampler=sampler, **kwargs)


# ---------------------------------------------------------------------------
# RunContext.register_worker
# ---------------------------------------------------------------------------

def test_a_run_without_resource_accounting_labels_nothing_and_says_so():
    """runctx.py:1341-1342 -- no sampler means the empty identity.

    ``register_worker`` is called unconditionally by workers that do not
    know whether accounting is on, so "off" has to be an answer rather than
    an AttributeError on ``None._register_worker``.
    """
    off = _context()
    assert off.register_worker("cellpose", 3) == ""

    # ... and with a sampler the same call is answered with the sampler's
    # own identity, so the empty string above is the *absence* of accounting
    # and not this method's only reply.
    sampler = _RecordingSampler(answer="1234:9.5")
    on = _context(sampler)
    assert on.register_worker("cellpose", 3) == "1234:9.5"
    assert sampler.stamps[0]["worker_kind"] == "cellpose"
    assert sampler.stamps[0]["worker_id"] == "3"


def test_a_worker_that_brings_its_own_stamp_has_it_passed_through_whole():
    """runctx.py:1343-1344 -- a Mapping is the stamp, not a worker kind.

    A spawned worker reports its own ``(pid, create_time)`` because the
    parent cannot read them without a PID-reuse race. That stamp must reach
    the sampler unrebuilt: rebuilding it here would substitute the PARENT's
    pid and creation time and attribute the child's memory to the wrong
    process.
    """
    sampler = _RecordingSampler()
    run = _context(sampler)

    stamp = {"pid": 4242, "create_time": 111.5,
             "worker_kind": "fastq-saver", "worker_id": "7"}
    assert run.register_worker(stamp) == "stamped"
    assert sampler.stamps[0] == stamp
    assert sampler.stamps[0] is not stamp        # copied, not aliased

    # The same arguments spelled as a kind rebuild the stamp around THIS
    # process instead, which is what the Mapping branch exists to avoid.
    assert run.register_worker("fastq-saver", "7") == "stamped"
    assert sampler.stamps[1]["pid"] == os.getpid()


def test_a_creation_time_that_was_supplied_is_not_looked_up_again(
        monkeypatch):
    """runctx.py:1348->1355 -- ``create_time`` short-circuits psutil.

    The caller that has the number already is the one that can be trusted
    with it; asking psutil again for a pid that has since exited would
    replace a good creation time with ``None``.
    """
    import psutil

    def _refuse(_pid):
        raise AssertionError("psutil must not be consulted here")

    monkeypatch.setattr(psutil, "Process", _refuse)

    sampler = _RecordingSampler()
    run = _context(sampler)
    run.register_worker("trial", 17, pid=4242, create_time=111.5)
    assert sampler.stamps[0] == {"pid": 4242, "create_time": 111.5,
                                 "worker_kind": "trial", "worker_id": "17"}


def test_a_pid_psutil_cannot_read_costs_the_creation_time_not_the_run():
    """runctx.py:1353-1354 -- an unreadable process leaves ``create_time`` None.

    A worker can exit between being spawned and being labelled. The stamp is
    still worth making -- it names the kind and the id -- so the lookup
    failure is swallowed and only the creation time is lost.
    """
    sampler = _RecordingSampler()
    run = _context(sampler)

    # A pid above the kernel's maximum cannot exist, so psutil raises.
    run.register_worker("ghost", 1, pid=2 ** 30)
    assert sampler.stamps[0]["create_time"] is None
    assert sampler.stamps[0]["worker_kind"] == "ghost"

    # This process CAN be read, so the None above is the failed lookup and
    # not a creation time this method never fills in.
    run.register_worker("live", 2)
    assert isinstance(sampler.stamps[1]["create_time"], float)
    assert sampler.stamps[1]["create_time"] > 0


def test_a_sampler_that_refuses_a_stamp_costs_the_label_not_the_run(caplog):
    """runctx.py:1363-1365 -- a refused registration is logged and shrugged off.

    ``_register_worker`` reaches into the sampler's own bookkeeping, which
    is racing a sampling thread. Losing that race must cost the caller its
    identity string and nothing else.
    """
    broken = _RecordingSampler(explode=True)
    run = _context(broken)

    with caplog.at_level(logging.DEBUG, logger="spacr.mask"):
        assert run.register_worker("cellpose", 1) == ""
    assert broken.stamps                       # it really was attempted
    assert any("could not label resource worker" in record.message
               for record in caplog.records)

    # A sampler that accepts the stamp answers with an identity, so the
    # empty string above is the refusal and not this method's usual reply.
    assert _context(_RecordingSampler("9:1.0")).register_worker(
        "cellpose", 1) == "9:1.0"


# ---------------------------------------------------------------------------
# _performance_logging_preference
# ---------------------------------------------------------------------------

def test_the_environment_beats_the_gui_preference_by_declining_to_answer(
        monkeypatch):
    """runctx.py:1412-1415 -- ``SPACR_PERFORMANCE_LOG`` returns ``None``.

    ``None`` is not "off" here: it hands the decision to the sampler, which
    resolves the environment itself. Returning a mode instead would record
    a preference as the reason the run was profiled, when the environment
    was.
    """
    monkeypatch.setenv("SPACR_PERFORMANCE_LOG", "detailed")

    # An explicit setting still wins -- it is tested one line earlier.
    assert runctx._performance_logging_preference(
        {"performance_logging": "summary"}) == "summary"
    # With no setting, the environment's presence is answered with None.
    assert runctx._performance_logging_preference({}) is None


def test_a_headless_run_never_imports_qt_to_ask_about_logging(monkeypatch):
    """runctx.py:1416-1417 -- Qt not already loaded means no preference.

    The membership test is on ``sys.modules`` precisely so that resolving
    this preference cannot be what drags PySide6 into a batch job. The
    import below the test must therefore not run when Qt is absent.
    """
    monkeypatch.delenv("SPACR_PERFORMANCE_LOG", raising=False)
    monkeypatch.delitem(sys.modules, "PySide6.QtCore", raising=False)

    def _never(*_args, **_kwargs):
        raise AssertionError("Qt preferences must not be imported")

    monkeypatch.setitem(sys.modules, "spacr.qt",
                        types.SimpleNamespace(preferences=None))
    monkeypatch.setitem(
        sys.modules, "spacr.qt.preferences",
        types.SimpleNamespace(get_performance_logging=_never))

    assert runctx._performance_logging_preference({}) is None

    # With Qt already loaded the same call DOES read the preference, which
    # is what makes the None above a decision about sys.modules.
    preferences = types.SimpleNamespace(
        get_performance_logging=lambda: "detailed")
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", types.ModuleType(
        "PySide6.QtCore"))
    monkeypatch.setitem(sys.modules, "spacr.qt",
                        types.SimpleNamespace(preferences=preferences))
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", preferences)
    assert runctx._performance_logging_preference({}) == "detailed"


def test_a_gui_preference_that_cannot_be_read_is_not_a_failed_run(monkeypatch):
    """runctx.py:1422-1423 -- an unreadable preference falls back to ``None``.

    The preference store is a QSettings file a user can corrupt. Reading it
    is a nicety; failing a headless pipeline over it is not.
    """
    monkeypatch.delenv("SPACR_PERFORMANCE_LOG", raising=False)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore",
                        types.ModuleType("PySide6.QtCore"))

    def _corrupt():
        raise RuntimeError("preferences file is not readable")

    broken = types.SimpleNamespace(get_performance_logging=_corrupt)
    monkeypatch.setitem(sys.modules, "spacr.qt",
                        types.SimpleNamespace(preferences=broken))
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", broken)
    assert runctx._performance_logging_preference({}) is None

    # The same wiring with a readable store answers with the preference, so
    # the None above is the exception being swallowed.
    working = types.SimpleNamespace(
        get_performance_logging=lambda: "summary")
    monkeypatch.setitem(sys.modules, "spacr.qt",
                        types.SimpleNamespace(preferences=working))
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", working)
    assert runctx._performance_logging_preference({}) == "summary"


# ---------------------------------------------------------------------------
# the three except arms of resource accounting
#
# These three are module-level privates with no public seam of their own: a
# run reaches them only through `run_context`, which would also have to be
# made to fail. They are called directly so that exactly one collaborator is
# broken per test.
# ---------------------------------------------------------------------------

def test_a_sampler_that_will_not_start_leaves_the_run_unaccounted(
        tmp_path, monkeypatch, caplog):
    """runctx.py:1439-1441 -- accounting that cannot start is a warning.

    ``_resource_sampler`` must be left as ``None`` rather than as a
    half-built object, because every later call -- ``register_worker``,
    ``_stop_resource_accounting`` -- tests exactly that attribute for
    ``None`` to decide whether accounting is on.
    """
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    from spacr import fit_resources

    def _refuse(*_args, **_kwargs):
        raise OSError("no room for a resource log")

    monkeypatch.setattr(fit_resources, "_ResourceSampler", _refuse)

    run = _context(sampler="left over from before")
    with caplog.at_level(logging.WARNING, logger="spacr.mask"):
        runctx._start_resource_accounting(run)

    assert run._resource_sampler is None
    assert run.resource_log_path == ""
    assert any("could not start resource accounting" in record.message
               for record in caplog.records)
    # The run is still usable: the seam that reads the sampler answers.
    assert run.register_worker("cellpose", 1) == ""

    # The unbroken constructor does arm it, so the None above is the
    # failure and not what this function always leaves behind.
    monkeypatch.undo()
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    working = _context(settings={"performance_logging": "off"})
    runctx._start_resource_accounting(working)
    try:
        assert working._resource_sampler is not None
    finally:
        runctx._stop_resource_accounting(working, "completed")


def test_a_resource_log_that_cannot_be_registered_is_still_written(
        tmp_path, monkeypatch, caplog):
    """runctx.py:1475-1476 -- a refused artifact registration is a warning.

    The document on disk is the measurement; the registry row is only how it
    is found again. Losing the row must not lose the run, and must not leave
    a ``resource_artifact_id`` naming a record that was never made.
    """
    project = tmp_path / "project"
    project.mkdir()
    log = tmp_path / "r7.resources.json"
    log.write_text('{"schema_version": 3, "mode": "summary"}')

    run = _context(settings={"src": str(project)})
    run.resource_log_path = str(log)
    assert ports.project_root(run.settings, run.module)

    document = {"schema_version": 3, "mode": "summary", "summary": {}}

    def _refuse(**_kwargs):
        raise OSError("registry is read-only")

    monkeypatch.setattr(artifacts, "register", _refuse)
    with caplog.at_level(logging.WARNING, logger="spacr.mask"):
        runctx._register_resource_artifact(run, document, "completed")

    assert run.resource_artifact_id == ""
    assert any("could not register the resource log" in record.message
               for record in caplog.records)
    assert log.exists()

    # With a working registry the same call DOES record an id, so the empty
    # string above is the refusal rather than the fixture.
    monkeypatch.undo()
    runctx._register_resource_artifact(run, document, "completed")
    assert run.resource_artifact_id
    records = artifacts.by_kind("resource-log", project=project)
    assert [record.artifact_id for record in records] == [
        run.resource_artifact_id]
    assert records[0].status == artifacts.STATUS_COMPLETE


def test_a_sampler_that_will_not_stop_is_a_warning_not_a_failed_run(
        tmp_path, monkeypatch, caplog):
    """runctx.py:1503-1504 -- stopping accounting cannot fail the run.

    This runs on the way out of ``run_context``, after the pipeline has
    already produced its outputs. An exception here would replace a
    successful run's result with a diagnostics error.
    """
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))

    class _Stubborn:
        stopped = False

        def _stop(self, reason):
            _Stubborn.stopped = True
            raise RuntimeError(f"sampler thread will not join ({reason})")

    run = _context(_Stubborn())
    with caplog.at_level(logging.WARNING, logger="spacr.mask"):
        runctx._stop_resource_accounting(run, "completed")

    assert _Stubborn.stopped                    # it really was attempted
    assert any("could not finish resource accounting" in record.message
               for record in caplog.records)

    # A run with no sampler returns before any of that, so the warning above
    # is the failure and not something every stop emits.
    caplog.clear()
    quiet = _context()
    with caplog.at_level(logging.WARNING, logger="spacr.mask"):
        runctx._stop_resource_accounting(quiet, "completed")
    assert caplog.records == []


# ---------------------------------------------------------------------------
# Proved unreachable
# ---------------------------------------------------------------------------

def test_giving_up_on_a_unit_always_has_an_exception_to_give_up_over():
    """Why ``if exc is None`` in ``_give_up`` (runctx.py:1131) cannot be true.

    ``_give_up`` has exactly one caller, ``attempts_for`` at runctx.py:1120,
    reached only by falling out of ``for number in range(1, total + 1)``.
    Every iteration either returns -- on success at 1105, or at 1109 when the
    body never ran -- or assigns ``last = attempt.exc`` at 1110, which the
    line above has just tested is not ``None``. So the loop can only be left
    by exhausting a range whose last iteration assigned a real exception.

    The remaining way to reach 1120 with ``last`` still ``None`` would be an
    empty range, i.e. ``total < 1``; the constructor refuses that outright,
    which is the half of this worth keeping: a retry budget of zero would
    otherwise be a stop that reports itself as a retry.
    """
    with pytest.raises(ValueError, match="at least 1"):
        ErrorPolicy(ON_ERROR_RETRY, attempts=0)

    # The smallest budget the constructor does accept still runs the body,
    # so the exception that reaches _give_up is the unit's own.
    policy = ErrorPolicy(ON_ERROR_RETRY, attempts=1, sleep=lambda _s: None)
    with pytest.raises(ZeroDivisionError):
        for attempt in policy.attempts_for("plate1", stage="plate"):
            with attempt:
                1 / 0

    # And a body that never ran is returned on at 1109 rather than being
    # given up on, so it never reaches the guard either: nothing is
    # recorded as a failure.
    quiet = ErrorPolicy(ON_ERROR_RETRY, attempts=2, sleep=lambda _s: None)
    visits = 0
    for _attempt in quiet.attempts_for("plate2", stage="plate"):
        visits += 1                  # no inner `with`: the body never ran
    assert visits == 1               # returned at 1109, not retried
    assert quiet.skips == []
    assert quiet.retries == []
