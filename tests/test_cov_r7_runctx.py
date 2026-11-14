"""Round-7 ``spacr.runctx``: resource accounting that must never cost the run.

Resource accounting's three ``except`` arms --
:func:`_start_resource_accounting`, :func:`_register_resource_artifact` and
:func:`_stop_resource_accounting`. Each is driven here by breaking exactly
one collaborator, and each test also drives the working path in the same
place, so "no artifact was registered" is a fact about the failure and not
about the fixture. Worker labelling and the GUI-preference resolver live in
tests/test_runctx_run_lifecycle.py.

CPU-only and offline throughout.
"""
from __future__ import annotations

import logging

from spacr import artifacts, ports, runctx
from spacr.runctx import RunContext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _context(sampler=None, **kwargs):
    return RunContext(run_id="r7", module="mask",
                      _resource_sampler=sampler, **kwargs)


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
    log.write_text('{"schema_version": 3, "mode": "summary"}', encoding="utf-8")

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
