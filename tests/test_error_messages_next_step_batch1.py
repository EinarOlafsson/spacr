"""Recovery messages retain failure details and existing failure behavior."""
from __future__ import annotations

import logging
import subprocess
from unittest.mock import Mock

import pytest

from spacr import updater
from spacr.errors import RunLedger, STATUS_PARTIAL
from spacr.runctx import ErrorPolicy


def test_partial_run_names_the_failed_item_and_how_to_recover(caplog):
    """Log the failed item and recovery action without changing the ledger."""
    ledger = RunLedger("measure")
    with caplog.at_level(logging.ERROR):
        failure = ledger.record_failure("A01", stage="read", exc=ValueError("bad shape"))
        returned = ledger.finalize()
    assert returned is ledger
    assert ledger.status == STATUS_PARTIAL
    assert ledger.n_failed == 1
    assert failure.message == "bad shape"
    messages = [record.getMessage() for record in caplog.records]
    assert any("FAILED A01 (stage read): ValueError: bad shape" in text
               and "resolve the reported cause" in text for text in messages)
    assert any("RUN INCOMPLETE" in text
               and "Read the failed-item list in the run report." in text
               and "before using the artifacts as a complete result" in text
               for text in messages)


@pytest.mark.parametrize("mode", ["stop", "retry"])
def test_run_policy_keeps_the_original_exception_and_adds_a_next_step(mode, caplog):
    """Both abort policies retain the exception while explaining recovery."""
    error = ValueError("A01 has an invalid channel")
    operation = Mock(side_effect=error)
    policy = ErrorPolicy(mode, attempts=2, backoff=0)
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as caught:
        policy.run("A01", operation, stage="measure")
    assert caught.value is error
    assert operation.call_count == (2 if mode == "retry" else 1)
    assert policy.ledger.n_failed == 1
    assert any(f"on_error={mode}: A01 failed" in record.getMessage()
               and "Read the error for this unit in the run log" in record.getMessage()
               for record in caplog.records)


@pytest.mark.parametrize("outcome", ["timeout", "failure"])
def test_update_failures_log_translatable_recovery_and_keep_the_result(
        outcome, monkeypatch, caplog):
    """An offline packaging failure keeps its return value and gains advice."""
    from spacr.qt import i18n

    monkeypatch.setattr(updater, "_apply_network_settings", Mock())
    translate = Mock(wraps=str)
    monkeypatch.setattr(i18n, "tr", translate)
    if outcome == "timeout":
        runner = Mock(side_effect=subprocess.TimeoutExpired(["package-tool"], 120))
        expected = (1, "The command timed out after 2 minutes.")
        diagnostic = "Command timed out after 120 seconds"
        recovery = "Read the command log to find the stalled step"
    else:
        runner = Mock(return_value=subprocess.CompletedProcess(
            ["package-tool"], 2, "", "permission denied"))
        expected = (2, "permission denied")
        diagnostic = "Command failed (2):"
        recovery = "Read the command output above"
    monkeypatch.setattr(updater.subprocess, "run", runner)
    with caplog.at_level(logging.ERROR):
        result = updater.run_install_command(["package-tool"], timeout=120)
    assert result == expected
    runner.assert_called_once_with(
        ["package-tool"], capture_output=True, text=True, timeout=120)
    assert any(diagnostic in record.getMessage()
               and recovery in record.getMessage() for record in caplog.records)
    assert any(recovery in call.args[0] for call in translate.call_args_list)


@pytest.mark.parametrize("problem", ["header", "dtype", "shape", "truncated"])
def test_import_preview_keeps_the_bad_file_and_explains_recovery(problem, tmp_path):
    """Preview reports unsafe arrays without importing or repairing their data."""
    import numpy as np
    from spacr import convert

    path = tmp_path / "field.npy"
    if problem == "header":
        path.write_bytes(b"not an npy file")
        diagnostic, recovery = "NPY header is missing", "Choose a valid NPY image"
    else:
        if problem == "dtype":
            data = np.array([[1j]], dtype=complex)
            diagnostic, recovery = "real numeric array", "Export the image as a numeric NPY"
        elif problem == "shape":
            data = np.zeros((2, 2, 2), dtype=np.uint8)
            diagnostic, recovery = "two-dimensional YX image", "Select an image plane"
        else:
            data = np.zeros((2, 2), dtype=np.uint8)
            diagnostic, recovery = "NPY data is truncated", "Restore or export the complete file"
        np.save(path, data, allow_pickle=False)
        if problem == "truncated":
            path.write_bytes(path.read_bytes()[:-1])
    before = path.read_bytes()
    sources = convert.scan(str(tmp_path), extensions=(".npy",))
    assert len(sources) == 1
    assert diagnostic in sources[0].meta["error"]
    assert recovery in sources[0].meta["error"]
    assert path.read_bytes() == before


def test_missing_import_folder_keeps_exception_type_path_and_next_step(tmp_path):
    """An absent folder is refused before scanning and names the recovery action."""
    from spacr import convert
    from spacr.errors import ConfigurationError

    missing = tmp_path / "missing"
    with pytest.raises(ConfigurationError) as caught:
        convert.scan(str(missing))
    assert str(missing) in str(caught.value)
    assert "Choose an existing image folder" in str(caught.value)
    assert not missing.exists()


def test_blocked_queue_job_explains_the_dependency_without_running_it(
        tmp_path, monkeypatch, caplog):
    """A failed prerequisite still blocks the job and is named in recovery advice."""
    from spacr import batch

    prerequisite = batch.Job(module="mask", id="mask-1", status=batch.STATUS_FAILED)
    dependent = batch.Job(module="measure", id="measure-1", depends_on=["mask-1"])
    queue = batch.Queue(jobs=[prerequisite, dependent])
    monkeypatch.setattr(batch, "validate_queue", Mock(return_value=[]))
    runner = Mock()
    with caplog.at_level(logging.ERROR):
        result = batch.run_queue(queue, log_dir=tmp_path, runner=runner, echo=False)
    assert result.queue is queue
    assert dependent.status == batch.STATUS_SKIPPED
    runner.assert_not_called()
    assert any("SKIPPED measure-1" in record.getMessage()
               and "mask-1" in record.getMessage()
               and "run that dependency successfully before this job" in record.getMessage()
               for record in caplog.records)
