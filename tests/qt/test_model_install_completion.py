"""Install cancellation, download endings and explicit package consent."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QMessageBox

from spacr.qt import model_install as install


@pytest.mark.parametrize("answer,accepted", [
    (QMessageBox.Yes, True), (QMessageBox.Cancel, False),
])
def test_package_confirmation_names_the_environment_risk_and_defaults_to_cancel(
        monkeypatch, answer, accepted):
    monkeypatch.setattr(install, "can_install_packages", lambda: True)
    prompt = Mock(return_value=answer)
    monkeypatch.setattr(QMessageBox, "warning", prompt)
    assert install.confirm_backend_install(None, "SAMCell", "spacr[samcell]") is accepted
    args = prompt.call_args.args
    assert 'pip install "spacr[samcell]"' in args[2]
    assert "torch" in args[2] and "environment" in args[2]
    assert args[-1] == QMessageBox.Cancel


@pytest.mark.parametrize("error", [ImportError, ValueError])
def test_unresolvable_backend_is_reported_as_unavailable(monkeypatch, error):
    monkeypatch.setattr(install, "find_spec", Mock(side_effect=error("no spec")))
    assert not install.is_importable("missing_backend")


def test_import_probe_does_not_import_a_backend(monkeypatch):
    probe = Mock(side_effect=[object(), None])
    monkeypatch.setattr(install, "find_spec", probe)
    assert install.is_importable("present_backend")
    assert not install.is_importable("missing_backend")
    assert probe.call_count == 2


def test_unknown_backend_cannot_open_an_installer(qtbot, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    box = install.SegmentationBackendCombo()
    qtbot.addWidget(box)
    dialog = Mock(side_effect=AssertionError("unknown backend offered an install"))
    monkeypatch.setattr(model_zoo_picker, "install_backend", dialog)
    assert install.backend_row("unlisted-backend") is None
    assert box.offer_install("unlisted-backend") is False
    previous = box.currentIndex()
    box.setCurrentText("unlisted-backend")
    assert box.currentIndex() == previous
    dialog.assert_not_called()


def test_cancelling_a_running_installer_stops_it_and_reports_once(qtbot):
    # A harmless real process, contained by the test runner's memory cgroup.
    job = install.PackageInstall("unused", command=[
        "/bin/sh", "-c", "printf 'ready\\n'; exec sleep 30",
    ])
    endings = []
    job.finished.connect(lambda *args: endings.append(args))
    assert job.start()
    try:
        qtbot.waitUntil(lambda: "ready" in job.output(), timeout=5000)
        assert job.is_running()
        job.cancel()
        assert not job.is_running()
        assert endings == [(False, "cancelled")]
        assert job not in install._RUNNING
        job.cancel()
        job._ended(9, QProcess.CrashExit)
        assert endings == [(False, "cancelled")]
    finally:
        if job.is_running():
            job.cancel()


def test_cancelling_before_start_never_launches_an_installer():
    job = install.PackageInstall("unused", command=["/must-not-run"])
    endings = []
    job.finished.connect(lambda *args: endings.append(args))
    job.cancel()
    assert endings == [(False, "cancelled")]
    assert not job.is_running() and job not in install._RUNNING


@pytest.mark.parametrize("unverified", [False, True])
def test_checkpoint_fetch_forwards_checksum_requirement_and_callbacks(
        monkeypatch, tmp_path, unverified):
    from spacr import model_zoo

    entry, progress, cancel = object(), Mock(), Mock(return_value=False)
    fetch = Mock(return_value=SimpleNamespace(path=tmp_path / "model"))
    monkeypatch.setattr(model_zoo, "install", fetch)
    job = install.CheckpointDownload(entry, tmp_path, unverified=unverified)
    assert job._fetch(progress=progress, cancel=cancel) is fetch.return_value
    fetch.assert_called_once_with(entry, str(tmp_path),
                                  require_checksum=not unverified,
                                  progress=progress, cancel=cancel)
    assert job.wait(1) is True and not job.is_running()


@pytest.mark.parametrize("failure", [None, RuntimeError("connection lost"), ValueError()])
def test_download_worker_reports_progress_and_exactly_one_ending(failure):
    progress, done, failed = [], [], []
    sentinel = object()

    def fetch(*, progress, cancel):
        assert cancel() is False
        progress(12, None)
        progress(20, 40)
        if failure is not None:
            raise failure
        return sentinel

    worker = install._DownloadWorker(fetch)
    worker.progressed.connect(lambda *args: progress.append(args))
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()  # Exercise work on the test thread; start no QThread.
    assert progress == [(12, 0), (20, 40)]
    if failure is None:
        assert done == [sentinel] and failed == []
    else:
        assert done == [] and failed == [str(failure) or type(failure).__name__]


@pytest.mark.parametrize("late_success", [False, True])
def test_checkpoint_cancel_wins_over_a_late_download_ending(late_success):
    job = install.CheckpointDownload(object(), "/unused")
    job._thread = Mock()
    job._worker = install._DownloadWorker(Mock())
    endings = []
    job.finished.connect(lambda *args: endings.append(args))
    install._RUNNING.add(job)
    try:
        job.cancel()
        assert job._worker.stop is True
        if late_success:
            job._succeeded(SimpleNamespace(path="/unused/model"))
        else:
            job._failed("connection lost")
        job._thread.quit.assert_called_once_with()
        assert endings == [(False, "cancelled")]
        job._release()
        assert job not in install._RUNNING
    finally:
        install._RUNNING.discard(job)


@pytest.mark.parametrize("size,expected", [
    (None, ""), (-1, ""), (12, "12 B"), (1536, "1.5 kB"),
    (1572864, "1.5 MB"), (1610612736, "1.5 GB"),
])
def test_download_size_uses_binary_units_and_handles_unknown_totals(size, expected):
    assert install.human_bytes(size) == expected
