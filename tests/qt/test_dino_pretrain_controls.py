"""DINO UI operations select paths and pass settings without loading models."""
from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QThread
from PySide6.QtWidgets import QFileDialog, QProgressBar, QPushButton, QSpinBox, QWidget

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _summary(path, epochs=20):
    return {"path": path, "epochs": epochs, "loss": [1.25], "seconds": 0.1}


def test_controls_offer_existing_checkpoint_and_twenty_epochs(screen):
    container = screen.findChild(QWidget, "EmbeddingsDinoControls")
    assert container is not None
    assert container.findChild(QPushButton, "EmbeddingsDinoCheckpointButton")
    epochs = container.findChild(QSpinBox, "EmbeddingsDinoEpochs")
    assert epochs is not None and epochs.value() == 20
    assert epochs.minimum() == 1
    assert container.findChild(QProgressBar, "EmbeddingsDinoProgress")


def test_existing_checkpoint_picker_cancel_dedup_and_no_computation(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint selection must not load, train or embed")

    for name in ("_dino_pretrain", "_dino_encoder", "embed_array"):
        monkeypatch.setattr(emb, name, forbidden)
    path = tmp_path / "own.pt"
    path.write_text("Path-only picker must not decode this file")
    button = screen.findChild(QPushButton, "EmbeddingsDinoCheckpointButton")
    assert button is not None
    original = screen._foundation.currentData()
    count = screen._foundation.count()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    button.click()
    assert screen._foundation.currentData() == original
    assert screen._foundation.count() == count
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    button.click()
    assert screen.spec().backbone == emb._DINO_PREFIX + str(path)
    assert screen._foundation.count() == count + 1
    alias = tmp_path / "alias.pt"
    alias.symlink_to(path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(alias), ""))
    button.click()
    assert screen._foundation.count() == count + 1
    assert screen.spec().backbone == emb._DINO_PREFIX + str(path)
    assert not hasattr(screen, "_dino_summary")


def test_checkpoint_picker_refuses_a_missing_path(screen, monkeypatch, tmp_path):
    count = screen._foundation.count()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(tmp_path / "missing.pt"), ""))
    screen._dino_checkpoint.click()
    assert screen._foundation.count() == count
    assert "existing" in screen._status.text()


def test_gui_epochs_and_channel_policy_reach_backend_and_resume_entry(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    calls = []

    def fake(crops, path, **kwargs):
        calls.append((path, kwargs.copy(), screen._dino.isEnabled(),
                      screen._dino_checkpoint.isEnabled(), screen._dino_epochs.isEnabled()))
        kwargs["progress"](3, kwargs["epochs"], 2.5)
        return _summary(path, kwargs["epochs"])

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    path = str(tmp_path / "resume.pt")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
    screen.set_crops(np.zeros((2, 8, 8, 2), dtype=np.float32))
    screen._policy.setCurrentIndex(screen._policy.findData("project"))
    screen._dino_epochs.setValue(7)
    screen._dino.click()
    assert calls[0][0] == path
    assert calls[0][1]["epochs"] == 7
    assert calls[0][1]["channel_policy"] == "project"
    assert calls[0][1]["pretrained"] is False
    assert calls[0][2:] == (False, False, False)
    count = screen._foundation.count()
    screen._dino.click()
    assert screen._foundation.count() == count
    assert screen.spec().backbone == emb._DINO_PREFIX + path
    assert screen._dino.isEnabled() and screen._dino_epochs.isEnabled()
    assert screen._dino_progress_bar.value() == 7


def test_programmatic_default_stays_twenty_and_cancel_starts_nothing(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    seen = []
    monkeypatch.setattr(emb, "_dino_pretrain", lambda crops, path, **kw:
                        seen.append(kw["epochs"]) or _summary(path, kw["epochs"]))
    screen.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    screen._dino_epochs.setValue(7)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert screen._pretrain_dino() == ""
    assert not seen and screen._dino.isEnabled()
    screen._pretrain_dino(str(tmp_path / "default.pt"))
    assert seen == [20]


def test_failure_restores_controls_and_rejects_late_progress(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    callbacks = []

    def fail(crops, path, **kwargs):
        callbacks.append(kwargs["progress"])
        kwargs["progress"](1, 20, 3.0)
        raise ValueError("incompatible checkpoint")

    monkeypatch.setattr(emb, "_dino_pretrain", fail)
    screen.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    screen._pretrain_dino(str(tmp_path / "bad.pt"))
    assert screen._dino.isEnabled() and screen._dino_checkpoint.isEnabled()
    assert "incompatible checkpoint" in screen._status.text()
    value = screen._dino_progress_bar.value()
    callbacks[0](19, 20, 0.1)
    assert screen._dino_progress_bar.value() == value
    assert "incompatible checkpoint" in screen._status.text()


def test_old_run_progress_cannot_overwrite_a_new_run(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    callbacks = []

    def fake(crops, path, **kwargs):
        callbacks.append(kwargs["progress"])
        kwargs["progress"](1, 20, 2.0)
        if len(callbacks) == 2:
            callbacks[0](19, 20, 0.1)
            assert screen._dino_progress_bar.value() == 1
        return _summary(path)

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    screen.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    for filename in ("first.pt", "second.pt"):
        screen._pretrain_dino(str(tmp_path / filename))
    assert len(callbacks) == 2


def test_real_worker_progress_is_applied_on_gui_thread_and_busy_is_guarded(
        qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    started = threading.Event()
    release = threading.Event()
    seen = []

    class ObservedScreen(em.EmbeddingsScreen):
        def _on_dino_progress(self, generation, epoch, total, loss):
            seen.append((QThread.currentThread(), epoch, total, loss))
            super()._on_dino_progress(generation, epoch, total, loss)

    def fake(crops, path, **kwargs):
        assert QThread.currentThread() != widget.thread()
        kwargs["progress"](2, 7, 1.25)
        started.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release its owned worker")
        return _summary(path, 7)

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    widget = ObservedScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    path = str(tmp_path / "threaded.pt")
    try:
        assert widget._pretrain_dino(path, epochs=7) == path
        qtbot.waitUntil(lambda: started.is_set() and bool(seen), timeout=5000)
        assert seen == [(widget.thread(), 2, 7, 1.25)]
        assert widget._dino_progress_bar.value() == 2
        assert "1.250" in widget._dino_progress_bar.format()
        assert widget.is_busy() and not widget._dino.isEnabled()
        assert widget._pretrain_dino(path, epochs=7) == ""
    finally:
        release.set()
        qtbot.waitUntil(lambda: not widget.is_busy(), timeout=5000)
    assert widget._dino.isEnabled()
    assert widget.spec().backbone == emb._DINO_PREFIX + path


def test_close_ignores_queued_progress_and_deleted_signal_is_safe(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    callbacks = []

    def fake(crops, path, **kwargs):
        callbacks.append(kwargs["progress"])
        return _summary(path)

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    screen.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    path = str(tmp_path / "closed.pt")
    screen._pretrain_dino(path)
    screen.close()
    old = screen._dino_progress_bar.value()
    callbacks[0](1, 20, 99.0)
    assert screen._dino_progress_bar.value() == old
    assert screen._pretrain_dino(path) == ""
    screen.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    callbacks[0](2, 20, 99.0)


def test_closing_active_worker_drops_late_progress_and_completion(
        qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()

    def fake(crops, path, **kwargs):
        started.set()
        try:
            if not release.wait(5):
                raise RuntimeError("Test did not release its owned worker")
            kwargs["progress"](18, 20, 0.1)
            return _summary(path)
        finally:
            ended.set()

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    widget = em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    selected = widget._foundation.currentData()
    try:
        widget._pretrain_dino(str(tmp_path / "closed-active.pt"))
        qtbot.waitUntil(started.is_set, timeout=5000)
        timer = threading.Timer(0.05, release.set)
        timer.start()
        widget.close()
        timer.join(timeout=1)
        qtbot.waitUntil(ended.is_set, timeout=5000)
        QCoreApplication.processEvents()
        assert widget._dino_progress_bar.value() == 0
        assert widget._foundation.currentData() == selected
        assert not hasattr(widget, "_dino_summary")
        assert not widget.is_busy()
    finally:
        release.set()
        widget.close()


def test_cancel_before_worker_start_restores_controls_without_an_error(
        qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb
    from spacr.qt import job_runner

    original = job_runner.make_thread

    def cancelled(*args, **kwargs):
        thread, worker = original(*args, **kwargs)
        worker.request_cancel("test cancellation before start")
        return thread, worker

    def forbidden(*args, **kwargs):
        pytest.fail("A cancelled worker must not call the backend")

    monkeypatch.setattr(job_runner, "make_thread", cancelled)
    monkeypatch.setattr(emb, "_dino_pretrain", forbidden)
    widget = em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    failures = []
    widget._dino_jobs.job_failed.connect(failures.append)
    widget._pretrain_dino(str(tmp_path / "cancelled.pt"))
    qtbot.waitUntil(lambda: widget._dino_jobs.active_jobs() == 0
                   and widget._dino_jobs.pending_jobs() == 0, timeout=5000)
    assert failures == []
    assert not widget.is_busy()
    assert widget._dino.isEnabled() and widget._dino_epochs.isEnabled()
    assert "stopped" in widget._status.text()


def test_an_old_failed_job_cannot_clear_a_new_pending_run(
        qtbot, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    release = threading.Event()
    started = threading.Event()
    calls = []
    terminals = []

    def fake(crops, path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("first checkpoint refused")
        started.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release its owned worker")
        return _summary(path)

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    widget = em.EmbeddingsScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    retry_path = str(tmp_path / "retry.pt")
    widget._dino_jobs.job_failed.connect(lambda message: widget._pretrain_dino(retry_path))
    widget._dino_jobs.job_finished.connect(terminals.append)
    try:
        widget._pretrain_dino(str(tmp_path / "first.pt"))
        qtbot.waitUntil(lambda: started.is_set() and False in terminals, timeout=5000)
        assert widget._dino_jobs.pending_jobs() == 1
        assert widget._dino_running and not widget._dino.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not widget.is_busy(), timeout=5000)
    assert widget.spec().backbone == emb._DINO_PREFIX + retry_path
