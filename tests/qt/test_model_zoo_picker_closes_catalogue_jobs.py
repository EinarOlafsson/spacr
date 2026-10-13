"""Closing the picker abandons catalogue callbacks without waiting for HTTP."""
from __future__ import annotations

import threading
import time

import pytest

from spacr import model_zoo
from spacr import _segmentation_backends as backends
from spacr.qt.widgets.model_zoo_picker import ModelZooPicker


@pytest.mark.parametrize('action,shown', [('close', True), ('reject', True),
                                        ('accept', True), ('done', True),
                                        ('reject', False), ('accept', False),
                                        ('done', False)])
def test_every_dialog_exit_retires_catalogue_updates(
        qtbot, monkeypatch, action, shown):
    from spacr.qt.bridge import prune_parked_threads, thread_has_stopped

    started = threading.Event()
    release = threading.Event()
    refreshes = []

    def load_catalogue(**kwargs):
        started.set()
        release.wait(5)
        return []

    monkeypatch.setattr(model_zoo, 'shared_catalogue_is_stale', lambda: True)
    monkeypatch.setattr(model_zoo, 'shared_catalogue', load_catalogue)
    monkeypatch.setattr(model_zoo, 'bioimageio_entries', lambda **kwargs: [])
    monkeypatch.setattr(backends, '_probe_blockers', lambda: {})
    monkeypatch.setattr(ModelZooPicker, 'refresh', lambda self: refreshes.append(1))
    dialog = ModelZooPicker()
    qtbot.addWidget(dialog)
    if shown:
        dialog.show()
    runner = dialog._catalogue_job
    qtbot.waitUntil(started.is_set)
    thread = next(iter(runner._jobs.values()))[0]
    refreshes.clear()
    try:
        closing_started = time.monotonic()
        if action == 'done':
            dialog.done(0)
        else:
            getattr(dialog, action)()
        assert time.monotonic() - closing_started < .5, 'closing waited for HTTP'
        assert not runner.is_busy(), 'the hidden picker still owns a pending callback'
        assert runner.active_jobs() == 0
        assert not dialog._probe_timer.isActive()
        release.set()
        qtbot.waitUntil(lambda: thread_has_stopped(thread), timeout=3000)
        qtbot.wait(25)
        assert not refreshes, 'a completed request refreshed the closed picker'
    finally:
        release.set()
        runner.shutdown(timeout_ms=1000)
        qtbot.waitUntil(lambda: thread_has_stopped(thread), timeout=3000)
        prune_parked_threads()
