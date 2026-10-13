"""Closing Annotate cannot native-delete a parked, parented model worker."""
from __future__ import annotations

import threading
from types import MethodType

import pytest

from spacr.qt import bridge
from spacr.qt.screens import annotate as module


def _worker(kind, screen):
    if kind == 'similar':
        return module._SimilarityWorker('/controlled/no-model.db', None, 'crop', parent=screen)
    if kind == 'retrain':
        return module._RetrainWorker('/controlled/no-model.db', 'annotate', {}, parent=screen)
    return module._SuggestWorker('/controlled/no-model.db', 'annotate', {}, parent=screen)


@pytest.mark.parametrize('kind', ['similar', 'retrain', 'suggest'])
def test_parented_timeout_keeps_worker_owned_and_rejects_late_publication(
        qtbot, qt_theme_applied, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    interrupted, delivered = [], []
    original_drain = module.drain_thread
    screen = module.AnnotateScreen()
    qtbot.addWidget(screen)
    worker = _worker(kind, screen)

    def blocked(self):
        entered.set()
        assert release.wait(10)
        interrupted.append(self.isInterruptionRequested())
        self.done.emit({'late': True})
        self.failed.emit('controlled late result')

    def record(_screen, *args):
        delivered.append(args)

    def immediate_timeout(thread, *, timeout_ms):
        assert timeout_ms == module.CLOSE_DRAIN_MS == 15000
        return original_drain(thread, timeout_ms=0)

    for name in ('done', 'failed', 'finished'):
        callback_name = '_on_' + kind + '_' + name
        monkeypatch.setattr(screen, callback_name, MethodType(record, screen))
        getattr(worker, name).connect(getattr(screen, callback_name))
    monkeypatch.setattr(type(worker), 'run', blocked)
    monkeypatch.setattr(module, 'drain_thread', immediate_timeout)
    setattr(screen, '_' + kind + '_worker', worker)
    worker.start()
    qtbot.waitUntil(entered.is_set)
    try:
        screen.close()
        assert worker.isRunning() and worker.parent() is None
        assert getattr(screen, '_' + kind + '_worker') is None
        assert bridge.parked_thread_count() >= 1 and screen._closing
        release.set()
        assert worker.wait(2000)
        qtbot.wait(10)
        assert interrupted == [True]
        assert delivered == []
        assert getattr(screen, '_' + kind + '_worker') is None
    finally:
        release.set()
        assert worker.wait(2000)
        bridge.prune_parked_threads()
        worker.deleteLater()


@pytest.mark.parametrize('kind', ['similar', 'retrain', 'suggest'])
def test_stopped_worker_keeps_native_parent_until_normal_retirement(
        qtbot, qt_theme_applied, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    screen = module.AnnotateScreen()
    qtbot.addWidget(screen)
    worker = _worker(kind, screen)

    def blocked(self):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(type(worker), 'run', blocked)
    setattr(screen, '_' + kind + '_worker', worker)
    worker.start()
    qtbot.waitUntil(entered.is_set)
    try:
        release.set()
        screen.close()
        assert not worker.isRunning() and worker.parent() is screen
        assert getattr(screen, '_' + kind + '_worker') is None
    finally:
        release.set()
        assert worker.wait(2000)
