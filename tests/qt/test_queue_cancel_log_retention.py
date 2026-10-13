"""Cancellation logs must not retain a queue runner's exception frame."""

from __future__ import annotations

import logging
import weakref

import pytest

pytest.importorskip("PySide6")

from spacr.cancellation import PipelineCancelled
from spacr.qt import bridge
from spacr.qt.plate_queue import PlateQueue, QueueItem, Status
from spacr.qt.screens.queue import _QueueRunner


def test_cancelled_runner_log_releases_its_exception_frame(
        tmp_path, monkeypatch, caplog):
    """A retained INFO record must not keep the completed runner alive."""
    queue = PlateQueue(path=tmp_path / "queue.json")
    item = QueueItem.build("mask", {"src": "first"})
    queue.add(item)

    def cancel(_settings):
        raise PipelineCancelled("requested stop")

    monkeypatch.setattr(bridge, "resolve_pipeline_entry", lambda _key: cancel)
    runner = _QueueRunner(queue)
    reference = weakref.ref(runner)
    with caplog.at_level(logging.INFO, logger="spacr.qt.queue_screen"):
        runner.run()

    assert item.status == Status.QUEUED
    assert item.end_ts is None and not item.error
    assert any(
        record.name == "spacr.qt.queue_screen"
        and "requested stop" in record.getMessage()
        for record in caplog.records
    )
    del runner
    assert reference() is None
