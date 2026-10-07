"""Retained cancellation logs must release completed pipeline workers."""

from __future__ import annotations

import logging
import weakref

import pytest

pytest.importorskip("PySide6")

from spacr.cancellation import PipelineCancelled
from spacr.qt.bridge import PipelineWorker


def test_cancelled_pipeline_log_releases_its_exception_frame(caplog):
    """A retained INFO record must not hold the worker's traceback."""
    def cancel(_settings):
        raise PipelineCancelled("requested stop")

    worker = PipelineWorker(
        cancel, {}, app_key="retention-probe", journal=False,
        capture_figures=False,
    )
    reference = weakref.ref(worker)
    with caplog.at_level(logging.INFO, logger="spacr.qt.bridge"):
        worker.run()

    assert worker.was_cancelled
    assert any(
        record.name == "spacr.qt.bridge"
        and "requested stop" in record.getMessage()
        for record in caplog.records
    )
    del worker
    assert reference() is None
