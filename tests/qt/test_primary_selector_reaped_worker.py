"""Closing the primary source picker tolerates a reaped QThread wrapper."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread
from shiboken6 import delete, isValid

from spacr.qt.widgets.primary_mask_selector import PrimaryMaskSelector


def test_shutdown_accepts_a_native_deleted_worker(qtbot):
    """The drain helper's deleted-wrapper contract also applies to its caller."""
    selector = PrimaryMaskSelector()
    qtbot.addWidget(selector)
    worker = QThread(selector)
    selector._worker = worker
    delete(worker)
    assert not isValid(worker)

    selector.shutdown()

    assert selector._closed and selector._worker is None
    assert isValid(selector)
