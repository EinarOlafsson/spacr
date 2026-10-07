"""Closing the primary source picker tolerates a reaped QThread wrapper."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread
from shiboken6 import delete, isValid

from spacr.qt.secondary_masks import read_primary_source
from spacr.qt.widgets.primary_mask_selector import PrimaryMaskSelector
from spacr.tiff_io import write_tiff


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


def test_restored_standard_source_is_cleared_when_its_field_closes(
        qtbot, tmp_path):
    """A saved known-class source cannot follow the editor to another field."""
    path = tmp_path / "primary.tif"
    write_tiff(path, np.full((10, 10), 7, dtype=np.uint16))
    field = tmp_path / "field.tif"
    output = tmp_path / "masks" / "field.tif"
    source = read_primary_source(
        path, field, (10, 10), output, bound_image=field,
        primary_class="nucleus", secondary_class="cell",
    )
    selector = PrimaryMaskSelector()
    qtbot.addWidget(selector)
    selector.bind_field(field, (10, 10), output)
    selector.restore_source(source.provenance())
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert selector.snapshot.identity == source.identity
    assert selector.primary_class.currentData() == "nucleus"
    assert selector.secondary_class.currentData() == "cell"

    selector.clear_field()

    assert selector._field is None and selector.snapshot is None
    assert selector._worker is None and selector._pending is None
    selector.shutdown()
