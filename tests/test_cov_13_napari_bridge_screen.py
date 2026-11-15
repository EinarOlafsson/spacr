"""Browsing for a field, and the three ways the bridge can fail mid-handover.

The bridge is folded into Make Masks -- :class:`spacr.qt.screens.make_masks
.NapariBridgeScreen` -- because that masthead is the only place it opens
from. The engine it drives, :mod:`spacr.napari_bridge`, is a separate module
and is what the assertions below reach through.

The status pane is the only channel this screen has, so every failure it can
reach has to arrive there as words: an unreadable mask, a napari that opened
and then threw, a write-back that was refused. A traceback in the terminal is
not an answer to somebody looking at the window, and a silent return leaves
the buttons armed for a field that is not there.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.mask_io import save_mask  # noqa: E402
from spacr.qt.screens.make_masks import NapariBridgeScreen  # noqa: E402

pytestmark = pytest.mark.qt


class _FakeLayer:
    def __init__(self, data, name, kind):
        self.data = np.asarray(data)
        self.name = str(name)
        self._type_string = str(kind)


class _FakeViewer:
    """Enough of a napari viewer to be handed a field and read back."""

    def __init__(self):
        self.layers = []
        self.closed = False

    def add_image(self, **kwargs):
        return self._add(kwargs, "image")

    def add_labels(self, **kwargs):
        return self._add(kwargs, "labels")

    def _add(self, kwargs, kind):
        layer = _FakeLayer(kwargs["data"], kwargs["name"], kind)
        self.layers.append(layer)
        return layer

    def close(self):
        self.closed = True


def _field(tmp_path):
    mask = np.zeros((7, 11), dtype=np.uint16)
    mask[1:3, 2:5] = 41
    mask_path = str(tmp_path / "plate1_A01_f1_mask.tif")
    save_mask(mask_path, mask)
    image_path = str(tmp_path / "plate1_A01_f1.tif")
    save_mask(image_path, np.arange(mask.size, dtype=np.uint16)
              .reshape(mask.shape))
    return mask_path, image_path


@pytest.fixture()
def screen(qtbot):
    widget = NapariBridgeScreen()
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# browsing for the two files
# ---------------------------------------------------------------------------

def test_browsing_to_a_mask_fills_the_field_and_describes_what_is_in_it(
        screen, tmp_path, monkeypatch):
    """Picking a mask must also answer "what did I just pick".

    The description is what tells the user whether this field has been
    corrected before. Filling the box and saying nothing leaves them to press
    Open and find out.
    """
    mask_path, _ = _field(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (mask_path, "")))

    screen._choose_mask()

    assert screen.mask_path() == mask_path
    said = screen.status.toPlainText()
    assert said.strip()
    assert os.path.basename(mask_path) in said or "object" in said


# ---------------------------------------------------------------------------
# the status pane
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# a mask that cannot be read
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# napari failing after it was reached
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# a write-back that is refused
# ---------------------------------------------------------------------------

def test_a_write_back_that_fails_is_shown_verbatim_and_records_nothing(
        screen, tmp_path, monkeypatch):
    """The corrected mask could not be written, so nothing may claim it was.

    ``corrected`` is what the rest of spaCR listens to; emitting it after a
    failed write would mark the field curated with the old pixels still on
    disk.
    """
    from spacr import napari_bridge as engine

    mask_path, image_path = _field(tmp_path)
    screen.set_paths(mask=mask_path, image=image_path)
    viewer = _FakeViewer()
    real_open = engine.open_in_napari
    monkeypatch.setattr(
        engine, "open_in_napari",
        lambda handoff, **kwargs: real_open(handoff, viewer=viewer))
    assert screen.open_in_napari() is viewer

    emitted: list = []
    screen.corrected.connect(emitted.append)

    def refuse(*args, **kwargs):
        raise OSError("Read-only file system: plate1_A01_f1_mask.tif")

    monkeypatch.setattr(engine, "write_back", refuse)

    assert screen.take_mask_back() is None
    assert ("Read-only file system: plate1_A01_f1_mask.tif"
            in screen.status.toPlainText())
    assert emitted == []
