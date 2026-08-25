"""Browsing for a field, and the three ways the bridge can fail mid-handover.

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
from spacr.qt.screens import napari_bridge as screen_module  # noqa: E402
from spacr.qt.screens.napari_bridge import NapariBridgeScreen  # noqa: E402

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


def test_cancelling_the_mask_chooser_changes_nothing(screen, tmp_path,
                                                     monkeypatch):
    """A cancelled dialog returns an empty path, which is not a selection.

    Writing it in would blank a path the user had already typed, and
    describing it would replace a real description with "choose a mask first".
    """
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    screen.say("something already said")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._choose_mask()

    assert screen.mask_path() == mask_path
    assert screen.status.toPlainText() == "something already said"


def test_browsing_to_an_image_fills_only_the_image_field(screen, tmp_path,
                                                         monkeypatch):
    """The image underneath is context, so picking one describes nothing.

    The description belongs to the mask; re-running it here would overwrite
    what the pane says about the mask with the same text, or with a refusal
    when no mask has been chosen yet.
    """
    _, image_path = _field(tmp_path)
    screen.say("about the mask")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (image_path, "")))

    screen._choose_image()

    assert screen.image_path() == image_path
    assert screen.status.toPlainText() == "about the mask"


def test_cancelling_the_image_chooser_changes_nothing(screen, monkeypatch):
    """Same rule for the second field."""
    screen.set_paths(image="/somewhere/plate1_A01_f1.tif")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._choose_image()

    assert screen.image_path() == "/somewhere/plate1_A01_f1.tif"


# ---------------------------------------------------------------------------
# the status pane
# ---------------------------------------------------------------------------

def test_appending_keeps_what_the_pane_already_said(screen):
    """A second message must not erase the first when the caller appends.

    The two things this pane says most are a multi-line refusal and a
    multi-line instruction, and the second is only actionable beside the
    first.
    """
    screen.say("Open in napari.")

    shown = screen.say("Nothing was written.", append=True)

    assert shown == "Open in napari.\n\nNothing was written."


def test_appending_to_an_empty_pane_does_not_leave_leading_blank_lines(screen):
    """With nothing to append to, an append is just a message."""
    shown = screen.say("Choose a mask file first.", append=True)

    assert shown == "Choose a mask file first."


# ---------------------------------------------------------------------------
# a mask that cannot be read
# ---------------------------------------------------------------------------

def test_describing_a_mask_that_will_not_load_names_the_file(screen, tmp_path):
    """The refusal has to name the file, because a run has thousands.

    ``describe_mask`` runs on every browse, so the message the user gets when
    they pick the wrong file is the only thing telling them which one it was.
    """
    broken = tmp_path / "plate1_A01_f1_mask.tif"
    broken.write_bytes(b"not a tiff at all")
    screen.set_paths(mask=str(broken))

    said = screen.describe_mask()

    assert "Could not read" in said
    assert "plate1_A01_f1_mask.tif" in said


# ---------------------------------------------------------------------------
# napari failing after it was reached
# ---------------------------------------------------------------------------

def test_a_napari_that_throws_on_open_is_reported_and_leaves_the_buttons_off(
        screen, tmp_path, monkeypatch, caplog):
    """A viewer that raises is not a missing install and not a viewer.

    Arming *Take the mask back* here would let the user press it against a
    handoff that was never opened. The failure is logged with its traceback
    for the report, and summarised in the pane for the person looking at it.
    """
    from spacr import napari_bridge as engine

    mask_path, image_path = _field(tmp_path)
    screen.set_paths(mask=mask_path, image=image_path)

    def explode(handoff, **kwargs):
        raise RuntimeError("no OpenGL context")

    monkeypatch.setattr(engine, "open_in_napari", explode)

    with caplog.at_level("ERROR", logger="spacr.qt.screens.napari_bridge"):
        assert screen.open_in_napari() is None

    said = screen.status.toPlainText()
    assert "napari could not open this field" in said
    assert "no OpenGL context" in said
    assert screen.take_button.isEnabled() is False
    assert screen.close_button.isEnabled() is False
    assert any("could not open napari" in record.message
               for record in caplog.records)


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
