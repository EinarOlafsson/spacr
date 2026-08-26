"""``A18`` — the napari bridge, in the screen that folded it in.

THE BRIDGE HAS NO MODULE OF ITS OWN. It is a button on the Make Masks
masthead and has no tile, so nothing but Make Masks ever built its screen;
a file beside every real screen under ``spacr/qt/screens/`` was a front door
onto nothing. :class:`spacr.qt.screens.make_masks.NapariBridgeScreen` is
that screen, in the one module that opens it.

:mod:`spacr.napari_bridge` -- the engine -- is a DIFFERENT file and stays
where it is. It is public API, ``spacr.napari_bridge.correct_mask`` is the
headless route the CLI names, and the label-mask drop handler routes through
it. The fold took the screen in, not the library, and the first test here
says so.

Everything about *fidelity* — what survives the round trip, what is refused —
is asserted against the engine in ``tests/test_napari_bridge.py``. This file
asserts what is the screen's own responsibility:

* it **never imports napari at module scope, and neither does its host**, so
  a settings panel does not drag a second Qt stack into a spaCR session that
  will never use it;
* a missing install produces the ``pip install "spacr[napari]"`` paragraph in
  the status pane, not a traceback and not a modal;
* the buttons do the round trip through :mod:`spacr.napari_bridge` and record
  the correction, rather than reimplementing any of it.

The viewer is a stand-in, injected through the same seam the real one comes
back on. That is not a weaker test than one with napari installed: what could
break here is the wiring, and the wiring is exactly what a fake viewer
exercises.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import napari_bridge as nb
from spacr.curation import CurationLog, is_curated
from spacr.mask_io import load_mask, save_mask
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import NapariBridgeScreen

pytestmark = pytest.mark.qt


class FakeLayer:
    """A napari layer: an array, a name and its own ``_type_string``."""

    def __init__(self, data, name, kind):
        self.data = np.asarray(data)
        self.name = str(name)
        self._type_string = str(kind)


class FakeViewer:
    """A napari viewer, painted in place the way napari's brush paints."""

    def __init__(self):
        self.layers = []
        self.closed = False

    def add_image(self, **kwargs):
        return self._add(kwargs, "image")

    def add_labels(self, **kwargs):
        return self._add(kwargs, "labels")

    def _add(self, kwargs, kind):
        layer = FakeLayer(kwargs["data"], kwargs["name"], kind)
        self.layers.append(layer)
        return layer

    def close(self):
        self.closed = True


def _field(tmp_path):
    """A 7x11 mask (not square) and its image, on disk."""
    mask = np.zeros((7, 11), dtype=np.uint16)
    mask[1:3, 2:5] = 41
    mask[5, 8:10] = 900
    mask_path = str(tmp_path / "plate1_A01_f1_mask.tif")
    save_mask(mask_path, mask)
    image_path = str(tmp_path / "plate1_A01_f1.tif")
    save_mask(image_path, np.arange(mask.size, dtype=np.uint16)
              .reshape(mask.shape))
    return mask_path, image_path


@pytest.fixture
def screen(qtbot):
    widget = NapariBridgeScreen()
    qtbot.addWidget(widget)
    return widget


def _open_with(screen, monkeypatch, viewer):
    """Open the field, with ``viewer`` standing in for a napari window."""
    from spacr import napari_bridge as engine

    real = engine.open_in_napari
    monkeypatch.setattr(
        engine, "open_in_napari",
        lambda handoff, **kwargs: real(handoff, viewer=viewer))
    return screen.open_in_napari()


# ---------------------------------------------------------------------------
# The fold itself
# ---------------------------------------------------------------------------

def test_the_screen_has_no_module_of_its_own_and_the_engine_still_does():
    """Two files carried this name; exactly one of them went.

    Deleting the wrong one would take a documented headless entry point
    (``spacr.napari_bridge.correct_mask``, named in ``spacr.cli``) and the
    label-mask drop handler with it, so the two halves are asserted
    separately rather than as one import.
    """
    import importlib

    import spacr

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("spacr.qt.screens.napari_bridge")

    engine = importlib.import_module("spacr.napari_bridge")
    assert spacr.napari_bridge is engine
    assert callable(engine.correct_mask)


def test_the_bridge_is_a_button_on_make_masks_and_not_a_tile():
    """A key in both tables is a module with two front doors.

    Asserted against the LAUNCHED registry -- ``register_self_registering
    _modules`` is what the window runs -- because the row this screen used to
    add was added there and nowhere else, so a check against the bare table
    would pass while the tile was still on Home.
    """
    import spacr.qt
    from spacr.qt import app as app_module

    spacr.qt.register_self_registering_modules()

    assert "napari_bridge" in mm.FOLD_ORDER
    assert "napari_bridge" not in {row[0] for row in app_module.APPS}
    # With no row, the fallback is the only thing left to name the button.
    assert mm.fold_description("napari_bridge")[0] == "Napari Bridge"


def test_the_masthead_button_builds_the_folded_bridge(qtbot,
                                                      qt_theme_applied):
    """Pressed through the host, which is now the only way in."""
    host = mm.MakeMasksScreen()
    qtbot.addWidget(host)

    built = host.folded_screen("napari_bridge")

    assert isinstance(built, NapariBridgeScreen)
    # Built once and kept: a second copy would answer a hand-off the first
    # one was given.
    assert host.folded_screen("napari_bridge") is built


# ---------------------------------------------------------------------------
# The lazy import, which is the reason the screen is shaped like this
# ---------------------------------------------------------------------------

def test_the_host_imports_neither_napari_nor_a_second_qt_stack():
    """The invariant this screen exists inside, now paid by its host.

    Importing the module that holds the bridge must cost PySide6 and nothing
    else Qt-shaped: no napari, and none of the bindings or toolkits napari
    brings with it. Asserted as an absence in a fresh interpreter, because
    ``'napari' in sys.modules`` is a fact and a timing is a coin flip.
    """
    code = (
        "import os; os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen');"
        "import sys, spacr.qt.screens.make_masks;"
        "bad=[m for m in ('napari','qtpy','PyQt5','PyQt6','PySide2','vispy',"
        "'magicgui','superqt','torch','spacr.utils') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout


def test_no_import_of_napari_sits_outside_a_function_in_the_host():
    """Checked against the source, not against a run that took one path.

    The engine module is on the list too: it is cheap, but importing it at
    module scope would put ``spacr.napari_bridge`` in the import graph of
    every settings panel in spaCR, and the next thing added to it would be
    paid for by all of them.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mm))
    module_scope = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_scope.append((node.module or "").split(".")[0])
    assert "napari" not in module_scope
    assert "napari_bridge" not in module_scope


@pytest.mark.skipif(nb.napari_available(),
                    reason="napari is installed in this environment")
def test_a_missing_install_is_an_instruction_in_the_pane_not_a_traceback(
        screen, tmp_path):
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    assert screen.open_in_napari() is None
    text = screen.status.toPlainText()
    assert 'pip install "spacr[napari]"' in text
    assert "Traceback" not in text
    # And it still says what to do instead, because most people do not need it.
    assert "Curate" in text
    assert screen.take_button.isEnabled() is False


# ---------------------------------------------------------------------------
# The round trip through the buttons
# ---------------------------------------------------------------------------

def test_opening_hands_the_field_over_and_arms_the_take_back_button(
        screen, monkeypatch, tmp_path):
    mask_path, image_path = _field(tmp_path)
    screen.set_paths(mask=mask_path, image=image_path)
    viewer = FakeViewer()
    assert _open_with(screen, monkeypatch, viewer) is viewer
    assert [layer.name for layer in viewer.layers] == ["image", "mask"]
    assert viewer.layers[1].data.shape == (7, 11)
    assert screen.take_button.isEnabled() and screen.close_button.isEnabled()
    assert "3 object(s)" not in screen.status.toPlainText()
    assert "2 object(s)" in screen.status.toPlainText()
    assert "nothing is written until you do" in screen.status.toPlainText()


def test_taking_the_mask_back_writes_it_and_records_the_correction(
        screen, monkeypatch, tmp_path):
    """The screen's whole job, end to end."""
    mask_path, image_path = _field(tmp_path)
    screen.set_paths(mask=mask_path, image=image_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)

    viewer.layers[1].data[3, 2] = 41           # the hand edit, in "napari"
    result = screen.take_mask_back()

    assert result is not None and result.written is True
    assert result.changed_pixels == 1 and result.altered == (41,)
    assert load_mask(mask_path)[3, 2] == 41
    assert load_mask(mask_path)[5, 8] == 900   # nothing else moved
    assert is_curated(mask_path) is True
    assert CurationLog.read_beside(mask_path).edits[0].kind == nb.EDIT_KIND
    assert "1 pixel(s)" in screen.status.toPlainText()


def test_the_corrected_signal_carries_the_path_that_was_written(
        qtbot, screen, monkeypatch, tmp_path):
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)
    viewer.layers[0].data[3, 2] = 41
    with qtbot.waitSignal(screen.corrected, timeout=2000) as caught:
        screen.take_mask_back()
    assert caught.args == [mask_path]


def test_pressing_take_back_twice_does_not_record_the_edit_twice(
        screen, monkeypatch, tmp_path):
    """The second press has nothing to take, and says so.

    Without refreshing the handoff after a write, the second press would diff
    the new mask against the original again and append an identical entry —
    a ledger that says an edit happened twice when it happened once.
    """
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)
    viewer.layers[0].data[3, 2] = 41

    first = screen.take_mask_back()
    second = screen.take_mask_back()

    assert first.written is True and second.written is False
    assert len(CurationLog.read_beside(mask_path)) == 1
    assert "unchanged" in screen.status.toPlainText()


def test_a_refusal_from_the_engine_is_shown_verbatim(
        screen, monkeypatch, tmp_path):
    """The engine's refusals are written to be read; they are not replaced."""
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)
    # A label that cannot survive the cast back to uint16.
    viewer.layers[0].data = viewer.layers[0].data.astype(np.int64)
    viewer.layers[0].data[0, 0] = 70000

    assert screen.take_mask_back() is None
    text = screen.status.toPlainText()
    assert "70000" in text and "4464" in text
    # Nothing was written, and nothing was recorded.
    assert is_curated(mask_path) is False


def test_taking_back_before_opening_says_so(screen):
    assert screen.take_mask_back() is None
    assert "Open a field in napari first" in screen.status.toPlainText()


def test_opening_without_a_mask_says_so(screen):
    assert screen.open_in_napari() is None
    assert "Choose a mask file first" in screen.status.toPlainText()


def test_opening_something_unreadable_says_so_rather_than_raising(
        screen, tmp_path):
    broken = tmp_path / "not_a_mask.tif"
    broken.write_bytes(b"this is not a tiff")
    screen.set_paths(mask=str(broken))
    assert screen.open_in_napari() is None
    assert "Could not read" in screen.status.toPlainText()


# ---------------------------------------------------------------------------
# Describing what is there
# ---------------------------------------------------------------------------

def test_the_screen_says_what_is_in_the_mask_and_whether_it_was_edited(
        screen, tmp_path):
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    assert "2 object(s)" in screen.describe_mask()
    assert "already curated" not in screen.describe_mask()

    log = CurationLog(mask_path, source="spacr-qt curation")
    log.append("paint", 41, n_changed=5)
    log.write_beside(mask_path)
    assert "already curated" in screen.describe_mask()


def test_describing_nothing_asks_for_a_mask(screen):
    assert "Choose a mask file first" in screen.describe_mask()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_closing_the_viewer_disarms_the_buttons(screen, monkeypatch, tmp_path):
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)
    screen.close_viewer()
    assert viewer.closed is True
    assert screen.viewer() is None
    assert screen.take_button.isEnabled() is False


def test_a_viewer_that_will_not_close_does_not_take_the_screen_with_it(
        screen, monkeypatch, tmp_path):
    class Stubborn(FakeViewer):
        def close(self):
            raise RuntimeError("the window manager said no")

    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    _open_with(screen, monkeypatch, Stubborn())
    screen.close_viewer()
    assert screen.viewer() is None


def test_closing_the_screen_closes_the_viewer(screen, monkeypatch, tmp_path):
    mask_path, _ = _field(tmp_path)
    screen.set_paths(mask=mask_path)
    viewer = FakeViewer()
    _open_with(screen, monkeypatch, viewer)
    screen.close()
    assert viewer.closed is True


def test_a_dropped_label_mask_still_lands_on_the_folded_screen(screen,
                                                               tmp_path):
    """The drop handler is keyed on the app key, which the fold kept.

    Dropping a mask anywhere on the bridge is how a user gets a field into it
    without the file dialog, and the handler is registered against
    ``napari_bridge`` in ``spacr.qt.dnd_handlers`` -- a table the fold does
    not touch, which is exactly why it is asserted here.
    """
    from pathlib import Path

    from spacr.qt import dnd_handlers as dh

    mask_path, _ = _field(tmp_path)
    handler = dh.get_handler("napari_bridge")
    assert handler is not None

    handler.apply(Path(mask_path), screen)

    assert screen.mask_path() == mask_path
