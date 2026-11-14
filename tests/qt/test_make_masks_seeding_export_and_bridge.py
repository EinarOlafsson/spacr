"""Make Masks: seeding folded modules, the loader queue, vvvv export, uncertainty, napari.

The mask editor is a long-lived window a user leaves open while folders come
and go, so each of these paths answers with an exact value or sentence rather
than a traceback: what a folded module was seeded with, which load runs next
after a failure, where an export was written (or the warning that says why
not), whether an uncertainty run started, and what the napari bridge's status
pane says.

Offscreen and CPU-only: no model runs; the uncertainty snapshot and the
napari viewer are stubbed where a test needs them to fail.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog                      # noqa: E402

from spacr.mask_io import save_mask                             # noqa: E402
from spacr.qt.screens import make_masks as mm                   # noqa: E402
from spacr.qt.screens.make_masks import (                       # noqa: E402
    MakeMasksScreen, NapariBridgeScreen)

pytestmark = pytest.mark.qt

IMG_N = 64


def _field_image():
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[8:56, 8:56] = 20000
    return img


def _objects_mask():
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    mask[4:10, 4:10] = 3
    mask[20:40, 20:40] = 7
    mask[54:60, 44:54] = 9
    return mask


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Three identical fields."""
    root = tmp_path / "field"
    root.mkdir()
    for index in range(3):
        imageio.imwrite(root / f"img_{index:02d}.tif", _field_image())
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    """Make Masks open on ``folder``, the first field showing three objects."""
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                    and widget._load_worker is None)
    widget._canvas.set_image_and_mask(_field_image(), _objects_mask())
    yield widget
    widget._uncertainty_worker = None
    widget._uncertainty_request = None


@pytest.fixture
def warnings(screen, monkeypatch):
    """Every ``_warn`` the screen raises, as ``(title, text)``."""
    said = []
    monkeypatch.setattr(screen, "_warn",
                        lambda title, text: said.append((title, text)))
    return said


@pytest.fixture
def bridge(qtbot):
    """A bare napari bridge screen."""
    widget = NapariBridgeScreen()
    qtbot.addWidget(widget)
    return widget


def _mask_field(tmp_path):
    mask = np.zeros((7, 11), dtype=np.uint16)
    mask[1:3, 2:5] = 41
    mask_path = str(tmp_path / "plate1_A01_f1_mask.tif")
    save_mask(mask_path, mask)
    image_path = str(tmp_path / "plate1_A01_f1.tif")
    save_mask(image_path,
              np.arange(mask.size, dtype=np.uint16).reshape(mask.shape))
    return mask_path, image_path


def test_make_masks_seed_folded_hands_each_module_the_open_folder(
        subtests, screen, monkeypatch):
    """The zoo gets the folder, applying Cellpose gets ``src``, training nothing."""
    folder = screen._folder
    zoo = screen.folded_screen("model_zoo")
    asked = []
    monkeypatch.setattr(type(zoo), "set_fields_source",
                        lambda self, source: asked.append(source) or True)
    cases = (
        ("the model zoo", "model_zoo", {"folder": folder}),
        ("applying Cellpose", mm.MASK_FOLDER_KEY, {"src": folder}),
        ("training Cellpose keeps its own path", "train_cellpose", {}),
    )
    for case_id, key, seeded in cases:
        with subtests.test(msg=case_id):
            assert screen.seed_folded(key) == seeded
    assert asked == [folder]

    with subtests.test(msg="a module with no folded screen"):
        monkeypatch.setattr(type(screen), "folded_screen",
                            lambda self, key: None)
        assert screen.seed_folded("curate") == {}

    with subtests.test(msg="no folder open"):
        screen._folder = None
        assert screen.seed_folded("model_zoo") == {}
        assert asked == [folder]


def test_make_masks_background_load_finished_reports_then_drains_the_queue(
        subtests, screen, monkeypatch):
    """No worker is a no-op; a failed load is reported, then the pending one runs."""
    reported, started = [], []
    monkeypatch.setattr(type(screen), "_handle_load_failure",
                        lambda self, error: reported.append(error))
    monkeypatch.setattr(type(screen), "_start_background_load",
                        lambda self, *request: started.append(request))

    with subtests.test(msg="no worker left"):
        screen._load_worker = None
        assert screen._on_background_load_finished() is None
        assert reported == [] and started == []

    with subtests.test(msg="a failed load, then the next request"):
        error = OSError("the share went away mid-read")

        class _Failed:
            token = screen._load_token
            filename = "img_01.tif"
            result = None

            def deleteLater(self):
                pass

        _Failed.error = error
        screen._load_worker = _Failed()
        pending = (screen._folder, "img_02.tif", screen._load_token)
        screen._pending_load = pending
        screen._on_background_load_finished()
        assert reported == [error]
        assert started == [pending]
        assert screen._pending_load is None
        assert screen._load_worker is None


def test_make_masks_vvvv_export_writes_or_warns_and_writes_nothing(
        subtests, screen, warnings, monkeypatch, tmp_path):
    """The labels land beside a manifest; every refusal leaves the disk alone."""
    out = tmp_path / "vvvv"
    root = {"path": str(out)}
    monkeypatch.setattr(type(screen), "_vvvv_export_root",
                        lambda self: root["path"])

    with subtests.test(msg="export writes and says where"):
        written = screen._on_export_vvvv()
        assert written == str(out / "img_00")
        np.testing.assert_array_equal(
            imageio.imread(out / "img_00" / "img_00_labels.png"),
            _objects_mask())
        assert screen._status_label.text() == (
            f"Exported for vvvv → {out / 'img_00'}")
        assert warnings == []

    with subtests.test(msg="quiet leaves the status line alone"):
        screen._status_label.setText("Saved.")
        assert screen._on_export_vvvv(str(tmp_path / "quiet"),
                                      quiet=True) == str(
            tmp_path / "quiet" / "img_00")
        assert screen._status_label.text() == "Saved."

    with subtests.test(msg="no export folder"):
        root["path"] = ""
        screen._status_label.setText("before")
        assert screen._on_export_vvvv() is None
        assert screen._status_label.text() == "before"
        assert warnings == []

    with subtests.test(msg="a refused label image"):
        bad = np.zeros((2, IMG_N, IMG_N), dtype=np.uint16)
        assert screen._on_export_vvvv(str(tmp_path / "bad"),
                                      labels=bad) is None
        assert warnings == [("vvvv export failed",
                             "vvvv export needs a 2-D label image")]
        assert not (tmp_path / "bad" / "img_00").exists()

    with subtests.test(msg="blinded"):
        warnings.clear()
        screen._blind = {"codes": {}}
        try:
            assert screen._on_export_vvvv(str(tmp_path / "blind")) is None
        finally:
            screen._blind = None
        assert warnings == [(
            "vvvv export is off while blinded",
            "Unblind this session before exporting folders named after "
            "source images.")]
        assert not (tmp_path / "blind").exists()

    with subtests.test(msg="no field"):
        warnings.clear()
        screen._image_files = []
        assert screen._on_export_vvvv(str(tmp_path / "none")) is None
        assert warnings == []
        assert not (tmp_path / "none").exists()


def test_make_masks_start_uncertainty_runs_one_request_at_a_time(
        subtests, screen, warnings, monkeypatch):
    """Inline runs deliver at once; threaded runs share one worker; busy refuses."""
    built, submitted = [], []

    class _Worker:
        def __init__(self, work, done, name=None):
            built.append((work, done, name))

        def submit(self, request):
            submitted.append(request)

    monkeypatch.setattr(mm, "_NewestRequestWorker", _Worker)

    with subtests.test(msg="busy refuses a second"):
        screen._uncertainty_request = {"busy": True}
        assert screen._start_uncertainty({}, threaded=False) is False
        assert screen._uncertainty_request == {"busy": True}
        screen._uncertainty_request = None

    with subtests.test(msg="an inline error is delivered and warned"):
        def broken(request, models=None):
            raise RuntimeError("model failed")

        monkeypatch.setattr(mm, "_uncertainty_snapshot", broken)
        assert screen._start_uncertainty({"kind": "field"},
                                         threaded=False) is True
        assert warnings == [("Uncertainty failed", "model failed")]
        assert screen._uncertainty_request is None
        assert screen._btn_uncertainty.isEnabled()

    with subtests.test(msg="a threaded run is queued on a new worker"):
        request = {"kind": "field"}
        assert screen._start_uncertainty(request, threaded=True) is True
        assert submitted == [request]
        assert screen._uncertainty_request is request
        assert not screen._btn_uncertainty.isEnabled()
        (work, done, name), = built
        assert isinstance(work, partial)
        assert work.func is mm._uncertainty_snapshot
        assert work.keywords == {"models": screen._cp_loaded}
        assert done == screen._uncertainty_done
        assert name == "spacr-uncertainty"

    with subtests.test(msg="a second while the first runs is refused"):
        assert screen._start_uncertainty({"kind": "field"}, True) is False
        assert submitted == [request]

    with subtests.test(msg="the next run reuses the worker"):
        screen._uncertainty_request = None
        again = {"kind": "fields"}
        assert screen._start_uncertainty(again, threaded=True) is True
        assert submitted == [request, again]
        assert len(built) == 1


def test_napari_bridge_choosers_and_status_pane(subtests, bridge, tmp_path,
                                                monkeypatch):
    """Cancelled choosers change nothing; the pane appends or replaces exactly."""
    mask_path, image_path = _mask_field(tmp_path)
    answer = {"path": ""}
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (answer["path"], "")))

    with subtests.test(msg="cancelling the mask chooser"):
        bridge.set_paths(mask=mask_path)
        bridge.say("something already said")
        bridge._choose_mask()
        assert bridge.mask_path() == mask_path
        assert bridge.status.toPlainText() == "something already said"

    with subtests.test(msg="browsing to an image fills only that field"):
        answer["path"] = image_path
        bridge.say("about the mask")
        bridge._choose_image()
        assert bridge.image_path() == image_path
        assert bridge.mask_path() == mask_path
        assert bridge.status.toPlainText() == "about the mask"

    with subtests.test(msg="cancelling the image chooser"):
        answer["path"] = ""
        bridge.set_paths(image="/somewhere/plate1_A01_f1.tif")
        bridge._choose_image()
        assert bridge.image_path() == "/somewhere/plate1_A01_f1.tif"

    cases = (
        ("appending keeps what was said", "Open in napari.",
         "Nothing was written.", True,
         "Open in napari.\n\nNothing was written."),
        ("appending to an empty pane", "", "Choose a mask file first.", True,
         "Choose a mask file first."),
        ("replacing", "Open in napari.", 42, False, "42"),
    )
    for case_id, before, text, append, shown in cases:
        with subtests.test(msg=case_id):
            bridge.status.setPlainText(before)
            assert bridge.say(text, append=append) == shown
            assert bridge.status.toPlainText() == shown


def test_napari_bridge_refusals_name_the_file_and_leave_the_buttons_off(
        subtests, bridge, tmp_path, monkeypatch, caplog):
    """An unreadable mask and a napari that throws are reported in the pane."""
    from spacr import napari_bridge as engine

    with subtests.test(msg="no mask chosen"):
        assert bridge.describe_mask() == "Choose a mask file first."

    with subtests.test(msg="a mask that will not load"):
        broken = tmp_path / "broken" / "plate1_A01_f1_mask.tif"
        broken.parent.mkdir()
        broken.write_bytes(b"not a tiff at all")
        bridge.set_paths(mask=str(broken))
        said = bridge.describe_mask()
        assert said.startswith("Could not read plate1_A01_f1_mask.tif: ")
        assert bridge.status.toPlainText() == said

    with subtests.test(msg="a napari that throws on open"):
        mask_path, image_path = _mask_field(tmp_path)
        bridge.set_paths(mask=mask_path, image=image_path)

        def explode(handoff, **kwargs):
            raise RuntimeError("no OpenGL context")

        monkeypatch.setattr(engine, "open_in_napari", explode)
        with caplog.at_level("ERROR", logger="spacr.qt.make_masks"):
            assert bridge.open_in_napari() is None
        assert bridge.status.toPlainText() == (
            "napari could not open this field: no OpenGL context")
        assert bridge.take_button.isEnabled() is False
        assert bridge.close_button.isEnabled() is False
        assert bridge.viewer() is None
        assert [record.getMessage() for record in caplog.records
                if record.name == "spacr.qt.make_masks"] == [
            "could not open napari"]
