"""Make Masks' micro-SAM prompting, ROI menu and blind curation, as a user
drives them: real mouse and key events on the canvas, the ROI menu's file
dialogs answered, the Blind switch flipped and the window closed.

micro-SAM itself runs in its own environment and is never started here; a
stand-in client answers every prompt with a disk round its first point, or
its box, and a stand-in ``_PromptClient`` answers the readiness check.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, Signal  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

import spacr._segmentation_backends as SB  # noqa: E402
from spacr import mask_io  # noqa: E402
from spacr import run_journal as rj  # noqa: E402

pytestmark = pytest.mark.qt

IMG_N = 64


class _Client:
    """Answers a prompt with a disk round its first point, or its box; can
    instead fail, be cancelled, outline nothing or answer the wrong shape."""

    def __init__(self):
        self.prompts = []
        self.mode = "ok"

    def readiness(self):
        return True, "stand-in"

    def segment(self, key, image, points=(), labels=(), box=None, *,
                should_cancel=None, on_start=None, on_embed=None):
        self.prompts.append((list(points), list(labels), box))
        field = image() if callable(image) else image
        if self.mode == "error":
            raise RuntimeError("the model fell over")
        if self.mode == "cancelled":
            raise SB._BackendCancelled("cancelled")
        shape = (7, 7) if self.mode == "shape" else field.shape[:2]
        mask = np.zeros(shape, bool)
        if self.mode != "empty" and self.mode != "shape":
            if box is not None:
                y0, x0, y1, x1 = box
                mask[y0:y1, x0:x1] = True
            else:
                y, x = points[0]
                yy, xx = np.ogrid[:shape[0], :shape[1]]
                mask[(yy - y) ** 2 + (xx - x) ** 2 <= 16] = True
        return {"mask": mask, "seconds": 0.01, "score": 0.9,
                "embed_seconds": None, "device": "cpu", "model": "vit_b_lm",
                "versions": {"micro_sam": "1.8.14"}}


@pytest.fixture
def journal(tmp_path, monkeypatch):
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    return runs


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    folder = tmp_path / "field"
    folder.mkdir()
    image = np.zeros((IMG_N, IMG_N), np.uint16)
    image[20:40, 20:40] = 30000
    for i in range(3):
        imageio.imwrite(folder / f"img_0{i}.tif", image)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder, journal, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    widget.resize(1200, 800)
    widget.show()
    qtbot.waitExposed(widget)
    widget._canvas.resize(600, 400)
    widget._canvas.refresh()
    widget._prompter._client = _Client()
    return widget


def _at(screen, x, y) -> QPoint:
    return screen._canvas._image_to_canvas(x + 0.5, y + 0.5)


def _prompting(screen, monkeypatch):
    monkeypatch.setattr(type(screen), "_prompt_ready", lambda self: True)
    screen._btn_prompt.setChecked(True)
    assert screen._prompter.enabled
    return screen._prompter


def _pixel(picture, point):
    return picture.pixelColor(point.x(), point.y())


def test_mouse_and_keys_on_the_canvas_prompt_draw_and_accept_an_object(
        screen, qtbot, monkeypatch):
    canvas = screen._canvas
    prompter = _prompting(screen, monkeypatch)
    before = canvas.grab().toImage()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, _at(screen, 30, 30))
    qtbot.waitUntil(lambda: prompter.pending is not None, timeout=5000)
    assert prompter.points == [(30, 30, True)]
    assert screen._btn_prompt_discard.isEnabled()

    drawn = canvas.grab().toImage()
    assert _pixel(drawn, _at(screen, 30, 30)) != _pixel(before, _at(screen, 30, 30)), (
        "the point and the outline must be drawn over the field")

    QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, _at(screen, 21, 21))
    QTest.mouseMove(canvas, _at(screen, 38, 38))
    assert prompter._drag is not None
    dragging = canvas.grab().toImage()
    assert not dragging.isNull()
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, _at(screen, 38, 38))
    assert prompter.box == (21, 21, 39, 39)
    qtbot.waitUntil(lambda: prompter.pending is not None
                    and prompter.pending["box"] is not None, timeout=5000)
    boxed = canvas.grab().toImage()
    assert boxed.size() == drawn.size()

    override = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Escape, Qt.NoModifier)
    QApplication.sendEvent(canvas, override)
    assert override.isAccepted(), "Escape belongs to the prompt, not the zoom"

    QTest.keyClick(canvas, Qt.Key_Backspace)
    assert prompter.points == [] and prompter.box is not None
    qtbot.waitUntil(lambda: prompter.pending is not None
                    and not prompter.pending["points"], timeout=5000)

    mask_before = np.array(canvas.mask, copy=True)
    QTest.keyClick(canvas, Qt.Key_Return)
    added = np.unique(canvas.mask[canvas.mask != mask_before])
    assert added.size == 1 and canvas.mask[30, 30] == added[0]
    assert screen._log.edits[-1].kind == "prompt"
    assert screen._log.edits[-1].detail["box"] == [21, 21, 39, 39]
    assert prompter.pending is None

    QTest.keyClick(canvas, Qt.Key_Escape)
    assert not prompter.wants_key(Qt.Key_Escape)

    screen._btn_prompt.setChecked(False)
    assert not prompter.enabled
    assert "Prompting off" in screen._status_label.text()
    assert not prompter.key(Qt.Key_Return)
    assert not prompter.wants_key(Qt.Key_Return)


def test_backspace_takes_prompts_back_one_at_a_time_and_keys_it_cannot_use_pass(
        screen, qtbot, monkeypatch):
    prompter = _prompting(screen, monkeypatch)
    assert not prompter.key(Qt.Key_Return), "Enter with no outline does nothing"
    assert not prompter.wants_key(Qt.Key_Return)
    assert not prompter.key(Qt.Key_Escape)
    assert not prompter.key(Qt.Key_A)
    assert not prompter.wants_key(Qt.Key_A)
    assert not prompter.undo_prompt()

    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 22, 22)))
    assert prompter.move(QPointF(_at(screen, 36, 36)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 36, 36)))
    assert prompter.press(Qt.RightButton, QPointF(_at(screen, 5, 5)))
    assert prompter.release(Qt.RightButton, QPointF(_at(screen, 5, 5)))
    qtbot.waitUntil(lambda: prompter.pending is not None
                    and len(prompter.pending["points"]) == 1, timeout=5000)
    assert prompter.wants_key(Qt.Key_Return)

    assert prompter.key(Qt.Key_Backspace)
    assert prompter.points == [] and prompter.box is not None
    qtbot.waitUntil(lambda: not prompter.busy, timeout=5000)
    assert prompter.key(Qt.Key_Backspace)
    assert prompter.box is None and prompter.pending is None
    assert not prompter.busy
    assert not prompter.key(Qt.Key_Backspace)

    assert not prompter.press(Qt.MiddleButton, QPointF(_at(screen, 30, 30)))
    assert not prompter.move(QPointF(10, 10))
    assert not prompter.release(Qt.LeftButton, QPointF(10, 10))
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.RightButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(-50, -50)) is True
    assert prompter.points == [(30, 30, True)]


def test_off_image_clicks_and_drags_add_nothing(screen, monkeypatch):
    prompter = _prompting(screen, monkeypatch)
    outside = QPointF(-40, -40)
    assert prompter.press(Qt.LeftButton, outside)
    assert prompter.release(Qt.LeftButton, outside)
    assert prompter.points == [] and prompter.box is None

    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.move(QPointF(-40, -40))
    assert prompter.release(Qt.LeftButton, QPointF(-40, -40))
    assert prompter.box is None and prompter.points == []
    assert screen._prompter._client.prompts == []


@pytest.mark.parametrize("mode,said", (
    ("error", "micro-SAM could not segment: the model fell over"),
    ("cancelled", None),
    ("shape", None),
))
def test_a_prompt_that_fails_or_answers_the_wrong_field_leaves_no_outline(
        screen, qtbot, monkeypatch, mode, said):
    prompter = _prompting(screen, monkeypatch)
    prompter._client.mode = mode
    before = screen._masks_console.text()
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    qtbot.waitUntil(lambda: bool(prompter._client.prompts) and not prompter.busy,
                    timeout=5000)
    assert prompter.pending is None
    assert screen._accept_prompt() == []
    grown = screen._masks_console.text()[len(before):]
    if said:
        assert said in grown
    else:
        assert "could not segment" not in grown


def test_an_empty_outline_or_one_the_overlap_rule_refuses_adds_nothing(
        screen, qtbot, monkeypatch):
    prompter = _prompting(screen, monkeypatch)
    prompter._client.mode = "empty"
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    qtbot.waitUntil(lambda: prompter.pending is not None, timeout=5000)
    assert prompter._overlay is None
    assert screen._accept_prompt() == []
    assert "outlined nothing" in screen._status_label.text()

    screen._canvas.mask = np.full((IMG_N, IMG_N), 5, np.uint16)
    screen._prompt_overlap.setCurrentIndex(
        screen._prompt_overlap.findData("skip"))
    prompter._client.mode = "ok"
    prompter.forget()
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    qtbot.waitUntil(lambda: prompter.pending is not None, timeout=5000)
    assert screen._accept_prompt() == []
    assert "Nothing was added" in screen._status_label.text()
    assert np.all(screen._canvas.mask == 5)

    prompter.pending = dict(prompter.pending, mask=np.ones((3, 3), bool))
    assert screen._accept_prompt() == []

    generation = prompter.generation
    prompter.forget()
    from spacr.qt.screens.make_masks import _PromptRequest

    old = _PromptRequest(key=(), generation=generation, field="", source=None,
                         low=0.0, high=1.0, points=((30, 30, True),), box=None)
    prompter._take(old, {"mask": np.ones((IMG_N, IMG_N), bool)}, None)
    assert prompter.pending is None, "an answer to a discarded prompt is dropped"


def test_prompting_asks_the_backend_and_offers_to_install_it(
        screen, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    class _Ready:
        answer = (False, "missing")

        def readiness(self):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    monkeypatch.setattr(SB, "_PromptClient", _Ready)
    screen._prompter._client = None
    assert screen._prompt_ready() is False
    assert isinstance(screen._prompter.client(), _Ready)
    _Ready.answer = OSError("no environment folder")
    assert screen._prompt_ready() is False

    class _Dialog(QObject):
        job_started = Signal()
        job_progressed = Signal(str)
        job_failed = Signal(str)
        job_cancelled = Signal()

    def install(parent, name, *, watch, why):
        dialog = _Dialog()
        watch(dialog)
        dialog.job_started.emit()
        dialog.job_progressed.emit("resolving packages")
        dialog.job_failed.emit("first mirror down")
        dialog.job_cancelled.emit()
        _Ready.answer = (True, "installed")
        return True

    monkeypatch.setattr(model_zoo_picker, "install_backend", install)
    screen._canvas.ruler.set_active(True)
    screen._btn_prompt.setChecked(True)
    console = screen._masks_console.text()
    assert "Installing micro-SAM failed" in console
    assert "Cancelled. Nothing was left behind." in console
    assert "micro-SAM is installed" in console
    assert screen._prompter.enabled
    assert not screen._canvas.ruler.active

    screen._canvas.ruler.set_active(True)
    screen._btn_magnifier.setChecked(True)
    assert not screen._canvas.ruler.active
    assert not screen._btn_prompt.isChecked()


def _roi_mask() -> np.ndarray:
    mask = np.zeros((IMG_N, IMG_N), np.uint16)
    mask[4:14, 4:14] = 1
    mask[30:40, 30:40] = 2
    return mask


def _answer(monkeypatch, **answers):
    for name, value in answers.items():
        monkeypatch.setattr(QFileDialog, name,
                            staticmethod(lambda *a, _v=value, **k: _v))


def test_the_roi_menu_exports_through_its_file_dialogs(screen, tmp_path,
                                                       monkeypatch):
    screen._canvas.mask = _roi_mask()
    out = tmp_path / "out.geojson"
    _answer(monkeypatch, getSaveFileName=("", ""))
    screen._on_export_field_rois("geojson")
    assert not out.exists()

    _answer(monkeypatch, getSaveFileName=(str(out), ""))
    screen._on_export_field_rois("geojson")
    assert out.exists()
    assert "ROIs exported" in screen._status_label.text()
    back = mask_io.import_rois(str(out), (IMG_N, IMG_N), "geojson")
    assert np.array_equal(next(iter(back.values())), _roi_mask())

    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    _answer(monkeypatch, getSaveFileName=(str(blocker / "x.geojson"), ""))
    screen._on_export_field_rois("geojson")
    assert screen._status_label.text().startswith("Export failed")

    _answer(monkeypatch, getExistingDirectory="")
    screen._on_export_all_rois("geojson")
    folder = tmp_path / "all"
    _answer(monkeypatch, getExistingDirectory=str(folder))
    screen._on_export_all_rois("geojson")
    assert (folder / "img_00.geojson").exists()
    assert "ROI file(s) written" in screen._status_label.text()

    coco = tmp_path / "coco.json"
    _answer(monkeypatch, getSaveFileName=(str(coco), ""))
    screen._on_export_all_rois("coco")
    names = [i["file_name"] for i in json.loads(coco.read_text())["images"]]
    assert names == ["img_00.tif"]

    _answer(monkeypatch, getExistingDirectory=str(blocker))
    screen._on_export_all_rois("geojson")
    assert screen._status_label.text().startswith("Export failed")


def test_the_roi_menu_imports_through_its_file_dialogs(screen, tmp_path,
                                                       monkeypatch):
    from spacr.qt.screens import make_masks as mm

    rois = tmp_path / "rois"
    mask_io.export_rois(_roi_mask(), rois / "img_00.geojson",
                        object_type="cell")
    mask_io.export_rois(_roi_mask(), rois / "img_01.geojson",
                        object_type="cell")
    screen._canvas.mask = np.zeros((IMG_N, IMG_N), np.uint16)

    _answer(monkeypatch, getOpenFileName=("", ""))
    screen._on_import_field_rois()
    assert not screen._canvas.mask.any()

    _answer(monkeypatch, getOpenFileName=(str(rois / "img_00.geojson"), ""))
    screen._on_import_field_rois()
    assert "2 object(s) imported from img_00.geojson" in \
        screen._status_label.text()
    assert np.array_equal(screen._canvas.mask > 0, _roi_mask() > 0)

    broken = tmp_path / "broken.geojson"
    broken.write_text("{not json")
    _answer(monkeypatch, getOpenFileName=(str(broken), ""))
    screen._on_import_field_rois()
    assert screen._status_label.text().startswith("Import failed")

    empty = tmp_path / "empty.geojson"
    empty.write_text(json.dumps({"type": "FeatureCollection",
                                 "features": []}))
    _answer(monkeypatch, getOpenFileName=(str(empty), ""))
    screen._status_label.setText("unchanged")
    screen._on_import_field_rois()
    assert screen._status_label.text() == "unchanged"

    _answer(monkeypatch, getExistingDirectory="")
    screen._on_import_all_rois("geojson")
    _answer(monkeypatch, getExistingDirectory=str(rois))
    screen._on_import_all_rois("geojson")
    assert "cancelled" in screen._status_label.text(), (
        "with nobody to confirm, replacing every saved mask is refused")

    warned = []
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[1:])))
    screen._on_import_all_rois("geojson")
    assert screen._status_label.text() == "ROIs imported for 2 field(s)"
    assert mm.engine.load_image_and_mask(
        str(screen._folder), "img_01.tif",
        **screen._layout_kwargs())[1].max() > 0

    coco = tmp_path / "coco.json"
    mask_io.export_rois(_roi_mask(), coco, "coco", file_name="img_02.tif")
    _answer(monkeypatch, getOpenFileName=(str(coco), ""))
    screen._on_import_all_rois("coco")
    assert screen._status_label.text() == "ROIs imported for 1 field(s)"

    _answer(monkeypatch, getOpenFileName=(str(broken), ""))
    screen._on_import_all_rois("coco")
    assert warned and warned[-1][0] == "Import failed"


def test_a_file_of_several_object_types_asks_which_one_goes_on_the_field(
        screen, tmp_path, monkeypatch):
    from spacr.qt.screens import make_masks as mm

    both = tmp_path / "both.geojson"
    nuclei = np.zeros((IMG_N, IMG_N), np.uint16)
    nuclei[50:55, 50:55] = 1
    mask_io.export_rois({"nucleus": nuclei, "pathogen": _roi_mask()}, both)
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    asked = []

    def pick(parent, title, text, names, current, editable):
        asked.append(list(names))
        return names[1], len(asked) == 1

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(pick))
    screen._canvas.mask = np.zeros((IMG_N, IMG_N), np.uint16)
    assert screen.import_field_rois(str(both)) == 2
    assert asked and set(asked[0]) == {"nucleus", "pathogen"}
    assert screen._canvas.mask[35, 35] > 0 and screen._canvas.mask[52, 52] == 0
    assert screen.import_field_rois(str(both)) == -1, "a cancelled choice imports nothing"
    assert screen.import_field_rois(str(both), object_type="nucleus") == 1


def test_the_blind_switch_refuses_without_fields_and_asks_before_unblinding(
        qtbot, qt_theme_applied, journal, folder, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens import make_masks as mm

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    empty = mm.MakeMasksScreen()
    qtbot.addWidget(empty)
    empty._btn_blind.setChecked(True)
    assert not empty._btn_blind.isChecked() and empty._blind is None
    assert "Open a folder of images" in empty._status_label.text()
    assert empty._end_blind()

    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder))
    screen._btn_blind.setChecked(True)
    key = screen._blind["key_id"]
    screen._btn_blind.setChecked(False)
    assert screen._btn_blind.isChecked() and screen._blind is not None, (
        "with nobody to answer, the session stays blind")

    monkeypatch.setattr(mm, "is_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    screen._btn_blind.setChecked(False)
    assert screen._blind is None and not screen._btn_blind.isChecked()
    assert [e["event"] for e in rj._blinding_events(key)] == [
        "blinded", "unblinded"]
    assert "3 images" in screen._src_label.text()

    screen._btn_blind.setChecked(True)
    second = screen._blind["key_id"]
    other = tmp_path / "other"
    other.mkdir()
    imageio.imwrite(other / "b.tif", np.zeros((16, 16), np.uint16))
    assert screen._open_folder(str(other))
    assert screen._blind is None
    assert [e["event"] for e in rj._blinding_events(second)] == [
        "blinded", "closed"]

    screen._btn_blind.setChecked(True)
    third = screen._blind["key_id"]

    class _Download:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    download = _Download()
    screen._cp_download = download
    screen.close()
    assert download.cancelled and screen._blind is None
    events = rj._blinding_events(third)
    assert [e["event"] for e in events] == ["blinded", "closed"]
    assert events[-1].get("reason") == "the screen was closed"


def test_the_roi_menu_does_nothing_before_a_folder_is_open(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    asked = []
    for name in ("getSaveFileName", "getOpenFileName"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(
            lambda *a, _n=name, **k: asked.append(_n) or ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda *a, **k: asked.append("dir") or ""))
    blank = MakeMasksScreen()
    qtbot.addWidget(blank)
    blank._on_export_field_rois("geojson")
    blank._on_export_all_rois("coco")
    blank._on_import_field_rois()
    blank._on_import_all_rois("imagej")
    assert asked == [], "no file dialog opens with nothing to export or import"
    assert blank.export_field_rois("x.geojson") == ""
    assert blank.import_field_rois("x.geojson") == -1


def test_imports_that_find_nothing_or_ids_too_big_for_the_mask(
        screen, tmp_path):
    rois = tmp_path / "rois"
    rois.mkdir()
    empty = json.dumps({"type": "FeatureCollection", "features": []})
    (rois / "img_00.geojson").write_text(empty)
    (rois / "img_01.geojson").write_text(empty)
    screen._canvas.mask = np.zeros((IMG_N, IMG_N), np.uint16)
    assert screen.import_all_rois(str(rois), "geojson") == []

    big = np.zeros((IMG_N, IMG_N), np.uint16)
    big[10:20, 10:20] = 300
    path = mask_io.export_rois(big, tmp_path / "big.geojson",
                               object_type="cell")
    screen._canvas.mask = np.zeros((IMG_N, IMG_N), np.uint8)
    assert screen.import_field_rois(str(path)) == 1
    assert screen._canvas.mask.max() > 255, (
        "an id the mask's type cannot hold widens the mask, not wraps it")


def test_the_prompt_is_not_drawn_where_the_canvas_has_no_picture(
        screen, qtbot, monkeypatch):
    from PySide6.QtGui import QImage, QPainter

    prompter = _prompting(screen, monkeypatch)
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    qtbot.waitUntil(lambda: prompter.pending is not None, timeout=5000)
    prompter.box = (20, 20, 40, 40)
    screen._canvas.mask = None
    picture = QImage(200, 200, QImage.Format_ARGB32)
    picture.fill(0)
    painter = QPainter(picture)
    prompter.paint(painter)
    painter.end()
    assert not np.any(np.frombuffer(picture.constBits(), np.uint8)), (
        "nothing maps onto a canvas without a field, so nothing is drawn")


def test_a_prompt_on_a_canvas_with_no_pixels_sends_nothing(screen,
                                                           monkeypatch):
    prompter = _prompting(screen, monkeypatch)
    screen._canvas.image = None
    assert prompter.press(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter.release(Qt.LeftButton, QPointF(_at(screen, 30, 30)))
    assert prompter._client.prompts == [] and not prompter.busy


def test_the_magnifier_takes_the_mouse_it_is_under_and_waits_for_its_stroke(
        screen, monkeypatch):
    from PySide6.QtGui import QCursor

    hovered = []
    magnifier = screen._magnifier
    monkeypatch.setattr(magnifier, "hover", lambda where: hovered.append(where))
    QCursor.setPos(screen._canvas.mapToGlobal(_at(screen, 30, 30)))
    screen._canvas.setAttribute(Qt.WA_UnderMouse, True)
    screen._btn_magnifier.setChecked(True)
    assert len(hovered) == 1, "switching on under the mouse shows the lens"

    magnifier._stroke = object()
    magnifier._stroke_from = (QPointF(0, 0), magnifier._field)
    assert magnifier.press() is False
    assert "segmenting the last regions" in screen._status_label.text()
    assert magnifier.drag() is None
    assert magnifier.release() is False
    assert magnifier._blocked_stroke_press is False


def test_a_blinded_curation_session_and_many_folders_restore_their_names(
        qtbot, qt_theme_applied, journal, folder, tmp_path, monkeypatch):
    from spacr.curation_queue import build_queue
    from spacr.qt import preferences
    from spacr.qt.screens import make_masks as mm

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    (folder / "masks").mkdir()
    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen.open_queue(build_queue(folder, order="name"))
    screen._btn_blind.setChecked(True)
    assert "Blinded: 3 fields" in screen._src_label.text()
    screen._btn_blind.setChecked(False)
    assert "3 to curate this session" in screen._src_label.text()

    other = tmp_path / "second"
    other.mkdir()
    imageio.imwrite(other / "z.tif", np.zeros((16, 16), np.uint16))
    assert screen.open_paths([str(folder / "img_00.tif"),
                              str(other / "z.tif")])
    screen._btn_blind.setChecked(True)
    assert sorted(screen._field_folders) == sorted([str(folder), str(other)])
    screen._btn_blind.setChecked(False)
    assert screen._field_pairs() == [(str(folder), "img_00.tif"),
                                     (str(other), "z.tif")]


def test_closing_a_blinded_screen_whose_log_cannot_be_written_still_closes(
        screen, monkeypatch):
    screen._btn_blind.setChecked(True)
    assert screen._blind is not None

    def broken(*_a, **_k):
        raise OSError("journal is read-only")

    monkeypatch.setattr(rj, "_close_blinding", broken)
    screen.close()
    assert screen._blind is None and not screen.isVisible()
