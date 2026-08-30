"""Make Masks where the collaborator is missing: no matplotlib, no flows, no app.

The canvas and the screen both do most of their work through something else --
matplotlib for the probability ramp, Cellpose for the flows, a ``QApplication``
for the wait cursor, a background thread for the image. Each of those can be
absent or can answer with less than the happy path expects, and this file
drives the arms that only run then: a recrop box with no caption, a pan too
small to re-anchor, a drag that started outside the image, a flows list with
only two members, a fold that cannot be hung as a page, and a stale background
load whose field the user has already moved off.

Every "nothing happened" assertion here is paired, in the same test, with the
input that makes something happen -- otherwise a screen that did nothing at all
would pass every one of them.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt      # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent        # noqa: E402
from PySide6.QtWidgets import (                              # noqa: E402
    QDialogButtonBox, QPushButton, QTabWidget, QWidget,
)

from spacr.qt import mask_engine as engine                   # noqa: E402
from spacr.qt.screens import make_masks as mm                # noqa: E402
from spacr.qt.screens.make_masks import (                    # noqa: E402
    MODE_RECROP,
    FoldedModulePanel,
    MakeMasksScreen,
    _MaskCanvas,
    cellprob_heatmap,
    cellpose_detect,
    cellpose_intermediates,
)
from spacr.qt.theme import DARK_PALETTE                      # noqa: E402
from tests.conftest import (MISSING_CHANNEL_AXIS,
                            check_cellpose_eval_call)

pytestmark = pytest.mark.qt

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2


def canvas_xy(img_x: float, img_y: float) -> tuple:
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton,
         modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def _colour_pixels(qimg, colour) -> int:
    image = qimg.convertToFormat(QImage.Format_RGB32)
    array = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
        image.height(), image.bytesPerLine() // 4)
    return int((array == np.uint32(QColor(colour).rgb())).sum())


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """A canvas at a known size over a black field, so text shows up."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(CANVAS_W, CANVAS_H)
    widget.set_image_and_mask(np.zeros((IMG_N, IMG_N), np.uint16),
                              np.zeros((IMG_N, IMG_N), np.uint8))
    assert widget.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return widget


@pytest.fixture
def field_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "fields"
    (folder / "masks").mkdir(parents=True)
    for index in range(2):
        field = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
        field[10 + index:40, 10:40] = 3000
        imageio.imwrite(folder / f"f_{index:02d}.tif", field)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, field_folder):
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(field_folder))
    if made._loading:
        made._load_worker.wait(30_000)
        made._on_background_load_finished()
    assert made._canvas.image is not None
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    yield made
    made.close_folded()


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------



class TestTheSweepDeleteLedger:

    def test_one_label_is_recorded_once_however_often_it_is_hit(self, canvas):
        """The sweep's ledger entry names the objects, not the clicks."""
        mask = np.zeros((IMG_N, IMG_N), np.uint8)
        mask[10:20, 10:20] = 7
        mask[30:40, 30:40] = 9
        canvas.set_image_and_mask(canvas.image, mask)
        canvas._sweep_labels = [7]

        assert canvas._sweep_delete_at((12, 12)) is True
        assert canvas._sweep_labels == [7]

        assert canvas._sweep_delete_at((32, 32)) is True
        assert canvas._sweep_labels == [7, 9]




class TestARecropDragThatStartedOffTheImage:

    def test_nothing_is_asked_for_when_neither_corner_is_on_the_field(
            self, canvas, qtbot):
        """The margin either side of the pixmap is not part of the field."""
        canvas.mode = MODE_RECROP
        asked = []
        canvas.recrop_requested.connect(
            lambda *box: asked.append(tuple(box)))

        canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, 4, 4))
        canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove, 40, 40,
                                   button=Qt.NoButton))
        canvas.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease, 40, 40,
                                      buttons=Qt.NoButton))

        assert asked == []
        assert canvas._zoom_drag_start is None

        # The same gesture inside the image does ask for a crop.
        canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress,
                                    *canvas_xy(8, 8)))
        canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove,
                                   *canvas_xy(30, 30), button=Qt.NoButton))
        canvas.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease,
                                      *canvas_xy(30, 30),
                                      buttons=Qt.NoButton))

        assert len(asked) == 1
        assert asked[0][0] < asked[0][2] and asked[0][1] < asked[0][3]


# ---------------------------------------------------------------------------
# Reading what Cellpose hands back
# ---------------------------------------------------------------------------

class TestTheProbabilityRamp:

    def test_without_matplotlib_the_map_is_a_grey_ramp_not_a_failure(
            self, monkeypatch):
        """The pane is never the thing that fails; it just loses its colour."""
        values = np.linspace(-6.0, 6.0, 64, dtype=np.float32).reshape(8, 8)

        coloured = cellprob_heatmap(values)
        assert coloured.shape == (8, 8, 3)
        assert not np.array_equal(coloured[..., 0], coloured[..., 2])

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        grey = cellprob_heatmap(values)

        assert grey.shape == (8, 8, 3)
        assert grey.dtype == np.uint8
        # Grey means the three channels agree, and it is still monotone in
        # the probability -- the ramp is what was lost, not the reading.
        assert np.array_equal(grey[..., 0], grey[..., 1])
        assert np.array_equal(grey[..., 1], grey[..., 2])
        assert grey[0, 0, 0] < grey[-1, -1, 0]


class TestReadingAShortFlowsList:

    def test_no_flows_at_all_is_two_missing_maps_not_a_crash(self):
        assert cellpose_intermediates(None) == (None, None)

    def test_a_two_member_list_has_no_probability_map(self):
        """``flows[2]`` is the cell probability; a wrapper may not send one."""
        rgb = np.zeros((6, 5, 3), dtype=np.uint8)
        rgb[..., 0] = 200
        vectors = np.zeros((2, 6, 5), dtype=np.float32)

        cellprob, picture = cellpose_intermediates([rgb, vectors])

        assert cellprob is None
        assert np.array_equal(picture, rgb)

        # With the third member present it IS read, so the None above is the
        # short list and not a function that never looks.
        probability = np.full((6, 5), 2.5, dtype=np.float32)
        cellprob, _picture = cellpose_intermediates(
            [rgb, vectors, probability])
        assert np.array_equal(cellprob, probability)

    def test_the_vectors_are_drawn_when_the_picture_is_not_a_picture(self):
        """``flows[0]`` is the entry most likely to be missing or wrong."""
        vectors = np.zeros((2, 8, 8), dtype=np.float32)
        vectors[0, :4] = 3.0
        vectors[1, :, :4] = -2.0

        _cellprob, picture = cellpose_intermediates([None, vectors])

        assert picture is not None
        assert picture.shape == (8, 8, 3)
        assert picture[..., 0].tolist() != picture[..., 1].tolist()


class TestAnEvalWhoseSignatureCannotBeRead:

    def test_every_setting_is_offered_when_the_arguments_cannot_be_listed(
            self):
        """Filtering against a signature nobody can read would silently drop
        every threshold the user chose."""
        seen = {}

        class _Opaque:
            # ``numpy.frombuffer`` is a C function inspect cannot describe,
            # which is what a compiled or wrapped ``eval`` looks like.
            eval = np.frombuffer

        class _Recording:
            def eval(self, batch, channel_axis=MISSING_CHANNEL_AXIS,
                     **kwargs):
                # NAMED, and read. `_segment_one_field` filters its kwargs
                # against `inspect.signature(model.eval).parameters`, so a
                # double that swallowed the axis into **kwargs would never
                # be handed it at all -- the call site passes
                # channel_axis=cellpose_channel_axis(field), and the double
                # has to be able to tell that from an omission.
                check_cellpose_eval_call(batch, channel_axis)
                seen["channel_axis"] = channel_axis
                seen.update(kwargs)
                labels = np.zeros((8, 8), dtype=np.uint16)
                labels[2:4, 2:4] = 1
                return ([labels], [[None, None, None]],
                        np.zeros(4, dtype=np.float32))

        with pytest.raises(Exception):
            # Proves the stand-in really has no readable signature.
            import inspect
            inspect.signature(_Opaque.eval)

        model = _Recording()
        model.eval = _Recording.eval.__get__(model)
        # Give the recording model the unreadable signature.
        opaque = _Recording()
        opaque.__dict__["eval"] = lambda batch, **kwargs: model.eval(
            batch, **kwargs)
        import inspect
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(opaque.eval).parameters.values())

        labels, _cellprob, _flow = cellpose_detect(
            np.zeros((8, 8), dtype=np.uint16), opaque,
            flow_threshold=0.7, cellprob_threshold=-1.5, min_size=11)

        assert int(labels.max()) == 1
        assert seen["flow_threshold"] == 0.7
        assert seen["cellprob_threshold"] == -1.5
        assert seen["min_size"] == 11


class TestLoadingTheModelThroughSpacrsOwnResolver:

    def test_the_resolved_checkpoint_is_what_cellpose_is_handed(self,
                                                                monkeypatch):
        """Going round ``_resolve_cellpose_pretrained`` would give this screen
        a different model from the run it is correcting."""
        pytest.importorskip("cellpose")
        from cellpose import models as cp_models

        from spacr import utils

        built = {}

        class _Model:
            def __init__(self, gpu=False, pretrained_model=None, device=None):
                built.update(gpu=gpu, pretrained_model=pretrained_model,
                             device=device)

        monkeypatch.setattr(utils, "_resolve_cellpose_pretrained",
                            lambda name: f"resolved::{name}")
        monkeypatch.setattr(cp_models, "CellposeModel", _Model)

        model = mm.load_cellpose_model("cyto2")

        assert isinstance(model, _Model)
        assert built["pretrained_model"] == "resolved::cyto2"
        import torch
        assert built["gpu"] is torch.cuda.is_available()
        assert built["device"].type in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# The folded panels
# ---------------------------------------------------------------------------

class TestAFoldedPanelBecomingAWindow:

    def test_the_close_button_is_added_once_and_shows_the_row(self, qtbot):
        panel = FoldedModulePanel("mask", QWidget(), "Mask Generation")
        qtbot.addWidget(panel)
        assert "Close" not in panel.actions

        panel.add_close_button()

        assert "Close" in panel.actions
        assert panel.buttons.isVisible() or not panel.isVisible()
        assert panel.buttons.button(QDialogButtonBox.Close) is not None
        first = panel.actions["Close"]

        panel.add_close_button()

        assert panel.actions["Close"] is first
        assert len([b for b in panel.buttons.buttons()
                    if b is first]) == 1

    def test_a_fold_with_no_body_to_page_it_becomes_a_window(self, screen,
                                                              monkeypatch):
        """The window is the last resort, and it is the shape that needs a
        Close button, because there is no tab to close it by."""
        from spacr.qt.screens import map_barcodes

        held = QWidget()
        monkeypatch.setattr(screen, "folded_screen", lambda key: held)
        monkeypatch.setattr(screen, "seed_folded", lambda key: {})
        monkeypatch.setattr(map_barcodes, "show_as_page",
                            lambda panel, host, title: None)
        windowed = []
        monkeypatch.setattr(
            map_barcodes, "show_as_window",
            lambda panel, host, title: windowed.append((panel, title)))

        key = mm.FOLD_ORDER[0]
        panel = screen.open_folded(key)

        assert panel is not None
        assert "Close" in panel.actions
        assert [item[0] for item in windowed] == [panel]


class TestRestatingAFoldButton:

    def test_the_button_takes_the_name_sentence_and_stage_of_its_tile(self):
        key = mm.FOLD_ORDER[0]
        name, description, stage = mm.fold_description(key)
        button = QPushButton()
        assert button.property("stage") != stage

        MakeMasksScreen._restate_fold_button(button, key)

        assert button.property("stage") == stage
        assert button.accessibleName() == name
        assert button.toolTip() == f"{name}\n{description}".strip()

    def test_there_is_nothing_to_restate_without_a_button(self):
        assert MakeMasksScreen._restate_fold_button(
            None, mm.FOLD_ORDER[0]) is None


class TestSeedingTheCellposeWorkbench:

    def test_a_workbench_with_no_tab_stack_is_still_pointed_at_the_half(
            self, screen, qtbot):
        """Training keeps its own path; only applying is given this folder."""
        applied = []
        train = QWidget()
        apply_screen = QWidget()
        apply_screen.apply_settings_dict = lambda values: applied.append(
            dict(values))
        flat = QWidget()
        qtbot.addWidget(flat)
        flat.train_screen = train
        flat.apply_screen = apply_screen
        assert flat.findChild(QTabWidget) is None

        assert screen._seed_cellpose(flat, "train_cellpose") == {}
        assert applied == []

        assert screen._seed_cellpose(flat, "apply_cellpose") == {
            "src": screen._folder}
        assert applied == [{"src": screen._folder}]

    def test_a_workbench_with_tabs_is_moved_to_the_half_that_was_asked_for(
            self, screen, qtbot):
        tabbed = QWidget()
        qtbot.addWidget(tabbed)
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(tabbed)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        train, apply_screen = QWidget(), QWidget()
        apply_screen.apply_settings_dict = lambda values: None
        tabs.addTab(train, "Train")
        tabs.addTab(apply_screen, "Apply")
        tabbed.train_screen = train
        tabbed.apply_screen = apply_screen

        screen._seed_cellpose(tabbed, "train_cellpose")

        assert tabs.currentWidget() is train


# ---------------------------------------------------------------------------
# The settings column
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# The Cellpose workbench on this screen
# ---------------------------------------------------------------------------

class TestSyncingTheModelList:

    def test_a_model_the_live_cellpose_reports_is_added_once(self, screen,
                                                              monkeypatch):
        """The combo is built from a fallback list until Cellpose is loaded."""
        from spacr import settings as spacr_settings

        existing = screen._cp_model.itemData(0)
        monkeypatch.setattr(spacr_settings, "cellpose_model_choices",
                            lambda: [existing, "a_new_checkpoint"])
        before = screen._cp_model.count()

        screen._sync_model_choices()

        assert screen._cp_model.count() == before + 1
        assert screen._cp_model.findData("a_new_checkpoint") >= 0

        screen._sync_model_choices()

        assert screen._cp_model.count() == before + 1


class TestTheIntermediatePanes:

    def test_a_run_with_neither_map_clears_both_panes(self, screen):
        """A pane still showing the last field's run is a picture of the
        wrong image with nothing on screen saying so."""
        probability = np.linspace(-6.0, 6.0, 64,
                                  dtype=np.float32).reshape(8, 8)
        flow = np.zeros((8, 8, 3), dtype=np.uint8)
        flow[..., 1] = 180

        screen._show_intermediates(probability, flow)
        assert screen._prob_pane.pixmap() is not None
        assert not screen._prob_pane.pixmap().isNull()
        assert not screen._flow_pane.pixmap().isNull()

        screen._show_intermediates(None, None)

        assert screen._prob_pane.pixmap().isNull()
        assert screen._flow_pane.pixmap().isNull()


class TestARunWhoseMasksCannotBeCombined:

    def test_the_mask_is_left_alone_and_the_failure_is_named(self, screen,
                                                              monkeypatch):
        """A mask somebody has been correcting for an hour is not something a
        failed combine may quietly replace."""
        labels = np.zeros((IMG_N, IMG_N), dtype=np.int32)
        labels[10:20, 10:20] = 1
        monkeypatch.setattr(
            mm, "cellpose_detect",
            lambda image, model, **kwargs: (labels, None, None))
        monkeypatch.setattr(screen, "_cellpose_model", lambda name: object())
        # No QApplication to put a wait cursor on: the headless half of the
        # same call, which must not skip the run or the restore.
        monkeypatch.setattr(mm, "QApplication",
                            types.SimpleNamespace(instance=lambda: None))

        def _boom(current, new, mode):
            raise ValueError("shapes do not match")

        monkeypatch.setattr(engine, "combine_masks", _boom)
        warned = []
        monkeypatch.setattr(screen, "_warn",
                            lambda title, text: warned.append((title, text)))
        before = screen._canvas.mask.copy()

        assert screen.run_cellpose() == 0

        assert warned == [("Cellpose-SAM detect failed",
                           "shapes do not match")]
        assert np.array_equal(screen._canvas.mask, before)
        assert screen._btn_cellpose.isEnabled()

    def test_the_toolbar_button_runs_the_same_thing(self, screen,
                                                     monkeypatch):
        runs = []
        monkeypatch.setattr(screen, "run_cellpose",
                            lambda: runs.append(1) or 0)

        screen._on_detect_cellpose()

        assert runs == [1]


# ---------------------------------------------------------------------------
# The background image loader
# ---------------------------------------------------------------------------

class _Worker:
    def __init__(self, token, *, error=None, result=None, filename="f_00.tif"):
        self.token = token
        self.error = error
        self.result = result
        self.filename = filename
        self.deleted = 0

    def deleteLater(self):
        self.deleted += 1


class TestAStaleBackgroundLoad:

    def test_a_result_for_a_field_the_user_left_is_dropped(self, screen):
        """The token is what says which field the bytes belong to."""
        applied = []
        screen._apply_loaded_pair = (
            lambda *args: applied.append(args))
        screen._handle_load_failure = lambda error: applied.append(error)
        image = np.zeros((IMG_N, IMG_N), np.uint16)
        mask = np.zeros((IMG_N, IMG_N), np.uint8)

        stale = _Worker(screen._load_token - 1, result=(image, mask))
        screen._load_worker = stale
        screen._loading = True
        screen._pending_load = None

        screen._on_background_load_finished()

        assert applied == []
        assert stale.deleted == 1
        assert screen._load_worker is None
        assert screen._loading is False

        # The current token's result IS applied, so the drop above is the
        # token and not a handler that never applies anything.
        current = _Worker(screen._load_token, result=(image, mask))
        screen._load_worker = current
        screen._on_background_load_finished()

        assert len(applied) == 1
        assert applied[0][0] == "f_00.tif"

    def test_a_worker_that_brought_back_neither_error_nor_image_is_dropped(
            self, screen):
        applied = []
        screen._apply_loaded_pair = lambda *args: applied.append(args)
        screen._handle_load_failure = lambda error: applied.append(error)
        empty = _Worker(screen._load_token)
        screen._load_worker = empty
        screen._pending_load = None

        screen._on_background_load_finished()

        assert applied == []
        assert empty.deleted == 1

        failed = _Worker(screen._load_token, error=OSError("unreadable"))
        screen._load_worker = failed
        screen._on_background_load_finished()

        assert [type(item) for item in applied] == [OSError]

    def test_there_is_no_worker_left_to_finish_twice(self, screen):
        screen._load_worker = None
        screen._loading = True

        screen._on_background_load_finished()

        assert screen._loading is True       # untouched: nothing to finish


# ---------------------------------------------------------------------------
# Retiring a recropped field
# ---------------------------------------------------------------------------

class TestArchivingAFieldWithNoMask:

    def test_a_field_that_never_had_a_mask_is_still_archived(self, screen,
                                                             monkeypatch):
        """The archive is the recovery; there is simply nothing to save first."""
        saved = []
        retired = []
        monkeypatch.setattr(engine, "save_mask",
                            lambda *args, **kwargs: saved.append(args))
        monkeypatch.setattr(
            engine, "retire_recropped_original",
            lambda folder, filename, children, boxes: retired.append(
                (filename, tuple(children), tuple(boxes))))
        screen._canvas.mask = None
        screen._recrop_children = ["f_00_crop_1.tif"]
        screen._canvas.recrop_boxes = [(1, 2, 3, 4, "crop_1")]
        filename = screen._image_files[screen._current_index]

        assert screen.finish_recrop() is True

        assert saved == []
        assert retired == [(filename, ("f_00_crop_1.tif",), ((1, 2, 3, 4),))]

    def test_a_mask_that_will_not_write_does_not_stop_the_archive(
            self, screen, monkeypatch, caplog):
        """The original still has to reach somewhere safe."""
        import logging

        def _boom(*args, **kwargs):
            raise OSError("read-only filesystem")

        retired = []
        monkeypatch.setattr(engine, "save_mask", _boom)
        monkeypatch.setattr(
            engine, "retire_recropped_original",
            lambda folder, filename, children, boxes: retired.append(filename))
        screen._canvas.mask = np.zeros((IMG_N, IMG_N), np.uint8)
        screen._recrop_children = ["f_00_crop_1.tif"]
        caplog.set_level(logging.WARNING, logger=mm.LOG.name)

        assert screen.finish_recrop() is True

        assert len(retired) == 1
        assert "read-only filesystem" in caplog.text
