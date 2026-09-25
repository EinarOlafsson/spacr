"""Item 513 part 3: the plaque preview shows Mask generation's four views.

    "there should also be the ability to show the plaques as outlines ,
     masks, flows, or cell probabilities, just like in the live preview for
     mask generation." -- the maintainer, 2026-09-24

The plaque preview's image pane is the shared
:class:`~spacr.qt.widgets.segmentation_views.SegmentationViews` (item 505)
with the plaque view as its canvas: the same view names in the same order,
the same colours for masks and probability, the same sentence for a run
without a probability map. Each view is checked against the array the
faked segmentation handed back, drawn by the shared renderers, so a second
drawing path would show up as a mismatch.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402
from spacr.qt.widgets import segmentation_views as sv  # noqa: E402

GREY = 40


def _flat_png(path, size=(60, 60)):
    from PIL import Image

    Image.fromarray(np.full(size + (3,), GREY, dtype=np.uint8)).save(path)
    return path


def _labels(shape=(60, 60)):
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[35:50, 35:55] = 2
    return labels


def _flows(shape=(60, 60)):
    flow = np.zeros(shape + (3,), dtype=np.uint8)
    flow[..., 1] = 180
    flow[20:40, :, 2] = 90
    prob = np.full(shape, -4.0, dtype=np.float32)
    prob[10:30, 10:30] = 5.0
    return {"flow_rgb": flow, "cellprob": prob}


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def ran(panel, tmp_path):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(segment=lambda path: (_labels(), _flows()))
    return panel


def _show(panel, view):
    """Switch to ``view``; the canvas's pixels, or the sentence shown."""
    panel.views().set_view(view)
    if panel.views().is_showing_message():
        return panel.views().message_text()
    return panel._view.array()


def test_the_views_are_mask_generations_views(panel):
    from spacr.qt.widgets.live_preview import VIEW_MODES

    selector = panel._view_selector
    offered = [selector.itemData(i) for i in range(selector.count())]
    assert offered == list(VIEW_MODES) == list(sv.VIEWS)
    assert isinstance(panel.views(), sv.SegmentationViews)
    assert panel.views().canvas is panel._view, "one canvas, one ruler"
    assert ppv.render_cellprob is sv.render_cellprob, "no second renderer"


def test_each_view_shows_its_own_array_of_the_run(ran):
    labels, flows = _labels(), _flows()
    overlay = _show(ran, sv.OVERLAY)
    expected = ppv.render_overlay(
        np.full((60, 60, 3), GREY, dtype=np.uint8), labels, ppv.OverlayStyle())
    assert np.array_equal(overlay, expected)
    assert tuple(overlay[10, 20]) == ppv.OUTLINE_COLOUR
    assert np.array_equal(_show(ran, sv.MASKS), sv.render_labels(labels))
    assert np.array_equal(_show(ran, sv.FLOWS), flows["flow_rgb"])
    prob = _show(ran, sv.CELLPROB)
    assert np.array_equal(prob, sv.render_cellprob(flows["cellprob"]))
    assert prob[20, 20].sum() > prob[0, 0].sum(), "probable is brighter"


def test_a_run_without_a_probability_map_says_so(panel, tmp_path):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    assert _show(panel, sv.CELLPROB) == sv.WAITING
    assert panel.run_preview(segment=lambda path: _labels())
    assert _show(panel, sv.CELLPROB) == sv.NO_CELLPROB
    assert _show(panel, sv.FLOWS) == sv.NO_FLOWS
    assert panel.save_picture(str(tmp_path / "none.png")) is None
    assert "no picture to save" in panel.preview_status()


def test_a_run_that_found_nothing_says_so_on_the_masks_view(panel, tmp_path):
    _flat_png(tmp_path / "a.png")
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(
        segment=lambda path: np.zeros((60, 60), np.int32))
    assert _show(panel, sv.MASKS) == ppv.NO_PLAQUES


def test_right_click_saves_the_view_shown(ran, tmp_path, monkeypatch):
    menus = []
    monkeypatch.setattr(ppv.PlaquePreviewPanel, "_exec_menu",
                        staticmethod(lambda menu, pos: menus.append(menu)))
    offered = []

    def ask(parent, title, name, filters):
        offered.append(name)
        return str(tmp_path / "saved.png"), ""

    monkeypatch.setattr(ppv.QFileDialog, "getSaveFileName", ask)
    ran.views().set_view(sv.CELLPROB)
    ran._view.context_requested.emit(QPoint(4, 4))
    actions = [a for a in menus[0].actions() if a.text()]
    assert [a.text() for a in actions] == ["Save picture…"], (
        "no outline options on a view without outlines")
    actions[0].trigger()
    assert offered == ["plaque_cell_probability.png"]
    saved = QImage(str(tmp_path / "saved.png")).convertToFormat(
        QImage.Format_RGB888)
    assert (saved.width(), saved.height()) == (60, 60)
    shown = sv.render_cellprob(_flows()["cellprob"])
    assert saved.pixelColor(20, 20).getRgb()[:3] == tuple(shown[20, 20])
    ran.views().set_view(sv.OVERLAY)
    ran._view.context_requested.emit(QPoint(4, 4))
    assert "Outlines" in [a.text() for a in menus[1].actions()]


def test_line_weight_and_ruler_keep_working_across_the_views(ran):
    ran.open_overlay_settings()
    dialog = ran._overlay_dialog
    dialog.thickness.setValue(5)
    assert tuple(_show(ran, sv.OVERLAY)[12, 20]) == ppv.OUTLINE_COLOUR
    _show(ran, sv.MASKS)
    dialog.thickness.setValue(1)
    assert not ran.views().is_showing_message()
    assert np.array_equal(ran._view.array(), sv.render_labels(_labels())), (
        "a style change leaves the masks view alone")
    assert tuple(_show(ran, sv.OVERLAY)[12, 20]) == (GREY, GREY, GREY)
    dialog.reject()
    ruler = ran._view.ruler
    ran._ruler_btn.setChecked(True)
    ruler.start, ruler.end = (5.0, 5.0), (8.0, 9.0)
    assert ruler.length() == 5
    for view in (sv.FLOWS, sv.CELLPROB, sv.MASKS, sv.OVERLAY):
        _show(ran, view)
        assert ran._view.ruler is ruler and ruler.length() == 5, (
            "the same line on every view")


def test_figure_mode_keeps_the_four_views(panel):
    """Item 524 lifted Figure mode's Overlay-only rule."""
    panel.views().set_view(sv.FLOWS)
    panel.set_mode("figure")
    assert panel.views().view() == sv.FLOWS
    assert panel._view_selector.isVisibleTo(panel)
    panel.set_mode("plaque")
    assert panel._view_selector.isVisibleTo(panel)
