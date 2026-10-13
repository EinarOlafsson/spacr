"""The shared four-view widget: overlay, masks, flows and cell probability.

`spacr.qt.widgets.segmentation_views.SegmentationViews` is built for both
the Mask module's live preview (item 505) and the plaque preview (item 513).
These tests hold it to what either owner needs from it, with no owner.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint                            # noqa: E402
from PySide6.QtGui import QImage                             # noqa: E402

from spacr.qt.widgets import picture_export                  # noqa: E402
from spacr.qt.widgets import segmentation_views as SV        # noqa: E402


def _labels():
    labels = np.zeros((24, 24), np.int32)
    labels[2:8, 2:8] = 1
    labels[12:20, 12:20] = 2
    return labels


def _logits():
    return np.linspace(-4.0, 4.0, 24 * 24, dtype=np.float32).reshape(24, 24)


@pytest.fixture()
def views(qtbot):
    widget = SV.SegmentationViews()
    qtbot.addWidget(widget)
    return widget


def test_four_views_in_the_order_offered(views):
    assert views.views() == ("Overlay", "Masks", "Flows", "Cell probability")
    combo = views.make_selector()
    assert [combo.itemData(i) for i in range(combo.count())] == \
        list(SV.VIEWS)


def test_it_waits_for_a_run(views):
    assert views.is_showing_message()
    assert views.message_text() == SV.WAITING


def test_the_probability_is_drawn_on_a_fixed_scale(views):
    views.set_arrays(image=np.ones((24, 24)), labels=_labels(),
                     cellprob=_logits(), refresh=False)
    views.set_view(SV.CELLPROB)
    assert not views.is_showing_message()
    assert np.array_equal(views.rendered(), SV.render_cellprob(_logits()))
    zero = SV.render_cellprob(np.zeros((2, 2), np.float32))
    brighter = SV.render_cellprob(np.full((2, 2), 3.0, np.float32))
    assert int(brighter.sum()) > int(zero.sum()), "more probable is brighter"


def test_a_run_without_a_probability_says_so(views):
    views.set_arrays(image=np.ones((24, 24)), labels=_labels())
    views.set_view(SV.CELLPROB)
    assert views.message_text() == SV.NO_CELLPROB


def test_two_objects_give_the_higher_probability(views):
    low = np.full((4, 4), -2.0, np.float32)
    high = np.full((4, 4), 2.0, np.float32)
    prob = SV.cell_probability({"cell": low, "nucleus": high})
    assert np.allclose(prob, 1.0 / (1.0 + np.exp(-2.0)))


def test_an_owner_renderer_replaces_one_view(views):
    views.set_renderer(SV.OVERLAY, lambda arrays: "drawn by the owner")
    views.set_arrays(image=np.ones((24, 24)), labels=_labels())
    assert views.message_text() == "drawn by the owner"
    views.set_renderer(SV.OVERLAY, None)
    views.refresh()
    assert not views.is_showing_message()


def test_every_view_draws_and_every_selector_follows(views):
    flows = np.zeros((24, 24, 3), np.uint8)
    flows[..., 2] = 180
    views.set_arrays(image=np.ones((24, 24)), labels=_labels(),
                     flows=flows, cellprob=_logits(), refresh=False)
    combo = views.make_selector()
    drawn = {}
    for view in SV.VIEWS:
        views.set_view(view)
        assert combo.currentData() == view
        drawn[view] = views.rendered().tobytes()
    assert len(set(drawn.values())) == 4


def test_the_canvas_saves_what_it_shows_from_a_right_click(views, tmp_path,
                                                           monkeypatch):
    views.set_arrays(image=np.ones((24, 24)), labels=_labels(),
                     cellprob=_logits(), refresh=False)
    views.set_view(SV.CELLPROB)
    asked = {}

    def _where(parent, stem):
        asked["stem"] = stem
        return str(tmp_path / f"{stem}.png")

    monkeypatch.setattr(picture_export, "choose_format",
                        lambda widget, point, enabled: ".png")
    monkeypatch.setattr(picture_export, "ask_where_to_save", _where)
    written = views.canvas._spacr_picture_menu(QPoint(3, 3))
    assert asked["stem"] == "cell_probability"
    assert (QImage(written).width(), QImage(written).height()) == (24, 24)


def test_an_owner_can_hand_in_its_own_canvas(qtbot):
    from spacr.qt.widgets.live_preview import _ZoomView

    canvas = _ZoomView()
    widget = SV.SegmentationViews(canvas=canvas)
    qtbot.addWidget(widget)
    widget.set_arrays(image=np.ones((24, 24)), labels=_labels())
    assert widget.canvas is canvas
    assert canvas.picture() is not None
    assert canvas.picture_name() == "overlay"
