"""The four segmentation views with the inputs a run does not always give.

``tests/qt/test_segmentation_views.py`` covers the views an ordinary run
draws. These cases are the rest: pictures with one or two channels, a flat
field, several objects of different shapes, a thicker outline, a colour map
that cannot be loaded, and an owner renderer that raises.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

from spacr.qt.widgets import segmentation_views as sv


def test_to_rgb8_takes_every_shape_a_run_hands_over():
    assert sv.to_rgb8(None) is None
    assert sv.to_rgb8(np.zeros((4,), np.uint8)) is None
    assert sv.to_rgb8(np.zeros((0, 3), np.uint8)) is None

    single = sv.to_rgb8(np.arange(4, dtype=np.uint8).reshape(2, 2, 1))
    assert single.shape == (2, 2, 3)
    assert (single[..., 0] == single[..., 2]).all()

    flat = sv.to_rgb8(np.full((2, 2), 7.5, np.float32))
    assert flat.dtype == np.uint8 and not flat.any()

    two = np.zeros((2, 2, 2), np.uint16)
    two[..., 0] = 1000
    padded = sv.to_rgb8(two)
    assert padded.shape == (2, 2, 3)
    assert (padded[..., 2] == 0).all()


def test_nothing_drawable_is_a_null_pixmap(qapp):
    assert sv.to_qpixmap(None).isNull()
    assert not sv.to_qpixmap(np.zeros((3, 3), np.uint8)).isNull()


def test_a_thicker_outline_grows_by_one_ring_per_step():
    labels = np.zeros((21, 21), np.int32)
    labels[3:18, 3:18] = 1
    thin = sv.render_overlay(np.zeros((21, 21), np.uint8), labels)
    thick = sv.render_overlay(np.zeros((21, 21), np.uint8), labels,
                              thickness=3)
    thin_pixels = int(thin.any(axis=-1).sum())
    thick_pixels = int(thick.any(axis=-1).sum())
    assert thick_pixels > thin_pixels
    assert thick[5, 5].any() and not thin[5, 5].any()
    assert thick[10, 10].tolist() == [0, 0, 0], "the middle stays unpainted"


def test_masks_of_another_shape_or_empty_are_left_out():
    first = np.zeros((4, 4), np.int32)
    first[1:3, 1:3] = 5
    picture = sv.render_labels({"cell": first,
                                "nucleus": np.zeros((4, 4), np.int32),
                                "pathogen": np.ones((6, 6), np.int32)})
    assert picture.shape == (4, 4, 3)
    assert picture[1, 1].any() and not picture[0, 0].any()


def test_flows_are_blended_only_where_the_shapes_agree():
    left = np.zeros((3, 3, 3), np.uint8)
    left[0, 0] = 200
    right = np.zeros((3, 3, 3), np.uint8)
    right[2, 2] = 100
    other = np.full((5, 5, 3), 255, np.uint8)

    picture = sv.render_flows({"a": left, "b": right, "c": other,
                               "d": np.zeros((3, 3), np.uint8)})

    assert picture.shape == (3, 3, 3)
    assert picture[0, 0].tolist() == [200, 200, 200]
    assert picture[2, 2].tolist() == [100, 100, 100]
    assert sv.render_flows({"flat": np.zeros((3, 3), np.uint8)}) is None


def test_cell_probability_takes_the_highest_over_matching_maps():
    low = np.full((2, 2), -5.0, np.float32)
    high = np.full((2, 2), 5.0, np.float32)
    prob = sv.cell_probability({"a": low, "b": high,
                                "c": np.zeros((3, 3), np.float32),
                                "d": np.zeros((2, 2, 2), np.float32)})
    assert prob.shape == (2, 2)
    assert prob[0, 0] == pytest.approx(1 / (1 + np.exp(-5.0)))


def test_without_a_colour_map_the_probability_is_drawn_in_grey(monkeypatch):
    import matplotlib

    class _NoMaps:
        def __getitem__(self, name):
            raise KeyError(name)

    monkeypatch.setattr(matplotlib, "colormaps", _NoMaps())
    picture = sv.render_cellprob(np.zeros((2, 2), np.float32))
    assert picture.shape == (2, 2, 3)
    assert picture[0, 0].tolist() == [128, 128, 128]


def test_an_unknown_view_is_refused(qtbot):
    views = sv.SegmentationViews()
    qtbot.addWidget(views)
    with pytest.raises(ValueError):
        views.set_view("Histogram")
    with pytest.raises(ValueError):
        views.set_renderer("Histogram", lambda arrays: None)


def test_a_renderer_that_raises_says_so_instead_of_crashing(qtbot):
    views = sv.SegmentationViews()
    qtbot.addWidget(views)

    def _broken(arrays):
        raise RuntimeError("renderer bug")

    views.set_renderer(sv.OVERLAY, _broken)
    views.set_arrays(image=np.zeros((4, 4), np.uint8))

    assert views._pages.currentWidget() is views.message
    assert views.message.text() == sv.COULD_NOT_DRAW
    assert views.arrays()["image"].shape == (4, 4)

    views.set_renderer(sv.OVERLAY, None)
    views.refresh()
    assert views._pages.currentWidget() is views.canvas


def test_a_selector_already_on_the_view_is_left_alone(qtbot):
    views = sv.SegmentationViews()
    qtbot.addWidget(views)
    holder = QLabel()
    qtbot.addWidget(holder)
    first = views.make_selector()
    second = views.make_selector(holder)
    seen = []
    views.view_changed.connect(seen.append)

    views.set_view(sv.OVERLAY)
    first.setCurrentIndex(first.findData(sv.MASKS))

    assert seen == [sv.MASKS]
    assert second.currentData() == sv.MASKS
    assert views.view() == sv.MASKS


def test_a_canvas_without_a_name_setter_still_shows_the_picture(qtbot):
    from PySide6.QtWidgets import QWidget

    class _BareCanvas(QWidget):
        def __init__(self):
            super().__init__()
            self.pixmap = None

        def set_pixmap(self, pixmap):
            self.pixmap = pixmap

        def picture(self):
            return self.pixmap

    canvas = _BareCanvas()
    views = sv.SegmentationViews(canvas=canvas)
    qtbot.addWidget(views)
    views.show_picture(np.zeros((3, 3), np.uint8))

    assert canvas.pixmap is not None and not canvas.pixmap.isNull()
    assert views._pages.currentWidget() is canvas
