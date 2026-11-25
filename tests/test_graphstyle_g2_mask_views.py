"""Actual native histogram paints and image views preserve their scientific inputs."""

import numpy as np
import pytest
from types import SimpleNamespace

from spacr.figures import style
from spacr.qt.screens import annotate, make_masks
from spacr.qt.widgets import segmentation_views as views


pytestmark = pytest.mark.qt


@pytest.fixture(params=["light", "dark"])
def mask_plot_theme(request, monkeypatch):
    """Supply both actual native paint themes without changing saved preferences."""
    dark = request.param == "dark"
    palette = {"bg": "#1b1e20" if dark else "#ffffff",
               "fg": "#ffffff" if dark else "#000000"}
    tile, active = annotate.tile_palette, make_masks.active_palette
    monkeypatch.setattr(annotate, "tile_palette", lambda: dict(tile(), **palette))
    monkeypatch.setattr(make_masks, "active_palette", lambda: dict(active(), **palette))
    return palette


def _assert_native_histogram(widget):
    """Check policy against the actual paint and completed text geometry."""
    observed = widget._spacr_style_observation()
    assert observed["width_inches"] == pytest.approx(observed["height_inches"])
    assert all(edge["visible"] for edge in observed["frame"])
    assert all(item["family"] == "Open Sans" and item["size"] >= 8
               and item["weight"] == 300 for item in observed["fonts"])
    assert all(colour[3] * item["opacity"] == pytest.approx(0.6, abs=1e-6)
               for item in observed["marks"] for colour in item["colours"])
    assert all(item["colour"][3] == 1.0 for item in observed["legend_swatches"])
    assert not observed["text_geometry"]["outside"]
    assert not observed["text_geometry"]["overlaps"]
    report = style.finalize_figure(widget)
    assert report.compliant, report.problems
    return observed


def test_probability_histogram_preserves_boundary_counts_and_explicit_thresholds(qtbot, mask_plot_theme):
    """True class labels cannot silently become controls or alter probability bins."""
    widget = annotate._ProbabilityHistogram()
    qtbot.addWidget(widget)
    widget.resize(540, 620)
    values = {"reference": [0.0, 0.04, 0.1, 0.15, 0.99, 1.0], "treated": [0.25, 0.75]}
    widget.set_distribution(["reference", "treated"], values, "Probability")
    widget.set_lines([0.2, 0.8])
    observed = _assert_native_histogram(widget)
    assert widget.counts()["reference"] == [2, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    assert widget.lines() == observed["threshold_values"] == [0.2, 0.8]
    assert not widget._spacr_style_spec.controls
    assert len(set(widget._spacr_style_spec.colours)) == 2
    assert [item["bin_counts"] for item in observed["marks"] if item["role"] == "data"] == list(widget.counts().values())
    assert values == {"reference": [0.0, 0.04, 0.1, 0.15, 0.99, 1.0], "treated": [0.25, 0.75]}


def test_nine_probability_classes_have_distinct_visible_keys(qtbot, mask_plot_theme):
    """Every supported annotation class needs a distinct shared color/pattern key."""
    widget = annotate._ProbabilityHistogram()
    qtbot.addWidget(widget)
    widget.resize(640, 900)
    classes = list(range(1, 10))
    widget.set_distribution(classes, {str(value): [0.05, 0.55, 0.95] for value in classes}, "Probability")
    widget.set_lines([0.2, 0.8])
    observed = _assert_native_histogram(widget)
    marks = [item for item in observed["marks"] if item["role"] == "data"]
    keys = [item for item in observed["legend_swatches"] if item["role"] == "data"]
    encodings = [(tuple(item["colours"][0][:3]), item["brush_style"]) for item in marks]
    assert len(encodings) == len(set(encodings)) == 9
    assert [(tuple(item["colour"][:3]), item["brush_style"]) for item in keys] == encodings
    assert all(sum(item["bin_counts"]) == 3 for item in marks)


def test_otsu_histogram_includes_corrected_levels_without_changing_bins(qtbot, mask_plot_theme):
    """A corrected level beyond the image range remains distinct from its maximum."""
    counts = np.array([2.0, 5.0, 3.0, 1.0])
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    original = (counts.tobytes(), edges.tobytes())
    widget = make_masks._OtsuHistogramPlot(counts, edges, [-0.1, 1.2])
    qtbot.addWidget(widget)
    widget.resize(540, 620)
    observed = _assert_native_histogram(widget)
    assert observed["data_ranges"][0] == [-0.1, 1.2]
    assert observed["threshold_positions"] == pytest.approx([widget.level_x(-0.1), widget.level_x(1.2)])
    assert widget.level_x(1.0) < widget.level_x(1.2)
    assert observed["marks"][0]["bin_counts"] == counts.tolist()
    assert (counts.tobytes(), edges.tobytes()) == original


def test_levels_pointer_inverse_uses_the_same_painted_square(qtbot, mask_plot_theme):
    """Inset axes must not shift a normalization cutoff under the pointer."""
    widget = make_masks._LevelsPlot([1, 3, 2, 1], [0, 1, 2, 3, 4], [1, 3])
    qtbot.addWidget(widget)
    widget.resize(540, 620)
    emitted = []
    widget.cutoff_changed.connect(lambda index, value: emitted.append((index, value)))
    widget._drag_cutoff = 0
    for value in (0, 0.5, 2.25, 4):
        widget._move_cutoff(widget.level_x(value))
        assert emitted[-1] == pytest.approx((0, value))
    assert widget.levels == [1.0, 3.0]
    _assert_native_histogram(widget)


def test_flat_histogram_still_paints_its_existing_counts(qtbot, mask_plot_theme):
    """Constant input retains degenerate source bin edges without a paint error."""
    widget = make_masks._OtsuHistogramPlot([4], [7, 7], [])
    qtbot.addWidget(widget)
    widget.resize(540, 620)
    observed = _assert_native_histogram(widget)
    assert observed["bin_edges"] == [7.0, 7.0]
    assert observed["marks"][0]["bin_counts"] == [4.0]
    assert widget.level_x(7) == widget._plot_geometry()["plot_rect"].left()


def test_otsu_minimum_height_follows_the_actual_legend(qtbot, mask_plot_theme):
    """Many explicit cut levels must reserve their measured key space in the producer."""
    counts = np.ones(14)
    edges = np.arange(15, dtype=float)
    levels = np.arange(0.5, 12.0, 1.0).tolist()
    before = counts.tobytes(), edges.tobytes(), list(levels)
    widget = make_masks._OtsuHistogramPlot(counts, edges, levels)
    qtbot.addWidget(widget)
    widget.resize(420, 120)
    observed = _assert_native_histogram(widget)
    assert widget.minimumHeight() >= widget._plot_geometry()["minimum_height"]
    assert widget.height() >= widget.minimumHeight()
    assert observed["width_inches"] == pytest.approx(observed["height_inches"])
    assert (counts.tobytes(), edges.tobytes(), widget.levels) == before


def test_image_view_legends_leave_raw_pixels_labels_and_probability_unchanged(qtbot):
    """Captions describe each actual renderer without recoloring its pixels."""
    widget = views.SegmentationViews()
    qtbot.addWidget(widget)
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    labels = np.zeros((4, 4), dtype=np.uint16)
    labels[1:3, 1:3] = 39
    flow = np.full((4, 4, 3), [19, 47, 83], dtype=np.uint8)
    logits = np.linspace(-4, 4, 16, dtype=np.float32).reshape(4, 4)
    before = tuple(array.tobytes() for array in (image, labels, flow, logits))
    widget.set_arrays(image=image, labels={"cells": labels}, flows=flow, cellprob=logits)
    assert widget._image_display_metadata["overlays"][0]["identity"] == "cells"
    np.testing.assert_array_equal(widget.rendered()[0], image[0])
    np.testing.assert_array_equal(widget.rendered()[1, 1], views.DEFAULT_COLOURS[0])
    widget.set_view(views.FLOWS)
    np.testing.assert_array_equal(widget.rendered(), flow)
    assert "maximum" in widget.display_legend.text()
    widget.set_view(views.MASKS)
    assert "intensity" in widget.display_legend.text()
    widget.set_view(views.CELLPROB)
    np.testing.assert_array_equal(widget.rendered(), views.render_cellprob(logits))
    assert widget._image_display_metadata["display_range"] == (0.0, 1.0)
    assert widget._image_display_metadata["colormap"] == views.CELLPROB_COLORMAP
    assert tuple(array.tobytes() for array in (image, labels, flow, logits)) == before
    font = widget.display_legend.font()
    points = font.pointSizeF() if font.pointSizeF() > 0 else font.pixelSize() * 72 / 96
    assert points >= 8
    assert not hasattr(widget, "_spacr_style_diagnostics")


def test_custom_image_renderer_cannot_inherit_an_inaccurate_default_caption(qtbot):
    """A caller-supplied renderer must not be described as a built-in pipeline."""
    widget = views.SegmentationViews()
    qtbot.addWidget(widget)
    widget.set_arrays(image=np.zeros((3, 3), np.uint8))
    assert widget.display_legend.text()
    picture = np.full((3, 3, 3), 71, np.uint8)
    widget.set_renderer(views.OVERLAY, lambda arrays: picture)
    widget.refresh()
    assert widget._image_display_metadata == {}
    assert widget._spacr_style_spec is None
    assert not widget.display_legend.text()
    np.testing.assert_array_equal(widget.rendered(), picture)


@pytest.mark.parametrize("view", [views.OVERLAY, views.FLOWS])
def test_image_metadata_reports_the_actual_float32_constant_branch(qtbot, view):
    """Adjacent large float64 inputs can collapse during the existing float32 display conversion."""
    widget = views.SegmentationViews()
    qtbot.addWidget(widget)
    source = np.array([[16777216.0, 16777217.0]], dtype=np.float64)
    if view == views.FLOWS:
        source = np.repeat(source[..., None], 3, axis=2)
        widget.set_arrays(flows={"cells": source}, refresh=False)
    else:
        widget.set_arrays(image=source, refresh=False)
    original = source.tobytes()
    widget.set_view(view)
    metadata = widget._image_display_metadata
    if view == views.FLOWS:
        metadata = metadata["flow_normalization"]["cells"]
    assert metadata["source_range"] == (16777216.0, 16777217.0)
    assert metadata["normalization_range"] == (16777216.0, 16777216.0)
    assert metadata["normalization_dtype"] == "float32"
    assert metadata["normalization_branch"] == "constant_zero"
    assert not widget.rendered().any()
    assert source.tobytes() == original


def test_annotation_count_paint_preserves_the_supplied_crop(qtbot):
    """Readable count typography must not bake borders or text into crop pixels."""
    widget = annotate._Thumbnail(0)
    qtbot.addWidget(widget)
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    picture = views.to_qpixmap(image)
    key = picture.cacheKey()
    widget.setPixmap(picture)
    widget.set_occupied(True)
    widget.set_count_text("39")
    widget.resize(180, 180)
    widget.grab()
    assert widget.pixmap().cacheKey() == key
    assert widget.count_text() == "39"
    assert widget.pixmap().toImage() == picture.toImage()


@pytest.mark.parametrize("gathered", [True, False])
def test_layer_compositors_remain_exact_rgb32_boundaries(gathered):
    """Worker and direct compositing retain RGB channels and opaque image pixels."""
    rgb = np.array([[[18, 52, 86], [255, 0, 1]], [[0, 128, 255], [9, 8, 7]]], dtype=np.uint8)
    original = rgb.tobytes()

    class Stack:
        """Supply an already-composited image without rendering a real layer stack."""

        def _render_uint8_gathered(self, canvas, rows=slice(None), cols=slice(None)):
            """Return the existing crop when gathered rendering is available."""
            return rgb[rows, cols] if gathered else None

        def render_uint8(self, canvas):
            """Return the same source image for the fallback route."""
            return rgb

    stack = Stack()
    expected = np.array([[0xFF123456, 0xFFFF0001], [0xFF0080FF, 0xFF090807]], dtype=np.uint32)
    direct = make_masks._MaskCanvas._layered_picture(SimpleNamespace(field_layers=stack), None)
    worker = make_masks._layers_picture_for(SimpleNamespace(stack=stack, canvas=None))
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(worker, expected)
    assert rgb.tobytes() == original


@pytest.mark.parametrize("kind", ["boxes", "recrop"])
def test_zoomed_canvas_labels_require_a_visible_box(qtbot, monkeypatch, mask_plot_theme, kind):
    """Offscreen annotations stay absent while partial and visible boxes keep their names."""
    from PySide6.QtGui import QPainter

    widget = make_masks._MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    mask = np.zeros((64, 64), dtype=np.uint16)
    original = image.tobytes(), mask.tobytes()
    widget.set_image_and_mask(image, mask)
    widget.zoom_at(32, 32, 4.0)
    x0, y0, _, _ = widget._viewport_bounds()
    rectangles = [(x0 + 3, y0 + 3, x0 + 7, y0 + 7),
                  (x0 - 8, y0 - 8, x0 - 4, y0 - 4),
                  (x0 - 3, y0 + 8, x0 + 3, y0 + 12)]
    names = ["inside name", "outside name", "partial name"]
    if kind == "boxes":
        widget.box_classes = list(names)
        widget.boxes = [(index, *rectangle) for index, rectangle in enumerate(rectangles)]
    else:
        widget.recrop_boxes = [(*rectangle, name) for rectangle, name in zip(rectangles, names)]
    boxes, recrops = list(widget.boxes), list(widget.recrop_boxes)
    drawn = []

    class RecordingPainter(QPainter):
        """Observe text while the actual native painter draws the real canvas."""

        def drawText(self, *args):
            """Retain text supplied to the normal draw operation."""
            if args and isinstance(args[-1], str):
                drawn.append(args[-1])
            return super().drawText(*args)

    monkeypatch.setattr(make_masks, "QPainter", RecordingPainter)
    widget.grab()
    assert "outside name" not in drawn
    assert "inside name" in drawn and "partial name" in drawn
    assert widget.boxes == boxes and widget.recrop_boxes == recrops
    assert (image.tobytes(), mask.tobytes()) == original
