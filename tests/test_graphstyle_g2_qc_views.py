"""Small source-identity checks for G2 image previews and native QC layouts."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.qt


@pytest.mark.parametrize("primaries", ["rgb", "cmy"])
@pytest.mark.parametrize("ratio", [1., 2.])
def test_measure_worker_and_gui_thumbnails_keep_the_same_pixels(qapp, primaries, ratio):
    """Both existing preview routes retain source RGB, rounding and device scale."""
    from PySide6.QtGui import QImage, QPixmap
    from spacr.qt.hidpi import scaled_for
    from spacr.qt.widgets import measure_preview as preview

    class Ratio:
        """Supply only the same display ratio used by the worker."""

        def devicePixelRatioF(self):
            """Return the explicit saved ratio."""
            return ratio

    pixels = np.arange(11 * 17 * 3, dtype=np.uint16).reshape(11, 17, 3).astype(np.uint8)
    original = pixels.tobytes()
    worker = preview._thumbnail_image(pixels, primaries, 48, ratio, preview._opaque_pixmap_format())
    visible = preview._preview_crop_pixels(pixels, primaries)
    image = QImage(visible.data, visible.shape[1], visible.shape[0],
                   visible.strides[0], QImage.Format_RGB888)
    gui = preview._rounded_pixmap(scaled_for(QPixmap.fromImage(image.copy()), Ratio(), 48))
    assert worker == gui.toImage()
    assert worker.devicePixelRatio() == ratio
    assert pixels.tobytes() == original
    assert worker.pixelColor(worker.width() // 2, worker.height() // 2).alpha() == 255


def test_measure_heading_keeps_category_counts_and_uses_shared_font(qtbot):
    """Readable heading styling does not alter filter membership or counts."""
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    entries = [(0, {"included": True}), (1, {"included": False})]
    label = MeasurePreviewPanel._category_header(None, "Fixed phenotype", entries)
    qtbot.addWidget(label)
    assert label.text() == "Fixed phenotype  ·  1 kept  ·  1 excluded by filters"
    assert label.font().family() == "Open Sans" and label.font().pointSizeF() >= 8.
    assert entries == [(0, {"included": True}), (1, {"included": False})]


def test_qc_render_declaration_keeps_exact_payload_and_composite(qapp):
    """The dispatch owner changes no channel, mask, source array or output pixel."""
    from spacr.qt.widgets import qc_field_browser as browser

    intensities = np.arange(8 * 9 * 2, dtype=np.uint16).reshape(8, 9, 2)
    mask = np.zeros((8, 9), dtype=np.uint16)
    mask[2:6, 3:7] = 5
    payload = browser.QCFieldImage(intensities=intensities, masks={"nucleus": mask})
    before = intensities.tobytes(), mask.tobytes()
    expected = browser.render_qc_field(payload, 1, ("nucleus",))

    class Channel:
        """Expose the explicit selected display channel."""

        def currentData(self):
            """Return channel one without normalization or selection changes."""
            return 1

    class Checked:
        """Expose one selected object-role outline."""

        def isChecked(self):
            """Keep the role selected."""
            return True

    class Jobs:
        """Exercise the supplied callback synchronously without a worker job."""

        cancelled = False

        def cancel(self):
            """Record the owner's usual stale-render cancellation."""
            self.cancelled = True

        def submit(self, function, receiver):
            """Evaluate only the already-small field composite."""
            receiver(function())

    output = []
    jobs = Jobs()
    owner = SimpleNamespace(_payload=payload, _channel=Channel(),
                            _layer_checks={"nucleus": Checked()},
                            _render_jobs=jobs, _on_rendered=output.append)
    browser.QCFieldBrowser._render(owner)
    assert jobs.cancelled and len(output) == 1
    np.testing.assert_array_equal(output[0], expected)
    assert (intensities.tobytes(), mask.tobytes()) == before
    assert owner._payload is payload
    declared = dict(owner._spacr_style_spec.comparison)
    assert declared["channel"] == 1 and declared["visible_masks"] == ("nucleus",)
    assert owner._spacr_style_spec.kind == "image"


@pytest.mark.parametrize("dark", [False, True])
def test_qc_histogram_keeps_counts_sqrt_and_inverse_handle_mapping(qtbot, monkeypatch, dark):
    """The measured square layout preserves all histogram and drag semantics."""
    from PySide6.QtCore import QPointF, QRectF
    from spacr.qt import theme
    from spacr.qt.widgets.qc_field_browser import _OverviewLevelsHistogram

    ground, ink = ("#000000", "#FFFFFF") if dark else ("#FFFFFF", "#000000")
    palette = {"surface_alt": ground, "fg": ink, "border": ink, "accent": "#0A77B8"}

    def explicit_palette():
        """Supply a fixed theme without changing persistent user settings."""
        return palette

    monkeypatch.setattr(theme, "_painting_palette", explicit_palette)
    widget = _OverviewLevelsHistogram()
    qtbot.addWidget(widget)
    counts = np.zeros(256)
    counts[[4, 32, 100, 224]] = [1., 4., 9., 16.]
    original = counts.tobytes()
    widget.set_counts(counts)
    widget.set_levels(.1, .9, 1.5)
    widget.resize(360, 480)
    widget.show()
    image = widget.grab().toImage()
    assert not image.isNull()
    observed = widget._spacr_style_observation()
    plot = QRectF(*observed["plot_rect"])
    assert plot.width() == pytest.approx(plot.height())
    data = [mark for mark in observed["marks"] if mark["role"] == "data"]
    np.testing.assert_array_equal(data[0]["bin_counts"], counts)
    np.testing.assert_allclose(data[0]["displayed_heights"], np.sqrt(counts / 16.))
    np.testing.assert_allclose(observed["threshold_values"], [value for _, value in widget._handles()])
    assert counts.tobytes() == original
    assert observed["text_geometry"]["outside"] == []
    assert observed["text_geometry"]["overlaps"] == []
    assert all(font["family"] == "Open Sans" and font["size"] >= 8. for font in observed["fonts"])
    assert all(swatch["colour"][3] == 1. for swatch in observed["legend_swatches"])
    assert data[0]["colours"][0][3] == pytest.approx(.6, abs=1 / 65535)
    assert observed["incomplete_checks"]
    from spacr.figures.style import finalize_figure

    diagnostics = finalize_figure(widget)
    assert "actual native plot area" in diagnostics.checks
    assert not diagnostics.compliant
    assert any("triangle" in problem for problem in diagnostics.problems)
    for level in (0., .1, .5, .9, 1.):
        assert widget._level(widget._x(level)) == pytest.approx(level)

    class Event:
        """Use the actual plot mapping without relying on screen coordinates."""

        def position(self):
            """Move the selected black handle to display level 0.2."""
            return QPointF(widget._x(.2), plot.center().y())

        def accept(self):
            """Accept the same event contract as Qt mouse motion."""

    changes = []
    widget.levels_changed.connect(_collect_levels(changes))
    widget._dragging = "black"
    widget.mouseMoveEvent(Event())
    assert widget.black == pytest.approx(.2)
    assert widget.white == .9 and widget.gamma == 1.5
    assert len(changes) == 1 and changes[0] == pytest.approx((.2, .9, 1.5))
    assert counts.tobytes() == original


@pytest.mark.parametrize("counts", [None, np.array([])])
def test_missing_histogram_is_distinct_from_observed_zero_counts(qtbot, counts):
    """An asynchronous missing histogram never becomes invented measured pixels."""
    from spacr.qt.widgets.qc_field_browser import _OverviewLevelsHistogram

    widget = _OverviewLevelsHistogram()
    qtbot.addWidget(widget)
    widget.resize(360, 480)
    widget.set_counts(counts)
    missing = widget._spacr_style_observation()
    assert not missing["histogram_data_available"] and missing["input_bin_count"] == 0
    assert not any(mark["role"] == "data" for mark in missing["marks"])
    assert "unavailable" in dict(widget._spacr_style_spec.legend)["caption"]
    assert any("unavailable" in problem for problem in missing["incomplete_checks"])
    zeros = np.zeros(256)
    widget.set_counts(zeros)
    measured = widget._spacr_style_observation()
    assert measured["histogram_data_available"] and measured["input_bin_count"] == 256
    data = [mark for mark in measured["marks"] if mark["role"] == "data"]
    assert len(data) == 1 and data[0]["bin_counts"] == zeros.tolist()
    assert not any("unavailable" in problem for problem in measured["incomplete_checks"])


def _collect_levels(destination):
    """Return a three-value receiver for the existing levels signal."""
    def collect(black, white, gamma):
        """Keep the complete emitted level tuple."""
        destination.append((black, white, gamma))
    return collect


def test_native_constant_histogram_keeps_original_bins(qapp):
    """Supported all-equal source edges gain only display-range padding."""
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QWidget
    from spacr.figures.style import paint_native_histogram

    widget = QWidget()
    widget.resize(320, 320)
    image = QImage(widget.size(), QImage.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    try:
        observed = paint_native_histogram(widget, painter,
            series=[{"label": "Pixels", "values": np.array([4.])}], edges=np.array([7., 7.]),
            xlabel="Intensity", ylabel="Count", caption="Fixed constant image.",
            reference="Supplied histogram.")
    finally:
        painter.end()
        widget.close()
    assert observed["bin_edges"] == [7., 7.]
    assert observed["data_ranges"][0] == [7., 8.]
    assert observed["marks"][0]["bin_counts"] == [4.]
    assert observed["marks"][0]["rectangles"][0][2] == 1.


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [180, 400])
def test_align_key_preserves_plan_and_click_identity(qtbot, monkeypatch, dark, width):
    """An outside confidence key changes only screen space, never solved placement."""
    from PySide6.QtCore import QRectF, Qt
    from spacr import align
    from spacr.qt.screens import align as screen

    original_palette = dict(screen.active_palette())
    original_palette.update(surface="#000000" if dark else "#FFFFFF",
                            surface_hi="#1f2124" if dark else "#e6e9ee",
                            fg="#FFFFFF" if dark else "#000000")

    def explicit_palette():
        """Exercise exact saved light and dark appearance independently."""
        return dict(original_palette)

    def long_localized_label(source):
        """Exercise a longer complete label instead of assuming English width."""
        return {"Registration confidence: 0.3": "Registrierungsvertrauen der Kachel: 0.3",
                "Registration confidence: 1.0": "Registrierungsvertrauen der Kachel: 1.0",
                "Stage-position placement": "Platzierung anhand der Tischposition",
                "Unreadable tile": "Nicht lesbare Bildkachel"}.get(source, source)

    monkeypatch.setattr(screen, "active_palette", explicit_palette)
    monkeypatch.setattr(screen, "tr", long_localized_label)

    tiles = [align.Tile(path=f"unused-{index}.npy", index=index, field=index + 1,
                        shape=(20, 30, 1)) for index in range(2)]
    placements = [align.Placement(tiles[0], y=0., x=0., confidence=.75,
                                  method=align.METHOD_REGISTRATION),
                  align.Placement(tiles[1], y=0., x=40., method=align.METHOD_NOMINAL)]
    plan = align.AlignPlan(tiles=tiles, placements=placements, canvas_shape=(20, 70, 1))
    before = asdict(plan)
    widget = screen.TileLayoutWidget()
    qtbot.addWidget(widget)
    widget.resize(width, 300)
    widget.set_plan(plan)
    widget.show()
    assert not widget.grab().isNull()
    assert widget.plan() is plan and asdict(plan) == before
    key = widget._spacr_layout_key
    assert len(key) == 4 and key[2]["hatched"]
    assert key[0]["colour"] == screen.confidence_colour(.3, align.METHOD_REGISTRATION).name()
    assert key[0]["outline"] == ("#ffffff" if dark else "#000000")
    assert key[2]["colour"] == screen.confidence_colour(0., align.METHOD_NOMINAL).name()
    assert all(row["inside_canvas"] and row["outside_tiles"] and row["font_points"] >= 8.
               for row in key)
    assert all(row["foreground"] == ("#ffffff" if dark else "#000000") for row in key)
    bounds = [QRectF(*row["text_bounds"]) for row in key]
    assert all(not first.intersects(second) for index, first in enumerate(bounds)
               for second in bounds[index + 1:])
    clicks = []
    widget.tile_clicked.connect(clicks.append)
    for index, rect in widget.tile_rects():
        assert all(not rect.intersects(QRectF(*row["rect"])) for row in key)
        qtbot.mouseClick(widget, Qt.LeftButton, pos=rect.center().toPoint())
        assert clicks[-1] == index
    assert asdict(plan) == before


def test_overview_export_keeps_exact_rgb_and_selection(qapp, tmp_path, monkeypatch):
    """The normal export owner declares image axes without restyling their pixels."""
    from spacr import plot
    from spacr.figures import style
    from spacr.qt.widgets import qc_field_browser as browser

    pixels = np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3)
    original = pixels.tobytes()
    field = browser._OverviewField("plate", "A01", "A", 1, "4", (("", "unused.npy"),))
    calls, written = [], []

    def supplied_pixels(selected, channels, percentiles, **kwargs):
        """Return only the fixed already-rendered RGB view for this layout test."""
        calls.append((selected, channels, percentiles, kwargs))
        return pixels

    def channel_info(selected):
        """Avoid source I/O while preserving this explicit channel identity."""
        assert selected is field
        return {"files": ()}

    def capture(figure, path, **kwargs):
        """Inspect the exact normal-writer input without an extra render."""
        written.append((figure, path, kwargs))

    def no_preferences(kind=None):
        """Remove workstation-specific manual appearance from the fixture."""
        return {}

    monkeypatch.setattr(browser, "_overview_pixels", supplied_pixels)
    monkeypatch.setattr(browser, "_overview_channel_info", channel_info)
    monkeypatch.setattr(plot, "save_figure", capture)
    monkeypatch.setattr(style, "_preference_deltas", no_preferences)
    output = tmp_path / "overview"
    paths, errors = browser._overview_export((field,), output, (0, 1, 2), (1, 99), seed=17,
                                            quality_edge=None, dpi=72)
    assert len(paths) == len(written) == 1 and errors == ()
    assert calls[0][:3] == (field, (0, 1, 2), (1, 99))
    assert calls[0][3]["edge"] is None
    figure = written[0][0]
    text_specification = style.resolve_style("print", "image")
    assert len(figure.texts) == 2
    assert all(text_specification.font_family in text.get_fontfamily()
               and text.get_fontweight() == dict(text_specification.font_weights)["body"]
               for text in figure.texts)
    assert figure.texts[0].get_text() == "Channels: (0, 1, 2); percentiles: (1, 99); seed: 17"
    assert figure.texts[1].get_text() == field.caption
    axis = figure.axes[0]
    np.testing.assert_array_equal(axis.images[0].get_array(), pixels)
    assert axis.images[0].get_alpha() is None
    assert axis._spacr_style_spec.kind == "image"
    assert pixels.tobytes() == original
    provenance = json.loads((output / "overview_selection.json").read_text())
    assert provenance["seed"] == 17 and provenance["channels"] == [0, 1, 2]
    assert provenance["fields"][0]["field"] == "4"
    assert provenance["percentiles"] == [1, 99]


def test_normal_dark_writer_keeps_colourbar_readable_and_scalar_mapping_exact(qapp, tmp_path, monkeypatch):
    """Real saved preferences and writer retain readable keys without changing values."""
    from PySide6.QtCore import QSettings
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text
    from spacr.figures import style
    from spacr.plot import save_figure
    from spacr.qt import preferences

    def temporary_settings():
        """Keep real preference setters confined to the temporary test store."""
        return QSettings(str(tmp_path / "figures.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr(preferences, "_settings", temporary_settings)
    preferences.set_figure_style({"background": "#181818", "foreground": "#F0F0F0"})
    preferences.set_figure_colors("#181818", "#F0F0F0")
    values = np.array([[0., .25], [.75, 1.]])
    source = values.tobytes()
    with style.figure_style("screen"):
        figure, ax = plt.subplots(figsize=(3., 3.))
        try:
            image = style.heatmap_image(ax, values, cmap=style.heatmap_cmap(), vmin=0., vmax=1.)
            key = style.attach_colourbar(ax, image, label="Fraction", unit="dimensionless")
            specification = style.resolve_style("screen", "heatmap", legend={
                "caption": "Fixed supplied fractions.", "reference": "Supplied zero to one range.",
                "quantities": {"x": "column", "y": "row", "value": "fraction"},
                "units": {"x": "index", "y": "index", "value": "dimensionless"},
                "colour_quantity": "Fraction", "colour_unit": "dimensionless"})
            style.apply_axes_style(ax, specification, layout=False)
            norm, cmap, limits, alpha = image.norm, image.get_cmap(), image.get_clim(), image.get_alpha()
            key_norm, key_cmap = key.norm, key.cmap
            key_array, key_alpha = key.solids.get_array().copy(), key.solids.get_alpha()
            ticks = key.get_ticks().copy()
            assert style.apply_colourbar_theme(ax) == (key,)
            path = tmp_path / "dark-key.png"
            assert save_figure(figure, str(path), fmt="png", dpi=72, save_mode="screen") == str(path)
            assert path.is_file() and path.stat().st_size > 0
            assert values.tobytes() == source
            np.testing.assert_array_equal(image.get_array(), values)
            assert image.norm is norm and image.get_cmap() is cmap
            assert image.get_clim() == limits and image.get_alpha() == alpha
            assert key.norm is key_norm and key.cmap is key_cmap
            np.testing.assert_array_equal(key.solids.get_array(), key_array)
            np.testing.assert_array_equal(key.get_ticks(), ticks)
            assert key.solids.get_alpha() == key_alpha
            assert to_rgba(key.ax.get_facecolor()) == to_rgba("#181818")
            texts = [text for text in key.ax.findobj(Text) if text.get_visible() and text.get_text().strip()]
            assert texts and any(text.get_text() == "Fraction (dimensionless)" for text in texts)
            assert all(to_rgba(text.get_color(), text.get_alpha()) == to_rgba("#F0F0F0") for text in texts)
            diagnostics = style.finalize_figure(figure)
            assert not any("colour-key" in problem for problem in diagnostics.problems)
        finally:
            plt.close(figure)
