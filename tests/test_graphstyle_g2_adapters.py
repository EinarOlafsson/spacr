"""Additive G2 adapters preserve source values and expose measurable external keys."""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from spacr.figures import style


@pytest.fixture(autouse=True)
def isolated_appearance(monkeypatch):
    """Exercise producer defaults independently of workstation preferences."""
    def no_overrides(kind=None):
        """Return no saved workstation appearance override."""
        return {}

    monkeypatch.setattr(style, "_preference_deltas", no_overrides)


@pytest.mark.parametrize("dark,ground,ink", [(False, "white", "black"), (True, "black", "white")])
def test_theme_retains_saved_roles_and_falls_back_only_for_invalid_values(dark, ground, ink):
    """Saved RGB values win individually; invalid roles use the explicit fallback."""
    saved = {"background": [0.1, 0.2, 0.3], "foreground": "not-a-colour", "border": "#abcdef", "extra": "red"}
    result = style.resolve_plot_theme(dark, saved)
    assert result == {"background": [0.1, 0.2, 0.3], "foreground": ink, "border": "#abcdef"}
    result["background"][0] = 0.9
    assert saved["background"] == [0.1, 0.2, 0.3]
    assert style.resolve_plot_theme(dark) == {"background": ground, "foreground": ink, "border": ink}
    assert style.resolve_plot_theme(dark, "invalid mapping") == style.resolve_plot_theme(dark)


@pytest.mark.parametrize("method,unit", [("tsne", None), ("pca", "a.u."), ("embedding", None)])
@pytest.mark.parametrize("preserve", [False, True])
def test_embedding_preserves_geometry_colours_and_truthful_method(method, unit, preserve):
    """Actual coordinates, hulls, limits and saved opacity survive without a UMAP claim."""
    figure = Figure(figsize=(4., 4.))
    ax = figure.add_subplot()
    coordinates = np.array([[1., 2.], [3., 4.], [5., 3.]])
    labels = np.array([2, 2, 7])
    colours = {2: style.Palette.ACCENT, 7: style.Palette.TEAL}
    points = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                        c=[colours[value] for value in labels], alpha=np.array([0., .35, .8]))
    hull, = ax.plot([1., 3., 5., 1.], [2., 4., 3., 2.], color=style.Palette.NAVY_DARK,
                    linewidth=1.7, alpha=.35)
    centroid = ax.text(3., 3., "2")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xlim(-1., 8.)
    ax.set_ylim(0., 6.)
    style.annotate(ax, "An existing source note.")
    before = (points.get_offsets().copy(), hull.get_xydata().copy(), points.get_facecolors().copy(),
              figure.get_size_inches().copy(), coordinates.tobytes(), labels.tobytes())
    spec = style.apply_embedding_style(ax, labels, caption="Fixed supplied coordinates.",
            colours=colours, method=method, unit=unit, preserve_opacity=preserve,
            centroid_labels=[centroid], outline_colours={2: style.Palette.NAVY_DARK},
            theme_colors={"background": "black", "foreground": (1., 1., 1., 1.), "border": "#abcdef"})
    np.testing.assert_array_equal(points.get_offsets(), before[0])
    np.testing.assert_array_equal(hull.get_xydata(), before[1])
    np.testing.assert_array_equal(points.get_facecolors()[:, :3], before[2][:, :3])
    np.testing.assert_array_equal(figure.get_size_inches(), before[3])
    assert coordinates.tobytes() == before[4] and labels.tobytes() == before[5]
    np.testing.assert_array_equal(points.get_alpha(), [0., .35, .8] if preserve else [0., .6, .6])
    assert hull.get_alpha() == (.35 if preserve else .6)
    assert hull.get_linewidth() == 1.7
    assert ax.get_xlim() == (-1., 8.) and ax.get_ylim() == (0., 6.)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Component 1", "Component 2")
    assert ax.get_box_aspect() == 1 and not spec.controls
    assert centroid not in ax.texts and not figure.legends
    notes = " ".join(row["label"] for row in ax._spacr_legend_entries)
    assert "UMAP" not in notes and "An existing source note." in notes
    assert dict(spec.comparison)["embedding_method"] == method
    assert dict(spec.legend).get("unit") == ("dimensionless" if method == "tsne" else unit)
    assert to_rgba(ax.get_facecolor()) == to_rgba("black")
    assert to_rgba(ax.xaxis.label.get_color()) == to_rgba("white")
    assert to_rgba(ax.spines["left"].get_edgecolor()) == to_rgba("#abcdef")
    assert all(to_rgba(row["colour"]) for row in ax._spacr_legend_entries)


def test_embedding_replaces_only_its_own_method_notes_and_refuses_images():
    """Reusing axes cannot leave a stale UMAP caption or restyle image pixels."""
    figure = Figure()
    ax = figure.add_subplot()
    ax.scatter([1., 2.], [3., 4.], color=style.Palette.ACCENT)
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    style.annotate(ax, "Preserve this source note.")
    for method in ("umap", "tsne"):
        style.apply_embedding_style(ax, [0, 0], caption="Supplied coordinates.",
                                    colours={0: style.Palette.ACCENT}, method=method)
    labels = [row["label"] for row in ax._spacr_legend_entries]
    assert sum(text == "Preserve this source note." for text in labels) == 1
    assert not any("UMAP" in text for text in labels)
    image = ax.imshow([[1., 2.], [3., 4.]])
    before = image.get_array().copy()
    with pytest.raises(ValueError, match="non-image"):
        style.apply_embedding_style(ax, [0, 0], caption="Image is not an embedding.",
                                    colours={0: style.Palette.ACCENT}, method="embedding")
    np.testing.assert_array_equal(image.get_array(), before)


def _declared_panel(figure):
    """Create one fixed source panel with caption, key and reference metadata."""
    ax = figure.add_subplot()
    ax.scatter([.2, .8], [.4, .6], color=style.Palette.ACCENT, alpha=.6)
    ax.set_xlabel("Coordinate (a.u.)")
    ax.set_ylabel("Value (a.u.)")
    spec = style.resolve_style("print", "scatter", legend={
        "caption": "Fixed synthetic source.", "reference": "Threshold", "unit": "a.u.",
        "quantities": {"x": "Coordinate", "y": "Value"}})
    style.apply_axes_style(ax, spec, layout=False)
    style.annotate(ax, "Fixed synthetic source with a longer explanation that belongs outside the plotting area.")
    style.text_legend(ax, [("Observed", style.Palette.ACCENT)])
    style.reference_line(ax, y=.5, label="Threshold")
    return ax


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("dark", [False, True])
def test_external_key_is_visible_opaque_measured_and_outside_data(pack, columns, dark):
    """Rendered key text stays inside the canvas and outside unchanged source coordinates."""
    figure = Figure(figsize=(4., 3.), dpi=100)
    ax = _declared_panel(figure)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    area_before = ax.get_window_extent(canvas.get_renderer()).size.copy() / figure.dpi
    points = ax.collections[0].get_offsets().copy()
    limits = (ax.get_xlim(), ax.get_ylim())
    metadata = tuple(dict(row) for row in ax._spacr_legend_entries)
    theme = style.resolve_plot_theme(dark)
    legend = style.external_legend(figure, axes=[ax], columns=columns, column_width=1.4,
                                   fontsize=6., pack=pack, theme_colors=theme)
    assert figure.canvas is canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    key = legend.get_window_extent(renderer)
    assert not key.overlaps(ax.get_tightbbox(renderer))
    assert key.x0 >= 0 and key.y0 >= 0
    assert key.x1 <= figure.bbox.width and key.y1 <= figure.bbox.height
    assert len(legend.get_texts()) == 3 and legend.get_frame().get_alpha() == 1.
    for text in legend.get_texts():
        assert text.get_fontsize() >= 8. and text.get_alpha() == 1.
        assert "Open Sans" in text.get_fontfamily()
    keys = getattr(legend, "legend_handles", None)
    if keys is None:
        keys = legend.legendHandles
    assert all(handle.get_alpha() == 1. for handle in keys)
    assert to_rgba(keys[1].get_markerfacecolor()) == to_rgba(style.Palette.ACCENT)
    assert to_rgba(keys[2].get_color()) == to_rgba("black")
    assert bool(keys[2].get_path_effects()) is dark
    assert to_rgba(legend.get_frame().get_facecolor()) == to_rgba(theme["background"])
    assert all(to_rgba(text.get_color()) == to_rgba(theme["foreground"]) for text in legend.get_texts())
    receipt = figure._spacr_external_legend_layout
    assert receipt["outside_panels"] and receipt["inside_canvas"]
    assert tuple(ax._spacr_legend_entries) == metadata
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), points)
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    if not pack:
        np.testing.assert_allclose(ax.get_window_extent(renderer).size / figure.dpi, area_before, rtol=0., atol=1e-8)
    else:
        np.testing.assert_allclose(ax.get_window_extent(renderer).size / figure.dpi,
                                   [40. / 25.4, 40. / 25.4], rtol=0., atol=1e-8)
    before = figure.get_size_inches().copy()
    second = style.external_legend(figure, axes=[ax], columns=columns, column_width=1.4,
                                   pack=pack, theme_colors=theme)
    assert second is not legend and len(figure.legends) == 1
    np.testing.assert_allclose(figure.get_size_inches(), before, rtol=0., atol=1e-8)


def test_external_key_leaves_raw_image_values_and_dimensions_unchanged():
    """Image metadata can gain a visible key without statistical alpha on pixels."""
    figure = Figure(figsize=(3., 2.))
    ax = figure.add_subplot()
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    image = ax.imshow(pixels, vmin=0, vmax=15, interpolation="nearest")
    ax.set_axis_off()
    style.annotate(ax, "Fixed raw field; no pixel transformation.")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    before = (image.get_array().copy(), image.get_clim(), image.get_alpha(),
              ax.get_window_extent(canvas.get_renderer()).size.copy())
    style.external_legend(figure, axes=[ax], pack=False)
    canvas.draw()
    np.testing.assert_array_equal(image.get_array(), before[0])
    assert image.get_clim() == before[1] and image.get_alpha() == before[2]
    np.testing.assert_allclose(ax.get_window_extent(canvas.get_renderer()).size, before[3], rtol=0., atol=1e-8)


@pytest.mark.parametrize("dark", [False, True])
def test_unpacked_key_preserves_panel_inches_and_pads_overflowing_labels(dark):
    """Existing small canvases gain margins rather than clipping labels or shrinking data."""
    figure = Figure(figsize=(3., 3.), dpi=100)
    ax = _declared_panel(figure)
    ax.set_position([.04, .02, .86, .86])
    ax.set_xlabel("Embedding coordinate 1", labelpad=12.)
    ax.set_ylabel("Embedding coordinate 2", labelpad=12.)
    footer = figure.text(.05, -.08, "Explicit supplied coordinate system.", va="top")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    assert ax.get_tightbbox(renderer).x0 < 0 and ax.get_tightbbox(renderer).y0 < 0
    before = ax.get_window_extent(renderer).size.copy() / figure.dpi
    points, limits = ax.collections[0].get_offsets().copy(), (ax.get_xlim(), ax.get_ylim())
    style.external_legend(figure, axes=[ax], pack=False, theme_colors=style.resolve_plot_theme(dark))
    canvas.draw()
    renderer = canvas.get_renderer()
    for bounds in (ax.get_tightbbox(renderer), footer.get_window_extent(renderer)):
        assert bounds.x0 >= .1 * figure.dpi - 1e-8
        assert bounds.y0 >= .1 * figure.dpi - 1e-8
        assert bounds.x1 <= figure.bbox.width and bounds.y1 <= figure.bbox.height
    np.testing.assert_allclose(ax.get_window_extent(renderer).size / figure.dpi, before, rtol=0., atol=1e-8)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), points)
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert all(value > 0 for value in figure._spacr_external_legend_layout["content_shift_inches"])
    size, position = figure.get_size_inches().copy(), ax.get_position().bounds
    style.external_legend(figure, axes=[ax], pack=False, theme_colors=style.resolve_plot_theme(dark))
    np.testing.assert_allclose(figure.get_size_inches(), size, rtol=0., atol=1e-8)
    np.testing.assert_allclose(ax.get_position().bounds, position, rtol=0., atol=1e-8)


def test_empty_external_key_does_not_resize_the_figure():
    """Axes without retained metadata do not gain an empty box or blank column."""
    figure = Figure(figsize=(3., 2.))
    ax = figure.add_subplot()
    before = figure.get_size_inches().copy()
    assert style.external_legend(figure, axes=[ax], pack=False) is None
    np.testing.assert_array_equal(figure.get_size_inches(), before)
    assert not figure.legends and figure._spacr_external_legend_layout is None


@pytest.mark.parametrize("dark", [False, True])
def test_below_key_bounds_canvas_without_shrinking_image_pixels_or_panel(dark):
    """A complete below-panel caption fits the declared width while image inches stay exact."""
    maximum_width = 170. / 25.4
    figure = Figure(figsize=(maximum_width, 4.), dpi=100)
    ax = figure.add_axes([.1, .15, .8, .75])
    pixels = np.arange(24, dtype=np.uint16).reshape(4, 6)
    image = ax.imshow(pixels, vmin=0., vmax=30., interpolation="nearest")
    ax.set_axis_off()
    style.annotate(ax, "Supplied microscopy pixels and masks remain unchanged; only the complete explanatory key moves below the image panels.")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    size = ax.get_window_extent(renderer).size.copy() / figure.dpi
    norm, cmap, alpha, limits = image.norm, image.get_cmap(), image.get_alpha(), image.get_clim()
    legend = style.external_legend(figure, axes=[ax], pack=False, placement="below",
        maximum_width=maximum_width, column_width=8., theme_colors=style.resolve_plot_theme(dark))
    canvas.draw()
    renderer = canvas.get_renderer()
    bounds = legend.get_window_extent(renderer)
    assert figure.get_size_inches()[0] <= maximum_width + 1e-8
    assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= figure.bbox.width
    assert bounds.y1 + .25 * figure.dpi <= ax.get_tightbbox(renderer).y0 + 1e-8
    np.testing.assert_allclose(ax.get_window_extent(renderer).size / figure.dpi, size, rtol=0., atol=1e-8)
    np.testing.assert_array_equal(image.get_array(), pixels)
    assert image.norm is norm and image.get_cmap() is cmap
    assert image.get_alpha() == alpha and image.get_clim() == limits
    assert image.get_interpolation() == "nearest"
    assert " ".join(legend.get_texts()[0].get_text().split()) == ax._spacr_legend_entries[0]["label"]
    old_size, old_position = figure.get_size_inches().copy(), ax.get_position().bounds
    style.external_legend(figure, axes=[ax], pack=False, placement="below",
        maximum_width=maximum_width, column_width=8., theme_colors=style.resolve_plot_theme(dark))
    np.testing.assert_allclose(figure.get_size_inches(), old_size, rtol=0., atol=1e-8)
    np.testing.assert_allclose(ax.get_position().bounds, old_position, rtol=0., atol=1e-8)


def test_below_key_refuses_width_that_requires_resizing_the_data_panel():
    """An impossible canvas bound cannot silently scale or crop a source image."""
    figure = Figure(figsize=(8., 3.))
    ax = figure.add_axes([0., .1, 1., .8])
    pixels = np.arange(12).reshape(3, 4)
    image = ax.imshow(pixels, aspect="auto")
    ax.set_axis_off()
    style.annotate(ax, "Fixed eight-inch data panel.")
    before = figure.get_size_inches().copy()
    with pytest.raises(ValueError, match="physical size"):
        style.external_legend(figure, axes=[ax], pack=False, placement="below", maximum_width=170. / 25.4)
    np.testing.assert_array_equal(figure.get_size_inches(), before)
    np.testing.assert_array_equal(image.get_array(), pixels)


def test_explicit_reference_dashes_match_keys_through_normal_writer_and_appearance_replay(tmp_path):
    """Exact dash geometry survives writing and replay for thresholds and their keys.

    Matplotlib reports every tuple linestyle as ``--``, including a restored
    dotted tuple. The unscaled and rendered scaled patterns are the evidence,
    rather than that symbolic alias.
    """
    import json
    from spacr.plot import save_figure

    def dash_state(artist):
        """Freeze both exact patterns used by normal appearance capture and drawing."""
        return tuple((offset, None if pattern is None else tuple(pattern))
                     for offset, pattern in (artist._unscaled_dash_pattern, artist._dash_pattern))

    figure = Figure(figsize=(4., 3.))
    ax = _declared_panel(figure)
    used = style.styled_reference_line(ax, x=.25, label="Used threshold", linestyle="--", role="used")
    reference = style.styled_reference_line(ax, x=.75, label="Reference threshold", linestyle=":", role="reference")
    before = [line.get_xydata().copy() for line in (used, reference)]
    key = style.external_legend(figure, axes=[ax], pack=False)
    FigureCanvasAgg(figure).draw()
    handles = dict(zip((text.get_text() for text in key.get_texts()), key.legend_handles))
    lines = {"Used threshold": used, "Reference threshold": reference}
    patterns = {label: (dash_state(line), dash_state(handles[label])) for label, line in lines.items()}
    for line_patterns, key_patterns in patterns.values():
        assert line_patterns == key_patterns
    assert patterns["Used threshold"][0][0] != patterns["Reference threshold"][0][0]
    assert patterns["Used threshold"][0][1] != patterns["Reference threshold"][0][1]
    path = tmp_path / "reference-patterns.png"
    save_figure(figure, str(path), fmt="png", dpi=72, save_mode="screen", preferences=False)
    assert path.is_file()
    handles = dict(zip((text.get_text() for text in key.get_texts()), key.legend_handles))
    assert used.get_linestyle() == handles["Used threshold"].get_linestyle() == "--"
    assert reference.get_linestyle() == handles["Reference threshold"].get_linestyle() == ":"
    for label, line in lines.items():
        assert (dash_state(line), dash_state(handles[label])) == patterns[label]
    for line, values in zip((used, reference), before):
        np.testing.assert_array_equal(line.get_xydata(), values)
        assert to_rgba(line.get_color()) == to_rgba("black") and line.get_alpha() == .6
    for label in lines:
        assert to_rgba(handles[label].get_color()) == to_rgba("black") and handles[label].get_alpha() == 1.
    saved = json.loads(json.dumps(style._capture_appearance(figure)))
    captured_keys = dict(zip(saved["figure_legends"][0]["labels"], saved["figure_legends"][0]["handles"]))
    for label, line in lines.items():
        line_record = saved["axes"][0]["lines"][list(ax.lines).index(line)]
        for record, artist, expected in zip((line_record, captured_keys[label]),
                                          (line, handles[label]), patterns[label]):
            offset, pattern = record["linestyle"]
            assert (offset, tuple(pattern)) == expected[0]
            assert record["linewidth"] == artist.get_linewidth()
            assert dash_state(artist) == expected
    used.set_linestyle("-")
    reference.set_linestyle("-")
    assert not style._restore_appearance(figure, saved)
    handles = dict(zip((text.get_text() for text in figure.legends[0].get_texts()),
                       figure.legends[0].legend_handles))
    for label, line in lines.items():
        assert (dash_state(line), dash_state(handles[label])) == patterns[label]
        assert to_rgba(line.get_color()) == to_rgba("black") and line.get_alpha() == .6
        assert to_rgba(handles[label].get_color()) == to_rgba("black") and handles[label].get_alpha() == 1.
    assert dash_state(used)[0] != dash_state(reference)[0]
    assert dash_state(used)[1] != dash_state(reference)[1]
    for line, values in zip((used, reference), before):
        np.testing.assert_array_equal(line.get_xydata(), values)


@pytest.mark.parametrize("size", [100., 200., 500.])
def test_large_external_key_measures_without_allocating_a_full_canvas(monkeypatch, size):
    """Previously undrawn large image grids must not allocate multi-gigabyte raster buffers."""
    from matplotlib.backends.backend_agg import RendererAgg

    allocated = []
    original_init = RendererAgg.__init__

    def bounded_init(renderer, width, height, dpi):
        """Record real buffer sizes while refusing an accidental full-size allocation."""
        allocated.append((width, height))
        assert width <= 8 and height <= 8
        original_init(renderer, width, height, dpi)

    def refuse_canvas(*args, **kwargs):
        """Fail if metadata layout asks to render a complete source canvas."""
        raise AssertionError("External-key measurement must not rasterize the figure")

    monkeypatch.setattr(RendererAgg, "__init__", bounded_init)
    monkeypatch.setattr(FigureCanvasAgg, "get_renderer", refuse_canvas)
    monkeypatch.setattr(FigureCanvasAgg, "draw", refuse_canvas)
    figure = Figure(figsize=(size, size), dpi=100)
    canvas = figure.canvas
    ax = figure.add_subplot()
    pixels = np.arange(12).reshape(3, 4)
    image = ax.imshow(pixels, interpolation="nearest")
    ax.set_axis_off()
    ax.apply_aspect()
    before = ax.get_window_extent().size.copy() / figure.dpi
    style.annotate(ax, "A large source image grid; layout measurement only.")
    legend = style.external_legend(figure, axes=[ax], pack=False)
    assert legend is not None and allocated == [(8, 8)]
    assert figure.canvas is canvas
    assert figure.get_size_inches()[0] >= size and figure.get_size_inches()[1] >= size
    np.testing.assert_array_equal(image.get_array(), pixels)
    np.testing.assert_allclose(ax.get_window_extent().size / figure.dpi, before, rtol=0., atol=1e-8)
    assert figure._spacr_external_legend_layout["inside_canvas"]
    assert figure._spacr_external_legend_layout["outside_panels"]


def test_reference_backing_keeps_actual_black_core_and_source_values():
    """Dark-ground references gain contrast without a new line or changed alpha."""
    figure = Figure()
    ax = figure.add_subplot()
    figure.set_facecolor("black")
    ax.set_facecolor("black")
    line = style.reference_line(ax, y=.5, label="Fixed threshold")
    unrelated, = ax.plot([0., 1.], [0., 1.], color="black", alpha=.6)
    legend = ax.legend(facecolor="black", framealpha=1.)
    before = (line.get_xydata().copy(), line.get_alpha(), line.get_color(), line.get_linewidth())
    backed = style.ensure_reference_contrast(ax)
    assert line in backed and len(backed) == 2
    assert not unrelated.get_path_effects()
    np.testing.assert_array_equal(line.get_xydata(), before[0])
    assert (line.get_alpha(), line.get_color(), line.get_linewidth()) == before[1:]
    assert len(line.get_path_effects()) == 2
    style.ensure_reference_contrast(ax)
    assert len(line.get_path_effects()) == 2
    ax.set_facecolor("white")
    legend.get_frame().set_facecolor("white")
    assert style.ensure_reference_contrast(ax) == ()
    assert not line.get_path_effects()
    assert len(ax.lines) == 2


def test_image_report_css_has_shared_theme_font_and_no_pixel_transform():
    """Report chrome can change while CSS has no image alpha or colour filter."""
    css = style.image_report_css("screen", {"background": "#123456", "foreground": "#abcdef", "border": "#ffffff"})
    assert "#123456ff" in css and "#abcdefff" in css and "#ffffffff" in css
    assert "'Open Sans'" in css and "12pt" in css
    assert "opacity" not in css and "filter" not in css
    assert "grid-template-columns:repeat(auto-fit,minmax(320px,1fr))" in css


@pytest.mark.parametrize("float32_colours", [False, True])
def test_identity_backing_preserves_core_hidden_members_and_manual_effects(float32_colours):
    """Actual dark rendering reveals low-contrast identities without creating hidden points."""
    from matplotlib.patheffects import Normal

    figure = Figure(figsize=(3., 2.), dpi=100, facecolor="black")
    ax = figure.add_axes([0., 0., 1., 1.], facecolor="black")
    ax.set_axis_off()
    colour = "#0A77B8" if float32_colours else "#283868"
    points = ax.scatter([.3, 1., 1.7], [.5, .5, .5], s=100., color=colour,
                        alpha=np.array([0., .6, .35]))
    if float32_colours:
        colours = np.asarray([to_rgba(colour)] * 3, dtype=np.float32)
        points.set_facecolors(colours)
        points.set_edgecolors(colours)
        assert points.get_facecolors().dtype == points.get_edgecolors().dtype == np.float32
    hull, = ax.plot([.3, 1.7], [1.2, 1.2], color=colour, alpha=.6, linewidth=1.)
    manual, = ax.plot([.3, 1.7], [1.7, 1.7], color=colour, alpha=.6, linewidth=1.)
    effect = Normal()
    manual.set_path_effects([effect])
    ax.set_xlim(0., 2.)
    ax.set_ylim(0., 2.)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    before = np.asarray(canvas.buffer_rgba()).copy()
    offsets, faces, edges = points.get_offsets().copy(), points.get_facecolors().copy(), points.get_edgecolors().copy()
    alpha, sizes, widths = points.get_alpha().copy(), points.get_sizes().copy(), points.get_linewidths().copy()
    line = (hull.get_xydata().copy(), hull.get_color(), hull.get_alpha(), hull.get_linewidth())
    installed = style.ensure_identity_contrast(ax, artists=[points, hull, manual], colours=[colour])
    assert installed == (points, hull)
    assert manual.get_path_effects() == [effect]
    canvas.draw()
    after = np.asarray(canvas.buffer_rgba()).copy()

    def region(x, y, radius):
        """Read the same fixed pixel neighborhood in both actual Agg images."""
        px, py = ax.transData.transform((x, y))
        row, column = int(figure.bbox.height - py), int(px)
        return (slice(row - radius, row + radius + 1), slice(column - radius, column + radius + 1))

    np.testing.assert_array_equal(before[region(.3, .5, 10)], after[region(.3, .5, 10)])
    np.testing.assert_array_equal(before[region(1., .5, 2)], after[region(1., .5, 2)])
    np.testing.assert_array_equal(before[region(1., 1.7, 5)], after[region(1., 1.7, 5)])
    assert np.max(after[region(1., .5, 10)][:, :, :3]) > np.max(before[region(1., .5, 10)][:, :, :3])
    assert np.max(after[region(1., 1.2, 5)][:, :, :3]) > np.max(before[region(1., 1.2, 5)][:, :, :3])
    np.testing.assert_array_equal(points.get_offsets(), offsets)
    np.testing.assert_array_equal(points.get_facecolors(), faces)
    np.testing.assert_array_equal(points.get_edgecolors(), edges)
    np.testing.assert_array_equal(points.get_alpha(), alpha)
    np.testing.assert_array_equal(points.get_sizes(), sizes)
    np.testing.assert_array_equal(points.get_linewidths(), widths)
    np.testing.assert_array_equal(hull.get_xydata(), line[0])
    assert (hull.get_color(), hull.get_alpha(), hull.get_linewidth()) == line[1:]


def test_identity_backing_leaves_unaffected_collection_rendering_and_manual_choices_untouched():
    """Invisible, high-contrast and unbound colors retain their original collection path."""
    figure = Figure(facecolor="black")
    ax = figure.add_subplot(facecolor="black")
    hidden = ax.scatter([0.], [0.], color="#283868", alpha=0.)
    bright = ax.scatter([1.], [1.], color="white", alpha=1.)
    unbound = ax.scatter([2.], [2.], color="#000000", alpha=.6)
    assert style.ensure_identity_contrast(ax, artists=[hidden, bright, unbound],
                                          colours=["#283868", "white"]) == ()
    assert all(not artist.get_path_effects() for artist in (hidden, bright, unbound))
    hidden.set_alpha(.6)
    assert style.ensure_identity_contrast(ax, artists=[hidden], colours=["#283868"]) == (hidden,)
    hidden.set_alpha(0.)
    assert style.ensure_identity_contrast(ax, artists=[hidden], colours=["#283868"]) == ()
    assert not hidden.get_path_effects()


def test_identity_backing_figure_pickle_preserves_pixels_properties_and_axes_binding():
    """Whole-Figure transport retains dark identities, hidden members and restored axes."""
    import pickle

    figure = Figure(figsize=(3., 2.), dpi=100, facecolor="black")
    ax = figure.add_axes([0., 0., 1., 1.], facecolor="black")
    ax.set_axis_off()
    points = ax.scatter([.3, 1., 1.7], [.5, .5, .5], s=[80., 100., 120.],
                        color="#283868", alpha=np.array([0., .6, .35]))
    hull, = ax.plot([.3, 1.7], [1.2, 1.2], color="#283868", alpha=.6, linewidth=1.)
    ax.set_xlim(0., 2.)
    ax.set_ylim(0., 2.)
    assert style.ensure_identity_contrast(ax, artists=[points, hull], colours=["#283868"]) == (points, hull)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba()).copy()
    restored = pickle.loads(pickle.dumps(figure, protocol=pickle.HIGHEST_PROTOCOL))
    restored_ax = restored.axes[0]
    restored_points, restored_hull = restored_ax.collections[0], restored_ax.lines[0]
    assert restored_ax is not ax and restored_ax.figure is restored
    assert restored_points.axes is restored_hull.axes is restored_ax
    for getter in ("get_offsets", "get_facecolors", "get_edgecolors", "get_alpha",
                   "get_sizes", "get_linewidths"):
        np.testing.assert_array_equal(getattr(restored_points, getter)(), getattr(points, getter)())
    np.testing.assert_array_equal(restored_hull.get_xydata(), hull.get_xydata())
    assert (restored_hull.get_color(), restored_hull.get_alpha(), restored_hull.get_linewidth()) == (
        hull.get_color(), hull.get_alpha(), hull.get_linewidth())
    for artist in (restored_points, restored_hull):
        assert tuple(artist.get_path_effects()) == artist._spacr_identity_contrast_effects
    restored_canvas = FigureCanvasAgg(restored)
    restored_canvas.draw()
    np.testing.assert_array_equal(np.asarray(restored_canvas.buffer_rgba()), pixels)
    px, py = restored_ax.transData.transform((.3, .5))
    row, column = int(restored.bbox.height - py), int(px)
    assert not np.any(pixels[row - 10:row + 11, column - 10:column + 11, :3])
    ax.set_facecolor("white")
    figure.set_facecolor("white")
    restored_canvas.draw()
    np.testing.assert_array_equal(np.asarray(restored_canvas.buffer_rgba()), pixels)


def test_explicit_ci_artists_gain_backing_without_changing_intervals():
    """Caller-declared CI collections and summary ticks keep exact values and alpha."""
    figure = Figure()
    ax = figure.add_subplot()
    ax.set_facecolor("black")
    container = ax.errorbar([1., 2.], [2., 4.], yerr=[.3, .7], fmt="_", capsize=3,
                            color="black", ecolor="black", alpha=.6)
    artists = (container.lines[0], *container.lines[1], *container.lines[2])
    segments = [line.copy() for line in container.lines[2][0].get_segments()]
    values = container.lines[0].get_xydata().copy()
    colours = container.lines[2][0].get_colors().copy()
    assert len(style.ensure_reference_contrast(ax, artists=artists)) == len(artists)
    np.testing.assert_array_equal(container.lines[0].get_xydata(), values)
    np.testing.assert_array_equal(container.lines[2][0].get_colors(), colours)
    for before, after in zip(segments, container.lines[2][0].get_segments()):
        np.testing.assert_array_equal(before, after)
    assert all(artist.get_alpha() == .6 for artist in artists)
