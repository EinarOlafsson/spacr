"""A saved figure keeps its source data and user-visible edits."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from spacr.figures import bundle, style


def test_an_axes_source_and_ragged_groups_become_reusable_rows():
    figure = Figure()
    axes = figure.add_subplot()
    series = pd.Series([2, 5, 8], name="signal")

    bundle._register_figure_data(axes, lambda: series, y="signal", kind="histogram")

    frame, spec = bundle._figure_record(figure)
    assert frame["signal"].tolist() == [2, 5, 8]
    assert spec["kind"] == "hist"

    groups = bundle._as_frame({"control": [1, 2], "treated": [4]},
                              "group", "signal")
    assert groups.to_dict("list") == {
        "group": ["control", "control", "treated"],
        "signal": [1, 2, 4],
    }


def test_axes_grids_vectors_and_image_stacks_keep_their_original_shape():
    figure = Figure()
    axes = figure.subplots(1, 2)
    assert bundle._as_figure(axes) is figure

    matrix = np.arange(6).reshape(3, 2)
    assert bundle._as_frame(matrix, "", "").shape == (3, 2)
    assert bundle._as_frame(np.arange(3), "", "signal")["signal"].tolist() == [0, 1, 2]

    bundle._register_figure_data(figure, np.arange(24).reshape(3, 2, 4))
    assert figure._spacr_data is None
    assert figure._spacr_image[0].shape == (3, 2, 4)

    bundle._register_figure_data(figure, [matrix, matrix + 1])
    assert [array.shape for array in figure._spacr_image] == [(3, 2), (3, 2)]


def test_a_headless_figure_still_gets_a_pdf_and_a_png(monkeypatch):
    from spacr import plot

    def unavailable():
        raise RuntimeError("Preferences unavailable")

    monkeypatch.setattr(plot, "figure_output_preferences", unavailable)
    assert bundle._default_formats() == ["pdf", "png"]
    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("PNG", None))
    assert bundle._default_formats() == ["png"]


def test_a_redraw_recipe_keeps_the_visible_view_and_group_order():
    figure = Figure()
    axes = figure.add_subplot()
    axes.set(title="Edited title", xlabel="new x", ylabel="new y",
             xlim=(2, 8), ylim=(-1, 6))
    figure._spacr_replot = {
        "df": pd.DataFrame({"group": ["a", "b"], "value": [1, 3]}),
        "grouping_column": "group", "data_column": "value",
        "graph_type": "jitter_box", "order": ["b", "a"],
    }

    frame, spec = bundle._figure_record(figure)
    spec["keep_limits"] = True
    visible = bundle._capture_view(figure, spec)

    assert frame["value"].tolist() == [1, 3]
    assert visible["kind"] == "box_strip"
    assert visible["order"] == ["b", "a"]
    assert visible["title"] == "Edited title"
    assert visible["xlabel"] == "new x"
    assert visible["xlim"] == [2, 8]
    assert visible["ylim"] == [-1, 6]


def test_an_image_zip_retains_pixels_when_tiff_writing_fails(tmp_path,
                                                              monkeypatch):
    from spacr import tiff_io

    def unavailable(*_args, **_kwargs):
        raise OSError("TIFF backend unavailable")

    monkeypatch.setattr(tiff_io, "write_tiff", unavailable)
    pixels = np.arange(16, dtype=np.uint16).reshape(4, 4)
    figure = Figure()
    bundle._register_figure_data(figure, pixels, kind="image")

    path = bundle._save_zip(figure, str(tmp_path / "image.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["arrays"] == [{
            "file": "image_0.npy", "shape": [4, 4], "dtype": "uint16",
        }]
        np.testing.assert_array_equal(
            np.load(io.BytesIO(archive.read("image_0.npy")), allow_pickle=False),
            pixels,
        )


def test_live_style_applies_background_fonts_markers_and_image_map():
    figure = Figure()
    axes = figure.add_subplot()
    figure.suptitle("Overview")
    line, = axes.plot([0, 1], [0, 1], marker="o", label="series")
    points = axes.scatter([0.5], [0.5], s=[10])
    image = axes.imshow(np.arange(4).reshape(2, 2), cmap="viridis")
    axes.legend()

    style._restyle(figure, {
        "background": "#dedede", "font_family": "DejaVu Sans",
        "title_size": 18, "tick_size": 9, "spines": "left_bottom",
        "spine_width": 2, "despine_offset": 5,
        "line_width": 3, "marker_size": 36, "colormap": "magma",
    })

    assert figure.get_facecolor()[:3] == (222 / 255,) * 3
    assert figure._suptitle.get_fontsize() == 18
    assert axes.get_facecolor()[:3] == (222 / 255,) * 3
    assert line.get_linewidth() == 3
    assert line.get_markersize() == 6
    assert points.get_sizes().tolist() == [36]
    assert image.get_cmap().name == "magma"
    assert not axes.spines["top"].get_visible()
    assert axes.spines["left"].get_linewidth() == 2
    assert axes.spines["left"].get_position() == ("outward", 5)
