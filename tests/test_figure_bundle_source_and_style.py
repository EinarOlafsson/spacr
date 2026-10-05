"""A saved figure keeps its source data and user-visible edits."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import tifffile
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


def test_recreated_histogram_keeps_the_saved_zoom():
    frame = pd.DataFrame({"signal": [1, 2, 2, 3, 4]})
    figure = Figure()

    axes = bundle._draw(figure, frame, {
        "kind": "hist", "x": "signal", "xlim": [0.5, 4.5],
        "ylim": [0, 6],
    })

    assert axes.get_xlim() == (0.5, 4.5)
    assert axes.get_ylim() == (0, 6)
    assert sum(patch.get_height() for patch in axes.patches) == len(frame)


def test_stale_group_annotation_is_omitted_when_recreating_filtered_data():
    frame = pd.DataFrame({
        "group": ["control", "control", "treated", "treated"],
        "value": [1, 2, 3, 4],
    })
    figure = Figure()

    axes = bundle._draw(figure, frame, {
        "kind": "box", "x": "group", "y": "value",
        "annotations": [{"pair": ["control", "removed"], "label": "*"}],
    })

    assert not any(line.get_gid() == "spacr-stats" for line in axes.lines)
    assert axes.get_ylim()[1] < 5


def test_graph_zip_keeps_data_and_recipe_when_one_renderer_fails(
        tmp_path, monkeypatch):
    from spacr import plot

    figure = Figure()
    figure.add_subplot().plot([1, 2, 3], [2, 4, 6])
    bundle._register_figure_data(
        figure, pd.DataFrame({"time": [1, 2, 3], "signal": [2, 4, 6]}),
        x="time", y="signal", kind="line",
    )
    render = plot.save_figure

    def one_format_unavailable(fig, path, *, fmt=None, **kwargs):
        if fmt == "pdf":
            raise OSError("PDF output is unavailable")
        return render(fig, path, fmt=fmt, **kwargs)

    monkeypatch.setattr(plot, "save_figure", one_format_unavailable)
    path = bundle._save_zip(
        figure, str(tmp_path / "partial"), formats=["pdf", "png"],
        name="partial",
    )

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "partial.pdf" not in names
        assert {"partial.png", "data.csv", "statistics.csv", "spec.json",
                "recreate_figure.py"} <= names
        assert pd.read_csv(io.BytesIO(archive.read("data.csv")))["signal"].tolist() \
            == [2, 4, 6]


def test_image_zip_keeps_pixels_and_metadata_when_one_renderer_fails(
        tmp_path, monkeypatch):
    from spacr import plot

    pixels = np.arange(16, dtype=np.uint16).reshape(4, 4)
    figure = Figure()
    bundle._register_figure_data(figure, pixels, kind="image")
    render = plot.save_figure

    def one_format_unavailable(fig, path, *, fmt=None, **kwargs):
        if fmt == "pdf":
            raise OSError("PDF output is unavailable")
        return render(fig, path, fmt=fmt, **kwargs)

    monkeypatch.setattr(plot, "save_figure", one_format_unavailable)
    path = bundle._save_zip(
        figure, str(tmp_path / "image-partial"), formats=["pdf", "png"],
        name="image-partial",
    )

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "image-partial.pdf" not in names
        assert {"image-partial.png", "image_0.tif", "metadata.json"} <= names
        assert "statistics.csv" not in names
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["arrays"][0]["shape"] == [4, 4]
        np.testing.assert_array_equal(
            tifffile.imread(io.BytesIO(archive.read("image_0.tif"))), pixels,
        )


def test_group_only_figure_zip_exports_the_values_used_for_statistics(tmp_path):
    groups = {
        "control": [1.0, 1.3, 1.7, 2.0, 2.2, 2.5],
        "treated": [2.1, 2.4, 2.7, 3.0, 3.2, 3.5],
    }
    figure = Figure()
    figure.add_subplot().boxplot(list(groups.values()))
    figure._spacr_groups = groups

    path = bundle._save_zip(
        figure, str(tmp_path / "groups.zip"), formats=["png"],
    )

    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read("data.csv")))
        spec = json.loads(archive.read("spec.json"))
        statistics = pd.read_csv(io.BytesIO(archive.read("statistics.csv")))
        archive.extractall(tmp_path / "unpacked")
    assert frame.groupby("group")["value"].apply(list).to_dict() == groups
    assert (spec["x"], spec["y"], spec["kind"]) == ("group", "value", "box")
    assert (statistics["test_stage"] == "pairwise").any()
    recreated = subprocess.run(
        [sys.executable, "recreate_figure.py"], cwd=tmp_path / "unpacked",
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, MPLBACKEND="Agg"),
    )
    assert recreated.returncode == 0, recreated.stderr[-1000:]
    assert (tmp_path / "unpacked" / "recreated.png").is_file()
