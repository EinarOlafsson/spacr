"""Extra QC renderers retain supplied summaries, source pixels and threshold calculations."""

import ast
import base64
from copy import deepcopy
import io
from html import unescape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from spacr import image_quality, seg_qc, sequencing
from spacr.figures import style


def nested_threshold_function():
    """Load the exact nested production function without invoking CSV or plate orchestration."""
    tree = ast.parse(Path(sequencing.__file__).read_text())
    outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "graph_sequencing_stats")
    function = next(node for node in outer.body if isinstance(node, ast.FunctionDef)
                    and node.name == "find_and_visualize_fraction_threshold")
    namespace = vars(sequencing).copy()
    exec(compile(ast.Module(body=[function], type_ignores=[]), sequencing.__file__, "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("target", ["print", "screen"])
def test_nested_threshold_render_preserves_the_actual_grid_and_selected_value(target, monkeypatch, tmp_path):
    """Two supplied guide fractions make the first one-guide threshold exactly grid index fifty."""
    from spacr import plot

    captured = []
    monkeypatch.setattr(plot, "save_figure", lambda figure, path, **kwargs: captured.append(figure) or path)
    monkeypatch.setattr(sequencing, "theme_target", lambda: target)
    monkeypatch.setattr(sequencing, "_show_if_interactive", lambda: None)
    frame = pd.DataFrame({"plateID": ["p", "p"], "rowID": ["r1", "r1"],
                          "columnID": ["c1", "c1"], "grna": ["a", "b"],
                          "fraction": [.05, .95]})
    frame.attrs["evidence"] = "supplied fractions; no decoding or model"
    original = frame.copy(deep=True)
    with style.pipeline_figures():
        result = nested_threshold_function()(frame, target_unique_count=1, dst=str(tmp_path))
    grid = np.linspace(.001, .99, 1000)
    assert result == grid[50]
    figure = captured[0]
    axis = figure.axes[0]
    assert figure.get_facecolor() == axis.get_facecolor()
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), grid)
    assert np.all(axis.lines[0].get_ydata()[:50] == 2)
    assert np.all(axis.lines[0].get_ydata()[50:959] == 1)
    assert np.all(np.isnan(axis.lines[0].get_ydata()[959:]))
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), [grid[50], grid[50]])
    np.testing.assert_array_equal(axis.lines[2].get_ydata(), [1, 1])
    assert axis.get_xlim() == (0., .1) and axis.get_ylim() == (0., 20.)
    assert axis._spacr_style_spec.target == target
    assert figure._spacr_external_legend is not None
    assert not axis.texts and not axis.get_title()
    diagnostics = style.finalize_figure(figure)
    assert not any("texts overlap" in problem for problem in diagnostics.problems)
    renderer = figure.canvas.get_renderer()
    zero_x = next(tick.label1 for tick in axis.xaxis.get_major_ticks() if tick.get_loc() == 0.)
    zero_y = next(tick.label1 for tick in axis.yaxis.get_major_ticks() if tick.get_loc() == 0.)
    assert not zero_x.get_window_extent(renderer).overlaps(zero_y.get_window_extent(renderer))
    pd.testing.assert_frame_equal(frame, original)
    assert frame.attrs == original.attrs
    plt.close(figure)


@pytest.mark.parametrize("target", ["print", "screen"])
def test_robustness_heatmap_retains_nonfinite_sentinel_and_reference_summary(target, monkeypatch):
    """The existing shown array and scale remain exact while values move out of seven-point cells."""
    monkeypatch.setattr(style, "theme_target", lambda: target)
    summary = pd.DataFrame({"parameter": ["baseline", "flow_threshold", "diameter"],
        "value": ["run", "0.8", "40"], "fragile": [False, True, True],
        "object_count_change": [0., -.5, np.nan], "median_area_change": [0., .1, np.inf],
        "mean_intensity_change": [0., -.2, -.3], "matched_fraction": [1., .75, .8]})
    summary.attrs["settings"] = {"tolerance": .2}
    original = summary.copy(deep=True)
    with style.pipeline_figures():
        figure = seg_qc._robustness_figure(summary, .2, "cell")
    axis = figure.axes[0]
    assert figure.get_facecolor() == axis.get_facecolor()
    colourbar_axes = style._axis_colourbars(axis)
    assert len(colourbar_axes) == 1
    colourbar = colourbar_axes[0]._colorbar
    assert colourbar.ax.get_facecolor() == axis.get_facecolor()
    from matplotlib.colors import to_rgba
    for text in (colourbar.ax.yaxis.label, *colourbar.ax.get_yticklabels()):
        assert to_rgba(text.get_color()) == to_rgba(axis.xaxis.label.get_color())
        assert text.get_alpha() == 1.
    expected = np.array([[0., 0., 0., 0.], [.5, .1, .2, .25], [.4, .4, .3, 1.-.8]])
    np.testing.assert_array_equal(axis.images[0].get_array(), expected)
    assert axis.images[0].get_clim() == (0., .4)
    assert axis.images[0].get_alpha() == .6
    assert axis.images[0].cmap.name == "Blues"
    assert not axis.texts
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["baseline run", "! flow_threshold 0.8", "! diameter 40"]
    assert figure._spacr_external_legend is not None
    assert "Nonfinite" in " ".join(text.get_text() for text in figure._spacr_external_legend.get_texts())
    pd.testing.assert_frame_equal(summary, original)
    assert summary.attrs == original.attrs
    plt.close(figure)


def test_image_gallery_retains_raw_files_and_exact_thumbnail_bytes(tmp_path):
    """Shared HTML styling must not alter the original channel/projection/percentile/Pillow path."""
    first = np.arange(16*20, dtype=np.uint16).reshape(16, 20)
    second = np.flip(first, axis=0).copy()
    paths = [tmp_path / "plain.npy", tmp_path / "flagged.npy"]
    for path, values in zip(paths, (first, second)):
        np.save(path, values)
    originals = {path: path.read_bytes() for path in paths}
    fields = [{"field": path.name, "status": "flagged" if index else "accepted",
               "reasons": ["focus_below_threshold"] if index else [],
               "channels": [{"channel": 0, "reasons": ["focus_below_threshold"] if index else [],
                             "focus_variance": 1.25, "saturation_fraction": .01}]}
              for index, path in enumerate(paths)]
    before = deepcopy(fields)
    image_quality._write_gallery(tmp_path / "gallery", fields, paths, None)
    html = (tmp_path / "gallery.html").read_text()
    encoded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    assert len(encoded) == 2
    metadata = [json.loads(unescape(value)) for value in re.findall(r'data-display="([^\"]+)"', html)]
    assert len(metadata) == 2
    assert "Open Sans" in html
    for item in metadata:
        assert item["gamma"] == 1. and item["percentiles"] == [1, 99]
        assert item["finite_sample_count"] == first.size
        assert item["range_source"] == "finite_sample_percentiles"
        assert item["display_dtype"] == "uint8" and item["maximum_projection"] is False
        assert item["stride"] == 1
    for payload, original in zip(encoded, (second, first)):
        image = np.asarray(original, dtype=np.float32)
        low, high = np.percentile(image, (1, 99))
        scaled = np.nan_to_num((image-low)/max(float(high-low), 1e-12), nan=0., posinf=1., neginf=0.)
        thumbnail = Image.fromarray((scaled.clip(0, 1)*255).astype(np.uint8))
        thumbnail.thumbnail((256, 256))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="PNG")
        assert base64.b64decode(payload) == buffer.getvalue()
    assert fields == before
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert html.index("<b>flagged.npy") < html.index("<b>plain.npy")


@pytest.mark.parametrize("values", [[np.nan, np.nan, np.nan], [np.nan, np.inf, -np.inf]])
def test_image_gallery_marks_nonfinite_fallback_without_changing_thumbnail_bytes(tmp_path, values):
    """An empty finite sample must disclose its original 0–1 fallback, not measured percentiles."""
    original = np.asarray([values, values], dtype=np.float32)
    path = tmp_path / "nonfinite.npy"
    np.save(path, original)
    original_bytes = path.read_bytes()
    fields = [{"field": path.name, "status": "flagged", "reasons": ["nonfinite"],
               "channels": [{"channel": 0, "reasons": ["nonfinite"],
                             "focus_variance": 0., "saturation_fraction": None}]}]
    before = deepcopy(fields)
    image_quality._write_gallery(tmp_path / "fallback", fields, [path], None)
    html = (tmp_path / "fallback.html").read_text()
    encoded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    metadata = [json.loads(unescape(value)) for value in re.findall(r'data-display="([^\"]+)"', html)]
    assert len(encoded) == len(metadata) == 1
    assert metadata[0]["display_range"] == [0., 1.]
    assert metadata[0]["percentiles"] is None
    assert metadata[0]["finite_sample_count"] == 0
    assert metadata[0]["range_source"] == "no_finite_sample_fallback"
    assert "no finite values; no percentiles were measured" in html
    scaled = np.nan_to_num((original - 0.) / max(float(1. - 0.), 1e-12),
                           nan=0., posinf=1., neginf=0.)
    thumbnail = Image.fromarray((scaled.clip(0, 1) * 255).astype(np.uint8))
    thumbnail.thumbnail((256, 256))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="PNG")
    assert base64.b64decode(encoded[0]) == buffer.getvalue()
    assert path.read_bytes() == original_bytes
    assert fields == before
