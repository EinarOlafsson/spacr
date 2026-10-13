"""The Preferences figure settings reach the figures spaCR produces.

A representative producer from each figure module is drawn twice: once with an
untouched preference store and once with the tick size, resolution and palette
changed. Every figure is then written through `spacr.plot.save_figure`, the
one writer, and the change must be visible in the figure and the file: the
tick labels at the chosen size, the PNG stamped with the chosen DPI, and the
coloured series drawn from the chosen palette.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("PySide6")

import matplotlib.pyplot as plt  # noqa: E402

TICK = 21.0
DPI = 137
PALETTE = "bright"


@pytest.fixture()
def prefs(monkeypatch):
    """Drive the preference stores directly, as the dialog would save them."""
    from spacr.qt import preferences

    state = {"general": {}, "per_graph": {}}
    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: dict(state["general"]))
    monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                        lambda: {k: dict(v)
                                 for k, v in state["per_graph"].items()})
    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 100)
    yield state
    plt.close("all")


def _grouped_frame():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "grp": np.repeat(["a", "b", "c"], 15),
        "v1": rng.normal(0, 1, 45),
        "v2": rng.normal(1, 1, 45),
    })


def _volcano():
    from spacr.figures import build_panel

    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "gene": [f"g{i}" for i in range(40)],
        "coefficient": rng.normal(0, 0.4, 40),
        "p_value": rng.uniform(1e-6, 1, 40),
    })
    return build_panel("volcano", frame, target="print")[0]


def _grouped_graph():
    from spacr.plot import spacrGraph

    graph = spacrGraph(_grouped_frame(), "grp", ["v1", "v2"],
                       graph_type="jitter", representation="object")
    graph.create_plot()
    return graph.get_figure()


def _histogram():
    from spacr.plot import plot_histogram

    before = set(plt.get_fignums())
    plot_histogram(_grouped_frame(), "v1")
    new = sorted(set(plt.get_fignums()) - before)
    return plt.figure(new[-1])


def _residuals():
    from unittest import mock

    from spacr import regression_diagnostics

    rng = np.random.default_rng(5)
    fitted = rng.normal(0, 1, 60)
    with mock.patch.object(regression_diagnostics, "_finish",
                           lambda fig, *_a, **_k: fig):
        figure, _report = regression_diagnostics.plot_residual_diagnostics(
            fitted + rng.normal(0, 0.3, 60), fitted)
    return figure


def _unstyled_module_figure():
    """A figure drawn the way older modules draw: no style context at all."""
    figure, ax = plt.subplots()
    for offset in range(3):
        ax.plot([0, 1, 2], [offset, offset + 1, offset + 2])
    ax.set_title("legacy")
    return figure


PRODUCERS = {
    "figures.build_panel volcano": (_volcano, False),
    "plot.spacrGraph jitter": (_grouped_graph, True),
    "plot.plot_histogram": (_histogram, False),
    "regression_diagnostics.plot_residual_diagnostics": (_residuals, False),
    "an unstyled module figure": (_unstyled_module_figure, True),
}


def _tick_sizes(figure):
    return {label.get_fontsize() for ax in figure.get_axes()
            for label in ax.get_xticklabels() + ax.get_yticklabels()}


def _colours(figure):
    from matplotlib.colors import to_hex

    found = set()
    for ax in figure.get_axes():
        for line in ax.get_lines():
            found.add(to_hex(line.get_color()).lower())
        for patch in ax.patches:
            found.add(to_hex(patch.get_facecolor()).lower())
        for collection in ax.collections:
            for row in np.atleast_2d(collection.get_facecolor()):
                if len(row) == 4:
                    found.add(to_hex(row).lower())
    return found


def _saved_dpi(path):
    from PIL import Image

    with Image.open(path) as image:
        return round(float(image.info["dpi"][0]))


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_a_changed_preference_shows_up_in_the_figure(name, prefs, tmp_path):
    from spacr.figure_style import palette_colours
    from spacr.plot import save_figure

    producer, coloured = PRODUCERS[name]
    untouched = producer()
    plain_path = save_figure(untouched, tmp_path / "plain.png")
    assert TICK not in _tick_sizes(untouched)
    assert _saved_dpi(plain_path) == 100

    prefs["general"].update({"tick_size": TICK, "dpi": DPI,
                             "palette": PALETTE})
    figure = producer()
    path = save_figure(figure, tmp_path / "chosen.png")

    assert _tick_sizes(figure) == {TICK}, name
    assert _saved_dpi(path) == DPI, name
    if coloured:
        chosen = {c.lower() for c in palette_colours(PALETTE)}
        assert _colours(figure) & chosen, name


def test_the_other_settings_reach_an_unstyled_figure(prefs):
    from spacr.figures.style import _apply_user_style

    prefs["general"].update({"title_size": 19.0, "label_size": 15.0,
                             "legend_size": 7.0, "line_width": 3.0,
                             "grid": False, "spines": "none"})
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="one")
    ax.set_title("t")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True)
    applied = _apply_user_style(figure)

    assert applied["title_size"] == 19.0
    assert ax.title.get_fontsize() == 19.0
    assert ax.xaxis.label.get_fontsize() == 15.0
    assert ax.get_legend().get_texts()[0].get_fontsize() == 7.0
    assert ax.get_lines()[0].get_linewidth() == 3.0
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())


def test_a_figure_is_styled_once_so_a_restyle_survives_the_save(prefs):
    from spacr.figures.style import _apply_user_style

    prefs["general"]["tick_size"] = TICK
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    _apply_user_style(figure)
    ax.tick_params(labelsize=5)

    assert _apply_user_style(figure) == {}
    assert _tick_sizes(figure) == {5}


def test_the_new_settings_reach_the_rc_house_style(prefs):
    from spacr.figures.style import rc

    prefs["general"].update({"legend_size": 6.5, "colormap": "cividis",
                             "error_capsize": 2.0, "vector_text": False,
                             "page_shape": "custom", "figure_width": 3.0,
                             "figure_height": 2.0})
    params = rc("print")

    assert params["legend.fontsize"] == 6.5
    assert params["image.cmap"] == "cividis"
    assert params["errorbar.capsize"] == 2.0
    assert params["pdf.fonttype"] == 3
    assert list(params["figure.figsize"]) == [3.0, 2.0]


def test_the_grouped_graph_follows_the_error_bar_and_overlay_settings(prefs):
    from spacr.plot import spacrGraph

    prefs["general"].update({"error_bars": "none", "point_overlay": False})
    graph = spacrGraph(_grouped_frame(), "grp", "v1", graph_type="bar",
                       representation="object")
    graph.create_plot()
    from matplotlib.container import ErrorbarContainer

    assert not [c for c in graph.get_figure().get_axes()[0].containers
                if isinstance(c, ErrorbarContainer)]

    graph = spacrGraph(_grouped_frame(), "grp", "v1",
                       graph_type="jitter_box", representation="object")
    graph.create_plot()
    assert not graph.get_figure().get_axes()[0].collections


def test_a_second_format_is_written_beside_the_first(prefs, tmp_path):
    from spacr.plot import save_figure

    prefs["general"].update({"format": "svg", "also_save": "png"})
    path = save_figure(_unstyled_module_figure(), tmp_path / "both")

    assert path.endswith(".svg")
    assert os.path.exists(path[:-4] + ".png")
    assert "<text" in open(path, encoding="utf-8").read()


@pytest.mark.parametrize("path, hook", [
    ("qt/widgets/figure_queue.py", "_apply_user_style(fig)"),
    ("qt/widgets/umap_explorer.py", "_apply_user_style(self._figure"),
    ("qt/widgets/graph_builder.py", "_apply_user_style(self._figure"),
    ("qt/screens/train_compare.py", "_apply_user_style(self._figure"),
    ("plot.py", "_apply_user_style(fig)"),
])
def test_every_embedded_canvas_and_the_writer_apply_the_settings(path, hook):
    import pathlib

    import spacr

    text = (pathlib.Path(spacr.__file__).parent / path).read_text(
        encoding="utf-8")
    assert hook in text, f"{path} shows or writes figures without the settings"
