"""Table-driven tests of how a figure bundle rebuilds and records a figure.

``spacr.figures.bundle`` stores a figure as a recipe beside the rows behind it,
so the standalone script can redraw the same picture without Qt or torch.  One
general test per unit walks a case table with pytest's ``subtests`` fixture:
``_recipe_frame`` (the exact rows a recipe draws: object / well / plate
representations, outlier removal, log axes, plate heatmap aggregation, melted
measurements and the refusals), ``_draw`` (the exact artists each recipe kind
puts on the axes: grouped lines, training curves, scatter, histogram, QQ,
Lorenz, Venn and regression panels) and ``save`` (what a bundle folder holds
when the renderer fails, when the folder exists, and when a setting is not
JSON).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import to_rgb, to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import bundle  # noqa: E402


@pytest.fixture(autouse=True)
def _figures_closed():
    """Close whatever a case drew through pyplot."""
    yield
    plt.close("all")


def _grouped(**overrides):
    """A grouped-line recipe as the GUI records it."""
    recipe = {"grouping_column": "condition", "columns": ["time", "signal"],
              "representation": "object", "summary_func": "mean",
              "order": ["b", "a"], "mode": "line", "log_x": False,
              "log_y": False, "palette": ["#112233", "#445566"]}
    recipe.update(overrides)
    return recipe


def _frame():
    """Two conditions on two plates, plus one row with no time."""
    return pd.DataFrame({
        "condition": ["a", "a", "a", "a", "b", "b", "b", "b", "a"],
        "time": [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0, np.nan],
        "signal": [2.0, 4.0, 6.0, 8.0, 10.0, 30.0, 100.0, 300.0, 1.0],
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2", "p1_r1_c2",
                "p2_r1_c1", "p2_r1_c1", "p2_r1_c2", "p2_r1_c2", "p1_r2_c1"],
    })


RAW_TIME = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0]
RAW_SIGNAL = [2.0, 4.0, 6.0, 8.0, 10.0, 30.0, 100.0, 300.0]
RAW_ROWS = {"condition": ["a"] * 4 + ["b"] * 4, "time": RAW_TIME,
            "signal": RAW_SIGNAL,
            "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2", "p1_r1_c2",
                    "p2_r1_c1", "p2_r1_c1", "p2_r1_c2", "p2_r1_c2"]}


def _plate_spec(grouping):
    """A two-by-two plate heatmap recipe aggregating with ``grouping``."""
    return {"kind": "plate_heatmap", "plate": {
        "coordinates": {"p1_r1_c1": [1, 1], "p1_r2_c2": [2, 2]},
        "grouping": grouping, "variable": "value", "shape": [2, 2]}}


PLATE_ROWS = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c2", "other"],
                           "value": [1.0, 3.0, 5.0, 100.0]})
NAN = float("nan")
OUTLIERS = pd.DataFrame({
    "condition": ["a"] * 6 + ["b"] * 6,
    "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
    "signal": [1.0, 2.0, 2.0, 3.0, 2.0, 50.0, 5.0, 5.0, 6.0, 6.0, 5.0, -40.0],
})
MEASURES = pd.DataFrame({"id": [1, 2], "cell_area": [3.0, 4.0],
                         "nucleus_area": [5.0, 6.0]})


def _melt(labels=None):
    spec = {"var_name": "measurement", "value_name": "value",
            "id_vars": ["id"], "columns": ["cell_area", "nucleus_area"]}
    if labels is not None:
        spec["labels"] = labels
    return {"melt": spec}


# (case id, rows, recipe, exact rows drawn as lists, ordered categories)
RECIPE_FRAMES = [
    ("object rows drop the incomplete one and order the groups", _frame(),
     {"grouped_line": _grouped()}, RAW_ROWS, ("condition", ["b", "a"])),
    ("well rows are one summary per well", _frame(),
     {"grouped_line": _grouped(representation="well", summary_func="median")},
     {"prc": ["p1_r1_c1", "p1_r1_c2", "p2_r1_c1", "p2_r1_c2"],
      "condition": ["a", "a", "b", "b"], "time": [1.0, 10.0, 1.0, 10.0],
      "signal": [3.0, 7.0, 20.0, 200.0]}, ("condition", ["b", "a"])),
    ("plate rows split prc and summarise per plate", _frame(),
     {"grouped_line": _grouped(representation="plate")},
     {"plateID": ["p1", "p2"], "condition": ["a", "b"], "time": [5.5, 5.5],
      "signal": [5.0, 110.0]}, ("condition", ["b", "a"])),
    ("plate rows grouped by plate use the existing plate column",
     _frame().assign(plateID=["x"] * 4 + ["y"] * 5),
     {"grouped_line": _grouped(grouping_column="plateID",
                               representation="plate", order=["y", "x"])},
     {"plateID": ["x", "y"], "time": [5.5, 5.5], "signal": [5.0, 110.0]},
     ("plateID", ["y", "x"])),
    ("IQR outliers are removed per group and column", OUTLIERS,
     {"grouped_line": _grouped(order=["a", "b"], remove_outliers=True)},
     {"condition": ["a"] * 5 + ["b"] * 5, "time": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
      "signal": [1.0, 2.0, 2.0, 3.0, 2.0, 5.0, 5.0, 6.0, 6.0, 5.0]},
     ("condition", ["a", "b"])),
    ("log y transforms only the y column", _frame().dropna(),
     {"grouped_line": _grouped(log_y=True)},
     dict(RAW_ROWS, signal=list(np.log10(RAW_SIGNAL))),
     ("condition", ["b", "a"])),
    ("log x transforms only the x column", _frame().dropna(),
     {"grouped_line": _grouped(log_x=True)},
     dict(RAW_ROWS, time=list(np.log10(RAW_TIME))),
     ("condition", ["b", "a"])),
    ("group mode ignores both log flags", _frame().dropna(),
     {"grouped_line": _grouped(mode="groups", log_x=True, log_y=True)},
     RAW_ROWS, ("condition", ["b", "a"])),
    ("a plate heatmap takes the mean of each well", PLATE_ROWS,
     _plate_spec("mean"), {1: [2.0, NAN], 2: [NAN, 5.0]}, None),
    ("a plate heatmap takes the sum of each well", PLATE_ROWS,
     _plate_spec("sum"), {1: [4.0, NAN], 2: [NAN, 5.0]}, None),
    ("melted measurements take their display labels", MEASURES,
     _melt({"cell_area": "Cell"}),
     {"id": [1, 2, 1, 2],
      "measurement": ["Cell", "Cell", "nucleus_area", "nucleus_area"],
      "value": [3.0, 4.0, 5.0, 6.0]}, None),
    ("melted measurements keep their names without labels", MEASURES,
     _melt(), {"id": [1, 2, 1, 2],
               "measurement": ["cell_area", "cell_area", "nucleus_area",
                               "nucleus_area"],
               "value": [3.0, 4.0, 5.0, 6.0]}, None),
]


def _as_lists(frame):
    return {column: frame[column].tolist() for column in frame.columns}


def test_recipe_frame_rebuilds_the_exact_rows_a_recipe_draws(subtests):
    """Every recipe resolves to the same rows and never edits its input."""
    for case, rows, spec, expected, ordered in RECIPE_FRAMES:
        with subtests.test(msg=case):
            before = rows.copy()
            data = bundle._recipe_frame(rows, spec)
            got = _as_lists(data)
            assert list(got) == list(expected)
            for column, values in expected.items():
                assert got[column] == pytest.approx(values, nan_ok=True)
            if ordered is not None:
                column, categories = ordered
                assert list(data[column].cat.categories) == categories
                assert data[column].cat.ordered
            pd.testing.assert_frame_equal(rows, before)

    refusals = [
        ("an unknown representation", _frame(),
         {"grouped_line": _grouped(representation="field")},
         "Invalid grouped line representation"),
        ("an unknown plate aggregation",
         pd.DataFrame({"prc": ["p1_r1_c1"], "value": [1.0]}),
         _plate_spec("median"), "Invalid plate aggregation"),
    ]
    for case, rows, spec, message in refusals:
        with subtests.test(msg=case):
            with pytest.raises(ValueError, match=message):
                bundle._recipe_frame(rows, spec)


# ---------------------------------------------------------------------------
# _draw
# ---------------------------------------------------------------------------

def _line_spec(recipe, **spec):
    return {"kind": "line", "x": "time", "y": "signal",
            "grouped_line": recipe, **spec}


def _named(ax):
    return [line for line in ax.lines if not line.get_label().startswith("_")]


def _band_colours_and_edges(ax, alpha=None):
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)
             and (alpha is None or c.get_alpha() == pytest.approx(alpha))]
    return [(tuple(np.round(b.get_facecolor()[0][:3], 6)),
             float(b.get_paths()[0].vertices[:, 1].min()),
             float(b.get_paths()[0].vertices[:, 1].max())) for b in bands]


def test_draw_grouped_line_recipes(subtests):
    """Groups, SD bands, per-group means and plain lines draw exact data."""
    with subtests.test(msg="groups mode draws error bars in recorded order"):
        frame = pd.DataFrame({"condition": ["a", "a", "b", "b"],
                              "signal": [1.0, 3.0, 10.0, 10.0]})
        ax = bundle._draw(Figure(), frame, _line_spec(
            _grouped(mode="groups", columns=["signal"]), x="condition"))
        line = ax.lines[0]
        assert list(line.get_xdata()) == [0, 1]
        assert list(line.get_ydata()) == pytest.approx([10.0, 2.0])
        assert line.get_color() == "#112233"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["b", "a"]

    with subtests.test(msg="std_area draws the mean and a one-SD band"):
        frame = pd.DataFrame({"condition": ["a"] * 4,
                              "time": [1.0, 1.0, 2.0, 2.0],
                              "signal": [1.0, 3.0, 4.0, 8.0]})
        ax = bundle._draw(Figure(), frame, _line_spec(_grouped(
            mode="std_area", order=["a"], band_color="#aa0000",
            band_width=2.5)))
        line = ax.lines[0]
        assert line.get_label() == "mean_signal"
        assert list(line.get_ydata()) == pytest.approx([2.0, 6.0])
        assert line.get_linewidth() == pytest.approx(2.5)
        sd_first, sd_last = np.std([1.0, 3.0], ddof=1), np.std([4.0, 8.0],
                                                               ddof=1)
        assert _band_colours_and_edges(ax, alpha=0.25) == [(
            tuple(np.round(to_rgb("#aa0000"), 6)),
            pytest.approx(2.0 - sd_first), pytest.approx(6.0 + sd_last))]

    with subtests.test(msg="line mode draws one mean line per group"):
        ax = bundle._draw(Figure(), _frame().dropna(), _line_spec(
            _grouped(order=["b", "a"]), hue="condition"))
        drawn = {tuple(np.round(line.get_ydata(), 6)) for line in ax.lines
                 if len(line.get_ydata()) == 2}
        assert {(20.0, 200.0), (3.0, 7.0)} <= drawn

    with subtests.test(msg="a plain line keeps its raw rows in order"):
        frame = pd.DataFrame({"time": [3.0, 1.0, 2.0],
                              "signal": [30.0, 10.0, 20.0]})
        ax = bundle._draw(Figure(), frame, {
            "kind": "line", "x": "time", "y": "signal",
            "line_options": {"linewidth": 3.0}})
        (line,) = ax.lines
        assert list(line.get_xdata()) == [3.0, 1.0, 2.0]
        assert list(line.get_ydata()) == [30.0, 10.0, 20.0]
        assert line.get_linewidth() == pytest.approx(3.0)


def _training_frame():
    return pd.DataFrame({
        "curve": ["train"] * 3 + ["val"] * 3 + ["fold"] * 3,
        "epoch": [1, 2, 3] * 3,
        "loss": [0.9, 0.4, 0.6, 0.8, 0.7, 0.75, 0.5, 0.3, 0.2],
        "loss__sd": [0.1, 0.1, 0.1, NAN, NAN, NAN, 0.05, 0.05, 0.05],
    })


def _training(**overrides):
    recipe = {"band": True, "mark_best": True, "direction": "min", "curves": [
        {"label": "train", "kind": "mean", "color": "#ff0000",
         "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
        {"label": "val", "kind": "mean", "color": "#00ff00", "alpha": 0.8,
         "marker": "s", "markersize": 4},
        {"label": "fold", "kind": "fold", "color": "#0000ff", "alpha": 0.5,
         "band_style": {"color": "#000000"},
         "best_style": {"color": "#123456", "marker": "x",
                        "linestyle": "None"}},
    ], "notes": [{"x": 0.1, "y": 0.9, "text": "best 0.2", "fontsize": 7}]}
    recipe.update(overrides)
    return recipe


def _first_loss_missing():
    frame = _training_frame()
    frame.loc[0, "loss"] = NAN
    return frame


def _val_missing():
    frame = _training_frame()
    frame.loc[frame["curve"] == "val", "loss"] = NAN
    return frame


# (case id, rows, recipe, best marks (x, y), number of SD bands, curves)
TRAINING = [
    ("min direction marks each curve's lowest value", _training_frame(),
     _training(), [(2.0, 0.4), (2.0, 0.7), (3.0, 0.2)], 1,
     ["train", "val", "fold"]),
    ("max direction marks the highest finite value", _first_loss_missing(),
     _training(band=False, direction="max"), [(3.0, 0.6), (1.0, 0.8),
                                              (1.0, 0.5)], 0,
     ["train", "val", "fold"]),
    ("min direction skips a missing value", _first_loss_missing(),
     _training(band=False, notes=[]), [(2.0, 0.4), (2.0, 0.7), (3.0, 0.2)],
     0, ["train", "val", "fold"]),
    ("no direction marks nothing", _val_missing(), _training(direction=None),
     [], 1, ["train", "val", "fold"]),
    ("mark_best off marks nothing", _val_missing(), _training(mark_best=False),
     [], 1, ["train", "val", "fold"]),
    ("a curve with no finite value gets no mark", _val_missing(), _training(),
     [(2.0, 0.4), (3.0, 0.2)], 1, ["train", "val", "fold"]),
    ("a band needs its SD column",
     _training_frame().drop(columns="loss__sd"), _training(),
     [(2.0, 0.4), (2.0, 0.7), (3.0, 0.2)], 0, ["train", "val", "fold"]),
    ("no curves draws only the notes", _training_frame(),
     _training(curves=[]), [], 0, []),
]


def test_draw_training_curves(subtests):
    """Training curves draw their lines, SD bands, best marks and notes."""
    for case, frame, recipe, best, bands, curves in TRAINING:
        with subtests.test(msg=case):
            ax = bundle._draw(Figure(), frame, {
                "kind": "line", "x": "epoch", "y": "loss", "hue": "curve",
                "training_curves": recipe})
            marks = [line for line in ax.lines
                     if line.get_label().startswith("_")]
            assert [(float(m.get_xdata()[0]), float(m.get_ydata()[0]))
                    for m in marks] == pytest.approx(best)
            assert len([c for c in ax.collections
                        if isinstance(c, PolyCollection)]) == bands
            assert [line.get_label() for line in _named(ax)] == curves
            legend = ax.get_legend()
            assert ([t.get_text() for t in legend.get_texts()]
                    if legend else []) == curves
            notes = recipe["notes"]
            assert [t.get_text() for t in ax.texts] == [n["text"]
                                                        for n in notes]

    with subtests.test(msg="styles reach the curves, marks, band and notes"):
        ax = bundle._draw(Figure(), _training_frame(), {
            "kind": "line", "x": "epoch", "y": "loss", "hue": "curve",
            "training_curves": _training()})
        curves = {line.get_label(): line for line in _named(ax)}
        assert list(curves["train"].get_ydata()) == pytest.approx(
            [0.9, 0.4, 0.6])
        assert curves["train"].get_linewidth() == pytest.approx(2.0)
        assert curves["val"].get_marker() == "s"
        marks = [line for line in ax.lines if line.get_label().startswith("_")]
        assert [m.get_marker() for m in marks] == ["o", "o", "x"]
        assert marks[0].get_alpha() == pytest.approx(0.9)
        ((_, low, high),) = _band_colours_and_edges(ax)
        assert (low, high) == (pytest.approx(0.3), pytest.approx(1.0))
        assert ax.texts[0].get_transform() == ax.transAxes
        assert ax.texts[0].get_fontsize() == 7

    with subtests.test(msg="no curves draws no grid either"):
        ax = bundle._draw(Figure(), _training_frame(), {
            "kind": "line", "x": "epoch", "y": "loss", "hue": "curve",
            "training_curves": _training(curves=[])})
        assert len(ax.lines) == 0
        assert not ax.xaxis._major_tick_kw.get("gridOn", False)


def _qq(fit, line):
    return {"fit": fit, "line": line, "data_color": "#112233",
            "reference_color": "#445566", "reference_width": 2.5}


def test_draw_statistical_recipes(subtests, monkeypatch):
    """Scatter, histogram, QQ, Lorenz, Venn and regression recipes redraw."""
    with subtests.test(msg="scatter draws with its recorded options"):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        ax = bundle._draw(Figure(), frame, {
            "kind": "scatter", "x": "a", "y": "b",
            "scatter": {"color": "#ff0000", "s": 9}})
        (points,) = ax.collections
        assert points.get_offsets().tolist() == [[1.0, 4.0], [2.0, 5.0],
                                                 [3.0, 6.0]]
        assert tuple(points.get_facecolors()[0]) == to_rgba("#ff0000")
        assert list(points.get_sizes()) == [9.0]

    with subtests.test(msg="hist uses the recorded bins and reference lines"):
        frame = pd.DataFrame({"v": [0.0, 1.0, 1.5, 2.0, 3.9]})
        ax = bundle._draw(Figure(), frame, {
            "kind": "hist", "x": "v", "histogram": {"bins": [0, 2, 4]},
            "references": [
                {"axis": "x", "value": 1.0, "color": "#00ff00",
                 "dashes": [2, 1]},
                {"axis": "y", "value": 2.0, "color": "#0000ff"}]})
        assert [patch.get_height() for patch in ax.patches] == [3.0, 2.0]
        vertical, horizontal = ax.lines
        assert (vertical.get_xdata()[0], vertical.get_color(),
                vertical.get_linestyle()) == (1.0, "#00ff00", "--")
        assert (horizontal.get_ydata()[0], horizontal.get_color(),
                horizontal.get_linestyle()) == (2.0, "#0000ff", "-")

    pytest.importorskip("statsmodels")
    qq_cases = [
        ("qq recolours the points and the 45-degree line",
         pd.DataFrame({"v": np.random.default_rng(0).normal(size=40)}),
         _qq(True, "45"), 40, [("None", "#112233"), ("--", "#445566")]),
        ("qq without a reference line leaves only points",
         pd.DataFrame({"v": [0.1, 0.4, -0.2, 1.3, -0.8]}), _qq(False, None),
         5, [("None", "#112233")]),
    ]
    for case, frame, recipe, n, lines in qq_cases:
        with subtests.test(msg=case):
            ax = bundle._draw(Figure(), frame,
                              {"kind": "qq", "y": "v", "qq": recipe})
            assert [(line.get_linestyle(),
                     line.get_markerfacecolor() if line.get_linestyle()
                     == "None" else line.get_color())
                    for line in ax.lines] == lines
            assert ax.lines[0].get_markeredgecolor() == "none"
            assert len(ax.lines[0].get_xdata()) == n
            if len(lines) == 2:
                assert ax.lines[1].get_linewidth() == 2.5

    with subtests.test(msg="qq that draws no lines still sets its title"):
        import statsmodels.api as sm

        with monkeypatch.context() as patch:
            patch.setattr(sm, "qqplot", lambda *args, **kwargs: None)
            ax = bundle._draw(Figure(), pd.DataFrame({"v": [1.0, 2.0]}),
                              {"kind": "qq", "y": "v", "qq": _qq(False, None),
                               "title": "Q"})
        assert (len(ax.lines), ax.get_title()) == (0, "Q")

    with subtests.test(msg="lorenz uses the helpers in the module namespace"):
        helpers = {}
        exec(bundle._lorenz_script(), helpers)
        calls = []

        def counted(values):
            calls.append(len(values))
            return helpers["lorenz_curve"](values)

        frame = pd.DataFrame({"input": [0, 0, 0, 1, 1],
                              "row": [0, 1, 2, 0, 1],
                              "reads": [1.0, 2.0, 7.0, 3.0, 3.0]})
        with monkeypatch.context() as patch:
            patch.setattr(bundle, "lorenz_curve", counted, raising=False)
            for name in ("gini_coefficient", "text_legend"):
                patch.setattr(bundle, name, helpers[name], raising=False)
            ax = bundle._draw(Figure(), frame, {
                "kind": "lorenz", "y": "reads", "lorenz": {
                    "input_column": "input", "row_column": "row",
                    "combined_color": "#000000", "curves": [
                        {"input": 0, "rows": [0, 1, 2], "label": "A",
                         "color": "#ff0000", "linestyle": "-"},
                        {"input": 1, "rows": [0, 1], "label": "B",
                         "color": "#00ff00", "linestyle": ":"}]}})
        assert calls == [3, 2, 5]
        gini = helpers["gini_coefficient"]
        assert [line.get_label() for line in ax.lines] == [
            f"A (Gini: {gini(np.array([1.0, 2.0, 7.0])):.4f})",
            "B (Gini: 0.0000)",
            f"Combined (Gini: {gini(np.array([1.0, 2.0, 7.0, 3.0, 3.0])):.4f})"]

    with subtests.test(msg="venn skips regions and labels that are empty"):
        pytest.importorskip("matplotlib_venn")
        frame = pd.DataFrame({"source": ["a", "a", "b"],
                              "gene": ["g1", "g2", "g3"],
                              "coefficient": [1.0, 2.0, 3.0]})
        ax = bundle._draw(Figure(), frame, {"kind": "venn", "venn": {
            "input_column": "source", "gene_column": "gene",
            "filter_coeff": None, "labels": ["A", "B"], "fontsize": 7,
            "ink": "#123456", "inputs": [{"input": "a"}, {"input": "b"}],
            "colors": {"10": "#ff0000", "01": "#00ff00", "11": "#0000ff"}}})
        colours = {tuple(np.round(patch.get_facecolor(), 3))
                   for patch in ax.patches}
        assert to_rgba("#ff0000") in colours and to_rgba("#00ff00") in colours
        assert to_rgba("#0000ff") not in colours
        texts = {text.get_text(): text for text in ax.texts}
        assert {"A", "B", "2", "1"} <= set(texts)
        assert texts["2"].get_fontsize() == 7
        assert texts["A"].get_color() == "#123456"

    with subtests.test(msg="regression panels use the renderers in namespace"):
        seen, lettered = {}, []

        class _Panel:
            data = pd.DataFrame({"drawn": [1]})

        def renderer(ax, frame, **options):
            seen.update(options=options, rows=len(frame))
            ax.plot([0, 1], [0, 1])
            return _Panel()

        figure = Figure()
        with monkeypatch.context() as patch:
            patch.setattr(bundle, "_REGRESSION_PANELS", {"fake": renderer},
                          raising=False)
            patch.setattr(bundle, "panel_letter",
                          lambda ax, letter: lettered.append(letter),
                          raising=False)
            patch.setattr(bundle, "_REGRESSION_LOCALISATIONS", None,
                          raising=False)
            bundle._draw(figure, pd.DataFrame({"x": [1, 2, 3]}), {
                "kind": "regression_panel", "regression": "fake",
                "regression_options": {"alpha": 0.1}, "letter": "C",
                "localisations": {"g": "nucleus"}})
            assert bundle._REGRESSION_LOCALISATIONS == {"g": "nucleus"}
        assert seen == {"options": {"alpha": 0.1}, "rows": 3}
        assert lettered == ["C"]
        assert figure._spacr_drawn_data is _Panel.data


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class _Opaque:
    def __str__(self):
        return "threshold=0.5"


def _touch(path):
    Path(path).touch()


def _fail(_path):
    raise RuntimeError("no backend")


def test_save_writes_a_new_bundle_whatever_the_renderer_does(
        tmp_path, subtests):
    """A bundle keeps data, statistics and settings beside any picture."""
    cases = [
        ("a failed renderer still leaves the data and statistics", "my graph",
         _fail, {"data": pd.DataFrame({"value": [1, 2, 3]})},
         ["data.csv", "settings.json", "statistics.csv"],
         {"graph": "my graph", "rows": 3}),
        ("a working renderer adds both pictures", "graph", _touch, {},
         ["data.csv", "graph.pdf", "graph.png", "settings.json",
          "statistics.csv"], {"graph": "graph", "rows": 0}),
        ("a setting JSON cannot hold is recorded as its text", "graph",
         _touch, {"settings": {"gate": _Opaque(), "keep": {1}}},
         ["data.csv", "graph.pdf", "graph.png", "settings.json",
          "statistics.csv"],
         {"gate": "threshold=0.5", "keep": "{1}", "graph": "graph",
          "rows": 0}),
    ]
    for case, name, render, extra, files, settings in cases:
        with subtests.test(msg=case):
            folder = tmp_path / case.replace(" ", "_")
            out = bundle.save(str(folder), name, render=render, **extra)
            assert sorted(os.listdir(out)) == files
            payload = json.loads(
                (Path(out) / "settings.json").read_text(encoding="utf-8"))
            assert payload == settings
            if settings["rows"]:
                assert pd.read_csv(Path(out) / "data.csv")["value"].tolist(
                ) == [1, 2, 3]

    with subtests.test(msg="a second save sits beside the first"):
        first = bundle.save(str(tmp_path), "twice", render=_touch)
        second = bundle.save(str(tmp_path), "twice", render=_touch)
        assert (first, second) == (str(tmp_path / "twice"),
                                   str(tmp_path / "twice-2"))
        assert sorted(os.listdir(first)) == sorted(os.listdir(second))
