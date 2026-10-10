"""Recipe drawing branches of the figure bundle, plates and sheet modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import bundle  # noqa: E402
from spacr.figures.plates import build_plates, well_matrices  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _plate_spec(grouping):
    return {"kind": "plate_heatmap", "plate": {
        "coordinates": {"p1_r1_c1": [1, 1], "p1_r2_c2": [2, 2]},
        "grouping": grouping, "variable": "value", "shape": [2, 2]}}


def test_plate_mean_and_sum_aggregate_each_well():
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c2", "other"],
                          "value": [1.0, 3.0, 5.0, 100.0]})
    mean = bundle._recipe_frame(frame, _plate_spec("mean"))
    total = bundle._recipe_frame(frame, _plate_spec("sum"))
    assert mean.loc[1, 1] == 2.0 and mean.loc[2, 2] == 5.0
    assert total.loc[1, 1] == 4.0 and total.loc[2, 2] == 5.0
    assert np.isnan(mean.loc[1, 2]) and np.isnan(total.loc[2, 1])


def test_an_unknown_plate_aggregation_is_refused():
    frame = pd.DataFrame({"prc": ["p1_r1_c1"], "value": [1.0]})
    with pytest.raises(ValueError, match="Invalid plate aggregation"):
        bundle._recipe_frame(frame, _plate_spec("median"))


def test_melted_measurements_take_their_display_labels():
    frame = pd.DataFrame({"id": [1, 2], "cell_area": [3.0, 4.0],
                          "nucleus_area": [5.0, 6.0]})
    spec = {"melt": {"var_name": "measurement", "value_name": "value",
                     "id_vars": ["id"], "columns": ["cell_area", "nucleus_area"],
                     "labels": {"cell_area": "Cell"}}}
    data = bundle._recipe_frame(frame, spec)
    assert data["measurement"].tolist() == ["Cell", "Cell", "nucleus_area",
                                            "nucleus_area"]
    assert data["value"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert list(frame.columns) == ["id", "cell_area", "nucleus_area"]


def test_melted_measurements_keep_their_column_names_without_labels():
    frame = pd.DataFrame({"id": [1], "a": [3.0], "b": [5.0]})
    spec = {"melt": {"var_name": "measurement", "value_name": "value",
                     "id_vars": ["id"], "columns": ["a", "b"]}}
    data = bundle._recipe_frame(frame, spec)
    assert data.to_dict("list") == {"id": [1, 1], "measurement": ["a", "b"],
                                    "value": [3.0, 5.0]}


def test_scatter_recipe_draws_with_its_recorded_options():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    figure = Figure()
    ax = bundle._draw(figure, frame, {"kind": "scatter", "x": "a", "y": "b",
                                      "scatter": {"color": "#ff0000", "s": 9}})
    (points,) = ax.collections
    assert points.get_offsets().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert tuple(points.get_facecolors()[0]) == to_rgba("#ff0000")


def test_hist_recipe_uses_the_recorded_bins_and_reference_lines():
    frame = pd.DataFrame({"v": [0.0, 1.0, 1.5, 2.0, 3.9]})
    figure = Figure()
    spec = {"kind": "hist", "x": "v", "histogram": {"bins": [0, 2, 4]},
            "references": [
                {"axis": "x", "value": 1.0, "color": "#00ff00", "dashes": [2, 1]},
                {"axis": "y", "value": 2.0, "color": "#0000ff"}]}
    ax = bundle._draw(figure, frame, spec)
    assert [patch.get_height() for patch in ax.patches] == [3.0, 2.0]
    vertical, horizontal = ax.lines
    assert vertical.get_xdata()[0] == 1.0
    assert vertical.get_linestyle() != "-"
    assert vertical.get_color() == "#00ff00"
    assert horizontal.get_ydata()[0] == 2.0
    assert horizontal.get_linestyle() == "-"


def test_qq_recipe_recolours_the_points_and_the_reference_line():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"v": rng.normal(size=40)})
    recipe = {"fit": True, "line": "45", "data_color": "#112233",
              "reference_color": "#445566", "reference_width": 2.5}
    figure = Figure()
    ax = bundle._draw(figure, frame, {"kind": "qq", "y": "v", "qq": recipe})
    markers = [line for line in ax.lines if line.get_linestyle() == "None"]
    references = [line for line in ax.lines if line.get_linestyle() != "None"]
    assert len(markers) == 1 and len(references) == 1
    assert markers[0].get_markerfacecolor() == "#112233"
    assert markers[0].get_markeredgecolor() == "none"
    assert references[0].get_color() == "#445566"
    assert references[0].get_linewidth() == 2.5
    assert len(markers[0].get_xdata()) == 40


def test_qq_recipe_without_a_reference_line_leaves_only_points():
    pytest.importorskip("statsmodels")
    frame = pd.DataFrame({"v": [0.1, 0.4, -0.2, 1.3, -0.8]})
    recipe = {"fit": False, "line": None, "data_color": "#112233",
              "reference_color": "#445566", "reference_width": 2.5}
    ax = bundle._draw(Figure(), frame, {"kind": "qq", "y": "v", "qq": recipe})
    assert [line.get_linestyle() for line in ax.lines] == ["None"]


def test_qq_recipe_with_no_drawn_lines_still_finishes(monkeypatch):
    sm = pytest.importorskip("statsmodels.api")
    monkeypatch.setattr(sm, "qqplot", lambda *args, **kwargs: None)
    recipe = {"fit": False, "line": None, "data_color": "#112233",
              "reference_color": "#445566", "reference_width": 2.5}
    ax = bundle._draw(Figure(), pd.DataFrame({"v": [1.0, 2.0]}),
                      {"kind": "qq", "y": "v", "qq": recipe, "title": "Q"})
    assert len(ax.lines) == 0 and ax.get_title() == "Q"


def test_lorenz_recipe_uses_helpers_already_in_the_module_namespace(monkeypatch):
    helpers = {}
    exec(bundle._lorenz_script(), helpers)
    calls = []

    def counted(values):
        calls.append(len(values))
        return helpers["lorenz_curve"](values)

    monkeypatch.setattr(bundle, "lorenz_curve", counted, raising=False)
    monkeypatch.setattr(bundle, "gini_coefficient", helpers["gini_coefficient"],
                        raising=False)
    monkeypatch.setattr(bundle, "text_legend", helpers["text_legend"], raising=False)
    frame = pd.DataFrame({"input": [0, 0, 0, 1, 1], "row": [0, 1, 2, 0, 1],
                          "reads": [1.0, 2.0, 7.0, 3.0, 3.0]})
    spec = {"kind": "lorenz", "y": "reads", "lorenz": {
        "input_column": "input", "row_column": "row", "combined_color": "#000000",
        "curves": [
            {"input": 0, "rows": [0, 1, 2], "label": "A", "color": "#ff0000",
             "linestyle": "-"},
            {"input": 1, "rows": [0, 1], "label": "B", "color": "#00ff00",
             "linestyle": ":"}]}}
    ax = bundle._draw(Figure(), frame, spec)
    assert calls == [3, 2, 5]
    labels = [line.get_label() for line in ax.lines]
    assert labels[0].startswith("A (Gini: ")
    assert labels[1] == "B (Gini: 0.0000)"
    assert labels[2].startswith("Combined (Gini: ")


def test_venn_recipe_skips_regions_and_labels_that_are_not_drawn():
    pytest.importorskip("matplotlib_venn")
    frame = pd.DataFrame({"source": ["a", "a", "b"],
                          "gene": ["g1", "g2", "g3"],
                          "coefficient": [1.0, 2.0, 3.0]})
    spec = {"kind": "venn", "venn": {
        "input_column": "source", "gene_column": "gene", "filter_coeff": None,
        "labels": ["A", "B"], "fontsize": 7, "ink": "#123456",
        "inputs": [{"input": "a"}, {"input": "b"}],
        "colors": {"10": "#ff0000", "01": "#00ff00", "11": "#0000ff"}}}
    ax = bundle._draw(Figure(), frame, spec)
    colours = {tuple(np.round(patch.get_facecolor(), 3)) for patch in ax.patches}
    assert to_rgba("#ff0000") in colours and to_rgba("#00ff00") in colours
    assert to_rgba("#0000ff") not in colours
    texts = {text.get_text(): text for text in ax.texts}
    assert {"A", "B", "2", "1"} <= set(texts)
    assert texts["2"].get_fontsize() == 7
    assert texts["A"].get_color() == "#123456"


def test_regression_panel_recipe_uses_renderers_in_the_namespace(monkeypatch):
    seen = {}

    class _Panel:
        data = pd.DataFrame({"drawn": [1]})

    def renderer(ax, frame, **options):
        seen["options"] = options
        seen["rows"] = len(frame)
        ax.plot([0, 1], [0, 1])
        return _Panel()

    lettered = []
    monkeypatch.setattr(bundle, "_REGRESSION_PANELS", {"fake": renderer},
                        raising=False)
    monkeypatch.setattr(bundle, "panel_letter",
                        lambda ax, letter: lettered.append(letter), raising=False)
    figure = Figure()
    bundle._draw(figure, pd.DataFrame({"x": [1, 2, 3]}), {
        "kind": "regression_panel", "regression": "fake",
        "regression_options": {"alpha": 0.1}, "letter": "C",
        "localisations": {"g": "nucleus"}})
    assert seen == {"options": {"alpha": 0.1}, "rows": 3}
    assert lettered == ["C"]
    assert figure._spacr_drawn_data is _Panel.data
    assert bundle._REGRESSION_LOCALISATIONS == {"g": "nucleus"}


def test_plate_keys_without_a_plate_prefix_are_read_as_one_plate():
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "r2_c2"], "value": [4.0, 8.0]})
    names, matrices, _grid = well_matrices(frame, "value", plates=["p1"])
    assert names == ["p1"]
    assert matrices[0][0, 0] == 4.0
    assert np.isfinite(matrices[0]).sum() == 1


def test_plate_recipe_skips_keys_whose_well_cannot_be_read():
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r2_c2", "p1_r3_cX"],
                          "value": [1.0, 2.0, 3.0]})
    figure, panel = build_plates(frame, "value", plates=["p1"], target="print")
    assert panel.drawn
    (recipe,) = figure._spacr_spec["panels"]
    assert recipe["plate"]["coordinates"] == {"p1_r1_c1": [1, 1],
                                              "p1_r2_c2": [2, 2]}


def test_a_sheet_with_no_drawable_panel_registers_a_plain_sheet():
    from spacr.figures import sheet as sheet_module

    sheet = sheet_module.build_sheet(pd.DataFrame({"unrelated": [1, 2]}),
                                     target="print")
    assert sheet.panels == []
    assert sheet.skipped
    assert all(not panel.drawn for panel in sheet.skipped)
    assert sheet.figure._spacr_spec["kind"] == "sheet"
    assert "panels" not in sheet.figure._spacr_spec
    assert not any(axes.axison or axes.has_data() for axes in sheet.figure.axes)
