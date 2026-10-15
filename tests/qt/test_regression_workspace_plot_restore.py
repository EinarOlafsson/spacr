"""Workspace restoration must retain the selections the real plot exposes."""

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def results(tmp_path, name="run_a", *, extra_colours=False):
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    path = folder / "results.csv"
    frame = pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(6)],
        "coefficient": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "p_value": [0.001, 0.1, 0.8, 0.3, 0.02, 0.0001],
        "condition": ["positive", "negative", "other"] * 2,
    })
    if extra_colours:
        frame["qc"] = ["pass", "fail", "pass"] * 2
        frame["batch"] = ["b1", "b1", "b2"] * 2
    frame.to_csv(path, index=False)
    return path


def test_workspace_restores_all_rows_selected_through_the_volcano(
        panel, qtbot, tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    assert panel.load(str(results(tmp_path)))
    chosen = ["gene_fraction:gene[G0]", "gene_fraction:gene[G3]",
              "gene_fraction:gene[G4]"]
    panel.volcano.keys_selected.emit(chosen)
    assert panel.selected_keys() == chosen
    saved = json.loads(json.dumps(panel.workspace_state()))

    fresh = RegressionResultsPanel()
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(saved)
    assert fresh.selected_keys() == chosen
    assert fresh.table.selected_keys() == chosen


def test_workspace_restores_an_explicitly_uncoloured_plot(
        panel, qtbot, tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    assert panel.load(str(results(tmp_path)))
    assert panel.colour_channels() == ["condition"]
    panel._colour_by.setCurrentIndex(panel._colour_by.findData(None))
    assert panel.colour_channels() == []
    saved = json.loads(json.dumps(panel.workspace_state()))

    fresh = RegressionResultsPanel()
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(saved)
    assert fresh.colour_channels() == []


def test_workspace_saved_view_wins_over_the_same_open_run(panel, tmp_path):
    assert panel.load(str(results(tmp_path)))
    panel.set_threshold_multiplier(2.0)
    panel.volcano.set_axis_limits(x=(-1.5, 1.5))
    saved = json.loads(json.dumps(panel.workspace_state()))
    panel.set_threshold_multiplier(4.0)
    panel.volcano.set_axis_limits(x=(-3.0, 3.0))

    assert panel.apply_workspace_state(saved)
    restored = panel.plot_state()
    assert restored["threshold_multiplier"] == 2.0
    assert restored["x_limits"] == (-1.5, 1.5)


def test_legacy_single_selection_workspace_still_restores(panel, qtbot, tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    assert panel.load(str(results(tmp_path)))
    chosen = "gene_fraction:gene[G3]"
    panel.table.select_key(chosen)
    saved = json.loads(json.dumps(panel.workspace_state()))
    for state in saved["runs"].values():
        state.pop("selected_keys")

    fresh = RegressionResultsPanel()
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(saved)
    assert fresh.selected_keys() == [chosen]
    assert fresh.volcano.selected_keys() == [chosen]


def test_explicit_empty_selection_clears_the_table_and_plot(panel, tmp_path):
    assert panel.load(str(results(tmp_path)))
    panel.table.select_keys(["gene_fraction:gene[G0]", "gene_fraction:gene[G3]"])

    assert panel.apply_plot_state({"selected_keys": [], "selected_key": None})

    assert panel.selected_keys() == []
    assert panel.table.selected_keys() == []
    assert panel.volcano.selected_keys() == []


def test_omitted_colour_keeps_the_current_choice(panel, tmp_path):
    assert panel.load(str(results(tmp_path)))
    panel._colour_by.setCurrentIndex(panel._colour_by.findData(None))

    assert panel.apply_plot_state({"threshold_multiplier": 2.0})

    assert panel.colour_channels() == []


def test_restore_preserves_an_unrelated_unsaved_live_view(panel, tmp_path):
    first = results(tmp_path)
    second = results(tmp_path, "run_b")
    assert panel.load(str(first))
    panel.set_threshold_multiplier(2.0)
    saved = json.loads(json.dumps(panel.workspace_state()))
    assert panel.load(str(second))
    panel.set_threshold_multiplier(4.0)

    assert panel.apply_workspace_state(saved)
    assert panel.plot_state()["threshold_multiplier"] == 2.0
    assert panel.load(str(second))
    assert panel.plot_state()["threshold_multiplier"] == 4.0


@pytest.mark.parametrize("shape,opacity", [("qc", None), (None, "batch"),
                                          ("qc", "batch")])
def test_workspace_restores_secondary_colour_channels(
        panel, qtbot, tmp_path, shape, opacity):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    assert panel.load(str(results(tmp_path, extra_colours=True)))
    panel._colour_by_2.setCurrentIndex(panel._colour_by_2.findData(shape))
    panel._colour_by_3.setCurrentIndex(panel._colour_by_3.findData(opacity))
    expected = ["condition"] + [value for value in (shape, opacity) if value]
    assert panel.colour_channels() == expected
    saved = json.loads(json.dumps(panel.workspace_state()))

    fresh = RegressionResultsPanel()
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(saved)

    assert fresh.colour_channels() == expected
    assert fresh._colour_by_2.currentData() == shape
    assert fresh._colour_by_3.currentData() == opacity
    labels = [label for label, _ in fresh.volcano._legend_entries()]
    assert any("(shape)" in label for label in labels) is bool(shape)
    assert any("(opacity)" in label for label in labels) is bool(opacity)


def test_secondary_channels_remain_owned_by_their_run(panel, tmp_path):
    first = results(tmp_path, extra_colours=True)
    second = results(tmp_path, "run_b", extra_colours=True)
    assert panel.load(str(first))
    panel._colour_by_2.setCurrentIndex(panel._colour_by_2.findData("qc"))
    panel._colour_by_3.setCurrentIndex(panel._colour_by_3.findData("batch"))
    assert panel.load(str(second))
    panel._colour_by_2.setCurrentIndex(panel._colour_by_2.findData("batch"))
    panel._colour_by_3.setCurrentIndex(panel._colour_by_3.findData(None))

    assert panel.load(str(first))
    assert panel.colour_channels() == ["condition", "qc", "batch"]
    assert panel.load(str(second))
    assert panel.colour_channels() == ["condition", "batch"]
    assert panel._colour_by_3.currentData() is None


def test_legacy_state_omitting_secondary_channels_keeps_their_choices(panel, tmp_path):
    assert panel.load(str(results(tmp_path, extra_colours=True)))
    panel._colour_by_2.setCurrentIndex(panel._colour_by_2.findData("qc"))
    panel._colour_by_3.setCurrentIndex(panel._colour_by_3.findData("batch"))

    assert panel.apply_plot_state({"colour_by": "condition"})

    assert panel.colour_channels() == ["condition", "qc", "batch"]


def test_explicit_empty_secondary_channels_disable_their_encodings(panel, tmp_path):
    assert panel.load(str(results(tmp_path, extra_colours=True)))
    panel._colour_by_2.setCurrentIndex(panel._colour_by_2.findData("qc"))
    panel._colour_by_3.setCurrentIndex(panel._colour_by_3.findData("batch"))

    assert panel.apply_plot_state({"symbol_by": None, "opacity_by": None})

    assert panel.colour_channels() == ["condition"]
    labels = [label for label, _ in panel.volcano._legend_entries()]
    assert not any("(shape)" in label or "(opacity)" in label for label in labels)
