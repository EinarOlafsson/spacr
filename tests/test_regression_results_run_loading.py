"""Finding a regression run on disk, reading it, and the panel that shows it.

The Results tab is handed a CSV, a run folder, or a parent holding many runs.
These tests drive the walk that turns one of those into tables, the summary
text printed above the statsmodels block, the worker and synchronous load
paths with every way they fail, and the panel's optional readings (run
settings, localisations, saved views) that may each be lost without losing
the run.  One general test per unit, each walking a case table with pytest's
``subtests`` fixture against real files in ``tmp_path``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from tests.refused_access import refused

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr.qt.widgets.regression_results import (  # noqa: E402
    MAX_SEARCH_DEPTH, RESULT_FILENAMES, SPACR_SUMMARY_HEADING,
    VERBATIM_HEADING, RegressionResultsPanel, _spacr_summary_text,
    _with_spacr_summary, find_results_table, find_results_tables,
)

_UNPARSABLE = 'a,b\n"unterminated,1\n2,3,4,5\n'


def _run_folder(root, name, filenames=("results.csv",)):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (folder / filename).write_text("gene,coefficient\na,1.0\n",
                                       encoding="utf-8")
    return folder


def _write_run(tmp_path, name="run1", frame=None, filenames=("results.csv",)):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    frame = frame if frame is not None else pd.DataFrame({
        "feature": ["fraction:grna[a_1]", "fraction:grna[b_1]"],
        "coefficient": [1.0, -0.5],
        "p_value": [0.01, 0.6],
        "q_value": [0.02, 0.8],
    })
    for filename in filenames:
        frame.to_csv(folder / filename, index=False)
    return folder


def _write_summary(tmp_path, text):
    from spacr.ml import SUMMARY_FILENAME

    folder = _run_folder(tmp_path, "run1")
    (folder / SUMMARY_FILENAME).write_text(text, encoding="utf-8")
    return folder


def _results_frame():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(size=n),
        "q_value": np.sort(rng.uniform(size=n)),
        "condition": rng.choice(["nc", "pc", "other"], n, p=[0.1, 0.1, 0.8]),
    })


def _guides() -> pd.DataFrame:
    return pd.DataFrame({"feature": ["fraction:grna[G0_1]",
                                     "fraction:grna[G0_2]"],
                         "coefficient": [1.0, 2.0],
                         "p_value": [0.01, 0.4]})


def _panel(qtbot) -> RegressionResultsPanel:
    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def _no_tables_sentence(searched) -> str:
    return (f"Searched {searched} and found none of "
            f"{', '.join(RESULT_FILENAMES)} in it or in any folder up to "
            f"{MAX_SEARCH_DEPTH} deep.")


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------

def test_the_results_walk_finds_tables_within_its_depth_and_limit(
        tmp_path, monkeypatch, subtests):
    """Non-paths and non-CSVs find nothing; depth, limit and races hold."""
    with subtests.test(msg="something that is not a path"):
        assert find_results_tables(3.5) == []
        assert find_results_table(3.5) is None

    with subtests.test(msg="a file that is not a CSV"):
        other = tmp_path / "notes.txt"
        other.write_text("hello", encoding="utf-8")
        assert find_results_tables(str(other)) == []

    with subtests.test(msg="the walk stops at its depth"):
        root = tmp_path / "depth"
        shallow = _run_folder(root, "shallow")
        deep = _run_folder(root, os.path.join("a", "b", "deep"))
        assert find_results_tables(str(root), max_depth=1) == [
            str(shallow / "results.csv")]
        assert sorted(find_results_tables(str(root), max_depth=6)) == sorted(
            [str(shallow / "results.csv"), str(deep / "results.csv")])

    with subtests.test(msg="the walk stops at the candidate limit"):
        root = tmp_path / "many"
        for index in range(5):
            _run_folder(root, f"run{index}")
        assert len(find_results_tables(str(root), limit=1)) == 1
        assert len(find_results_tables(str(root), limit=200)) == 5

    with subtests.test(msg="a table that vanishes before its stat"):
        root = tmp_path / "race"
        folder = _run_folder(root, "run1", RESULT_FILENAMES)
        real_getmtime = os.path.getmtime

        def vanishing(path):
            if str(path).endswith("results_grna.csv"):
                raise OSError(2, "No such file or directory")
            return real_getmtime(path)

        monkeypatch.setattr(os.path, "getmtime", vanishing)
        assert find_results_tables(str(root)) == [
            str(folder / name) for name in RESULT_FILENAMES
            if name != "results_grna.csv"]


# ---------------------------------------------------------------------------
# the summary text
# ---------------------------------------------------------------------------

def test_the_spacr_summary_is_read_trimmed_and_prepended(tmp_path, subtests):
    """Only a spaCR-headed file is prepended, without its verbatim tail."""
    from spacr.ml import SUMMARY_FILENAME

    with subtests.test(msg="an unreadable summary reads as none"):
        folder = _write_summary(tmp_path / "a",
                                f"{SPACR_SUMMARY_HEADING}\nbody\n")
        with refused(folder / SUMMARY_FILENAME, 0o000, windows="read"):
            assert _spacr_summary_text(str(folder)) == ""

    with subtests.test(msg="a file without the spaCR heading is not used"):
        folder = _write_summary(tmp_path / "b",
                                "                   OLS Regression\n")
        assert _spacr_summary_text(str(folder)) == ""

    with subtests.test(msg="the verbatim tail is trimmed"):
        folder = _write_summary(
            tmp_path / "c", f"{SPACR_SUMMARY_HEADING}\nsections here\n\n"
                            f"{VERBATIM_HEADING}\nan old statsmodels block\n")
        assert _spacr_summary_text(str(folder)) == (
            f"{SPACR_SUMMARY_HEADING}\nsections here")

    summarised = _write_summary(tmp_path / "d",
                                f"{SPACR_SUMMARY_HEADING}\nsections\n")
    bare = _run_folder(tmp_path / "e", "run1")
    cases = [
        ("a missing statsmodels text beside a spaCR summary", summarised,
         "this backend keeps none", True,
         f"{SPACR_SUMMARY_HEADING}\nsections\n\n{VERBATIM_HEADING}\n"
         "No statsmodels summary: this backend keeps none"),
        ("the statsmodels text kept verbatim", summarised,
         "OLS Regression Results", False,
         f"{SPACR_SUMMARY_HEADING}\nsections\n\n{VERBATIM_HEADING}\n"
         "OLS Regression Results"),
        ("no spaCR summary and no fit", bare, "no fit here", True,
         "No summary: no fit here"),
        ("no spaCR summary, a fit", bare, "OLS text", False, "OLS text"),
    ]
    for case_id, folder, text, missing, expected in cases:
        with subtests.test(msg=case_id):
            assert _with_spacr_summary(str(folder), text,
                                       missing=missing) == expected


# ---------------------------------------------------------------------------
# loading: the worker half and the synchronous path say the same thing
# ---------------------------------------------------------------------------

def test_the_worker_reports_every_failure_as_a_sentence(tmp_path, subtests):
    """``_read_run`` returns ``{"error": ...}`` rather than raising."""
    gone = str(tmp_path / "gone")
    unparsable = tmp_path / "bad" / "run1"
    unparsable.mkdir(parents=True)
    (unparsable / "results.csv").write_text(_UNPARSABLE, encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    cases = [
        ("a missing folder", gone,
         f"{gone} does not exist, so there is nothing to load from it."),
        ("a folder with no tables", str(empty),
         _no_tables_sentence(str(empty))),
    ]
    for case_id, path, expected in cases:
        with subtests.test(msg=case_id):
            assert RegressionResultsPanel._read_run(path) == {
                "error": expected}

    with subtests.test(msg="a table it could not parse"):
        outcome = RegressionResultsPanel._read_run(str(unparsable))
        assert list(outcome) == ["error"]
        assert outcome["error"].startswith(
            f"Could not read {unparsable / 'results.csv'}: ")

    with subtests.test(msg="a good run comes back whole"):
        folder = _write_run(tmp_path / "good")
        outcome = RegressionResultsPanel._read_run(str(folder))
        assert outcome["found"] == str(folder / "results.csv")
        assert outcome["tables"] == [str(folder / "results.csv")]
        assert list(outcome["frame"]["feature"]) == [
            "fraction:grna[a_1]", "fraction:grna[b_1]"]


def test_loading_a_run_names_the_folder_the_tables_and_the_runs(
        qtbot, tmp_path, monkeypatch, subtests):
    """Every load outcome is one status sentence naming where and why."""
    nothing = ("Nothing was handed to the results panel, so there is no "
               "folder to search. Use “Load results…”.")
    with subtests.test(msg="loading nothing says which button to press"):
        panel = _panel(qtbot)
        assert panel.load("") is False
        assert panel._status == nothing

    with subtests.test(msg="a folder that does not exist"):
        panel = _panel(qtbot)
        missing = str(tmp_path / "not_here")
        assert panel.load(missing) is False
        assert panel._status == (
            f"{missing} does not exist, so there is nothing to load from it.")

    with subtests.test(msg="a folder with no result table"):
        panel = _panel(qtbot)
        (tmp_path / "empty").mkdir()
        assert panel.load(str(tmp_path / "empty")) is False
        assert panel._status == _no_tables_sentence(str(tmp_path / "empty"))

    with subtests.test(msg="a CSV pandas cannot parse"):
        panel = _panel(qtbot)
        folder = tmp_path / "bad"
        folder.mkdir()
        (folder / "results.csv").write_text(_UNPARSABLE, encoding="utf-8")
        assert panel.load(str(folder)) is False
        assert panel._status.startswith(
            f"Could not read {folder / 'results.csv'}: ")

    with subtests.test(msg="a refused table names where it was found"):
        panel = _panel(qtbot)
        root = tmp_path / "refused"
        folder = _write_run(root, frame=pd.DataFrame(
            {"feature": [], "coefficient": [], "p_value": []}))
        assert panel.load(str(root)) is False
        assert panel._status.endswith(
            f" ({folder / 'results.csv'}, found under {root})")

    with subtests.test(msg="the newest of several runs is loaded"):
        panel = _panel(qtbot)
        root = tmp_path / "two"
        _write_run(root, "run1")
        _write_run(root, "run2")
        assert panel.load(str(root)) is True
        assert panel._status.endswith(f" — newest of 2 runs under {root}")

    with subtests.test(msg="one run that wrote several tables"):
        panel = _panel(qtbot)
        root = tmp_path / "several"
        _write_run(root, "run1", filenames=RESULT_FILENAMES)
        assert panel.load(str(root)) is True
        assert panel._status.endswith(
            f" — the newest run under {root} wrote "
            f"{len(RESULT_FILENAMES)} tables; this is the full one.")

    with subtests.test(msg="browsing starts beside the open run"):
        from PySide6.QtWidgets import QFileDialog

        panel = _panel(qtbot)
        folder = _write_run(tmp_path / "browse", "run1")
        assert panel.load(str(folder)) is True
        seen = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            lambda parent, title, start: seen.append(start)
                            or "")
        assert panel.browse_for_results() is False
        assert seen == [str(folder)]


# ---------------------------------------------------------------------------
# the optional readings a run may lose
# ---------------------------------------------------------------------------

def test_a_run_still_loads_when_an_optional_reading_fails(qtbot, monkeypatch,
                                                          subtests):
    """Settings, localisations, axes, saved views and diagnostics are extras."""
    import spacr.localisation as localisation
    import spacr.refit as refit

    def _raise(*_args, **_kwargs):
        raise RuntimeError("not available")

    with subtests.test(msg="unreadable run settings"):
        panel = _panel(qtbot)
        monkeypatch.setattr(refit, "settings_of_run", _raise)
        assert panel.set_frame(_results_frame(), source="results.csv") is True
        assert panel._run_settings is None
        monkeypatch.undo()

    with subtests.test(msg="localisations that cannot be looked up"):
        panel = _panel(qtbot)
        monkeypatch.setattr(localisation, "present", _raise)
        assert panel.set_frame(_results_frame(), source="results.csv") is True
        assert panel._colour_by.findData(panel.LOPIT_KEY) == -1
        monkeypatch.undo()

    with subtests.test(msg="a volcano that will not release its axes"):
        panel = _panel(qtbot)
        monkeypatch.setattr(panel.volcano, "auto_range_axes", _raise)
        assert panel.set_frame(_results_frame(), source="results.csv") is True
        assert panel.table.table.rowCount() == 60

    for case_id, target, state in [
            ("saved axis limits that will not apply", "set_axis_limits",
             {"x_limits": (-1.0, 1.0), "y_limits": (0.0, 3.0)}),
            ("a saved selection no longer visible", "_select_key",
             {"selected_key": "gene[244480]"})]:
        with subtests.test(msg=case_id):
            panel = _panel(qtbot)
            panel.set_frame(_results_frame(), source="results.csv")
            owner = panel.volcano if target == "set_axis_limits" else panel
            monkeypatch.setattr(owner, target, _raise)
            assert panel.apply_plot_state(state) is True

    with subtests.test(msg="diagnostics without the QC module"):
        panel = _panel(qtbot)
        panel.set_frame(_results_frame())
        monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
        model = object()
        assert panel.set_diagnostics(model, "ols") is False
        assert panel._model is model
        assert "could not load the diagnostics module" in (
            panel._diagnostics_error)


# ---------------------------------------------------------------------------
# selection, level filter, colouring and the volcano axis
# ---------------------------------------------------------------------------

def test_the_panel_selects_filters_and_labels_what_it_shows(qtbot, monkeypatch,
                                                            subtests):
    """Keyed plots select, histograms narrow, the axis is named or defaults."""
    from spacr.localisation import ALL as ALL_COMPARTMENTS
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    results = _results_frame()

    with subtests.test(msg="the selection falls back to the last click"):
        panel = _panel(qtbot)
        panel.set_frame(results)
        panel.table.table.clearSelection()
        panel._selected_key = results["feature"].iloc[3]
        assert panel.selected_keys() == ["fraction:grna[3_1]"]

    with subtests.test(msg="with no level filter every key is reachable"):
        panel = _panel(qtbot)
        panel.set_frame(results)
        panel._level = ""
        assert panel._reachable(results["feature"].iloc[0]) is True

    with subtests.test(msg="colouring by every localisation"):
        panel = _panel(qtbot)
        panel.set_frame(results)
        panel.set_compartment(ALL_COMPARTMENTS)
        assert panel._compartment == ALL_COMPARTMENTS
        assert "all-localisations" not in panel._status
        assert "TAGM/LOPIT localisation" in panel._status

    with subtests.test(msg="histograms narrow where point plots select"):
        panel = _panel(qtbot)
        panel.set_frame(_guides(), "/runs/guides/results.csv")
        keyed = panel._keyed_plots()
        assert panel.volcano in keyed
        assert panel.p_values not in keyed
        assert panel.effect_distribution not in keyed
        panel.volcano.keys_selected.emit(["fraction:grna[G0_1]",
                                          "fraction:grna[G0_2]"])
        assert panel.selected_keys() == ["fraction:grna[G0_1]",
                                         "fraction:grna[G0_2]"]
        panel.p_values.keys_selected.emit(["fraction:grna[G0_1]"])
        table = panel.table.table
        visible = [row for row in range(table.rowCount())
                   if not table.isRowHidden(row)]
        assert len(visible) == 1
        texts = [table.item(visible[0], col).text()
                 if table.item(visible[0], col) else ""
                 for col in range(table.columnCount())]
        assert "fraction:grna[G0_1]" in texts

    permuted = _guides()
    permuted["permutation_p_value"] = [0.01, 0.4]
    permuted["standardized_marginal_effect"] = [0.11, 0.22]

    def _bottom(panel):
        return panel.volcano.plot.getPlotItem().getAxis("bottom").labelText

    with subtests.test(msg="a permutation axis is named"):
        panel = _panel(qtbot)
        panel.set_frame(permuted, "/runs/perm/results.csv")
        assert panel._analysis_path() == "permutation"
        assert _bottom(panel) == (
            "standardized marginal effect (partial correlation)")

    with subtests.test(msg="a volcano that cannot name its axis still draws"):
        older = _panel(qtbot)
        monkeypatch.delattr(VolcanoPlot, "name_the_effect")
        older.set_frame(permuted, "/runs/perm/results.csv")
        assert list(older.results_frame()["feature"]) == [
            "fraction:grna[G0_1]", "fraction:grna[G0_2]"]
        assert older.table.table.rowCount() == 2
        assert _bottom(older) == "coefficient"
