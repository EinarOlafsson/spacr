"""The hyperparameter panel: its controls, its clustering, its reports, and its settings window.

A search runs for hours and produces figures, clusterings and a table of
trials; none of the side jobs may end the run.  Each failure lands as an exact
sentence on the status line (or the preview, on a panel with no UMAP view),
with whatever did complete still on screen.  The guards for half-built panels
(an attribute the constructor creates later) are reached by deleting that
attribute, which is the only way to present one.

Offscreen and CPU-only: no UMAP is fitted, HDBSCAN is stubbed, and the
search backend is never run.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtWidgets import QDialogButtonBox, QTableWidgetItem  # noqa: E402

from spacr import umap_search                                   # noqa: E402
from spacr.hyperparam import (                                  # noqa: E402
    UMAP_CRITERIA, SearchResult, SearchSpace, Trial,
)
from spacr.qt.screens import hyperparam as hp                   # noqa: E402
from spacr.qt.screens.hyperparam import (                       # noqa: E402
    HyperparamPanel, _NumericTableItem,
)
from spacr.qt.widgets.sortable_table import table_item          # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """A modal dialog in a headless run hangs, and every error belongs inline."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened -- errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))


@pytest.fixture()
def make(qtbot, qt_theme_applied):
    """Build a fresh panel for an app key, offscreen."""
    def _build(app_key="umap"):
        widget = HyperparamPanel(app_key)
        qtbot.addWidget(widget)
        return widget
    return _build


def _col(widget, name):
    return widget.COLUMNS.index(name)


def _trial(index=0, score=0.9, *, embedding=False, n=12, **extra):
    metrics = dict(extra)
    if embedding:
        metrics["embedding"] = np.random.default_rng(index).normal(size=(n, 2))
    return Trial(params={"n_neighbors": 5 * (index + 1)}, score=score,
                 extra_metrics=metrics, duration=0.01, index=index)


def _result(trials, metric="trustworthiness", notes=()):
    space = SearchSpace({"n_neighbors": [t.params["n_neighbors"]
                                         for t in trials]})
    return SearchResult(trials=list(trials), best=trials[0], space=space,
                        metric=metric, notes=list(notes), partial=False)


def _tabled(panel, n=2):
    """A result with embeddings, written into the table, row 0 selected."""
    result = _result([_trial(i, 0.9 - 0.1 * i, embedding=True, n=24)
                      for i in range(n)])
    panel._result = result
    panel._rebuild_table(result)
    panel._table.setSortingEnabled(False)
    panel._table.selectRow(0)
    return result


def _stub_clustering(monkeypatch, labels):
    monkeypatch.setattr(umap_search, "cluster_embedding",
                        lambda embedding, **_kw: np.asarray(labels, dtype=int))


class _WalkRow:
    def __init__(self, size, n_clusters, silhouette=0.42, noise=0.1):
        self.min_cluster_size = size
        self.labels = np.array([0, 0, 1, 1, -1, 1, 0, 1, 0, 1, -1, 0])
        self.silhouette = silhouette
        self.score = 0.4
        self.n_clusters = n_clusters
        self.noise_fraction = noise


def test_the_criterion_explains_itself_and_heads_the_score_column(make, monkeypatch, subtests):
    """Each UMAP criterion writes its help and tooltip, toggles the objective box, and retitles the score header."""
    recommendations = {
        "trustworthiness": (
            "Best default for finding local structure without inventing "
            "apparent neighbours."),
        "continuity": (
            "Use when preserving existing neighbourhoods matters most; it may "
            "crowd unrelated points together."),
        "silhouette": (
            "Use only to test separation of labels you already have; it does "
            "not discover unknown structure."),
        "multi_objective": (
            "Recommended when you want unknown structure: inspect the Pareto "
            "front rather than treating the composite top row as a uniquely "
            "correct answer."),
    }
    panel = make()
    column = _col(panel, "score")
    for criterion, recommendation in recommendations.items():
        with subtests.test(msg=criterion):
            panel._criterion.setCurrentText(criterion)
            expected = (f"{criterion}: {UMAP_CRITERIA.get(criterion, '')} "
                        f"{recommendation}")
            assert panel._criterion_help.text() == expected
            assert panel._criterion.toolTip() == expected
            assert panel._multi_objective_controls.isEnabled() is (
                criterion == "multi_objective")
            header = panel._table.horizontalHeaderItem(column)
            title = criterion.replace("_", " ")
            assert header.text() == title
            assert header.toolTip() == (
                f"The value being optimised: {title}. Higher is better unless "
                "the criterion is one of the lower-is-better ones.")

    with subtests.test(msg="an unknown criterion has no detail"):
        panel._update_criterion_explanation("made_up")
        assert panel._criterion_help.text() == "made_up:  "

    with subtests.test(msg="no objective box yet still explains"):
        monkeypatch.delattr(panel, "_multi_objective_controls")
        panel._criterion.setCurrentText("continuity")
        assert not hasattr(panel, "_multi_objective_controls")
        assert panel._criterion_help.text() == (
            f"continuity: {UMAP_CRITERIA['continuity']} "
            f"{recommendations['continuity']}")

    with subtests.test(msg="a missing header item is not invented"):
        panel = make()
        panel._criterion.setCurrentText("continuity")
        taken = panel._table.takeHorizontalHeaderItem(column)
        panel._criterion.setCurrentText("silhouette")
        assert panel._table.horizontalHeaderItem(column) is None
        assert taken.text() == "continuity"

    with subtests.test(msg="a classifier criterion points at the tooltips"):
        panel = make("classify")
        panel._update_criterion_explanation("accuracy")
        assert panel._criterion_help.text() == (
            "accuracy ranks the trials. See the control tooltip and the "
            "per-trial score tooltips in the results table.")


def test_the_walk_toggle_collapses_grid_fields_and_restores_them(make, monkeypatch, subtests):
    """Walk mode keeps one centre value per field and gives the grid lists back, even on a half-built panel."""
    with subtests.test(msg="on and off again"):
        panel = make()
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        panel._value_edits["min_dist"].setText("0.25")
        panel._adaptive.setChecked(True)
        assert panel._walk_axes_button.isEnabled()
        assert panel._adaptive_n_step.isEnabled()
        assert panel._adaptive_improvement.isEnabled()
        assert not panel._mode.isEnabled()
        assert not panel._n_trials.isEnabled()
        assert panel._value_edits["n_neighbors"].text() == "5"
        assert panel._value_edits["min_dist"].text() == "0.25"
        assert panel._adaptive_grid_text == {"n_neighbors": "5, 15, 50"}

        panel._adaptive.setChecked(False)
        assert panel._value_edits["n_neighbors"].text() == "5, 15, 50"
        assert panel._mode.isEnabled()
        assert panel._n_trials.isEnabled()
        assert not panel._walk_axes_button.isEnabled()
        assert not panel._adaptive_n_step.isEnabled()

    with subtests.test(msg="the centre is the host's own value"):
        panel = make()
        panel.apply_settings({"n_neighbors": 30})
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        panel._adaptive.setChecked(True)
        assert panel._value_edits["n_neighbors"].text() == "30"

    with subtests.test(msg="missing optional run controls are skipped"):
        panel = make()
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        step = panel._adaptive_n_step
        mode = panel._mode
        for name in ("_adaptive_controls", "_walk_axes_button",
                     "_mode", "_n_trials"):
            monkeypatch.delattr(panel, name)
        panel._adaptive.setChecked(True)
        assert panel._adaptive_grid_text["n_neighbors"] == "5, 15, 50"
        assert panel._value_edits["n_neighbors"].text() == "5"
        assert not step.isEnabled()
        assert mode.isEnabled()

    with subtests.test(msg="a stale cache entry for a missing field"):
        panel = make()
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        panel._adaptive.setChecked(True)
        panel._adaptive_grid_text = {"ghost": "1, 2",
                                     "n_neighbors": "5, 15, 50"}
        panel._adaptive.setChecked(False)
        assert "ghost" not in panel._value_edits
        assert panel._value_edits["n_neighbors"].text() == "5, 15, 50"


def test_clustering_the_selected_map_records_labels_or_names_the_reason(make, monkeypatch, subtests):
    """Cluster and walk need a map, report a failure or an empty walk, and store what they chose on the trial."""
    for case_id, call, message in [
            ("cluster with nothing selected", "cluster_selected",
             "Select a completed UMAP before clustering it."),
            ("walk with nothing selected", "walk_selected_clusters",
             "Select a completed UMAP before walking clusters.")]:
        with subtests.test(msg=case_id):
            panel = make()
            assert getattr(panel, call)() is False
            assert panel._status.text() == message
            panel._displayed_trial = _trial(0)
            assert getattr(panel, call)() is False
            assert panel._status.text() == message

    def _refuse(*_a, **_k):
        raise RuntimeError("hdbscan is not installed")

    for case_id, name, call, message in [
            ("a clustering that fails", "cluster_embedding",
             "cluster_selected",
             "Could not cluster this UMAP: hdbscan is not installed"),
            ("a walk that fails", "walk_clusters", "walk_selected_clusters",
             "Could not walk cluster settings: hdbscan is not installed")]:
        with subtests.test(msg=case_id):
            panel = make()
            monkeypatch.setattr(umap_search, name, _refuse)
            trial = _trial(0, embedding=True)
            panel._displayed_trial = trial
            assert getattr(panel, call)() is False
            assert panel._status.text() == message
            assert "n_clusters" not in trial.extra_metrics

    with subtests.test(msg="a walk that finds no partition"):
        panel = make()
        monkeypatch.setattr(umap_search, "walk_clusters", lambda *a, **k: [])
        panel._displayed_trial = _trial(0, embedding=True)
        assert panel.walk_selected_clusters() is False
        assert panel._status.text() == (
            "The cluster walk produced no usable partition.")

    with subtests.test(msg="a walk that succeeds"):
        panel = make()
        seen = {}

        def _walk(embedding, min_cluster_sizes):
            seen["sizes"] = min_cluster_sizes
            return [_WalkRow(12, 2), _WalkRow(20, 3, silhouette=0.3)]

        monkeypatch.setattr(umap_search, "walk_clusters", _walk)
        panel._cluster_size.setValue(10)
        trial = _trial(0, embedding=True)
        panel._displayed_trial = trial
        assert panel.walk_selected_clusters() is True
        assert seen["sizes"] == (4, 6, 10, 17, 27)
        assert panel._cluster_size.value() == 12
        assert {key: trial.extra_metrics[key] for key in (
            "cluster_min_size", "cluster_silhouette", "cluster_score",
            "n_clusters", "cluster_noise_fraction")} == {
            "cluster_min_size": 12, "cluster_silhouette": 0.42,
            "cluster_score": 0.4, "n_clusters": 2,
            "cluster_noise_fraction": 0.1}
        assert [row["min_cluster_size"] for row in
                trial.extra_metrics["cluster_walk"]] == [12, 20]
        assert trial.extra_metrics["cluster_walk"][1] == {
            "min_cluster_size": 20, "silhouette": 0.3, "score": 0.4,
            "n_clusters": 3, "noise_fraction": 0.1}
        assert panel._status.text() == (
            "Cluster walk chose min_cluster_size=12: 2 clusters, 10.0% noise, "
            "silhouette 0.420.")

    with subtests.test(msg="clustering the selected row writes its cell"):
        panel = make()
        result = _tabled(panel)
        trial = panel.selected_trial()
        assert trial is result.trials[0]
        clusters = _col(panel, "clusters")
        _stub_clustering(monkeypatch, [0] * 12 + [1] * 10 + [-1] * 2)
        panel._cluster_size.setValue(7)
        assert panel.cluster_selected() is True
        assert panel._table.item(0, clusters).text() == "2"
        assert panel._table.item(1, clusters).text() == "-"
        assert trial.extra_metrics["n_clusters"] == 2
        assert trial.extra_metrics["cluster_min_size"] == 7
        assert trial.extra_metrics["cluster_noise_fraction"] == \
            pytest.approx(2 / 24)
        assert panel._status.text() == (
            "Clustered this stored map into 2 clusters; 8.3% of points are "
            "HDBSCAN noise.")

        taken = panel._table.takeItem(0, clusters)
        _stub_clustering(monkeypatch, [0] * 8 + [1] * 8 + [2] * 8)
        assert panel.cluster_selected() is True
        assert trial.extra_metrics["n_clusters"] == 3
        assert panel._table.item(0, clusters) is None
        assert taken.text() == "2"

        monkeypatch.delattr(panel, "_status")
        _stub_clustering(monkeypatch, [0] * 24)
        assert panel.cluster_selected() is True
        assert trial.extra_metrics["n_clusters"] == 1


def test_showing_a_trial_needs_stored_coordinates_and_a_umap_view(make, monkeypatch, subtests):
    """No embedding or no view returns False; a rejected embedding is a status line; a shown one enables clustering."""
    with subtests.test(msg="no stored coordinates"):
        panel = make()
        assert panel.show_trial(_trial(0)) is False
        assert panel._displayed_trial is None

    with subtests.test(msg="the viewer rejects the coordinates"):
        panel = make()

        def refuse(*_args, **_kwargs):
            raise ValueError("that is not a 2-D or 3-D embedding")

        monkeypatch.setattr(panel._umap_explorer.view, "set_embedding", refuse)
        assert panel.show_trial(_trial(0, embedding=True)) is False
        assert panel._status.text() == (
            "Could not display this UMAP: that is not a 2-D or 3-D embedding")
        assert panel._displayed_trial is None

    with subtests.test(msg="a shown trial carries its caption and backend"):
        panel = make()
        seen = {}
        monkeypatch.setattr(panel._umap_explorer.view, "set_embedding",
                            lambda embedding, **kw: seen.update(kw))
        trial = _trial(1, 0.5, embedding=True, backend="cuml")
        assert panel.show_trial(trial) is True
        assert seen == {"labels": None, "backend": "cuml",
                        "caption": "2D · n_neighbors=10 · 0.5000"}
        assert panel._displayed_trial is trial
        assert panel._cluster_btn.isEnabled()
        assert panel._cluster_walk_btn.isEnabled()

    with subtests.test(msg="a classifier panel has no UMAP to show"):
        panel = make("classify")
        assert panel.show_trial(_trial(0, embedding=True)) is False
        assert panel.open_umap_grid() is None


def test_a_finished_search_reports_failure_and_mixed_backends(make, subtests):
    """A missing result or an error is a failure on status and view; mixing UMAP backends is flagged."""
    for case_id, app_key, result, err, status in [
            ("umap: no result", "umap", None, "",
             "Search failed: the worker returned no result."),
            ("umap: an error", "umap", None, "out of memory",
             "Search failed: out of memory"),
            ("classify: no result", "classify", None, "",
             "Search failed: the worker returned no result."),
            ("classify: an error", "classify", None, "boom",
             "Search failed: boom")]:
        with subtests.test(msg=case_id):
            panel = make(app_key)
            panel._on_search_done(result, err)
            assert panel._status.text() == status
            assert panel._result is None
            if app_key == "classify":
                assert panel._preview.text() == "Search failed."

    with subtests.test(msg="two UMAP backends are not one walk"):
        panel = make()
        trials = [_trial(0, 0.9, backend="cpu"), _trial(1, 0.8, backend="cuml")]
        panel._on_search_done(_result(trials, notes=["hand built"]), "")
        assert panel._notes.text() == "• hand built"
        assert panel._status.text().endswith(
            "MIXED BACKENDS — these rows compare different UMAP "
            "implementations (cpu, cuml) as well as different settings; do "
            "not rank them as one walk.")
        assert panel._table.rowCount() == 2

    with subtests.test(msg="one backend is not flagged"):
        panel = make()
        trials = [_trial(0, 0.9, backend="cpu"), _trial(1, 0.8, backend="cpu")]
        panel._on_search_done(_result(trials), "")
        assert "MIXED BACKENDS" not in panel._status.text()


def test_the_preview_reports_why_it_is_not_a_picture(make, monkeypatch, subtests):
    """A failed build, nothing to plot, and a figure that will not close each leave the right preview."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def _cannot(*_a, **_k):
        raise RuntimeError("no axes could be laid out")

    for case_id, builder, text in [
            ("the figure cannot be built", _cannot,
             "Could not draw the preview: no axes could be laid out"),
            ("no trial produced a score", lambda *a, **k: None,
             "No trial produced a score to plot.")]:
        with subtests.test(msg=case_id):
            panel = make("classify")
            monkeypatch.setattr(hp, "build_panel_figure", builder)
            panel._draw_preview(_result([_trial(0), _trial(1)]))
            assert panel._preview.text() == text
            assert panel._preview_figure is None

    with subtests.test(msg="a figure that will not close is still drawn"):
        panel = make("classify")
        figure = Figure(figsize=(2, 2))
        seen = {}

        def _build(result, max_panels):
            seen["max_panels"] = max_panels
            return figure

        monkeypatch.setattr(hp, "build_panel_figure", _build)

        def refuse(_fig):
            raise RuntimeError("this figure is not in the pyplot registry")

        monkeypatch.setattr(plt, "close", refuse)
        panel._max_panels.setValue(4)
        panel._draw_preview(_result([_trial(0)]))
        assert seen == {"max_panels": 4}
        assert panel._preview_figure is figure
        assert not panel._preview.pixmap().isNull()
        assert panel._preview.text() == ""


def test_a_trial_figure_opens_only_when_there_is_a_file(make, monkeypatch, tmp_path, subtests):
    """No grid or no path opens nothing; a viewer that fails leaves the status line alone."""
    import PySide6.QtGui as QtGui

    png = tmp_path / "trial.png"
    png.write_bytes(b"x")

    class _Grid:
        def __init__(self, path):
            self.path = path
            self.asked = []

        def figure_path(self, index):
            self.asked.append(index)
            return self.path

    for case_id, grid, opener, opened_paths in [
            ("no figure grid", None, None, []),
            ("a cell with no figure", _Grid(""), None, []),
            ("a cell with a figure", _Grid(str(png)), None, [str(png)]),
            ("a viewer that will not open", _Grid(str(png)), "refuse", [])]:
        with subtests.test(msg=case_id):
            panel = make()
            opened = []

            def _open(url, opener=opener):
                if opener == "refuse":
                    raise RuntimeError("no application is registered")
                opened.append(url.toLocalFile())

            monkeypatch.setattr(panel, "_figure_grid", grid, raising=False)
            monkeypatch.setattr(QtGui.QDesktopServices, "openUrl",
                                staticmethod(_open))
            panel._status.setText("3 of 8 configurations evaluated")
            panel._open_trial_figure(2)
            assert opened == opened_paths
            if grid is not None:
                assert grid.asked == [2]
            assert panel._status.text() == "3 of 8 configurations evaluated"


def test_the_settings_window_is_built_once_with_its_own_buttons(make, qtbot, monkeypatch, subtests):
    """A second open raises the same window; the footer has Run and Propagate, with or without a Close button; empty tabs are dropped."""
    with subtests.test(msg="asking twice raises the open window"):
        panel = make()
        panel.open_settings()
        first = panel._settings_dialog
        qtbot.addWidget(first)
        qtbot.waitExposed(first)
        assert first.windowTitle() == "UMAP settings"
        panel.open_settings()
        assert panel._settings_dialog is first

    with subtests.test(msg="a classifier window has the standard footer"):
        panel = make("classify")
        panel.open_settings()
        dialog = panel._settings_dialog
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Hyperparameter search settings"
        assert [dialog._tabs.tabText(i)
                for i in range(dialog._tabs.count())] == ["Search"]
        assert dialog._close_btn is not None
        assert dialog._close_btn.icon().isNull()
        assert dialog._run_btn.text() == "Run search"
        assert dialog._run_btn.isEnabled()
        assert dialog._propagate.text() == "Propagate settings"
        assert panel._run_btn.isHidden() and panel._apply_btn.isHidden()

    with subtests.test(msg="a button box with no Close button"):
        class _NoCloseButtonBox(QDialogButtonBox):
            def button(self, _which):
                return None

        monkeypatch.setattr(hp, "QDialogButtonBox", _NoCloseButtonBox)
        panel = make("classify")
        panel.open_settings()
        dialog = panel._settings_dialog
        qtbot.addWidget(dialog)
        assert dialog._close_btn is None
        assert dialog._run_btn.text() == "Run search"
        assert dialog._propagate.text() == "Propagate settings"
        monkeypatch.undo()

    with subtests.test(msg="a tab with no controls is not added"):
        from spacr.qt.screens import settings_model

        real = settings_model.SettingsWidgets.build_sections
        kept = []

        def only_the_first(self):
            sections = real(self)
            kept.append(sections[0][0])
            return sections[:1]

        monkeypatch.setattr(settings_model.SettingsWidgets, "build_sections",
                            only_the_first)
        panel = make()
        panel.open_settings()
        dialog = panel._settings_dialog
        qtbot.addWidget(dialog)
        titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
        tab_of = {"Input Data": "Data", "Dimensionality Reduction": "Reducer",
                  "UMAP": "Reducer", "t-SNE": "Reducer", "PCA": "Reducer",
                  "Isomap": "Reducer", "Spectral Embedding": "Reducer",
                  "Clustering": "Clustering", "Points & Images": "Appearance",
                  "Canvas & Output": "Appearance",
                  "Plate & Batch Correction": "Batch", "Runtime": "Runtime"}
        assert titles == ["Search", tab_of[kept[0]]]
        assert set(dialog._module_model._widgets) == dialog._module_keys
        assert dialog._module_keys


def test_propagation_sends_parsed_fields_and_the_selected_row(make, qtbot, subtests):
    """No callback sends nothing; a typo is left out; a selected row overrides the fields."""
    def _dialog(panel):
        panel.open_settings()
        dialog = panel._settings_dialog
        qtbot.addWidget(dialog)
        return dialog

    with subtests.test(msg="nowhere to send it"):
        panel = make("classify")
        panel.set_apply_callback(None)
        panel._status.setText("before")
        _dialog(panel).propagate_settings()
        assert panel._status.text() == "before"

    with subtests.test(msg="a typo is left out, the rest is sent"):
        panel = make("classify")
        written = []
        panel.set_apply_callback(written.append)
        panel._value_edits["learning_rate"].setText("fast")
        panel._value_edits["dropout_rate"].setText("0.1")
        panel._value_edits["epochs"].setText("10, 20")
        panel._value_edits["weight_decay"].setText("")
        _dialog(panel).propagate_settings()
        assert written == [{"dropout_rate": 0.1}]
        assert panel._status.text() == (
            "Propagated UMAP search and graph settings to the module.")

    with subtests.test(msg="a selected row wins over the fields"):
        panel = make("classify")
        written = []
        panel.set_apply_callback(written.append)
        panel._value_edits["dropout_rate"].setText("0.3")
        chosen = Trial(params={"dropout_rate": 0.5, "epochs": 25}, score=0.9,
                       extra_metrics={}, duration=0.01, index=0)
        panel._on_search_done(SearchResult(
            trials=[chosen], best=chosen, metric="accuracy", notes=[],
            space=SearchSpace({"dropout_rate": [0.5], "epochs": [25]})), "")
        panel._table.selectRow(0)
        _dialog(panel).propagate_settings()
        assert written[0]["dropout_rate"] == 0.5
        assert written[0]["epochs"] == 25

    with subtests.test(msg="a umap window also sends its module tabs"):
        panel = make()
        written = []
        panel.set_apply_callback(written.append)
        panel._value_edits["n_neighbors"].setText("fifteen")
        panel._value_edits["min_dist"].setText("0.1")
        dialog = _dialog(panel)
        dialog.propagate_settings()
        collected = dialog._module_model.collect()
        module = {key: collected[key] for key in dialog._module_keys
                  if key in collected}
        assert written == [{**module, "min_dist": 0.1,
                            "metric": panel._value_edits["metric"].text()}]
        assert written[0]["n_neighbors"] == collected["n_neighbors"]
        assert panel._status.text() == (
            "Propagated UMAP search and graph settings to the module.")


def test_closing_survives_a_worker_that_will_not_stop(make):
    """Close finishes even when the sweep worker refuses to be stopped."""
    panel = make()
    calls = []

    class _Stubborn:
        def request_stop(self):
            calls.append("stop")
            raise RuntimeError("this worker will not stop")

    panel._worker = _Stubborn()
    panel.show()
    assert panel.close() is True
    assert calls == ["stop"]
    assert panel.isVisible() is False
    panel._worker = None


def test_numeric_cells_sort_on_their_numbers(subtests, qapp):
    """Trial cells compare by number with each other and by text against plain cells."""
    for case_id, left, right, expected in [
            ("0.1 below 0.9", _NumericTableItem("0.1", 0.1),
             _NumericTableItem("0.9", 0.9), True),
            ("0.9 not below 0.1", _NumericTableItem("0.9", 0.9),
             _NumericTableItem("0.1", 0.1), False),
            ("0.10 below 0.9 despite its text", _NumericTableItem("0.10", 0.10),
             _NumericTableItem("0.9", 0.9), True),
            ("against a plain cell, by text", _NumericTableItem("0.9000", 0.9),
             QTableWidgetItem("zzz"), True),
            ("a plain cell above, by text", _NumericTableItem("0.9000", 0.9),
             QTableWidgetItem("0.1"), False)]:
        with subtests.test(msg=case_id):
            assert (left < right) is expected

    with subtests.test(msg="sorted orders by number"):
        cells = [_NumericTableItem("0.5", 0.5), _NumericTableItem("0.05", 0.05),
                 _NumericTableItem("0.75", 0.75), _NumericTableItem("10", 10.0)]
        assert [cell.text() for cell in sorted(cells)] == [
            "0.05", "0.5", "0.75", "10"]

    with subtests.test(msg="the cell is the shared sortable item"):
        cell = _NumericTableItem("0.812", 0.812)
        assert type(cell) is type(table_item("0.812", key=0.812))
        assert cell.text() == "0.812"
        assert cell.data(Qt.DisplayRole) == "0.812"
