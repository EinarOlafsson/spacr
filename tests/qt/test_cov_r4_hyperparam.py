"""Hyperparameter-panel guards a healthy panel never trips.

Pins the branches this screen carries for controls that may not be there and
for rows that may not carry what the panel put on them: the criterion
explanation without its multi-objective box; ``_set_control_text`` handed a
choice the combo does not offer, and handed a widget that is neither field
nor combo; the Walk toggle on a panel missing its optional run controls, and
its restore cache naming a field that is gone; clustering with no status
line to report on; the settings popup whose footer has no Run button; the
score column with no header item; a table row written without a trial; the
selection falling back to the best row when the selected one carries no
configuration; the cluster-count cell that has been taken out; and the
button box that yields no Close button.

Offscreen and CPU-only: no UMAP is fitted and no model is trained -- the
search backend is injected and the HDBSCAN call is stubbed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import numpy as np                                              # noqa: E402
from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtWidgets import (                                 # noqa: E402
    QDialogButtonBox, QLabel, QLineEdit,
)

from spacr.hyperparam import SearchResult, SearchSpace, Trial   # noqa: E402
from spacr.qt.screens import hyperparam as hp                   # noqa: E402
from spacr.qt.screens.hyperparam import (                       # noqa: E402
    HyperparamPanel, SearchRequest,
)

pytestmark = pytest.mark.qt


@pytest.fixture()
def panel(qtbot, qt_theme_applied):
    """A UMAP panel, built offscreen."""
    widget = HyperparamPanel("umap")
    qtbot.addWidget(widget)
    return widget


def _col(widget: HyperparamPanel, name: str) -> int:
    """Column index by NAME -- the positions move, the names do not."""
    return widget.COLUMNS.index(name)


def _result(*, n: int = 3, embeddings: bool = False) -> SearchResult:
    """A hand-built result: n_neighbors 5, 10, 15 at falling scores."""
    rng = np.random.default_rng(0)
    trials = []
    for i in range(n):
        extra = {}
        if embeddings:
            extra["embedding"] = rng.normal(size=(24, 2))
        trials.append(Trial(params={"n_neighbors": 5 * (i + 1)},
                            score=0.9 - 0.1 * i, extra_metrics=extra,
                            duration=0.01, index=i))
    return SearchResult(
        trials=trials, best=trials[0], metric="trustworthiness",
        space=SearchSpace({"n_neighbors": [5 * (i + 1) for i in range(n)]}),
        notes=["hand built"])


def _stub_clustering(monkeypatch, labels) -> None:
    """Replace HDBSCAN with fixed labels; the table cell is what is tested."""
    import spacr.umap_search as umap_search
    monkeypatch.setattr(
        umap_search, "cluster_embedding",
        lambda embedding, **_kw: np.asarray(labels, dtype=int))


def _scripted_search(scores):
    """An injectable backend that emits one trial per configuration."""
    def _search(request: SearchRequest, on_trial, should_stop):
        combos = request.space.grid()
        result = SearchResult(space=request.space, metric=request.criterion,
                              notes=["scripted"])
        for i, params in enumerate(combos):
            if should_stop():
                result.partial = True
                break
            trial = Trial(params=dict(params), index=i,
                          score=scores[i % len(scores)])
            result.trials.append(trial)
            on_trial(trial, i + 1, len(combos))
        ok = [t for t in result.trials if t.ok]
        result.best = max(ok, key=lambda t: t.score) if ok else None
        return result
    return _search


# ---------------------------------------------------------------------------
# The search-space controls
# ---------------------------------------------------------------------------

def test_the_criterion_explanation_survives_a_missing_objective_box(
        panel, monkeypatch):
    """The help text is written whether or not the weights box is there."""
    panel._criterion.setCurrentText("silhouette")
    assert not panel._multi_objective_controls.isEnabled()
    panel._criterion.setCurrentText("multi_objective")
    assert panel._multi_objective_controls.isEnabled()

    # The panel creates `_multi_objective_controls` a hundred lines after it
    # connects this slot to the combo, so the getattr guard stands for "the
    # criterion changed before the box existed" -- a state no public call can
    # reach once __init__ has returned. Deleting the attribute is the only
    # way in.
    monkeypatch.delattr(panel, "_multi_objective_controls")
    panel._criterion.setCurrentText("continuity")

    assert not hasattr(panel, "_multi_objective_controls")
    assert panel._criterion_help.text().startswith("continuity:")
    assert "crowd unrelated points" in panel._criterion_help.text()
    assert panel._criterion.toolTip() == panel._criterion_help.text()


def test_a_metric_the_combo_does_not_offer_is_not_invented(panel):
    """A fixed-choice control is seeded only with a choice it actually has."""
    metric = panel._value_edits["metric"]

    panel.apply_settings({"metric": "cosine"})
    assert metric.currentText() == "cosine"

    assert metric.findText("wombat") == -1
    panel.apply_settings({"metric": "wombat"})

    assert metric.currentText() == "cosine"


def test_a_control_that_is_neither_a_field_nor_a_combo_is_left_alone(qtbot):
    """Only a QLineEdit or a QComboBox is written to.

    Every entry in ``_value_edits`` is one or the other, so no caller inside
    the panel can hand this static helper a third kind of widget; it is
    called directly with one.
    """
    field = QLineEdit()
    qtbot.addWidget(field)
    label = QLabel("untouched")
    qtbot.addWidget(label)

    HyperparamPanel._set_control_text(field, "5, 15")
    HyperparamPanel._set_control_text(label, "5, 15")

    assert field.text() == "5, 15"
    assert label.text() == "untouched"
    assert HyperparamPanel._control_text(label) == ""


# ---------------------------------------------------------------------------
# The Walk toggle
# ---------------------------------------------------------------------------

def test_the_walk_toggle_works_without_its_optional_run_controls(
        panel, monkeypatch):
    """Missing mode, trials, Axes and adaptive boxes cost no grid capture.

    All four are built after the toggle is connected, so their getattr and
    hasattr guards describe a half-built panel; deleting the attributes is
    the only way to present one.
    """
    panel._value_edits["n_neighbors"].setText("5, 15, 50")

    panel._adaptive.setChecked(True)
    assert panel._walk_axes_button.isEnabled()
    assert panel._adaptive_n_step.isEnabled()
    assert not panel._mode.isEnabled()
    assert not panel._n_trials.isEnabled()
    assert panel._value_edits["n_neighbors"].text() == "5"

    panel._adaptive.setChecked(False)
    assert panel._value_edits["n_neighbors"].text() == "5, 15, 50"
    assert panel._mode.isEnabled()
    assert panel._n_trials.isEnabled()
    assert not panel._walk_axes_button.isEnabled()
    assert not panel._adaptive_n_step.isEnabled()

    step = panel._adaptive_n_step
    for name in ("_adaptive_controls", "_walk_axes_button",
                 "_mode", "_n_trials"):
        monkeypatch.delattr(panel, name)

    panel._adaptive.setChecked(True)

    # The grid text is still captured and the field still collapses to one
    # centre value, while the fields belonging to the absent boxes keep the
    # state they had -- the guards skipped them instead of raising.
    assert panel._adaptive_grid_text["n_neighbors"] == "5, 15, 50"
    assert panel._value_edits["n_neighbors"].text() == "5"
    assert not step.isEnabled()


def test_grid_text_for_a_field_the_panel_no_longer_has_is_skipped(panel):
    """A stale cache entry is passed over, not written to a missing field."""
    panel._value_edits["n_neighbors"].setText("5, 15, 50")
    panel._adaptive.setChecked(True)
    assert panel._value_edits["n_neighbors"].text() == "5"

    # Nothing public writes this cache, and only a stale entry can name a
    # field the panel does not have. The ghost is FIRST, so a restored
    # n_neighbors proves the loop carried on past it.
    panel._adaptive_grid_text = {"ghost": "1, 2", "n_neighbors": "5, 15, 50"}

    panel._adaptive.setChecked(False)

    assert "ghost" not in panel._value_edits
    assert panel._value_edits["n_neighbors"].text() == "5, 15, 50"


# ---------------------------------------------------------------------------
# Status, popup footer and column heading
# ---------------------------------------------------------------------------

def test_clustering_still_records_its_labels_without_a_status_line(
        panel, monkeypatch):
    """The cluster count lands on the trial even with no label to report on.

    ``_status`` is created at the very end of ``_build_ui``; its getattr
    guard covers a report arriving before that, which nothing public can
    arrange once the panel exists.
    """
    result = _result(n=1, embeddings=True)
    panel._result = result
    panel._rebuild_table(result)
    panel._table.selectRow(0)
    trial = panel.selected_trial()
    assert trial is result.trials[0]

    _stub_clustering(monkeypatch, [0] * 12 + [1] * 12)
    assert panel.cluster_selected() is True
    assert "into 2 clusters" in panel._status.text()

    monkeypatch.delattr(panel, "_status")
    _stub_clustering(monkeypatch, [0] * 8 + [1] * 8 + [2] * 8)

    assert panel.cluster_selected() is True
    assert trial.extra_metrics["n_clusters"] == 3


def test_a_settings_popup_without_a_footer_run_still_locks_the_panel(
        panel, qtbot, monkeypatch):
    """A footer Run is disabled while a sweep runs; its absence costs none."""
    panel._value_edits["n_neighbors"].setText("5, 15")
    panel._value_edits["min_dist"].setText("")
    panel.set_search_fn(_scripted_search([0.5, 0.6]))
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(panel.search_finished, timeout=5000):
        assert panel.run_search() is True
        assert not dialog._run_btn.isEnabled()
        assert not panel._run_btn.isEnabled()
    assert dialog._run_btn.isEnabled()

    # The popup builds its footer Run in __init__, so only a dialog caught
    # mid-construction lacks one; delete the attribute to stand in for that.
    monkeypatch.delattr(dialog, "_run_btn")

    with qtbot.waitSignal(panel.search_finished, timeout=5000):
        assert panel.run_search() is True
        assert not panel._run_btn.isEnabled()
    assert panel._run_btn.isEnabled()


def test_a_score_column_with_no_header_item_is_not_retitled(panel):
    """The criterion renames the header it has, and invents none when gone."""
    column = _col(panel, "score")

    panel._criterion.setCurrentText("continuity")
    header = panel._table.horizontalHeaderItem(column)
    assert header.text() == "continuity"
    assert "The value being optimised: continuity" in header.toolTip()

    taken = panel._table.takeHorizontalHeaderItem(column)
    panel._criterion.setCurrentText("silhouette")

    assert panel._table.horizontalHeaderItem(column) is None
    assert taken.text() == "continuity"


# ---------------------------------------------------------------------------
# The results table
# ---------------------------------------------------------------------------

def test_a_row_written_without_a_trial_shows_dashes_and_stashes_no_trial(
        panel):
    """The trial-less row form fills backend and clusters with a dash.

    Every caller in the module passes a trial (``_on_trial_ready`` and both
    ``_rebuild_table`` loops), so the declared default is exercised here
    directly, against the same helper writing a full row.
    """
    panel._table.setSortingEnabled(False)      # keep the two rows in place
    trial = Trial(params={"n_neighbors": 5}, score=0.9, index=0,
                  extra_metrics={"backend": "cuml", "n_clusters": 4})

    panel._table.insertRow(0)
    panel._set_row(0, "1", "0.9000", "-", "-", "n_neighbors=5", "ok",
                   {"n_neighbors": 5}, None, trial)
    panel._table.insertRow(1)
    panel._set_row(1, "2", "0.8000", "-", "-", "n_neighbors=15", "ok",
                   {"n_neighbors": 15}, None)

    assert panel._table.item(0, _col(panel, "backend")).text() == "cuml"
    assert panel._table.item(0, _col(panel, "clusters")).text() == "4"
    assert panel._table.item(0, 0).data(Qt.UserRole + 2) is trial

    assert panel._table.item(1, _col(panel, "backend")).text() == "-"
    assert panel._table.item(1, _col(panel, "clusters")).text() == "-"
    assert panel._table.item(1, 0).data(Qt.UserRole) == {"n_neighbors": 15}
    assert panel._table.item(1, 0).data(Qt.UserRole + 1) is None
    assert panel._table.item(1, 0).data(Qt.UserRole + 2) is None


def test_a_selected_row_that_carries_nothing_falls_back_to_the_best(panel):
    """A row with no dict on it -- or no first cell -- yields the best."""
    result = _result(n=3)
    panel._result = result
    panel._rebuild_table(result)
    panel._table.setSortingEnabled(False)      # no re-sort mid-mutation
    panel._table.selectRow(1)
    assert panel._table.item(
        1, _col(panel, "parameters")).text() == "n_neighbors=10"
    assert panel.selected_params() == {"n_neighbors": 10}

    panel._table.item(1, 0).setData(Qt.UserRole, "not a configuration")
    assert panel.selected_params() == {"n_neighbors": 5}

    taken = panel._table.takeItem(1, 0)
    assert taken is not None
    assert panel._table.item(1, 0) is None
    assert panel.selected_params() == {"n_neighbors": 5}


def test_a_cluster_count_with_no_cell_to_write_in_is_still_recorded(
        panel, monkeypatch):
    """Clustering updates the row's cell when there is one, and the trial
    either way."""
    result = _result(n=2, embeddings=True)
    panel._result = result
    panel._rebuild_table(result)
    panel._table.setSortingEnabled(False)
    panel._table.selectRow(0)
    trial = panel.selected_trial()
    clusters = _col(panel, "clusters")

    _stub_clustering(monkeypatch, [0] * 12 + [1] * 12)
    assert panel.cluster_selected() is True
    assert panel._table.item(0, clusters).text() == "2"

    taken = panel._table.takeItem(0, clusters)
    _stub_clustering(monkeypatch, [0] * 8 + [1] * 8 + [2] * 8)
    assert panel.cluster_selected() is True

    assert trial.extra_metrics["n_clusters"] == 3
    assert panel._table.item(0, clusters) is None
    assert taken.text() == "2"


# ---------------------------------------------------------------------------
# The settings popup's button box
# ---------------------------------------------------------------------------

def test_a_button_box_with_no_close_button_still_gets_run_and_propagate(
        qtbot, qt_theme_applied, monkeypatch):
    """The Close button loses its platform icon; a box without one is fine.

    Qt always returns the standard button the box was constructed with, so
    nothing in production makes ``buttons.button(Close)`` None -- the guard
    is only reachable from a button box that declines to hand one back, and
    one is injected here so the guard is exercised against the real dialog.
    """
    panel = HyperparamPanel("classify")
    qtbot.addWidget(panel)
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    assert dialog._close_btn is not None
    assert dialog._close_btn.icon().isNull()

    class _NoCloseButtonBox(QDialogButtonBox):
        """A button box that reports no standard Close button."""

        def button(self, _which):
            return None

    monkeypatch.setattr(hp, "QDialogButtonBox", _NoCloseButtonBox)
    stripped_panel = HyperparamPanel("classify")
    qtbot.addWidget(stripped_panel)
    stripped_panel.open_settings()
    stripped = stripped_panel._settings_dialog
    qtbot.addWidget(stripped)

    assert stripped._close_btn is None
    assert stripped._run_btn.text() == "Run search"
    assert stripped._propagate.text() == "Propagate settings"
