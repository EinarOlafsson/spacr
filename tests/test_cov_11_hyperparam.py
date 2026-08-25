"""The sweep panel survives every part of itself that can fail.

A hyperparameter search runs for hours and produces figures, clusterings and
a table of trials. None of those side jobs may be able to end the run: a
figure that will not render, a clustering library that is not installed, a
worker that dies before it returns anything. Each of those has to land as a
sentence on the status line, with whatever DID complete still on screen.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog

from spacr.hyperparam import SearchResult, SearchSpace, Trial
from spacr.qt.screens import hyperparam as hp
from spacr.qt.screens.hyperparam import (
    HyperparamPanel,
    SearchRequest,
    _FixedChoiceCombo,
    _NumericTableItem,
    _parse_walk_start,
    _SearchWorker,
    criteria_disagree,
    figure_to_pixmap,
    format_scores,
)


@pytest.fixture(autouse=True)
def _no_modal_dialogs(qapp, monkeypatch):
    """A modal dialog in a headless run hangs forever, so ban them here.

    Every error this panel can produce belongs on its inline status label.
    ``qapp`` is required by every test in this module: constructing a
    ``QPixmap`` without a live application aborts the process.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened -- errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))


@pytest.fixture()
def panel(qtbot):
    view = HyperparamPanel("umap")
    qtbot.addWidget(view)
    return view


def _trial(index=0, score=0.9, *, embedding=False, n=12, **extra):
    metrics = dict(extra)
    if embedding:
        metrics["embedding"] = np.random.default_rng(index).normal(size=(n, 2))
    return Trial(params={"n_neighbors": 5 * (index + 1)}, score=score,
                 extra_metrics=metrics, duration=0.01, index=index)


def _result(trials, metric="trustworthiness", partial=False):
    space = SearchSpace({"n_neighbors": [t.params["n_neighbors"]
                                         for t in trials]})
    return SearchResult(trials=list(trials), best=trials[0], space=space,
                        metric=metric, notes=[], partial=partial)


# ---------------------------------------------------------------------------
# Reporting a trial
# ---------------------------------------------------------------------------

def test_the_extra_settings_a_trial_ran_with_are_listed_with_its_scores():
    """A score is only interpretable next to what produced it.

    Two trials at the same trustworthiness under different cluster methods
    are not the same result, and the report has to say which was which.
    """
    trial = _trial(0, 0.87, cluster_structure_method="hdbscan",
                   cluster_counts=[4, 5], objective_weights={"trust": 1.0})

    text = format_scores(trial)

    assert "cluster_structure_method = hdbscan" in text
    assert "cluster_counts = [4, 5]" in text
    assert "objective_weights = {'trust': 1.0}" in text


def test_an_empty_extra_setting_is_not_listed():
    """The control: blanks would be noise in every report."""
    text = format_scores(_trial(0, 0.87, cluster_counts=[],
                                cluster_structure_method=""))

    assert "cluster_counts" not in text
    assert "cluster_structure_method" not in text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_render_becomes_an_empty_pixmap():
    """A failed render is a blank preview, not a dead sweep.

    The figure is decoration; the trials it summarises took hours.
    """
    class _Unrenderable:
        def get_facecolor(self):
            return "white"

        def savefig(self, *_args, **_kwargs):
            raise RuntimeError("this figure cannot be written")

    assert figure_to_pixmap(_Unrenderable()).isNull()


def test_a_preview_that_cannot_be_built_says_so_under_the_table(panel,
                                                                monkeypatch):
    """The finished sweep is still in the table when its summary fails."""
    def refuse(*_args, **_kwargs):
        raise RuntimeError("no axes could be laid out")

    monkeypatch.setattr(hp, "build_panel_figure", refuse)

    panel._draw_preview(_result([_trial(0), _trial(1)]))

    assert "Could not draw the preview" in panel._preview.text()
    assert "no axes could be laid out" in panel._preview.text()


def test_a_sweep_with_nothing_to_plot_says_that_instead(panel, monkeypatch):
    """No scored trial is a different message from a failed render."""
    monkeypatch.setattr(hp, "build_panel_figure", lambda *a, **k: None)

    panel._draw_preview(_result([_trial(0)]))

    assert panel._preview.text() == "No trial produced a score to plot."


def test_a_preview_that_rasterises_to_nothing_says_so(panel, monkeypatch):
    """A figure that builds but will not rasterise is its own failure."""
    from matplotlib.figure import Figure

    monkeypatch.setattr(hp, "build_panel_figure",
                        lambda *a, **k: Figure(figsize=(1, 1)))
    monkeypatch.setattr(hp, "figure_to_pixmap",
                        lambda _fig: hp.QPixmap())

    panel._draw_preview(_result([_trial(0)]))

    assert panel._preview.text() == "Could not render the preview."


def test_a_trial_figure_that_will_not_be_placed_does_not_stop_the_table(
        panel, monkeypatch, tmp_path):
    """The row still lands when the thumbnail grid rejects the image."""
    class _Grid:
        def add_figure(self, *_args, **_kwargs):
            raise RuntimeError("the grid has no cell for that")

        def clear(self):
            pass

    monkeypatch.setattr(panel, "_figure_grid", _Grid(), raising=False)
    png = tmp_path / "trial.png"
    png.write_bytes(b"not really a png")

    panel._on_trial_ready(_trial(0), 1, 3, str(png))

    assert panel._table.rowCount() == 1
    assert "1 of 3 configurations evaluated" in panel._status.text()


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _request(**kwargs):
    kwargs.setdefault("space", SearchSpace({"n_neighbors": [5, 10]}))
    return SearchRequest(**kwargs)


def test_a_search_that_raises_is_reported_rather_than_lost(qtbot):
    """The worker turns any exception into a message the panel can show.

    A sweep dying silently leaves Run disabled and Stop enabled forever.
    """
    def explode(_request, _on_trial, _should_stop):
        raise RuntimeError("the reducer refused these settings")

    worker = _SearchWorker(_request(), explode)

    worker.run()

    assert worker.result is None
    assert worker.error == "RuntimeError: the reducer refused these settings"
    assert worker.completion_ready is True


def test_a_search_that_finishes_stores_its_result(qtbot):
    """The control: the same path carries a real result through."""
    answer = _result([_trial(0)])

    worker = _SearchWorker(_request(), lambda *a: answer)
    worker.run()

    assert worker.result is answer
    assert worker.error == ""
    assert worker.completion_ready is True


def test_a_trial_whose_figure_fails_is_still_announced(qtbot, tmp_path,
                                                       monkeypatch):
    """The figure is rendered on the worker thread, and may fail there.

    Losing the trial because its picture failed would lose the measurement
    the sweep exists to make.
    """
    def refuse(*_args, **_kwargs):
        raise RuntimeError("the figure could not be drawn")

    monkeypatch.setattr(hp, "render_trial_figure", refuse)

    worker = _SearchWorker(_request(app_key="classify"), lambda *a: None)
    monkeypatch.setattr(worker, "_figure_dir", tmp_path, raising=False)
    seen = []
    worker.trial_ready.connect(lambda *args: seen.append(args))

    worker._emit_trial(_trial(0, embedding=True), 1, 2)

    assert len(seen) == 1
    assert seen[0][3] == ""


def test_a_trial_figure_that_is_written_travels_with_the_trial(qtbot, tmp_path,
                                                               monkeypatch):
    """The control: a successful render hands the path to the panel."""
    monkeypatch.setattr(hp, "render_trial_figure",
                        lambda _t, _m, path: True)

    worker = _SearchWorker(_request(app_key="classify"), lambda *a: None)
    monkeypatch.setattr(worker, "_figure_dir", tmp_path, raising=False)
    seen = []
    worker.trial_ready.connect(lambda *args: seen.append(args))

    worker._emit_trial(_trial(3, embedding=True), 1, 2)

    assert seen[0][3] == str(tmp_path / "trial_0003.png")


def test_a_worker_that_exits_without_a_result_says_so(panel, qtbot):
    """A thread that dies before finishing must not look like a clean sweep."""
    worker = _SearchWorker(_request(), lambda *a: None, panel)
    panel._worker = worker

    panel._on_worker_finished()

    assert panel._worker is None
    assert "Search worker exited without returning a result" in \
        panel._status.text()


def test_a_second_finished_signal_from_a_gone_worker_is_ignored(panel, qtbot):
    """Only the first completion is consumed; a repeat must change nothing.

    Re-running the completion path would re-enable Run and rewrite the
    status line for a sweep that has already been reported.
    """
    panel._worker = None
    panel._status.setText("Search finished.")

    panel._on_worker_finished()

    assert panel._worker is None
    assert panel._status.text() == "Search finished."


def test_a_search_that_returned_nothing_is_a_failure_on_the_status_line(panel):
    """`None` with no error message still has to read as a failure."""
    panel._on_search_done(None, "")

    assert "the worker returned no result" in panel._status.text()


# ---------------------------------------------------------------------------
# Sorting and small controls
# ---------------------------------------------------------------------------

def test_two_numeric_cells_sort_by_their_numbers_not_their_text():
    """0.9 above 0.1 even though "0.1" sorts before "0.9" as text."""
    assert _NumericTableItem("0.1", 0.1) < _NumericTableItem("0.9", 0.9)
    assert not (_NumericTableItem("0.9", 0.9) < _NumericTableItem("0.1", 0.1))


@pytest.mark.xfail(strict=True,
                   reason="_NumericTableItem.__lt__ falls back to "
                          "super().__lt__, which PySide6 dispatches straight "
                          "back into the Python override -- the comparison "
                          "recurses until it is abandoned and answers False")
def test_a_numeric_cell_sorts_against_a_plain_cell_by_its_text():
    """A number column holding "-" for a failed trial must still sort.

    Qt orders plain items by their display text, so a scored row and an
    unscored one have a defined order; without the fallback working, the
    table's sort silently reports "not less than" for every such pair.
    """
    from PySide6.QtWidgets import QTableWidgetItem

    number = _NumericTableItem("0.9000", 0.9)
    plain = QTableWidgetItem("zzz")

    assert (number < plain) is True


def test_a_fixed_choice_field_takes_a_value_it_offers(qtbot):
    """The combo keeps the tiny text API the plain field had."""
    combo = _FixedChoiceCombo()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])

    combo.setText("cosine")
    assert combo.text() == "cosine"

    combo.setText("manhattan")
    assert combo.text() == "cosine", "an unoffered value must not be invented"

    combo.setText("")
    assert combo.currentIndex() == -1


def test_a_walk_starts_from_one_value_not_a_list():
    """A walk needs a starting POINT; a grid of starts is a different request."""
    assert _parse_walk_start("n_neighbors", "15") == 15

    with pytest.raises(ValueError, match="starts from one value"):
        _parse_walk_start("n_neighbors", "5, 10, 15")


# ---------------------------------------------------------------------------
# Clustering a stored map
# ---------------------------------------------------------------------------

def test_clustering_with_nothing_selected_asks_for_a_map(panel):
    """There is nothing to cluster until a completed UMAP is chosen."""
    assert panel.cluster_selected() is False
    assert "Select a completed UMAP before clustering it." in \
        panel._status.text()


def test_walking_clusters_with_nothing_selected_asks_for_a_map(panel):
    """The same precondition, said in the walk's own words."""
    assert panel.walk_selected_clusters() is False
    assert "Select a completed UMAP before walking clusters." in \
        panel._status.text()


def test_a_clustering_that_fails_names_the_reason(panel, monkeypatch):
    """HDBSCAN is optional, so its absence must be a sentence, not a crash."""
    from spacr import umap_search

    def refuse(*_args, **_kwargs):
        raise RuntimeError("hdbscan is not installed")

    monkeypatch.setattr(umap_search, "cluster_embedding", refuse)
    panel._displayed_trial = _trial(0, embedding=True)

    assert panel.cluster_selected() is False
    assert "Could not cluster this UMAP: hdbscan is not installed" in \
        panel._status.text()


def test_a_cluster_walk_that_fails_names_the_reason(panel, monkeypatch):
    """The walk over HDBSCAN scales fails the same way and says so."""
    from spacr import umap_search

    def refuse(*_args, **_kwargs):
        raise RuntimeError("hdbscan is not installed")

    monkeypatch.setattr(umap_search, "walk_clusters", refuse)
    panel._displayed_trial = _trial(0, embedding=True)

    assert panel.walk_selected_clusters() is False
    assert "Could not walk cluster settings: hdbscan is not installed" in \
        panel._status.text()


def test_a_cluster_walk_that_finds_no_partition_says_so(panel, monkeypatch):
    """An empty walk is not an error, and is not a clustering either."""
    from spacr import umap_search

    monkeypatch.setattr(umap_search, "walk_clusters", lambda *a, **k: [])
    panel._displayed_trial = _trial(0, embedding=True)

    assert panel.walk_selected_clusters() is False
    assert "no usable partition" in panel._status.text()


# ---------------------------------------------------------------------------
# The stored-map viewer
# ---------------------------------------------------------------------------

def test_a_trial_with_no_stored_coordinates_cannot_be_displayed(panel):
    """A classifier trial has no embedding, and asking is not an error."""
    assert panel.show_trial(_trial(0)) is False


def test_coordinates_the_viewer_rejects_land_on_the_status_line(panel,
                                                                monkeypatch):
    """A malformed embedding is a message, not an exception in a slot."""
    explorer = getattr(panel, "_umap_explorer", None)
    if explorer is None:
        pytest.skip("this build has no native UMAP explorer")

    def refuse(*_args, **_kwargs):
        raise ValueError("that is not a 2-D or 3-D embedding")

    monkeypatch.setattr(explorer.view, "set_embedding", refuse)

    assert panel.show_trial(_trial(0, embedding=True)) is False
    assert "Could not display this UMAP" in panel._status.text()


def test_the_grid_says_so_when_no_umap_has_finished(panel):
    """An empty wall of embeddings is explained rather than shown blank."""
    assert panel.open_umap_grid() is None
    assert "no completed UMAPs" in panel._status.text()


# ---------------------------------------------------------------------------
# Opening a trial's figure file
# ---------------------------------------------------------------------------

def test_clicking_a_cell_with_no_figure_opens_nothing(panel, monkeypatch):
    """A grid cell whose trial produced no file must not open a blank viewer."""
    opened = []

    class _Grid:
        def figure_path(self, _index):
            return ""

    import PySide6.QtGui as QtGui

    monkeypatch.setattr(panel, "_figure_grid", _Grid(), raising=False)
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url)))

    panel._open_trial_figure(0)

    assert opened == []


def test_a_viewer_that_will_not_open_is_not_fatal(panel, monkeypatch,
                                                  tmp_path):
    """The desktop has no viewer for a PDF on some machines, and that is fine.

    The failure is logged rather than shown: the status line belongs to the
    sweep, and overwriting it with a file-association problem would hide how
    far the search had got.
    """
    png = tmp_path / "trial.png"
    png.write_bytes(b"x")

    class _Grid:
        def figure_path(self, _index):
            return str(png)

    monkeypatch.setattr(panel, "_figure_grid", _Grid(), raising=False)
    panel._status.setText("3 of 8 configurations evaluated")

    import PySide6.QtGui as QtGui

    def refuse(_url):
        raise RuntimeError("no application is registered for that file")

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl",
                        staticmethod(refuse))

    panel._open_trial_figure(0)

    assert panel._status.text() == "3 of 8 configurations evaluated"


# ---------------------------------------------------------------------------
# Criteria that disagree
# ---------------------------------------------------------------------------

def test_two_criteria_picking_different_winners_is_the_result():
    """There is no ground truth for attribution, so disagreement IS the finding.

    Presenting only the top row would hide that the configurations differ in
    which property they satisfy.
    """
    trials = [_trial(0, 0.9, trustworthiness=0.9, continuity=0.1),
              _trial(1, 0.5, trustworthiness=0.5, continuity=0.9)]

    sentence = criteria_disagree(_result(trials),
                                 ["trustworthiness", "continuity"])

    assert sentence is not None
    assert "THE CRITERIA DISAGREE" in sentence


def test_a_single_trial_cannot_disagree_with_itself():
    """One row has nothing to be re-ranked against."""
    assert criteria_disagree(_result([_trial(0)]), ["trustworthiness"]) is None


# ---------------------------------------------------------------------------
# The settings window
# ---------------------------------------------------------------------------

def test_asking_for_the_settings_window_twice_raises_the_one_that_is_open(
        panel, qtbot):
    """A second Settings click must focus the window, not build another.

    Two settings windows over one panel is two views of one state, and the
    one the user is not typing into silently goes stale.
    """
    panel.open_settings()
    first = panel._settings_dialog
    qtbot.addWidget(first)
    first.show()
    qtbot.waitExposed(first)

    panel.open_settings()

    assert panel._settings_dialog is first


def test_the_axis_picker_is_only_offered_for_umap(qtbot):
    """A classifier sweep has no UMAP axes to walk over."""
    other = HyperparamPanel("classify")
    qtbot.addWidget(other)

    assert other.open_walk_axes() is None


def test_cancelling_the_axis_picker_leaves_the_axes_alone(panel, monkeypatch):
    """A rejected dialog must not rewrite the walk."""
    before = dict(panel.walk_axes())
    monkeypatch.setattr(hp.WalkAxesDialog, "exec",
                        lambda self: QDialog.Rejected)

    dialog = panel.open_walk_axes()

    assert dialog is not None
    assert panel.walk_axes() == before


# ---------------------------------------------------------------------------
# Dispatching a request to the search backend
# ---------------------------------------------------------------------------

def test_a_request_is_handed_to_the_backend_for_its_app(monkeypatch):
    """The default search function is only a translation of the request.

    Anything it forgot to pass would be a setting the user chose that the
    sweep never saw.
    """
    seen = {}

    def record(app_key, settings, space, **kwargs):
        seen["app_key"] = app_key
        seen["space"] = space
        seen.update(kwargs)
        return "the result"

    monkeypatch.setattr(hp, "run_search_for_app", record)
    space = SearchSpace({"n_neighbors": [5, 10]})
    request = SearchRequest(app_key="umap", space=space, criterion="continuity",
                            mode="random", n_trials=7, seed=3)

    assert hp._default_search_fn(request, None, None) == "the result"
    assert seen["app_key"] == "umap"
    assert seen["space"] is space
    assert seen["criterion"] == "continuity"
    assert seen["n_trials"] == 7


# ---------------------------------------------------------------------------
# Switching between a grid and a walk
# ---------------------------------------------------------------------------

def test_turning_the_walk_off_puts_the_grid_back(panel):
    """A walk replaces the grid list with one centre; leaving restores it.

    Losing the list would make the adaptive checkbox destructive -- a user
    who ticks it to read the tooltip loses the space they typed.
    """
    panel._value_edits["n_neighbors"].setText("5, 10, 15, 20")
    panel._adaptive.setChecked(True)
    assert "," not in panel._control_text(panel._value_edits["n_neighbors"])

    panel._adaptive.setChecked(False)

    assert panel._control_text(panel._value_edits["n_neighbors"]) == \
        "5, 10, 15, 20"


def test_a_parameter_this_app_has_no_field_for_is_skipped(panel):
    """The walk centres only the fields that exist on this panel."""
    panel._value_edits["n_neighbors"].setText("5, 10")
    panel._value_edits.pop("min_dist")

    panel._adaptive.setChecked(True)   # must not raise

    assert "," not in panel._control_text(panel._value_edits["n_neighbors"])


@pytest.mark.parametrize("field,text,message", [
    ("_adaptive_n_step", "many", "n_neighbors increment must be a number"),
    ("_adaptive_rounds", "lots", "maximum rounds must be a number"),
])
def test_an_adaptive_field_that_is_not_a_number_says_which_one(panel, field,
                                                               text, message):
    """The message names the field, because four of them look alike."""
    getattr(panel, field).setText(text)

    with pytest.raises(ValueError, match=message):
        panel.adaptive_parameters()


def test_adaptive_increments_have_to_move_the_search_forward(panel):
    """A zero increment or a zero round count is a walk that never walks."""
    panel._adaptive_n_step.setText("0")

    with pytest.raises(ValueError, match="must be positive"):
        panel.adaptive_parameters()


# ---------------------------------------------------------------------------
# Choosing the walk's axes
# ---------------------------------------------------------------------------

def test_accepting_the_axis_picker_writes_its_choice_back(panel, monkeypatch):
    """The dialog is the only way the walk's axes change."""
    chosen = {"n_neighbors": {"start": "15", "resolution": 3}}
    monkeypatch.setattr(hp.WalkAxesDialog, "exec",
                        lambda self: QDialog.Accepted)
    monkeypatch.setattr(hp.WalkAxesDialog, "selection", lambda self: chosen)

    panel.open_walk_axes()

    assert panel.walk_axes()["n_neighbors"]["start"] == "15"


def test_a_walk_axis_with_no_start_falls_back_to_the_field(panel):
    """An axis ticked without a value starts from what the panel already holds."""
    panel._adaptive.setChecked(True)
    panel._value_edits["n_neighbors"].setText("17")
    panel.set_walk_axes({"n_neighbors": {"start": "", "resolution": 3}})

    space = panel.current_space()

    assert list(space.params["n_neighbors"]) == [17]


def test_a_walk_axis_with_nowhere_to_start_from_says_so(panel, monkeypatch):
    """A walk needs a starting point, and the message names where to give it.

    Starting the walk from a silent default would search a neighbourhood
    around a value nobody chose and report it as the user's space.
    """
    panel._adaptive.setChecked(True)
    panel.set_walk_axes({"n_neighbors": {"start": "", "resolution": 3}})
    monkeypatch.setattr(panel, "walk_start_for", lambda _name: "")

    with pytest.raises(ValueError, match="no value to start"):
        panel.current_space()


# ---------------------------------------------------------------------------
# Starting and stopping
# ---------------------------------------------------------------------------

def test_a_second_run_while_one_is_going_is_refused(panel, monkeypatch):
    """Two sweeps writing into one table is two answers to one question."""
    class _Busy:
        def isRunning(self):
            return True

    panel._worker = _Busy()

    assert panel.run_search() is False
    assert panel._status.text() == "A search is already running."


def test_a_figure_grid_that_rejects_the_axes_is_cleared_instead(panel,
                                                               monkeypatch):
    """A grid that cannot lay out the space still has to accept figures.

    Falling back to no axes keeps the thumbnails arriving in order rather
    than losing them because the layout was refused.
    """
    seen = []

    class _Grid:
        def clear(self):
            pass

        def set_parameters(self, params):
            seen.append(list(params))
            if params:
                raise RuntimeError("those axes cannot be laid out")

        def setVisible(self, _flag):
            pass

    monkeypatch.setattr(panel, "_figure_grid", _Grid(), raising=False)
    monkeypatch.setattr(panel, "_search_fn",
                        lambda *a: _result([_trial(0)]), raising=False)
    panel._value_edits["n_neighbors"].setText("5, 10")

    assert panel.run_search() is True
    panel.stop_search()
    if panel._worker is not None:
        panel._worker.wait(3000)

    assert len(seen) == 2
    assert "n_neighbors" in seen[0]
    assert seen[1] == [], "the fallback must clear the axes, not keep them"


def test_the_popup_footer_run_button_follows_the_panels(panel, qtbot):
    """The settings window has its own Run; it must not stay live mid-sweep."""
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)
    footer = getattr(dialog, "_run_btn", None)
    if footer is None:
        pytest.skip("this build's settings window has no footer Run")

    panel._set_search_running(True)
    assert footer.isEnabled() is False

    panel._set_search_running(False)
    assert footer.isEnabled() is True


# ---------------------------------------------------------------------------
# The score column's heading
# ---------------------------------------------------------------------------

def test_a_criterion_that_cannot_be_read_leaves_a_plain_score_heading(panel,
                                                                     monkeypatch):
    """The heading degrades to "score" rather than to a stale criterion name."""
    class _Broken:
        def currentText(self):
            raise RuntimeError("this combo is gone")

    monkeypatch.setattr(panel, "_criterion", _Broken(), raising=False)

    panel._retitle_score_column()

    column = panel.COLUMNS.index("score")
    assert panel._table.horizontalHeaderItem(column).text() == "score"


def test_with_no_table_there_is_no_heading_to_change(panel, monkeypatch):
    """Called from a constructor, before the table exists.

    The criterion is not even read in that case: reading it is only worth
    doing when there is a heading to put it in.
    """
    reads = []

    class _Counting:
        def currentText(self):
            reads.append(1)
            return "trustworthiness"

    monkeypatch.setattr(panel, "_table", None, raising=False)
    monkeypatch.setattr(panel, "_criterion", _Counting(), raising=False)

    panel._retitle_score_column()

    assert reads == []


# ---------------------------------------------------------------------------
# The first UMAP to land
# ---------------------------------------------------------------------------

def test_the_first_finished_umap_is_selected_and_shown(panel):
    """The user sees a map as soon as there is one, not only at the end.

    Waiting until the sweep finishes is the complaint the live grid exists
    to answer.
    """
    panel._on_trial_ready(_trial(0, embedding=True), 1, 4, "")

    assert panel._grid_btn.isEnabled() is True
    assert panel._displayed_trial is not None
    assert panel._table.currentRow() == 0


# ---------------------------------------------------------------------------
# What a finished sweep says about itself
# ---------------------------------------------------------------------------

def test_rows_from_two_umap_backends_are_flagged_as_not_one_walk(panel):
    """cuML and the CPU implementation are different reducers.

    Ranking them together reads as a settings comparison when half the
    difference is which library produced the map.
    """
    trials = [_trial(0, 0.9, backend="cpu"), _trial(1, 0.8, backend="cuml")]

    panel._on_search_done(_result(trials), "")

    assert "MIXED BACKENDS" in panel._notes.text() + panel._status.text()


# ---------------------------------------------------------------------------
# A panel that is not showing UMAPs
# ---------------------------------------------------------------------------

@pytest.fixture()
def classify_panel(qtbot):
    view = HyperparamPanel("classify")
    qtbot.addWidget(view)
    return view


def test_a_classifier_panel_has_no_umap_to_show(classify_panel):
    """`show_trial` and the grid are UMAP-only, and say so by returning None."""
    assert classify_panel.show_trial(_trial(0, embedding=True)) is False
    assert classify_panel.open_umap_grid() is None


def test_a_classifier_sweep_that_returns_nothing_says_so_in_the_preview(
        classify_panel):
    """With no explorer, the failure message goes on the preview label."""
    classify_panel._on_search_done(None, "")

    assert classify_panel._preview.text() == "Search failed."


# ---------------------------------------------------------------------------
# The gallery
# ---------------------------------------------------------------------------

def test_reopening_the_gallery_refreshes_it_instead_of_building_another(
        panel, qtbot):
    """One gallery per panel, so a second click does not orphan the first."""
    panel._result = _result([_trial(0, embedding=True)])

    first = panel.open_umap_grid()
    assert first is not None
    qtbot.addWidget(first)

    panel._result = _result([_trial(0, embedding=True),
                             _trial(1, embedding=True)])
    again = panel.open_umap_grid()

    assert again is first


# ---------------------------------------------------------------------------
# Clustering that works
# ---------------------------------------------------------------------------

def test_a_cluster_walk_that_succeeds_records_what_it_chose(panel,
                                                            monkeypatch):
    """The chosen scale, its partition and the whole walk are stored.

    The walk is stored as well as its winner because the winner alone does
    not say whether the choice was clear or arbitrary.
    """
    from spacr import umap_search

    class _Row:
        min_cluster_size = 12
        labels = np.array([0, 0, 1, 1, -1, 1, 0, 1, 0, 1, -1, 0])
        silhouette = 0.42
        score = 0.4
        n_clusters = 2
        noise_fraction = 0.1

    monkeypatch.setattr(umap_search, "walk_clusters",
                        lambda *a, **k: [_Row(), _Row()])
    trial = _trial(0, embedding=True)
    panel._displayed_trial = trial

    assert panel.walk_selected_clusters() is True
    assert trial.extra_metrics["cluster_min_size"] == 12
    assert trial.extra_metrics["n_clusters"] == 2
    assert len(trial.extra_metrics["cluster_walk"]) == 2
    assert "min_cluster_size=12" in panel._status.text()


# ---------------------------------------------------------------------------
# Opening a figure with no grid at all
# ---------------------------------------------------------------------------

def test_a_panel_with_no_figure_grid_opens_nothing(classify_panel,
                                                   monkeypatch):
    """Not every app key builds a thumbnail grid, and a click on none opens none."""
    import PySide6.QtGui as QtGui

    opened = []
    monkeypatch.setattr(classify_panel, "_figure_grid", None, raising=False)
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url)))

    classify_panel._open_trial_figure(0)

    assert opened == []


# ---------------------------------------------------------------------------
# Tearing the panel down
# ---------------------------------------------------------------------------

def test_a_worker_that_will_not_stop_does_not_take_the_close_with_it(panel,
                                                                     monkeypatch):
    """Closing has to finish even when the sweep refuses to be stopped.

    Destroying a QWidget whose thread is still running aborts the process,
    so the shutdown attempt is guarded rather than assumed.
    """
    class _Stubborn:
        def request_stop(self):
            raise RuntimeError("this worker will not stop")

    panel._worker = _Stubborn()

    panel.close()      # must not raise

    assert panel.isVisible() is False


# ---------------------------------------------------------------------------
# Reading the host's settings before each sweep
# ---------------------------------------------------------------------------

def test_a_settings_provider_that_answers_nonsense_stops_the_sweep(panel):
    """A provider that is not a settings mapping must not be merged in.

    Running against a half-read settings object is how a source path
    dropped after the panel opened turns into a sweep over the wrong data.
    """
    panel.set_settings_provider(lambda: ["not", "a", "dict"])

    assert panel.run_search() is False
    assert "Could not read the current module settings" in panel._status.text()
    assert "did not return a dict" in panel._status.text()


# ---------------------------------------------------------------------------
# Closing the preview figure
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_close_still_leaves_the_preview_drawn(
        panel, monkeypatch):
    """Releasing the figure is housekeeping; the picture is the deliverable."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    monkeypatch.setattr(hp, "build_panel_figure",
                        lambda *a, **k: Figure(figsize=(2, 2)))

    def refuse(_fig):
        raise RuntimeError("this figure is not in the pyplot registry")

    monkeypatch.setattr(plt, "close", refuse)

    panel._draw_preview(_result([_trial(0)]))

    assert panel._preview.pixmap() is not None
    assert not panel._preview.text()


# ---------------------------------------------------------------------------
# Propagating the settings window's values back to the module
# ---------------------------------------------------------------------------

def test_propagating_with_nowhere_to_send_it_does_nothing(panel, qtbot):
    """A panel with no host has no settings form to write into."""
    panel.set_apply_callback(None)
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    dialog.propagate_settings()      # must not raise

    assert "Propagated" not in panel._status.text()


def test_a_search_field_holding_a_typo_is_left_out_of_the_propagation(panel,
                                                                     qtbot):
    """One unparseable field must not stop the other fields reaching the module.

    Propagation is the user pressing a button that says it will apply what
    they typed; refusing all of it over one typo is a worse answer than
    applying the rest.
    """
    written = {}
    panel.set_apply_callback(written.update)
    panel._value_edits["n_neighbors"].setText("fifteen")
    panel._value_edits["min_dist"].setText("0.1")
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    dialog.propagate_settings()

    assert written["n_neighbors"] != "fifteen", \
        "an unparseable field must not travel to the module"
    assert written["min_dist"] == pytest.approx(0.1)
    assert "Propagated" in panel._status.text()


def test_a_selected_row_wins_over_the_search_fields(panel, qtbot):
    """A chosen result is more authoritative than what is still in the boxes.

    The row is a measured configuration; the field is whatever the user
    last typed while browsing.
    """
    written = {}
    panel.set_apply_callback(written.update)
    panel._value_edits["n_neighbors"].setText("5")
    panel._on_search_done(_result([_trial(4)]), "")
    panel._table.selectRow(0)
    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    dialog.propagate_settings()

    assert written["n_neighbors"] == 25


# ---------------------------------------------------------------------------
# What one round of the walk will cost
# ---------------------------------------------------------------------------

def test_a_half_typed_starting_value_says_nothing_rather_than_a_wrong_number(
        panel, qtbot):
    """The cost is taken from the engine, so an unparseable axis has no cost.

    Printing an arithmetic guess would promise a round size the search then
    does not deliver.
    """
    dialog = hp.WalkAxesDialog(panel)
    qtbot.addWidget(dialog)
    rows = dialog._rows
    name = "n_neighbors"
    if name not in rows:
        pytest.skip("this build's walk dialog has no n_neighbors axis")
    enable, start, _resolution = rows[name]
    enable.setChecked(True)
    start.setText("fifte")

    dialog._update_cost()

    assert "Fill in a starting value" in dialog._cost.text()
    assert "1 axes" in dialog._cost.text()


def test_no_axis_ticked_names_the_two_the_walk_falls_back_to(panel, qtbot):
    """An empty selection is not an empty walk; it is the original two."""
    dialog = hp.WalkAxesDialog(panel)
    qtbot.addWidget(dialog)
    for enable, _start, _resolution in dialog._rows.values():
        enable.setChecked(False)

    dialog._update_cost()

    assert "n_neighbors and min_dist" in dialog._cost.text()


# ---------------------------------------------------------------------------
# Building the settings window's tabs from whatever categories exist
# ---------------------------------------------------------------------------

def test_a_tab_with_no_controls_is_not_added_and_its_widgets_are_destroyed(
        panel, qtbot, monkeypatch):
    """A tab set is built from the categories that exist, not a fixed list.

    Every UMAP control is materialised with the dialog as its parent before
    the tabs are laid out, so a control that no tab claims has to be
    destroyed rather than hidden -- a merely hidden compound control is
    still eligible for a transient paint at the window's top-left corner.
    """
    from spacr.qt.screens import settings_model

    real = settings_model.SettingsWidgets.build_sections

    def only_the_first(self):
        sections = real(self)
        return sections[:1]

    monkeypatch.setattr(settings_model.SettingsWidgets, "build_sections",
                        only_the_first)

    panel.open_settings()
    dialog = panel._settings_dialog
    qtbot.addWidget(dialog)

    titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    assert "Search" in titles
    assert len(titles) < 7, "empty tabs were added anyway"
    assert set(dialog._module_model._widgets) == dialog._module_keys
