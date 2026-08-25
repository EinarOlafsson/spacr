"""Finding a regression run on disk, and reading its summary back.

The Results tab is handed whatever the user has: a CSV, the run folder, or
the parent that holds a dozen runs. Everything here is about the walk that
turns one of those into a table, and about the summary text that gets printed
above the statsmodels block. Both run against real files in ``tmp_path``.

The failures pinned down are the ones a shared filesystem produces on its own:
a path that is not a path at all, a file that vanished between the listing and
the ``stat``, a summary that cannot be opened, and a tree deep or wide enough
that the walk has to stop. In every case the tab must come back with something
-- a shorter list, an empty string -- rather than take the panel down with it.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.regression_results import (  # noqa: E402
    RESULT_FILENAMES, SPACR_SUMMARY_HEADING, VERBATIM_HEADING,
    _identifiability_warning, _match_column, _spacr_summary_text,
    _with_spacr_summary, find_results_table, find_results_tables,
    find_summary_file,
)


def _run_folder(root, name, filenames=("results.csv",)):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (folder / filename).write_text("gene,coefficient\na,1.0\n")
    return folder


# ---------------------------------------------------------------------------
# _match_column
# ---------------------------------------------------------------------------

def test_matching_a_column_of_nothing_is_nothing():
    """The panel asks before it knows whether a table was loaded."""
    assert _match_column(None, ("p_value",)) is None


def test_a_column_is_found_through_a_difference_in_spelling():
    """A CSV written by hand spells it 'P Value'; the panel still finds it.

    The exact-name pass runs first so a table carrying both spellings keeps
    the canonical one. The normalised pass is what rescues everything else.
    """
    frame = pd.DataFrame({"P Value": [0.1], "coefficient": [1.0]})
    assert _match_column(frame, ("p_value",)) == "P Value"
    assert _match_column(frame, ("q_value",)) is None


# ---------------------------------------------------------------------------
# find_results_tables
# ---------------------------------------------------------------------------

def test_something_that_is_not_a_path_finds_no_tables():
    """``os.fspath`` raises on a number, and the tab must not."""
    assert find_results_tables(3.5) == []
    assert find_results_table(3.5) is None


def test_the_walk_stops_at_the_depth_it_was_given(tmp_path):
    """A results folder six levels down is a search, not a scan of the disk."""
    _run_folder(tmp_path, "shallow")
    _run_folder(tmp_path, os.path.join("a", "b", "deep"))
    found = find_results_tables(str(tmp_path), max_depth=1)
    assert [os.path.basename(os.path.dirname(p)) for p in found] == ["shallow"]
    deeper = find_results_tables(str(tmp_path), max_depth=6)
    assert len(deeper) == 2


def test_a_table_that_vanishes_between_the_listing_and_the_stat_is_skipped(
        tmp_path, monkeypatch):
    """A pipeline still writing into the folder deletes files as we walk it.

    The run is ordered by its newest table's modification time, so a table
    that has gone cannot contribute one. Skipping it keeps the other tables
    of the same run, which is what the user came to look at.
    """
    _run_folder(tmp_path, "run1", RESULT_FILENAMES)
    real_getmtime = os.path.getmtime

    def vanishing(path):
        if str(path).endswith("results_grna.csv"):
            raise OSError(2, "No such file or directory")
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", vanishing)
    found = find_results_tables(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["results.csv",
                                                    "results_gene.csv"]


def test_the_walk_stops_once_the_candidate_limit_is_reached(tmp_path):
    """A parent holding hundreds of runs must not be enumerated in full."""
    for index in range(5):
        _run_folder(tmp_path, f"run{index}")
    assert len(find_results_tables(str(tmp_path), limit=1)) == 1
    assert len(find_results_tables(str(tmp_path), limit=200)) == 5


def test_a_file_that_is_not_a_csv_is_not_a_results_table(tmp_path):
    """Handed a settings file, the tab must not try to read it as results."""
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert find_results_tables(str(other)) == []


# ---------------------------------------------------------------------------
# find_summary_file
# ---------------------------------------------------------------------------

def test_a_summary_is_not_looked_for_under_something_that_is_not_a_path():
    """Same guard as the table walk, on the same kind of caller mistake."""
    assert find_summary_file(3.5) is None


def test_the_summary_is_found_beside_the_table_the_walk_chose(tmp_path):
    """The user names the parent; the summary lives beside the chosen run."""
    from spacr.ml import SUMMARY_FILENAME

    folder = _run_folder(tmp_path, "run1")
    (folder / SUMMARY_FILENAME).write_text(f"{SPACR_SUMMARY_HEADING}\nbody\n")
    assert find_summary_file(str(tmp_path)) == str(folder / SUMMARY_FILENAME)


# ---------------------------------------------------------------------------
# _spacr_summary_text and _with_spacr_summary
# ---------------------------------------------------------------------------

def _write_summary(tmp_path, text):
    from spacr.ml import SUMMARY_FILENAME

    folder = _run_folder(tmp_path, "run1")
    (folder / SUMMARY_FILENAME).write_text(text)
    return folder


def test_a_summary_that_cannot_be_opened_reads_as_no_summary(tmp_path):
    """An unreadable file is not a reason to lose the statsmodels block."""
    folder = _write_summary(tmp_path, f"{SPACR_SUMMARY_HEADING}\nbody\n")
    from spacr.ml import SUMMARY_FILENAME

    path = folder / SUMMARY_FILENAME
    os.chmod(path, 0o000)
    try:
        assert _spacr_summary_text(str(folder)) == ""
    finally:
        os.chmod(path, 0o644)


def test_a_file_written_before_spacr_had_its_own_summary_is_not_prepended(
        tmp_path):
    """Only the spaCR heading marks a file this reader owns the front of."""
    folder = _write_summary(tmp_path, "                   OLS Regression\n")
    assert _spacr_summary_text(str(folder)) == ""


def test_the_verbatim_tail_is_trimmed_so_it_is_not_printed_twice(tmp_path):
    """The live fit's text is the one on screen; the saved copy is dropped."""
    folder = _write_summary(
        tmp_path,
        f"{SPACR_SUMMARY_HEADING}\nsections here\n\n"
        f"{VERBATIM_HEADING}\nan old statsmodels block\n")
    text = _spacr_summary_text(str(folder))
    assert text.endswith("sections here")
    assert "an old statsmodels block" not in text


def test_a_run_with_a_spacr_summary_and_no_statsmodels_one_says_which_is_missing(
        tmp_path):
    """"No summary" would deny the sections that are right there above it."""
    folder = _write_summary(tmp_path, f"{SPACR_SUMMARY_HEADING}\nsections\n")
    text = _with_spacr_summary(str(folder), "this backend keeps none",
                               missing=True)
    assert text.startswith(SPACR_SUMMARY_HEADING)
    assert VERBATIM_HEADING in text
    assert "No statsmodels summary: this backend keeps none" in text


def test_a_run_with_a_spacr_summary_keeps_the_statsmodels_text_verbatim(
        tmp_path):
    """The point of asking for it is to get it unchanged."""
    folder = _write_summary(tmp_path, f"{SPACR_SUMMARY_HEADING}\nsections\n")
    text = _with_spacr_summary(str(folder), "OLS Regression Results")
    assert text.endswith("OLS Regression Results")
    assert VERBATIM_HEADING in text


def test_a_run_with_no_spacr_summary_at_all_says_no_summary(tmp_path):
    """That sentinel is what every caller tests for."""
    folder = _run_folder(tmp_path, "run1")
    assert _with_spacr_summary(str(folder), "no fit here",
                               missing=True) == "No summary: no fit here"


# ---------------------------------------------------------------------------
# _identifiability_warning
# ---------------------------------------------------------------------------

def test_a_model_that_exposes_no_shape_raises_no_warning():
    """The warning is read off the fit; a fit that cannot say stays silent."""

    class _Opaque:
        nobs = "many"

    assert _identifiability_warning(_Opaque()) == ""
    assert _identifiability_warning(None) == ""


def test_more_parameters_than_wells_is_warned_about():
    """The refusal above must not be swallowing the real warning."""

    class _Wide:
        nobs = 10
        params = list(range(12))

    warning = _identifiability_warning(_Wide())
    assert "10" in warning and "12" in warning


# ---------------------------------------------------------------------------
# The panel itself
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import sys  # noqa: E402

pytest.importorskip("pyqtgraph")

from spacr.qt.widgets.regression_results import (  # noqa: E402
    RegressionResultsPanel,
)


@pytest.fixture()
def results():
    """A small guide-level table with everything the panel colours by."""
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(size=n),
        "q_value": np.sort(rng.uniform(size=n)),
        "condition": rng.choice(["nc", "pc", "other"], n, p=[0.1, 0.1, 0.8]),
    })


@pytest.fixture()
def panel(qtbot):
    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


# --- the static key helpers ------------------------------------------------

def test_a_table_with_no_rows_at_all_has_no_key_column():
    """The panel asks before a table has been loaded."""
    assert RegressionResultsPanel._key_column(None) is None


def test_a_repeated_feature_is_not_a_key():
    """Joining on it would select an arbitrary member of the group."""
    frame = pd.DataFrame({"feature": ["a", "a", "b"]})
    assert RegressionResultsPanel._key_column(frame) is None
    unique = pd.DataFrame({"feature": ["a", "b"]})
    assert RegressionResultsPanel._key_column(unique) == "feature"


def test_a_table_with_no_feature_column_cannot_say_what_was_tested():
    """"cannot say" and "nothing was tested" are opposite answers."""
    assert RegressionResultsPanel._tested_mask(
        pd.DataFrame({"gene": ["a"]})) is None


def test_a_table_with_no_feature_column_offers_no_gene_terms():
    """The bridge between the bare id and the design term needs the term."""
    assert RegressionResultsPanel._gene_terms(pd.DataFrame({"x": [1]})) == {}
    assert RegressionResultsPanel._gene_terms(None) == {}


# --- level, reachability and colour channels -------------------------------

def test_a_table_with_no_feature_column_opens_at_no_level(panel):
    """An unrecognised table is shown whole rather than filtered to nothing."""
    panel._frame = pd.DataFrame({"coefficient": [1.0]})
    assert panel._default_level() is None


def test_every_key_is_reachable_when_the_table_carries_no_feature_column(
        panel):
    """With no feature column the level filter is hiding nothing."""
    panel._frame = pd.DataFrame({"coefficient": [1.0]})
    panel._level = "grna"
    assert panel._reachable("anything") is True


def test_every_key_is_reachable_when_no_level_filter_is_in_force(panel,
                                                                 results):
    """Nothing is hidden, so moving the filter could not help."""
    panel.set_frame(results)
    panel._level = ""
    assert panel._reachable(results["feature"].iloc[0]) is True


def test_the_lopit_channel_is_never_offered_as_a_second_colour(panel):
    """It is materialised onto the frame copy for the FIRST channel only.

    Asking for it on channel two or three names a column that is not on the
    frame at all, so the encoding would silently draw nothing.
    """

    class _Combo:
        def currentData(self):
            return panel.LOPIT_KEY

    assert panel._extra_colour_column(_Combo(), "q_value") is None


def test_no_second_colour_is_encoded_without_a_first(panel):
    """A shape meaning one thing beside a colour meaning another is two
    claims on one dot with nothing on screen saying which."""

    class _Combo:
        def currentData(self):
            return "coefficient"

    assert panel._extra_colour_column(_Combo(), None) is None


def test_the_selection_falls_back_to_the_last_single_click(panel, results):
    """Linked views follow the click even when the table has no multiselect."""
    panel.set_frame(results)
    panel.table.table.clearSelection()
    panel._selected_key = results["feature"].iloc[3]
    assert panel.selected_keys() == [results["feature"].iloc[3]]


# --- the model identity line ----------------------------------------------

def test_a_model_that_cannot_be_named_leaves_the_caption_empty(panel,
                                                               monkeypatch):
    """An empty muted strip under the graph is pixels that mean nothing."""
    monkeypatch.setitem(sys.modules, "spacr.regression_summary", None)
    assert panel._name_the_model(object(), "ols") == ""
    assert not panel._model_line.isVisible()


def test_the_no_model_reason_names_the_diagnostics_error_that_lost_it(panel):
    """Telling the disk story again would send the user to the wrong place."""
    panel._model = None
    panel._diagnostics_error = "context_from_model: singular design"
    reason = panel._no_model_reason()
    assert "the diagnostics failed and the fit was not kept" in reason
    assert "singular design" in reason


def test_diagnostics_without_the_qc_module_report_which_module_is_missing(
        panel, results, monkeypatch):
    """The fit is stored before anything is drawn, so only the panels go."""
    panel.set_frame(results)
    monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
    model = object()
    assert panel.set_diagnostics(model, "ols") is False
    assert panel._model is model
    assert "could not load the diagnostics module" in panel._diagnostics_error


# --- tabs that were never built -------------------------------------------

def test_asking_for_a_panel_this_build_has_no_tab_for_is_declined(panel):
    """The available tabs depend on the fit and on where the volcano lives."""
    assert panel.show_panel("no_such_panel") is False
    panel.annotation_umap = None
    assert panel.show_panel("annotation_check") is False


# --- compartments ----------------------------------------------------------

def test_a_frame_that_cannot_be_scanned_offers_no_compartments(panel,
                                                               monkeypatch):
    """A menu of 22 choices that colour nothing is a broken-looking menu."""
    import spacr.localisation as localisation

    def unreadable(frame):
        raise RuntimeError("no reference table")

    offered = []
    monkeypatch.setattr(localisation, "present", unreadable)
    monkeypatch.setattr(panel.volcano, "offer_compartments", offered.append)
    panel._frame = pd.DataFrame({"feature": ["gene_fraction:gene[1]"]})
    panel._offer_compartments()
    assert offered == [[]], "a menu was built from compartments nobody found"


def test_colouring_by_every_localisation_counts_what_carries_one(panel,
                                                                 results):
    """``mask`` takes ONE compartment, so ALL needs its own sentence.

    Handing the sentinel to the single-compartment branch reported
    "0 annotated all-localisations" -- a confident number about nothing.
    """
    from spacr.localisation import ALL as ALL_COMPARTMENTS

    panel.set_frame(results)
    panel.set_compartment(ALL_COMPARTMENTS)
    assert panel._compartment == ALL_COMPARTMENTS
    assert "all-localisations" not in panel._status
    assert "TAGM/LOPIT localisation" in panel._status


def test_the_volcano_redraw_is_a_no_op_before_a_table_is_loaded(panel,
                                                                monkeypatch):
    """Every control connects to it, and they exist before the data does.

    The redraw reaches for the effect column on the very next line, so
    without the guard the first signal fired during construction would take
    the panel down before it had a table.
    """
    asked = []
    monkeypatch.setattr(panel, "_effect_column", asked.append)
    panel._frame = None
    assert panel._redraw_volcano() is None
    assert asked == [], "the redraw carried on past a table that is not there"


# ---------------------------------------------------------------------------
# Loading a run, and every way it fails
# ---------------------------------------------------------------------------

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


def test_loading_nothing_says_which_button_to_press(panel):
    """A panel with columns and no rows is indistinguishable from a bad run."""
    assert panel.load("") is False
    assert "Nothing was handed to the results panel" in panel._status


def test_starting_a_load_of_nothing_says_the_same_thing(panel):
    """Both entry points end at the same message, or they will drift."""
    assert panel.start_load("") is False
    assert "Nothing was handed to the results panel" in panel._status
    assert panel.is_loading() is False


def test_a_folder_that_does_not_exist_says_so(panel, tmp_path):
    """The path is named, because the caller may have built it wrongly."""
    missing = str(tmp_path / "not_here")
    assert panel.load(missing) is False
    assert missing in panel._status
    assert "does not exist" in panel._status


def test_a_folder_with_no_result_table_names_what_was_looked_for(panel,
                                                                 tmp_path):
    """The user has to know which filenames would have satisfied the search."""
    (tmp_path / "empty").mkdir()
    assert panel.load(str(tmp_path / "empty")) is False
    for name in RESULT_FILENAMES:
        assert name in panel._status


def test_a_results_csv_that_pandas_cannot_parse_says_which_file(panel,
                                                                tmp_path):
    """Naming the file is what turns a stack trace into something actionable."""
    folder = tmp_path / "run1"
    folder.mkdir()
    (folder / "results.csv").write_text('a,b\n"unterminated,1\n2,3,4,5\n')
    assert panel.load(str(folder)) is False
    assert "Could not read" in panel._status
    assert "results.csv" in panel._status


def test_a_table_the_panel_refuses_still_names_the_folder_it_came_from(
        panel, tmp_path):
    """``set_frame`` says WHY; the load adds WHERE, and both are needed."""
    folder = _write_run(tmp_path, frame=pd.DataFrame(
        {"feature": [], "coefficient": [], "p_value": []}))
    assert panel.load(str(tmp_path)) is False
    assert str(folder / "results.csv") in panel._status
    assert str(tmp_path) in panel._status


def test_the_newest_of_several_runs_is_loaded_and_the_rest_are_counted(
        panel, tmp_path):
    """Silently picking one of five runs is how the wrong table gets read."""
    _write_run(tmp_path, "run1")
    _write_run(tmp_path, "run2")
    assert panel.load(str(tmp_path)) is True
    assert "newest of 2 runs" in panel._status
    assert str(tmp_path) in panel._status


def test_a_single_run_that_wrote_several_tables_says_which_one_is_shown(
        panel, tmp_path):
    """results.csv is the full table; the gene and guide views are subsets."""
    _write_run(tmp_path, "run1", filenames=RESULT_FILENAMES)
    assert panel.load(str(tmp_path)) is True
    assert f"wrote {len(RESULT_FILENAMES)} tables" in panel._status
    assert "this is the full one" in panel._status


# --- the worker half -------------------------------------------------------

def test_the_worker_reports_a_missing_folder_rather_than_raising(tmp_path):
    """A job that raises loses the detail crossing the thread boundary."""
    outcome = RegressionResultsPanel._read_run(str(tmp_path / "gone"))
    assert "does not exist" in outcome["error"]
    assert "frame" not in outcome


def test_the_worker_reports_a_folder_with_no_tables(tmp_path):
    """Same sentence as the synchronous path, from the worker's side."""
    outcome = RegressionResultsPanel._read_run(str(tmp_path))
    assert "Searched" in outcome["error"]
    for name in RESULT_FILENAMES:
        assert name in outcome["error"]


def test_the_worker_reports_a_table_it_could_not_parse(tmp_path):
    """The GUI half only relays; the reading is all done out here."""
    folder = tmp_path / "run1"
    folder.mkdir()
    (folder / "results.csv").write_text('a,b\n"unterminated,1\n2,3,4,5\n')
    outcome = RegressionResultsPanel._read_run(str(folder))
    assert outcome["error"].startswith("Could not read")


def test_a_loader_that_comes_back_with_nothing_is_called_a_loader_bug(panel):
    """Blaming the run for a loader fault sends the user to the wrong file."""
    seen = []
    panel.load_finished.connect(seen.append)
    assert panel._finish_load(None) is False
    assert "a bug in the loader rather than in the run" in panel._status
    assert seen == [False]


def test_a_load_job_that_fails_clears_the_spinner_and_says_why(panel):
    """A caller waiting on ``load_finished`` must not be left waiting."""
    seen = []
    panel.load_finished.connect(seen.append)
    panel._loading = True
    panel._on_load_job_failed("the worker died")
    assert panel.is_loading() is False
    assert "The run could not be read: the worker died" in panel._status
    assert seen == [False]


def test_closing_the_panel_survives_a_loader_that_will_not_stop(panel,
                                                                monkeypatch):
    """Qt needs the thread to outlive its owner; the close must still happen."""
    from PySide6.QtGui import QCloseEvent

    def refuse():
        raise RuntimeError("the worker is wedged")

    monkeypatch.setattr(panel._load_jobs, "shutdown", refuse)
    event = QCloseEvent()
    panel.closeEvent(event)
    assert event.isAccepted(), "the panel refused to close"


# ---------------------------------------------------------------------------
# set_frame: the optional readings it can lose
# ---------------------------------------------------------------------------

def test_a_run_whose_settings_cannot_be_read_still_loads_its_table(
        panel, results, monkeypatch):
    """The settings are context; the coefficients are the deliverable."""
    import spacr.refit as refit

    def unreadable(source):
        raise RuntimeError("settings/ was overwritten by a later run")

    monkeypatch.setattr(refit, "settings_of_run", unreadable)
    assert panel.set_frame(results, source="results.csv") is True
    assert panel._run_settings is None


def test_a_table_whose_localisations_cannot_be_looked_up_still_loads(
        panel, results, monkeypatch):
    """LOPIT is joined from a bundled table; losing it costs one menu entry."""
    import spacr.localisation as localisation

    def unreadable(frame):
        raise RuntimeError("the TAGM table is not installed")

    monkeypatch.setattr(localisation, "present", unreadable)
    assert panel.set_frame(results, source="results.csv") is True
    assert panel._colour_by.findData(panel.LOPIT_KEY) == -1


def test_a_volcano_that_will_not_release_its_axes_does_not_stop_the_load(
        panel, results, monkeypatch):
    """A previous run's limits are a convenience, not part of the table."""
    def refuse():
        raise RuntimeError("no view box yet")

    monkeypatch.setattr(panel.volcano, "auto_range_axes", refuse)
    assert panel.set_frame(results, source="results.csv") is True


# ---------------------------------------------------------------------------
# Restoring saved views
# ---------------------------------------------------------------------------

def test_saved_axis_limits_that_will_not_apply_do_not_lose_the_run(
        panel, results, monkeypatch):
    """The table is restored either way; the zoom is a convenience."""
    panel.set_frame(results, source="results.csv")

    def refuse(**kwargs):
        raise RuntimeError("no view box")

    monkeypatch.setattr(panel.volcano, "set_axis_limits", refuse)
    assert panel.apply_plot_state({"x_limits": (-1.0, 1.0),
                                   "y_limits": (0.0, 3.0)}) is True


def test_a_saved_selection_that_is_no_longer_visible_is_not_an_error(
        panel, results, monkeypatch):
    """A gene picked at level=None is not a row in the guide table."""
    panel.set_frame(results, source="results.csv")

    def refuse(key):
        raise KeyError(key)

    monkeypatch.setattr(panel, "_select_key", refuse)
    assert panel.apply_plot_state({"selected_key": "gene[244480]"}) is True


def test_a_workspace_that_is_not_a_mapping_restores_nothing(panel):
    """A document read from an older or corrupt file must not take the panel."""
    assert panel.apply_workspace_state(None) is False
    assert panel.apply_workspace_state("runs") is False


def test_a_panel_whose_optional_umap_tab_will_not_build_is_still_a_panel(
        qtbot, monkeypatch, results):
    """The annotation check is one tab; the results are the other twelve.

    The UMAP tab is imported lazily at construction because it drags in the
    embedding stack. An environment without it must still open a run.
    """
    monkeypatch.setitem(sys.modules,
                        "spacr.qt.widgets.annotation_umap_tab", None)
    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    assert widget.annotation_umap is None
    assert widget.set_frame(results, source="results.csv") is True
    assert widget.show_panel("annotation_check") is False
    titles = [widget.tabs.tabText(i).split(" (")[0]
              for i in range(widget.tabs.count())]
    assert "Volcano" in titles
    assert "Annotation check" not in titles


def test_browsing_starts_beside_the_run_that_is_already_open(panel, tmp_path,
                                                             monkeypatch):
    """Opening the file dialog at the last run's parent is the whole point.

    Starting at the process's working directory instead makes a user who has
    one run open navigate back to it from wherever spaCR was launched.
    """
    from PySide6.QtWidgets import QFileDialog

    folder = _write_run(tmp_path, "run1")
    assert panel.load(str(folder)) is True

    seen = {}

    def record(parent, title, start):
        seen["start"] = start
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", record)
    assert panel.browse_for_results() is False
    assert seen["start"] == str(folder)


def test_browsing_from_an_empty_panel_starts_nowhere_in_particular(panel,
                                                                   monkeypatch):
    """With no run open there is no better guess than the dialog's own."""
    from PySide6.QtWidgets import QFileDialog

    seen = {}

    def record(parent, title, start):
        seen["start"] = start
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", record)
    assert panel.browse_for_results() is False
    assert seen["start"] == ""
