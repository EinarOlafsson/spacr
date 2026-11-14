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

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.regression_results import (  # noqa: E402
    SPACR_SUMMARY_HEADING, _identifiability_warning, _match_column,
    find_summary_file,
)


def _run_folder(root, name, filenames=("results.csv",)):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (folder / filename).write_text("gene,coefficient\na,1.0\n", encoding="utf-8")
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
# find_summary_file
# ---------------------------------------------------------------------------

def test_a_summary_is_not_looked_for_under_something_that_is_not_a_path():
    """Same guard as the table walk, on the same kind of caller mistake."""
    assert find_summary_file(3.5) is None


def test_the_summary_is_found_beside_the_table_the_walk_chose(tmp_path):
    """The user names the parent; the summary lives beside the chosen run."""
    from spacr.ml import SUMMARY_FILENAME

    folder = _run_folder(tmp_path, "run1")
    (folder / SUMMARY_FILENAME).write_text(f"{SPACR_SUMMARY_HEADING}\nbody\n", encoding="utf-8")
    assert find_summary_file(str(tmp_path)) == str(folder / SUMMARY_FILENAME)


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


def test_starting_a_load_of_nothing_says_the_same_thing(panel):
    """Both entry points end at the same message, or they will drift."""
    assert panel.start_load("") is False
    assert "Nothing was handed to the results panel" in panel._status
    assert panel.is_loading() is False


# --- the worker half -------------------------------------------------------


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
# Restoring saved views
# ---------------------------------------------------------------------------


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
