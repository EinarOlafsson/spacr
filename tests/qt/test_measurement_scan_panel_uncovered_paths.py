"""Paths of the Measurements tab that only a worn-in session reaches.

The four steps of this tab -- attach, choose, merge, regress -- are driven
elsewhere along their happy road. What is driven here is the state a tab
arrives in after it has been used: a layout stored last session being put
back, a section folded away, a worker still reporting into a panel whose
window has gone, and a fit whose pipeline handed back something other than a
table of coefficients.

Offscreen, CPU-only, offline. Preferences are sandboxed onto a temporary
config home, because the section layout this file restores and stores lives in
a real user settings file.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.measurement_scan import MeasurementEffect, ScanResult  # noqa: E402
from spacr.qt.widgets import measurement_scan_panel as msp        # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def preferences_are_sandboxed():
    """Refuse to run unless the preference store is the suite's throwaway one.

    Folding a section of this tab STORES the layout, and restoring one reads
    it back, so these tests write preferences. ``tests/conftest.py`` gives
    every test its own QSettings directory; this states the dependency rather
    than assuming it, because without it the file being edited is the
    developer's own.
    """
    from spacr.qt.preferences import _settings

    where = _settings().fileName()
    assert "spacr-qsettings" in where, \
        f"the preference store is not sandboxed: {where}"
    return where


def _database(directory, plate):
    """One plate's ``measurements.db``, in spaCR's own shape."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), "measurements.db")
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    cell = pd.DataFrame({
        "plateID": [plate] * 3, **{k: [v] * 3 for k, v in identity.items()},
        "object_label": [1, 2, 3],
        "area": [100.0, 200.0, 300.0],
        "perimeter": [10.0, 20.0, 30.0],
    })
    with sqlite3.connect(path) as database:
        cell.to_sql("cell", database, index=False)
    return path


def _rows(paths):
    """Input-table rows, in the shape the paired input table emits."""
    return [{"plate": f"plate{i + 1}", "score": f"plate{i + 1}_scores.csv",
             "count": f"plate{i + 1}_counts.csv", "database": path}
            for i, path in enumerate(paths)]


@pytest.fixture()
def two_plates(tmp_path):
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


@pytest.fixture()
def scan_panel(qtbot):
    widget = msp.MeasurementScanPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _effect(measurement, effect_size, *, across=0.9, within=0.9):
    """One scanned measurement, with both corrections filled in."""
    return MeasurementEffect(
        measurement=measurement, n_wells=24, n_genes=4, top_gene="ATG5",
        effect_size=effect_size, coefficient=effect_size * 2.0,
        p_value=0.01, within_run_q=within, within_run_hits=1,
        measurement_p=0.02, across_scan_q=across,
        survives_within_run=within < 0.05,
        survives_across_scan=across < 0.05)


def _result(rows):
    return ScanResult(
        rows=tuple(rows), skipped={}, block_columns=("plateID",),
        gene_column="gene", control_genes=(), within_run_method="fdr_bh",
        across_scan_method="fdr_bh", alpha=0.05, effective_n_tests=2.0)


# --------------------------------------------------------------------------- #
#  One fit's settings, and what came back from it
# --------------------------------------------------------------------------- #

def test_a_queue_over_an_unpaired_run_leaves_the_runs_own_score_alone():
    """With no pairs to rewrite, only the response is changed.

    ``column_run_settings`` points every pair's score side at the merged frame
    so the count side stays paired as the input table pairs it. A run that was
    never paired has nothing to point, and the fit is handed the settings it
    already had with one new dependent variable.
    """
    base = {"regression_type": "ols", "score_data": ["/runs/one/scores.csv"]}

    settings = msp.column_run_settings(base, "cell_area", "/merged/frame.csv")

    assert settings["dependent_variable"] == "cell_area"
    assert "paired_data" not in settings
    assert settings["score_data"] == ["/runs/one/scores.csv"]
    assert base == {"regression_type": "ols",
                    "score_data": ["/runs/one/scores.csv"]}, \
        "the live settings panel's dict must not be edited in place"


class _FittedModel:
    """A fit result that is a model, not a table -- it has no length."""

    def __init__(self, rsquared):
        self.rsquared = rsquared


def test_a_fit_that_returns_a_model_rather_than_a_table_is_still_a_run():
    """``perform_regression`` may hand back an object with no ``len``.

    The queue's job is to say whether the run happened and where its folder
    is. A payload it cannot count rows in is a run with an unknown row count,
    not a failure -- reporting it as one would put "did not fit" next to a
    results folder that is on disk.
    """
    payload = {"res_folder": "/runs/cell_area", "results": _FittedModel(0.42)}

    outcome = msp._fit_outcome("cell_area", payload)

    assert outcome.ok is True
    assert outcome.folder == "/runs/cell_area"
    assert outcome.n_results == 0
    assert "cell_area" in outcome.describe()


# --------------------------------------------------------------------------- #
#  A worker still reporting into a panel whose C++ half has gone
# --------------------------------------------------------------------------- #

def test_merge_progress_arriving_after_the_panel_is_destroyed_is_dropped(qapp):
    """A merge outlives its tab, and its progress must not take it down.

    ``_relay_progress`` runs ON THE WORKER THREAD. When the panel's C++ half
    has been destroyed, PySide6 raises ``RuntimeError: Signal source has been
    deleted`` from the emit -- inside the worker, where nothing would catch
    it, aborting a join that is still holding databases open.
    """
    import shiboken6

    panel = msp.DatabaseMergePanel(threaded=False)
    shiboken6.delete(panel)

    assert panel._relay_progress("joining cell", 120, 4000) is None
    assert panel._relay_progress("writing", 4000, 4000) is None


def test_a_fit_starting_after_the_queue_panel_is_destroyed_is_dropped(qapp):
    """The same race on the queue's side, announced before each fit."""
    import shiboken6

    panel = msp.ColumnRegressionPanel(threaded=False)
    shiboken6.delete(panel)

    assert panel._relay_started("cell_area", 0, 3) is None


def test_a_fit_finishing_after_the_queue_panel_is_destroyed_is_dropped(qapp):
    """And on the way back: an outcome nobody is left to show."""
    import shiboken6

    panel = msp.ColumnRegressionPanel(threaded=False)
    shiboken6.delete(panel)

    outcome = msp.ColumnFit(column="cell_area", ok=True, folder="/runs/a")
    assert panel._relay_result(outcome) is None


# --------------------------------------------------------------------------- #
#  A queue with no settings panel behind it
# --------------------------------------------------------------------------- #

def test_a_queue_with_no_settings_panel_fits_the_response_and_nothing_else(
        qtbot, tmp_path):
    """Headless, or before a run is loaded, there are no settings to copy.

    The queue still has to run: the merged frame and the chosen column are
    everything a fit strictly needs. What each fit is handed is then exactly
    the column it was queued for, with nothing carried over from a settings
    panel that was not there.
    """
    frame = pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "cell_area": [100.0, 210.0, 305.0, 400.0],
        "cell_perimeter": [10.0, 21.0, 30.5, 40.0],
    })
    artefact = tmp_path / "merged_measurements.csv"
    frame.to_csv(artefact, index=False)

    handed = []

    def fit(settings):
        handed.append(dict(settings))
        folder = tmp_path / "runs" / str(settings["dependent_variable"])
        folder.mkdir(parents=True, exist_ok=True)
        return {"res_folder": str(folder), "results": pd.DataFrame({"b": [1]})}

    panel = msp.ColumnRegressionPanel(
        frame_provider=lambda: frame, settings_provider=None,
        score_provider=lambda: str(artefact), threaded=False, fit=fit)
    qtbot.addWidget(panel)
    assert panel.set_selected_columns(["cell_area"]) == 1

    assert panel.start_regressions() is True

    assert handed == [{"dependent_variable": "cell_area"}], \
        "a fit must carry the response and nothing invented around it"
    assert [(fit.column, fit.ok) for fit in panel.outcomes()] == \
        [("cell_area", True)]
    assert panel.is_running() is False
    assert "1 run(s) fitted" in panel.progress.text()


# --------------------------------------------------------------------------- #
#  Sharing the height between folded, open and hidden sections
# --------------------------------------------------------------------------- #

def test_a_folded_section_keeps_its_header_and_the_filler_takes_the_rest(
        scan_panel, qtbot):
    """Fold everything and the headers stack at the top of the tab.

    A folded section is pinned to its header height; the space it gave up goes
    to the filler below, which is what makes a fold hand its height over
    instead of leaving a hole where the panel was.
    """
    scan_panel.resize(600, 900)
    scan_panel.show()
    qtbot.waitExposed(scan_panel)
    for title in scan_panel.section_titles():
        scan_panel.set_section_expanded(title, False)

    scan_panel._share_the_height()

    splitter = scan_panel._sections
    sizes = splitter.sizes()
    filler_index = splitter.indexOf(scan_panel._filler)
    for index in range(splitter.count()):
        widget = splitter.widget(index)
        if index == filler_index or not widget.isVisible():
            continue
        assert sizes[index] == widget.minimumHeight(), \
            f"the folded {widget.title()!r} kept more than its header"
    assert sizes[filler_index] > max(
        sizes[:filler_index] + sizes[filler_index + 1:]), \
        "the space the folds gave up did not reach the filler"


def test_the_height_is_shared_even_when_the_filler_is_not_in_the_splitter(
        scan_panel, qtbot):
    """The filler is one widget among the splitter's children, not a promise.

    ``_share_the_height`` hands the unused height to it when everything is
    folded. With no filler among the children there is nowhere to put that
    height, and the folded headers must still get exactly their own -- not a
    stretched section and not an exception on the way past.
    """
    scan_panel.resize(600, 900)
    scan_panel.show()
    qtbot.waitExposed(scan_panel)
    for title in scan_panel.section_titles():
        scan_panel.set_section_expanded(title, False)
    scan_panel._filler.setParent(None)
    assert scan_panel._sections.indexOf(scan_panel._filler) == -1

    scan_panel._share_the_height()

    splitter = scan_panel._sections
    sizes = splitter.sizes()
    visible = [index for index in range(splitter.count())
               if splitter.widget(index).isVisible()]
    assert visible, "nothing was left on screen to share height between"
    for index in visible:
        assert sizes[index] == splitter.widget(index).minimumHeight()


# --------------------------------------------------------------------------- #
#  The layout the tab was left in last session
# --------------------------------------------------------------------------- #

def test_a_stored_layout_comes_back_and_is_not_written_over_as_it_lands(
        scan_panel):
    """Restoring folds emits the same toggles a click does.

    Each of those toggles stores the layout, so a restore that did not say it
    was restoring would write the half-restored arrangement back over the one
    it was reading -- and the section it had not reached yet would come back
    open next session.
    """
    from spacr.qt.preferences import get_section_layout, set_section_layout

    titles = scan_panel.section_titles()
    stored_sizes = [40] * (scan_panel._sections.count() - 1)
    set_section_layout(scan_panel.LAYOUT_KEY, folded=[titles[-1]],
                       sizes=stored_sizes)

    assert scan_panel.restore_section_layout() is True

    assert scan_panel.is_section_expanded(titles[-1]) is False
    assert all(scan_panel.is_section_expanded(title)
               for title in titles[:-1]), \
        "only the stored fold should be folded"
    assert get_section_layout(scan_panel.LAYOUT_KEY) == {
        "folded": [titles[-1]], "sizes": stored_sizes}, \
        "the restore wrote its own half-finished state back out"
    assert scan_panel._restoring is False


def test_a_stored_layout_whose_sizes_do_not_fit_restores_the_folds_only(
        scan_panel):
    """A sizes list from a tab with a different number of sections.

    The folds still name themselves and are still put back. The dividers are
    left where they are rather than being handed a list that does not describe
    this splitter -- which would drop sections off the bottom of the tab.
    """
    from spacr.qt.preferences import set_section_layout

    titles = scan_panel.section_titles()
    scan_panel.resize(600, 900)
    for title in titles:
        scan_panel.set_section_expanded(title, True)
    before = scan_panel._sections.sizes()
    set_section_layout(scan_panel.LAYOUT_KEY, folded=list(titles[:1]),
                       sizes=[11, 22])

    assert scan_panel.restore_section_layout() is True

    assert scan_panel.is_section_expanded(titles[0]) is False
    assert scan_panel._sections.sizes() == before, \
        "a sizes list of the wrong length must not be applied"


def test_folding_a_section_this_tab_does_not_have_changes_nothing(scan_panel):
    """A stored layout may name a section a later version dropped.

    Naming one is not an error -- the arrangement is keyed by title so it
    survives a tab being rebuilt -- and the sections that ARE here keep the
    state they were in.
    """
    before = {title: scan_panel.is_section_expanded(title)
              for title in scan_panel.section_titles()}

    scan_panel.set_section_expanded("Sweep runs", True)

    assert {title: scan_panel.is_section_expanded(title)
            for title in scan_panel.section_titles()} == before


# --------------------------------------------------------------------------- #
#  What the tab can say about the two halves it needs
# --------------------------------------------------------------------------- #

def test_merged_databases_alone_say_nothing_about_a_run_that_is_not_loaded(
        qtbot, tmp_path, two_plates):
    """One half is not an overlap.

    ``what_is_available`` names both halves and whether their wells meet. With
    databases merged and no run loaded there is no second half to compare, and
    a sentence about an overlap with nothing would be worse than silence.
    """
    panel = msp.MeasurementScanPanel(
        database_provider=lambda: _rows(two_plates),
        destination_provider=lambda: str(tmp_path / "out"),
        threaded=False)
    qtbot.addWidget(panel)

    assert panel.databases.merge() is not None
    assert len(panel.databases_frame()) > 0
    panel.set_frame_provider(None)

    assert panel.what_is_available() == ""


# --------------------------------------------------------------------------- #
#  Re-ranking the result table
# --------------------------------------------------------------------------- #

class _ResultWithoutTheAcrossScanColumn:
    """A scan result whose table does not carry the across-scan correction.

    The ranking column comes from the combo box and the table comes from the
    result, and nothing binds the two together: ``set_result`` takes any
    result that can produce a table and name its rows. A result assembled
    without one of the offered rankings is the case the re-sort guards.
    """

    skipped: dict = {}
    genes_dropped: dict = {}

    def __init__(self, rows):
        self.rows = tuple(rows)

    def frame(self):
        table = pd.DataFrame([vars(row) for row in self.rows])
        return table.drop(columns=["across_scan_q"])

    def surviving(self):
        return tuple(row for row in self.rows if row.survives_across_scan)


def test_ranking_on_a_column_the_result_does_not_carry_keeps_its_own_order(
        scan_panel):
    """The table is still shown, in the order the result produced it.

    Sorting on a column that is not there would raise inside a slot on a combo
    box -- a tab that goes blank when a user changes how it is ranked. Leaving
    the order alone shows the same rows the previous ranking showed.
    """
    rows = [_effect("cell_area", 2.5), _effect("cell_perimeter", 0.4),
            _effect("nucleus_area", 1.2)]
    scan_panel.set_result(_ResultWithoutTheAcrossScanColumn(rows))
    shown_first = scan_panel.table._frame["measurement"].tolist()

    scan_panel._rank.setCurrentIndex(
        scan_panel._rank.findData("across_scan_q"))

    assert scan_panel._rank.currentData() == "across_scan_q"
    assert scan_panel.table._frame["measurement"].tolist() == shown_first
    assert "verdict" in scan_panel.table._frame.columns


def test_clearing_the_selection_announces_no_measurement(scan_panel, qtbot):
    """Nothing selected is not a measurement.

    The tab emits the selected row so a host can draw it. Emitting on the way
    out of a selection would redraw the measurement the user just stopped
    looking at.
    """
    scan_panel.set_result(_result([_effect("cell_area", 2.5),
                                   _effect("nucleus_area", 1.2)]))
    with qtbot.waitSignal(scan_panel.measurement_selected, timeout=1000) as got:
        scan_panel.table.table.selectRow(0)
    assert got.args == ["cell_area"]

    with qtbot.assertNotEmitted(scan_panel.measurement_selected):
        scan_panel.table.table.clearSelection()
