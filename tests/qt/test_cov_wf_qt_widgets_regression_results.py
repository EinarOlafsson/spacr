"""The results panel's recovery paths, and the edges of its level filter.

Every route here is one the panel takes when a part it leans on is missing or
has been taken away: the helper in :mod:`spacr.hits` that says which fit a row
came from, the two columns a permutation run writes under different names, the
dialog that offers a re-fit, the button that becomes the cancel, the background
runner that owns the read, a volcano too old to name its own axis, and tabs
that a host has lifted out of the panel's own tab bar to hang somewhere else.

Each is asserted from the outside -- the text on the tab, the label on the
axis, the sentence in the header -- because that is the whole of what a user
gets when one of them goes wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

import spacr.hits as hits  # noqa: E402
import spacr.regression_qc as regression_qc  # noqa: E402
from spacr.qt.widgets import refit_dialog  # noqa: E402
from spacr.qt.widgets.fast_plots import VolcanoPlot  # noqa: E402
from spacr.qt.widgets.regression_results import (  # noqa: E402
    RegressionResultsPanel,
    for_table,
    read_run_tables,
)


def _guides() -> pd.DataFrame:
    """A guide-only coefficient table, the shape a grna fit writes."""
    return pd.DataFrame({"feature": ["fraction:grna[G0_1]",
                                     "fraction:grna[G0_2]"],
                         "coefficient": [1.0, 2.0],
                         "p_value": [0.01, 0.4]})


def _panel(qtbot) -> RegressionResultsPanel:
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    return panel


def _bottom_label(panel) -> str:
    """The volcano's horizontal axis title, as the reader sees it."""
    return panel.volcano.plot.getPlotItem().getAxis("bottom").labelText


# ------------------------------------------------------- reading a run folder

def test_a_gene_sibling_is_still_folded_in_when_the_level_reader_fails(
        tmp_path, monkeypatch):
    """An older permutation run kept its genes in a second file, and the panel
    finds them by asking :mod:`spacr.hits` which levels the primary table
    already holds. If that question raises, the honest answer is "I know of
    none" -- which must still merge the sibling, because refusing to merge is
    how a reader ends up staring at an empty Gene tab while the rows sit in a
    file beside the one that was opened."""
    primary = tmp_path / "results.csv"
    sibling = tmp_path / "results_gene.csv"
    _guides().to_csv(primary, index=False)
    pd.DataFrame({"feature": ["gene_fraction:gene[G0]"],
                  "coefficient": [3.0], "p_value": [0.03]}).to_csv(
        sibling, index=False)
    tables = [str(primary), str(sibling)]

    whole, _, merged = read_run_tables(tables)
    assert merged == [str(sibling)]
    assert list(whole["level"]) == ["grna", "grna", "gene"], (
        "with the level reader working, the primary table's rows are labelled "
        "with the one level it was found to hold")

    def unreadable(frame):
        raise RuntimeError("family_labels is not available in this build")

    monkeypatch.setattr(hits, "coefficient_levels", unreadable)
    frame, found, merged = read_run_tables(tables)

    assert found == str(primary)
    assert merged == [str(sibling)], (
        "the gene rows were dropped because the level of the guide rows could "
        "not be read, which is the exact bug the merge exists to prevent")
    assert list(frame["feature"]) == ["fraction:grna[G0_1]",
                                      "fraction:grna[G0_2]",
                                      "gene_fraction:gene[G0]"]
    assert frame["level"].tolist()[2] == "gene"
    assert frame["level"].isna().tolist() == [True, True, False], (
        "with no readable level for the primary rows the merge must leave "
        "them blank rather than guess a level for them")


# ----------------------------------------------- one number under two names

def test_the_coefficient_column_survives_columns_that_cannot_be_compared():
    """A permutation run copies its partial correlation into ``coefficient``,
    so the table would show the same numbers twice under two names and a
    reader cannot tell which is real. The duplicate is dropped only when the
    two columns are PROVED equal; a table whose columns refuse to be compared
    keeps both, because dropping a column on a failed comparison would hide a
    real, different coefficient."""
    identical = pd.DataFrame({
        "feature": ["a", "b"],
        "coefficient": [0.1, 0.2],
        "standardized_marginal_effect": [0.1, 0.2]})

    kept = list(for_table(identical).columns)
    assert kept == ["feature", "standardized_marginal_effect"], (
        "the accurate name is the one that stays when the two are the same")

    class Incomparable:
        """A column that answers no question about itself."""

        def equals(self, other):
            raise RuntimeError("this column cannot be compared")

        def notna(self):
            raise RuntimeError("this column cannot be counted")

    class Table:
        columns = ["feature", "coefficient", "standardized_marginal_effect"]

        def __len__(self):
            return 3

        def __getitem__(self, name):
            return Incomparable()

    table = Table()
    narrowed = for_table(table)

    assert narrowed is table, "nothing could be dropped, so nothing was"
    assert "coefficient" in narrowed.columns, (
        "a comparison that raised was read as 'these are the same' and threw "
        "away a column the reader needs")


# ------------------------------------------------------------------- re-fit

def test_a_refit_that_changes_nothing_says_nothing_about_changes(
        qtbot, monkeypatch):
    """The re-fit dialog reports the settings it had to adjust, and the panel
    repeats them in the header so a run that quietly changed the correction
    method is not a surprise twenty minutes later. When the dialog adjusted
    nothing there is nothing to repeat, and the header must keep whatever it
    was already telling the reader instead of announcing an empty list."""
    panel = _panel(qtbot)
    panel.set_run_settings({"count_data": "/runs/counts.csv",
                            "regression_type": "ols"})
    asked = []
    panel.refit_requested.connect(asked.append)

    monkeypatch.setattr(
        refit_dialog, "ask_refit",
        lambda base, parent=None: ({"count_data": "/runs/counts.csv",
                                    "level": "both"},
                                   ["level 'grna' -> 'both'"]))
    assert panel.ask_refit() is True
    assert panel.status_text() == "Re-fitting: level 'grna' -> 'both'"

    panel.say("Nothing has been re-fitted yet.")
    monkeypatch.setattr(
        refit_dialog, "ask_refit",
        lambda base, parent=None: ({"count_data": "/runs/counts.csv"}, []))

    assert panel.ask_refit() is True
    assert panel.status_text() == "Nothing has been re-fitted yet.", (
        "an empty note list was announced as a change to the header")
    assert len(asked) == 2 and asked[1] == {"count_data": "/runs/counts.csv"}, (
        "the re-fit must still be requested when there was nothing to report")


# ------------------------------------------------------- the spread verdict

def test_the_spread_verdict_is_returned_even_with_its_tab_lifted_out(
        qtbot, monkeypatch):
    """The constant-spread verdict is written onto the Scale-location tab as a
    tooltip AND returned to the caller. A host that hangs that tab somewhere
    else -- the folded layout does exactly this -- leaves the panel with a
    widget its own tab bar does not hold, and the verdict must still reach the
    caller rather than dying on a missing tab index."""
    panel = _panel(qtbot)
    panel._qc_context = object()

    def refuses(*args, **kwargs):
        raise ValueError("no standardised residual for this fit")

    monkeypatch.setattr(regression_qc, "draw_panel", refuses)

    verdict = panel.judge_homogeneity()
    index = panel.tabs.indexOf(panel._scale_location_tab)

    assert index >= 0
    assert "could not be computed" in verdict
    assert panel.tabs.tabToolTip(index) == verdict, (
        "the tab carries the verdict while the tab is in the bar")

    panel.tabs.removeTab(index)
    assert panel.tabs.indexOf(panel._scale_location_tab) == -1

    again = panel.judge_homogeneity()

    assert again == verdict, (
        "the verdict stopped being returned once its tab left the tab bar")
    assert panel.homogeneity_verdict() == verdict
    assert panel.homogeneity_stats() == {}, (
        "a refused test has no statistics to report")


# ------------------------------------------------------------- load and stop

def test_the_loading_flag_is_recorded_when_the_button_is_already_gone(
        qtbot, monkeypatch):
    """The one button that starts a read becomes the cancel while the read
    runs. During a close the button is destroyed before the panel is, so the
    flag that says a read is in flight must be recorded whether or not there
    is still a button to relabel -- otherwise a teardown turns into a crash on
    the way out."""
    panel = _panel(qtbot)
    button = panel._load_button

    panel._set_loading(True)
    assert panel.is_loading() is True
    assert button.text() == "Cancel load", (
        "the way out of a slow read is the button that started it")

    panel._set_loading(False)
    assert button.text() == "Load results…"

    monkeypatch.delattr(panel, "_load_button")
    panel._set_loading(True)

    assert panel.is_loading() is True, (
        "the panel forgot it was reading because it had no button to relabel")
    assert button.text() == "Load results…", (
        "the detached button must not be touched")


def test_a_cancel_the_runner_refuses_still_ends_the_load(qtbot, monkeypatch):
    """Cancel has to end the load from the reader's point of view even when
    the background runner will not co-operate. A cancel that propagated the
    runner's failure would leave the panel showing "Cancel load" forever and
    every caller waiting on ``load_finished`` waiting for good."""
    panel = _panel(qtbot)
    finished = []
    panel.load_finished.connect(finished.append)

    panel._set_loading(True)
    assert panel.cancel_load() is True
    assert finished == [False]
    assert panel.is_loading() is False

    def refuses():
        raise RuntimeError("the runner was already shut down")

    monkeypatch.setattr(panel._load_jobs, "cancel", refuses)
    panel._set_loading(True)

    assert panel.cancel_load() is True, (
        "a runner that refused the cancel was reported to the user as 'no "
        "load was running'")
    assert panel.is_loading() is False
    assert finished == [False, False], (
        "a caller waiting on load_finished was left waiting by the refusal")
    assert panel.status_text().startswith("Loading was cancelled")
    assert panel.cancel_load() is False, (
        "there is nothing left to cancel once the load has ended")


# -------------------------------------------------------- the level filter

def test_a_run_with_neither_family_is_not_told_to_switch_level(qtbot):
    """When the chosen level has no rows the panel says why, and offers the
    other family when the other family is actually there. A table of nothing
    but nuisance terms has neither, and inviting the reader to "set Level to
    gRNA to read them" would send them to a second empty tab."""
    panel = _panel(qtbot)

    panel.set_frame(_guides(), "/runs/guides/results.csv")
    panel.set_level("gene")
    with_guides = panel.missing_level_note()

    assert with_guides.startswith("No gene-level coefficients in this run:")
    assert "Its 2 guide-level coefficients are still here" in with_guides
    assert "set Level to gRNA" in with_guides

    # THE SPELLING spaCR ACTUALLY WRITES. `prepare_formula` emits
    # "fraction ~ fraction:grna + plateID + rowID + columnID", so the layout
    # covariates come back as `columnID[T.c3]`. This said `column[c3]`, which
    # no spaCR run produces and which `hits.NUISANCE_TERMS` therefore does
    # not match -- so the row counted as a gene-level coefficient and the
    # test failed against a fixture, not against the code.
    nuisance = pd.DataFrame({"feature": ["Intercept", "columnID[T.c3]"],
                             "coefficient": [0.5, 0.6],
                             "p_value": [0.2, 0.3]})
    panel.set_frame(nuisance, "/runs/nuisance/results.csv")
    panel.set_level("gene")
    note = panel.missing_level_note()

    assert panel.level_counts() == {None: 2, "gene": 0, "grna": 0}
    assert note.startswith("No gene-level coefficients in this run:")
    assert "still here" not in note, (
        "the reader was pointed at a guide family this run does not have")
    assert panel.status_text().startswith(
        "No gene-level coefficients in this run:")


def test_the_family_is_written_on_plots_whose_tabs_were_lifted_out(qtbot):
    """The level filter is written onto every tab label AND into every plot's
    own title, because a plot's status line is overwritten by the next click
    while a title is not. A host that re-parents a tab out of the panel -- the
    folded layout hangs the Q-Q and the guide support elsewhere -- must still
    get the family written into those plots, or a reader looking at a filtered
    Q-Q has nothing on screen saying which family it is."""
    panel = _panel(qtbot)
    panel.set_frame(_guides(), "/runs/guides/results.csv")
    panel.set_level("gene")

    qq_index = panel.tabs.indexOf(panel.qq)
    assert panel.tabs.tabText(qq_index) == "Q-Q (genes)"
    assert panel.tabs.tabToolTip(qq_index) == panel.family_note()

    panel.tabs.removeTab(qq_index)
    panel.tabs.removeTab(panel.tabs.indexOf(panel._support_tab))
    assert panel.tabs.indexOf(panel.qq) == -1
    assert panel.tabs.indexOf(panel._support_tab) == -1

    panel.set_level("grna")

    assert panel.qq.plot.plotItem.titleLabel.text == (
        "p-value Q-Q — guides only"), (
        "a Q-Q lifted out of the tab bar stopped saying which family it draws")
    assert panel.agreement.plot.plotItem.titleLabel.text == (
        "Guide agreement — guides only")
    volcano_index = panel.tabs.indexOf(panel._volcano_tab)
    assert panel.tabs.tabText(volcano_index).endswith("(guides)"), (
        "the tabs still in the bar are still labelled")
    assert panel.tabs.tabToolTip(volcano_index) == panel.family_note()
    assert "guides only — 2 of 2 coefficients" in panel.family_note()


def test_the_histograms_narrow_the_table_where_a_point_plot_selects(qtbot):
    """A dot on the volcano IS one coefficient, so clicking a band of them
    selects them. A bar on the p-value histogram stands for many rows, so it
    narrows the table instead -- selecting "the rows behind this bar" would
    make the gene tile show one guide while the image tabs showed another.
    The two histograms are therefore not keyed plots at all."""
    panel = _panel(qtbot)
    panel.set_frame(_guides(), "/runs/guides/results.csv")
    keyed = panel._keyed_plots()

    assert panel.volcano in keyed
    assert panel.p_values not in keyed and panel.effect_distribution not in keyed, (
        "a histogram joined the select-many route, so a bar of 40 rows now "
        "claims to be a selection of 40 coefficients")

    panel.volcano.keys_selected.emit(["fraction:grna[G0_1]",
                                      "fraction:grna[G0_2]"])
    assert panel.selected_keys() == ["fraction:grna[G0_1]",
                                     "fraction:grna[G0_2]"], (
        "a band on a keyed plot selects every coefficient it encloses")

    panel.p_values.keys_selected.emit(["fraction:grna[G0_1]"])
    visible = [row for row in range(panel.table.table.rowCount())
               if not panel.table.table.isRowHidden(row)]
    assert len(visible) == 1, (
        "a histogram bar narrowed nothing, so the rows behind it were never "
        "brought to the front")
    table = panel.table.table
    row = visible[0]
    texts = [table.item(row, col).text() if table.item(row, col) else ""
             for col in range(table.columnCount())]
    assert "fraction:grna[G0_1]" in texts, (
        f"the bar's own row is not the one left showing; got {texts}")


# ----------------------------------------------------------- the volcano axis

def test_a_volcano_that_cannot_name_its_axis_still_draws_the_run(
        qtbot, monkeypatch):
    """A permutation run's horizontal axis is a partial correlation bounded in
    [-1, 1], not a coefficient, and the volcano is asked to say so on every
    redraw. A volcano too old to be asked must not take the whole redraw down
    with it -- the run is still worth drawing, with the axis under its older
    and more cautious name."""
    permuted = _guides()
    permuted["permutation_p_value"] = [0.01, 0.4]
    permuted["standardized_marginal_effect"] = [0.11, 0.22]

    panel = _panel(qtbot)
    panel.set_frame(permuted, "/runs/perm/results.csv")

    assert panel._analysis_path() == "permutation"
    assert _bottom_label(panel) == (
        "standardized marginal effect (partial correlation)"), (
        "a bounded correlation was left labelled 'coefficient'")

    older = _panel(qtbot)
    monkeypatch.delattr(VolcanoPlot, "name_the_effect")
    older.set_frame(permuted, "/runs/perm/results.csv")

    assert older.results_frame() is not None
    assert older.table.table.rowCount() == 2, (
        "an unnameable axis stopped the run being drawn at all")
    assert _bottom_label(older) == "coefficient", (
        "with nothing to name the axis it keeps the default label")
