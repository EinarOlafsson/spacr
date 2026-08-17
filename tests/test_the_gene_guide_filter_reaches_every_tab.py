"""One filter, every tab -- and the panel says which family it is drawing.

Instruction 128 L, in the maintainer's words:

    "i should be able to right click on the coeffisients table and only see
     grna or genes and this should also filer the subsequent data/graphs in
     the subsequent tabs"

129 A put genes / guides / both on the VOLCANO's right-click menu and stopped
there. The coefficient table, the p-value histogram, the Q-Q, the control
panel and the guide support all kept drawing the whole fit, so "genes only"
left FIVE tabs disagreeing with the sixth at the same time, with nothing on
screen saying which was which. That is worse than no filter: a reader who
trusts the volcano and then reads the inflation figure off the Q-Q beside it
has combined two multiple-testing families and cannot tell.

AND THE Q-Q IS NOT MERELY NARROWED, IT IS A DIFFERENT DIAGNOSTIC. Measured on
the fixture below -- 200 genes with uniform p and 600 guides with p ~ U^4, the
shape of a screen whose guide-level tests are inflated and whose gene-level
tests are not:

    whole fit    inflation at the median = 2.90, first bin holds 212 extra
    genes only   inflation at the median = 0.97, first bin holds   0 extra
    guides only  inflation at the median = 4.07, first bin holds 216 extra

Three numbers, three answers to "is this screen calibrated", one plot. So the
family is written into the tab label AND into the plot's own title, and a
title survives a click where a status line does not.

THE WELL-LEVEL TABS ARE NOT FILTERED, ON PURPOSE, AND SAY SO. A residual is
one WELL; a well is neither a gene nor a guide, and there is no honest split
to make. Silence there is what let five tabs narrow while three did not.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

GENES, GUIDES_PER_GENE = 200, 3
GUIDES = GENES * GUIDES_PER_GENE
ROWS = GENES + GUIDES


def _frame(seed: int = 3) -> pd.DataFrame:
    """Genes calibrated, guides inflated -- two families in one table.

    ``p ~ U(0, 1)`` for the gene terms and ``p ~ U(0, 1) ** 4`` for the guide
    terms, which is a null gene family and a heavily enriched guide family.
    Filtering to one or the other therefore has to move the Q-Q and the
    histogram, or the filter is not reaching them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gene in range(GENES):
        rows.append({"feature": f"gene_fraction:gene[{400000 + gene}]",
                     "coefficient": rng.normal(),
                     "p_value": rng.uniform(),
                     "condition": "other"})
        for guide in range(GUIDES_PER_GENE):
            rows.append({
                "feature": f"fraction:grna[{400000 + gene}_{guide}]",
                "coefficient": rng.normal(),
                "p_value": rng.uniform() ** 4,
                "condition": "nc" if guide == 0 else "other"})
    return pd.DataFrame(rows)


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    assert widget.set_frame(_frame(), source="results.csv")
    return widget


def _tabs(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count())]


def _tab(panel, starts_with):
    for index in range(panel.tabs.count()):
        if panel.tabs.tabText(index).startswith(starts_with):
            return index
    raise AssertionError(f"no tab starting {starts_with!r}: {_tabs(panel)}")


def _inflation(panel) -> float:
    """The genomic inflation figure the Q-Q tab is currently reporting."""
    found = re.search(r"Inflation at the median is ([0-9.]+)",
                      panel.qq._status.text())
    assert found, panel.qq._status.text()
    return float(found.group(1))


def _first_bin_excess(panel) -> int:
    """How many more coefficients the histogram's first bar holds than flat."""
    found = re.search(r"first bin holds (\d+) more",
                      panel.p_values._status.text())
    assert found, panel.p_values._status.text()
    return int(found.group(1))


def _title(plot) -> str:
    return str(plot.plot.plotItem.titleLabel.text)


def _fill_the_well_level_tabs(panel, wells: int = 240) -> None:
    """Give the panel a real fit, so the three well-level tabs have content.

    They are drawn from WELLS -- 240 of them here, against 800 coefficients in
    the table -- which is the whole reason the gene/guide filter cannot reach
    them.
    """
    statsmodels = pytest.importorskip("statsmodels.api")
    design = pd.DataFrame({"Intercept": np.ones(wells),
                           "a": np.linspace(0, 1, wells)})
    y = 0.3 + 1.2 * design["a"] + np.random.default_rng(1).normal(
        scale=0.2, size=wells)
    assert panel.set_diagnostics(statsmodels.OLS(y, design).fit(),
                                 regression_type="ols")


# --------------------------------------------------------------------------- #
#  The filter reaches every tab that draws coefficients
# --------------------------------------------------------------------------- #

def test_the_coefficient_table_narrows_with_the_volcano(panel):
    """The tab the maintainer named. It used to keep all 800 rows while the
    volcano beside it drew 200, which is two tabs disagreeing on screen."""
    assert panel.table.table.rowCount() == ROWS

    panel.set_level("gene")

    assert panel.table.table.rowCount() == GENES
    assert len(panel.volcano._row_xy) == GENES


def test_the_q_q_narrows_with_it(panel):
    panel.set_level("grna")

    assert f"{GUIDES} tests" in panel.qq._status.text(), (
        panel.qq._status.text())


def test_the_p_value_histogram_narrows_with_it(panel):
    panel.set_level("gene")

    assert f"{GENES} p-values" in panel.p_values._status.text(), (
        panel.p_values._status.text())


def test_the_control_panel_narrows_with_it(panel):
    """One guide per gene is a negative control in the fixture and no gene
    term is, so "genes only" has to empty the negative group entirely. A
    control panel still showing 200 negatives is drawing the whole table."""
    negatives, others = panel.controls.group_sizes()
    assert (negatives, others) == (GENES, GENES * 2 + GENES), (
        negatives, others)   # 200 nc guides; 400 other guides + 200 genes

    panel.set_level("gene")

    assert panel.controls.group_sizes() == [GENES], (
        "the control groups did not follow the filter")


def test_the_guide_support_tab_lists_only_the_filtered_genes(panel):
    """Narrowed by which genes are LISTED, never by how one was measured --
    see ``RegressionResultsPanel.GUIDE_SUPPORT_NEEDS_BOTH``."""
    frame = panel.results_frame()
    # Half the genes lose their gene-level term: in the fit they exist only as
    # a bundle of guides, so "genes only" has no dot for them on the volcano
    # and must have no row for them here either.
    kept = {f"gene_fraction:gene[{400000 + gene}]"
            for gene in range(0, GENES, 2)}
    is_gene_term = frame["feature"].str.startswith("gene_fraction:gene")
    trimmed = frame[~is_gene_term | frame["feature"].isin(kept)]
    assert panel.set_frame(trimmed.reset_index(drop=True))
    assert panel.support.table.rowCount() == GENES, (
        "every gene still has guides, so all of them are listed unfiltered")

    panel.set_level("gene")

    assert panel.support.table.rowCount() == len(kept)


def test_every_tab_agrees_on_the_row_count(panel):
    """The whole point. Six views of one table, and after a filter they must
    all be views of the SAME subset -- not four of them."""
    panel.set_level("grna")

    assert len(panel.filtered_frame()) == GUIDES
    assert panel.table.table.rowCount() == GUIDES
    assert len(panel.volcano._row_xy) == GUIDES
    assert f"{GUIDES} tests" in panel.qq._status.text()
    assert f"{GUIDES} p-values" in panel.p_values._status.text()
    assert sum(panel.controls.group_sizes()) == GUIDES


def test_going_back_to_both_restores_every_tab(panel):
    panel.set_level("gene")
    panel.set_level(None)

    assert panel.table.table.rowCount() == ROWS
    assert len(panel.volcano._row_xy) == ROWS
    assert f"{ROWS} tests" in panel.qq._status.text()


# --------------------------------------------------------------------------- #
#  Filtering the family changes the calibration, so the panel says which
# --------------------------------------------------------------------------- #

def test_the_inflation_figure_is_a_different_number_per_family(panel):
    """This is why a filtered Q-Q must be labelled. The same plot answers
    "is this screen calibrated" with 2.9, 0.97 and 4.07 depending only on
    which family is selected, and none of those is wrong."""
    both = _inflation(panel)
    panel.set_level("gene")
    genes = _inflation(panel)
    panel.set_level("grna")
    guides = _inflation(panel)

    assert genes < both < guides, (both, genes, guides)
    assert genes == pytest.approx(1.0, abs=0.15), (
        "the gene family is uniform by construction and should be calibrated")
    assert guides > 3.0, "the guide family is inflated by construction"


def test_the_first_bin_excess_is_a_different_number_per_family(panel):
    panel.set_level("gene")
    genes = _first_bin_excess(panel)
    panel.set_level("grna")
    guides = _first_bin_excess(panel)

    assert genes == 0, "uniform p-values have no excess in the first bin"
    assert guides > 100, guides


def test_the_q_q_tab_says_which_family_it_is_drawing(panel):
    panel.set_level("gene")

    assert panel.tabs.tabText(_tab(panel, "Q-Q")) == "Q-Q (genes)"
    tip = panel.tabs.tabToolTip(_tab(panel, "Q-Q"))
    assert "multiple-testing family" in tip, tip
    assert f"{GENES} of {ROWS}" in tip, tip


def test_the_plot_title_carries_the_family_too(panel):
    """A tab label is above the tab bar; the title is above the axes, which
    is where the reader's eye is. Both, because a status line is overwritten
    by whatever was last clicked and a title is not."""
    panel.set_level("grna")

    assert _title(panel.qq) == "p-value Q-Q — guides only"
    assert _title(panel.p_values) == "p-value distribution — guides only"
    assert _title(panel.volcano) == "Volcano — guides only"


def test_the_title_survives_a_click_where_a_status_line_would_not(panel):
    """The reason the family is in the title. Clicking a point rewrites every
    plot's status line -- so a family written there is gone the moment the
    reader uses the panel."""
    panel.set_level("gene")
    key = str(panel.filtered_frame()["feature"].iloc[0])

    panel._select_key(key)

    assert key in panel.qq._status.text(), "the click did not reach the Q-Q"
    assert _title(panel.qq) == "p-value Q-Q — genes only"


def test_unfiltered_is_said_as_plainly_as_filtered(panel):
    """"the whole fit" is a claim too, and the default is where a reader is
    most likely to assume rather than read."""
    note = panel.family_note()

    assert f"all {ROWS} coefficients" in note, note
    assert panel.tabs.tabText(_tab(panel, "Q-Q")) == "Q-Q"
    assert _title(panel.qq) == "p-value Q-Q"


# --------------------------------------------------------------------------- #
#  The tabs the filter cannot reach say so, rather than staying quiet
# --------------------------------------------------------------------------- #

def test_the_well_level_tabs_say_they_are_not_filtered(panel):
    """A residual is one WELL. There is no gene/guide split to make, and
    saying nothing is what let five tabs narrow while three did not."""
    _fill_the_well_level_tabs(panel)

    panel.set_level("gene")

    for plot in panel.diagnostic_plots():
        said = plot._status.text()
        assert "NOT FILTERED" in said, said
        assert "neither a gene nor a guide" in said, said


def test_the_well_level_note_goes_away_when_the_filter_does(panel):
    _fill_the_well_level_tabs(panel)
    panel.set_level("gene")

    panel.set_level(None)

    for plot in panel.diagnostic_plots():
        assert "NOT FILTERED" not in plot._status.text()


def test_a_run_that_arrives_after_the_filter_still_carries_the_note(panel):
    """`set_diagnostics` rewrites all three headlines, which clears the note.
    A user who filters and THEN finishes a run must not lose the sentence."""
    panel.set_level("grna")

    _fill_the_well_level_tabs(panel)

    assert "NOT FILTERED" in panel.residuals._status.text()


def test_the_guide_support_tab_says_it_needs_both_families(panel):
    """`guide_support` counts GUIDE rows: hand it a gene-only table and it
    returns nothing at all. So the concordance is always computed from the
    whole table, and the tab says so rather than looking like a filter that
    quietly did something else."""
    from spacr.guide_concordance import guide_support

    genes_only = panel.results_frame()[
        panel.results_frame()["feature"].str.startswith("gene_fraction")]
    assert not len(guide_support(genes_only)), (
        "the premise is gone: guide_support now works without guide rows")

    panel.set_level("gene")

    tip = panel.tabs.tabToolTip(_tab(panel, "Guide support"))
    assert "Computed from the whole table" in tip, tip
    assert panel.support.table.rowCount() == GENES, (
        "the support table lost its rows to the filter")


# --------------------------------------------------------------------------- #
#  The gesture on the coefficients table
# --------------------------------------------------------------------------- #

def test_the_coefficients_table_offers_the_three_levels_with_counts(panel):
    text = [action.text() for action in panel.build_level_menu().actions()]

    assert f"genes only ({GENES})" in text, text
    assert f"guides only ({GUIDES})" in text, text
    assert f"genes and guides ({ROWS})" in text, text


def test_the_menu_marks_what_is_already_chosen(panel):
    panel.set_level("grna")

    checked = [a.text() for a in panel.build_level_menu().actions()
               if a.isCheckable() and a.isChecked()]

    assert checked == [f"guides only ({GUIDES})"], checked


def test_choosing_from_the_table_menu_filters_every_tab(panel):
    """The user's own click, not a call to `set_level`: the action that the
    menu hands them is what has to be wired to the panel-wide filter."""
    action = next(a for a in panel.build_level_menu().actions()
                  if a.text().startswith("genes only"))

    action.trigger()

    assert panel.level() == "gene"
    assert panel.table.table.rowCount() == GENES
    assert f"{GENES} p-values" in panel.p_values._status.text()


def test_a_right_click_on_the_table_opens_that_menu(panel, monkeypatch):
    """The gesture itself, driven through the real signal the table emits.

    The menu is executed rather than returned, so it is intercepted at the
    instance -- patching `QMenu.exec` on the CLASS silently does not take on a
    Shiboken type, which was measured here before this test was written.
    """
    from PySide6.QtCore import QPoint
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    opened = {}
    build = RegressionResultsPanel.build_level_menu

    def intercepted(self):
        menu = build(self)
        opened["actions"] = [action.text() for action in menu.actions()]
        menu.exec = lambda *args, **kwargs: None
        return menu

    monkeypatch.setattr(RegressionResultsPanel, "build_level_menu",
                        intercepted)

    panel.table.table.customContextMenuRequested.emit(QPoint(5, 5))

    assert opened, "right-clicking the coefficients table opened no menu"
    assert f"genes only ({GENES})" in opened["actions"], opened


def test_the_table_is_wired_for_a_context_menu_at_all(panel):
    """Without this policy Qt never emits the signal, and the whole gesture
    is a connection to something that cannot fire."""
    from PySide6.QtCore import Qt

    assert panel.table.table.contextMenuPolicy() == Qt.CustomContextMenu


def test_both_gestures_set_the_same_state(panel):
    """ONE PIECE OF STATE. Two menus that each kept their own idea of the
    level is precisely the disagreement this instruction is about."""
    volcano_entry = next(
        a for a in panel.volcano.build_style_menu().actions()
        if a.text().startswith("guides only"))

    volcano_entry.trigger()

    assert panel.level() == "grna"
    checked = [a.text() for a in panel.build_level_menu().actions()
               if a.isCheckable() and a.isChecked()]
    assert checked == [f"guides only ({GUIDES})"], (
        "the table's menu does not know what the volcano's menu chose")


# --------------------------------------------------------------------------- #
#  What the filter must not break
# --------------------------------------------------------------------------- #

def test_the_run_s_table_is_never_edited(panel):
    """A view, not an edit. A caller exporting the results must get the fit
    rather than whatever the user last right-clicked -- which is why
    `results_frame` and `filtered_frame` are two methods."""
    panel.set_level("gene")

    assert len(panel.results_frame()) == ROWS
    assert len(panel.filtered_frame()) == GENES


def test_a_new_table_clears_the_filter_and_every_label_with_it(panel):
    """A new run is a new experiment. A tab left saying "(genes)" over the
    next screen's whole fit is the same lie the filter was built to stop."""
    panel.set_level("gene")

    panel.set_frame(_frame(seed=9))

    assert panel.level() is None
    assert _tabs(panel) == ["Volcano", "p-values", "Q-Q", "Controls",
                            "Residuals", "Scale-location", "Influence",
                            "Summary", "Guide support", "Gene"]
    assert _title(panel.qq) == "p-value Q-Q"


def test_the_selection_survives_the_filter_on_every_plot(panel):
    """The ring the user was reading must not vanish and leave them to find
    their guide again -- the rule the colouring and the baseline follow."""
    key = str(panel.results_frame()["feature"].iloc[0])   # a gene row
    panel._select_key(key)

    panel.set_level("gene")

    assert panel.volcano._selected_key == key
    assert panel.qq._selected_key == key


def test_the_filter_works_before_any_table_has_arrived(qtbot):
    """A widget is usable the moment it exists. Filtering an empty panel used
    to be the sort of half-built state that crashes on the run AFTER the one
    that set it -- and the labels have to follow `level()` even here, or they
    disagree with it for as long as it takes a run to finish."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)

    assert widget.filtered_frame() is None
    widget.refresh_views()
    widget.set_level("grna")

    assert widget.level() == "grna"
    assert widget.filtered_frame() is None
    assert widget.tabs.tabText(_tab(widget, "Q-Q")) == "Q-Q (guides)"
    assert "0 of 0 coefficients" in widget.status_text(), widget.status_text()


def test_an_empty_family_is_an_answer_not_a_blank_panel(qtbot):
    """A guide-only results table filtered to "genes only" has no rows. Every
    tab has to say so rather than showing an empty picture with no caption."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    guides = _frame()
    guides = guides[guides["feature"].str.startswith("fraction:grna")]
    assert widget.set_frame(guides.reset_index(drop=True))

    widget.set_level("gene")

    assert widget.table.table.rowCount() == 0
    assert f"0 of {GUIDES} coefficients" in widget.status_text(), (
        widget.status_text())
    assert "No p-values" in widget.p_values._status.text()
    said = widget.agreement._status.text()
    assert "was fitted a gene-level term" in said, said
    assert "Computed from the whole table" in said, said
