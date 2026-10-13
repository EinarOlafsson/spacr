"""The gene half of a run is reachable, and 'both' keeps the two apart.

Instruction 264, reported as: "in the regression module i still only get gRNA
level coefficients i cant plot the gene level coefficients ...or see the gene
level coefficients."

BOTH HALVES WERE FITTED AND BOTH WERE IN THE TABLE. What the reader could not
do was reach the gene half: the level lived on the volcano's own header and on
the coefficient table's right-click menu, so it was off screen on the six other
tabs and absent altogether when the host places the plot itself. It is now a
control on the PANEL, beside everything it changes.

WHY IT HAD BEEN PINNED, AND WHY THAT NO LONGER APPLIES. Unfiltered, a gene used
to appear once per guide plus once for itself -- `225160` was four points, all
four labelled `225160` -- which is the report "GRA14 and 225160 occur in the top
right side of the graph 4 times each". Checked on the real screen before the pin
was relaxed: the four rows now read `gene_fraction:gene[225160]`,
`fraction:grna[225160_1]`, `[_2]` and `[_3]`, and the run records the fit each
row came from in a `level` column. The distinction is real, so these tests pin
it: on the drawn table, the drawn plot and the copied export, never on a model
call.

FOUR QUESTIONS THE INSTRUCTION ASKS, one test each below:

  1. a permutation run reports guides only, so "Gene" must SAY there are none;
  2. so must a run fitted at level='grna';
  3. Benjamini-Hochberg ties are still real, so coincident points must stay
     tellable apart;
  4. a gene and its guide are different features, and selecting one must not
     highlight the other.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

GENES, GUIDES_PER_GENE = 40, 3

#: The real four-plate screen, when this is the machine that holds it. Every
#: shape claimed by the synthetic fixtures was measured here first: 1,171
#: coefficients, 790 from the guide fit and 381 from the gene fit -- including
#: one ``Intercept`` EACH, which is why the two cannot be told apart by their
#: term name.
REAL_BOTH_LEVELS = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/"
                    "results/glm_1/results.csv")
REAL_PERMUTATION = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/"
                    "results/guide_permutation_3/results.csv")


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #

def _gene_id(index: int) -> str:
    return f"{225160 + index * 10}"


def _both_levels(*, level_column: bool = True, seed: int = 0,
                 tie: bool = False) -> pd.DataFrame:
    """A run fitted at level='both', shaped like the real screen's table.

    Two fits, so TWO intercepts: the guide fit's and the gene fit's, with the
    same term name and different numbers. That pair is the reason the level
    cannot be recovered from the term alone.

    :param tie: give one gene and its first guide the SAME coefficient and the
        SAME p-value, so they land on one point of the volcano however the
        y-axis is measured. Question 3.
    """
    rng = np.random.default_rng(seed)
    rows = [{"feature": "Intercept", "coefficient": 0.4,
             "p_value": 3.1e-46, "level": "grna", "condition": "other"}]
    for gene in range(GENES):
        for guide in range(1, GUIDES_PER_GENE + 1):
            rows.append({
                "feature": f"fraction:grna[{_gene_id(gene)}_{guide}]",
                "coefficient": float(rng.normal()),
                "p_value": float(rng.uniform() ** 3),
                "level": "grna", "condition": "other"})
    rows.append({"feature": "Intercept", "coefficient": -1.6,
                 "p_value": 1.2e-30, "level": "gene", "condition": "other"})
    for gene in range(GENES):
        rows.append({
            "feature": f"gene_fraction:gene[{_gene_id(gene)}]",
            "coefficient": float(rng.normal()),
            "p_value": float(rng.uniform() ** 3),
            "level": "gene", "condition": "other"})
    frame = pd.DataFrame(rows)
    if tie:
        gene_term = f"gene_fraction:gene[{_gene_id(0)}]"
        guide_term = f"fraction:grna[{_gene_id(0)}_1]"
        for term in (gene_term, guide_term):
            frame.loc[frame["feature"] == term,
                      ["coefficient", "p_value"]] = [2.5, 1e-9]
    # THE RUN'S OWN CORRECTION, WITHIN EACH FAMILY. Two fits are two
    # multiple-testing families; a q-value pooled across the pair would be a
    # different number from the one the run wrote.
    frame["q_value"] = np.nan
    for family in ("grna", "gene"):
        rows_in = frame["level"] == family
        order = frame.loc[rows_in, "p_value"].rank(method="first")
        n = int(rows_in.sum())
        frame.loc[rows_in, "q_value"] = np.minimum(
            1.0, frame.loc[rows_in, "p_value"] * n / order)
    frame["multiple_testing_method"] = "fdr_bh"
    if not level_column:
        frame = frame.drop(columns=["level"])
    return frame


def _guides_only(seed: int = 1) -> pd.DataFrame:
    """A run fitted at level='grna': guide terms and nothing else."""
    frame = _both_levels(seed=seed)
    return frame[frame["level"] == "grna"].reset_index(drop=True)


def _permutation(seed: int = 2) -> pd.DataFrame:
    """A guide permutation's table, with the columns only it writes.

    A permutation resamples the well labels of ONE guide at a time, so the
    table is per guide by construction and holds no gene-level test at all.
    """
    frame = _guides_only(seed=seed)
    frame = frame.drop(columns=["level"])
    frame["permutations"] = 200000
    frame["permutation_exceedances"] = 0
    frame["permutation_p_value"] = frame["p_value"]
    return frame


# --------------------------------------------------------------------------- #
#  Driving the real widgets
# --------------------------------------------------------------------------- #

def _panel(qtbot, frame, source: str = ""):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.set_frame(frame, source)
    return widget


def _choose_level(qtbot, panel, wanted: str) -> None:
    """Pick a level off the PANEL's own control, the way a reader does.

    ``activated`` is the signal a combo box emits when a person chooses an
    entry, and it is what the panel is connected to -- a popup list cannot be
    clicked offscreen, so the signal is where the real gesture and the test
    meet. Everything after it is the production path.
    """
    box = panel._level_box
    index = next(i for i in range(box.count())
                 if box.itemText(i).startswith(wanted))
    box.setCurrentIndex(index)
    box.activated.emit(index)
    qtbot.wait(1)


def _drawn_features(panel) -> list:
    """Every row the coefficient TABLE is showing, as the reader reads it."""
    table = panel.table.table
    return [table.item(row, 0).text() for row in range(table.rowCount())
            if not table.isRowHidden(row)]


def _drawn_points(plot) -> list:
    """The identifier behind every point actually on ``plot``."""
    keys = []
    for item in plot.plot.listDataItems():
        if item is plot._highlight or not hasattr(item, "points"):
            continue
        for point in item.points():
            keys.append(plot.key_for_row(int(point.data())))
    return keys


def _significant_only(panel):
    """The 'significant only' checkbox -- the panel's hit list."""
    from PySide6.QtWidgets import QCheckBox

    return next(box for box in panel.table.findChildren(QCheckBox)
                if "significant" in box.text().lower())


# --------------------------------------------------------------------------- #
#  The choice is on the panel
# --------------------------------------------------------------------------- #

def test_the_three_levels_are_offered_on_the_panel_not_only_on_the_plot(qtbot):
    """gRNA, Gene and Both, on the panel, with the row count of each.

    The volcano has carried this control for a while and it was not enough:
    it is off screen on the p-value tab, the Q-Q, the effect ranks and the
    guide support, and it is not in this panel at all when a host places the
    plot itself. The reported symptom was "i cant plot the gene level
    coefficients", from a panel that could.
    """
    panel = _panel(qtbot, _both_levels())
    box = panel._level_box

    assert [box.itemData(i) for i in range(box.count())] == \
        ["grna", "gene", None]
    assert box.itemText(0) == f"gRNA ({GENES * GUIDES_PER_GENE + 1})"
    assert box.itemText(1) == f"Gene ({GENES + 1})"
    assert box.itemText(2) == f"Both ({len(panel.results_frame())})"

    # ON THE PANEL: in the control row above the tabs, so it is on screen
    # whichever tab is open -- which is the whole complaint.
    assert panel._controls_row.isAncestorOf(box)
    qq = next(i for i in range(panel.tabs.count())
              if panel.tabs.tabText(i).startswith("Q-Q"))
    panel.tabs.setCurrentIndex(qq)
    qtbot.wait(1)
    assert box.isVisible(), "the level control is hidden on the Q-Q tab"


def test_each_choice_draws_the_rows_it_names(qtbot):
    """Asserted on the drawn table and the drawn points, not on a model call."""
    panel = _panel(qtbot, _both_levels())

    _choose_level(qtbot, panel, "gRNA")
    drawn = _drawn_features(panel)
    assert len(drawn) == GENES * GUIDES_PER_GENE + 1
    assert all(":grna[" in row or row == "Intercept" for row in drawn)
    assert not any(":gene[" in row for row in drawn)

    _choose_level(qtbot, panel, "Gene")
    drawn = _drawn_features(panel)
    assert len(drawn) == GENES + 1
    assert all(":gene[" in row or row == "Intercept" for row in drawn)
    assert not any(":grna[" in row for row in drawn)

    _choose_level(qtbot, panel, "Both")
    drawn = _drawn_features(panel)
    assert len(drawn) == len(panel.results_frame())
    assert sum(":gene[" in row for row in drawn) == GENES
    assert sum(":grna[" in row for row in drawn) == GENES * GUIDES_PER_GENE


def test_gene_reaches_the_table_the_hit_list_and_the_volcano(qtbot):
    """All three, because the request named plotting AND seeing."""
    panel = _panel(qtbot, _both_levels())
    _choose_level(qtbot, panel, "Gene")

    on_the_plot = _drawn_points(panel.volcano)
    assert len(on_the_plot) == GENES, on_the_plot[:4]
    assert all(":gene[" in str(key) for key in on_the_plot)

    # THE HIT LIST is this table with the significance cut on. Driven through
    # the checkbox rather than the frame: the cut and the level are two
    # filters over one table and the question is whether they compose.
    hits = _significant_only(panel)
    hits.setChecked(True)
    qtbot.wait(1)
    listed = _drawn_features(panel)
    assert listed, "no gene passed the significance cut in this fixture"
    assert not any(":grna[" in row for row in listed), listed[:4]
    assert any(":gene[" in row for row in listed), listed[:4]


# --------------------------------------------------------------------------- #
#  'Both' keeps the two apart
# --------------------------------------------------------------------------- #

def test_a_gene_and_its_guides_are_four_rows_and_no_two_share_a_label(qtbot):
    """The original report, pinned so it cannot come back.

    `225160` was four points all labelled `225160`. Four rows is right -- the
    run fitted four coefficients -- and one label for the four is not.
    """
    panel = _panel(qtbot, _both_levels())
    _choose_level(qtbot, panel, "Both")

    gene = _gene_id(0)
    drawn = _drawn_features(panel)
    theirs = [row for row in drawn if gene in row]
    assert len(theirs) == 1 + GUIDES_PER_GENE
    assert len(set(theirs)) == len(theirs), theirs
    assert f"gene_fraction:gene[{gene}]" in theirs
    assert f"fraction:grna[{gene}_1]" in theirs

    # AND NOT JUST FOR THIS GENE. The two intercepts are the pair that a term
    # name cannot separate, so the whole drawn table is checked.
    table = panel.table.table
    columns = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    assert "level" in columns
    level = columns.index("level")
    labels = [(table.item(row, 0).text(), table.item(row, level).text())
              for row in range(table.rowCount())]
    assert len(set(labels)) == len(labels), "two drawn rows read the same"


def test_the_level_survives_into_the_exported_table(qtbot):
    """The report is where it was noticed, so the export has to carry it."""
    panel = _panel(qtbot, _both_levels(level_column=False))
    _choose_level(qtbot, panel, "Both")

    exported = panel.table.copy_visible().splitlines()
    header = exported[0].split("\t")
    assert "level" in header, header
    where = header.index("level")
    rows = {line.split("\t")[0]: line.split("\t")[where]
            for line in exported[1:]}
    gene = _gene_id(0)
    assert rows[f"gene_fraction:gene[{gene}]"] == "gene"
    assert rows[f"fraction:grna[{gene}_1]"] == "grna"
    # A term that is in NEITHER family says so rather than exporting a blank.
    assert rows["Intercept"] in ("grna", "gene", "nuisance")


def test_the_two_fits_intercepts_are_not_both_counted_as_genes(qtbot):
    """A run at level='both' fits twice, and each fit reports an intercept.

    Counting "gene" as "everything that is not a guide" files the GUIDE fit's
    intercept under genes: on the real screen that is 789 + 382 for a table of
    1,171 rows, where the run itself recorded 790 + 381. The table says which
    fit each row came from and it is believed.
    """
    panel = _panel(qtbot, _both_levels())
    counts = panel.level_counts()

    assert counts["grna"] == GENES * GUIDES_PER_GENE + 1
    assert counts["gene"] == GENES + 1
    assert counts["grna"] + counts["gene"] == counts[None]

    _choose_level(qtbot, panel, "Gene")
    drawn = _drawn_features(panel)
    assert drawn.count("Intercept") == 1, "both fits' intercepts are drawn"


# --------------------------------------------------------------------------- #
#  1 and 2 -- an empty level is an answer
# --------------------------------------------------------------------------- #

def test_a_permutation_run_says_there_are_no_gene_rows_and_why(qtbot):
    """Question 1. A permutation tests one guide at a time; there is no gene
    test in it. An empty table with no caption is indistinguishable from a
    panel that has broken."""
    panel = _panel(qtbot, _permutation())
    _choose_level(qtbot, panel, "Gene")

    assert _drawn_features(panel) == []
    said = panel.status_text()
    assert "No gene-level coefficients in this run" in said, said
    assert "permutation" in said, said
    assert "guide-level coefficients are still here" in said, said
    # ON THE PANEL, beside whichever tab is open -- not only in a header the
    # next click overwrites.
    assert panel._missing_level.isVisible()
    assert panel._missing_level.text() == said
    # AND ON THE PLOT the reader is staring at.
    assert "permutation" in panel.volcano._status.text()


def test_a_run_fitted_at_grna_says_why_it_has_no_genes(qtbot):
    """Question 2. Same rule, different reason: the gene terms were never fitted.

    The settings are read when they are there and the table when they are not,
    because a run opened from disk usually has no settings beside it -- which
    is exactly the case that needs the sentence.
    """
    panel = _panel(qtbot, _guides_only())
    panel.set_run_settings({"regression_type": "glm", "level": "grna"})
    _choose_level(qtbot, panel, "Gene")

    said = panel.status_text()
    assert "No gene-level coefficients in this run" in said, said
    assert "level='grna'" in said, said
    assert _drawn_features(panel) == []
    assert panel._missing_level.isVisible()


def test_an_empty_level_leaves_no_points_answering_for_the_last_one(qtbot):
    """Question 4's other half: the linkage must not report a hit it has not got.

    `set_results` returns early on an empty frame, and that is the one path
    through the plot that does not re-key it -- so after choosing a level this
    run has none of, the volcano drew nothing and still answered
    `highlight_key` for the guides it had drawn a moment earlier.
    """
    panel = _panel(qtbot, _permutation())
    a_guide = f"fraction:grna[{_gene_id(0)}_1]"
    assert panel.volcano.highlight_key(a_guide) is True

    _choose_level(qtbot, panel, "Gene")

    assert _drawn_points(panel.volcano) == []
    assert panel.volcano.highlight_key(a_guide) is False


# --------------------------------------------------------------------------- #
#  3 -- the ties
# --------------------------------------------------------------------------- #

def test_a_tied_gene_and_guide_stay_tellable_apart(qtbot):
    """Question 3. Two coefficients on one point are still two coefficients.

    The fixture forces the worst case: one gene and one of its guides with the
    same effect AND the same p, so they coincide on the raw axis and on the
    adjusted one. They must remain separate rows, separate identifiers,
    separately selectable -- and the level column has to say which is which.
    """
    panel = _panel(qtbot, _both_levels(tie=True))
    _choose_level(qtbot, panel, "Both")
    gene = f"gene_fraction:gene[{_gene_id(0)}]"
    guide = f"fraction:grna[{_gene_id(0)}_1]"

    volcano = panel.volcano
    gene_row = volcano._key_rows[gene]
    guide_row = volcano._key_rows[guide]
    assert gene_row != guide_row
    assert volcano._row_xy[gene_row] == volcano._row_xy[guide_row], \
        "the fixture is meant to put them on one point"

    # THE LABEL IS THE THING THAT DIFFERS, which is what the instruction asks
    # for: they may share a coordinate, they may not share a name.
    assert volcano._describe(gene_row) != volcano._describe(guide_row)
    assert gene in volcano._describe(gene_row)
    assert guide in volcano._describe(guide_row)

    # AND EITHER ONE CAN STILL BE PICKED OUT, from the table, which is the
    # route that does not depend on which of the two is drawn on top.
    assert panel.table.select_key(gene)
    qtbot.wait(1)
    assert panel.volcano._selected_keys == [gene]


def test_the_adjusted_axis_corrects_within_each_family(qtbot):
    """Ties are pulled to one height WITHIN a family, and the two fits are two.

    Benjamini-Hochberg is a cumulative minimum, so it manufactures ties -- that
    is why the original report looked like an adjusted-p bug. At 'both' the
    correction runs once per level, so a gene's q and its guide's q come off
    two different ladders and the run's own numbers are what is drawn.
    """
    panel = _panel(qtbot, _both_levels())
    _choose_level(qtbot, panel, "Both")
    volcano = panel.volcano
    volcano.set_p_axis("adjusted")
    qtbot.wait(1)

    families = {}
    for row, (_x, y) in volcano._row_xy.items():
        key = str(volcano.key_for_row(row))
        if "[" not in key:
            continue
        families.setdefault(round(y, 9), set()).add(
            "gene" if ":gene[" in key else "grna")
    mixed = [height for height, kinds in families.items() if len(kinds) > 1]
    assert not mixed, (
        f"{len(mixed)} adjusted heights hold both families, so the two fits "
        f"were corrected as one")


# --------------------------------------------------------------------------- #
#  4 -- the linkage
# --------------------------------------------------------------------------- #

def test_selecting_a_gene_does_not_highlight_its_guides(qtbot):
    """A gene and its guide are different features and different rows."""
    panel = _panel(qtbot, _both_levels())
    _choose_level(qtbot, panel, "Both")
    gene = f"gene_fraction:gene[{_gene_id(0)}]"
    guides = [f"fraction:grna[{_gene_id(0)}_{i}]"
              for i in range(1, GUIDES_PER_GENE + 1)]

    # Driven through the table's own row, which is what a reader clicks.
    table = panel.table.table
    row = next(r for r in range(table.rowCount())
               if table.item(r, 0).text() == gene)
    table.selectRow(row)
    qtbot.wait(1)

    assert panel.selected_keys() == [gene]
    assert panel.volcano._selected_keys == [gene]
    chosen = {table.item(item.row(), 0).text()
              for item in table.selectedItems()}
    assert chosen == {gene}
    for guide in guides:
        assert guide not in chosen

    # The other direction: picking a guide does not light the gene up.
    row = next(r for r in range(table.rowCount())
               if table.item(r, 0).text() == guides[0])
    table.selectRow(row)
    qtbot.wait(1)
    assert panel.selected_keys() == [guides[0]]
    assert panel.volcano._selected_keys == [guides[0]]


# --------------------------------------------------------------------------- #
#  The real screen
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_BOTH_LEVELS),
                    reason="not the machine with the real screen")
class TestTheRealScreen:
    """Synthetic tables prove the rules; the screen proves the shape.

    1,171 coefficients from one glm run at level='both' -- 790 rows from the
    guide fit and 381 from the gene fit, one intercept each.
    """

    def test_the_four_rows_for_one_gene_read_four_different_things(self,
                                                                   qtbot):
        panel = _panel(qtbot, pd.read_csv(REAL_BOTH_LEVELS),
                       REAL_BOTH_LEVELS)
        _choose_level(qtbot, panel, "Both")

        theirs = [row for row in _drawn_features(panel) if "225160" in row]
        assert sorted(theirs) == [
            "fraction:grna[225160_1]", "fraction:grna[225160_2]",
            "fraction:grna[225160_3]", "gene_fraction:gene[225160]"]

    def test_the_counts_are_the_run_s_own(self, qtbot):
        panel = _panel(qtbot, pd.read_csv(REAL_BOTH_LEVELS),
                       REAL_BOTH_LEVELS)
        counts = panel.level_counts()

        assert (counts["grna"], counts["gene"], counts[None]) == (790, 381,
                                                                  1171)

    def test_gene_draws_the_gene_fit_on_the_volcano(self, qtbot):
        panel = _panel(qtbot, pd.read_csv(REAL_BOTH_LEVELS),
                       REAL_BOTH_LEVELS)
        _choose_level(qtbot, panel, "Gene")

        drawn = _drawn_points(panel.volcano)
        assert len(drawn) == 380      # 381 rows, less the gene fit's intercept
        assert all(":gene[" in str(key) for key in drawn)

    @pytest.mark.skipif(not os.path.exists(REAL_PERMUTATION),
                        reason="not the machine with the real screen")
    def test_the_real_permutation_run_says_it_has_no_genes(self, qtbot):
        panel = _panel(qtbot, pd.read_csv(REAL_PERMUTATION), REAL_PERMUTATION)
        _choose_level(qtbot, panel, "Gene")

        assert _drawn_features(panel) == []
        said = panel.status_text()
        assert "No gene-level coefficients in this run" in said, said
        assert "permutation" in said, said
