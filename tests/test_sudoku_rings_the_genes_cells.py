"""A gene-level sudoku montage rings the cells it called.

Reported 2026-08-21: "i press sudoku and i get a bunch of wells with 3 cells
in each none with a blue ring".

`sudoku` CALLS A CELL FOR A GUIDE. A gene-level montage -- which is the
ordinary case -- is named for the GENE, and the picker compared the call
against that name. A gene is never a guide, so the comparison was false for
every cell of every gene montage ever drawn: `mine` came back empty, the
picker returned None, and where it did run nothing was ever marked.

NO ERROR ANYWHERE. The montage drew, the cells were there, and not one had a
ring -- which is exactly what was reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def screen():
    """Three wells, two genes, and measurements that separate the cells."""
    rng = np.random.default_rng(0)
    rows = []
    for column in (1, 2, 3):
        for obj in range(12):
            rows.append({
                "plateID": "p1", "rowID": "r1", "columnID": f"c{column}",
                "fieldID": "f1", "objectID": str(obj),
                "prcfo": f"p1_r1_c{column}_f1_o{obj}",
                "pred": float(rng.uniform(0.0, 1.0)),
                "cell_area": float(rng.normal(500 + 80 * (obj % 3), 20)),
                "nucleus_area": float(rng.normal(200 + 30 * (obj % 3), 10)),
            })
    objects = pd.DataFrame(rows)
    counts = pd.DataFrame([
        {"plateID": "p1", "rowID": "r1", "columnID": "c1",
         "grna": "GRA14_1", "gene": "GRA14", "fraction": 0.6},
        {"plateID": "p1", "rowID": "r1", "columnID": "c1",
         "grna": "OTHER_1", "gene": "OTHER", "fraction": 0.4},
        {"plateID": "p1", "rowID": "r1", "columnID": "c2",
         "grna": "GRA14_2", "gene": "GRA14", "fraction": 0.5},
        {"plateID": "p1", "rowID": "r1", "columnID": "c2",
         "grna": "OTHER_1", "gene": "OTHER", "fraction": 0.5},
        {"plateID": "p1", "rowID": "r1", "columnID": "c3",
         "grna": "OTHER_1", "gene": "OTHER", "fraction": 0.9},
        {"plateID": "p1", "rowID": "r1", "columnID": "c3",
         "grna": "GRA14_1", "gene": "GRA14", "fraction": 0.1},
    ])
    return objects, counts


def _plan(screen, **kwargs):
    from spacr.cell_montage import select_montage

    objects, counts = screen
    return select_montage(objects, counts, "GRA14", 0.2, level="gene",
                          picking="sudoku", show_all=True, **kwargs)


class TestTheGeneMontageRingsSomething:

    def test_some_cell_is_marked(self, screen):
        """The whole report: none had a ring."""
        plan = _plan(screen)
        assert "montage_candidate" in plan.objects.columns
        assert int(plan.objects["montage_candidate"].sum()) > 0

    def test_not_every_cell_is_marked(self, screen):
        """A picker that marks everything is not picking."""
        marked = int(_plan(screen).objects["montage_candidate"].sum())
        assert marked < len(_plan(screen).objects)

    def test_the_picker_actually_ran(self, screen):
        """It used to return None and fall back, silently."""
        notes = " ".join(_plan(screen).notes)
        assert "cell(s) annotated across" in notes

    def test_it_found_the_genes_wells(self, screen):
        """`mine` was empty because a gene is never a key in guide ->
        fraction."""
        notes = " ".join(_plan(screen).notes)
        assert "is in none of these wells" not in notes
        assert "well(s) hold GRA14" in notes


class TestItMatchesOnTheGuidesNotTheName:
    """Driven through `select_montage`, which is the only caller.

    A hand-built call to `_sudoku_calls` needs the frame and the counts
    prepared exactly as `select_montage` prepares them, and getting that
    wrong tests the fixture rather than the code -- which it did on the
    first attempt, refusing with "no well has a guide fraction".
    """

    def test_the_gene_reaches_its_guides_wells(self, screen):
        """`mine` is found through the coefficient's guides, and the note
        says how many wells hold it."""
        notes = " ".join(_plan(screen).notes)
        assert "well(s) hold GRA14" in notes

    def test_naming_a_guide_that_is_in_no_well_is_refused(self, screen):
        """The other half: when the guides genuinely are not there, the
        picker says so rather than marking nothing in silence."""
        from spacr.cell_montage import CoefficientNotFound, select_montage

        objects, counts = screen
        with pytest.raises(CoefficientNotFound):
            select_montage(objects, counts, "GRA14", 0.2, level="gene",
                           picking="sudoku", show_all=True,
                           guides=["NOT_A_GUIDE_1"])

    def test_the_marked_cells_are_a_subset_of_the_wells_holding_it(
            self, screen):
        """A cell in a well the gene is absent from cannot be its cell."""
        plan = _plan(screen)
        frame = plan.objects
        marked = frame[frame["montage_candidate"].astype(bool)]
        assert set(marked["columnID"]) <= {"c1", "c2", "c3"}
        assert len(marked) > 0


class TestAGuideLevelMontageStillWorks:
    """The same code path with one guide instead of a gene's several."""

    def test_it_rings_cells_too(self, screen):
        from spacr.cell_montage import select_montage

        objects, counts = screen
        plan = select_montage(objects, counts, "GRA14_1", 0.2,
                              level="grna", picking="sudoku",
                              show_all=True)
        assert int(plan.objects["montage_candidate"].sum()) > 0
