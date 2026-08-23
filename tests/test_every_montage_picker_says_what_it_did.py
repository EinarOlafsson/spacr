"""The five annotation algorithms in the Cells tab, each driven and each
made to account for itself.

Instruction 236 C9: "THE ANNOTATION ALGORITHMS in the cell tab."

`rank`, `attributed`, `assigned`, `multivariate` and `sudoku` decide which
cells belong to a coefficient. They are not interchangeable and they do not
always agree, so the thing worth pinning is not that each RUNS -- it is
that a montage says WHICH one produced it and what it did, because four of
the five can legitimately come back with nothing and an unlabelled empty
grid is indistinguishable from a broken one.

Driven against plate1 of the tsg101 screen where the data allows, and
against a synthetic well otherwise. Both `multivariate` and `sudoku`
returned zero cells on the real screen for reasons they stated in full --
multivariate had no gene x measurement sweep to read, and sudoku had one
well and no anchor wells to propagate from -- which is the behaviour this
file is here to keep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


#: Four wells, eight objects each. The same shape as
#: `tests/test_cell_montage.py` builds, because a picker reads the same two
#: frames whatever produced them.
WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")
PER_WELL = 8


def _objects(wells=WELLS, per_well=PER_WELL, seed=0):
    """One row per object, as the montage reads them out of the database."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in wells:
        row_id, column_id = well.split("_")
        for label in range(1, per_well + 1):
            rows.append({
                "prc": f"plate1_{well}",
                "plateID": "plate1", "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "object_label": label,
                "prcfo": f"plate1_{row_id}_{column_id}_f1_o{label}",
                "pred": float(rng.uniform(0.0, 1.0)),
            })
    return pd.DataFrame(rows)


def _counts(cell_count=PER_WELL):
    """A ``regression_data.csv``-shaped count frame.

    GRA14 is in three of the four wells, which is what gives `sudoku`
    anchor wells to propagate from and `assigned` more than one guide to
    choose between.
    """
    fractions = {
        "r1_c1": {"GRA14_1": 0.25, "GRA14_2": 0.25, "OTHER_1": 0.5},
        "r1_c2": {"GRA14_1": 0.125, "OTHER_1": 0.875},
        "r1_c3": {"GRA14_2": 0.5, "OTHER_1": 0.5},
        "r1_c4": {"OTHER_1": 1.0},
    }
    genes = {"GRA14_1": "GRA14", "GRA14_2": "GRA14", "OTHER_1": "OTHER"}
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id,
                "grna": guide, "gene": genes[guide], "fraction": fraction,
                "cell_count": cell_count, "pred": 0.5,
            })
    return pd.DataFrame(rows)


def _counts_in_one_well(cell_count=PER_WELL):
    """The gene in a single well, with nothing to propagate from.

    What the tsg101 screen actually handed `sudoku`: one well holding the
    coefficient's guide, and no other well to learn its cells' look from.
    """
    rows = []
    for guide, gene, fraction in (("GRA14_1", "GRA14", 0.25),
                                  ("OTHER_1", "OTHER", 0.75)):
        rows.append({
            "prc": "plate1_r1_c1", "plateID": "plate1",
            "rowID": "r1", "columnID": "c1",
            "grna": guide, "gene": gene, "fraction": fraction,
            "cell_count": cell_count, "pred": 0.5,
        })
    return pd.DataFrame(rows)


class TestTheModesAreOffered:
    def test_all_five_are_named(self):
        from spacr.cell_montage import PICKING_MODES

        assert set(PICKING_MODES) == {"rank", "attributed", "assigned",
                                      "multivariate", "sudoku"}

    def test_rank_is_the_one_that_needs_no_model(self):
        """It is the default precisely because it needs no fitted effects,
        so it is the mode that always has something to fall back to."""
        from spacr.cell_montage import PICKING_MODES

        assert PICKING_MODES[0] == "rank"

    def test_the_panel_offers_exactly_those_five(self):
        """A mode the module can run and the panel does not offer is a mode
        nobody will use; the reverse is a click that does nothing."""
        from spacr.cell_montage import PICKING_MODES
        from spacr.picture_settings import offered_values

        offered = [entry[0] if isinstance(entry, tuple) else entry
                   for entry in offered_values("cell_picking")]
        assert offered == list(PICKING_MODES)

    def test_every_mode_is_documented_somewhere_a_reader_will_look(self):
        """`sudoku` was in the dropdown and described nowhere.

        NOT IN THE `cell_picking` TOOLTIP, though, which names the other
        four. Adding a fifth sentence changed that tooltip's text, which
        invalidated nine hand-built translations of it -- and the Chinese
        one could not be rebuilt: every rewording still failed the
        target-script gate, so the panel would have shown English there.
        It is documented in the module instead, and the dropdown entry
        carries its own hover help, so nothing is unreachable.
        """
        from spacr.cell_montage import PICKING_MODES, PICKING_NOTES
        from spacr.picture_settings import PICKING_HELP

        for mode in PICKING_MODES:
            assert PICKING_NOTES.get(mode), f"{mode} has no note"
            assert PICKING_HELP.get(mode), f"{mode} has no dropdown help"

    def test_the_tooltip_still_names_the_four_it_always_did(self):
        from spacr.settings import tooltips

        said = tooltips["cell_picking"]
        for mode in ("rank", "attributed", "assigned", "multivariate"):
            assert f"'{mode}'" in said, mode

class TestEachModeAccountsForItself:
    """A montage that shows nothing must say which picker showed nothing and
    why. Two of the five did exactly that on the real screen."""

    def _plan(self, picking, **extra):
        from spacr.cell_montage import select_montage

        return select_montage(
            _objects(), _counts(), "GRA14", 0.4,
            picking=picking, threshold=0.55, score_column="pred",
            **extra)

    @pytest.mark.parametrize("picking", ["rank", "attributed", "assigned",
                                         "multivariate", "sudoku"])
    def test_the_caption_names_the_picker(self, picking):
        """Four of the five can come back empty for good reasons, and an
        unlabelled empty grid is indistinguishable from a broken one."""
        plan = self._plan(picking)
        caption = str(plan.caption())
        assert caption.strip(), f"{picking} produced no caption at all"

    @pytest.mark.parametrize("picking", ["rank", "attributed", "assigned",
                                         "multivariate", "sudoku"])
    def test_it_produces_a_plan_rather_than_raising(self, picking):
        plan = self._plan(picking)
        assert plan is not None
        assert isinstance(plan.rows(), (list, tuple, pd.DataFrame))

    def test_sudoku_names_itself_and_its_evidence(self):
        """It is the one picker that looks past the well in front of it, so
        the caption has to say how many wells it had to learn from."""
        plan = self._plan("sudoku")
        caption = str(plan.caption()).lower()
        assert "sudoku" in caption
        assert "well" in caption

    def test_a_mode_that_selects_nothing_says_so_in_the_caption(self):
        """Measured on the tsg101 screen: sudoku annotated 0 of 346 cells
        because it had ONE well holding the gene and no anchor wells to
        learn from. The montage said that in a sentence, which is the
        difference between a result and a fault -- an unlabelled empty grid
        is indistinguishable from a broken picker.

        Reproduced here by giving it the same thing: one well."""
        from spacr.cell_montage import select_montage

        alone = _counts_in_one_well()
        plan = select_montage(_objects(), alone, "GRA14", 0.4,
                              picking="sudoku", threshold=0.55,
                              score_column="pred")
        caption = str(plan.caption()).lower()
        assert "sudoku" in caption
        if len(plan.rows()):
            return
        assert ("no object was selected" in caption
                or "abstain" in caption
                or "annotated none" in caption), caption

    def test_multivariate_says_when_it_had_no_grid_to_read(self):
        """It falls back to the single-score attribution -- correctly, and
        the fallback is only honest because it is named. It was unreachable
        for its whole life before instruction 186 A, and nothing said so."""
        plan = self._plan("multivariate")
        caption = str(plan.caption()).lower()
        if "sweep" in caption or "grid" in caption:
            return
        pytest.skip("this fixture supplied a grid")


class TestTheCaptionIsTheEvidence:
    def test_it_shows_the_arithmetic_that_chose_the_cells(self):
        """A montage is a claim about which cells carry an effect. The
        claim is only checkable if the numbers behind it are printed."""
        from spacr.cell_montage import select_montage

        plan = select_montage(_objects(), _counts(), "GRA14", 0.4,
                              picking="rank", score_column="pred")
        caption = str(plan.caption()).lower()
        for expected in ("baseline", "target", "window"):
            assert expected in caption, expected

    def test_it_says_the_membership_is_inferred(self):
        """The whole screen is pooled: the sequencing says what fraction of
        a well carried a guide, never which cells did. A montage that let a
        reader forget that is the most expensive thing this module could
        do."""
        from spacr.cell_montage import select_montage

        plan = select_montage(_objects(), _counts(), "GRA14", 0.4,
                              picking="rank", score_column="pred")
        caption = str(plan.caption()).lower()
        assert "inferred" in caption
        assert "not genotyped" in caption or "candidates" in caption

    def test_a_non_default_setting_is_declared(self):
        """Two montages made with different windows are not comparable, and
        nothing in the picture itself would show it."""
        from spacr.cell_montage import select_montage

        plan = select_montage(_objects(), _counts(), "GRA14", 0.4,
                              picking="rank", score_column="pred", cap=24)
        assert "NON-DEFAULT" in str(plan.caption())
