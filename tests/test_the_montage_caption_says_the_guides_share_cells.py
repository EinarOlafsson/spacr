"""172's last open item: the guides share cells, and the caption says so.

    "sum over guides of x exceeds the well's cell count in 190 of 1,366
    wells. This is CORRECT for heuristic 1 -- guides sharing the top-ranked
    cells is exactly the flaw instruction 173 exists to fix -- but it is
    worth a word in the caption so a reader is not surprised by it."

A reader who adds up a well's guides and gets more cells than the well holds
concludes the count is wrong. It is not: every guide in a well is given
`round(n x share)` of the SAME top-ranked cells, because heuristic 1 has no
way to tell one guide's cells from another's. Unsaid, that reads as a bug.
"""
from __future__ import annotations

import pandas as pd

from spacr.cell_montage import (Coefficient, MontagePlan, ScoreWindow,
                                WellSelection)


def _plan(**kwargs) -> MontagePlan:
    window = ScoreWindow(baseline=0.5, target=0.7, scale=0.1, low=0.5,
                         high=0.9, half_widths=2.0, n_scored=100,
                         baseline_source="screen_median",
                         observed_low=0.0, observed_high=1.0)
    wells = kwargs.pop("wells", (
        WellSelection(well="1_A_01", fraction=0.4, n_objects=10,
                      n_reported=10, n_expected=4, n_in_window=8,
                      n_selected=4),))
    return MontagePlan(
        coefficient=Coefficient(name="000123", effect=0.2, level="grna"),
        window=window, wells=wells,
        objects=pd.DataFrame({"montage_well": ["1_A_01"] * 4}), **kwargs)


class TestTheCaptionSaysTheGuidesShareCells:

    def test_a_well_documents_and_displays_the_fraction_it_actually_used(self):
        well = WellSelection(well="1_A_01", fraction=0.2, share=0.75,
                             n_objects=8, n_reported=8, n_expected=6,
                             n_in_window=8, n_selected=6)

        assert ":param share:" in (WellSelection.__doc__ or "")
        assert "round(8 x 0.75) = 6" in well.describe()

    def test_the_note_is_in_the_arithmetic(self):
        said = _plan().arithmetic()

        assert "the guides are not given DIFFERENT cells" in said

    def test_it_names_the_arithmetic_a_reader_would_do(self):
        said = _plan().arithmetic()

        assert "can add to more than the well holds" in said

    def test_it_says_which_heuristic_this_is_and_what_fixes_it(self):
        """Not a bug and not a shrug: the per-cell attribution is the fix."""
        said = _plan().arithmetic()

        assert "not a miscount" in said
        assert "attribution" in said

    def test_it_reaches_the_caption_a_user_reads(self):
        assert "not given DIFFERENT cells" in _plan().caption()

    def test_a_well_over_its_own_count_is_counted(self):
        crowded = (WellSelection(well="1_A_01", fraction=0.9, n_objects=4,
                                 n_reported=4, n_expected=6, n_in_window=4,
                                 n_selected=6),)

        said = _plan(wells=crowded).arithmetic()

        assert ("6 object(s) here come from a well already at its own count"
                in said)

    def test_an_ordinary_plan_does_not_claim_crowding(self):
        assert "already at its own count" not in _plan().arithmetic()
