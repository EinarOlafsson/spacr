"""Instruction 173: the pre-flight is REPORTED, not merely computable.

"WHICH GUIDES CAN BE ATTRIBUTED AT ALL, reported BEFORE anything is
assigned." `attributable` and `preflight` answered that from the day they
landed and nothing called them, so the answer existed and no user ever saw
it. Asked for on 2026-08-20 -- "wire the pre flihght".

WHY IT IS WORTH A TEST RATHER THAN A GLANCE. A guide whose effect is too
small against the spread of scores reaches the 0.55 threshold in no well, so
the attributed picker selects nothing and the montage comes back EMPTY. An
empty montage reads as a broken viewer. The pre-flight is the difference
between that and a sentence saying the guide cannot support cell-level work,
which is arithmetic and not sample size.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.guide_attribution import (DEFAULT_THRESHOLD, Preflight,
                                     attributable, preflight)


WELLS = {
    "p1_r1_c1": {"strong": 0.5, "quiet": 0.5},
    "p1_r1_c2": {"strong": 0.2, "quiet": 0.8},
}


class TestTheVerdict:

    def test_a_guide_with_a_real_effect_can_be_called(self):
        got = preflight("strong", WELLS, {"strong": 2.5, "quiet": 0.0},
                        scale=1.0)

        assert isinstance(got, Preflight)
        assert got.wells == 2
        assert got.callable_wells == 2
        assert not got.hopeless
        assert "can be attributed in 2 of its 2 wells" in got.note()

    def test_a_guide_with_no_effect_and_a_small_share_can_never_be_called(self):
        """THE SHARE MATTERS AS MUCH AS THE EFFECT. A guide that is 80% of a
        well clears 0.55 on its prior alone, with no effect at all -- which
        is correct, and is why ~45% of cells come back "called" even under a
        permutation on the real screen. A guide that is a fiftieth of the
        well cannot."""
        wells = {"a": {"trace": 0.02, "bulk": 0.98},
                 "b": {"trace": 0.05, "bulk": 0.95}}
        got = preflight("trace", wells, {"trace": 0.0, "bulk": 0.0},
                        scale=1.0)

        assert got.hopeless
        note = got.note()
        assert "CANNOT BE ATTRIBUTED" in note
        assert "more cells will not change it" in note, (
            "the point of the report is that it is arithmetic, not sample size")
        assert "picked by rank" in note, "say what happens instead"

    def test_the_verdict_is_per_well_because_the_prior_is(self):
        """A guide at half a well and at a fiftieth of one is not the same
        question, so a single yes/no for the whole screen would be a lie."""
        wells = {"rich": {"g": 0.9, "other": 0.1},
                 "trace": {"g": 0.01, "other": 0.99}}
        got = preflight("g", wells, {"g": 0.35, "other": 0.0}, scale=1.0)

        assert 0 < got.callable_wells < got.wells

    def test_a_guide_no_well_carries_says_so(self):
        got = preflight("absent", WELLS, {}, scale=1.0)

        assert got.wells == 0
        assert not got.hopeless, "no wells is not the same as no hope"
        assert "no well carries it" in got.note()

    def test_the_ceiling_is_the_best_over_the_wells(self):
        effects = {"strong": 1.2, "quiet": 0.0}
        got = preflight("strong", WELLS, effects, scale=1.0)
        each = [attributable(1.2, 1.0, f["strong"] / sum(f.values()),
                             others=[(0.0, f["quiet"] / sum(f.values()))])[1]
                for f in WELLS.values()]

        assert got.best == pytest.approx(max(each))

    def test_it_sees_the_competition(self):
        """Against an opposite-signed competitor a guide separates twice as
        fast, and a ceiling blind to that under-reports it."""
        opposed = preflight("strong", {"w": {"strong": 0.1, "quiet": 0.9}},
                            {"strong": 0.5, "quiet": -0.5}, scale=1.0)
        flat = preflight("strong", {"w": {"strong": 0.1, "quiet": 0.9}},
                         {"strong": 0.5, "quiet": 0.0}, scale=1.0)

        assert opposed.best > flat.best


class TestItReachesTheMontage:
    """The wiring, which is the part that was missing."""

    @pytest.fixture
    def screen(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        rows = []
        for well in ("plate1_r1_c1", "plate1_r1_c2"):
            for index in range(40):
                rows.append({
                    "prc": well, "object_label": index,
                    "pred": float(rng.uniform(0.0, 1.0)),
                    "png_path": f"/tmp/{well}_{index}.png",
                })
        objects = pd.DataFrame(rows)
        counts = pd.DataFrame([
            {"prc": "plate1_r1_c1", "grna": "hopeless", "fraction": 0.5},
            {"prc": "plate1_r1_c1", "grna": "other", "fraction": 0.5},
            {"prc": "plate1_r1_c2", "grna": "hopeless", "fraction": 0.5},
            {"prc": "plate1_r1_c2", "grna": "other", "fraction": 0.5},
        ])
        return objects, counts

    def _plan(self, screen, picking, effects):
        from spacr.cell_montage import select_montage

        objects, counts = screen
        # THE FULL COUNT TABLE, not the coefficient's own rows. A posterior
        # is a comparison, and a lone guide has a prior of 1.0 and a ceiling
        # of 1.0 -- the pre-flight would call every guide attributable.
        return select_montage(
            objects,
            counts,
            "hopeless", 0.0,
            level="grna",
            score_column="pred",
            picking=picking,
            effects=effects,
        )

    def test_an_unattributable_guide_says_so_in_the_notes(self, screen):
        plan = self._plan(screen, "attributed",
                          {"hopeless": 0.0, "other": 0.0})

        assert any("CANNOT BE ATTRIBUTED" in note for note in plan.notes), (
            f"the caption has to explain the empty montage; got {plan.notes}")

    def test_the_rank_picker_does_not_pre_flight_an_attribution_it_never_makes(
            self, screen):
        plan = self._plan(screen, "rank", {"hopeless": 0.0, "other": 0.0})

        assert not any("CANNOT BE ATTRIBUTED" in note for note in plan.notes)

    def test_a_pre_flight_is_never_the_reason_a_montage_does_not_draw(
            self, screen):
        """A courtesy, not a precondition."""
        plan = self._plan(screen, "attributed", {"hopeless": 3.0})

        assert plan is not None
