""""Show" offers the three populations (instruction 205).

"i whould be able to gRNAs, gRNAs + other datapoints in selected wells, and
All datapoints. presently i cannot pick gRNAs + other datapoints in selected
wells."

THE MIDDLE ONE IS THE COMPARISON THAT MATTERS. A guide's cells against every
cell in the screen compares two different experiments; a guide's cells
against their own well-mates compares two populations that shared a plate, a
day, a stain and an imaging session.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.well_scope import (CHOSEN, MATE, MATE_COLUMN, SCOPES, describe,
                              select, wells_of)


@pytest.fixture
def objects():
    """Two wells hold g1; a third holds nobody's chosen guide."""
    return pd.DataFrame({
        "prc": ["p1_r1_c1"] * 3 + ["p1_r1_c2"] * 3 + ["p1_r1_c3"] * 2,
        "grna": ["g1", "x", "y", "g1", "z", "w", "q", "r"],
        "area": range(8),
    })


class TestAllThreeAreOffered:

    def test_there_are_exactly_three(self):
        assert len(SCOPES) == 3

    def test_they_are_the_three_asked_for(self):
        assert [v for v, _ in SCOPES] == ["guides", "wells", "all"]

    def test_each_draws_a_different_population(self, objects):
        sizes = {scope: len(select(objects, scope=scope, guides=["g1"])[0])
                 for scope, _ in SCOPES}
        assert len(set(sizes.values())) == 3, sizes

    def test_an_unknown_scope_raises(self, objects):
        """Falling back to 'all' would draw a different population from the
        one asked for and say nothing."""
        with pytest.raises(ValueError):
            select(objects, scope="everything", guides=["g1"])


class TestTheMiddleOne:

    def test_it_adds_the_well_mates(self, objects):
        out, report = select(objects, scope="wells", guides=["g1"])
        assert report["chosen"] == 2
        assert report["mates"] == 4
        assert len(out) == 6

    def test_it_leaves_out_wells_the_guide_is_not_in(self, objects):
        out, _ = select(objects, scope="wells", guides=["g1"])
        assert "p1_r1_c3" not in set(out["prc"])

    def test_the_two_sides_are_distinguishable(self, objects):
        """Drawing them in one colour would answer the question by hiding
        it."""
        out, _ = select(objects, scope="wells", guides=["g1"])
        assert MATE_COLUMN in out.columns
        assert set(out[MATE_COLUMN]) == {CHOSEN, MATE}

    def test_the_guides_only_scope_has_no_mates(self, objects):
        out, report = select(objects, scope="guides", guides=["g1"])
        assert len(out) == 2 and report["mates"] == 0

    def test_all_is_the_whole_table(self, objects):
        out, _ = select(objects, scope="all", guides=["g1"])
        assert len(out) == len(objects)


class TestTheWellSetIsDerivedThenEditable:

    def test_it_is_derived_from_the_guides(self, objects):
        assert wells_of(objects, ["g1"]) == ["p1_r1_c1", "p1_r1_c2"]

    def test_different_guides_give_different_wells(self, objects):
        assert wells_of(objects, ["q"]) == ["p1_r1_c3"]

    def test_narrowing_it_narrows_the_plot(self, objects):
        out, report = select(objects, scope="wells", guides=["g1"],
                             wells=["p1_r1_c1"])
        assert len(out) == 3
        assert report["wells"] == ["p1_r1_c1"]

    def test_narrowing_to_nothing_draws_nothing(self, objects):
        out, _ = select(objects, scope="wells", guides=["g1"], wells=[])
        assert len(out) == 0

    def test_the_order_is_the_tables_not_sorted(self, objects):
        """So a user narrowing the set sees them in the order the screen was
        laid out."""
        shuffled = objects.iloc[::-1]
        assert wells_of(shuffled, ["g1"]) == ["p1_r1_c2", "p1_r1_c1"]


class TestNoSelectionIsAnAnswer:

    def test_no_guide_draws_nothing_and_says_why(self, objects):
        """Drawing the whole table instead would look like a selection
        nobody made."""
        out, report = select(objects, scope="guides", guides=[])
        assert len(out) == 0
        assert "no gRNA was chosen" in report["note"]

    def test_all_still_draws_everything_with_no_selection(self, objects):
        out, _ = select(objects, scope="all", guides=[])
        assert len(out) == len(objects)

    def test_a_table_with_no_well_column_says_so(self):
        frame = pd.DataFrame({"grna": ["g1", "x"], "area": [1, 2]})
        out, report = select(frame, scope="wells", guides=["g1"])
        assert "names no well" in report["note"]
        assert len(out) == 1


class TestTheSentence:

    def test_it_names_the_comparison(self, objects):
        _, report = select(objects, scope="wells", guides=["g1"])
        text = describe(report)
        assert "well-mates" in text and "imaging session" in text

    def test_it_counts_both_sides(self, objects):
        _, report = select(objects, scope="wells", guides=["g1"])
        text = describe(report)
        assert "2 objects" in text and "4 of their well-mates" in text


class TestThroughThePanel:

    @pytest.fixture
    def panel(self, objects):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        QApplication.instance() or QApplication([])
        frame = objects.copy()
        frame["prcfo"] = [f"{p}_f1_o{i}" for i, p in
                          enumerate(frame["prc"])]
        return MeasurementComparePanel(frame, {"a": ["a"]})

    def test_the_box_offers_all_three(self, panel):
        assert [panel.scope.itemData(i)
                for i in range(panel.scope.count())] == \
            ["guides", "wells", "all"]

    def test_the_volcano_selection_sets_the_wells(self, panel):
        panel.set_selected_guides(["g1"])
        assert panel.selected_wells() == ["p1_r1_c1", "p1_r1_c2"]

    def test_a_new_selection_re_derives_the_wells(self, panel):
        """`None` for the wells means not narrowed yet, which is different
        from narrowed to nothing -- and the difference is what lets a new
        guide selection re-derive rather than stay empty."""
        panel.set_selected_guides(["g1"])
        panel.set_selected_wells(["p1_r1_c1"])
        panel.set_selected_guides(["q"])
        assert panel.selected_wells() == ["p1_r1_c3"]

    def test_narrowing_reaches_the_plot(self, panel):
        panel.set_selected_guides(["g1"])
        panel.scope.setCurrentIndex(panel.scope.findData("wells"))
        before = len(panel.scoped_objects()[0])
        panel.set_selected_wells(["p1_r1_c1"])
        assert len(panel.scoped_objects()[0]) < before
