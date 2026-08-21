"""187 B: the three comparisons that matter.

    "compare the annotated cells in the well, compare to the controlls
    (controlls would need to be specified.), compare to all other wells. and
    i whould be able to choose which wells to include from the gene
    annotation."

THE POINT OF THREE IS THAT THEY DISAGREE. If the same cells gave the same
answer under all three there would be no reason to offer a choice; the fixture
below is built so that they cannot -- a plate/well offset is added on top of
the gene effect, so "within the well" (which removes it) and "against every
other well" (which does not) see different differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (CONTRASTS, REST, build,
                                            contrast_note, control_wells,
                                            well_labels, wells_of)


def _objects() -> pd.DataFrame:
    """Two plates, six wells, ten cells each.

    A01/A02 hold the gene. A03/A04 are ordinary wells. B01/B02 are the
    controls. Every well carries an offset of its own, so a comparison that
    crosses wells is measuring the offset as well as the gene.
    """
    rows = []
    offsets = {"1_A_01": 0.0, "1_A_02": 5.0, "1_A_03": 20.0,
               "1_A_04": 25.0, "1_B_01": 40.0, "1_B_02": 45.0}
    for well, offset in offsets.items():
        plate, row, column = well.split("_")
        for cell in range(10):
            rows.append({
                "plateID": plate, "rowID": row, "columnID": column,
                "cell": f"{well}_{cell}",
                "area": offset + cell,       # the well offset dominates
                "text": "not a number",
            })
    frame = pd.DataFrame(rows)
    frame.index = pd.RangeIndex(len(frame))
    return frame


def _annotated(objects) -> dict:
    """The first three cells of each of the gene's two wells."""
    where = well_labels(objects)
    picked = []
    for well in ("1_A_01", "1_A_02"):
        picked.extend(objects.index[where == well][:3].tolist())
    return {"the gene": picked}


@pytest.fixture
def objects():
    return _objects()


@pytest.fixture
def groups(objects):
    return _annotated(objects)


class TestEachContrastHoldsADifferentSetOfCells:

    def test_within_the_well_keeps_only_that_well(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="within_well")

        where = made.frame["unit"].map(
            dict(zip(objects.index.astype(str), well_labels(objects))))
        assert set(where) == {"1_A_01", "1_A_02"}

    def test_against_every_other_well_keeps_no_annotated_well(
            self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="against_other_wells")

        where = made.frame["unit"].map(
            dict(zip(objects.index.astype(str), well_labels(objects))))
        rest = where[made.frame["group"].astype(str) == REST]
        assert set(rest) == {"1_A_03", "1_A_04", "1_B_01", "1_B_02"}

    def test_against_controls_keeps_only_the_control_wells(
            self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="against_controls",
                     controls=["1_B_01", "1_B_02"])

        where = made.frame["unit"].map(
            dict(zip(objects.index.astype(str), well_labels(objects))))
        rest = where[made.frame["group"].astype(str) == REST]
        assert set(rest) == {"1_B_01", "1_B_02"}

    def test_the_default_keeps_everything(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell")

        assert len(made.frame) == len(objects)

    def test_the_three_do_not_agree(self, objects, groups):
        """If they agreed there would be no reason to offer a choice."""
        def difference(contrast, **kwargs):
            made = build(objects, "area", groups=groups, level="cell",
                         contrast=contrast, **kwargs)
            by = made.frame.groupby("group")["value"].mean()
            return float(by["the gene"] - by[REST])

        within = difference("within_well")
        others = difference("against_other_wells")
        controls = difference("against_controls",
                              controls=["1_B_01", "1_B_02"])
        assert len({round(within, 6), round(others, 6),
                    round(controls, 6)}) == 3

    def test_the_annotated_cells_are_the_same_in_all_three(
            self, objects, groups):
        """Only the comparison group moves; the annotation does not."""
        counts = []
        for contrast, extra in (("within_well", {}),
                                ("against_other_wells", {}),
                                ("against_controls",
                                 {"controls": ["1_B_01"]})):
            made = build(objects, "area", groups=groups, level="cell",
                         contrast=contrast, **extra)
            counts.append(made.counts()["the gene"])
        assert counts == [6, 6, 6]


class TestTheContrastSaysWhatItRemoves:

    @pytest.mark.parametrize("contrast", [c for c, _l, _w in CONTRASTS if c])
    def test_every_contrast_has_a_reason(self, contrast):
        assert len(contrast_note(contrast)) > 40

    @pytest.mark.parametrize("contrast,extra", [
        ("within_well", {}),
        ("against_other_wells", {}),
        ("against_controls", {"controls": ["1_B_01"]}),
    ])
    def test_the_reason_reaches_the_comparison(self, objects, groups,
                                               contrast, extra):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast=contrast, **extra)

        assert contrast_note(contrast) in made.note

    def test_the_contrast_is_named_before_anything_else(self, objects,
                                                        groups):
        """It decides what the p-value is a p-value ABOUT, so it leads."""
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="within_well")

        assert made.note.startswith("within the well:")

    def test_an_unknown_contrast_is_refused(self, objects, groups):
        with pytest.raises(ValueError, match="within_well"):
            build(objects, "area", groups=groups, contrast="sideways")

    def test_an_unknown_contrast_note_is_empty_rather_than_raising(self):
        """A saved run from a future spaCR still draws."""
        assert contrast_note("sideways") == ""


class TestControlsHaveToBeNamed:

    def test_no_control_is_a_reason_not_a_crash(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="against_controls")

        assert not len(made.frame)
        assert "needs the controls named" in made.note

    def test_the_control_wells_come_from_the_count_data(self):
        counts = pd.DataFrame({
            "plateID": ["1", "1", "1"],
            "rowID": ["A", "B", "B"],
            "columnID": ["01", "01", "02"],
            "grna": ["000001_1", "000000_1", "000000_2"],
            "gene": ["000001", "000000", "000000"],
        })

        assert control_wells(counts, ["000000"]) == ("1_B_01", "1_B_02")

    def test_one_control_written_as_a_bare_string(self):
        """`resolve_controls` iterates its argument, so a bare string used to
        arrive as six separate one-character controls."""
        counts = pd.DataFrame({
            "plateID": ["1", "1"], "rowID": ["A", "B"],
            "columnID": ["01", "01"],
            "grna": ["000001_1", "000000_1"],
            "gene": ["000001", "000000"],
        })

        assert control_wells(counts, "000000") == ("1_B_01",)

    def test_a_control_named_as_a_guide_works_too(self):
        """184's whole point: gene or guide, either spelling."""
        counts = pd.DataFrame({
            "plateID": ["1", "1"], "rowID": ["A", "B"],
            "columnID": ["01", "02"],
            "grna": ["000000_1", "000000_2"],
            "gene": ["000000", "000000"],
        })

        assert control_wells(counts, ["000000_2"]) == ("1_B_02",)

    def test_a_control_that_matches_nothing_is_empty(self):
        counts = pd.DataFrame({
            "plateID": ["1"], "rowID": ["A"], "columnID": ["01"],
            "grna": ["000001_1"], "gene": ["000001"],
        })

        assert control_wells(counts, ["999999"]) == ()


class TestTheWellsAreChosen:

    def test_the_wells_of_a_gene_are_listed(self, objects, groups):
        assert wells_of(objects, groups) == {"the gene": ("1_A_01", "1_A_02")}

    def test_a_left_out_well_takes_its_annotated_cells_with_it(
            self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="within_well", wells=["1_A_01"])

        assert made.counts()["the gene"] == 3

    def test_a_left_out_well_is_not_promoted_into_the_comparison(
            self, objects, groups):
        """The rows the user threw out must not come back on the other side."""
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="against_other_wells", wells=["1_A_01"])

        where = made.frame["unit"].map(
            dict(zip(objects.index.astype(str), well_labels(objects))))
        assert "1_A_02" not in set(where)

    def test_leaving_a_well_out_is_said_out_loud(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="within_well", wells=["1_A_01"])

        assert "1 annotated well(s) left out: 1_A_02" in made.note

    def test_including_every_well_says_nothing(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     contrast="within_well", wells=["1_A_01", "1_A_02"])

        assert "left out" not in made.note


class TestTheWellKeyIsReadTheWayTheMontageWroteIt:

    def test_montage_well_wins(self, objects):
        objects = objects.assign(montage_well="whatever the montage said")

        assert set(well_labels(objects)) == {"whatever the montage said"}

    def test_prc_is_next(self, objects):
        objects = objects.assign(prc="1_A_01")

        assert set(well_labels(objects)) == {"1_A_01"}

    def test_no_well_column_is_none_not_one_big_well(self):
        """Both other spellings would silently make every row one well."""
        assert well_labels(pd.DataFrame({"area": [1.0, 2.0]})) is None

    def test_a_contrast_without_a_well_column_says_so(self):
        frame = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0]})

        made = build(frame, "area", groups={"g": [0, 1]}, level="cell",
                     contrast="within_well")

        assert not len(made.frame)
        assert "do not say which well" in made.note


class TestTheContrastSurvivesTheRestOfTheMachinery:

    def test_it_still_aggregates_to_wells(self, objects, groups):
        made = build(objects, "area", groups=groups, level="well",
                     contrast="within_well")

        # Two wells, each contributing an annotated row and a rest row.
        assert made.counts() == {"the gene": 2, REST: 2}

    def test_the_well_key_is_not_lost_when_rows_are_dropped(
            self, objects, groups):
        """Assigning the FULL column onto a shortened frame aligns on the
        index and leaves NaN, which joins every dropped-through row into one
        well called 'nan'."""
        made = build(objects, "area", groups=groups, level="well",
                     contrast="against_controls",
                     controls=["1_B_01", "1_B_02"])

        assert "nan" not in set(made.frame["unit"].astype(str))

    def test_it_composes_with_two_measurements(self, objects, groups):
        made = build(objects, "area", groups=groups, level="cell",
                     operator="/", second="area", contrast="within_well")

        assert made.measurement == "area / area"
        assert set(np.unique(made.frame["value"].dropna())) <= {1.0}
