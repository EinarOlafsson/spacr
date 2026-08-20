"""Instruction 184: read a control the way the user wrote it.

"the controll finding mechanism should know if there is a string with a _
then it is a guide if there is only a string or a string starting with a
common string before the _ like TGGT1_000000 it is a gene."

WHY IT MATTERS MORE THAN IT LOOKS. A control that does not match is not an
error anyone sees -- it is silently zero controls, and every normalisation,
every volcano baseline and every nc/pc reference is then computed against
nothing. spaCR's own defaults are bare gene ids (nc='233460'); the names a
user pastes out of a library file are full guide names (TGGT1_233460_4).
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.control_names import (COMMON_PREFIX_SHARE, GENE, GUIDE,
                                 ControlSpec, common_prefix, matches,
                                 resolve_control, resolve_controls)

#: A library shaped like the maintainer's: one organism tag, three genes.
LIBRARY = [f"TGGT1_{gene}_{n}"
           for gene in ("000000", "233460", "220950")
           for n in range(1, 11)]

#: The same library as spaCR actually stores it, after process_reads has
#: split off the organism token.
STORED = [name.split("_", 1)[1] for name in LIBRARY]


class TestTheCommonPrefixIsMeasured:
    """Never a hard-coded list: the next screen is a different organism."""

    def test_it_finds_the_organism_tag(self):
        assert common_prefix(LIBRARY) == "TGGT1"

    def test_a_library_with_no_shared_tag_has_no_prefix(self):
        assert common_prefix(STORED) == "", (
            "'000000' and '233460' share nothing, and inventing a prefix "
            "would make every gene look like an organism tag")

    def test_a_minority_tag_is_not_common(self):
        mixed = LIBRARY[:3] + [f"PBANKA_{n}" for n in range(50)]

        assert common_prefix(mixed) != "TGGT1"

    def test_it_counts_distinct_names_not_rows(self):
        """One guide with a million reads must not speak for the library."""
        skewed = ["TGGT1_000000_1"] * 1000 + STORED

        assert common_prefix(skewed) == ""

    def test_an_empty_library_has_no_prefix(self):
        assert common_prefix([]) == ""
        assert common_prefix(["", None]) == ""

    def test_the_share_is_adjustable_and_documented(self):
        assert 0.5 < COMMON_PREFIX_SHARE <= 1.0


class TestTheFourCases:
    """The rule exactly as the maintainer stated it."""

    @pytest.fixture
    def prefix(self):
        return common_prefix(LIBRARY)

    def test_no_underscore_is_a_gene(self, prefix):
        spec = resolve_control("000000", prefix=prefix)

        assert (spec.level, spec.value) == (GENE, "000000")

    def test_one_underscore_with_a_common_head_is_a_gene(self, prefix):
        spec = resolve_control("TGGT1_000000", prefix=prefix)

        assert (spec.level, spec.value) == (GENE, "000000")

    def test_one_underscore_without_a_common_head_is_a_guide(self, prefix):
        """000000 is a gene id, not an organism, so the whole thing is one
        guide."""
        spec = resolve_control("000000_1", prefix=prefix)

        assert (spec.level, spec.value) == (GUIDE, "000000_1")

    def test_two_underscores_take_everything_after_the_first(self, prefix):
        spec = resolve_control("TGGT1_000000_1", prefix=prefix)

        assert (spec.level, spec.value) == (GUIDE, "000000_1")
        assert spec.value in STORED, (
            "it has to match the form process_reads actually stores")

    def test_without_a_measured_prefix_a_two_part_name_is_a_guide(self):
        """The conservative reading: with nothing known about the library,
        'TGGT1_000000' is a name with an underscore in it."""
        spec = resolve_control("TGGT1_000000", prefix="")

        assert (spec.level, spec.value) == (GUIDE, "TGGT1_000000")

    def test_it_measures_the_prefix_itself_when_given_the_names(self):
        spec = resolve_control("TGGT1_000000", names=LIBRARY)

        assert (spec.level, spec.value) == (GENE, "000000")


class TestNoControlIsALegalAnswer:
    """A control-free screen must not gain a control named ''."""

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_blank_resolves_to_nothing(self, value):
        assert resolve_control(value, names=LIBRARY) is None

    def test_a_blank_matches_no_row(self):
        assert not matches(None, STORED).any()

    def test_a_list_drops_its_blanks(self):
        specs = resolve_controls(["000000", "", None, "233460_2"],
                                 names=LIBRARY)

        assert [s.value for s in specs] == ["000000", "233460_2"]


class TestMixedFormsInOneList:
    """"a user may give a gene and two guides together"."""

    def test_each_entry_is_read_on_its_own(self):
        specs = resolve_controls(
            ["000000", "TGGT1_233460_4", "220950_1"], names=LIBRARY)

        assert [(s.level, s.value) for s in specs] == [
            (GENE, "000000"), (GUIDE, "233460_4"), (GUIDE, "220950_1")]


class TestMatching:

    def test_a_gene_takes_every_guide_of_that_gene(self):
        spec = resolve_control("000000", names=LIBRARY)
        mask = matches(spec, STORED)

        assert mask.sum() == 10
        assert all(g.startswith("000000_") for g in pd.Series(STORED)[mask])

    def test_a_guide_takes_exactly_one(self):
        spec = resolve_control("000000_1", names=LIBRARY)

        assert matches(spec, STORED).sum() == 1

    def test_a_gene_uses_the_gene_column_when_there_is_one(self):
        genes = [g.split("_")[0] for g in STORED]
        spec = resolve_control("233460", names=LIBRARY)

        assert matches(spec, STORED, genes=genes).sum() == 10

    def test_it_never_matches_a_substring(self):
        """`nc='23346'` claiming `233460` is the failure this replaces."""
        spec = resolve_control("23346", names=LIBRARY)

        assert matches(spec, STORED).sum() == 0

    def test_a_longer_gene_id_is_not_claimed_by_a_shorter_one(self):
        stored = ["233460_1", "2334600_1"]
        spec = resolve_control("233460", names=LIBRARY)

        assert matches(spec, stored).sum() == 1


class TestItSaysWhatItMatched:
    """"A control that matches NOTHING must be an error, not a silent zero."""

    def test_the_note_names_the_level_and_the_value(self):
        spec = resolve_control("TGGT1_000000", names=LIBRARY)
        note = spec.note()

        assert "gene" in note and "000000" in note
        assert "TGGT1" in note, "say that the prefix was dropped"

    def test_the_note_carries_the_counts_when_it_has_them(self):
        spec = resolve_control("000000", names=LIBRARY)

        assert "30 guide(s)" in spec.note(30, 1412)
        assert "1412 well(s)" in spec.note(30, 1412)

    def test_a_spec_is_hashable_so_it_can_be_deduplicated(self):
        assert len({resolve_control("000000", names=LIBRARY),
                    resolve_control("000000", names=LIBRARY)}) == 1


class TestTheMaintainersFourSpellings:
    """All four reach the same 30 control guides on the reference screen."""

    def test_they_all_land_on_the_same_thirty(self):
        genes = [g.split("_")[0] for g in STORED]
        got = set()
        for typed in ("000000", "TGGT1_000000"):
            spec = resolve_control(typed, names=LIBRARY)
            got.add(tuple(pd.Series(STORED)[matches(spec, STORED, genes)]))

        assert len(got) == 1, "the two gene spellings must agree"
        assert len(next(iter(got))) == 10

    def test_the_two_guide_spellings_agree_too(self):
        one = resolve_control("000000_1", names=LIBRARY)
        two = resolve_control("TGGT1_000000_1", names=LIBRARY)

        assert one.value == two.value
        assert matches(one, STORED).equals(matches(two, STORED))


class TestItTakesTheContainersCallersActuallyHave:
    """Guide names live in a DataFrame column, so a Series must work.

    `names or ()` raised "The truth value of a Series is ambiguous" the first
    time this was pointed at a real library -- an error at the point of use,
    from the most natural possible call.
    """

    def test_a_series_of_names_is_accepted(self):
        spec = resolve_control("TGGT1_000000", names=pd.Series(LIBRARY))

        assert (spec.level, spec.value) == (GENE, "000000")

    def test_a_series_of_controls_is_accepted(self):
        specs = resolve_controls(pd.Series(["000000", "233460_2"]),
                                 names=pd.Series(LIBRARY))

        assert [s.value for s in specs] == ["000000", "233460_2"]

    def test_an_empty_series_is_no_controls(self):
        assert resolve_controls(pd.Series([], dtype=object),
                                names=LIBRARY) == ()

    def test_common_prefix_takes_a_series(self):
        assert common_prefix(pd.Series(LIBRARY)) == "TGGT1"
