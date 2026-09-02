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
                                 ControlNotFound,
                                 ControlSpec, common_prefix, matches,
                                 resolve_control, resolve_controls, rows_for)

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


class TestOnlyAThreePartNameCarriesASpeciesTag:
    """Caught by pointing the resolver at a one-guide library.

    `common_prefix(["000000_1"])` returned "000000" -- a token on 100% of the
    names -- so the GENE was read as the organism, the control resolved to
    gene "1", and it matched nothing. spaCR's own convention is
    <org>_<gene>_<guide> and `process_reads` splits on exactly three parts,
    so a two-part name's head is a gene and never a species.
    """

    def test_a_single_guide_library_has_no_prefix(self):
        assert common_prefix(["000000_1"]) == ""

    def test_and_that_guide_still_resolves_as_a_guide(self):
        spec = resolve_control("000000_1", names=["000000_1"])

        assert (spec.level, spec.value) == (GUIDE, "000000_1")

    def test_one_gene_two_guides_is_not_a_prefix_either(self):
        assert common_prefix(["000000_1", "000000_2"]) == ""

    def test_three_part_names_still_give_up_their_organism(self):
        assert common_prefix(["TGGT1_000000_1", "TGGT1_233460_2"]) == "TGGT1"


class TestOneMatcherForEveryCallSite:
    """184 C. There were two and they disagreed."""

    def test_the_labeller_no_longer_matches_a_substring(self):
        """`nc='23346'` claiming `233460` is the defect this replaces."""
        from spacr.ml import label_control_condition

        features = pd.Series(["fraction:grna[233460_1]",
                              "fraction:grna[2334600_1]"])
        guides = pd.Series(["233460_1", "2334600_1"])

        labels = label_control_condition(features, guides, nc="23346")

        assert list(labels) == ["other", "other"]

    def test_a_gene_control_takes_every_guide_of_that_gene(self):
        from spacr.ml import label_control_condition

        guides = pd.Series(["233460_1", "233460_2", "999999_1"])
        features = pd.Series([f"fraction:grna[{g}]" for g in guides])

        labels = label_control_condition(features, guides, nc="233460")

        assert list(labels) == ["nc", "nc", "other"]

    def test_the_level_matcher_reads_a_prefixed_control(self):
        """`name.split('_')[0]` read TGGT1 as the gene, so a control pasted
        from a library file selected nothing at gene level -- and the gene
        table silently got no effect-size cut."""
        from spacr.ml import _level_control_rows

        frame = pd.DataFrame({
            "grna": [None, None],
            "gene": ["000000", "233460"],
            "coefficient": [0.1, 0.2]})

        rows = _level_control_rows(frame, "gene", ["TGGT1_000000_1"])

        assert len(rows) == 1
        assert rows["gene"].iloc[0] == "000000"

    def test_the_guide_level_matcher_still_matches_whole_guides(self):
        from spacr.ml import _level_control_rows

        frame = pd.DataFrame({
            "grna": ["000000_1", "000000_2"],
            "gene": ["000000", "000000"],
            "coefficient": [0.1, 0.2]})

        rows = _level_control_rows(frame, "grna", ["000000_1"])

        assert len(rows) == 1 and rows["grna"].iloc[0] == "000000_1"

    def test_no_controls_selects_nothing_rather_than_raising(self):
        from spacr.ml import _level_control_rows

        frame = pd.DataFrame({"grna": ["a"], "gene": ["a"],
                              "coefficient": [0.1]})

        assert len(_level_control_rows(frame, "grna", None)) == 0


class TestItSaysWhenAControlMatchedNothing:
    """184 D. Not silence -- but not an exception either.

    spaCR SHIPS nc='233460' and pc='220950', Toxoplasma gene ids. Raising
    would make every screen that is not this one fail on a value the user
    never typed, so the fix for silence is a sentence nobody can miss.
    `strict=True` stays available for a caller that KNOWS the control was
    chosen rather than defaulted.
    """

    def test_no_control_selects_no_rows_and_says_nothing(self):
        guides = pd.Series(["000000_1", "233460_1"], index=[4, 9])

        mask, note = rows_for(None, guides)

        assert mask.index.tolist() == [4, 9]
        assert mask.tolist() == [False, False]
        assert note == ""

    def test_a_control_that_matches_nothing_is_named(self, capsys):
        from spacr.ml import label_control_condition

        guides = pd.Series(["a_1", "b_1"])
        features = pd.Series([f"fraction:grna[{g}]" for g in guides])
        labels = label_control_condition(features, guides, nc="nowhere")
        from spacr.ml import _say_when_a_control_matched_nothing

        _say_when_a_control_matched_nothing(
            pd.DataFrame({"condition": labels}), "nowhere", None, None)

        said = capsys.readouterr().out
        assert "negative_control" in said and "nowhere" in said
        assert "nothing to measure" in said, "name the consequence"
        assert "GENE" in said and "GUIDE" in said, "and how it was read"

    def test_a_control_that_matched_says_nothing(self, capsys):
        from spacr.ml import _say_when_a_control_matched_nothing

        _say_when_a_control_matched_nothing(
            pd.DataFrame({"condition": ["nc", "other"]}), "a", None, None)

        assert "WARNING" not in capsys.readouterr().out

    def test_an_empty_control_list_is_not_warned_about(self, capsys):
        from spacr.ml import _say_when_a_control_matched_nothing

        _say_when_a_control_matched_nothing(
            pd.DataFrame({"condition": ["other"]}), None, None, None)

        assert "WARNING" not in capsys.readouterr().out

    def test_strict_is_still_available_for_a_caller_that_knows(self):
        from spacr.control_names import ControlNotFound, rows_for

        with pytest.raises(ControlNotFound, match="matches nothing"):
            rows_for("nowhere", pd.Series(["a_1"]), strict=True)


class TestTheDataHasAlreadyDroppedThePrefix:
    """`process_reads` stores TGGT1_000000_11 as guide 000000_11.

    So the names in hand carry no organism token, `common_prefix` measures ""
    from them -- correctly -- and a control pasted from the LIBRARY file
    still says TGGT1_000000, whose head is then not a known prefix. Measured
    on the maintainer's screen: 'TGGT1_000000' found 0 where '000000' found
    28, the same control written the way the library writes it.
    """

    #: What spaCR actually holds after `process_reads`.
    HELD = ["000000_1", "000000_2", "233460_1", "233460_2"]

    def test_a_library_spelling_still_finds_the_gene(self):
        mask, note = rows_for("TGGT1_000000", pd.Series(self.HELD),
                              pd.Series([g.split("_")[0] for g in self.HELD]))

        assert int(mask.sum()) == 2
        assert "TGGT1" in note, "say the prefix was dropped"

    def test_the_bare_gene_and_the_library_spelling_agree(self):
        genes = pd.Series([g.split("_")[0] for g in self.HELD])
        bare, _ = rows_for("000000", pd.Series(self.HELD), genes)
        full, _ = rows_for("TGGT1_000000", pd.Series(self.HELD), genes)

        assert list(bare) == list(full)

    def test_it_does_not_rescue_a_control_that_is_simply_wrong(self):
        mask, _note = rows_for("TGGT1_999999", pd.Series(self.HELD),
                               pd.Series([g.split("_")[0] for g in self.HELD]))

        assert int(mask.sum()) == 0

    def test_it_is_not_substring_matching_by_another_route(self):
        """Dropping a leading token still matches WHOLE values."""
        held = ["233460_1", "2334600_1"]
        mask, _note = rows_for("TGGT1_23346", pd.Series(held),
                               pd.Series([g.split("_")[0] for g in held]))

        assert int(mask.sum()) == 0

    def test_a_head_the_library_does_use_is_left_alone(self):
        """Only an UNUSED leading token is dropped: if names really do start
        with it, the two-part reading was right."""
        held = ["TGGT1_000000", "TGGT1_233460"]
        mask, _note = rows_for("TGGT1_000000", pd.Series(held))

        assert int(mask.sum()) == 1

    def test_a_name_that_is_nothing_but_a_trailing_separator_matches_nothing(
            self):
        """What a spreadsheet paste leaves behind, and what it must not do.

        "sgCtrl_   " strips to "sgCtrl_", so the part after the separator is
        empty and there is nothing to retry with. The answer is zero rows and
        no exception: the name really does match nothing, and raising here
        would take down normalisation, every volcano baseline and every nc/pc
        reference over a stray character.
        """
        held = ["TGGT1_233460_1", "TGGT1_233460_2"]

        mask, _note = rows_for("sgCtrl_   ", pd.Series(held))

        assert int(mask.sum()) == 0

    def test_the_same_name_still_raises_under_strict(self):
        """And giving up quietly must not skip the raise a caller asked for.

        Silently zero controls is the failure this module exists to prevent,
        so a caller who said `strict` hears about it.
        """
        held = ["TGGT1_233460_1"]

        with pytest.raises(ControlNotFound):
            rows_for("sgCtrl_   ", pd.Series(held), strict=True)
