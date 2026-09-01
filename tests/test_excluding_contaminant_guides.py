"""Excluded guides leave the count table before anything reads it.

Asked 2026-08-21: "we already have exclude guides in the other category,
this should be moved to controls and filters, and make sure it is removed
first. right?" and "in the exclude i should be able to put in several guides
or genes."

REMOVED FIRST IS THE WHOLE POINT. `select_montage` computes `well_totals` by
summing the fraction column over the FULL count table, and `normalised_share`
divides by that sum. A contaminant filtered further downstream would still be
sitting in that denominator, holding every real guide's share down by its
own -- on one real plate, a fifth of all the reads.

So there is exactly ONE exclusion point and it is above every path: the
ranking, the per-well fractions, the posteriors and the totals all read a
table the contaminant has already left.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.cell_montage import select_montage


@pytest.fixture
def screen():
    """One well: a 50% contaminant, the guide of interest at 30%, one more."""
    rng = np.random.default_rng(0)
    n = 60
    objects = pd.DataFrame({
        "prc": ["p1_w1"] * n,
        "prcfo": [f"p1_w1_f1_o{i}" for i in range(n)],
        "pred": rng.random(n)})
    counts = pd.DataFrame({
        "prc": ["p1_w1"] * 3,
        "grna": ["TGGT1_220950_1", "TGGT1_233460_4", "TGGT1_500000_1"],
        "gene": ["TGGT1_220950", "TGGT1_233460", "TGGT1_500000"],
        "fraction": [0.5, 0.3, 0.2]})
    return objects, counts


def _cells(plan):
    return sum(w.n_expected for w in plan.wells) if plan.wells else 0


def _note(plan):
    return " | ".join(n for n in (plan.notes or []) if "exclud" in n.lower())


class TestRemovedFirst:

    def test_excluding_a_contaminant_raises_the_real_guides_share(self, screen):
        """0.3 of the well becomes 0.3/0.5 once the 50% contaminant is gone,
        so twice as many cells are expected. This is the suppression the
        ordering exists to prevent."""
        objects, counts = screen
        plain = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                               score_column="pred")
        cleaned = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                                 score_column="pred",
                                 exclude_grnas=["TGGT1_220950_1"])

        assert _cells(cleaned) == pytest.approx(2 * _cells(plain), rel=0.15)

    def test_the_well_total_no_longer_counts_it(self, screen):
        """The specific denominator: excluding both other guides leaves the
        guide of interest as the whole well."""
        objects, counts = screen
        plan = select_montage(
            objects, counts, "TGGT1_233460_4", 1.0, score_column="pred",
            exclude_grnas=["TGGT1_220950_1", "TGGT1_500000_1"])
        assert _cells(plan) == len(objects)

    def test_it_says_what_it_removed(self, screen):
        objects, counts = screen
        plan = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                              score_column="pred",
                              exclude_grnas=["TGGT1_220950_1"])
        assert "excluded 1 guide" in _note(plan)


class TestSeveralGuidesOrGenes:
    """"in the exclude i should be able to put in several guides or genes"."""

    @pytest.mark.parametrize("typed", [
        "TGGT1_220950_1",     # the guide
        "TGGT1_220950",       # its gene
        "220950",             # the gene without the organism prefix
    ])
    def test_every_spelling_reaches_the_same_guide(self, screen, typed):
        objects, counts = screen
        plan = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                              score_column="pred", exclude_grnas=[typed])
        assert "excluded 1 guide" in _note(plan), typed

    def test_several_entries_at_once(self, screen):
        objects, counts = screen
        plan = select_montage(
            objects, counts, "TGGT1_233460_4", 1.0, score_column="pred",
            exclude_grnas=["TGGT1_220950", "TGGT1_500000"])
        assert "excluded 2 guide" in _note(plan)

    def test_a_gene_takes_all_of_its_guides(self):
        """So excluding a contaminated amplicon does not mean listing its
        four guides and missing the fifth."""
        from spacr.read_background import resolve_exclusions

        guides = ["TGGT1_220950_1", "TGGT1_220950_2", "TGGT1_220950_3",
                  "TGGT1_233460_4"]
        assert len(resolve_exclusions(["TGGT1_220950"], guides)) == 3


class TestRegressionExcludesBeforeReadFractions:
    """The regression path must use the same first-step exclusion contract."""

    @staticmethod
    def _counts():
        return pd.DataFrame({
            "plateID": ["p1"] * 3,
            "rowID": ["r1"] * 3,
            "columnID": ["c1"] * 3,
            "grna": ["TGGT1_220950_1", "TGGT1_233460_4",
                     "TGGT1_500000_1"],
            "gene": ["TGGT1_220950", "TGGT1_233460", "TGGT1_500000"],
            "count": [50, 30, 20],
        })

    def test_tggt1_220950_1_leaves_before_the_denominator(self):
        """30/100 becomes 30/(30+20), rather than staying at 0.30."""
        from spacr.ml import process_reads

        record = {}
        result = process_reads(
            self._counts(), fraction_threshold=None, plate=None,
            exclude_grnas=["TGGT1_220950_1"], record=record)

        fractions = result.set_index("grna")["fraction"].to_dict()
        assert "220950_1" not in fractions
        assert fractions["233460_4"] == pytest.approx(0.6)
        assert fractions["500000_1"] == pytest.approx(0.4)
        assert sum(fractions.values()) == pytest.approx(1.0)
        assert record == {
            "exclude_grnas": 1,
            "exclude_grnas_of": 3,
            "exclude_grnas_guides": ["TGGT1_220950_1"],
            "exclude_grnas_unmatched": [],
        }

    @pytest.mark.parametrize("typed", ["TGGT1_220950", "220950"])
    def test_a_gene_name_removes_all_of_its_guides(self, typed):
        from spacr.ml import process_reads

        counts = pd.DataFrame({
            "plateID": ["p1"] * 3,
            "rowID": ["r1"] * 3,
            "columnID": ["c1"] * 3,
            "grna": ["TGGT1_220950_1", "TGGT1_220950_2",
                     "TGGT1_233460_4"],
            "gene": ["TGGT1_220950", "TGGT1_220950", "TGGT1_233460"],
            "count": [25, 25, 50],
        })
        record = {}

        result = process_reads(
            counts, fraction_threshold=None, plate=None,
            exclude_grnas=[typed], record=record)

        assert result[["grna", "fraction"]].to_dict("records") == [
            {"grna": "233460_4", "fraction": 1.0}]
        assert record["exclude_grnas"] == 2
        assert record["exclude_grnas_guides"] == [
            "TGGT1_220950_1", "TGGT1_220950_2"]

    def test_unmatched_entries_are_persisted_alongside_matches(self):
        from spacr.ml import process_reads

        record = {}
        process_reads(
            self._counts(), fraction_threshold=None, plate=None,
            exclude_grnas=["TGGT1_220950_1", "TGGT1_NOT_HERE"],
            record=record)

        assert record["exclude_grnas"] == 1
        assert record["exclude_grnas_unmatched"] == ["TGGT1_NOT_HERE"]


class TestAMisspelledExclusionIsLoud:
    """A misspelled exclusion excludes nothing and looks like it worked,
    which is how a known contaminant survives the filter meant to remove
    it."""

    def test_it_is_named_in_the_notes(self, screen):
        objects, counts = screen
        plan = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                              score_column="pred",
                              exclude_grnas=["TGGT1_NOT_HERE"])
        said = _note(plan)
        assert "match nothing" in said
        assert "TGGT1_NOT_HERE" in said

    def test_and_nothing_is_removed(self, screen):
        objects, counts = screen
        plain = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                               score_column="pred")
        missed = select_montage(objects, counts, "TGGT1_233460_4", 1.0,
                                score_column="pred",
                                exclude_grnas=["TGGT1_NOT_HERE"])
        assert _cells(missed) == _cells(plain)

    def test_unmatched_is_reported_alongside_what_did_match(self, screen):
        objects, counts = screen
        plan = select_montage(
            objects, counts, "TGGT1_233460_4", 1.0, score_column="pred",
            exclude_grnas=["TGGT1_220950_1", "TGGT1_NOT_HERE"])
        said = _note(plan)
        assert "excluded 1 guide" in said
        assert "match nothing" in said


class TestItIsAControlSetting:

    def test_it_sits_with_the_controls_and_not_in_other(self):
        """An uncategorised setting falls into the trailing 'Other' section,
        which is where this one started."""
        import spacr.settings as settings

        holders = [name for name, keys in settings.categories.items()
                   if "exclude_grnas" in keys]
        assert holders == ["Plate Layout & Controls"]

    def test_it_is_typed_and_documented(self):
        import spacr.settings as settings

        assert "exclude_grnas" in settings.expected_types
        assert "exclude_grnas" in settings.tooltips


class TestTheGeneColumnBranchThatWasMissed:
    """Found while excluding by gene name, and it is not about exclusion.

    `matches` strips the measured organism prefix off what the user typed.
    The guide branch and the no-gene-column branch both compare against the
    stripped AND unstripped spelling; the branch between them -- the one a
    screen WITH a gene column takes -- compared only against the stripped
    one, so `TGGT1_220950` against a gene column spelling it `TGGT1_220950`
    matched nothing.

    `controls`, `positive_control` and `negative_control` all arrive through
    `rows_for`, so a gene-level control on such a screen selected zero rows
    in silence.
    """

    @pytest.fixture
    def library(self):
        guides = pd.Series(["TGGT1_220950_1", "TGGT1_220950_2",
                            "TGGT1_233460_4"])
        genes = pd.Series(["TGGT1_220950", "TGGT1_220950", "TGGT1_233460"])
        return guides, genes

    @pytest.mark.parametrize("typed", ["TGGT1_220950", "220950"])
    def test_a_prefixed_gene_column_matches(self, library, typed):
        from spacr.control_names import rows_for

        guides, genes = library
        mask, _note = rows_for(typed, guides, genes, names=list(guides))
        assert int(mask.sum()) == 2, typed

    @pytest.mark.parametrize("typed", ["TGGT1_220950", "220950"])
    def test_it_agrees_with_the_no_gene_column_path(self, library, typed):
        """The two paths must reach the same rows, or which columns a screen
        happens to carry changes which cells are controls."""
        from spacr.control_names import rows_for

        guides, genes = library
        with_genes, _a = rows_for(typed, guides, genes, names=list(guides))
        without, _b = rows_for(typed, guides, None, names=list(guides))
        assert list(with_genes) == list(without), typed

    def test_an_unprefixed_gene_column_still_matches(self):
        """The fix must not break the spelling that already worked."""
        from spacr.control_names import rows_for

        guides = pd.Series(["TGGT1_220950_1", "TGGT1_233460_4"])
        genes = pd.Series(["220950", "233460"])
        mask, _note = rows_for("TGGT1_220950", guides, genes,
                               names=list(guides))
        assert int(mask.sum()) == 1
