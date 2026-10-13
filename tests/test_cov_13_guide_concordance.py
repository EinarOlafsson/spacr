"""Single-guide hits must be findable, and an unfittable table must not crash.

A gene carried by one guide has a gene-level p-value identical to that
guide's, which is not independent evidence. ``flag_single_guide_hits`` is how
a reader separates those from genes whose guides agree, so it has to work on
the degenerate tables a real run produces -- a fit with no guide terms at all,
a table with no ``feature`` column -- and return an empty answer rather than
raising in the middle of a report.
"""
from __future__ import annotations

import pandas as pd

from spacr import guide_concordance


def _coefficients():
    """A two-gene table: 225160 has three agreeing guides, 244480 has one."""
    return pd.DataFrame({
        "feature": [
            "fraction:grna[TGGT1_225160_1]",
            "fraction:grna[TGGT1_225160_2]",
            "fraction:grna[TGGT1_225160_3]",
            "gene_fraction:gene[T.225160]",
            "fraction:grna[TGGT1_244480_2]",
            "gene_fraction:gene[T.244480]",
        ],
        "coefficient": [0.4, 0.5, 0.6, 0.5, -0.9, -0.9],
        "p_value": [0.51, 0.14, 0.27, 4.6e-08, 1.6e-12, 1.6e-12],
    })


def test_a_gene_carried_by_one_guide_is_flagged_and_a_concordant_one_is_not():
    """The one-guide gene must be the only row returned.

    Both genes clear alpha at the gene level, and the single-guide one clears
    it by more. Ranking alone therefore puts the weaker claim on top, which is
    precisely the confusion this flag exists to undo.
    """
    flagged = guide_concordance.flag_single_guide_hits(_coefficients(), alpha=0.05)

    assert list(flagged.index) == ["244480"]
    assert bool(flagged["single_guide"].iloc[0]) is True
    assert int(flagged["n_guides"].iloc[0]) == 1


def test_a_single_guide_gene_that_misses_alpha_is_not_a_hit():
    """The flag lists single-guide HITS, not every single-guide gene.

    A gene nobody would have looked at is not a hit list problem, and putting
    it in the flagged table would bury the ones that are.
    """
    frame = _coefficients()
    frame.loc[frame["feature"] == "gene_fraction:gene[T.244480]", "p_value"] = 0.4

    flagged = guide_concordance.flag_single_guide_hits(frame, alpha=0.05)

    assert flagged.empty


def test_flagging_a_fit_with_no_guide_terms_returns_an_empty_table():
    """A gene-only fit has nothing to say about guides, and must say so quietly.

    ``guide_support`` returns its empty, fully-columned frame here; the flag
    has to pass that straight back instead of indexing columns that a real
    result would have.
    """
    gene_only = pd.DataFrame({
        "feature": ["gene_fraction:gene[T.225160]"],
        "coefficient": [0.5],
        "p_value": [1e-6],
    })

    flagged = guide_concordance.flag_single_guide_hits(gene_only)

    assert flagged.empty
    assert "single_guide" in flagged.columns
    assert flagged.index.name == "gene"


def test_flagging_a_table_without_a_feature_column_returns_an_empty_table():
    """A frame that is not a coefficient table must not raise mid-report."""
    flagged = guide_concordance.flag_single_guide_hits(
        pd.DataFrame({"value": [1.0, 2.0]}))

    assert flagged.empty


def test_annotating_a_fit_with_no_guides_returns_an_untouched_copy():
    """With no guide support there is nothing to join, and nothing to change.

    Returning the caller's own frame would let a later mutation of the
    annotated table rewrite the results the caller is about to save, so the
    no-support path still copies.
    """
    gene_only = pd.DataFrame({
        "feature": ["gene_fraction:gene[T.225160]"],
        "coefficient": [0.5],
        "p_value": [1e-6],
    })

    annotated = guide_concordance.annotate_results(gene_only)

    assert annotated is not gene_only
    pd.testing.assert_frame_equal(annotated, gene_only)
    assert "n_guides" not in annotated.columns


def test_annotating_nothing_returns_nothing_rather_than_raising():
    """``None`` reaches this from a fit that produced no coefficient table."""
    assert guide_concordance.annotate_results(None) is None


def test_annotating_a_real_fit_joins_guide_support_onto_every_row():
    """The columns a reader needs must land on the table they are reading."""
    annotated = guide_concordance.annotate_results(_coefficients())

    assert "_gene" not in annotated.columns
    assert len(annotated) == 6
    single = annotated[annotated["feature"] == "gene_fraction:gene[T.244480]"]
    assert int(single["n_guides"].iloc[0]) == 1
    assert bool(single["single_guide"].iloc[0]) is True
    concordant = annotated[annotated["feature"] == "gene_fraction:gene[T.225160]"]
    assert float(concordant["concordance"].iloc[0]) == 1.0
