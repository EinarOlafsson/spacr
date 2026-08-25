"""A hit list must drop unusable evidence rather than count it as support.

Every guard here is the same rule seen from a different angle: a value that
cannot be read is not a value that passed. A guide with no coefficient is not
a guide that agrees; a gene with a non-numeric q is not a gene that cleared
the FDR; a coefficient table with no ``feature`` column carries no guide-level
evidence at all. Letting any of those through inflates the corroboration count
beside a gene, which is the number a reader uses to decide whether to believe
it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr.hits import (Hit, HitList, build_hit_list, family_labels, gene_of,
                        grna_agreement)


def test_labelling_no_terms_at_all_returns_an_empty_label_array():
    """A restyle of a plot with nothing on it must not raise.

    ``family_labels`` runs on every interactive restyle, including the one
    that happens before a fit has produced any terms.
    """
    labels = family_labels([])

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (0,)
    assert labels.dtype == object


def test_an_empty_bracket_names_no_gene():
    """``gene[]`` is a malformed term, not a gene called the empty string.

    An empty gene id would become a hit-list row keyed on ``""``, which then
    collides with every other malformed term in the same table.
    """
    assert gene_of("gene_fraction:gene[]") is None
    assert gene_of("gene_fraction:gene[T.]") is None
    assert gene_of("gene_fraction:gene[233460]") == "233460"


def test_a_guide_table_with_no_feature_column_reports_no_guide_evidence():
    """A frame that is not a coefficient table gives every gene zero guides.

    The per-gRNA table is read from disk and a run can write something else
    under that name. Reporting agreement of 0/0 says "not measured", which is
    what the flag machinery downstream expects; raising would take the whole
    hit list down over a diagnostic column.
    """
    frame = pd.DataFrame({"coefficient": [0.4, -0.2]})

    agreement = grna_agreement({"233460": 0.5}, frame)

    assert agreement == {"233460": (0, 0, [])}


def test_a_guide_row_naming_no_gene_is_skipped_entirely():
    """An intercept row is in the table and belongs to no gene.

    Counting it against some gene would add a phantom guide to that gene's
    denominator and lower its agreement for free.
    """
    frame = pd.DataFrame({
        "feature": ["Intercept", "fraction:grna[233460_1]"],
        "grna": ["Intercept", "233460_1"],
        "coefficient": [1.2, 0.4],
    })

    agreement = grna_agreement({"233460": 0.5}, frame)

    assert agreement == {"233460": (1, 1, ["233460_1"])}


def test_a_guide_with_no_name_column_is_named_from_its_own_term():
    """The guide id has to come from somewhere, and the term always has it.

    Some backends write no ``grna`` column, and others write NaN into it for
    rows they did not resolve. The agreeing-guide list is what a reader checks
    against the library, so an unnamed guide is worse than a missing row.
    """
    frame = pd.DataFrame({
        "feature": ["fraction:grna[T.233460_1]", "fraction:grna[233460_2]"],
        "grna": [None, float("nan")],
        "coefficient": [0.4, 0.9],
    })

    n_agree, n_guides, names = grna_agreement({"233460": 0.5}, frame)["233460"]

    assert (n_agree, n_guides) == (2, 2)
    assert names == ["233460_1", "233460_2"]


def test_a_guide_whose_coefficient_cannot_be_read_is_not_a_guide():
    """Text or NaN in the coefficient column is absence, not agreement.

    Both must leave the denominator alone: a gene with one real guide and one
    unreadable row is a gene with ONE guide, and reporting 1/2 would make a
    corroborated gene look uncorroborated -- or, with the sign the other way,
    the reverse.
    """
    frame = pd.DataFrame({
        "feature": ["fraction:grna[233460_1]", "fraction:grna[233460_2]",
                    "fraction:grna[233460_3]", "fraction:grna[233460_4]"],
        "grna": ["233460_1", "233460_2", "233460_3", "233460_4"],
        "coefficient": ["not a number", None, float("nan"), 0.4],
    })

    agreement = grna_agreement({"233460": 0.5}, frame)

    assert agreement == {"233460": (1, 1, ["233460_4"])}


def test_an_infinite_coefficient_is_dropped_like_a_missing_one():
    """A separated fit can report an infinite coefficient.

    Its sign is arithmetically well defined, so nothing would raise -- it
    would simply be counted as the most emphatic agreement in the table.
    """
    frame = pd.DataFrame({
        "feature": ["fraction:grna[233460_1]", "fraction:grna[233460_2]"],
        "grna": ["233460_1", "233460_2"],
        "coefficient": [math.inf, 0.4],
    })

    agreement = grna_agreement({"233460": 0.5}, frame)

    assert agreement == {"233460": (1, 1, ["233460_2"])}


def _hits():
    return (
        Hit(gene="233460", effect=0.9, p_value=0.001, q_value=0.01,
            n_guides=3, n_agree=3, agreement=1.0, direction="up", rank=1),
        Hit(gene="244480", effect=-0.5, p_value=0.2, q_value=0.4,
            n_guides=1, n_agree=1, agreement=1.0, direction="down", rank=2),
    )


def test_filtering_on_p_keeps_only_the_rows_that_clear_it():
    """``max_p`` is a filter in its own right, not a synonym for ``max_q``.

    A reader who asks for p <= 0.01 after seeing a q-ranked list is asking a
    different question, and an unfiltered answer would look like agreement
    between the two.
    """
    listing = HitList(hits=_hits())

    kept = listing.filter(max_p=0.01)

    assert [hit.gene for hit in kept] == ["233460"]
    assert kept.filters["max_p"] == 0.01


def test_a_row_whose_numbers_cannot_be_read_fails_every_numeric_filter():
    """A missing value has not been shown to clear a threshold.

    This is the whole reason the filters convert rather than compare: a
    q-value of ``"NA"`` sorts and compares against a float in ways that differ
    by type, and any of those answers except "excluded" puts an untested gene
    into a hit list.
    """
    unreadable = Hit(gene="X", effect=float("nan"), p_value="NA", q_value="NA",
                     agreement="NA", selection_frequency="NA", n_guides=2)
    listing = HitList(hits=(unreadable,) + _hits())

    assert [h.gene for h in listing.filter(max_q=1.0)] == ["233460", "244480"]
    assert [h.gene for h in listing.filter(max_p=1.0)] == ["233460", "244480"]
    assert [h.gene for h in listing.filter(min_agreement=0.0)] == ["233460",
                                                                   "244480"]
    assert [h.gene for h in listing.filter(min_selection=0.0)] == []
    # A missing effect is missing the same way; it just arrives as NaN because
    # nothing takes the absolute value of a string successfully.
    assert [h.gene for h in listing.filter(min_effect=0.0)] == ["233460",
                                                                "244480"]


def test_an_effect_that_is_not_a_number_is_excluded_like_every_other_criterion():
    """``min_effect`` must exclude an unreadable effect, not raise on it.

    Every other numeric criterion in ``filter`` routes through a converter
    that answers False for anything it cannot read, and the method's own
    contract is that a missing value fails its criterion. ``min_effect`` calls
    ``abs()`` on the raw field first, so the one criterion whose whole job is
    to be robust to junk is the one that raises TypeError -- and it raises
    inside a filter a screen applies interactively, taking the panel down
    rather than dropping a row.
    """
    listing = HitList(hits=(Hit(gene="X", effect="n/a"),) + _hits())

    kept = listing.filter(min_effect=0.0)

    assert [hit.gene for hit in kept] == ["233460", "244480"]


def test_an_unreadable_number_is_printed_as_itself_not_as_a_dash():
    """A cell the formatter cannot parse must still show what was there.

    An em dash means "missing". Printing one for the text ``NA`` would hide
    that the coefficient table holds a string where a number belongs, which
    is a data problem the reader needs to see.
    """
    listing = HitList(hits=(Hit(gene="X", effect="n/a", p_value=float("nan"),
                                q_value=0.5),))

    table = listing.to_markdown()

    assert "| n/a |" in table
    assert "| — |" in table


def test_a_list_carries_its_notes_into_the_markdown_it_exports():
    """The caveats travel with the table or they do not travel at all.

    The Markdown is the form the list is pasted into an email in. A note
    saying the run had no per-gRNA table, or that a penalised backend has no
    p-value, changes how every number under it should be read.
    """
    listing = HitList(
        hits=_hits(), source="plate1",
        notes=("No per-gRNA coefficient table was found, so guide agreement "
               "could not be computed for any gene.",
               "2 duplicate gene term(s) in the coefficient table."))

    table = listing.to_markdown()

    assert "Note: No per-gRNA coefficient table was found" in table
    assert "Note: 2 duplicate gene term(s)" in table


def test_a_results_table_that_is_not_a_coefficient_table_says_what_was_expected():
    """A frame with no ``feature`` column has no gene terms to find.

    ``build_hit_list`` falls back from the gene file to the combined one, and
    that fallback must recognise a frame it cannot read instead of indexing a
    column that is not there.
    """
    frames = {"all": pd.DataFrame({"value": [1.0, 2.0]})}

    with pytest.raises(ValueError, match="no gene-level coefficients"):
        build_hit_list(frames)
