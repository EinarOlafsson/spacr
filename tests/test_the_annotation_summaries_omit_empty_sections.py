"""Sections these summaries leave out when there is nothing to put in them.

All five arcs are a heading not printed. That reads as cosmetic and is not: a
heading with nothing under it is a positive statement -- "Busiest wells:"
followed by silence says the wells were computed and were all equal, which is
a different fact from not having been computed. These summaries are read by
someone deciding whether to annotate more, so an empty section is a misleading
answer to the question they came with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _coverage(meta):
    """A coverage frame carrying ``meta`` the way annotation_coverage does."""
    frame = pd.DataFrame({"class": ["a"], "n": [1]})
    frame.attrs["spacr_annotation_coverage"] = meta
    return frame


def _base_meta(**changes):
    meta = {
        "db_path": "/data/plate1/measurements.db",
        "annotation_column": "annotate",
        "n_annotated": 10, "n_rows": 100, "n_classes": 2,
        "plates_annotated": 1, "plates_total": 1,
        "wells_annotated": 3, "wells_total": 384,
        "by_class": {"positive": 6, "negative": 4},
        "concentration": {},
        "notes": [],
    }
    meta.update(changes)
    return meta


# ---------------------------------------------------------------------------
# format_coverage_summary — three sections that can be empty
# ---------------------------------------------------------------------------

def test_no_well_or_round_breakdown_prints_neither_heading():
    """Arcs 1825 -> 1832 and 1833 -> 1840.

    A first annotation pass has neither: rounds have not started and the wells
    are not yet worth ranking. Printing the headings would tell the user
    spaCR looked and found nothing to say, when it has not looked.
    """
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta()))

    assert "Busiest wells" not in text
    assert "round" not in text.lower()


def test_a_well_and_round_breakdown_print_their_headings():
    """The taken sides, so the omissions above are visibly decisions."""
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta(
        by_well={"p1/r1/c1": 7, "p1/r2/c2": 3},
        by_round={0: 4, 1: 6})))

    assert "Busiest wells" in text
    assert "p1/r1/c1" in text
    assert "round 1" in text


def test_a_round_recorded_before_rounds_existed_is_named_in_words():
    """The negative-round key, which is a real value in older databases."""
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta(by_round={-1: 5})))

    assert "before rounds were recorded" in text


def test_no_effective_group_count_prints_no_spread_sentence():
    """Arc 1811 -> 1818.

    The sentence compares wells to their size-weighted equivalent, and without
    an effective count there is nothing to compare. Printing it with a zero
    would claim the labels are concentrated in no wells at all.
    """
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta(
        concentration={"__all__": {"n": 10, "n_groups": 3}})))

    assert "weighted by size" not in text


def test_an_effective_group_count_prints_the_spread_sentence():
    """The taken side."""
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta(
        concentration={"__all__": {"n": 10, "n_groups": 3,
                                   "effective_groups": 1.4}})))

    assert "weighted by size" in text


def test_nothing_annotated_says_so_and_stops():
    """The early return above all of it, which the tests above must not take."""
    from spacr.active_learning import format_coverage_summary

    text = format_coverage_summary(_coverage(_base_meta(by_class={},
                                                        notes=["no labels"])))

    assert "Nothing annotated yet." in text
    assert "! no labels" in text


# ---------------------------------------------------------------------------
# format_learning_curve — a round with no per-class scores
# ---------------------------------------------------------------------------

def test_a_round_without_per_class_scores_names_no_worst_class():
    """Arc 2307 -> 2310.

    Per-class scores are absent for a round trained before the column existed,
    and naming a worst class from an empty mapping would need a default -- any
    default here is a class name the user would go and look at.
    """
    from spacr.active_learning import format_learning_curve

    curve = pd.DataFrame({
        "round": [1], "n_labels": [50], "n_new_labels": [50],
        "n_holdout": [20], "holdout_accuracy": [0.80],
        "gain": [0.05], "per_class": [None],
    })

    text = format_learning_curve(curve)

    # The row is rendered, and the worst-class column is simply blank.
    assert "0.800" in text
    assert text.rstrip().endswith("0.050")


def test_a_round_with_per_class_scores_names_the_worst(tmp_path):
    """The taken side: the weakest class is named, which is the actionable bit."""
    from spacr.active_learning import format_learning_curve

    curve = pd.DataFrame({
        "round": [1], "n_labels": [50], "n_new_labels": [50],
        "n_holdout": [20], "holdout_accuracy": [0.80],
        "gain": [0.05], "per_class": [{"positive": 0.9, "negative": 0.4}],
    })

    text = format_learning_curve(curve)

    assert "negative" in text
