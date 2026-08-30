"""Three corners of the feature ranking that nothing else drives.

Every branch here is a place where the module has to cope with something
*missing* and must still produce a usable answer rather than a crash or a
number that lies:

* a saved :class:`~spacr.qt.widgets.feature_rank.ExplorerSpec` payload with no
  ``features`` key at all -- what a preset written before the field existed,
  or by hand, actually looks like on the way back in;
* the per-row caption for a score whose statistics could not be computed,
  which must not offer the reader ``AUC nan, higher in ctrl`` as if it were a
  finding;
* the label-shuffling null over a feature that was measured on too few objects
  for a shuffle to ever produce two populations -- the null has to come back
  ``0.0`` ("nothing beat chance here") and not ``NaN``, because every
  ``score > NaN`` comparison is False and would silently empty the
  "beats the null" list for the *other* features too.

No Qt widgets are built; ``feature_rank`` is numpy and pandas. The import
still needs PySide6 because the shared column classifier lives in a widget
module, which is why this file sits under ``tests/qt/``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import feature_rank as FR
from spacr.qt.widgets.feature_rank import (
    AUC, KS, DEFAULT_TOP, ExplorerSpec, rank_features,
)


@pytest.fixture
def two_conditions() -> pd.DataFrame:
    """Sixteen objects, eight per condition, one cleanly separating feature.

    ``cell_area`` is entirely above in ``trt`` (AUC 1.0), so a ranking over
    this table has a finite value for every statistic -- the contrast the
    tests below need in order to show what a *missing* value looks like.
    Eight per class keeps the classes above ``LOW_N``, so the captions carry
    the statistics and nothing else.
    """
    return pd.DataFrame({
        "condition": ["ctrl"] * 8 + ["trt"] * 8,
        "cell_area": [100.0 + 10.0 * i for i in range(8)]
                     + [200.0 + 10.0 * i for i in range(8)],
    })


# ---------------------------------------------------------------------------
# Restoring a spec that predates the feature list
# ---------------------------------------------------------------------------

def test_a_saved_ranking_without_a_features_key_restores_as_every_column():
    """A stored ranking is round-tripped through ``from_dict``, and the
    payload does not always carry every field: a preset saved before the
    feature list existed, or written by hand, has only a label and a
    statistic. That must restore as "rank every continuous column" -- the
    default the screen exists for -- rather than raising a KeyError and
    losing the user's saved comparison. The same call has to keep an explicit
    list when one *is* present, and hand it back as a tuple so the frozen
    spec stays hashable.
    """
    lean = ExplorerSpec.from_dict({"label": "condition", "statistic": KS,
                                   "top": 5})

    assert lean.features == ()
    assert lean.label == "condition"
    assert lean.statistic == KS
    assert lean.top == 5
    assert lean.describe().startswith("every continuous column split by "
                                      "condition")

    full = ExplorerSpec.from_dict({"label": "condition",
                                   "features": ["cell_area", "nucleus_area"]})
    assert full.features == ("cell_area", "nucleus_area")
    assert isinstance(full.features, tuple)
    assert full.statistic == AUC and full.top == DEFAULT_TOP
    assert full.describe().startswith("2 features split by condition")


def test_a_saved_ranking_with_an_empty_feature_list_is_not_a_crash():
    """``features`` present but null is what a serialiser writes for "no
    explicit selection". It has to mean the same as the key being absent --
    rank everything -- because the alternative is ``tuple(None)`` raising
    while the user reopens a saved screen. An unknown extra key in the same
    payload must be dropped rather than passed to the constructor, which is
    what lets a newer spaCR read a spec an older one wrote and the reverse.
    """
    restored = ExplorerSpec.from_dict(
        {"label": "condition", "features": None, "colour": "red"})

    assert restored.features == ()
    assert restored.label == "condition"
    assert not hasattr(restored, "colour")
    assert ExplorerSpec.from_json(restored.to_json()) == restored


# ---------------------------------------------------------------------------
# The caption for a score with nothing in it
# ---------------------------------------------------------------------------

def test_a_row_whose_statistics_are_missing_prints_none_of_them(two_conditions):
    """Every ranked feature carries a one-line caption that the panel puts
    beside it. Each statistic is printed only when it is finite, and that
    guard is the whole point: a feature whose groups collapsed has NaN for
    AUC, for Cohen's d and for KS, and printing them anyway would put
    ``AUC nan, higher in trt`` in front of a reader as though the direction of
    the difference were known. The caption for such a row is the feature and
    its score and nothing else; a real row still says all three.
    """
    real = rank_features(two_conditions,
                         ExplorerSpec(label="condition")).score_for("cell_area")

    said = real.describe()
    assert said.startswith("cell_area: 1.000")
    assert "AUC 1.000, higher in trt" in said
    assert "KS 1.000" in said
    assert "d +" in said

    nothing = replace(real, score=float("nan"), auc=float("nan"),
                      cohen_d=float("nan"), ks=float("nan"),
                      mutual_info=float("nan"))
    empty = nothing.describe()

    assert empty == "cell_area: nan"
    assert "AUC" not in empty
    assert "KS" not in empty


def test_a_row_with_only_a_spread_statistic_still_names_the_spread(
        two_conditions):
    """The three guards are independent, and the caption must survive any
    combination of them. A KS-only row -- the shape of a feature whose ranks
    tie everywhere but whose distributions differ -- has to keep its KS
    number, because that value is the only evidence the reader has that the
    feature is worth a second look. Dropping it along with the two NaNs would
    turn the "shape, not shift" warning into an unexplained flag.
    """
    real = rank_features(two_conditions,
                         ExplorerSpec(label="condition")).score_for("cell_area")

    ks_only = replace(real, score=0.05, auc=float("nan"),
                      cohen_d=float("nan"), ks=0.42)
    said = ks_only.describe()

    assert said == "cell_area: 0.050 · KS 0.420"
    assert "AUC" not in said and "d " not in said

    both = replace(real, score=0.05, auc=0.52, cohen_d=float("nan"), ks=0.42)
    assert both.describe() == (
        "cell_area: 0.050 · AUC 0.520, higher in trt · KS 0.420 · "
        "distributional shape differs without a location shift; a rank "
        "statistic does not detect this pattern")


# ---------------------------------------------------------------------------
# The label-shuffling null over a feature that cannot be shuffled apart
# ---------------------------------------------------------------------------

def test_a_feature_with_one_measured_object_leaves_the_null_at_zero():
    """The null shuffles the class labels, re-ranks every feature and keeps
    the best score of each shuffle. A feature that is missing on all but one
    object contributes nothing to that: whichever class the shuffle deals to
    the single remaining row, the other group is empty and the separation is
    NaN. Those shuffles must contribute ``0.0`` to the null distribution
    rather than a NaN, because ``np.quantile`` propagates NaN and the
    threshold would come back NaN -- at which point ``score > threshold`` is
    False for *every* feature in the table and the "beats the null" list is
    silently empty. A feature with real values in both classes must still
    push the same null above zero.
    """
    keys = np.array(["ctrl", "ctrl", "ctrl", "trt", "trt", "trt"],
                    dtype=object)
    levels = ("ctrl", "trt")
    spec = ExplorerSpec(label="condition", n_permutations=8, seed=7)
    nan = float("nan")

    notices: list = []
    lonely = {"cell_area": np.array([1.0, nan, nan, nan, nan, nan])}
    threshold = FR._null_threshold(lonely, keys, levels, spec, notices)

    assert threshold == 0.0
    assert notices == []

    measured = {"cell_area": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
    real = FR._null_threshold(measured, keys, levels, spec, [])
    assert real > 0.0
    assert np.isfinite(real)


def test_a_feature_measured_on_two_objects_makes_the_null_unbeatable(
        two_conditions):
    """The same code end to end, because the direct call above proves the
    arithmetic and not the wiring -- and because what it does to a real
    ranking is worth stating. The null is the best score of *any* feature
    under shuffling, so a column measured on two objects, a stain only a
    couple of wells got, reaches a separation of 1.0 on any shuffle that
    deals those two rows to different classes, and the family-wise threshold
    goes to 1.0 with it. Everything then stops beating the null, including a
    feature that separates perfectly on all sixteen objects. That is the
    honest answer -- with that column in the ranking the best score really is
    reachable by chance -- and the screen has to be able to say it, which
    means the shuffles where the sparse column collapses into one class must
    contribute 0.0 rather than a NaN that would take the whole quantile with
    it.
    """
    alone = rank_features(two_conditions,
                          ExplorerSpec(label="condition", n_permutations=12,
                                       seed=3))
    assert alone.null_threshold is not None and alone.null_threshold < 1.0
    assert [s.feature for s in alone.above_null()] == ["cell_area"]

    frame = two_conditions.copy()
    sparse = [np.nan] * 16
    sparse[0], sparse[15] = 5.0, 9.0
    frame["rare_stain"] = sparse

    result = rank_features(frame, ExplorerSpec(label="condition",
                                               n_permutations=12, seed=3))

    assert result.n_considered == 2
    assert result.skipped == {}
    assert result.null_threshold == 1.0
    assert result.above_null() == ()
    assert "shuffling the labels reaches 1.000" in result.summary()
    assert "0 feature(s) beat it" in result.summary()
