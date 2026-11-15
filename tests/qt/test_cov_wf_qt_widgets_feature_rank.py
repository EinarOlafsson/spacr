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
    AUC,
    DEFAULT_TOP,
    KS,
    ExplorerSpec,
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
# The label-shuffling null over a feature that cannot be shuffled apart
# ---------------------------------------------------------------------------

def test_a_feature_with_one_measured_object_leaves_the_null_at_zero():
    """The null shuffles the class labels, re-ranks every feature and keeps
    the best score of each shuffle. A feature that is missing on all but one
    object contributes nothing to that: whichever class the shuffle deals to
    the single remaining row, the other group is empty and the separation is
    NaN. Those shuffles must not reach ``np.quantile`` as NaN, because it
    propagates and the threshold would come back NaN -- at which point ``score > threshold`` is
    False for *every* feature in the table and the "beats the null" list is
    silently empty. A feature with real values in both classes must still
    push the same null above zero.

    THE ANSWER IS NOW ``None``, NOT ``0.0``, AND IT CLOSES BOTH WORRIES.
    This asserted 0.0, which was the first fix for the NaN and carried a
    fault of its own: a shuffle that MEASURED NOTHING is not a shuffle
    that measured zero, and on a sparsely measured table those spurious
    zeros are a sizeable fraction of the null -- deflating the 95th
    percentile until `above_null` lists features that never beat chance.
    `_null_threshold` drops such a shuffle instead of scoring it, and
    returns None when every shuffle was dropped.

    None is not NaN, so the hazard this test was written for is still
    shut: `above_null` returns the WHOLE ranking on None and `summary()`
    says the null was not run -- "not run is not the same as passed".
    Nothing is silently empty, and nothing is silently zero.
    """
    keys = np.array(["ctrl", "ctrl", "ctrl", "trt", "trt", "trt"],
                    dtype=object)
    levels = ("ctrl", "trt")
    spec = ExplorerSpec(label="condition", n_permutations=8, seed=7)
    nan = float("nan")

    notices: list = []
    lonely = {"cell_area": np.array([1.0, nan, nan, nan, nan, nan])}
    threshold = FR._null_threshold(lonely, keys, levels, spec, notices)

    assert threshold is None
    assert notices and "measured no feature" in notices[0], (
        "dropping every shuffle must be SAID, not just returned as None")

    measured = {"cell_area": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
    real = FR._null_threshold(measured, keys, levels, spec, [])
    assert real > 0.0
    assert np.isfinite(real)


# ---------------------------------------------------------------------------
# The same three corners, reached by the other route into each of them
# ---------------------------------------------------------------------------

def test_an_unrecognised_preset_restores_the_default_ranking_not_a_crash():
    """The payload handed to ``from_dict`` comes off disk, so it can be a file
    that holds nothing this version understands -- an empty dict written by a
    truncated save, or a preset from a fork whose keys are all foreign. Both
    have to restore as the default ranking, which is what the screen opens
    with anyway, because the alternative is the settings load raising and the
    user losing every *other* preset in the same file. Note this is the branch
    where ``features`` is absent rather than empty: nothing may reach into
    ``known`` for a key that was never put there.
    """
    blank = ExplorerSpec.from_dict({})

    assert blank == ExplorerSpec()
    assert blank.features == ()
    assert blank.label == ""
    assert blank.statistic == AUC
    assert blank.top == DEFAULT_TOP
    assert blank.describe().startswith(
        "every continuous column split by (no class column), ranked by ")

    foreign = ExplorerSpec.from_dict({"colour": "red", "schema_version": 4,
                                      "order_by": "p_value"})
    assert foreign == blank
    assert ExplorerSpec.from_json(blank.to_json()) == blank


def test_a_row_with_only_an_effect_size_names_the_effect_size(two_conditions):
    """The remaining combination of the three caption guards, and the one a
    Cohen's-d ranking actually produces: a feature whose ranks are unusable --
    every object tied, so the AUC and the KS gap are both undefined -- while
    the means still differ. The caption has to carry the ``d`` and drop the
    other two silently, because ``d -1.25`` is the finding and ``AUC nan``
    beside it would read as a failed computation and get the row ignored. The
    small-class warning still has to be appended after it, since a d computed
    on four objects is exactly the number a reader must not take at face
    value.
    """
    thin = FR.FeatureScore(
        feature="mito_intensity", statistic=FR.COHEN_D, score=1.25,
        auc=float("nan"), cohen_d=-1.25, ks=float("nan"),
        mutual_info=float("nan"), higher_in="ctrl", against="trt",
        n_by_class={"ctrl": 4, "trt": 9})

    said = thin.describe()

    assert said == "mito_intensity: 1.250 · d -1.25 · n=4 in the smaller class"
    assert "AUC" not in said
    assert "KS" not in said

    complete = replace(thin, auc=0.18, ks=0.64)
    full = complete.describe()
    assert "AUC 0.180, higher in ctrl" in full
    assert "KS 0.640" in full
    assert "d -1.25" in full


def test_a_column_measured_only_on_unlabelled_rows_does_not_poison_the_null():
    """Rows with no condition are dropped before the labels are shuffled, so a
    column recorded *only* on those rows -- a stain added for the untreated
    wells that were never assigned a condition -- arrives at the null with
    nothing finite left in it at all. Every shuffle then compares two empty
    groups and the separation is NaN. Those NaNs must be skipped rather than
    accumulated, because ``np.quantile`` propagates a single NaN into the
    whole threshold, and a NaN threshold makes ``score > threshold`` False for
    every feature in the table: the panel would report that nothing beat
    chance while showing a column that separates perfectly. A real column
    measured beside it must still lift the same null off zero.
    """
    nan = float("nan")
    frame = pd.DataFrame({
        "condition": ["ctrl", "ctrl", "ctrl", "trt", "trt", "trt", None, None],
        "orphan_stain": [nan] * 6 + [3.0, 9.0],
        "cell_area": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, nan, nan],
    })
    keys, levels = FR._class_levels(frame, "condition")
    spec = ExplorerSpec(label="condition", statistic=KS, n_permutations=6,
                        seed=11)

    orphan = {"orphan_stain": frame["orphan_stain"].to_numpy(float)}
    # None rather than 0.0: a column measured only on unlabelled rows
    # contributes no measurement to any shuffle, and scoring that as
    # zero is exactly what would poison the null this test is named for.
    assert FR._null_threshold(orphan, keys, levels, spec, []) is None

    both = {name: frame[name].to_numpy(float)
            for name in ("orphan_stain", "cell_area")}
    together = FR._null_threshold(both, keys, levels, spec, [])

    assert np.isfinite(together)
    assert together > 0.0
    assert together == FR._null_threshold(
        {"cell_area": frame["cell_area"].to_numpy(float)}, keys, levels,
        spec, [])
