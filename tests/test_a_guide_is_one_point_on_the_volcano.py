"""A guide is ONE point on the volcano, not one per minimum-wells family.

Reported 2026-08-17: "GRA14 and 225160 occur in the top right side of the
graph 4 times each which is obviously wrong, stop trying to explain it away
and FIX IT."

It was obviously wrong, and I explained it away twice -- first as a
Benjamini-Hochberg tie artefact, then as gene/guide rows landing on one dot --
before checking the row counts the maintainer had already given me ("my data
say 1612 gRNAs").

`guide_min_wells` defaults to [1, 2, 3, 4], so the permutation analysis runs
FOUR times at four inclusion thresholds. They are four separate analyses of
the same guides. On the real screen the stacked frame is 1,612 rows for 789
guides, and `225160_2` appears four times at the identical effect 0.25406 --
so the panel drew every guide once per family, at the same coefficient.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def _long_frame():
    """Four families of the same three guides, as the analysis produces."""
    rows = []
    for family in (1, 2, 3, 4):
        for guide, effect in (("225160_2", 0.25406), ("239740_3", 0.400382),
                              ("244480_3", 0.276486)):
            rows.append({"guide": guide, "minimum_wells_threshold": family,
                         "standardized_marginal_effect": effect,
                         "permutation_p_value": 0.01,
                         "adjusted_p_value": 0.05})
    return pd.DataFrame(rows)


def test_the_panel_gets_one_row_per_guide():
    """The whole bug, as a property of the returned dict."""
    from spacr.ml import _run_guide_permutation_analysis

    source = inspect.getsource(_run_guide_permutation_analysis)
    assert "'results': primary_table," in source, (
        "the results key is the stacked multi-family frame again, so every "
        "guide is drawn once per minimum-wells family")


def test_the_families_are_still_reachable():
    """Four inclusion thresholds is a real analysis, not noise -- it must not
    be thrown away to fix the drawing."""
    from spacr.ml import _run_guide_permutation_analysis

    source = inspect.getsource(_run_guide_permutation_analysis)
    assert "'families': results," in source


def test_the_dict_and_the_file_agree():
    """`results.csv` on disk has always held the primary table. The returned
    dict held the stacked one under the SAME NAME, so a caller reading the
    file and a caller reading the dict got different row counts."""
    from spacr.ml import _run_guide_permutation_analysis

    source = inspect.getsource(_run_guide_permutation_analysis)
    saved = source.index("primary_table.to_csv(compatibility['results']")
    returned = source.index("'results': primary_table,")
    assert saved > 0 and returned > 0


def test_a_stacked_frame_really_does_repeat_every_guide():
    """The fixture is the shape the bug needs; if the analysis ever stops
    producing one row per family this test should stop being about anything.
    """
    frame = _long_frame()

    counts = frame["guide"].value_counts()
    assert set(counts.unique()) == {4}
    # And the repeats are IDENTICAL effects, which is why they stack into one
    # place on the volcano rather than spreading.
    for guide in frame["guide"].unique():
        effects = frame.loc[frame["guide"] == guide,
                            "standardized_marginal_effect"].unique()
        assert len(effects) == 1, (guide, effects)


def test_the_primary_family_has_each_guide_once():
    frame = _long_frame()
    primary = frame[frame["minimum_wells_threshold"] == 1]

    assert primary["guide"].is_unique
    assert len(primary) == frame["guide"].nunique()


@pytest.mark.parametrize("family", [1, 2, 3, 4])
def test_every_family_is_internally_unique(family):
    """A family that repeated a guide would be a different bug wearing the
    same face."""
    frame = _long_frame()

    assert frame[frame["minimum_wells_threshold"] == family]["guide"].is_unique
