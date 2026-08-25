"""Background correction survives a broken resolver and unusable numbers.

Three ways this module can be handed something it cannot use, and each has a
wrong answer that would pass unnoticed:

* the shared name resolver fails, and returning an empty exclusion set would
  leave a known contaminant in the denominator;
* a guide's background is not a number, and letting NaN through would poison
  every corrected fraction in the well;
* a measurement carries no control wells, and reporting suspicious guides
  from it would accuse guides on no evidence.
"""
from __future__ import annotations

import math

import numpy as np

from spacr import control_names, read_background


def test_a_resolver_that_raises_falls_back_to_the_exact_names(monkeypatch):
    """Excluding nothing is the dangerous answer here, so a resolver that
    cannot run must still exclude the guides the user named exactly."""
    def refuse(*_args, **_kwargs):
        raise RuntimeError("the resolver is unavailable")

    monkeypatch.setattr(control_names, "rows_for", refuse)

    resolved = read_background.resolve_exclusions(
        ["guide_a", "GRA14"], ["guide_a", "guide_b", "GRA14_1"])

    assert resolved == {"guide_a", "GRA14"}


def test_the_working_resolver_expands_a_gene_to_its_guides():
    """The contrast that shows the fallback is a fallback: with the resolver
    present a gene name selects the guides it names, which the exact-name
    fallback cannot do."""
    resolved = read_background.resolve_exclusions(
        ["GRA14"], ["GRA14_1", "GRA14_2", "ROP18_1"], genes=None)

    assert "GRA14_1" in resolved and "GRA14_2" in resolved
    assert "ROP18_1" not in resolved


def test_a_background_that_is_not_a_number_leaves_the_estimate_undefined():
    """A guide with no usable background makes the median undefined, and the
    suggested threshold has to come back as NaN rather than as a number
    computed from a comparison every value lost."""
    result = read_background.suggest_threshold(
        {"background": {"g1": 0.01, "g2": float("nan"), "g3": 0.02}})

    assert math.isnan(result["threshold"])
    assert result["guides"] == 3.0
    # Nothing survived the outlier cut, so the estimate fell back to using
    # every guide rather than reporting a threshold from an empty sample.
    assert result["guides_used"] == 3.0
    assert result["guides_needing_their_own"] == 0.0


def test_a_non_finite_fraction_is_dropped_from_the_corrected_well():
    """A NaN share must not reach the output or the renormalisation total;
    one unreadable guide would otherwise make the whole well NaN."""
    corrected = read_background.subtract_background(
        {"g1": 0.5, "g2": float("nan"), "g3": 0.5},
        {"g1": 0.1, "g3": 0.0})

    assert set(corrected) == {"g1", "g3"}
    assert all(np.isfinite(v) for v in corrected.values())
    # Renormalised back onto the mass the finite guides started with.
    assert corrected["g1"] + corrected["g3"] == 1.0
    assert corrected["g3"] > corrected["g1"], (
        "the guide with background subtracted must end below the one without")


def test_no_control_wells_accuses_no_guide():
    """A measurement with background but no wells behind it cannot support a
    'seen everywhere' claim, so the verdict list is empty."""
    assert read_background.suspicious({}) == []
    assert read_background.suspicious(
        {"background": {"g1": 0.5}, "seen_in_wells": {"g1": 3},
         "control_wells": 0}) == []
    assert read_background.suspicious(
        {"background": {}, "control_wells": 8}) == []
