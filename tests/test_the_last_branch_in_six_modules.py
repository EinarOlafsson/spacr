"""Six modules that were each one branch short, and the branch each one missed.

Item 288 counts branches as well as statements, and a module sitting at "100%"
on statements alone can still have a decision whose other side has never been
taken. All six below were at full statement coverage and one arc short, which
is the most misleading state a coverage report has: the number reads as done.

The arcs are not arbitrary. In every case the untaken side is the QUIET one --
the duplicate that should not be added twice, the empty result that should not
be summarised, the correction that should not be applied. A branch that does
nothing visible is exactly the branch a test suite forgets, and exactly the one
whose regression nobody notices until a number in a paper is wrong.

Each test names the arc it closes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# well_scope.wells_of — arc 66 -> 65, the loop going round again
# ---------------------------------------------------------------------------

def test_a_well_holding_two_selected_guides_is_listed_once():
    """The de-duplicating branch, which only fires on a repeat.

    Every earlier test gave each well one matching row, so ``value not in
    seen`` was true every time and the ``seen`` set was never once consulted
    in anger. A real object table has many objects per well, so the repeat is
    the common case in production and was the missing case under test.
    """
    from spacr.well_scope import wells_of

    frame = pd.DataFrame({
        "well": ["A01", "A02", "A01", "A03", "A02"],
        "grna": ["g1", "g1", "g2", "g9", "g2"],
    })
    assert wells_of(frame, ["g1", "g2"]) == ["A01", "A02"]


def test_wells_come_back_in_first_occurrence_order_not_sorted():
    """The order the docstring promises, which de-duplication could destroy."""
    from spacr.well_scope import wells_of

    frame = pd.DataFrame({
        "well": ["B12", "A01", "B12", "A01"],
        "grna": ["g1", "g1", "g1", "g1"],
    })
    assert wells_of(frame, ["g1"]) == ["B12", "A01"]


# ---------------------------------------------------------------------------
# trial_metrics.guide_support_summary — arc 438 -> 442, no hits to describe
# ---------------------------------------------------------------------------

def test_a_screen_with_no_gene_hits_reports_the_counts_and_nothing_else():
    """The ``if len(hits):`` branch not taken.

    A screen that found nothing is a legitimate and common outcome, and the
    three statistics inside the branch are all reductions over an empty frame:
    ``median()`` of nothing is NaN, and a NaN written into a trial summary
    reads as a measurement rather than as an absence. Skipping them is the
    correct behaviour and had never been exercised.
    """
    from spacr.trial_metrics import guide_support_summary

    results = pd.DataFrame({
        "feature": ["grna[TGGT1_231640_1]", "grna[TGGT1_231640_2]",
                    "gene_fraction:gene[TGGT1_231640]"],
        "p_value": [0.71, 0.83, 0.64],       # nothing anywhere near alpha
        "coefficient": [0.01, 0.02, 0.015],
    })
    out = guide_support_summary(results, alpha=0.05)

    # The counts are reported -- the screen WAS tested, and said no.
    assert out["n_genes_tested"] == 1
    assert out["n_gene_hits"] == 0
    # The three statistics over the hits are absent rather than NaN. A NaN
    # here would print in a trial summary as though it had been measured.
    assert "n_single_guide_hits" not in out
    assert "n_discordant_hits" not in out
    assert "median_guides_per_hit" not in out


# ---------------------------------------------------------------------------
# roi.Roi.save — arc 451 -> 453, a path with no directory part
# ---------------------------------------------------------------------------

def test_an_roi_saves_to_a_bare_filename_in_the_working_directory(tmp_path,
                                                                  monkeypatch):
    """A bare filename resolves against the cwd and writes there.

    This test does NOT close roi.py's arc 451->453, and the attempt to make it
    do so is what proved that arc unreachable. ``save`` calls
    ``os.path.abspath`` at line 448 BEFORE taking the dirname, and
    ``dirname(abspath(x))`` is never empty on POSIX -- checked against ``''``,
    ``'.'``, ``'/'``, ``'//'`` and a bare name, all of which yield a non-empty
    directory. So ``if parent:`` is unconditionally true and its false side is
    dead. Recorded in instruction 310; the guard is left in place because it
    is correct for a ``target`` that has not been made absolute, which is what
    the next edit to this function could easily produce.

    What is asserted here is the behaviour a user relies on instead.
    """
    from spacr.roi import RegionOfInterest, RoiSet

    monkeypatch.chdir(tmp_path)
    region = RegionOfInterest(kind="rectangle",
                              vertices=[[0.0, 0.0], [4.0, 0.0],
                                        [4.0, 4.0], [0.0, 4.0]])
    written = RoiSet(fields={"*": (region,)}).save("region.json")

    assert os.path.isfile(written)
    with open(written, encoding="utf-8") as handle:
        assert json.load(handle)["spacr_roi_version"] == 1


# ---------------------------------------------------------------------------
# object_distances._sample — arc 149 -> 152, nothing inside the field
# ---------------------------------------------------------------------------

def test_points_entirely_outside_the_field_sample_as_infinite():
    """The ``if inside.any():`` branch not taken.

    Infinity is the right answer and the reason the guard exists: a distance
    that cannot be measured must not read as zero, because zero means
    TOUCHING. A centroid outside the field is what a bad mask or a mismatched
    stack produces, and reporting those objects as being in contact would be a
    scientific error, not a cosmetic one.
    """
    from spacr.object_distances import _sample

    field = np.arange(9, dtype=float).reshape(3, 3)
    out = _sample(field, [(-5.0, -5.0), (99.0, 99.0)])

    assert out.shape == (2,)
    assert np.isinf(out).all()


def test_a_point_inside_the_field_samples_its_value():
    """The other side, so the test above is a contrast and not a coincidence."""
    from spacr.object_distances import _sample

    field = np.arange(9, dtype=float).reshape(3, 3)
    out = _sample(field, [(1.0, 2.0), (-5.0, -5.0)])

    assert out[0] == field[1, 2]
    assert np.isinf(out[1])


# ---------------------------------------------------------------------------
# fraction_calibration._disagreement — arc 142 -> 144, an unusable correction
# ---------------------------------------------------------------------------

def test_an_unusable_correction_leaves_the_measurement_alone(monkeypatch):
    """The ``if fixed.get("usable"):`` branch not taken.

    Rogan-Gladen can return a proportion outside [0, 1] when the classifier's
    sensitivity and specificity do not support the observed rate, and it says
    so with ``usable``. Using the number anyway would put an impossible
    fraction into a calibration, so the branch that declines it is the one
    protecting the result -- and it had never run.
    """
    from spacr import fraction_calibration as fc
    import spacr.classifier_quality as cq

    monkeypatch.setattr(cq, "rogan_gladen",
                        lambda *_a, **_k: {"usable": False, "corrected": 99.0})

    per_well = {"A01": (0.50, 0.40), "A02": (0.20, 0.30)}
    gap = fc._disagreement(per_well, {"sensitivity": 0.9, "specificity": 0.9})

    # Median of |0.40-0.50| and |0.30-0.20| — the RAW seen values, uncorrected.
    assert gap == pytest.approx(0.10)


def test_a_usable_correction_is_applied(monkeypatch):
    """The taken side, so the refusal above is visibly a different outcome."""
    from spacr import fraction_calibration as fc
    import spacr.classifier_quality as cq

    monkeypatch.setattr(cq, "rogan_gladen",
                        lambda *_a, **_k: {"usable": True, "corrected": 0.50})

    per_well = {"A01": (0.50, 0.40)}
    gap = fc._disagreement(per_well, {"sensitivity": 0.9, "specificity": 0.9})

    assert gap == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_recommendations.recommend — arc 134 -> 141, already robust
# ---------------------------------------------------------------------------

def test_an_influential_point_is_not_flagged_when_the_fit_is_already_robust():
    """The ``if kind not in (...)`` branch not taken.

    Recommending 'rlm' to a run that IS 'rlm' is the kind of advice that
    teaches users to ignore the panel. The suppression is the whole value of
    the check and was the untested half of it.
    """
    from spacr.run_recommendations import recommend

    diagnostics = {"max_cooks_distance": 4.0}
    for robust in ("rlm", "huber", "quantile"):
        made = recommend(diagnostics, settings={"regression_type": robust})
        assert not [r for r in made if r.setting == "regression_type"], robust


def test_an_influential_point_is_flagged_for_least_squares():
    """The taken side: the same diagnostics, a fit that cannot absorb them."""
    from spacr.run_recommendations import recommend

    made = recommend({"max_cooks_distance": 4.0},
                     settings={"regression_type": "ols"})
    flagged = [r for r in made if r.setting == "regression_type"]
    assert len(flagged) == 1
    assert "4.00" in flagged[0].because or "4.0" in flagged[0].because
