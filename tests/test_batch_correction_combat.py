"""ComBat: does the plate effect go, and does the biology survive?

Every test here plants a known answer. The batch shift and the treatment
effect are both written into the synthetic data by hand, so "the correction
worked" is not a matter of the numbers looking tidier -- it is whether the
treatment effect that goes in comes back out at the right size and sign.

The negative controls matter as much as the positive ones. A batch correction
that removes everything scores perfectly on the batch diagnostic, so the suite
also checks that a null screen (no treatment effect anywhere) does not acquire
one, and that a screen whose design is confounded is refused rather than
silently emptied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


#: True treatment effect written into every feature, in raw units.
TREATMENT_EFFECT = 3.0

#: Per-plate additive offsets. The spread between them is far larger than the
#: effect, which is the situation the correction exists for.
PLATE_SHIFTS = {"p1": 0.0, "p2": 4.0, "p3": -2.0}


def _screen(
    *,
    effect: float = TREATMENT_EFFECT,
    layout=(("p1", 10, 50), ("p2", 50, 10), ("p3", 30, 30)),
    n_features: int = 40,
    plate_scales=None,
    seed: int = 1,
):
    """Build a screen with a known plate effect and a known treatment effect.

    The default ``layout`` is deliberately **unbalanced**: plate ``p1`` is
    mostly treated wells and ``p2`` mostly controls. That is what a real
    screen looks like when conditions are laid out plate by plate, and it is
    the only configuration in which forgetting the covariate is visibly
    destructive -- with a perfectly balanced design every plate has the same
    mixture, so the treatment effect is orthogonal to the plate effect and
    survives whatever you do.

    :returns: ``(features, metadata)``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for plate, n_control, n_treated in layout:
        rows += [(plate, "ctrl", 0.0)] * n_control
        rows += [(plate, "treat", effect)] * n_treated
    metadata = pd.DataFrame(rows, columns=["plateID", "condition", "_effect"])
    shift = metadata["plateID"].map(PLATE_SHIFTS).to_numpy(dtype=float)
    scale = metadata["plateID"].map(plate_scales or {}).fillna(1.0)
    scale = scale.to_numpy(dtype=float)
    signal = metadata["_effect"].to_numpy(dtype=float)
    features = pd.DataFrame({
        f"feature_{index}": rng.normal(0.0, 1.0, len(metadata)) * scale
        + shift + signal
        for index in range(n_features)
    })
    return features, metadata


def _effect_size(frame: pd.DataFrame, metadata: pd.DataFrame) -> float:
    """Mean treated-minus-control difference, averaged over features."""
    means = frame.groupby(metadata["condition"].to_numpy()).mean()
    return float((means.loc["treat"] - means.loc["ctrl"]).mean())


def _plate_spread(frame: pd.DataFrame, metadata: pd.DataFrame) -> float:
    """How far apart the plates sit, measured *within* one condition.

    Measured inside the control wells only. Across all wells the plates of an
    unbalanced screen legitimately differ because they hold different mixtures
    of conditions, and a correction that drove *that* to zero would have eaten
    the treatment effect. Comparing like with like is the only way this number
    means "residual plate effect".
    """
    control = metadata["condition"].to_numpy() == "ctrl"
    subset = frame.loc[control]
    plates = metadata.loc[control, "plateID"].to_numpy()
    centers = subset.groupby(plates).mean()
    return float(centers.std(axis=0, ddof=0).mean())


# -- the point of the whole exercise ----------------------------------------


def test_combat_removes_the_plate_effect_and_keeps_the_treatment_effect():
    """Both halves, in one assertion block. The second half is the point."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    before_effect = _effect_size(features, metadata)
    before_spread = _plate_spread(features, metadata)

    corrected, report = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"],
    )

    after_effect = _effect_size(corrected, metadata)
    after_spread = _plate_spread(corrected, metadata)

    # The plate effect goes.
    assert before_spread > 2.0
    assert after_spread < 0.15
    assert report.centroid_spread_after < report.centroid_spread_before

    # The biology survives -- at the right size and the right sign. The raw
    # data understates the effect because the plate offsets fight it; the
    # correction recovers the value that was planted.
    assert before_effect < 0.6 * TREATMENT_EFFECT
    assert after_effect == pytest.approx(TREATMENT_EFFECT, abs=0.2)
    assert after_effect > 0


def test_forgetting_the_covariate_eats_the_treatment_effect():
    """Why the covariate is mandatory, demonstrated rather than asserted.

    Same data, same method, the only difference being whether the design knows
    about the condition. Without it the batch coefficients absorb part of the
    treatment contrast, the batch diagnostic looks *better* than the honest
    run, and the effect comes out too small.
    """
    from spacr.batch_correction import NO_COVARIATE, correct_batch_effects

    features, metadata = _screen()
    honest, _ = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"],
    )
    blind, _ = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=NO_COVARIATE,
    )

    honest_effect = _effect_size(honest, metadata)
    blind_effect = _effect_size(blind, metadata)

    assert honest_effect == pytest.approx(TREATMENT_EFFECT, abs=0.2)
    assert blind_effect < honest_effect - 0.5

    # ... and the run that destroyed the biology is the one whose *naive*
    # batch diagnostic -- plate centers over all wells, which is what anyone
    # eyeballing a PCA computes -- looks cleanest. That inversion is the trap:
    # the tidier picture is the wrong answer.
    def naive_spread(frame):
        centers = frame.groupby(metadata["plateID"].to_numpy()).mean()
        return float(centers.std(axis=0, ddof=0).mean())

    assert naive_spread(blind) < naive_spread(honest)

    # Comparing like with like tells the true story the other way round: among
    # control wells only, where no treatment effect can be confused for a
    # plate effect, the honest run is the one that actually flattened the
    # plates.
    assert _plate_spread(honest, metadata) < _plate_spread(blind, metadata)


def test_a_balanced_design_does_not_need_the_covariate_to_survive():
    """The complement: when every plate holds the same mixture, nothing is at
    risk, and ``NO_COVARIATE`` is an honest answer rather than a shortcut."""
    from spacr.batch_correction import NO_COVARIATE, correct_batch_effects

    features, metadata = _screen(layout=(("p1", 30, 30), ("p2", 30, 30),
                                         ("p3", 30, 30)))
    blind, _ = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=NO_COVARIATE,
    )
    assert _effect_size(blind, metadata) == pytest.approx(
        TREATMENT_EFFECT, abs=0.2)


# -- the null case ----------------------------------------------------------


def test_combat_does_not_invent_an_effect_that_was_never_there():
    """The null case, and it is not the trivial one.

    ``effect=0.0`` writes no treatment difference into the data at all. But
    the plates are unbalanced, so the *raw* table already shows a large
    apparent effect -- treated wells live mostly on the plate with the low
    offset -- of the wrong sign and about half the size of a real hit. That is
    precisely the false positive a screener reports if nobody corrects.

    The correction must drive it to zero. Not "reduce it": a shrinkage
    estimator that pulled it toward some non-zero prior would be inventing
    biology, and that failure is more expensive than doing nothing.
    """
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(effect=0.0)
    spurious = _effect_size(features, metadata)
    assert abs(spurious) > 1.0, "the unbalanced layout should fake an effect"

    corrected, report = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"],
    )

    assert _effect_size(corrected, metadata) == pytest.approx(0.0, abs=0.2)
    assert _plate_spread(corrected, metadata) < 0.15
    assert report.covariate_spread_after is not None


def test_combat_on_a_single_plate_is_an_explicit_no_op():
    """Nothing to correct, and the report says so instead of pretending."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(layout=(("p1", 30, 30),))
    corrected, report = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"],
    )
    pd.testing.assert_frame_equal(corrected, features.astype(float))
    assert any("no-op" in warning for warning in report.warnings)
    assert report.covariate_spread_after == report.covariate_spread_before


def test_combat_leaves_data_with_no_plate_effect_essentially_alone():
    """Applied to a screen that never had a plate effect, the correction is
    close to the identity -- it does not manufacture structure."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    flat = features.subtract(
        metadata["plateID"].map(PLATE_SHIFTS).to_numpy(dtype=float), axis=0)

    corrected, _ = correct_batch_effects(
        flat, metadata["plateID"],
        method="combat", covariate=metadata["condition"],
    )
    assert float((corrected - flat).abs().to_numpy().max()) < 0.75
    assert _effect_size(corrected, metadata) == pytest.approx(
        _effect_size(flat, metadata), abs=0.2)


# -- refusals ---------------------------------------------------------------


def test_combat_refuses_to_run_without_an_answer_about_the_biology():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    with pytest.raises(ValueError, match="which biology to keep"):
        correct_batch_effects(
            features, metadata["plateID"], method="combat")


def test_combat_refuses_when_batch_and_biology_are_confounded():
    """One plate, one condition. There is no arithmetic that separates them,
    and answering anyway would be the worst outcome available."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    confounded = pd.Series(
        np.where(metadata["plateID"] == "p1", "treat", "ctrl"),
        index=metadata.index, name="condition")
    with pytest.raises(ValueError, match="confounded"):
        correct_batch_effects(
            features, metadata["plateID"],
            method="combat", covariate=confounded)


def test_combat_refuses_a_plate_with_a_single_row():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(layout=(("p1", 10, 10), ("p2", 1, 0)))
    with pytest.raises(ValueError, match="at least 2 rows"):
        correct_batch_effects(
            features, metadata["plateID"], method="combat",
            covariate=metadata["condition"], min_samples=1)


def test_combat_refuses_a_covariate_with_a_missing_value():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    covariate = metadata["condition"].copy()
    covariate.iloc[3] = None
    with pytest.raises(ValueError, match="cannot be protected"):
        correct_batch_effects(
            features, metadata["plateID"],
            method="combat", covariate=covariate)


def test_combat_refuses_a_design_with_no_residual_degrees_of_freedom():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(layout=(("p1", 2, 2), ("p2", 2, 2)))
    # One level per row: the design saturates and nothing is left to
    # estimate the noise from.
    covariate = pd.Series(
        [f"gene_{index}" for index in range(len(metadata))],
        index=metadata.index, name="gene")
    with pytest.raises(ValueError, match="nothing left to estimate"):
        correct_batch_effects(
            features, metadata["plateID"], method="combat",
            covariate=covariate, min_samples=2)


def test_a_non_series_covariate_is_rejected_by_type():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    with pytest.raises(ValueError, match="must be a pandas Series"):
        correct_batch_effects(
            features, metadata["plateID"], method="combat",
            covariate=["ctrl"] * len(metadata))


# -- knobs and reporting ----------------------------------------------------


def test_mean_only_leaves_each_plate_dispersion_untouched():
    """A plate whose noise is genuinely wider stays wider under
    ``mean_only``, and is rescaled without it."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(
        plate_scales={"p1": 1.0, "p2": 3.0, "p3": 1.0},
        layout=(("p1", 40, 40), ("p2", 40, 40), ("p3", 40, 40)),
    )

    def spread_ratio(frame: pd.DataFrame) -> float:
        """p2's noise divided by p1's, measured inside the control wells.

        Inside one condition, so the treatment step is not counted as noise.
        """
        control = metadata["condition"].to_numpy() == "ctrl"
        by_plate = frame.loc[control].groupby(
            metadata.loc[control, "plateID"].to_numpy()).std(ddof=0)
        return float(by_plate.loc["p2"].mean() / by_plate.loc["p1"].mean())

    assert spread_ratio(features) > 2.0

    mean_only, _ = correct_batch_effects(
        features, metadata["plateID"], method="combat",
        covariate=metadata["condition"], combat_mean_only=True)
    full, _ = correct_batch_effects(
        features, metadata["plateID"], method="combat",
        covariate=metadata["condition"])

    assert spread_ratio(mean_only) > 2.0
    assert spread_ratio(full) == pytest.approx(1.0, abs=0.25)
    # Both still centre the plates.
    assert _plate_spread(mean_only, metadata) < 0.2
    assert _plate_spread(full, metadata) < 0.2


def test_empirical_bayes_shrinks_the_per_feature_estimates():
    """The prior buys stability, and that is what is measured here.

    With six wells per plate, the per-feature batch estimate is mostly noise.
    Shrinking it toward the across-feature distribution should make the
    *applied correction* less erratic from feature to feature -- which is the
    whole claim, and is directly observable as the spread of the per-plate
    shift across features.
    """
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen(
        layout=(("p1", 3, 3), ("p2", 3, 3), ("p3", 3, 3)),
        n_features=200, seed=7)

    def shift_spread(corrected: pd.DataFrame) -> float:
        delta = corrected - features
        by_plate = delta.groupby(metadata["plateID"].to_numpy()).mean()
        return float(by_plate.std(axis=1, ddof=0).mean())

    shrunk, _ = correct_batch_effects(
        features, metadata["plateID"], method="combat",
        covariate=metadata["condition"], combat_empirical_bayes=True)
    raw, _ = correct_batch_effects(
        features, metadata["plateID"], method="combat",
        covariate=metadata["condition"], combat_empirical_bayes=False)

    assert shift_spread(shrunk) < shift_spread(raw)
    # Both still remove the plate offsets; shrinkage is not a weaker
    # correction, it is a steadier one.
    assert _plate_spread(shrunk, metadata) < 1.0


def test_a_continuous_covariate_is_used_as_a_slope_not_a_factor():
    """A float column is a quantity. Dummy-coding a dose would produce one
    design column per distinct concentration and saturate the fit."""
    from spacr.batch_correction import correct_batch_effects

    rng = np.random.default_rng(11)
    plates = np.repeat(["p1", "p2"], 60)
    dose = rng.uniform(0.0, 10.0, 120)
    shift = np.where(plates == "p2", 5.0, 0.0)
    features = pd.DataFrame({
        f"feature_{index}": rng.normal(0, 0.5, 120) + 2.0 * dose + shift
        for index in range(15)
    })

    corrected, report = correct_batch_effects(
        features, pd.Series(plates), method="combat",
        covariate=pd.Series(dose, name="dose"))

    assert report.covariate_terms == ["dose"]
    # One term, not one per distinct dose.
    assert len(report.covariate_terms) == 1
    # The dose slope survives at its planted value of 2.0 per unit.
    slope = np.polyfit(dose, corrected.mean(axis=1).to_numpy(), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.15)
    # A continuous covariate has no groups, so no group diagnostic is faked.
    assert report.covariate_spread_before is None


def test_an_integer_coded_condition_is_read_as_a_factor():
    """Plate metadata routinely codes conditions as 0/1/2. Reading that as a
    slope would fit the wrong model without saying so."""
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    coded = (metadata["condition"] == "treat").astype(int)
    coded.name = "condition_code"

    corrected, report = correct_batch_effects(
        features, metadata["plateID"], method="combat", covariate=coded)

    assert report.covariate_terms == ["condition_code=1"]
    assert _effect_size(corrected, metadata) == pytest.approx(
        TREATMENT_EFFECT, abs=0.2)


def test_a_single_level_covariate_warns_that_it_constrained_nothing():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    flat = pd.Series(["only"] * len(metadata), index=metadata.index,
                     name="condition")
    _corrected, report = correct_batch_effects(
        features, metadata["plateID"], method="combat", covariate=flat)
    assert any("single level" in warning for warning in report.warnings)


def test_a_constant_feature_is_left_alone_and_reported():
    from spacr.batch_correction import correct_batch_effects

    features, metadata = _screen()
    features = features.copy()
    features["dead_channel"] = 7.0

    corrected, report = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"])

    assert (corrected["dead_channel"] == 7.0).all()
    assert any("no residual variance" in w for w in report.warnings)


def test_the_report_records_the_covariate_and_serializes(tmp_path):
    import json

    from spacr.batch_correction import correct_batch_effects, write_report

    features, metadata = _screen()
    _corrected, report = correct_batch_effects(
        features, metadata["plateID"],
        method="combat", covariate=metadata["condition"])

    assert report.method == "combat"
    assert report.covariate_columns == ["condition"]
    assert report.covariate_terms == ["condition=treat"]

    path = write_report(report, tmp_path / "batch_correction.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["covariate_columns"] == ["condition"]
    assert payload["covariate_spread_after"] > 0
    assert payload["centroid_spread_after"] < payload["centroid_spread_before"]


# -- the metadata adapter ---------------------------------------------------


def test_correct_from_metadata_resolves_the_covariate_column():
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen()
    corrected, report = correct_from_metadata(
        features, metadata,
        batch_correction="combat",
        batch_column="plateID",
        batch_covariate_column="condition",
    )
    assert report.covariate_columns == ["condition"]
    assert _effect_size(corrected, metadata) == pytest.approx(
        TREATMENT_EFFECT, abs=0.2)


def test_correct_from_metadata_accepts_several_covariate_columns():
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen()
    metadata = metadata.copy()
    rng = np.random.default_rng(3)
    metadata["operator"] = rng.choice(["ann", "bo"], len(metadata))

    _corrected, report = correct_from_metadata(
        features, metadata,
        batch_correction="combat",
        batch_covariate_column="condition, operator",
    )
    assert report.covariate_columns == ["condition", "operator"]
    assert report.covariate_terms == ["condition=treat", "operator=bo"]


def test_correct_from_metadata_accepts_the_explicit_none_answer():
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen(layout=(("p1", 30, 30), ("p2", 30, 30)))
    _corrected, report = correct_from_metadata(
        features, metadata,
        batch_correction="combat",
        batch_covariate_column="none",
    )
    assert report.covariate_columns == []


def test_correct_from_metadata_refuses_a_blank_covariate_setting():
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen()
    with pytest.raises(ValueError, match="which biology to keep"):
        correct_from_metadata(
            features, metadata,
            batch_correction="combat", batch_covariate_column="")


def test_correct_from_metadata_names_the_missing_covariate_column():
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen()
    with pytest.raises(ValueError, match="absent from the input metadata"):
        correct_from_metadata(
            features, metadata,
            batch_correction="combat", batch_covariate_column="treatment")


def test_the_covariate_setting_is_ignored_by_every_other_method():
    """Only ``combat`` models a covariate; the others must not start
    demanding one, or every existing settings file breaks."""
    from spacr.batch_correction import correct_from_metadata

    features, metadata = _screen()
    for method in ("center", "zscore", "robust_zscore"):
        corrected, report = correct_from_metadata(
            features, metadata,
            batch_correction=method, batch_column="plateID")
        assert report.method == method
        assert len(corrected) == len(features)


def test_combat_is_advertised_in_the_method_list():
    from spacr.batch_correction import METHODS

    assert "combat" in METHODS
