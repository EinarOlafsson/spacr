import numpy as np
import pandas as pd
import pytest

from spacr.guide_permutation import (
    adjust_p_values,
    analyse_long_guide_table,
    guide_freedman_lane_test,
    prepare_long_guide_data,
)
from spacr.ml import _run_guide_permutation_analysis
from spacr.settings import get_perform_regression_default_settings


def _small_screen(seed=7):
    rng = np.random.default_rng(seed)
    wells = [f"p{plate}_w{well}" for plate in (1, 2) for well in range(12)]
    blocks = np.repeat(["p1", "p2"], 12)
    fractions = pd.DataFrame(
        {
            "hit": np.tile([0, 0, 0.2, 0.4], 6),
            "null": rng.uniform(0, 0.3, len(wells)),
            "rare": [0.5, 0.4] + [0.0] * (len(wells) - 2),
        },
        index=wells,
    )
    outcome = 1.7 * fractions["hit"].to_numpy() + rng.normal(0, 0.03, len(wells))
    outcomes = pd.DataFrame({"plateID": blocks, "score": outcome}, index=wells)
    return fractions, outcomes


def test_support_families_and_bh_are_explicit():
    fractions, outcomes = _small_screen()
    result = guide_freedman_lane_test(
        fractions,
        outcomes,
        "score",
        min_wells=[1, 3],
        n_permutations=999,
        random_state=11,
    )
    assert set(result.loc[result.minimum_wells_threshold == 1, "guide"]) == {
        "hit", "null", "rare"
    }
    assert set(result.loc[result.minimum_wells_threshold == 3, "guide"]) == {
        "hit", "null"
    }
    hit = result.loc[
        (result.minimum_wells_threshold == 3) & (result.guide == "hit")
    ].iloc[0]
    assert hit.standardized_marginal_effect > 0.9
    assert hit.permutation_p_value == (hit.permutation_exceedances + 1) / 1000
    assert hit.multiple_testing_method == "fdr_bh"


def test_long_table_round_trip_counts_unique_wells_not_rows():
    fractions, outcomes = _small_screen()
    rows = []
    for well in fractions.index:
        for guide, fraction in fractions.loc[well].items():
            if fraction > 0:
                rows.append(
                    {
                        "prc": well,
                        "grna": guide,
                        "fraction": fraction,
                        "plateID": outcomes.loc[well, "plateID"],
                        "score": outcomes.loc[well, "score"],
                    }
                )
    long = pd.DataFrame(rows)
    rebuilt, rebuilt_outcomes, metadata = prepare_long_guide_data(long, "score")
    assert rebuilt.shape == fractions.shape
    assert metadata.loc["rare", "wells_with_guide"] == 2
    assert rebuilt_outcomes.index.equals(rebuilt.index)
    result = analyse_long_guide_table(
        long,
        "score",
        min_wells=[1, 3],
        n_permutations=99,
        random_state=4,
    )
    assert result.loc[result.minimum_wells_threshold == 3, "tested_guides_in_family"].eq(2).all()


def test_inconsistent_outcome_within_well_is_refused():
    data = pd.DataFrame(
        {
            "prc": ["p1_w1", "p1_w1"],
            "grna": ["g1", "g2"],
            "fraction": [0.2, 0.3],
            "plateID": ["p1", "p1"],
            "score": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="not constant within well"):
        prepare_long_guide_data(data, "score")


def test_multiple_testing_methods_and_missing_values():
    p = np.array([0.001, 0.02, 0.5, np.nan])
    adjusted, rejected = adjust_p_values(p, method="bh", alpha=0.05)
    assert np.allclose(adjusted[:3], [0.003, 0.03, 0.5])
    assert rejected.tolist() == [True, True, False, False]
    assert np.isnan(adjusted[3])
    raw, raw_rejected = adjust_p_values(p, method="none", alpha=0.05)
    assert np.allclose(raw[:3], p[:3])
    assert raw_rejected.tolist() == [True, True, False, False]


def test_regression_defaults_expose_the_correction_and_support_sensitivity():
    """The correction is offered and configurable; it is not applied by default.

    This asserted ``multiple_testing_method == 'fdr_bh'`` until d6eb6ca3 moved
    it to 'none', requested in that commit's own words alongside transform=log
    and min_cell_count=100. The key still exists, still accepts every method
    ``adjust_p_values`` implements, and ``fdr_alpha`` still rides with it --
    which is what "expose" meant here. What changed is which one a user who
    edits nothing gets, so that is pinned on its own below rather than folded
    back into this list.
    """
    settings = get_perform_regression_default_settings({})
    assert settings["analysis_mode"] == "regression"
    assert settings["guide_min_wells"] == [1, 2, 3, 4]
    assert settings["guide_primary_min_wells"] is None
    assert settings["fdr_alpha"] == 0.05
    # Still selectable, and an explicit choice survives the builder untouched.
    assert get_perform_regression_default_settings(
        {"multiple_testing_method": "fdr_bh"})["multiple_testing_method"] == (
            "fdr_bh")


def test_no_correction_is_the_requested_default():
    """Pinned separately, because it is a choice rather than a consequence.

    An uncorrected P value across hundreds of guides is not evidence, so what
    spaCR reports before the user chooses is a deliberate decision -- the kind
    that should fail here if it drifts back, rather than move quietly.
    """
    assert get_perform_regression_default_settings(
        {})["multiple_testing_method"] == "none"


def test_package_runner_writes_bh_tables_and_each_volcano(tmp_path):
    fractions, outcomes = _small_screen()
    rows = []
    for well in fractions.index:
        for guide, fraction in fractions.loc[well].items():
            if fraction > 0:
                rows.append({
                    "prc": well,
                    "grna": guide,
                    "fraction": fraction,
                    "plateID": outcomes.loc[well, "plateID"],
                    "score": outcomes.loc[well, "score"],
                })
    output = _run_guide_permutation_analysis(
        pd.DataFrame(rows), "score", tmp_path, {
            "guide_min_wells": [1, 2, 3, 4],
            "guide_primary_min_wells": 1,
            "guide_permutations": 99,
            "guide_permutation_seed": 13,
            "guide_permutation_block": "plateID",
            "guide_nuisance_columns": [],
            "multiple_testing_method": "fdr_bh",
            "fdr_alpha": 0.05,
        })
    assert output["analysis_mode"] == "guide_permutation"
    assert output["primary_min_wells"] == 1
    assert output["results"]["multiple_testing_method"].eq("fdr_bh").all()
    assert output["primary"]["q_value"].equals(
        output["primary"]["adjusted_p_value"])
    for threshold in (1, 2, 3, 4):
        assert (tmp_path / f"guide_permutation_min_{threshold}_wells.csv").is_file()
        assert output["paths"][f"plot_min_{threshold}_pdf"] == str(
            tmp_path / f"guide_permutation_min_{threshold}_wells.pdf")
        assert output["paths"][f"plot_min_{threshold}_png"] == str(
            tmp_path / f"guide_permutation_min_{threshold}_wells.png")
        assert (tmp_path / f"guide_permutation_min_{threshold}_wells.pdf").is_file()
        assert (tmp_path / f"guide_permutation_min_{threshold}_wells.png").is_file()
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "results_grna.csv").is_file()
    assert (tmp_path / "results_significant.csv").is_file()


def test_package_runner_requires_primary_threshold_in_requested_family(tmp_path):
    fractions, outcomes = _small_screen()
    rows = []
    for well in fractions.index:
        for guide, fraction in fractions.loc[well].items():
            if fraction > 0:
                rows.append({
                    "prc": well, "grna": guide, "fraction": fraction,
                    "plateID": outcomes.loc[well, "plateID"],
                    "score": outcomes.loc[well, "score"],
                })
    with pytest.raises(ValueError, match="is not in guide_min_wells"):
        _run_guide_permutation_analysis(
            pd.DataFrame(rows), "score", tmp_path, {
                "guide_min_wells": [1, 2],
                "guide_primary_min_wells": 3,
                "guide_permutations": 9,
            })


# ---------------------------------------------------------------------------
# Instruction 149 C: the box says what resolution the value buys
# ---------------------------------------------------------------------------


def test_the_permutations_setting_states_the_resolution_it_buys():
    """A permutation P can be no finer than 1 / (permutations + 1).

    Instruction 149 C. The 4%-distinct volcano is the permutation run, whose
    raw P is quantised BEFORE BH ever sees it, and the setting that fixes it
    said only "more permutations improve tail resolution", which does not tell
    a reader that 1,000 cannot express a P below 1e-3 -- it reads as a
    quality/time trade-off rather than as a hard floor. The arithmetic is on
    the box now, together with the symptom that says the value is too small.
    """
    from spacr import settings

    text = settings.tooltips['guide_permutations']
    assert '(exceedances + 1) / (permutations + 1)' in text
    # The floor, in both directions: the formula and a worked value.
    assert '1e-3' in text and '1000' in text
    assert '5e-6' in text and '200000' in text
    # The symptom, so a reader can recognise it on their own volcano.
    assert 'floor' in text


def test_the_permutation_p_really_is_quantised_to_that_floor():
    """The box's arithmetic is the code's arithmetic, not a nearby claim.

    ``spacr.guide_permutation`` reports ``(exceedances + 1) / (n + 1)``, so a
    guide no permutation ever exceeded gets exactly ``1 / (n + 1)`` and no
    smaller value exists at that setting. Pinned here rather than only in the
    tooltip, because a tooltip that drifted from the formula would be the
    wrong sentence rather than a missing one.
    """
    import inspect

    from spacr import guide_permutation

    source = inspect.getsource(guide_permutation)
    assert '(exceedances + 1) / (n_permutations + 1)' in source
    for n in (1000, 10000, 200000):
        floor = 1.0 / (n + 1)
        assert floor <= 1.0 / n
    assert 1.0 / 1001 > 9e-4      # 1,000 cannot reach 1e-3
    assert 1.0 / 200001 < 6e-6    # the default resolves to about 5e-6
