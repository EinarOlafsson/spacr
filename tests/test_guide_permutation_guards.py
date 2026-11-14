"""Input guards and the arithmetic of the guide Freedman-Lane permutation test.

One table-driven test per unit of :mod:`spacr.guide_permutation` that turns a
long regression table into a tested guide list: ``_normalise_thresholds``,
``prepare_long_guide_data``, ``_nuisance_design`` and
``guide_freedman_lane_test``.  Every refusal is driven with an otherwise valid
long table carrying one bad value, because each guard exists to catch that
value before 200,000 permutations are spent on it.  Each row runs as a subtest
so a failing case names itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import guide_permutation as gp


def _long_table(wells=8, guides=("g1", "g2", "g3"), seed=0):
    """A valid long guide table: two plates, one fraction per well/guide."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(wells):
        well = f"p{index // 4 + 1}_r1_c{index + 1}"
        plate = f"plate{index // 4 + 1}"
        phenotype = float(rng.normal())
        for order, guide in enumerate(guides):
            rows.append({
                "prc": well,
                "grna": guide,
                "fraction": float(rng.uniform(0.05, 0.9)) + 0.01 * order,
                "plateID": plate,
                "pathogen_rate": phenotype,
                "cell_count": float(200 + index * 7),
                "batch": "A" if index % 2 else "B",
            })
    return pd.DataFrame(rows)


def _prepared():
    return gp.prepare_long_guide_data(
        _long_table(), "pathogen_rate", nuisance_columns=["cell_count"])


def _within_plate_centred(values, plates):
    """Values minus their plate mean: the block-only Freedman-Lane residual."""
    series = pd.Series(np.asarray(values, dtype=float))
    return (series - series.groupby(np.asarray(plates)).transform("mean")).to_numpy()


def test_normalise_thresholds(subtests):
    """Support thresholds become sorted unique positive ints, or are refused."""
    cases = [
        ("int", 4, (4,)),
        ("numpy int", np.int64(2), (2,)),
        ("repeats collapse and sort", [3, 1, 3, 2], (1, 2, 3)),
        ("tuple", (5, 5), (5,)),
        ("string", "4", (TypeError, "positive integer or iterable")),
        ("None", None, (TypeError, "positive integer or iterable")),
        ("float", 3.5, (TypeError, "positive integer or iterable")),
        ("empty", [], (ValueError, "at least one positive integer")),
        ("zero", [0], (ValueError, "at least one positive integer")),
        ("one negative", [2, -1], (ValueError, "at least one positive integer")),
    ]
    for case_id, value, expected in cases:
        with subtests.test(msg=case_id):
            if isinstance(expected, tuple) and isinstance(expected[0], type):
                with pytest.raises(expected[0], match=expected[1]):
                    gp._normalise_thresholds(value)
            else:
                assert gp._normalise_thresholds(value) == expected


def test_prepare_long_guide_data(subtests):
    """A valid long table pivots exactly; one bad value anywhere is refused."""
    table = _long_table()
    fractions, outcomes, meta = gp.prepare_long_guide_data(
        table, "pathogen_rate", nuisance_columns=["cell_count"])

    wells = sorted(table["prc"].unique())
    with subtests.test(msg="pivoted fractions"):
        assert list(fractions.index) == wells
        assert list(fractions.columns) == ["g1", "g2", "g3"]
        expected = table.pivot(index="prc", columns="grna", values="fraction")
        np.testing.assert_array_equal(fractions.to_numpy(),
                                      expected.loc[wells].to_numpy())
    with subtests.test(msg="one outcome row per well"):
        assert list(outcomes.index) == wells
        assert list(outcomes.columns) == ["plateID", "pathogen_rate", "cell_count"]
        first = table.drop_duplicates("prc").set_index("prc").loc[wells]
        np.testing.assert_array_equal(outcomes["pathogen_rate"].to_numpy(),
                                      first["pathogen_rate"].to_numpy())
        assert list(outcomes["plateID"]) == ["plate1"] * 4 + ["plate2"] * 4
    with subtests.test(msg="guide support"):
        assert list(meta.index) == ["g1", "g2", "g3"]
        assert meta["wells_with_guide"].tolist() == [8, 8, 8]
    with subtests.test(msg="duplicate rows are summed, absent pairs are zero"):
        extra = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        extra = extra[~((extra["prc"] == wells[1]) & (extra["grna"] == "g3"))]
        f2, _o2, m2 = gp.prepare_long_guide_data(extra, "pathogen_rate")
        assert f2.loc[table.loc[0, "prc"], "g1"] == 2 * table.loc[0, "fraction"]
        assert f2.loc[wells[1], "g3"] == 0.0
        assert m2["wells_with_guide"].tolist() == [8, 8, 7]

    def _set(column, value, rows=(0,)):
        def edit(frame):
            frame.loc[list(rows), column] = value
        return edit

    first_well = table.index[table["prc"] == table["prc"].iloc[0]].tolist()
    refusals = [
        ("missing well id", _set("prc", np.nan),
         "well and guide identifiers must not contain missing values"),
        ("missing guide id", _set("grna", None),
         "well and guide identifiers must not contain missing values"),
        ("infinite fraction", _set("fraction", np.inf, (2,)),
         "guide fractions must be finite"),
        ("negative fraction", _set("fraction", -0.2, (3,)),
         "guide fractions must be non-negative"),
        ("infinite phenotype for a whole well",
         _set("pathogen_rate", np.inf, first_well),
         "Outcome 'pathogen_rate' must be finite for every analyzed well"),
        ("phenotype not constant within a well", _set("pathogen_rate", 99.0),
         "not constant within well 'p1_r1_c1'"),
        ("missing column", lambda frame: frame.drop(columns="plateID", inplace=True),
         r"missing required columns: \['plateID'\]"),
    ]
    for case_id, edit, message in refusals:
        with subtests.test(msg=case_id):
            bad = table.copy()
            edit(bad)
            with pytest.raises(ValueError, match=message):
                gp.prepare_long_guide_data(bad, "pathogen_rate")


def test_nuisance_design(subtests):
    """Intercept + plate dummies + nuisances, deduplicated and full rank."""
    _fractions, outcomes, _meta = _prepared()
    intercept = np.ones(8)
    plate2 = np.array([0.0] * 4 + [1.0] * 4)
    count = outcomes["cell_count"].to_numpy()
    batch = (np.arange(8) % 2 == 0).astype(float)  # "B" after drop_first of "A"
    with_batch = outcomes.assign(batch=np.where(np.arange(8) % 2, "A", "B"))
    with_site = outcomes.assign(
        site=outcomes["plateID"].map({"plate1": "north", "plate2": "south"}))
    one_plate = outcomes.assign(plateID="plate1")

    designs = [
        ("block only", outcomes, [], np.column_stack([intercept, plate2])),
        ("block named again is not added twice", outcomes, ["plateID"],
         np.column_stack([intercept, plate2])),
        ("numeric nuisance adds one column", outcomes, ["cell_count"],
         np.column_stack([intercept, plate2, count])),
        ("categorical nuisance adds dummies", with_batch, ["batch"],
         np.column_stack([intercept, plate2, batch])),
        ("a single block adds no dummy", one_plate, [], intercept[:, None]),
    ]
    for case_id, frame, nuisance, expected in designs:
        with subtests.test(msg=case_id):
            np.testing.assert_array_equal(
                gp._nuisance_design(frame, "plateID", nuisance), expected)

    infinite = outcomes.copy()
    infinite.loc[infinite.index[0], "cell_count"] = np.inf
    refusals = [
        ("absent block", outcomes, "no_such_block", [],
         "Block column 'no_such_block' is absent from outcomes"),
        ("infinite numeric nuisance", infinite, "plateID", ["cell_count"],
         "Nuisance column 'cell_count' must be finite"),
        ("nuisance repeating the block", with_site, "plateID", ["site"],
         "The nuisance fixed-effect design is rank deficient"),
    ]
    for case_id, frame, block, nuisance, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=message):
                gp._nuisance_design(frame, block, nuisance)


def _run(fractions, outcomes, outcome="pathogen_rate", **kwargs):
    kwargs.setdefault("n_permutations", 20)
    kwargs.setdefault("min_wells", 1)
    kwargs.setdefault("batch_size", 10)
    return gp.guide_freedman_lane_test(fractions, outcomes, outcome, **kwargs)


def test_guide_freedman_lane_test_refusals(subtests):
    """Every bad argument or value is refused by name before permuting."""
    fractions, outcomes, _meta = _prepared()
    negative = fractions.copy()
    negative.iloc[0, 0] = -1.0
    nan_outcome = outcomes.copy()
    nan_outcome.loc[nan_outcome.index[0], "pathogen_rate"] = np.nan
    flat_outcome = outcomes.assign(pathogen_rate=0.0)
    flat_guide = fractions.assign(g2=1e-300)
    renamed = outcomes.copy()
    renamed.index = [f"other_{name}" for name in renamed.index]

    cases = [
        ("unknown statistic", fractions, outcomes, {"statistic": "kendall"},
         r"statistic='kendall' must be 'pearson' or 'rank'"),
        ("zero permutations", fractions, outcomes, {"n_permutations": 0},
         "n_permutations must be at least 1"),
        ("zero batch", fractions, outcomes, {"batch_size": 0},
         "batch_size must be at least 1"),
        ("negative presence", fractions, outcomes, {"presence_threshold": -0.1},
         "presence_threshold must be non-negative"),
        ("absent outcome", fractions, outcomes, {"outcome": "missing_phenotype"},
         "Outcome column 'missing_phenotype' is absent"),
        ("wells that do not match", fractions, renamed, {},
         "Fraction and outcome well identifiers do not align"),
        ("negative fraction", negative, outcomes, {},
         "Guide fractions must be finite and non-negative"),
        ("unreachable support", fractions, outcomes, {"min_wells": 99},
         r"No guide occurs in at least 99 well\(s\) above presence_threshold=0.0"),
        ("non-finite outcome", fractions, nan_outcome, {},
         "Outcome 'pathogen_rate' must be finite"),
        ("outcome absorbed by the intercept", fractions, flat_outcome,
         {"nuisance_columns": []},
         "Outcome has no residual variation after nuisance adjustment"),
        ("guide with no residual variation is named", flat_guide, outcomes,
         {"nuisance_columns": []},
         r"Guide fractions have no residual variation .*\['g2'\]"),
    ]
    for case_id, frac, outs, kwargs, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=message):
                _run(frac, outs, **kwargs)


def test_guide_freedman_lane_test_results(subtests):
    """Effects are block-adjusted correlations; families, p-values and joins are exact."""
    fractions, outcomes, meta = _prepared()
    plates = outcomes["plateID"].to_numpy()
    y = _within_plate_centred(outcomes["pathogen_rate"], plates)
    expected = {}
    for guide in fractions.columns:
        x = _within_plate_centred(fractions[guide], plates)
        expected[guide] = float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

    result = _run(fractions, outcomes, min_wells=[1, 8], n_permutations=30,
                  batch_size=7)
    with subtests.test(msg="effect equals the within-plate correlation"):
        for _, row in result.iterrows():
            assert row["standardized_marginal_effect"] == pytest.approx(
                expected[row["guide"]], abs=1e-12)
    with subtests.test(msg="one family per threshold, every guide in both"):
        assert result["minimum_wells_threshold"].tolist() == [1] * 3 + [8] * 3
        assert result["tested_guides_in_family"].tolist() == [3] * 6
        assert set(result["guide"]) == {"g1", "g2", "g3"}
    with subtests.test(msg="p-value is (exceedances + 1) / (permutations + 1)"):
        np.testing.assert_allclose(
            result["permutation_p_value"],
            (result["permutation_exceedances"] + 1) / 31.0)
        assert result["permutations"].tolist() == [30] * 6
        assert result["permutation_exceedances"].between(0, 30).all()
    with subtests.test(msg="recorded settings"):
        assert set(result["statistic"]) == {"pearson"}
        assert set(result["nuisance_columns"]) == {""}
        assert set(result["block_column"]) == {"plateID"}
        assert set(result["multiple_testing_method"]) == {"fdr_bh"}
        assert set(result["alpha"]) == {0.05}
        assert result["wells_with_guide"].tolist() == [8] * 6
    with subtests.test(msg="sorted by family then p-value"):
        for _, family in result.groupby("minimum_wells_threshold"):
            assert family["permutation_p_value"].is_monotonic_increasing

    with subtests.test(msg="wells in another order are realigned"):
        straight = _run(fractions, outcomes)
        shuffled = _run(fractions, outcomes.iloc[::-1])
        pd.testing.assert_frame_equal(shuffled, straight)
    with subtests.test(msg="rank statistic correlates with the ranked outcome"):
        from scipy.stats import rankdata
        ranked = _run(fractions, outcomes, statistic="rank")
        ry = _within_plate_centred(rankdata(outcomes["pathogen_rate"]), plates)
        for _, row in ranked.iterrows():
            x = _within_plate_centred(fractions[row["guide"]], plates)
            assert row["standardized_marginal_effect"] == pytest.approx(
                float(x @ ry / np.linalg.norm(x) / np.linalg.norm(ry)), abs=1e-12)
            assert row["standardized_marginal_effect"] != pytest.approx(
                expected[row["guide"]], abs=1e-6)
        assert set(ranked["statistic"]) == {"rank"}
    with subtests.test(msg="metadata given as a column is joined by name"):
        metadata = meta.reset_index()
        metadata["gene"] = ["GENE_A", "GENE_B", "GENE_C"]
        joined = _run(fractions, outcomes, guide_metadata=metadata)
        assert dict(zip(joined["guide"], joined["gene"])) == {
            "g1": "GENE_A", "g2": "GENE_B", "g3": "GENE_C"}
        assert list(joined.columns).count("wells_with_guide") == 1
    with subtests.test(msg="presence threshold drops sparse guides"):
        sparse = fractions.copy()
        sparse.loc[sparse.index[:6], "g3"] = 0.0
        thinned = _run(sparse, outcomes, min_wells=3)
        assert sorted(thinned["guide"]) == ["g1", "g2"]
