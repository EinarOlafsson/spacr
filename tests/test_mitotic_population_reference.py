"""Training uses the full population; prediction uses the current plate population."""
from __future__ import annotations

import copy
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from spacr import ops_mitotic as om


FEATURES = ["nucleus_dapi_mean_intensity", "cell_gfp_max_intensity"]


def _population(seed, plate):
    """Make three complete wells with exactly 3% mitosis and distinct staining gains."""
    rng = np.random.default_rng(seed)
    frames = []
    for position, well in enumerate(("A01", "A02", "B01")):
        truth = np.zeros(1000, dtype=bool)
        truth[rng.choice(1000, 30, replace=False)] = True
        gain = 1.0 + 0.18 * position
        control = np.arange(1000) % 4 == 0
        frames.append(pd.DataFrame({
            "plate": plate, "well": well,
            "site": np.arange(1000) // 250,
            "nucleus_dapi_mean_intensity": gain * rng.normal(
                1000 + 650 * truth, 85, 1000),
            "cell_gfp_max_intensity": gain * rng.normal(
                300 + 240 * truth, 35, 1000),
            "gene": np.where(control, "nontargeting", "GENE1"),
            "sgRNA": [f"nt{i % 3}" if control[i] else f"gene{i % 2}" for i in range(1000)],
            "mitotic": truth, "truth": truth,
        }, index=[f"{plate}:{well}:{i}" for i in range(1000)]))
    return pd.concat(frames)


@pytest.fixture(scope="module")
def populations():
    """Reuse well names on a second plate with strongly shifted DNA and tubulin signals."""
    training = _population(941, "training-plate")
    labelled = []
    for _, well in training.groupby("well", sort=False):
        positive = well[well.truth]
        labelled.extend([positive, well[~well.truth].sample(len(positive), random_state=284)])
    query = _population(1187, "another-plate")
    for position, well in enumerate(("A01", "A02", "B01")):
        selected = query.well == well
        query.loc[selected, FEATURES[0]] = (
            query.loc[selected, FEATURES[0]] * (4 + position) + 1300)
        query.loc[selected, FEATURES[1]] = (
            query.loc[selected, FEATURES[1]] * (3 + position) + 450)
    return training, pd.concat(labelled).sample(frac=1, random_state=712), query


def _train(reference, labelled, **kwargs):
    """Train on enriched labels with an explicit full-well reference and fixed folds."""
    return om.train_mitotic_classifier(
        labelled, labelled.truth, FEATURES, groups=labelled.well,
        reference_cells=reference, n_splits=3, seed=318,
        shuffled_control=False, **kwargs)


@pytest.fixture(scope="module")
def trained(populations):
    """Train once against the full original plate, capturing its enrichment warning."""
    reference, labelled, _ = populations
    before = reference.copy(deep=True), labelled.copy(deep=True)
    with pytest.warns(UserWarning, match="enriched") as caught:
        bundle = _train(reference, labelled, reference_labels=reference.truth)
    pd.testing.assert_frame_equal(reference, before[0])
    pd.testing.assert_frame_equal(labelled, before[1])
    return bundle, [str(item.message) for item in caught]


def _current_population_matrix(cells, reference):
    """Independently normalize ordered query rows against their matching full wells."""
    result = np.empty((len(cells), len(FEATURES)), dtype=np.float32)
    for well in cells.well.unique():
        population = reference.loc[reference.well == well, FEATURES].to_numpy(np.float64)
        median = np.median(population, axis=0)
        scale = 1.4826 * np.median(np.abs(population - median), axis=0)
        positions = np.flatnonzero(cells.well.to_numpy() == well)
        result[positions] = (cells.iloc[positions][FEATURES].to_numpy(np.float64) - median) / scale
    return result


def _expected_probabilities(bundle, cells, reference):
    """Predict from independently computed current-population normalization."""
    probability = bundle["model"].predict_proba(_current_population_matrix(cells, reference))[:, 1]
    return pd.Series(probability, index=cells.index, name="mitotic_probability")


def _assert_inspection_statistics(actual, expected):
    """Check the saved diagnostic statistics without making them a prediction input."""
    assert actual.keys() == expected.keys()
    assert actual["schema"] == expected["schema"] == 1
    assert actual["group"] == expected["group"]
    assert actual["features"] == expected["features"]
    assert actual["reference_cells"] == expected["reference_cells"]
    assert actual["groups"].keys() == expected["groups"].keys()
    for group, stats in actual["groups"].items():
        assert stats["cells"] == expected["groups"][group]["cells"]
        np.testing.assert_array_equal(stats["median"], expected["groups"][group]["median"])
        np.testing.assert_array_equal(stats["scale"], expected["groups"][group]["scale"])


def test_enriched_labels_predict_the_three_percent_population(populations, trained):
    """A 50% annotation sample does not inflate the mitotic fraction of the whole screen."""
    reference, labelled, _ = populations
    bundle, _ = trained
    assert labelled.truth.mean() == pytest.approx(0.50)
    assert reference.truth.mean() == pytest.approx(0.03)
    called = om.predict_mitotic(bundle, reference)
    assert abs(float(called.mean()) - 0.03) <= 0.015
    assert called[reference.truth].mean() > 0.90
    assert called[~reference.truth].mean() < 0.01


def test_reused_well_names_on_a_brighter_plate_use_the_current_population(populations, trained):
    """Another A01 must use its own robust location and scale, not the training A01."""
    reference, _, query = populations
    bundle, _ = trained
    assert set(reference.well) == set(query.well)
    assert set(reference.plate).isdisjoint(query.plate)
    before = query.copy(deep=True)
    expected = _expected_probabilities(bundle, query, query)
    actual = om.predict_mitotic(bundle, query, probability=True)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert abs(float((actual >= bundle["threshold"]).mean()) - 0.03) <= 0.015
    wrong_population = _expected_probabilities(bundle, query, reference)
    assert float((wrong_population >= bundle["threshold"]).mean()) > 0.50
    pd.testing.assert_frame_equal(query, before)


def test_explicit_current_reference_overrides_matching_training_wells_for_subsets(populations, trained):
    """Positive-only, reordered and singleton queries use the supplied complete new plate."""
    _, _, query = populations
    bundle, _ = trained
    expected = _expected_probabilities(bundle, query, query)
    subsets = [query[query.truth], query.iloc[[0]],
               query.sample(83, random_state=51), query.iloc[::-1]]
    for subset in subsets:
        actual = om.predict_mitotic(bundle, subset, probability=True, reference_cells=query)
        pd.testing.assert_series_equal(actual, expected.loc[subset.index], check_exact=True)
    parts = [om.predict_mitotic(bundle, query.iloc[start:start + 173], probability=True,
                                reference_cells=query)
             for start in range(0, len(query), 173)]
    pd.testing.assert_series_equal(pd.concat(parts), expected, check_exact=True)


def test_nonfinite_values_have_identical_current_and_explicit_population_normalization(
        populations, trained, monkeypatch):
    """Infinity and NaN remain missing observations through both full-well prediction paths."""
    _, _, population = populations
    bundle, _ = trained
    query = population.copy(deep=True)
    query.loc[query.index[[0, 1, 2]], FEATURES[0]] = [np.inf, -np.inf, np.nan]
    query.loc[query.index[[1000, 1001, 1002]], FEATURES[1]] = [-np.inf, np.nan, np.inf]
    before = query.copy(deep=True)
    matrices = []
    original_predict = bundle["model"].predict_proba

    def capture(values):
        """Record the actual normalized model input while retaining real classifier predictions."""
        matrices.append(values.copy())
        return original_predict(values)

    monkeypatch.setattr(bundle["model"], "predict_proba", capture)
    actual = om.predict_mitotic(bundle, query, probability=True)
    explicit = om.predict_mitotic(bundle, query, probability=True, reference_cells=query)
    cleaned = query.replace([np.inf, -np.inf], np.nan)
    expected = om.predict_mitotic(bundle, cleaned, probability=True)
    np.testing.assert_array_equal(matrices[0], matrices[1])
    np.testing.assert_array_equal(matrices[0], matrices[2])
    assert not np.isinf(matrices[0]).any()
    assert np.isnan(matrices[0]).sum() == 6
    assert np.isfinite(actual).all()
    pd.testing.assert_series_equal(actual, explicit, check_exact=True)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(query, before)


def test_categorical_missing_groups_keep_all_rows_in_population_normalization(
        populations, trained, monkeypatch):
    """Missing categorical wells form one complete group and unused categories add no rows."""
    _, _, population = populations
    bundle, _ = trained
    query = population.copy(deep=True)
    query["well"] = pd.Categorical(query.well, categories=["A01", "A02", "B01", "unused"])
    query.loc[query.well == "A02", "well"] = np.nan
    before = query.copy(deep=True)
    independently_grouped = query.copy(deep=True)
    independently_grouped["well"] = independently_grouped.well.astype(object).fillna("missing-well")
    expected_matrix = _current_population_matrix(independently_grouped, independently_grouped)
    expected = pd.Series(bundle["model"].predict_proba(expected_matrix)[:, 1],
                         index=query.index, name="mitotic_probability")
    matrices = []
    original_predict = bundle["model"].predict_proba

    def capture(values):
        """Record every prediction input to detect rows left outside categorical groups."""
        matrices.append(values.copy())
        return original_predict(values)

    monkeypatch.setattr(bundle["model"], "predict_proba", capture)
    actual = om.predict_mitotic(bundle, query, probability=True)
    explicit = om.predict_mitotic(bundle, query, probability=True, reference_cells=query)
    subset = query[query.well.isna()].iloc[::-7]
    subset_result = om.predict_mitotic(bundle, subset, probability=True, reference_cells=query)
    np.testing.assert_array_equal(matrices[0], expected_matrix)
    np.testing.assert_array_equal(matrices[1], expected_matrix)
    positions = query.index.get_indexer(subset.index)
    np.testing.assert_array_equal(matrices[2], expected_matrix[positions])
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_series_equal(explicit, expected, check_exact=True)
    pd.testing.assert_series_equal(subset_result, expected.loc[subset.index], check_exact=True)
    pd.testing.assert_frame_equal(query, before)


def test_stored_training_statistics_are_inspection_only(populations, trained):
    """Changing diagnostic training statistics cannot change current-population predictions."""
    _, _, query = populations
    bundle, _ = trained
    altered = copy.deepcopy(bundle["normalization_reference"])
    for stats in altered["groups"].values():
        stats["median"] += 1000000
        stats["scale"] *= 100
    inspected = dict(bundle, normalization_reference=altered)
    expected = _expected_probabilities(bundle, query, query)
    pd.testing.assert_series_equal(
        om.predict_mitotic(inspected, query, probability=True), expected, check_exact=True)
    subset = query[query.truth]
    pd.testing.assert_series_equal(
        om.predict_mitotic(inspected, subset, probability=True, reference_cells=query),
        expected.loc[subset.index], check_exact=True)


def test_training_statistics_describe_all_reference_cells(populations, trained):
    """The diagnostic medians and MADs describe full wells rather than enriched labels."""
    population, _, _ = populations
    bundle, _ = trained
    reference = bundle["normalization_reference"]
    assert reference["schema"] == 1
    assert reference["group"] == "well"
    assert reference["features"] == FEATURES
    assert reference["reference_cells"] == len(population)
    assert set(reference["groups"]) == set(population.well)
    for well, frame in population.groupby("well", sort=False):
        values = frame[FEATURES].to_numpy(np.float64)
        median = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - median), axis=0)
        stats = reference["groups"][well]
        assert stats["cells"] == len(frame)
        np.testing.assert_allclose(stats["median"], median, rtol=0, atol=1e-12)
        np.testing.assert_allclose(stats["scale"], scale, rtol=0, atol=1e-12)


def test_new_bundles_preserve_inspection_statistics_without_changing_prediction(tmp_path, populations, trained):
    """Serialization retains format 1 and its optional full-population diagnostic reference."""
    _, _, query = populations
    bundle, _ = trained
    assert bundle["format"] == "spacr-ops-mitotic-classifier-1"
    path = om.save_mitotic_classifier(bundle, str(tmp_path / "current.joblib"))
    restored = om.load_mitotic_classifier(path)
    _assert_inspection_statistics(restored["normalization_reference"],
                                  bundle["normalization_reference"])
    pd.testing.assert_series_equal(
        om.predict_mitotic(restored, query, probability=True),
        _expected_probabilities(bundle, query, query), check_exact=True)


def test_legacy_bundle_loads_with_a_warning_and_predicts_current_wells(tmp_path, populations, trained):
    """Pre-reference format-1 bundles remain usable through their original prediction contract."""
    _, _, query = populations
    bundle, _ = trained
    legacy = dict(bundle)
    legacy.pop("normalization_reference")
    legacy["folds"] = legacy["folds"].to_dict(orient="records")
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    with pytest.warns(UserWarning, match="reference"):
        restored = om.load_mitotic_classifier(str(path))
    assert restored["format"] == "spacr-ops-mitotic-classifier-1"
    pd.testing.assert_series_equal(
        om.predict_mitotic(restored, query, probability=True),
        _expected_probabilities(bundle, query, query), check_exact=True)


def test_training_without_explicit_reference_warns_and_keeps_full_population_contract(populations):
    """Existing full-population training calls still work and explain the enriched-label risk."""
    population, _, _ = populations
    with pytest.warns(UserWarning, match="reference"):
        bundle = om.train_mitotic_classifier(
            population, population.truth, FEATURES, groups=population.well,
            n_splits=3, shuffled_control=False, model="logistic")
    assert bundle["normalization_reference"]["reference_cells"] == len(population)
    called = om.predict_mitotic(bundle, population)
    assert abs(float(called.mean()) - 0.03) <= 0.015


def test_enrichment_warning_uses_supplied_population_truth(trained):
    """The comparison identifies a 50% label sample enriched over known 3% truth."""
    _, messages = trained
    assert any("enriched" in message and "reference" in message for message in messages)


def test_features_alone_do_not_invent_a_population_prevalence(populations):
    """Normalization data without truth cannot justify an enrichment claim."""
    population, labelled, _ = populations
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _train(population, labelled, model="logistic")
    assert not any("enriched" in str(item.message).lower() for item in caught)


def test_population_sampling_weights_can_support_the_enrichment_warning(populations):
    """Supplied post-stratification weights provide prevalence evidence without inferred labels."""
    population, labelled, _ = populations
    weights = om.sampling_weights(labelled.truth, {True: 0.03, False: 0.97})
    with pytest.warns(UserWarning, match="enriched"):
        bundle = _train(population, labelled, model="logistic", sample_weight=weights)
    assert bundle["summary"]["weighted"] is True


@pytest.mark.parametrize("operation", ["train", "predict"])
def test_explicit_reference_must_cover_every_requested_well(populations, trained, operation):
    """An incomplete explicit population cannot fall back to either biased rows or training wells."""
    population, labelled, query = populations
    bundle, _ = trained
    with pytest.raises(ValueError, match="reference"):
        if operation == "train":
            _train(population[population.well != "A01"], labelled)
        else:
            om.predict_mitotic(bundle, query[query.well == "A01"],
                               reference_cells=query[query.well != "A01"])


@pytest.mark.parametrize("state", ["mitotic_classifier", "interphase_classifier"])
def test_tvn_profiles_classify_the_whole_current_well_before_selecting_a_state(
        tmp_path, populations, trained, state):
    """The frozen TVN profile reader preserves current-well normalization with no API changes."""
    from spacr import ops_tvn
    from spacr.tabular import write_table

    _, _, population = populations
    bundle, _ = trained
    well = population[population.well == "A01"].copy()
    path = write_table(well, tmp_path / "raw_A01.parquet", canonicalise=False)
    expected = _expected_probabilities(bundle, well, well) >= bundle["threshold"]
    if state == "interphase_classifier":
        expected = ~expected
    recipe = ops_tvn.profile_recipe({"lof": 0, "impute": "median"})

    meta, values, columns = ops_tvn._profile_well(
        path, "another-plate", "A01", FEATURES, recipe, None, "nontargeting",
        state=state, classifier=bundle)

    assert columns == FEATURES
    np.testing.assert_array_equal(values, well.loc[expected, FEATURES].to_numpy(np.float32))
    assert meta.cell_barcode_0.tolist() == well.loc[expected, "sgRNA"].tolist()
    assert len(meta) == int(expected.sum())


def test_tvn_mitotic_index_uses_full_current_wells_without_new_caller_arguments(
        tmp_path, populations, trained):
    """The frozen index-output route counts each new plate well using its own population."""
    from spacr import ops_tvn
    from spacr.tabular import read_table, write_table

    _, _, population = populations
    bundle, _ = trained
    folder = tmp_path / "cells"
    folder.mkdir()
    for well, frame in population.groupby("well", sort=False):
        write_table(frame, folder / f"raw_{well}.parquet", canonicalise=False)
    output = tmp_path / "index"
    output.mkdir()
    settings = {"ops_profile_cells": str(folder), "plate": "another-plate",
                "ops_control_prefix": "nontargeting", "ops_mitotic_index_min_cells": 5}
    expected = population.assign(
        clf=(_expected_probabilities(bundle, population, population) >= bundle["threshold"]))
    control = expected.gene == "nontargeting"
    messages = []

    paths = ops_tvn._mitotic_index_outputs(settings, bundle, str(output), messages.append)

    genes = read_table(paths["mitotic_index_per_gene.csv"], canonicalise=False, report=None)
    controls = read_table(paths["mitotic_index_nt_guides.csv"], canonicalise=False, report=None)
    gene = genes.set_index("gene").loc["GENE1"]
    assert int(gene.cells) == int((~control).sum())
    assert int(gene.mitotic) == int(expected.loc[~control, "clf"].sum())
    assert int(controls.cells.sum()) == int(control.sum())
    assert int(controls.mitotic.sum()) == int(expected.loc[control, "clf"].sum())
    with open(paths["mitotic_index_summary.json"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["nt_rate"] == pytest.approx(float(expected.loc[control, "clf"].mean()))
    assert messages
