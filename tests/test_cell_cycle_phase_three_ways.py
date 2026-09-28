"""Cell-cycle phase called three ways -- DNA gates, xgboost, torch -- into one column.

Nuclei of known phase are made two ways. A table of nuclei whose DNA content,
area and background are drawn per phase exercises the histogram fit, the
gates, the mitotic call, FUCCI and the tabular classifier directly. Fields of
nuclei drawn as ellipses of known DNA content (G1 at 2C, S between, G2 at 4C,
and M as a compact bright bar at 4C) go through Measure itself, so the phases
are called from the columns Measure really writes.

The torch route trains for a single epoch on the CPU here: that proves the
crops, the dataset, Classify's training and the scoring are wired, not that
the network is good -- its accuracy is measured separately and recorded in
features/future/535_cell_cycle_phase_classification_three_ways.txt.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from spacr import measure
from spacr.measure import (
    _CELL_CYCLE_TABLE,
    _CELL_CYCLE_WELL_TABLE,
    _cell_cycle_by_well,
    _cell_cycle_labels,
    _classify_cell_cycle,
    _consensus_phase,
    _fit_dna_content,
    _fucci_states,
    _gate_dna_content,
    _keep_content_calls,
    _normalise_phase,
    _phase_agreement_by_well,
    _phase_features,
    _phase_scores,
    _phases_by_measurements,
    _phases_by_xgboost,
    _resolve_cell_cycle_methods,
)

PHASES = ("G1", "S", "G2", "M")
UNIT = 1.0e5


def _nucleus_table(n=600, seed=0, fractions=(0.5, 0.25, 0.18, 0.07),
                   plates=("plate1",), background=100.0):
    """A nucleus table of known phases, in Measure's column names."""
    rng = np.random.default_rng(seed)
    phase = rng.choice(PHASES, size=n, p=fractions)
    dna = np.where(phase == "G1", rng.normal(1.0, 0.05, n),
                   np.where(phase == "S", rng.uniform(1.1, 1.9, n),
                            rng.normal(2.0, 0.06, n)))
    area = 300.0 * np.where(phase == "G1", 1.0, np.where(
        phase == "S", 1.0 + 0.6 * (dna - 1), np.where(
            phase == "G2", 1.6, 0.55))) * rng.normal(1.0, 0.1, n)
    content = dna * UNIT
    ring = rng.normal(background, 3.0, n)
    fields = rng.integers(1, 5, n)
    table = pd.DataFrame({
        "object_label": np.arange(1, n + 1),
        "plateID": rng.choice(list(plates), n),
        "rowID": "r1", "columnID": np.where(fields < 3, "c1", "c2"),
        "fieldID": [f"f{f}" for f in fields],
        "nucleus_area": area,
        "nucleus_eccentricity": np.where(phase == "M", 0.95, 0.4)
        + rng.normal(0, 0.03, n),
        "nucleus_channel_0_integrated_intensity": content + ring * area,
        "nucleus_channel_0_mean_intensity": content / area + ring,
        "nucleus_channel_0_outside_percentile_50": ring,
        "nucleus_channel_0_std_intensity": rng.normal(30, 3, n),
    })
    table["prcf"] = (table["plateID"] + "_" + table["rowID"] + "_"
                     + table["columnID"] + "_" + table["fieldID"])
    return table, pd.Series(phase, index=table.index)


def test_the_fit_finds_the_2n_and_4n_peaks_and_gates_between_them():
    table, truth = _nucleus_table()
    content = (table["nucleus_channel_0_integrated_intensity"]
               - table["nucleus_channel_0_outside_percentile_50"]
               * table["nucleus_area"])
    fit = _fit_dna_content(content)
    assert fit.g1 == pytest.approx(UNIT, rel=0.03)
    assert fit.g2 / fit.g1 == pytest.approx(2.0, abs=0.08)
    first, second = (float(fit.c_value(g)) for g in fit.gates)
    assert 2.0 < first < second < 4.0
    assert fit.fitted_gates
    phases, confidence, posterior = _gate_dna_content(content.to_numpy(), fit)
    assert set(phases) <= {"G1", "S", "G2"}
    assert np.all((confidence >= 0) & (confidence <= 1))
    assert np.allclose(posterior.sum(axis=1), 1.0)
    in_g1 = truth == "G1"
    assert (phases[in_g1.to_numpy()] == "G1").mean() > 0.95


def test_user_gates_replace_the_fitted_crossings_in_c_units():
    table, _ = _nucleus_table()
    content = table["nucleus_channel_0_integrated_intensity"]
    fitted = _fit_dna_content(content)
    fixed = _fit_dna_content(content, gates=[2.5, 3.5])
    assert not fixed.fitted_gates
    assert [float(fixed.c_value(g)) for g in fixed.gates] == pytest.approx(
        [2.5, 3.5])
    assert fixed.g1 == pytest.approx(fitted.g1)
    for bad in ([3.5, 2.5], [1], "x"):
        with pytest.raises(ValueError):
            _fit_dna_content(content, gates=bad)
    with pytest.raises(ValueError):
        _fit_dna_content(content[:10])


def test_content_far_outside_the_peaks_is_sub_g1_or_above_4n():
    table, _ = _nucleus_table()
    content = (table["nucleus_channel_0_integrated_intensity"]
               - table["nucleus_channel_0_outside_percentile_50"]
               * table["nucleus_area"]).to_numpy()
    fit = _fit_dna_content(content)
    phases, confidence, _ = _gate_dna_content(
        np.array([0.2 * UNIT, 5.0 * UNIT, np.nan]), fit)
    assert list(phases) == ["subG1", ">4N", None]
    assert confidence[0] == 1.0 and np.isnan(confidence[2])


def test_the_measurement_route_calls_all_four_phases_per_plate():
    table, truth = _nucleus_table(n=900, plates=("plate1", "plate2"))
    measured, fits = _phases_by_measurements(table, column=0)
    assert set(fits) == {("plate1",), ("plate2",)}
    scores = _phase_scores(truth, measured["phase_measurements"])
    assert scores["accuracy"] >= 0.93, scores
    assert scores["f1"]["M"] >= 0.9, scores
    by_truth = measured["dna_c"].groupby(truth).median()
    assert by_truth["G1"] == pytest.approx(2.0, abs=0.1)
    assert by_truth["G2"] == pytest.approx(4.0, abs=0.3)
    without_m, _ = _phases_by_measurements(table, column=0,
                                           mitotic_ratio=None)
    assert "M" not in set(without_m["phase_measurements"])


def test_a_plate_too_small_to_fit_borrows_the_pooled_fit():
    table, truth = _nucleus_table(n=500)
    table.loc[table.index[:12], "plateID"] = "tiny"
    measured, fits = _phases_by_measurements(table, column=0)
    assert fits[("tiny",)] is not fits[("plate1",)]
    assert measured.loc[table.index[:12], "phase_measurements"].notna().all()


def test_the_nucleus_table_must_carry_the_dna_channel():
    table, _ = _nucleus_table(n=60)
    with pytest.raises(ValueError, match="nucleus_channel_3_integrated"):
        _phases_by_measurements(table, column=3)


def test_fucci_states_come_from_two_reporters():
    table, _ = _nucleus_table(n=200)
    rng = np.random.default_rng(1)
    red = rng.random(200) < 0.5
    green = rng.random(200) < 0.5
    for channel, on in ((1, red), (2, green)):
        table[f"nucleus_channel_{channel}_integrated_intensity"] = 1.0
        table[f"nucleus_channel_{channel}_mean_intensity"] = np.where(
            on, 900.0, 110.0) + rng.normal(0, 5, 200)
        table[f"nucleus_channel_{channel}_outside_percentile_50"] = 100.0
    states = _fucci_states(table, {"channels": [0, 1, 2]}, [1, 2])
    expected = np.where(red & green, "G1/S", np.where(
        red, "G1", np.where(green, "S/G2/M", "early G1")))
    assert (states.to_numpy() == expected).all()
    with pytest.raises(ValueError):
        _fucci_states(table, {"channels": [0, 1, 2]}, [1])
    with pytest.raises(ValueError, match="not measured"):
        _fucci_states(table, {"channels": [0, 1]}, [1, 5])


def test_annotation_values_become_phases():
    assert [_normalise_phase(v) for v in (1, 2, 3, 4, 2.0, "3")] == [
        "G1", "S", "G2", "M", "S", "G2"]
    assert [_normalise_phase(v) for v in ("g2/m", "Mitotic", " s ")] == [
        "G2", "M", "S"]
    assert [_normalise_phase(v) for v in (0, 5, None, np.nan, "junk")] == [
        None] * 5


def test_xgboost_learns_the_phases_from_nucleus_features():
    table, truth = _nucleus_table(n=900)
    measured, _ = _phases_by_measurements(table, column=0)
    features = _phase_features(table, measured, [0])
    assert "nucleus_channel_0_mean_intensity" in features
    assert not any("centroid" in c or "distance" in c for c in features)
    labels = truth.where(np.random.default_rng(2).random(len(truth)) < 0.3)
    labels = labels.astype(object).where(labels.notna(), None)
    phases, confidence, report, _model = _phases_by_xgboost(
        features, labels, table["prcf"].to_numpy())
    scores = _phase_scores(truth[labels.isna()], phases[labels.isna()])
    assert scores["accuracy"] >= 0.93, scores
    assert report["classes"] == list(PHASES)
    assert report["held_out"]["n"] > 0
    assert json.loads(_model.get_booster().save_config())["learner"][
        "generic_param"]["device"] == "cpu"
    assert np.all((confidence > 0) & (confidence <= 1))
    one = pd.Series(["G1"] * len(truth), dtype=object)
    with pytest.raises(ValueError, match="at least two"):
        _phases_by_xgboost(features, one, table["prcf"].to_numpy())


def test_labels_come_from_annotate_through_the_cell_or_from_the_gates(
        tmp_path):
    table, truth = _nucleus_table(n=120)
    table["cell_id"] = table["object_label"] + 1000
    measured, _ = _phases_by_measurements(table, column=0)
    weak, source = _cell_cycle_labels(None, table, measured, "")
    assert source == "gates"
    assert weak.notna().sum() > 0.6 * len(table)
    assert set(weak.dropna()) <= set(PHASES)

    db = tmp_path / "measurements.db"
    crops = pd.DataFrame({
        "prcfo": [f"{p}_o{c}" for p, c in zip(table["prcf"][:40],
                                               table["cell_id"][:40])],
        "phase_annotation": [PHASES.index(t) + 1 for t in truth[:40]],
    })
    with sqlite3.connect(db) as conn:
        crops.to_sql("png_list", conn, index=False)
    labels, source = _cell_cycle_labels(str(db), table, measured,
                                        "phase_annotation")
    assert source == "annotation:phase_annotation"
    assert list(labels[:40]) == list(truth[:40])
    assert labels[40:].isna().all()
    with pytest.raises(ValueError, match="png_list"):
        _cell_cycle_labels(str(db), table, measured, "no_such_column")


def test_per_well_fractions_split_by_infection_and_the_consensus():
    frame = pd.DataFrame({
        "plateID": ["p"] * 6, "rowID": ["r1"] * 6,
        "columnID": ["c1"] * 3 + ["c2"] * 3,
        "phase_measurements": ["G1", "S", "G2", "G1", "M", "subG1"],
        "phase_xgboost": ["G1", "G1", "G2", "G1", "M", "subG1"],
        "phase_torch": ["S", "G1", "G2", "G1", "G2", "subG1"],
        "infected": [1.0, 0.0, 1.0, np.nan, 0.0, 1.0],
    })
    methods = ("measurements", "xgboost", "torch")
    frame["cell_cycle_phase"] = _consensus_phase(frame, methods)
    assert list(frame["cell_cycle_phase"]) == [
        "G1", "G1", "G2", "G1", "M", "subG1"]
    wells = _cell_cycle_by_well(frame, list(methods) + ["consensus"])
    first = wells[(wells["columnID"] == "c1")
                  & (wells["method"] == "measurements")].iloc[0]
    assert first["n"] == 3
    assert first["fraction_G1"] == pytest.approx(1 / 3)
    assert first["fraction_G2M"] == pytest.approx(1 / 3)
    assert first["infected_n"] == 2 and first["uninfected_n"] == 1
    assert first["infected_fraction_G2"] == pytest.approx(0.5)
    assert set(wells["method"]) == {*methods, "consensus"}


def test_methods_are_compared_on_the_nuclei_they_all_called():
    frame = pd.DataFrame({
        "plateID": ["p"] * 4, "rowID": ["r1"] * 4, "columnID": ["c1"] * 4,
        "phase_measurements": ["G1", "G1", "S", "G2"],
        "phase_torch": ["G1", "G1", None, None],
    })
    agreement = _phase_agreement_by_well(frame, ("measurements", "torch"))
    row = agreement.iloc[0]
    assert row["n"] == 2 and row["max_fraction_difference"] == 0.0
    frame.loc[1, "phase_torch"] = "S"
    row = _phase_agreement_by_well(frame, ("measurements", "torch")).iloc[0]
    assert row["max_fraction_difference"] == pytest.approx(0.5)
    assert _phase_agreement_by_well(frame, ("measurements",)).empty


def test_out_of_peak_nuclei_keep_their_content_call_only_when_called():
    measured = pd.DataFrame({"phase_measurements": ["G1", "subG1", ">4N",
                                                    "subG1"]})
    learned = _keep_content_calls(["S", "G1", "G2", None], measured)
    assert list(learned) == ["S", "subG1", ">4N", None]


def test_the_method_setting_names_what_runs():
    assert _resolve_cell_cycle_methods("measurements") == ("measurements",)
    assert _resolve_cell_cycle_methods("xgboost") == ("measurements",
                                                      "xgboost")
    assert _resolve_cell_cycle_methods("ALL") == ("measurements", "xgboost",
                                                  "torch")
    with pytest.raises(ValueError, match="cell_cycle_method"):
        _resolve_cell_cycle_methods("svm")


def _field(rng, n=26, size=256, fractions=(0.45, 0.25, 0.2, 0.1)):
    """One field of nuclei: an intensity plane, their labels and phases."""
    image = np.zeros((size, size))
    labels = np.zeros((size, size), np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = {}
    for phase in rng.choice(PHASES, n, p=fractions):
        dna = {"G1": rng.normal(1, 0.04), "S": rng.uniform(1.15, 1.85),
               "G2": rng.normal(2, 0.05), "M": rng.normal(2, 0.05)}[phase]
        area = 200.0 * {"G1": 1.0, "S": 1 + 0.6 * (dna - 1), "G2": 1.6,
                        "M": 0.55}[phase]
        ratio = 2.8 if phase == "M" else 1.2
        a, b = np.sqrt(area * ratio / np.pi), np.sqrt(area / (ratio * np.pi))
        for _ in range(80):
            cy, cx = rng.uniform(a + 4, size - a - 4, 2)
            angle = rng.uniform(0, np.pi)
            u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            inside = (u / a) ** 2 + (v / b) ** 2 <= 1
            if not labels[ndi.binary_dilation(inside, iterations=3)].any():
                break
        else:
            continue
        label = int(labels.max()) + 1
        labels[inside] = label
        image[inside] += dna * 400.0 * 200.0 / inside.sum()
        truth[label] = phase
    image = ndi.gaussian_filter(image, 0.7) + 120.0
    image = rng.poisson(image).astype(np.uint16)
    return image, labels, truth


def _write_plate(root, fields=6, seed=0):
    merged = root / "merged"
    merged.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    truths = {}
    for number in range(1, fields + 1):
        well = "plate1_A01" if number <= fields // 2 else "plate1_B02"
        image, labels, truth = _field(rng)
        name = f"{well}_{number}"
        np.save(merged / f"{name}.npy",
                np.stack([image, labels.astype(np.uint16)], axis=-1))
        truths[name] = truth
    return merged, truths


def _measure_settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0], "nucleus_channel": 0,
        "cell_mask_dim": None, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1,
        "cell_cycle": True, "cell_cycle_method": "xgboost",
    })
    settings.update(over)
    return settings


def _truth_for(table, truths):
    return [truths[f].get(int(o)) for f, o in
            zip(table["file_name"], table["object_label"])]


@pytest.mark.integration
def test_measure_writes_the_phase_column_and_the_per_well_fractions(
        tmp_path):
    merged, truths = _write_plate(tmp_path)
    measure.measure_crop(_measure_settings(merged, plot=True))

    db = tmp_path / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        table = pd.read_sql_query(f"SELECT * FROM {_CELL_CYCLE_TABLE}", conn)
        wells = pd.read_sql_query(f"SELECT * FROM {_CELL_CYCLE_WELL_TABLE}",
                                  conn)
    assert len(table) == sum(len(t) for t in truths.values())
    assert table["prcfo"].str.match(r"^plate1_r\d+_c\d+_f\d+_o\d+$").all()
    assert (table["cell_cycle_method"] == "xgboost").all()
    assert table["cell_cycle_phase"].equals(table["phase_xgboost"])
    truth = _truth_for(table, truths)
    for method in ("measurements", "xgboost"):
        scores = _phase_scores(truth, table[f"phase_{method}"])
        assert scores["accuracy"] >= 0.88, (method, scores)
    assert set(wells["method"]) == {"measurements", "xgboost"}
    assert len(wells) == 4
    assert (wells[[f"fraction_{p}" for p in PHASES]].sum(axis=1)
            <= 1.0 + 1e-9).all()
    assert wells["fit_fraction_S"].notna().sum() == 2
    assert (tmp_path / "measurements" / "cell_cycle_xgboost.json").is_file()
    figures = list((tmp_path / "results" / "cell_cycle").glob(
        "dna_content_plate1.*"))
    assert figures, list((tmp_path / "results").rglob("*"))


def test_a_failed_step_is_reported_and_does_not_fail_the_run(tmp_path,
                                                             capsys):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame({"object_label": [1]}).to_sql("cell", conn, index=False)
    assert measure._run_cell_cycle_step(str(db), {"cell_cycle": True}) is None
    assert "no nucleus table" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.slow
def test_the_torch_route_trains_with_classify_and_reuses_its_model(
        tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    merged, truths = _write_plate(tmp_path, fields=4, seed=3)
    measure.measure_crop(_measure_settings(
        merged, cell_cycle=False))
    db = str(tmp_path / "measurements" / "measurements.db")
    settings = _measure_settings(merged, cell_cycle_method="all",
                                 cell_cycle_epochs=1)
    table, report = _classify_cell_cycle(db, settings)

    assert report["torch"]["crops"] == len(table)
    assert report["torch"]["crop_size"] in range(32, 129, 8)
    model = report["torch"]["model"]
    order = json.loads((Path(model).parent / "cell_cycle_phases.json")
                       .read_text())
    assert order["phases"] == sorted(order["phases"])
    assert table["phase_torch"].notna().all()
    assert set(table["phase_torch"]) <= set(PHASES) | {"subG1", ">4N"}
    assert (table["cell_cycle_method"] == "consensus").all()
    crops = sorted(os.listdir(tmp_path / "cell_cycle" / "crops"))
    assert len(crops) == len(table)

    reused, again = _classify_cell_cycle(db, {
        **settings, "cell_cycle_method": "torch", "cell_cycle_model": model})
    assert again["torch"]["model"] == model
    assert reused["phase_torch"].notna().all()
    with sqlite3.connect(db) as conn:
        agreement = pd.read_sql_query("SELECT * FROM cell_cycle_agreement",
                                      conn)
    assert "max_fraction_difference" in agreement


def test_a_model_without_its_phase_order_is_refused(tmp_path):
    merged, _ = _write_plate(tmp_path, fields=2, seed=5)
    measure.measure_crop(_measure_settings(merged, cell_cycle=False))
    db = str(tmp_path / "measurements" / "measurements.db")
    stray = tmp_path / "stray.pth"
    stray.write_bytes(b"")
    with pytest.raises(ValueError, match="cell_cycle_phases.json"):
        _classify_cell_cycle(db, _measure_settings(
            merged, cell_cycle_method="torch", cell_cycle_model=str(stray)))


def test_the_settings_are_measure_defaults_and_alpha_registered():
    from spacr.settings import (ALPHA_FEATURES, expected_types,
                                get_measure_crop_settings, tooltips)

    defaults = get_measure_crop_settings({})
    keys = ALPHA_FEATURES[535]["settings"]
    assert {k for k in defaults if k.startswith("cell_cycle")} == set(keys)
    assert defaults["cell_cycle"] is False
    assert defaults["cell_cycle_method"] == "measurements"
    for key in keys:
        assert key in expected_types and key in tooltips, key
        assert len(tooltips[key]) <= 600, key
        assert re.search(r"Default [^.]+(\.\d+)?\.$", tooltips[key]), key


def test_the_real_plate1_nuclei_fit_a_g1_and_a_g2_peak(tmp_path):
    """The example plate's Hoechst nuclei: a 2N and a 4N peak, and S between."""
    source = Path(os.environ.get("SPACR_PLATE1_DB") or Path.home()
                  / ".cache/spacr/example_data/plate1/measurements"
                  / "measurements.db")
    if not source.is_file():
        pytest.skip("plate1 example data not downloaded")
    nuclei = pd.read_sql_query(
        "SELECT * FROM nucleus", sqlite3.connect(f"file:{source}?mode=ro",
                                                 uri=True))
    measured, fits = _phases_by_measurements(nuclei, column=0)
    fit = fits[("plate1",)]
    assert 1.8 <= fit.g2 / fit.g1 <= 2.2
    assert fit.weights[1] > 0.1
    shares = measured["phase_measurements"].value_counts(normalize=True)
    assert shares["G1"] > shares["G2"] > shares.get("M", 0)
