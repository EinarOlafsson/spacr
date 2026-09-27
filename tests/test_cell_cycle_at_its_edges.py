"""Cell-cycle phase calling on the inputs at its edges: histograms too flat
or too small to fit, tables without the columns a helper looks for, one
field only, FUCCI reporters that do not split, nuclei with no cell, and the
torch route with Classify's training and prediction stood in for so the
crop, dataset and model bookkeeping around them is what is exercised."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m
from spacr.tabular import read_table, write_database
from tests.test_cell_cycle_phase_three_ways import (_measure_settings,
                                                    _nucleus_table,
                                                    _write_plate)


def test_a_flat_or_tiny_histogram_is_seeded_and_refused_where_it_must_be():
    assert m._dna_seed(np.full(50, 8.0)) == pytest.approx(8.0)
    assert m._dna_g2_seed(np.array([1.0, 2.0, 3.0]), 1.0) == 2.0
    with pytest.raises(ValueError, match="needs at least"):
        m._fit_dna_content(np.linspace(1, 2, 10))
    table, _truth = _nucleus_table(n=400)
    dna = m._nucleus_dna(table, 0)["dna_content"]
    fit = m._fit_dna_content(dna, max_iter=1)
    assert fit.g2 > fit.g1 > 0


def test_the_channel_helpers_fall_back_to_the_measured_channels():
    assert m._cell_cycle_channel({"channels": [3, 1]}) == 3
    assert m._cell_cycle_channel({}) == 0
    assert m._measured_channel_column({}, 2) == 2
    with pytest.raises(ValueError, match="Channel 5 was not measured"):
        m._measured_channel_column({"channels": [0, 1]}, 5)


def test_dna_content_uses_whichever_background_ring_was_measured():
    table, _ = _nucleus_table(n=20)
    ring = table.pop("nucleus_channel_0_outside_percentile_50")
    bare = m._nucleus_dna(table, 0)
    assert np.allclose(bare["dna_content"],
                       table["nucleus_channel_0_integrated_intensity"])
    table["nucleus_channel_0_outside_mean"] = ring
    subtracted = m._nucleus_dna(table, 0)
    assert np.allclose(subtracted["dna_content"],
                       table["nucleus_channel_0_integrated_intensity"]
                       - ring * table["nucleus_area"])


def test_a_table_without_plates_is_fitted_whole_and_small_plates_share_a_fit():
    table, _ = _nucleus_table(n=300)
    groups = m._plate_groups(table.drop(columns="plateID"))
    assert groups[0][0] == "all" and len(groups[0][1]) == 300
    table["plateID"] = np.where(np.arange(300) % 2, "tiny_a", "tiny_b")
    table.loc[table.index[20:], "plateID"] = "big"
    measured, fits = m._phases_by_measurements(table, column=0)
    assert fits[("tiny_a",)] is fits[("tiny_b",)], (
        "two plates too small to fit borrow the same pooled fit")
    assert measured["phase_measurements"].notna().all()


def test_fucci_reporters_that_do_not_split_leave_the_plate_early_g1():
    table, _ = _nucleus_table(n=40)
    table["nucleus_channel_1_mean_intensity"] = 500.0
    table["nucleus_channel_1_integrated_intensity"] = 500.0
    states = m._fucci_states(table, {"channels": [0, 1]}, [1, 1])
    assert set(states) == {"early G1"}


def test_features_skip_other_channels_and_text_columns():
    table, _ = _nucleus_table(n=30)
    table["nucleus_channel_3_mean_intensity"] = 1.0
    table["nucleus_note"] = "x"
    measured, _ = m._phases_by_measurements(table, column=0)
    features = m._phase_features(table, measured, [0])
    assert "nucleus_channel_3_mean_intensity" not in features.columns
    assert "nucleus_note" not in features.columns
    assert "nucleus_channel_0_mean_intensity" in features.columns


def test_labels_scores_and_splits_on_degenerate_input():
    assert m._normalise_phase([1, 2]) is None
    scores = m._phase_scores([None, "x"], ["G1", "S"])
    assert scores["n"] == 0 and np.isnan(scores["accuracy"])
    assert not m._field_split(["f1", "f1", "f1"]).any()
    assert m._nucleus_crop_size(pd.DataFrame(
        {"nucleus_major_axis_length": [np.nan]})) == 64


def test_xgboost_on_one_field_trains_on_everything_it_has():
    pytest.importorskip("xgboost")
    table, truth = _nucleus_table(n=200)
    measured, _ = m._phases_by_measurements(table, column=0)
    features = m._phase_features(table, measured, [0])
    labels = pd.Series(truth.where(truth.isin(["G1", "G2"])), dtype=object)
    phases, _conf, _report, _model = m._phases_by_xgboost(
        features, labels, np.array(["f1"] * len(table)))
    called = pd.Series(phases)
    assert called.notna().all()
    assert set(called) <= {"G1", "G2"}


def test_consensus_and_well_fractions_of_nothing_called():
    frame = pd.DataFrame({"phase_measurements": [None],
                          "phase_xgboost": [np.nan]})
    assert list(m._consensus_phase(frame, ("measurements", "xgboost"))) == [
        None]
    assert m._cell_cycle_by_well(pd.DataFrame(), ["measurements"]).empty
    timed = pd.DataFrame({"plateID": ["p"] * 2, "rowID": ["r1"] * 2,
                          "columnID": ["c1"] * 2, "timeID": [1, 2],
                          "phase_measurements": ["G1", "S"]})
    wells = m._cell_cycle_by_well(timed, ["measurements", "torch"])
    assert sorted(wells["timeID"]) == [1, 2]
    assert set(wells["method"]) == {"measurements"}


def test_infection_is_unknown_where_the_pathogens_cannot_be_counted(
        tmp_path, monkeypatch):
    import spacr.infection as infection

    nuclei = pd.DataFrame({"plateID": ["p"] * 3, "rowID": ["r1"] * 3,
                           "columnID": ["c1"] * 3, "fieldID": ["f1"] * 3,
                           "cell_id": [1, np.nan, 9]})

    def broken(_db):
        raise OSError("database is locked")

    monkeypatch.setattr(infection, "parasites_per_cell", broken)
    assert m._nucleus_infection("x.db", nuclei).isna().all()
    monkeypatch.setattr(infection, "parasites_per_cell", lambda _db: pd.DataFrame({
        "plateID": ["p"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "object_label": [1], "pathogen_count": [2]}))
    assert m._nucleus_infection("x.db", nuclei).tolist()[0] == 1.0
    out = m._nucleus_infection("x.db", nuclei)
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])


def test_one_method_with_fucci_writes_no_agreement_table(tmp_path):
    table, _ = _nucleus_table(n=300)
    table["nucleus_channel_1_mean_intensity"] = np.where(
        np.arange(300) % 2, 50.0, 900.0)
    table["nucleus_channel_1_integrated_intensity"] = table[
        "nucleus_channel_1_mean_intensity"]
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    write_database(table, str(db), "nucleus", if_exists="replace")
    out, report = m._classify_cell_cycle(str(db), {
        "channels": [0, 1], "cell_cycle_channel": 0,
        "cell_cycle_method": "measurements",
        "cell_cycle_fucci_channels": [0, 1], "cell_cycle_mitotic_ratio": ""},
        plot=False)
    assert report["methods"] == ["measurements"]
    assert set(out["fucci_state"]) <= {"G1", "G1/S", "S/G2/M", "early G1"}
    assert "M" not in set(out["phase_measurements"])
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "cell_cycle_agreement" not in names and "cell_cycle" in names


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    root = tmp_path_factory.mktemp("cc_plate")
    merged, _truths = _write_plate(root, fields=2, seed=1)
    m.measure_crop(_measure_settings(merged, cell_cycle=False))
    db = root / "measurements" / "measurements.db"
    nuclei = read_table(str(db), table="nucleus", report=None).reset_index(
        drop=True)
    return root, merged, nuclei


def _stub_training(monkeypatch, *, product=".pth", drop_one=True):
    import spacr.deep_spacr as deep

    trained = {}

    def train(settings):
        trained.update(settings)
        if product is None:
            return None
        path = os.path.join(os.path.dirname(settings["src"]), "model" + product)
        open(path, "wb").close()
        return path

    def apply(crops, model_path, **kw):
        paths = sorted(os.path.join(crops, n) for n in os.listdir(crops))
        if drop_one:
            paths = paths[1:]
        return pd.DataFrame({"path": paths, "pred": [0.8] * len(paths)})

    monkeypatch.setattr(deep, "train_test_model", train)
    monkeypatch.setattr(deep, "apply_model", apply)
    return trained


def test_the_torch_route_splits_labelled_crops_and_reads_two_classes(
        plate, tmp_path, monkeypatch):
    root, _merged, nuclei = plate
    trained = _stub_training(monkeypatch)
    measured, _ = m._phases_by_measurements(nuclei, column=0)
    labels = measured["phase_measurements"].where(
        measured["phase_measurements"].isin(["G1", "G2"]))
    work = tmp_path / "work"
    (work / "dataset" / "stale").mkdir(parents=True)
    settings = _measure_settings(root / "merged", cell_cycle_epochs=1)
    phases, confidence, report = m._phases_by_torch(
        str(root), nuclei, measured, labels, settings, channel=0,
        work=str(work))
    assert trained["class_folder_names"] == ["G1", "G2"]
    assert not (work / "dataset" / "stale").exists()
    assert report["classes"] == ["G1", "G2"]
    assert report["crops"] == len(nuclei)
    assert phases.isna().sum() == 1, "the crop the model did not score"
    assert set(phases.dropna()) == {"G2"}
    assert confidence.dropna().round(6).eq(0.8).all()
    assert (work / "model.pth").is_file()


def test_the_torch_route_refuses_one_phase_or_a_run_without_a_model(
        plate, tmp_path, monkeypatch):
    root, _merged, nuclei = plate
    measured, _ = m._phases_by_measurements(nuclei, column=0)
    settings = _measure_settings(root / "merged")
    one = pd.Series("G1", index=nuclei.index, dtype=object)
    _stub_training(monkeypatch)
    with pytest.raises(ValueError, match="at least two phases"):
        m._phases_by_torch(str(root), nuclei, measured, one, settings,
                           channel=0, work=str(tmp_path / "a"))
    _stub_training(monkeypatch, product=None)
    two = measured["phase_measurements"].where(
        measured["phase_measurements"].isin(["G1", "G2"]))
    with pytest.raises(RuntimeError, match="produced no model"):
        m._phases_by_torch(str(root), nuclei, measured, two, settings,
                           channel=0, work=str(tmp_path / "b"))


def test_crops_skip_what_cannot_be_cut_and_name_the_time_point(
        plate, tmp_path, monkeypatch):
    root, merged, nuclei = plate
    measured, _ = m._phases_by_measurements(nuclei, column=0)
    extra = nuclei.iloc[:3].copy()
    extra["path_name"] = [str(tmp_path / "gone.npy"),
                          nuclei["path_name"].iloc[0],
                          nuclei["path_name"].iloc[0]]
    extra["object_label"] = [1, 0, 999]
    frame = pd.concat([nuclei, extra], ignore_index=True)
    frame["timeID"] = 3
    frame_measured = pd.concat([measured, measured.iloc[:3]],
                               ignore_index=True)
    stack_name = sorted(os.listdir(merged))[0]
    root2 = tmp_path / "root2"
    (root2 / "merged").mkdir(parents=True)
    image, labels = np.moveaxis(np.load(merged / stack_name), -1, 0)
    blank = np.zeros_like(image)
    six = np.stack([image, labels, blank, blank, blank, labels], axis=-1)
    np.save(root2 / "merged" / stack_name, np.stack([six, six]))
    on_stack = frame["path_name"].map(os.path.basename) == stack_name
    on_stack.iloc[-3] = True
    frame, frame_measured = frame[on_stack], frame_measured[on_stack]
    paths = m._write_nucleus_crops(str(root2), frame, frame_measured,
                                   {"nucleus_mask_dim": None}, channel=0,
                                   size=24, folder=str(tmp_path / "crops"))
    assert paths.iloc[-3:].isna().all()
    written = sorted(os.listdir(tmp_path / "crops"))
    assert written and all("_t3_o" in name for name in written)


def test_a_crop_is_copied_where_it_cannot_be_linked(tmp_path, monkeypatch):
    source = tmp_path / "a.png"
    source.write_bytes(b"png")

    def refuse(*_a):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    m._link_or_copy(str(source), str(tmp_path / "b.png"))
    assert (tmp_path / "b.png").read_bytes() == b"png"
