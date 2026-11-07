"""The trained mitotic classifier (item 856) on cells with known labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ops_mitotic as om


def _cells(seed=0, wells=("A1", "A2", "A3", "B1", "B2", "B3"), sites=4, n=150, rate=0.08):
    """Mitotic cells have bright, small, round nuclei and a tubulin spindle;
    each well has its own staining level, which the per-well robust z must
    take out."""
    rng = np.random.default_rng(seed)
    rows = []
    for w, well in enumerate(wells):
        gain = 1 + 0.3 * w
        for site in range(sites):
            m = rng.random(n) < rate
            rows.append(pd.DataFrame({
                "well": well, "site": site,
                "nucleus_area": rng.normal(200, 30, n) * np.where(m, 0.7, 1.0),
                "nucleus_eccentricity": np.clip(rng.normal(0.6, 0.1, n) - 0.25 * m, 0, 1),
                "nucleus_dapi_mean_intensity": gain * rng.normal(1000, 120, n) * np.where(m, 1.6, 1),
                "nucleus_dapi_std_intensity": gain * rng.normal(100, 20, n) * np.where(m, 1.8, 1),
                "cell_gfp_max_intensity": gain * rng.normal(300, 50, n) * np.where(m, 1.7, 1),
                "cell_gfp_mean_intensity": gain * rng.normal(80, 10, n),
                "cytoplasm_gfp_std_intensity": gain * rng.normal(20, 4, n) * np.where(m, 1.5, 1),
                "cell_a594_mean_intensity": rng.normal(50, 5, n),
                "nucleus_a594_mean_intensity": rng.normal(60, 5, n),
                "cell_area": rng.normal(800, 100, n),
                "label": m}))
    return pd.concat(rows, ignore_index=True)


def test_features_are_dna_tubulin_and_nuclear_shape_only():
    cells = _cells(n=5)
    features = om.mitotic_features(cells.columns)
    assert features == ["nucleus_area", "nucleus_eccentricity", "nucleus_dapi_mean_intensity",
                        "nucleus_dapi_std_intensity", "cell_gfp_max_intensity",
                        "cell_gfp_mean_intensity", "cytoplasm_gfp_std_intensity"]
    assert "cell_gfp_max_intensity" not in om.mitotic_features(cells.columns, tubulin=None)


def test_grouped_cross_validation_beats_its_shuffled_control():
    cells = _cells()
    out = om.train_mitotic_classifier(cells, cells.label, groups=cells.well, n_splits=3)
    s = out["summary"]
    assert s["balanced_accuracy"] > 0.9
    assert s["roc_auc"] > 0.95
    assert s["shuffled_roc_auc"] < 0.65
    assert s["shuffled_balanced_accuracy"] < 0.65
    assert s["cells"] == len(cells) and s["mitotic"] == int(cells.label.sum())
    folds = out["folds"]
    assert len(folds) == 3 and folds.test_groups.sum() == 6
    assert np.isfinite(out["out_of_fold"]).all()


def test_folds_never_split_a_group():
    from sklearn.model_selection import GroupKFold

    cells = _cells(n=40)
    groups = cells.well + "_" + cells.site.astype(str)
    for train, test in GroupKFold(5).split(cells, cells.label, groups):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
    out = om.train_mitotic_classifier(cells, cells.label, groups=groups, n_splits=5,
                                      shuffled_control=False)
    assert out["folds"].test_groups.sum() == groups.nunique()
    assert "shuffled_roc_auc" not in out["summary"]


def test_the_per_well_normalisation_lets_the_model_travel_to_a_brighter_well():
    train = _cells(wells=("A1", "A2", "A3"), seed=1)
    out = om.train_mitotic_classifier(train, train.label, groups=train.well, n_splits=3,
                                      shuffled_control=False)
    new = _cells(wells=("C1",), seed=2)
    bright = [c for c in new if c.endswith("intensity")]
    new[bright] *= 3.0
    called = om.predict_mitotic(out, new)
    from sklearn.metrics import balanced_accuracy_score
    assert balanced_accuracy_score(new.label, called) > 0.85
    p = om.predict_mitotic(out, new, probability=True)
    assert ((p >= 0) & (p <= 1)).all() and p.name == "mitotic_probability"


def test_save_and_load_give_the_same_calls(tmp_path):
    cells = _cells(n=60)
    out = om.train_mitotic_classifier(cells, cells.label, groups=cells.well, n_splits=3,
                                      shuffled_control=False)
    path = om.save_mitotic_classifier(out, str(tmp_path / "m.joblib"))
    back = om.load_mitotic_classifier(path)
    pd.testing.assert_series_equal(om.predict_mitotic(back, cells), om.predict_mitotic(out, cells))
    import joblib
    joblib.dump({"format": "other"}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="not a spaCR mitotic classifier"):
        om.load_mitotic_classifier(str(tmp_path / "x.joblib"))


def test_logistic_baseline_runs():
    cells = _cells(n=60)
    out = om.train_mitotic_classifier(cells, cells.label, groups=cells.well, n_splits=3,
                                      model="logistic", shuffled_control=False)
    assert out["summary"]["model"] == "logistic"


@pytest.mark.parametrize("kwargs, match", [
    ({"groups": None}, "groups is required"),
    ({"n_splits": 9}, "cannot make"),
    ({"model": "forest"}, "model must be"),
    ({"normalise_by": "plate"}, "normalise_by"),
    ({"features": []}, "no features"),
])
def test_bad_training_calls_are_refused(kwargs, match):
    cells = _cells(n=30)
    args = {"groups": cells.well, **kwargs}
    with pytest.raises(ValueError, match=match):
        om.train_mitotic_classifier(cells, cells.label, **args)


def test_one_class_labels_are_refused():
    cells = _cells(n=30)
    with pytest.raises(ValueError, match="both"):
        om.train_mitotic_classifier(cells, np.zeros(len(cells), bool), groups=cells.well)


def test_prediction_names_missing_columns():
    cells = _cells(n=60)
    out = om.train_mitotic_classifier(cells, cells.label, groups=cells.well, n_splits=3,
                                      shuffled_control=False)
    with pytest.raises(ValueError, match="nucleus_area"):
        om.predict_mitotic(out, cells.drop(columns=["nucleus_area"]))


def test_mitotic_index_agreement_has_a_shuffled_control():
    rng = np.random.default_rng(0)
    truth = rng.random(300) * 0.1
    genes = pd.DataFrame({"gene": [f"g{i}" for i in range(300)], "n_cells": 600,
                          "mitotic_index": truth + rng.normal(0, 0.01, 300)})
    reference = pd.DataFrame({"gene": genes.gene, "mitotic_index": truth})
    reference.loc[0, "mitotic_index"] = 7.0
    out = om.mitotic_index_agreement(genes, reference, min_cells=(50, 1000))
    assert list(out.min_cells) == [50]
    assert out.genes.iloc[0] == 299
    assert out.spearman.iloc[0] > 0.9 > 0.2 > out.shuffled_p95_abs.iloc[0]


def _label_files(tmp_path):
    labels = pd.DataFrame({"crop_id": ["c1", "c2", "c3", "c4", "c5"],
                           "class": ["interphase", "Prophase/Prometaphase", "metaphase",
                                     "anaphase_telophase", "artefact"]})
    mapping = pd.DataFrame({"crop_id": ["c1", "c2", "c3", "c4", "c5"],
                            "plate": "P", "well": ["A1", "A1", "A2", "A2", "B1"],
                            "site": [1, 2, 1, 3, 1], "label": [10, 11, 12, 13, 14]})
    labels.to_csv(tmp_path / "labels_final.csv", index=False)
    mapping.to_csv(tmp_path / "mapping_hidden.csv", index=False)
    return tmp_path / "labels_final.csv", tmp_path / "mapping_hidden.csv"


def test_label_table_joins_its_hidden_mapping_and_drops_artefacts(tmp_path):
    labels, mapping = _label_files(tmp_path)
    out = om.read_mitotic_labels(str(labels), str(mapping))
    assert list(out.columns) == ["plate", "well", "site", "label", "stage", "mitotic"]
    assert list(out.label) == [10, 11, 12, 13]
    assert list(out.mitotic) == [False, True, True, True]
    assert out.stage.iloc[1] == "prophase-prometaphase"


def test_label_table_with_cell_keys_needs_no_mapping():
    table = pd.DataFrame({"plate": "P", "well": "A1", "site": [1, 1], "label": [3, 4],
                          "stage": ["metaphase", "interphase"]})
    assert list(om.read_mitotic_labels(table).mitotic) == [True, False]


@pytest.mark.parametrize("change, match", [
    ({"class": ["interphase", "cytokinesis?", "metaphase", "metaphase", "artefact"]}, "unknown"),
    ({"crop_id": ["c1", "c2", "c3", "c4", "zz"]}, "no row in the mapping"),
])
def test_bad_label_tables_are_refused(tmp_path, change, match):
    labels, mapping = _label_files(tmp_path)
    table = pd.read_csv(labels)
    for k, v in change.items():
        table[k] = v
    with pytest.raises(ValueError, match=match):
        om.read_mitotic_labels(table, str(mapping))


def test_one_cell_with_two_classes_is_refused():
    table = pd.DataFrame({"plate": "P", "well": "A1", "site": [1, 1], "label": [3, 3],
                          "stage": ["metaphase", "interphase"]})
    with pytest.raises(ValueError, match="same cell"):
        om.read_mitotic_labels(table)


def test_the_aggregate_phase_calls_mitotic_cells_with_a_saved_model(tmp_path):
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings, _tables

    db, wells, _ = _plate(tmp_path, n=300)
    cells = pd.concat([pd.read_parquet(tmp_path / f"ops_phenotype_cells_{w}.parquet")
                       for w in wells], ignore_index=True)
    labels = cells["nucleus_dapi_mean_intensity"] > 800
    features = ["nucleus_area", "nucleus_dapi_mean_intensity"]
    out = om.train_mitotic_classifier(cells, labels, features, groups=cells.well + "_" +
                                      cells.site.astype(str), n_splits=3, shuffled_control=False)
    path = om.save_mitotic_classifier(out, str(tmp_path / "m.joblib"))
    report = ops_engine._ph_aggregate(db, PLATE, wells, _settings(
        tmp_path, ops_bootstrap=0, ops_mitotic_model=path), str(tmp_path))
    guides, _genes = _tables(report)
    assert report["mitotic_rule"]["classifier"] == path
    assert report["mitotic_rule"]["features"] == 2
    assert guides.mitotic_cells.sum() > 0
    bad = _settings(tmp_path, ops_bootstrap=0, ops_mitotic_model=path)
    out["features"] = ["nucleus_missing_column"]
    om.save_mitotic_classifier(out, str(tmp_path / "bad.joblib"))
    bad["ops_mitotic_model"] = str(tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="nucleus_missing_column"):
        ops_engine._ph_aggregate(db, PLATE, wells, bad, str(tmp_path))
