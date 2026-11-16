"""Item 963: Annotate picks the classifier and keeps per-class thresholds.

The synthetic set separates its two classes only along f0 - f1, a diagonal
boundary: f0 and f1 share a large common term, so either alone says little.
Logistic regression reads the difference directly; axis-aligned trees can
only approximate it. Auto must therefore pick logistic regression, by a
clear margin, over every tree model and both control rows. The XOR set
reverses the roles, so the pick follows the data and not a fixed order.
"""
import filecmp
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al

WELLS = [(r, c) for r in ("r1", "r2", "r3") for c in ("c1", "c2", "c3", "c4")]


def build(root, *, mode="diagonal", per_well=40, labelled_per_well=30,
          n_features=10, seed=0, images=False):
    """A measurements.db with a cell table and a labelled png_list."""
    rng = np.random.default_rng(seed)
    db = os.path.join(str(root), "measurements.db")
    cells, crops = [], []
    for row_id, column_id in WELLS:
        prcf = f"plate1_{row_id}_{column_id}_f1"
        offset = rng.normal(0, 0.3, n_features)
        for obj in range(1, per_well + 1):
            values = rng.normal(0, 1, n_features) + offset
            if mode == "diagonal":
                cls = 1 if obj % 2 else 2
                sign = -1.0 if cls == 1 else 1.0
                common = rng.normal(0, 5.0)
                values[0] = common + sign * 0.5 + rng.normal(0, 0.2)
                values[1] = common - sign * 0.5 + rng.normal(0, 0.2)
            else:
                values[0], values[1] = rng.uniform(-1, 1, 2)
                cls = 1 if values[0] * values[1] > 0 else 2
            cell = {"plateID": "plate1", "rowID": row_id,
                    "columnID": column_id, "fieldID": "f1", "prcf": prcf,
                    "object_label": obj}
            cell.update({f"cell_channel_1_feature_{j:02d}": float(v)
                         for j, v in enumerate(values)})
            cells.append(cell)
            name = f"plate1_{row_id}{column_id}_1_{obj}.png"
            crops.append({
                "png_path": os.path.join(str(root), "crops", name),
                "file_name": name, "plateID": "plate1", "rowID": row_id,
                "columnID": column_id, "fieldID": "f1",
                "prcfo": f"{prcf}_o{obj}", "cell_id": f"o{obj}",
                "annotate": cls if obj <= labelled_per_well else None,
                "_truth": cls})
    if images:
        from PIL import Image
        os.makedirs(os.path.join(str(root), "crops"), exist_ok=True)
        for crop in crops:
            shade = 60 if crop["_truth"] == 1 else 190
            Image.new("RGB", (24, 24), (shade, 80, 255 - shade)).save(
                crop["png_path"])
    con = sqlite3.connect(db)
    try:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        pd.DataFrame(crops).drop(columns=["_truth"]).to_sql(
            "png_list", con, index=False)
        con.commit()
    finally:
        con.close()
    return db


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def diagonal_auto(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagonal")
    db = build(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        result = al._run_classification({
            "db_path": db, "annotation_column": "annotate",
            "classifier_choice": "auto"})
    return db, result


def _rows(result):
    return {r["classifier"]: r for r in result["comparison"]["rows"]}


def test_auto_picks_the_clear_winner_over_every_candidate_and_control(
        diagonal_auto):
    _db, result = diagonal_auto
    assert result["classifier"] == "logistic_regression"
    rows = _rows(result)
    best = rows["logistic_regression"]["f1_macro"]
    others = [r["f1_macro"] for name, r in rows.items()
              if r["role"] == "candidate" and "f1_macro" in r
              and name != "logistic_regression"]
    assert len(others) == 5
    assert best - max(others) >= 0.05
    for name in (al._CONTROL_MAJORITY, al._CONTROL_SHUFFLED):
        assert rows[name]["role"] == "control"
        assert rows[name]["f1_macro"] <= 0.60
        assert best - rows[name]["f1_macro"] >= 0.30


def test_the_folds_never_put_one_well_on_both_sides(tmp_path):
    db = build(tmp_path)
    inputs = al._classify_inputs(db, "annotate")
    assert inputs["groups"] is not None
    pairs = al._classify_folds(inputs["y"], inputs["groups"])
    assert len(pairs) == 5
    groups = np.asarray(inputs["groups"])
    seen = np.zeros(len(groups), dtype=int)
    for train, held in pairs:
        assert not set(groups[train]) & set(groups[held])
        seen[held] += 1
    assert (seen == 1).all()


def test_resnet50_is_skipped_with_its_reason_when_no_gpu_is_free(
        diagonal_auto):
    _db, result = diagonal_auto
    row = _rows(result)["resnet50"]
    assert "f1_macro" not in row
    assert row["note"].startswith("skipped: no GPU visible")
    assert "resnet50" not in result["settings"]["auto_candidates"]


def test_the_comparison_and_the_choice_are_recorded(diagonal_auto):
    _db, result = diagonal_auto
    table = pd.read_csv(result["paths"]["comparison"])
    chosen = table[table["chosen"]]
    assert chosen["classifier"].tolist() == ["logistic_regression"]
    assert set(table["role"]) == {"candidate", "control"}
    assert table["split"].iloc[0] == "5-fold, grouped by well"
    from spacr.cli import load_settings_file
    saved = load_settings_file(result["paths"]["settings"])
    assert saved["classifier_choice"] == "auto"
    assert saved["classifier"] == "logistic_regression"


def test_auto_follows_the_data_on_an_xor_set(tmp_path):
    db = build(tmp_path, mode="xor")
    result = al._run_classification({
        "db_path": db, "annotation_column": "annotate",
        "classifier_choice": "auto"})
    rows = _rows(result)
    assert result["classifier"] != "logistic_regression"
    assert rows["logistic_regression"]["f1_macro"] < 0.65
    assert rows[result["classifier"]]["f1_macro"] >= 0.80


def test_a_threshold_leaves_crops_under_it_unassigned():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]])
    best, confidence, assigned = al._apply_class_thresholds(
        probs, [1, 2], {"1": 0.7, "2": 0.6})
    assert best.tolist() == [0, 0, 1, 1]
    assert confidence.tolist() == [0.9, 0.6, 0.7, 0.55]
    assert assigned == ["1", "", "2", ""]
    _b, _c, everyone = al._apply_class_thresholds(probs, [1, 2], None)
    assert everyone == ["1", "1", "2", "2"]


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_thresholds_round_trip_through_the_settings_file(tmp_path, suffix):
    db = build(tmp_path)
    first = al._run_classification({
        "db_path": db, "annotation_column": "annotate",
        "classifier_choice": "xgboost",
        "class_thresholds": {"1": 0.8, "2": 0.65}})
    table_path = first["paths"]["table"]
    kept = tmp_path / "first.csv"
    kept.write_bytes(open(table_path, "rb").read())
    frame = pd.read_csv(table_path, keep_default_na=False)
    assert (frame["class_thresholded"].astype(str) == "").any()
    assert (frame["class_thresholded"].astype(str) != "").any()
    os.remove(table_path)
    settings = os.path.splitext(first["paths"]["settings"])[0] + suffix
    again = al._rerun_classification(settings)
    assert again["settings"]["class_thresholds"] == {"1": 0.8, "2": 0.65}
    assert filecmp.cmp(str(kept), table_path, shallow=False)


def test_saving_one_class_threshold_changes_only_that_class(tmp_path):
    db = build(tmp_path)
    first = al._run_classification({
        "db_path": db, "annotation_column": "annotate",
        "classifier_choice": "xgboost"})
    before = pd.read_csv(first["paths"]["table"], keep_default_na=False,
                         dtype=str)
    al._save_class_thresholds(first["paths"]["settings"], {"2": 0.9})
    after = pd.read_csv(first["paths"]["table"], keep_default_na=False,
                        dtype=str)
    changed = before["class_thresholded"] != after["class_thresholded"]
    assert changed.any()
    assert set(before.loc[changed, "class_predicted"]) == {"2"}
    assert set(after.loc[changed, "class_thresholded"]) == {""}
    kept = tmp_path / "saved.csv"
    kept.write_bytes(open(first["paths"]["table"], "rb").read())
    al._rerun_classification(first["paths"]["settings"])
    assert filecmp.cmp(str(kept), first["paths"]["table"], shallow=False)


def test_resnet50_is_refused_by_the_tabular_run(tmp_path):
    db = build(tmp_path)
    with pytest.raises(ValueError, match="Classify \\(CV\\) screen"):
        al._run_classification({"db_path": db,
                                "annotation_column": "annotate",
                                "classifier_choice": "resnet50"})


def test_a_rerun_that_compared_resnet50_refuses_without_a_gpu(tmp_path):
    db = build(tmp_path)
    with pytest.raises(ValueError, match="no GPU is free"):
        al._run_classification({
            "db_path": db, "annotation_column": "annotate",
            "classifier_choice": "auto",
            "auto_candidates": ["xgboost", "resnet50"]})


def test_the_resnet50_candidate_is_scored_on_the_same_folds(tmp_path):
    db = build(tmp_path)
    inputs = al._classify_inputs(db, "annotate")
    image_x = np.column_stack([inputs["x"][:, 0] - inputs["x"][:, 1],
                               np.zeros(len(inputs["y"]))])
    comparison = al._compare_classifiers(
        inputs["x"], inputs["y"], inputs["groups"],
        candidates=("xgboost",), image_x=image_x)
    rows = _rows({"comparison": comparison})
    assert comparison["candidates"] == ["xgboost", "resnet50"]
    assert rows["resnet50"]["note"] == "ImageNet features, linear head"
    assert comparison["winner"] == "resnet50"


def test_existing_defaults_are_unchanged():
    import inspect
    assert inspect.signature(al.retrain_round).parameters[
        "model_type"].default == "logistic_regression"
    from spacr.qt.screens import annotate
    source = inspect.getsource(annotate.AnnotateScreen._start_suggest)
    assert '"model_type": "gradient_boosting"' in source
