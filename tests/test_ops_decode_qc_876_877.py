"""Items 876 and 877: the false-call model and the calibrated genotype confidence.

Both are pinned on synthetic calls whose decoys are told apart by their
evidence, so a model that learns anything scores them, and a shuffled-label
control does not. Validation against the decode-yield study's own outputs is
recorded in the items' feature files.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from spacr import ops_decode
from spacr.ops_store import read_table, write_table
from spacr.tabular import write_table as write_file
from spacr.settings import ALPHA_FEATURES


def _calls(rng, wells=("A1", "A2"), sites=10, per_site=150, decoy_rate=0.08):
    """Evidence rows: decoy calls carry fewer exact reads and lower quality."""
    rows = []
    object_id = 1
    for well in wells:
        for site in range(sites):
            for _ in range(per_site):
                decoy = rng.random() < decoy_rate
                q = float(np.clip(rng.normal(0.2 if decoy else 0.6, 0.15), 0, 1))
                exact = int(rng.integers(0, 2) if decoy else rng.integers(1, 5))
                pixel = rng.random() < 0.1
                rows.append({
                    "well": well, "site": site, "object_id": object_id,
                    "mapped_guide": "DECOY" if decoy else "GUIDE",
                    "is_decoy": decoy,
                    "source": "pixel_fallback" if pixel else "primary",
                    "n_reads": exact + 1, "n_matched": exact + 1,
                    "n_guides_seen": 1, "n_exact": exact, "n_corr": 1,
                    "vote_top": 1.0, "vote_total": 1.0,
                    "vote_frac": np.nan if pixel else 1.0,
                    "runner_margin": 1.0, "q_best": q, "q_mean": q,
                    "strength_max": 500.0, "n_amb": 0, "pix_n": 2 if pixel else 0,
                    "pix_total": 2 if pixel else 0, "pix_frac": 1.0 if pixel else 0.0,
                    "field_reads_per_nucleus": 2.0, "field_nuclei": per_site,
                })
                object_id += 1
    return pd.DataFrame(rows)


def test_the_confidence_learns_the_decoys_and_the_shuffled_control_does_not():
    calls = _calls(np.random.default_rng(1))
    out = ops_decode.genotype_confidence(calls, model="hist_gbm", seed=3,
                                         max_real=5000)
    metrics = out["metrics"]
    assert metrics["HGB"]["AUC_decoy"] > 0.8
    assert abs(metrics["control: shuffled decoy labels"]["AUC_decoy"] - 0.5) < 0.1
    assert metrics["chosen"] == "HGB"
    table = out["table"]
    assert len(table) == len(calls)
    real = table[~table["is_decoy"]]
    low = real["q_best"] < 0.3
    assert real.loc[low, "genotype_confidence"].mean() < \
        real.loc[~low, "genotype_confidence"].mean()
    assert ((table["genotype_confidence"] >= 0) & (table["genotype_confidence"] <= 1)).all()
    assert set(out["reliability"]["model"]) == {"LR", "ISO", "HGB", "STRATA"}
    assert len(out["reliability"]) == 40
    auto = ops_decode.genotype_confidence(calls, model="auto", seed=3,
                                          calibration="none", max_real=5000)
    assert auto["metrics"]["chosen"] == auto["metrics"]["best_by_logloss"]
    assert (auto["table"]["e"] == auto["table"][f"e_{auto['metrics']['chosen']}"]).all()


def test_the_confidence_refuses_what_it_cannot_fit():
    calls = _calls(np.random.default_rng(2), sites=5, per_site=20)
    with pytest.raises(ValueError, match="ops_confidence_model"):
        ops_decode.genotype_confidence(calls, model="forest")
    with pytest.raises(ValueError, match="ops_confidence_calibration"):
        ops_decode.genotype_confidence(calls, calibration="platt")
    with pytest.raises(ValueError, match="decoy and real"):
        ops_decode.genotype_confidence(calls.assign(is_decoy=False))


def _qc_field(rng, n_objects=40, cycles=6):
    """A cache-layout field with spot features: each object has reads of one
    guide, some objects' reads spell a decoy."""
    guides = ["GTACGT", "CATGCA", "TGCATG", "ACGTAC"]
    decoys = ["GGTTAC", "TTGGCA"]
    peaks, values, gap, ids, owned0, strength = [], [], [], [], [], []
    for obj in range(1, n_objects + 1):
        word = decoys[obj % 2] if obj % 5 == 0 else guides[obj % 4]
        for _ in range(3):
            v = np.full((cycles, 4), 10.0, np.float32)
            for c, letter in enumerate(word):
                v[c, "GTAC".index(letter)] = 1000.0 if obj % 5 else 300.0
            values.append(v + rng.normal(0, 5, v.shape).astype(np.float32))
            peaks.append(rng.integers(50, 1400, 2))
            gap.append([1.0, 20.0])
            ids.append(obj)
            owned0.append(True)
            strength.append(400.0)
    n = len(values)
    return guides, decoys, {
        "peaks": np.array(peaks, np.int32), "strength": np.array(strength, np.float32),
        "values": np.stack(values), "gap": np.array(gap, np.float32),
        "id0": np.array(ids), "owned0": np.array(owned0),
        "owned_ids": np.arange(1, n_objects + 1),
        "owned_areas": np.full(n_objects, 80.0),
        "f_raw": np.stack(values), "f_bg": np.stack(values) * 0 + 5,
        "f_sharp": rng.random(n).astype(np.float32),
        "f_focus": rng.random(n).astype(np.float32),
        "f_density": rng.integers(0, 5, n).astype(np.int16)}


def _qc_fields(n_fields=8):
    rng = np.random.default_rng(5)
    fields = []
    for i in range(n_fields):
        guides, decoys, z = _qc_field(rng)
        fields.append({"plate": "P", "well": "A1", "site": i, "z": z})
    return guides, decoys, fields


def test_the_false_call_tables_label_each_nucleus_and_split_fields_and_decoys():
    guides, decoys, fields = _qc_fields()
    reads, cells = ops_decode.false_call_tables(fields, guides, decoys,
                                                keep_reads=True)
    assert len(cells) == 8 * 40
    assert set(cells["label"]) == {"TP", "FP"}
    assert (cells.loc[cells["obj"] % 5 == 0, "label"] == "FP").all()
    assert cells["test"].sum() == 4 * 40
    assert set(cells.loc[cells["label"] == "FP", "decoy_half"]) <= {"train", "eval"}
    for column in ("n_exact_support", "min_mismatch", "q_best", "margin_c1",
                   "raw_max_c1", "bg_c1", "sharpness", "_gc", "_homopolymer"):
        assert column in cells.columns
    assert cells.loc[cells["obj"] == 1, "_homopolymer"].iloc[0] >= 1
    assert len(reads) == 8 * 40 * 3 and "label_read" in reads.columns
    _, lean = ops_decode.false_call_tables(fields, guides, decoys)
    pd.testing.assert_frame_equal(lean, cells)


def test_the_false_call_model_reports_main_control_and_chance():
    guides, decoys, fields = _qc_fields(12)
    _, cells = ops_decode.false_call_tables(fields, guides, decoys)
    halves = cells["decoy_half"].where(cells["label"] != "FP", "train")
    halves[(cells["label"] == "FP") & cells["test"]] = "eval"
    cells = cells.assign(decoy_half=halves)
    pytest.importorskip("shap")
    out = ops_decode.false_call_model(cells, model="hist_gbm", seed=0,
                                      permutation_repeats=2, boot=20)
    metrics = out["metrics"].set_index("model")
    assert list(metrics.index) == ["main_fp_vs_tp", "control_sequence_only", "chance"]
    assert metrics.loc["main_fp_vs_tp", "auc"] > 0.9
    assert metrics.loc["chance", "auc"] == 0.5
    assert out["shap_mean_abs"]["mean_abs_shap"].is_monotonic_decreasing
    assert set(out["permutation_importance"]["feature"]) == set(out["features"])
    with pytest.raises(ValueError, match="ops_decode_qc_model"):
        ops_decode.false_call_model(cells, model="forest")
    with pytest.raises(ValueError, match="decoy calls"):
        ops_decode.false_call_model(cells.assign(label="TP"))


def test_the_xgboost_model_has_shap_values_without_the_shap_package(monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.random(300), "b": rng.random(300)})
    model = xgboost.XGBClassifier(n_estimators=10, max_depth=2, n_jobs=1)
    model.fit(frame, (frame["a"] > 0.5).astype(int))
    import builtins

    real_import = builtins.__import__

    def no_shap(name, *args, **kwargs):
        """Refuse shap, import the rest."""
        if name == "shap":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_shap)
    values, base = ops_decode._tree_shap(model, frame)
    assert values.shape == (300, 2)
    assert np.abs(values[:, 0]).mean() > np.abs(values[:, 1]).mean()
    with pytest.raises(ImportError, match="shap"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        ops_decode._tree_shap(HistGradientBoostingClassifier().fit(
            frame, (frame["a"] > 0.5).astype(int)), frame)


def test_a_missing_xgboost_says_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        """Refuse xgboost, import the rest."""
        if name == "xgboost":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    with pytest.raises(ImportError, match="pip install xgboost"):
        ops_decode._xgboost_classifier({})


def test_the_decode_qc_phase_writes_its_reports_and_the_confidence_column(tmp_path):
    db = str(tmp_path / "measurements.db")
    calls = _calls(np.random.default_rng(4))
    folder = ops_decode.results_folder(db)
    for well, group in calls.groupby("well"):
        write_file(group, os.path.join(folder, f"calls_{well}.parquet"),
                   canonicalise=False)
    real = calls[~calls["is_decoy"]]
    write_table(db, "ops_barcodes", pd.DataFrame({
        "plate": "P", "well": real["well"], "object_id": real["object_id"],
        "barcode": "GUIDE", "quality": real["q_best"], "n_reads": real["n_reads"],
        "n_agreeing": 1, "fraction": 1.0, "n_cycles": 6, "mapped_guide": "GUIDE"}))
    settings = {"ops_genotype_confidence": True, "ops_confidence_min": 0.99,
                "ops_decoy_library_k": 4}
    report = ops_decode.run_decode_qc(db, "P", ["A1", "A2"], settings, ["GUIDE"])
    confidence = report["genotype_confidence"]
    assert confidence["model"] == "HGB" and 0 <= confidence["ece"] < 1
    assert confidence["auc"] > 0.7
    reliability = pd.read_csv(confidence["paths"]["reliability"])
    assert list(reliability.columns[:4]) == ["bin", "predicted", "observed", "n"]
    with open(confidence["paths"]["metrics"], encoding="utf-8") as handle:
        assert "shuffled_auc" in json.load(handle)
    barcodes = read_table(db, "ops_barcodes")
    assert "genotype_confidence" in barcodes.columns
    assert (barcodes["genotype_confidence"] >= 0.99).all()
    assert confidence["ops_barcodes"]["dropped"] == len(real) - len(barcodes)
    with pytest.raises(ValueError, match="per-call evidence"):
        ops_decode.run_decode_qc(db, "P", ["B9"], settings, ["GUIDE"])
    with pytest.raises(ValueError, match="field evidence"):
        ops_decode.run_decode_qc(db, "P", ["A1"], {"ops_decode_qc": True}, ["GUIDE"])
    with pytest.raises(ValueError, match="ops_decode_qc_split"):
        ops_decode.run_decode_qc(db, "P", ["A1"], {"ops_decode_qc": True,
                                                   "ops_decode_qc_split": "random"},
                                 ["GUIDE"])
    assert ops_decode.run_decode_qc(db, "P", ["A1"], {}, ["GUIDE"]) == {}


def test_the_decode_qc_phase_trains_the_false_call_model_from_the_cache(tmp_path):
    guides, decoys, fields = _qc_fields(12)
    db = str(tmp_path / "measurements.db")
    for fd in fields:
        ops_decode.save_field_cache(str(tmp_path / "A1" / "decode_cache"),
                                    fd["site"], fd["z"])
    settings = {"ops_decode_qc": True, "ops_decode_qc_model": "hist_gbm",
                "ops_decode_qc_max_calls": 50}
    pytest.importorskip("shap")
    from spacr import ops_sbs

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ops_sbs, "decoy_library", lambda library, k=4: decoys)
    try:
        report = ops_decode.run_decode_qc(db, "P", ["A1"], settings, guides)
    finally:
        monkey.undo()
    paths = report["false_call_model"]["paths"]
    for name in ("model_metrics.csv", "shap_mean_abs.csv",
                 "permutation_importance.csv", "model.json"):
        assert os.path.isfile(paths[name])
    metrics = pd.read_csv(paths["model_metrics.csv"])
    assert list(metrics["model"]) == ["main_fp_vs_tp", "control_sequence_only", "chance"]


def test_the_qc_settings_are_alpha_and_declared():
    from spacr import ops_settings

    for item in (876, 877):
        for key in ALPHA_FEATURES[item]["settings"]:
            assert key in ops_settings.OPS_DEFAULTS
            assert "Alpha." in ops_settings.OPS_TOOLTIPS[key]
    assert ops_settings.OPS_DEFAULTS["ops_confidence_seed"] == ops_decode.CONFIDENCE_SEED
    rule = ops_decode.decode_rule({"ops_decode_qc": True})
    assert rule["active"] and rule["store_cache"] and rule["features"]
