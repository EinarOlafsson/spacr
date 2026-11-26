"""One decode threshold set per dataset (item 1033, alpha, off by default).

``ops_decode_threshold_mode`` 'dataset' decodes every well at the entered
thresholds, then tunes one threshold set for the plate from the stored field
caches: on the tuning half of the fields the setting with the highest
``f = Gn (1 - e_total)^2`` whose decoy FDR meets the target and whose
split-read disagreement stays within the guard wins, and it is used only
when the paired held-out gain over the entered thresholds lies above 0.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode, ops_engine, ops_settings
from spacr.settings import ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION

NEW = ("ops_decode_target_fdr", "ops_decode_split_guard", "ops_decode_dataset_weights")
GUIDES = ["GTACGT", "CATGCA", "TGCATG"]
A, B, C = GUIDES

#: The synthetic grid: read threshold by single-read quality by pixel quality.
AXES = [("ops_read_threshold", [300.0, 315.0]),
        ("ops_single_read_quality", [0.05, 0.1, 0.2]),
        ("ops_pixel_quality", [0.2])]
MANUAL = (1, 1, 0)
PLANTED = (0, 1, 0)

#: Real calls per field at each setting; the loosest setting carries decoys.
REAL = {(0, 0, 0): 700, (0, 1, 0): 680, (0, 2, 0): 640,
        (1, 0, 0): 660, (1, 1, 0): 600, (1, 2, 0): 580}


def _config(**change):
    config = {"target_fdr": 0.1, "split_guard": 0.1, "ci": True,
              "weights": dict(ops_decode.DATASET_WEIGHTS)}
    config.update(change)
    return config


def _arrays(fields=40, real=None, agree=None, report_real=None):
    """Per-field counts of a dataset whose best passing setting is ``PLANTED``."""
    real = dict(real or REAL)
    shape = tuple(len(v) for _, v in AXES)
    rng = np.random.default_rng(7)
    out = {name: np.zeros((fields,) + shape, np.int64)
           for name in ("real", "decoy", "agree", "scored")}
    for pos, value in real.items():
        noise = rng.integers(-5, 6, fields)
        out["real"][(slice(None),) + pos] = value + noise
        out["scored"][(slice(None),) + pos] = 50
        out["agree"][(slice(None),) + pos] = (agree or {}).get(pos, 49)
        if report_real and pos in report_real:
            out["real"][(slice(fields // 2, None),) + pos] = report_real[pos] + noise[fields // 2:]
    out["decoy"][(slice(None), 0, 0, 0)] = 3
    out["ref_agree"] = np.full(fields, 20, np.int64)
    out["ref_scored"] = np.full(fields, 20, np.int64)
    return out


def _choice(arrays, config=None, fields=40):
    tune, report = list(range(fields // 2)), list(range(fields // 2, fields))
    return ops_decode._dataset_choice(arrays, np.full(fields, 1000), tune, report, AXES,
                                      MANUAL, config or _config(), k=4, scale=1.268,
                                      reps=300, seed=5)


def _values(pos):
    return {name: values[pos[a]] for a, (name, values) in enumerate(AXES)}


def test_off_is_the_default_and_adds_no_schema_row():
    assert ops_settings.OPS_DEFAULTS["ops_decode_threshold_mode"] == "auto"
    assert "dataset" in ops_decode.THRESHOLD_MODES
    assert ALPHA_FEATURES[1033]["choices"] == {"ops_decode_threshold_mode": ("dataset",)}
    assert set(ALPHA_FEATURES[1033]["settings"]) == set(NEW)
    assert SETTINGS_SCHEMA_VERSION == 985
    assert not any(set(NEW) & set(row) for row in DEFAULT_CHANGES.values())
    for key in NEW:
        assert key in ops_settings.OPS_TYPES
        assert ops_settings.OPS_TOOLTIPS[key].endswith("Alpha.")
        assert any(key in keys for keys in ops_settings.OPS_CATEGORIES.values())
    settings = {"ops_decode_threshold_mode": "auto"}
    assert ops_engine._dataset_mode_settings(settings, ("decode",)) is settings


def test_dataset_mode_decodes_each_well_as_manual_does():
    rule = ops_decode.decode_rule(ops_decode.resolve_settings({"ops_decode_preset": "rule_a_s2"}))
    for calibration in ("none", "never"):
        base = {"ops_decode_preset": "rule_a_s2", "ops_decode_fdr_calibration": calibration}
        manual = ops_decode.threshold_settings({**base, "ops_decode_threshold_mode": "manual"},
                                               rule)
        dataset = ops_decode.threshold_settings({**base, "ops_decode_threshold_mode": "dataset"},
                                                rule)
        assert dataset == {**manual, "dataset": True}
    config = ops_decode.dataset_threshold_settings(
        {"ops_decode_preset": "rule_a_s2", "ops_read_threshold": 330.0,
         "ops_single_read_quality": 0.07, "ops_decode_auto_spot_grid": [250, 315]}, rule)
    assert config["manual"] == {"ops_read_threshold": 330.0, "ops_single_read_quality": 0.07,
                                "ops_pixel_quality": rule["pixel_quality"]}
    assert config["spot_thresholds"] == [250.0, 315.0, 330.0]
    assert 0.07 in config["floors"] and rule["pixel_quality"] in config["pixel_qualities"]
    assert config["target_fdr"] == 0.1 and config["split_guard"] == 0.1
    assert config["weights"] == ops_decode.DATASET_WEIGHTS
    for bad in ({"ops_decode_target_fdr": 0}, {"ops_decode_split_guard": -1},
                {"ops_decode_dataset_weights": {"split": -1}},
                {"ops_decode_dataset_weights": "not json"}):
        with pytest.raises(ValueError):
            ops_decode.dataset_threshold_settings({"ops_decode_preset": "rule_a_s2", **bad},
                                                  rule)
    with pytest.raises(ValueError, match="ops_decoy_library_k"):
        ops_decode.dataset_threshold_settings({}, {**rule, "decoy_k": 0})


def test_the_tuner_finds_the_planted_optimum_under_the_fdr_target():
    out = _choice(_arrays())
    assert out["auto_met"] and out["adopted"]
    assert out["thresholds"]["chosen"] == _values(PLANTED)
    assert out["thresholds"]["used"] == _values(PLANTED)
    assert out["thresholds"]["manual"] == _values(MANUAL)
    loosest = next(r for r in out["grid"] if r["ops_single_read_quality"] == 0.05
                   and r["ops_read_threshold"] == 300.0)
    assert loosest["f"] > next(r for r in out["grid"] if r["chosen"])["f"]
    assert not loosest["fdr_passes"]
    gain = out["report"]["gain"]
    assert gain["ci95"][0] > 0 and gain["f"] > 0
    assert out["report"]["chosen"]["fdr_per_100"] <= 0.1
    assert out["report"]["chosen"]["split_disagreement_pct"] == pytest.approx(2.0)
    loose = _choice(_arrays(), _config(target_fdr=1.0))
    assert loose["thresholds"]["chosen"] == _values((0, 0, 0))


def test_it_refuses_to_adopt_when_the_held_out_gain_is_not_above_0():
    arrays = _arrays(report_real={PLANTED: 600})
    out = _choice(arrays)
    assert out["thresholds"]["chosen"] == _values(PLANTED)
    assert not out["adopted"]
    assert out["thresholds"]["used"] == _values(MANUAL)
    assert out["report"]["gain"]["ci95"][0] <= 0
    assert "does not lie above 0" in out["reason"]
    same = _choice(_arrays(real={**REAL, (1, 1, 0): 800}))
    assert same["thresholds"]["chosen"] == _values(MANUAL) and not same["adopted"]
    assert same["reason"] == "the tuning half chose the manual thresholds"


def test_the_split_read_guard_binds():
    worse = {PLANTED: 40}
    free = _config(weights={"split": 0.0, "decoy": 0.0})
    guarded = _choice(_arrays(agree=worse), free)
    row = next(r for r in guarded["grid"] if r["ops_read_threshold"] == 300.0
               and r["ops_single_read_quality"] == 0.1)
    assert row["fdr_passes"] and not row["split_passes"]
    assert guarded["thresholds"]["chosen"] == _values((1, 0, 0))
    loose = _choice(_arrays(agree=worse), _config(weights=free["weights"], split_guard=100.0))
    assert loose["thresholds"]["chosen"] == _values(PLANTED)
    weighted = _choice(_arrays(agree=worse), _config(split_guard=100.0))
    assert weighted["thresholds"]["chosen"] == _values((1, 0, 0))


def test_the_choice_is_deterministic():
    first, again = _choice(_arrays()), _choice(_arrays())
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_calibrated_weights_recover_a_known_error():
    rng = np.random.default_rng(0)
    terms = rng.uniform(0, 0.1, (50, 2))
    truth = terms @ np.array([0.8, 2.0])
    weights = ops_decode.calibrate_error_weights(terms, truth, ["split", "decoy"])
    assert weights == pytest.approx({"split": 0.8, "decoy": 2.0})
    with pytest.raises(ValueError):
        ops_decode.calibrate_error_weights(terms, truth[:3], ["split", "decoy"])


def _block(codes):
    values = np.full((len(codes), 6, 4), 10.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] = 1000.0
    return values


def _cache_field(own, other, own_guide, other_guide, strengths):
    """A field owning nucleus ``own`` and holding inside reads of ``other``."""
    reads = [(own, own_guide, True, 0.0, s) for s in strengths]
    reads += [(other, other_guide, False, -1.0, 500.0)] * 2
    return {"strength": np.array([r[4] for r in reads], np.float32),
            "values": _block([r[1] for r in reads]),
            "gap": np.array([[r[3], 30.0] for r in reads], np.float32),
            "id0": np.array([r[0] for r in reads], np.int64),
            "owned0": np.array([r[2] for r in reads], bool),
            "owned_ids": np.array([own], np.int64),
            "nuc_ids": np.array([own, other], np.int64),
            "nuc_centroids": np.array([[20.0, 20.0], [20.0, 80.0]]),
            "nuc_areas": np.array([100.0, 100.0])}


def _write_cache(folder):
    folder.mkdir()
    plan = [(0, 1, 2, A, B), (1, 2, 1, B, A), (2, 3, 4, C, C), (3, 4, 3, C, C)]
    for site, own, other, guide, seen in plan:
        np.savez(folder / f"site{site:04d}.npz",
                 **_cache_field(own, other, guide, seen, [500.0, 500.0, 320.0]))
    return folder


SETTINGS = {"ops_decode_preset": "rule_a_s2", "ops_decode_library_calibration": False,
            "ops_decoy_library_k": 1, "ops_crosstalk": "none",
            "ops_read_neighbour_discount": 1.0,
            "ops_read_threshold": 315.0, "ops_single_read_quality": 0.06,
            "ops_decode_auto_grid": {"ops_single_read_quality": [0.06, 0.1]},
            "ops_decode_auto_spot_grid": [315.0, 400.0]}


def test_the_tuner_replays_stored_field_caches(tmp_path):
    folder = _write_cache(tmp_path / "decode_cache")
    out = ops_decode.tune_dataset_thresholds({"A1": str(folder)}, SETTINGS, GUIDES, reps=50)
    assert out["fields"] == 4 and out["nuclei"] == 4
    assert out["thresholds"]["manual"]["ops_read_threshold"] == 315.0
    assert out["axes"]["ops_read_threshold"] == [315.0, 400.0]
    assert len(out["grid"]) == 4 and sum(r["manual"] for r in out["grid"]) == 1
    assert len(out["tune_sites"]) == 2 and len(out["report_sites"]) == 2
    loose = next(r for r in out["grid"] if r["ops_read_threshold"] == 315.0)
    strict = next(r for r in out["grid"] if r["ops_read_threshold"] == 400.0)
    assert loose["real"] >= strict["real"]
    scored = (out["tune"]["manual"]["split_reference_nuclei"]
              + out["report"]["manual"]["split_reference_nuclei"])
    assert scored == 4
    named = ops_decode.tune_dataset_thresholds(
        {"A1": str(folder)}, SETTINGS, GUIDES, reps=50, tune_sites={"A1": [0, 1]})
    assert named["tune_sites"] == [["A1", 0], ["A1", 1]]
    again = ops_decode.tune_dataset_thresholds({"A1": str(folder)}, SETTINGS, GUIDES, reps=50)
    assert json.dumps(again, sort_keys=True) == json.dumps(out, sort_keys=True)
    paths = ops_decode.write_dataset_thresholds(str(tmp_path / "measurements.db"), "P1", out)
    with open(paths["json"], encoding="utf-8") as handle:
        assert json.load(handle)["plate"] == "P1"
    with pytest.raises(ValueError, match="extra_terms"):
        ops_decode.tune_dataset_thresholds(
            {"A1": str(folder)}, {**SETTINGS, "ops_decode_dataset_weights": {"spike_in": 1}},
            GUIDES, reps=50)
    grid = (2, 2, 1)
    spike = {("A1", s): (np.zeros(grid), np.ones(grid)) for s in range(4)}
    with_spike = ops_decode.tune_dataset_thresholds(
        {"A1": str(folder)}, {**SETTINGS, "ops_decode_dataset_weights": {"spike_in": 1}},
        GUIDES, reps=50, extra_terms={"spike_in": spike})
    assert with_spike["weights"]["spike_in"] == 1.0


def test_the_engine_refuses_phases_that_read_the_decode_after_it():
    settings = {"ops_decode_threshold_mode": "dataset"}
    with pytest.raises(ValueError, match="dataset"):
        ops_engine._dataset_mode_settings(settings, ("decode", "merge"))
    kept = ops_engine._dataset_mode_settings(settings, ("decode",))
    assert kept["ops_store_field_cache"] is True
    assert ops_engine._dataset_mode_settings(settings, ("stitch",)) is settings


@pytest.fixture(scope="module")
def dataset_run(tmp_path_factory):
    """The synthetic well of item 873's tests, decoded manual and dataset."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setitem(ops_settings.OPS_DEFAULTS, "ops_raster_overlap", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_dataset_1033")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_preset": "rule_a_s2", "ops_decode_threshold_mode": "manual",
                "ops_decode_fdr_calibration": "none"}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    manual = ops_engine.run_ops(settings, library=library)
    dataset = ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "dataset"},
                                 phases=("decode",), library=library)
    yield manual, dataset, settings
    monkeypatch.undo()


def test_the_engine_tunes_the_plate_and_keeps_the_decode_unless_it_gains(dataset_run):
    manual, dataset, settings = dataset_run
    assert "dataset_thresholds" not in manual
    result = dataset["dataset_thresholds"]
    with open(result["paths"]["json"], encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["plate"] == "synthetic" and stored["wells"] == ["A1"]
    decode = dataset["wells"]["A1"]["decode"]
    assert decode["dataset_thresholds"]["adopted"] == result["adopted"]
    used = result["thresholds"]["used"]
    assert decode["decode_rules"]["single_read_quality"] == used["ops_single_read_quality"]
    assert decode["decode_rules"]["pixel_quality"] == used["ops_pixel_quality"]
    if not result["adopted"]:
        assert used == result["thresholds"]["manual"]
        assert decode["objects_assigned"] == manual["wells"]["A1"]["decode"]["objects_assigned"]
