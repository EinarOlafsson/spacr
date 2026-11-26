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
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode, ops_engine, ops_settings
from spacr.settings import ALPHA_FEATURES

NEW = ("ops_decode_target_fdr", "ops_decode_split_guard", "ops_decode_dataset_weights",
       "ops_decode_dataset_rule", "ops_brieflow_yield", "ops_brieflow_split_read",
       "ops_decode_target_fdr_reference", "ops_brieflow_fair_fdr")
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


def _choice(arrays, config=None, fields=40, choose=None):
    tune, report = list(range(fields // 2)), list(range(fields // 2, fields))
    return (choose or ops_decode._dataset_choice_target_gain)(arrays, np.full(fields, 1000), tune, report, AXES,
                                      MANUAL, config or _config(), k=4, scale=1.268,
                                      reps=300, seed=5)


def _values(pos):
    return {name: values[pos[a]] for a, (name, values) in enumerate(AXES)}


def test_off_is_the_default():
    assert ops_settings.OPS_DEFAULTS["ops_decode_threshold_mode"] == "auto"
    assert "dataset" in ops_decode.THRESHOLD_MODES
    assert ALPHA_FEATURES[1033]["choices"] == {"ops_decode_threshold_mode": ("dataset",)}
    assert set(ALPHA_FEATURES[1033]["settings"]) == set(NEW)
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
    assert config["target_fdr"] is None and config["rule"] == "beat_reference"
    assert config["brieflow_yield"] is None and config["brieflow_split_read"] is None
    assert config["weights"] == ops_decode.DATASET_KNEE_WEIGHTS
    legacy = ops_decode.dataset_threshold_settings(
        {"ops_decode_preset": "rule_a_s2", "ops_decode_dataset_rule": "target_gain"}, rule)
    assert legacy["target_fdr"] == 0.06 and legacy["split_guard"] == 0.1
    assert legacy["weights"] == ops_decode.DATASET_WEIGHTS
    for bad in ({"ops_decode_target_fdr": 0}, {"ops_decode_split_guard": -1},
                {"ops_decode_dataset_weights": {"split": -1}},
                {"ops_decode_dataset_weights": "not json"}):
        with pytest.raises(ValueError):
            ops_decode.dataset_threshold_settings({"ops_decode_preset": "rule_a_s2", **bad},
                                                  rule)
    with pytest.raises(ValueError, match="ops_decoy_library_k"):
        ops_decode.dataset_threshold_settings({}, {**rule, "decoy_k": 0})


def _held_decoys(arrays, per_field, fields=40):
    arrays["decoy"][(slice(fields // 2, None),) + PLANTED] = per_field
    return arrays


def test_a_held_out_fdr_estimate_over_the_target_is_not_adopted():
    out = _choice(_held_decoys(_arrays(), 5))
    assert out["thresholds"]["chosen"] == _values(PLANTED)
    assert out["report"]["gain"]["ci95"][0] > 0
    assert out["held_out_fdr"]["estimate"] > 0.1
    assert not out["held_out_fdr"]["at_or_below_target"]
    assert not out["adopted"] and out["thresholds"]["used"] == _values(MANUAL)
    assert "held-out FDR estimate" in out["reason"]
    row = next(r for r in out["grid"] if r["chosen"])
    assert row["held_out_fdr"] == pytest.approx(out["held_out_fdr"]["estimate"])
    assert row["held_out_fdr_hi"] >= row["held_out_fdr"]


def test_a_held_out_fdr_estimate_under_the_target_is_adopted_even_if_its_bound_is_over():
    arrays = _arrays()
    arrays["decoy"][(20,) + PLANTED] = 40
    out = _choice(arrays)
    held = out["held_out_fdr"]
    assert held["estimate"] <= 0.1 < held["ci95"][1]
    assert out["adopted"] and out["thresholds"]["used"] == _values(PLANTED)


def test_the_target_defaults_to_0_06_without_brieflow_and_follows_a_brieflow_reference():
    rule = ops_decode.decode_rule(ops_decode.resolve_settings({"ops_decode_preset": "rule_a_s2"}))
    base = {"ops_decode_preset": "rule_a_s2"}
    base = {**base, "ops_decode_dataset_rule": "target_gain"}
    plain = ops_decode.dataset_threshold_settings(base, rule)
    assert plain["target_fdr"] == 0.06 and plain["target_source"] == "default"
    given = ops_decode.dataset_threshold_settings(
        {**base, "ops_decode_target_fdr_reference": "brieflow", "ops_brieflow_fair_fdr": 0.056},
        rule)
    assert given["target_fdr"] == 0.056 and given["target_source"] == "brieflow"
    knee = ops_decode.dataset_threshold_settings(
        {"ops_decode_preset": "rule_a_s2", "ops_decode_target_fdr_reference": "brieflow",
         "ops_brieflow_fair_fdr": 0.056}, rule)
    assert knee["target_fdr"] is None and knee["brieflow_fair_fdr"] == 0.056
    entered = ops_decode.dataset_threshold_settings(
        {**base, "ops_decode_target_fdr": 0.2, "ops_decode_target_fdr_reference": "brieflow",
         "ops_brieflow_fair_fdr": 0.056}, rule)
    assert entered["target_fdr"] == 0.2 and entered["target_source"] == "entered"
    for bad in ({"ops_decode_target_fdr_reference": "brieflow"},
                {"ops_decode_target_fdr_reference": "other"}):
        with pytest.raises(ValueError):
            ops_decode.dataset_threshold_settings({**base, **bad}, rule)


def test_schema_986_987_and_988_replay_the_old_rule_for_older_files():
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, _fold_older_defaults

    assert SETTINGS_SCHEMA_VERSION >= 986
    assert DEFAULT_CHANGES[986] == {"ops_decode_target_fdr": 0.1}
    assert DEFAULT_CHANGES[987] == {"ops_decode_dataset_rule": "target_gain"}
    assert DEFAULT_CHANGES[988] == {"ops_decode_dataset_rule": "knee"}
    old = {"ops_decode_threshold_mode": "dataset"}
    _fold_older_defaults(old, 985)
    assert old["ops_decode_target_fdr"] == 0.1
    new = {"ops_decode_threshold_mode": "dataset"}
    _fold_older_defaults(new, 986)
    assert "ops_decode_target_fdr" not in new
    set_by_user = {"ops_decode_target_fdr": 0.3}
    _fold_older_defaults(set_by_user, 985)
    assert set_by_user["ops_decode_target_fdr"] == 0.3
    stamped_986 = {}
    _fold_older_defaults(stamped_986, 986)
    assert stamped_986["ops_decode_dataset_rule"] == "target_gain"
    stamped_987 = {}
    _fold_older_defaults(stamped_987, 987)
    assert stamped_987["ops_decode_dataset_rule"] == "knee"
    stamped_988 = {}
    _fold_older_defaults(stamped_988, 988)
    assert "ops_decode_dataset_rule" not in stamped_988
    chosen = {"ops_decode_dataset_rule": "knee"}
    _fold_older_defaults(chosen, 986)
    assert chosen["ops_decode_dataset_rule"] == "knee"


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
    assert loose["n"] >= strict["n"]
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
    assert stored["rule"] == "beat_reference" and os.path.getsize(result["paths"]["figure"]) > 1000
    decode = dataset["wells"]["A1"]["decode"]
    assert decode["dataset_thresholds"]["adopted"] == result["adopted"]
    used = result["thresholds"]["used"]
    assert decode["decode_rules"]["single_read_quality"] == used["ops_single_read_quality"]
    assert decode["decode_rules"]["pixel_quality"] == used["ops_pixel_quality"]
    if not result["adopted"]:
        assert used == result["thresholds"]["manual"]
        assert decode["objects_assigned"] == manual["wells"]["A1"]["decode"]["objects_assigned"]


def _frontier_arrays(fields=40, decoys=(20, 24, 28, 32, 80, 200), seed=3):
    """A dataset whose six frontier settings (``FRONT``) have a planted stop.

    Calls rise by 100 per setting. The decoys per field rise in step with
    the calls (a flat error rate) up to the fourth setting, then jump, so
    the first step whose marginal error rate is above the average one is
    the fourth to the fifth. Every other setting is beaten by one of them.
    """
    shape = tuple(len(v) for _, v in AXES)
    rng = np.random.default_rng(seed)
    out = {name: np.zeros((fields,) + shape, np.int64)
           for name in ("real", "decoy", "agree", "scored")}
    calls = [500, 600, 700, 800, 900, 1000]
    for pos, n, d in zip(FRONT, calls, decoys):
        out["real"][(slice(None),) + pos] = n
        out["decoy"][(slice(None),) + pos] = d + rng.integers(-3, 4, fields)
    for pos in np.ndindex(*shape):
        if pos not in FRONT:
            out["real"][(slice(None),) + pos] = 400
            out["decoy"][(slice(None),) + pos] = 50
    out["scored"][:] = 50
    out["agree"][:] = 50
    out["ref_agree"] = np.full(fields, 20, np.int64)
    out["ref_scored"] = np.full(fields, 20, np.int64)
    return out


FRONT = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0)]


def _knee(arrays=None, **change):
    config = _config(rule="knee", target_fdr=None, weights={"split": 1.0, "decoy": 1.0})
    config.update(change)
    return _choice(arrays or _frontier_arrays(), config, choose=ops_decode._dataset_choice)


def test_a_marginal_step_is_compared_with_the_average_error_rate():
    n = np.array([500.0, 600.0])
    boot_n = np.tile(n, (200, 1))
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (200, 1))
    above = ops_decode._marginal_steps(n, np.array([10.0, 40.0]), boot_n,
                                       np.array([10.0, 40.0]) + noise)[0]
    assert above["marginal"] == pytest.approx(0.3) and above["average"] == pytest.approx(0.02)
    assert above["significant"] and above["ci95"][0] > 0
    flat = ops_decode._marginal_steps(n, np.array([10.0, 12.0]), boot_n,
                                      np.array([10.0, 12.0]) + noise)[0]
    assert flat["marginal"] == pytest.approx(0.02) and not flat["significant"]
    falling = ops_decode._marginal_steps(n, np.array([10.0, 10.0]), boot_n,
                                         np.array([10.0, 10.0]) + noise)[0]
    assert falling["marginal"] == 0.0 and falling["diff"] < 0 and not falling["significant"]


def test_the_walk_stops_before_the_first_step_that_raises_the_error_rate():
    out = _knee()
    assert out["thresholds"]["chosen"] == _values(FRONT[3])
    assert out["adopted"] and out["thresholds"]["used"] == _values(FRONT[3])
    knee = out["knee"]
    assert knee["position"] == 3 and len(knee["points"]) == len(FRONT)
    assert [step["significant"] for step in knee["steps"]][:3] == [False] * 3
    assert knee["steps"][3]["significant"] and knee["steps"][3]["ci95"][0] > 0
    assert knee["tune"]["out_of"]["significant"]
    assert knee["report"]["into"] is not None and not knee["report"]["into"]["significant"]
    walked = [r for r in out["grid"] if r["walked"]]
    assert len(walked) == 4 and sum(r["knee"] for r in out["grid"]) == 1
    chosen, manual = out["report"]["chosen"], out["report"]["manual"]
    for part in (chosen, manual):
        assert part["n_ci95"][0] <= part["n"] <= part["n_ci95"][1]
        assert part["errors_ci95"][0] is not None and part["fdr_ci95"][0] is not None
    row = next(r for r in out["grid"] if r["knee"])
    assert {"decoy_errors", "split_errors", "errors", "n"} <= set(row)
    assert row["errors"] == pytest.approx(row["decoy_errors"] + row["split_errors"])


def test_a_flat_or_falling_error_rate_walks_to_the_most_calls():
    flat = _knee(_frontier_arrays(decoys=(20, 24, 28, 32, 36, 40)))
    assert flat["knee"]["position"] == len(FRONT) - 1
    assert flat["thresholds"]["chosen"] == _values(FRONT[-1])
    falling = _knee(_frontier_arrays(decoys=(20, 22, 24, 26, 28, 30)))
    assert falling["thresholds"]["chosen"] == _values(FRONT[-1])


def test_the_fdr_cap_leaves_points_off_the_frontier():
    free = _knee()
    capped = _knee(target_fdr=2.0)
    assert any(not r["within_cap"] for r in capped["grid"])
    assert all(r["within_cap"] for r in capped["grid"] if r["frontier"])
    assert sum(r["frontier"] for r in capped["grid"]) < sum(r["frontier"] for r in free["grid"])
    assert capped["held_out_fdr"]["target"] == 2.0
    nothing = _knee(target_fdr=1e-9)
    assert not nothing["adopted"] and "cap" in nothing["reason"]
    assert nothing["thresholds"]["used"] == _values(MANUAL)


def test_the_knee_is_deterministic():
    first, again = _knee(), _knee()
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(again, sort_keys=True,
                                                                         default=str)


def test_a_held_out_step_that_raises_the_error_rate_is_not_adopted():
    arrays = _frontier_arrays()
    arrays["decoy"][(slice(20, None),) + FRONT[3]] = 300
    out = _knee(arrays)
    assert out["thresholds"]["chosen"] == _values(FRONT[3])
    assert out["knee"]["report"]["into"]["significant"]
    assert not out["adopted"] and "significantly raises" in out["reason"]
    assert out["thresholds"]["used"] == _values(MANUAL)


def test_the_frontier_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    out = _knee()
    out.update({"brieflow_fair_fdr": 0.056})
    path = ops_decode.plot_dataset_frontier(out, str(tmp_path / "frontier.png"))
    assert (tmp_path / "frontier.png").stat().st_size > 1000 and path.endswith(".png")
    assert (tmp_path / "frontier.svg").exists()


def _reference(arrays=None, yield_=0.95, fdr=7.0, split=None, **change):
    config = _config(rule="beat_reference", target_fdr=None, brieflow_yield=yield_,
                     brieflow_fair_fdr=fdr, brieflow_split_read=split,
                     weights={"split": 1.0, "decoy": 1.0})
    config.update(change)
    return _choice(arrays or _frontier_arrays(), config, choose=ops_decode._dataset_choice)


def test_the_point_with_the_most_cells_that_beats_the_reference_on_both_is_chosen():
    out = _reference()
    assert out["knee"]["position"] == 5 and out["reference"]["rule_used"] == "beat_reference"
    assert out["thresholds"]["chosen"] == _values(FRONT[5])
    assert out["adopted"] and out["thresholds"]["used"] == _values(FRONT[5])
    beating = [r for r in out["grid"] if r["beats_reference"]]
    assert len(beating) == 1 and beating[0]["knee"]
    assert out["reference"]["report"]["beats_both"] and out["reference"]["beating_points"] == 1
    assert out["reference"]["n_ref"] == pytest.approx(0.95 * 20 * 1000)
    walk = _knee()
    assert walk["thresholds"]["chosen"] == _values(FRONT[3])
    two = _reference(yield_=0.75, fdr=3.0)
    assert two["thresholds"]["chosen"] == _values(FRONT[4])
    assert {r["ops_read_threshold"] for r in out["grid"] if r["beats_reference"]} == {315.0}


def test_a_split_read_reference_excludes_a_point_that_disagrees_too_often():
    arrays = _frontier_arrays()
    arrays["agree"][(slice(None),) + FRONT[5]] = 45
    out = _reference(arrays, yield_=0.75, fdr=7.0, split=0.0276)
    assert out["thresholds"]["chosen"] == _values(FRONT[4])
    assert not next(r for r in out["grid"] if r["ops_read_threshold"] == 315.0
                    and r["ops_single_read_quality"] == 0.2)["beats_reference"]
    assert out["reference"]["errors_ref"] is not None


def test_an_empty_region_keeps_the_manual_thresholds_and_says_so():
    """Einar 2026-11-26: with no point beating the reference, manual stays."""
    out = _reference(yield_=0.95, fdr=1.0)
    assert out["reference"]["beating_points"] == 0
    assert out["thresholds"]["chosen"] == out["thresholds"]["manual"]
    assert out["reason"].startswith("no frontier point beats the reference")
    assert "manual thresholds are kept" in out["reason"] and not out["adopted"]
    none = _reference(yield_=None, fdr=None)
    assert none["thresholds"]["chosen"] == _values(FRONT[3]) and not none["reference"]["used"]


def test_a_point_that_stops_beating_the_reference_on_the_held_out_half_is_not_adopted():
    arrays = _frontier_arrays()
    arrays["decoy"][(slice(20, None),) + FRONT[5]] = 2000
    out = _reference(arrays)
    assert out["thresholds"]["chosen"] == _values(FRONT[5])
    assert not out["reference"]["report"]["beats_both"]
    assert not out["adopted"] and "no longer beats the reference" in out["reason"]
    assert out["thresholds"]["used"] == _values(MANUAL)
    assert out["report"]["chosen"]["fdr_ci95"][0] is not None


def test_the_reference_settings_are_read_and_checked():
    rule = ops_decode.decode_rule(ops_decode.resolve_settings({"ops_decode_preset": "rule_a_s2"}))
    base = {"ops_decode_preset": "rule_a_s2"}
    config = ops_decode.dataset_threshold_settings(
        {**base, "ops_brieflow_yield": 0.605, "ops_brieflow_fair_fdr": 0.056,
         "ops_brieflow_split_read": 0.0276}, rule)
    assert (config["brieflow_yield"], config["brieflow_fair_fdr"],
            config["brieflow_split_read"]) == (0.605, 0.056, 0.0276)
    assert config["rule"] == "beat_reference" and config["target_fdr"] is None
    for bad in ({"ops_brieflow_yield": 1.5}, {"ops_brieflow_yield": 0},
                {"ops_brieflow_split_read": 1.0}, {"ops_decode_dataset_rule": "other"}):
        with pytest.raises(ValueError):
            ops_decode.dataset_threshold_settings({**base, **bad}, rule)
    assert ops_settings.OPS_DEFAULTS["ops_decode_dataset_rule"] == "beat_reference"


def test_the_figure_shades_the_region_that_beats_the_reference(tmp_path):
    pytest.importorskip("matplotlib")
    out = _reference(split=0.0276)
    out.update({"brieflow_fair_fdr": 7.0})
    ops_decode.plot_dataset_frontier(out, str(tmp_path / "ref.png"))
    assert (tmp_path / "ref.png").stat().st_size > 1000


def test_a_point_that_beats_brieflow_but_raises_the_internal_split_read_is_excluded():
    arrays = _frontier_arrays()
    arrays["agree"][(slice(None),) + FRONT[5]] = 45
    out = _reference(arrays)
    row = next(r for r in out["grid"] if r["ops_read_threshold"] == 315.0
               and r["ops_single_read_quality"] == 0.2)
    assert row["split_rise_hi"] > 0.1 and not row["split_passes"]
    assert not row["beats_reference"]
    assert out["reference"]["rule_used"] == "knee" and out["reference"]["beating_points"] == 0
    assert out["reason"].startswith("no frontier point beats the reference")
    easy = _reference(arrays, split=None, **{"split_guard": 100.0})
    assert easy["thresholds"]["chosen"] == _values(FRONT[5])
    assert easy["reference"]["rule_used"] == "beat_reference"


def test_the_split_guard_is_checked_again_on_the_held_out_half():
    arrays = _frontier_arrays()
    arrays["agree"][(slice(20, None),) + FRONT[5]] = 45
    out = _reference(arrays)
    assert out["thresholds"]["chosen"] == _values(FRONT[5])
    assert out["reference"]["rule_used"] == "beat_reference"
    assert not out["reference"]["report"]["split_guard_ok"]
    assert out["reference"]["report"]["split_rise"] > 0.1
    assert not out["adopted"] and "split-read" in out["reason"]
    assert out["thresholds"]["used"] == _values(MANUAL)


def test_the_guard_helper_is_the_one_the_target_gain_rule_uses():
    boot = np.zeros((50, 2, 3, 1))
    boot[:, 1, 2, 0] = 5.0
    rise_hi, ok = ops_decode._split_rise_guard(boot, (1, 1, 0), 0.1, (2, 3, 1))
    assert ok[0, 0, 0] and not ok[1, 2, 0] and rise_hi[1, 2, 0] == pytest.approx(5.0)
