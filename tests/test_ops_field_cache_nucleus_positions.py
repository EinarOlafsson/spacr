"""The OPS field cache stores the field's nucleus positions for the neighbour-bleed discount.

``ops_store_field_cache`` writes each field's decode evidence; it now also
writes the field task's object ids, centroids and areas (``nuc_ids``,
``nuc_centroids``, ``nuc_areas``), so a replay of the cache can apply
``ops_read_neighbour_discount`` (item 988's N1) without measuring the field
again. The evidence arrays are unchanged, a cache written before still loads,
and N1 on such a cache is refused with a message that says why.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode
from spacr.ops_sbs import _disc_neighbours

GUIDES = ["GTACGT", "CATGCA", "TGCATG"]
IDS = np.array([11, 12, 13], np.int64)
CENTROIDS = np.array([[0.25, 0.5], [0.0, 12.125], [0.0, 200.0]])
AREAS = np.array([50.0, 50.0, 50.0])


def _evidence():
    """Object 11 holds three reads of its guide and one of object 12's top guide."""
    codes = ["GTACGT"] * 3 + ["CATGCA"] * 3
    values = np.full((len(codes), 6, 4), 10.0, np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            values[i, c, "GTAC".index(letter)] = 1000.0
    return {"strength": np.full(len(codes), 500.0, np.float32), "values": values,
            "gap": np.tile(np.array([[0.0, 30.0]], np.float32), (len(codes), 1)),
            "id0": np.array([11, 11, 11, 11, 12, 12]),
            "owned0": np.ones(len(codes), bool), "owned_ids": IDS,
            "owned_areas": AREAS}


def _task():
    return {"ids": IDS, "centroids": CENTROIDS, "areas": AREAS}


def _calls(evidence, rule):
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    return ops_decode.field_calls(evidence, rule, GUIDES, matcher, decoys,
                                  threshold=315.0, footprint=10.0, max_distance=0,
                                  min_fraction=0.8, single_read_quality=0.0,
                                  crosstalk="none")


def _rule(discount):
    return ops_decode.decode_rule({"ops_decoy_library_k": 4,
                                   "ops_read_neighbour_discount": discount})


def _engine_rule(discount):
    """The rule as the engine decodes a field: the neighbours built from the task."""
    rule = _rule(discount)
    return {**rule, "neighbours": _disc_neighbours(IDS, CENTROIDS, AREAS, 10.0)}


def test_the_cache_adds_the_nucleus_positions_and_leaves_the_evidence_unchanged(tmp_path):
    evidence = _evidence()
    old = np.load(ops_decode.save_field_cache(str(tmp_path / "old"), 3, evidence))
    new = np.load(ops_decode.save_field_cache(str(tmp_path / "new"), 3, evidence,
                                              objects=_task()))
    assert set(new.files) - set(old.files) == set(ops_decode.CACHE_NUCLEUS_KEYS)
    for key in old.files:
        assert old[key].dtype == new[key].dtype
        assert old[key].tobytes() == new[key].tobytes()
    assert new["nuc_ids"].dtype == np.int64 and new["nuc_ids"].tolist() == IDS.tolist()
    assert new["nuc_centroids"].tobytes() == CENTROIDS.astype(np.float64).tobytes()
    assert new["nuc_areas"].tobytes() == AREAS.astype(np.float64).tobytes()
    assert set(evidence) == set(_evidence())


def test_the_cached_neighbours_are_the_ones_the_engine_builds(tmp_path):
    cache = np.load(ops_decode.save_field_cache(str(tmp_path), 1, _evidence(),
                                                objects=_task()))
    assert ops_decode.cached_neighbours(cache, 10.0) == _disc_neighbours(
        IDS, CENTROIDS, AREAS, 10.0) == {11: {12}, 12: {11}}


@pytest.mark.parametrize("discount", [0.0, 0.5])
def test_a_replay_with_the_discount_equals_the_decode_with_it(tmp_path, discount):
    evidence = _evidence()
    full = _calls(evidence, _engine_rule(discount))
    cache = np.load(ops_decode.save_field_cache(str(tmp_path), 1, evidence,
                                                objects=_task()))
    replay = _calls(cache, _rule(discount))
    assert replay == full
    assert full["calls"][11]["guide"] == "GTACGT"
    assert 11 not in _calls(evidence, _rule(1.0))["calls"]


def test_a_joint_grid_replay_equals_the_decode_with_the_discount(tmp_path):
    evidence = _evidence()
    matcher, decoys = ops_decode.matcher_for(GUIDES, ())
    grid = dict(spot_thresholds=[100.0, 315.0], floors=[0.0, 0.1], pixel_qualities=[0.1],
                footprint=10.0, max_distance=0, min_fraction=0.8, crosstalk="none")
    full = ops_decode._joint_field_candidates(evidence, _engine_rule(0.25), GUIDES, matcher,
                                              decoys, **grid)
    cache = np.load(ops_decode.save_field_cache(str(tmp_path), 1, evidence,
                                                objects=_task()))
    replay = ops_decode._joint_field_candidates(cache, _rule(0.25), GUIDES, matcher,
                                                decoys, **grid)
    assert replay["counts"].tobytes() == full["counts"].tobytes()
    assert [b["votes"]["primary"] for b in replay["by_spot"]] == \
        [b["votes"]["primary"] for b in full["by_spot"]]


def test_an_old_cache_still_loads_and_refuses_only_the_discount(tmp_path):
    evidence = _evidence()
    old = np.load(ops_decode.save_field_cache(str(tmp_path), 1, evidence))
    assert _calls(old, _rule(1.0)) == _calls(evidence, _rule(1.0))
    with pytest.raises(ValueError, match="written before they were stored"):
        _calls(old, _rule(0.5))
    with pytest.raises(ValueError, match="nucleus positions"):
        ops_decode.cached_neighbours(old, 10.0)


def test_the_engine_rule_keeps_its_own_neighbours():
    rule = {**_rule(0.5), "neighbours": {11: {13}}}
    assert ops_decode._rule_with_neighbours(rule, {}, 10.0) is rule
    plain = _rule(1.0)
    assert ops_decode._rule_with_neighbours(plain, {}, 10.0) is plain


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    """The synthetic well under the rule_a_s2 preset with the discount, cache off and on."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base
    from spacr import ops_engine

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_cache_positions")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1}
    ops_engine.run_ops(settings, library=library)
    captured = []
    real = ops_engine._decode_field

    def capture(task):
        out = real(task)
        captured.append((task, out))
        return out

    monkeypatch.setattr(ops_engine, "_decode_field", capture)
    runs = {}
    for name, cache in (("off", False), ("on", True)):
        captured.clear()
        preset = {**settings, "ops_decode_preset": "rule_a_s2",
                  "ops_read_neighbour_discount": 0.5, "ops_store_field_cache": cache}
        result = ops_engine.run_ops(preset, phases=("decode",), library=library)
        with sqlite3.connect(result["db"]) as conn:
            rows = conn.execute("SELECT * FROM ops_barcodes ORDER BY object_id").fetchall()
        runs[name] = {"result": result, "rows": rows, "fields": list(captured)}
    yield runs, ops_engine
    monkeypatch.undo()


def test_the_cache_changes_no_call_of_the_engine(synthetic_runs):
    runs, _ = synthetic_runs
    assert runs["on"]["rows"] and runs["on"]["rows"] == runs["off"]["rows"]
    decode = runs["on"]["result"]["wells"]["A1"]["decode"]
    assert decode["field_cache"]
    for path in decode["field_cache"]:
        assert set(ops_decode.CACHE_NUCLEUS_KEYS) <= set(np.load(path).files)


def test_an_engine_cache_replays_its_fields_with_the_discount(synthetic_runs):
    runs, ops_engine = synthetic_runs
    replayed = 0
    for task, out in runs["on"]["fields"]:
        if "cache" not in out:
            continue
        cache = np.load(out["cache"])
        assert cache["nuc_ids"].tolist() == np.asarray(task["ids"]).tolist()
        assert cache["nuc_centroids"].tobytes() == np.asarray(
            task["centroids"], np.float64).tobytes()
        rule = dict(task["rule_a"])
        auto = rule.get("auto")
        spots = list(auto.get("spot_thresholds") or ()) if auto else []
        key = "candidates" if auto else "rule_a"
        again = ops_engine._rule_a_calls(cache, rule, auto, spots, task)
        if auto:
            assert again["counts"].tobytes() == out[key]["counts"].tobytes()
        else:
            assert again["calls"] == out[key]["calls"]
        replayed += 1
    assert replayed
