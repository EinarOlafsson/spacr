"""Items N6 and N7 (schema 972): OPS storage that lets a run be redone without segmenting or decoding.

N6, ``ops_store_raw_masks``: segmentation keeps its masks as they came out,
before any pairing. A phenotype field gains ``nuclei_raw`` and ``cells_raw``
labels in its mask store, and each sequencing segmentation window is kept
in ``sbs_masks/<plate>.ome.zarr``. Pairing again from the stored phenotype
masks (:func:`spacr.ops_engine._ph_pair_stored`) and numbering the sequencing
objects again from the stored windows
(:func:`spacr.ops_engine._objects_from_stored`) give exactly what the live
run gave. Off, the default, nothing changes.

N7, ``ops_read_cache_floor``: the weakest spot the decode field cache keeps,
until now the fixed 100. At 100 the run is the run it was; lower keeps more
spots and changes no call. The cache with the stored masks and tables is
enough to redo read assignment at another footprint.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("tifffile")
pytest.importorskip("pyarrow")

from spacr import ops_decode, ops_engine, ops_settings
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION
from tests.ops_store_sql import ops_sql


def test_the_schema_row_keeps_old_files_at_the_old_behaviour():
    assert SETTINGS_SCHEMA_VERSION >= 972
    assert DEFAULT_CHANGES[972] == {"ops_store_raw_masks": False,
                                    "ops_read_cache_floor": 100.0}
    assert ops_settings.OPS_DEFAULTS["ops_store_raw_masks"] is False
    assert ops_settings.OPS_DEFAULTS["ops_read_cache_floor"] == 100.0
    assert ops_decode._CACHE_FLOOR == 100.0


def test_the_rule_carries_the_cache_floor_and_refuses_a_negative_one():
    assert ops_decode.decode_rule({})["cache_floor"] == 100.0
    assert ops_decode.decode_rule({"ops_read_cache_floor": 40})["cache_floor"] == 40.0
    with pytest.raises(ValueError, match="ops_read_cache_floor"):
        ops_decode.decode_rule({"ops_read_cache_floor": -1})


# --- N6, phenotype --------------------------------------------------------------

@pytest.fixture
def phenotype(tmp_path, monkeypatch):
    """Two planted phenotype fields segmented with the raw masks off and on."""
    import tests.test_ops_engine_phenotype_coverage as cover

    def fake(settings, gpu, key="cellpose_model"):
        return cover._ThresholdModel(grow=0 if "nucleus" in key else 5)

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    tiles = cover._write_fields(tmp_path / "pheno")
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    runs = {}
    for name, store in (("off", None), ("on", True)):
        root = tmp_path / name
        (root / "A1").mkdir(parents=True)
        db = str(root / "measurements.db")
        cover._placed(db)
        settings = dict(cover.SETTINGS)
        if store is not None:
            settings["ops_store_raw_masks"] = store
        report = ops_engine._ph_fields(db, cover.PLATE, "A1", index["A1"], settings,
                                       False, str(root), segment=True, measure=True)
        runs[name] = {"root": str(root), "report": report, "settings": settings,
                      "masks": ops_engine._ph_mask_paths(str(root), cover.PLATE, "A1",
                                                         settings)}
    return runs, index, cover


def test_raw_masks_off_writes_what_it_wrote_and_on_changes_no_output(phenotype):
    runs, _index, _cover = phenotype
    off, on = runs["off"], runs["on"]
    assert sorted(off["masks"]) == sorted(on["masks"]) == [0, 1]
    for site in off["masks"]:
        before = ops_engine._ph_read_masks(off["masks"][site])
        after = ops_engine._ph_read_masks(on["masks"][site])
        assert sorted(before) == ["cells", "nuclei"]
        assert sorted(after) == ["cells", "cells_raw", "nuclei", "nuclei_raw"]
        for name in before:
            assert before[name].dtype == after[name].dtype == np.uint32
            assert before[name].tobytes() == after[name].tobytes()
    assert off["report"]["ph_segment"]["objects"] == on["report"]["ph_segment"]["objects"]
    assert off["report"]["ph_measure"]["cells"] == on["report"]["ph_measure"]["cells"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(os.path.join(off["root"], "ops_phenotype_cells_A1.parquet")),
        pd.read_parquet(os.path.join(on["root"], "ops_phenotype_cells_A1.parquet")))


def test_the_stored_raw_labels_are_the_masks_before_pairing(phenotype):
    runs, index, _cover = phenotype
    on = runs["on"]
    channels = ops_engine._ph_channels(on["settings"])
    models = ops_engine._ph_model_pair(on["settings"], False, {})
    for site, path in on["masks"].items():
        stack = ops_engine._ph_read_field(index["A1"][min(index["A1"])][site], channels, [])
        raw: dict = {}
        nuclei, cells, _counts = ops_engine._ph_segment_field(
            stack, channels, models, on["settings"], raw=raw)
        stored = ops_engine._ph_read_masks(path)
        for role, name in ops_engine._PH_RAW_LABELS.items():
            assert stored[name].dtype == np.uint16
            assert np.array_equal(stored[name], raw[role])
        assert stored["nuclei"].tobytes() == nuclei.astype(np.uint32).tobytes()
        assert stored["cells"].tobytes() == cells.astype(np.uint32).tobytes()


def test_pairing_again_from_the_stored_masks_is_byte_identical(phenotype):
    runs, _index, _cover = phenotype
    on = runs["on"]
    for site, path in on["masks"].items():
        stored = ops_engine._ph_read_masks(path)
        nuclei, cells, counts = ops_engine._ph_pair_stored(path)
        assert nuclei.astype(np.uint32).tobytes() == stored["nuclei"].tobytes()
        assert cells.astype(np.uint32).tobytes() == stored["cells"].tobytes()
        assert counts["paired"] == int(np.unique(stored["cells"][stored["cells"] > 0]).size)


def test_pairing_from_a_field_without_raw_masks_says_why(phenotype):
    runs, _index, _cover = phenotype
    with pytest.raises(ValueError, match="ops_store_raw_masks"):
        ops_engine._ph_pair_stored(runs["off"]["masks"][0])


def test_raw_labels_take_the_smallest_lossless_dtype():
    small = {"nuclei": np.array([[0, 65535]], np.int32), "cells": np.array([[0, 1]], np.int32)}
    large = {"nuclei": np.array([[0, 65536]], np.int32), "cells": np.array([[0, 1]], np.int32)}
    assert ops_engine._ph_raw_arrays(small)["nuclei_raw"].dtype == np.uint16
    assert ops_engine._ph_raw_arrays(large)["nuclei_raw"].dtype == np.uint32
    assert int(ops_engine._ph_raw_arrays(large)["nuclei_raw"].max()) == 65536


# --- N6 sequencing windows, N7 cache floor, assignment from stored data ----------

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The synthetic sequencing well: objects and decode rerun with each storage setting."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setitem(ops_settings.OPS_DEFAULTS, "ops_raster_overlap", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_n6n7")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_threshold_mode": "manual", "ops_decode_preset": "none"}
    first = ops_engine.run_ops(settings, library=library)
    db = first["db"]
    objects_before = ops_sql(db, "SELECT * FROM ops_objects ORDER BY object_id")
    again = ops_engine.run_ops({**settings, "ops_store_raw_masks": True},
                               phases=("objects",), library=library)
    objects_after = ops_sql(db, "SELECT * FROM ops_objects ORDER BY object_id")

    captured = []
    real = ops_engine._decode_field

    def capture(task):
        out = real(task)
        captured.append((task, out))
        return out

    monkeypatch.setattr(ops_engine, "_decode_field", capture)

    def decode(extra):
        """One decode of the well with the field cache on, its rows, fields and caches."""
        captured.clear()
        preset = {**settings, "ops_decode_preset": "rule_a_s2",
                  "ops_decode_fdr_calibration": "none", "ops_store_field_cache": True,
                  **extra}
        result = ops_engine.run_ops(preset, phases=("decode",), library=library)
        rows = ops_sql(result["db"], "SELECT * FROM ops_barcodes ORDER BY object_id")
        caches = {}
        for task, out in captured:
            if "cache" in out:
                with np.load(out["cache"]) as saved:
                    caches[int(task["site"])] = {key: saved[key] for key in saved.files}
        return {"rows": rows, "fields": list(captured), "caches": caches}

    decodes = {name: decode(extra) for name, extra in {
        "base": {}, "floor100": {"ops_read_cache_floor": 100.0},
        "footprint6": {"ops_footprint": 6.0}}.items()}
    yield {"db": db, "settings": settings, "again": again, "decodes": decodes,
           "decode": decode, "objects_before": objects_before,
           "objects_after": objects_after}
    monkeypatch.undo()


def test_storing_sbs_raw_windows_changes_no_object(synthetic):
    pd.testing.assert_frame_equal(synthetic["objects_before"], synthetic["objects_after"])
    report = synthetic["again"]["wells"]["A1"]["objects"]
    assert "raw_masks" in report["seconds"]


def test_the_sbs_raw_windows_are_stored_compact_with_their_placement(synthetic):
    store = os.path.join(ops_engine._objects_raw_folder(synthetic["db"]), "synthetic.ome.zarr")
    with open(os.path.join(store, "spacr_commits.jsonl"), encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    assert entries and all(e["labels"] == ["nuclei_raw"] and len(e["window"]) == 4
                           for e in entries)
    from spacr.image_store import read_labels

    labels = read_labels(store, "nuclei_raw", key=entries[0]["key"])
    assert labels.dtype == np.uint16
    assert labels.shape == tuple(entries[0]["window"][2:])


def test_numbering_again_from_the_stored_windows_gives_the_stored_objects(synthetic):
    rebuilt = ops_engine._objects_from_stored(synthetic["db"], "synthetic", "A1")
    stored = synthetic["objects_after"]
    assert len(rebuilt) == len(stored)
    for column in rebuilt.columns:
        if column in stored.columns:
            assert np.array_equal(rebuilt[column].to_numpy(), stored[column].to_numpy()), column


def test_numbering_from_a_well_without_stored_windows_says_why(synthetic):
    with pytest.raises(ValueError, match="ops_store_raw_masks"):
        ops_engine._objects_from_stored(synthetic["db"], "synthetic", "B2")


def _same_cache(first, second):
    assert sorted(first) == sorted(second)
    for site in first:
        assert sorted(first[site]) == sorted(second[site])
        for key in first[site]:
            assert first[site][key].tobytes() == second[site][key].tobytes(), (site, key)


def test_the_floor_at_100_is_the_run_before_the_setting(synthetic):
    decodes = synthetic["decodes"]
    pd.testing.assert_frame_equal(decodes["base"]["rows"], decodes["floor100"]["rows"])
    assert decodes["base"]["caches"]
    _same_cache(decodes["base"]["caches"], decodes["floor100"]["caches"])


def test_a_lower_floor_keeps_more_spots_and_changes_no_call(synthetic):
    """Under a read threshold above some cached spots, a lower floor caches more of them.

    The cache keeps spots down to the lower of the floor and the read
    threshold, so the threshold is raised above part of the default cache
    and the two floors are compared under it: the calls are the same, and
    the lower floor's cache holds every spot of the higher one and more.
    """
    base = synthetic["decodes"]["base"]["caches"]
    strengths = np.sort(np.concatenate([c["strength"] for c in base.values()]))
    assert float(strengths[0]) > 100.0
    threshold = float(strengths[int(0.75 * (len(strengths) - 1))])
    high = float(strengths[int(0.4 * (len(strengths) - 1))])
    assert 100.0 < high < threshold
    low_run = synthetic["decode"]({"ops_read_threshold": threshold})
    high_run = synthetic["decode"]({"ops_read_threshold": threshold,
                                    "ops_read_cache_floor": high})
    pd.testing.assert_frame_equal(low_run["rows"], high_run["rows"])
    low, kept_high = low_run["caches"], high_run["caches"]
    assert sum(len(c["strength"]) for c in low.values()) > \
        sum(len(c["strength"]) for c in kept_high.values())
    for site in kept_high:
        assert float(kept_high[site]["strength"].min(initial=np.inf)) > high
        kept = low[site]["strength"] > high
        assert low[site]["peaks"][kept].tobytes() == kept_high[site]["peaks"].tobytes()
        assert np.array_equal(low[site]["strength"], base[site]["strength"])


def test_read_assignment_reruns_from_the_cache_alone_at_any_footprint(synthetic):
    """The cache of a footprint-10 run, replayed at footprint 6, gives the footprint-6 run's calls.

    The evidence does not depend on the footprint (every spot keeps its gap
    to its two nearest nuclei and the nucleus positions), so the two runs
    write the same cache, and the calls are a function of cache and rule.
    """
    decodes = synthetic["decodes"]
    _same_cache(decodes["base"]["caches"], decodes["footprint6"]["caches"])
    replayed = 0
    for task, out in decodes["footprint6"]["fields"]:
        site = int(task["site"])
        if site not in decodes["base"]["caches"]:
            continue
        cache = decodes["base"]["caches"][site]
        rule = dict(task["rule_a"])
        rule.pop("neighbours", None)
        auto = rule.get("auto")
        spots = list(auto.get("spot_thresholds") or ()) if auto else []
        again = ops_engine._rule_a_calls(cache, rule, auto, spots, task)
        key = "candidates" if auto else "rule_a"
        if auto:
            assert again["counts"].tobytes() == out[key]["counts"].tobytes()
        else:
            assert again["calls"] == out[key]["calls"]
        replayed += 1
    assert replayed
