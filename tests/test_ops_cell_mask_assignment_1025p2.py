"""Item 1025 part 2 (alpha): reads and fallback pixels go to the SBS cell they lie in.

Under ``ops_read_assignment`` 'cell_mask' every read and every fallback
pixel is owned by the SBS cell (``ops_sbs_cell_method``) whose label it
falls in, through the one ownership function of the decode,
``ops_decode._object_owners``, with the cell gaps of
``ops_sbs._cell_mask_boundaries``. The pixel bleed gate of item 1044 reads
the cells that touch as neighbours. The default 'nucleus_reach' decodes
exactly as before. The fields are painted as in the item-1044 test, run
through spaCR's own -LoG filter, read score, peaks, ``field_evidence``,
field cache and ``field_calls``; the end-to-end arm decodes the synthetic
well of the engine test with stored SBS cells.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from spacr import ops_decode, ops_engine, ops_sbs, ops_settings

GUIDE = "ACGTAC"
OTHER = "TGCATG"
LIBRARY = [GUIDE, OTHER, "CATGCA", "GTACGT"]
CYCLES = [1, 2, 3, 4, 5, 6]
SIZE = 64
RADIUS = 5.0

#: Object ids that are not their task index plus one, so the mapping from
#: cell label to object id is exercised.
IDS = np.array([11, 12, 13], np.int64)
NUCLEI = np.array([[32.0, 22.0], [32.0, 44.0], [8.0, 52.0]])

#: A's cytoplasm read 6 px beyond its disc, which the 3 px footprint misses.
CYTOPLASM = ((32.0, 11.0), GUIDE, 900.0)
#: A read 3 px beyond A's disc that lies in B's cell (B's cell grows past
#: the midpoint): the nucleus reach gives it to A, the cell mask to B.
IN_NEIGHBOUR = ((32.0, 30.0), OTHER, 900.0)
#: A pixel-only amplicon of A's read guide deep in B's touching cell, 7 px
#: beyond B's disc: beyond the 5 px pixel footprint, inside B's cell.
BLEED = ((32.0, 56.0), GUIDE, 150.0)
#: A pixel-only amplicon in C's cell; C touches no cell.
LONE = ((8.0, 52.0), OTHER, 150.0)


def _cells():
    """The task's SBS cells: label ``i + 1`` for object ``i``; A and B touch, C is apart."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cells = np.zeros((SIZE, SIZE), np.int32)
    cells[(np.hypot(yy - 32, xx - 22) <= 13) & (xx <= 29)] = 1
    cells[(np.hypot(yy - 32, xx - 44) <= 15) & (xx >= 30)] = 2
    cells[np.hypot(yy - 8, xx - 52) <= 6] = 3
    return cells


def _paint(layer, barcode, pos, amplitude, sigma=1.3):
    """One amplicon: a Gaussian per cycle on its base channel."""
    cy, cx = pos
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    for i, base in enumerate(barcode):
        layer[i, ops_sbs.BASES.index(base)] += (amplitude * spot).astype(np.float32)


def _task(cell_mask):
    """The field task the engine builds, with the cells under the cell mask."""
    task = {"cycles": CYCLES, "ids": IDS, "centroids": NUCLEI,
            "areas": np.full(len(IDS), np.pi * RADIUS ** 2),
            "owned": np.ones(len(IDS), bool)}
    if cell_mask:
        task["cell_labels"] = _cells()
    return task


def _field(spots, cell_mask, seed=3):
    """One painted field's evidence, through spaCR's read score and peaks."""
    rng = np.random.default_rng(seed)
    stack = rng.normal(40.0, 4.0, (len(CYCLES), 4, SIZE, SIZE)).astype(np.float32)
    for pos, barcode, amplitude in spots:
        _paint(stack, barcode, pos, amplitude)
    filtered = np.empty_like(stack)
    for c in range(stack.shape[0]):
        for k in range(stack.shape[1]):
            filtered[c, k] = np.clip(-ndimage.gaussian_laplace(stack[c, k], 1.0), 0, None)
    score = ops_sbs.estimate_read_locations(filtered)
    peaks = ops_sbs.find_peaks(score, min_distance=2, gpu=False)
    surface = score - ndimage.minimum_filter(score, size=5)
    strength = surface[peaks[:, 0], peaks[:, 1]]
    return ops_decode.field_evidence(stack, filtered, score, peaks, strength, CYCLES,
                                     _task(cell_mask), edge=3, floor=0.0,
                                     pixel_percentile=97.0)


def _rule(assignment, ownership="read_rule", **extra):
    """The item-1023 rule (footprint 3, territory, top share 0.9) with the pixel fallback."""
    settings = {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                "ops_footprint": 3.0, "ops_pixel_ownership": ownership,
                "ops_read_assignment": assignment, **extra}
    return ops_decode.decode_rule(ops_decode.resolve_settings(settings))


def _live(rule, evidence):
    """The rule as the engine hands it to a live decode: with the field's neighbours."""
    if ops_decode._evidence_text(evidence, "read_assignment") == "cell_mask":
        return {**rule, "neighbours": ops_decode.cached_neighbours(evidence, 10.0)}
    task = _task(False)
    return {**rule, "neighbours": ops_sbs._disc_neighbours(
        task["ids"], task["centroids"], task["areas"], 10.0)}


def _calls(evidence, rule, threshold=None):
    rule = _live(rule, evidence) if "neighbours" not in rule else rule
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    if threshold is None:
        threshold = 0.25 * float(np.max(evidence["strength"]))
    return ops_decode.field_calls(evidence, rule, LIBRARY, matcher, decoys,
                                  threshold=threshold, footprint=3.0, max_distance=0,
                                  min_fraction=0.6, single_read_quality=0.0,
                                  crosstalk="none")


def _guides(out, source=None):
    return {int(k): v["guide"] for k, v in out["calls"].items()
            if source is None or v["source"] == source}


def _cache(tmp_path, evidence, site=1):
    path = ops_decode.save_field_cache(str(tmp_path), site, evidence,
                                       objects=_task(False))
    with np.load(path) as stored:
        return path, {key: stored[key] for key in stored.files}


def test_the_default_is_the_nucleus_reach_and_adds_no_key():
    assert ops_settings.OPS_DEFAULTS["ops_read_assignment"] == "nucleus_reach"
    assert ops_decode.READ_ASSIGNMENTS[0] == "nucleus_reach"
    plain = ops_decode.decode_rule(ops_decode.resolve_settings({}))
    explicit = ops_decode.decode_rule(ops_decode.resolve_settings(
        {"ops_read_assignment": "nucleus_reach"}))
    assert plain == explicit and "read_assignment" not in plain
    assert _rule("cell_mask")["read_assignment"] == "cell_mask"
    evidence = _field([CYTOPLASM], cell_mask=False)
    assert not set(ops_decode.CELL_MASK_KEYS) & set(evidence)
    with pytest.raises(ValueError, match="ops_read_assignment must be one of"):
        ops_decode.decode_rule({"ops_read_assignment": "nearest"})


def test_the_setting_needs_sbs_cells():
    assert ops_engine._sbs_cell_options({}) is None
    assert ops_engine._sbs_cell_options({"ops_read_assignment": "nucleus_reach"}) is None
    with pytest.raises(ValueError, match="ops_sbs_cell_method is 'off'"):
        ops_engine._sbs_cell_options({"ops_read_assignment": "cell_mask"})
    reach = ops_engine._sbs_cell_options({"ops_sbs_cell_method": "watershed"})
    assert "assignment" not in reach
    cells = ops_engine._sbs_cell_options({"ops_sbs_cell_method": "watershed",
                                          "ops_read_assignment": "cell_mask"})
    assert cells["assignment"] == "cell_mask"


def test_the_cell_lookup_is_one_function():
    cells = _cells()
    points = np.array([[32, 11], [32, 30], [8, 52], [60, 2], [-4, 70]], float)
    gap, index = ops_sbs._cell_mask_boundaries(points, cells)
    assert index[:, 0].tolist() == [0, 1, 2, -1, -1]
    assert (index[:, 1] == -1).all() and np.isinf(gap[:, 1]).all()
    assert gap[:3, 0].tolist() == [-1.0, -1.0, -1.0]
    assert (gap[3:, 0] == ops_sbs._OUTSIDE_EVERY_CELL).all()
    stored = np.where(cells > 0, IDS[np.maximum(cells - 1, 0)], 0)
    pixels = ops_decode._cell_mask_pixels(
        {"pix_yx": points[:3].astype(np.int16), "owned_ids": IDS[:2]}, stored)
    assert pixels["pix_id0"].tolist() == IDS.tolist()
    assert pixels["pix_owned0"].tolist() == [True, True, False]
    assert ops_sbs._cell_neighbours(cells) == {1: {2}, 2: {1}}


def test_a_cytoplasm_read_6_px_out_goes_to_its_cell_only_under_cell_mask():
    reach = _field([CYTOPLASM], cell_mask=False)
    cell = _field([CYTOPLASM], cell_mask=True)
    assert np.array_equal(reach["peaks"], cell["peaks"])
    assert ops_decode._evidence_text(cell, "read_assignment") == "cell_mask"
    assert ops_decode._evidence_text(cell, "pix_assignment") == "cell_mask"
    assert set(np.unique(cell["cell_labels"]).tolist()) == {0, *IDS.tolist()}
    read = np.flatnonzero(np.all(np.abs(reach["peaks"] - [32, 11]) <= 1, axis=1))
    assert len(read) == 1
    assert 5.0 < float(reach["gap"][read[0], 0]) < 7.0
    assert float(cell["gap"][read[0], 0]) == -1.0 and int(cell["id0"][read[0]]) == 11
    assert 11 not in _guides(_calls(reach, _rule("nucleus_reach", ops_decode_pixel_fallback=False)))
    assert _guides(_calls(cell, _rule("cell_mask", ops_decode_pixel_fallback=False))) \
        == {11: GUIDE}


def test_a_read_in_a_touching_neighbours_cell_is_not_taken():
    spots = [CYTOPLASM, IN_NEIGHBOUR]
    rule = dict(ops_decode_pixel_fallback=False)
    reach = _guides(_calls(_field(spots, False), _rule("nucleus_reach", **rule)))
    cell = _guides(_calls(_field(spots, True), _rule("cell_mask", **rule)))
    assert reach == {11: OTHER}
    assert cell == {11: GUIDE, 12: OTHER}


def test_pixels_follow_the_cell_label_with_the_bleed_gate():
    spots = [CYTOPLASM, BLEED, LONE]
    evidence = _field(spots, cell_mask=True)
    cells = _cells()
    yx = np.asarray(evidence["pix_yx"], np.int64)
    expected = IDS[cells[yx[:, 0], yx[:, 1]] - 1]
    assert (cells[yx[:, 0], yx[:, 1]] > 0).all()
    assert np.array_equal(evidence["pix_id0"], expected)
    assert (np.asarray(evidence["pix_id1"]) == -1).all()
    gated = _calls(evidence, _rule("cell_mask"))
    assert _guides(gated, "primary") == {11: GUIDE}
    assert _guides(gated, "pixel_fallback") == {13: OTHER}
    assert gated["pixel_refused"]["bleed_read"] == 1
    ungated = _calls(evidence, _rule("cell_mask", "read_reach"))
    assert _guides(ungated, "pixel_fallback") == {12: GUIDE, 13: OTHER}
    reach = _calls(_field(spots, cell_mask=False), _rule("nucleus_reach"))
    assert 12 not in _guides(reach)


def test_the_neighbours_under_cell_mask_are_the_cells_that_touch():
    evidence = _field([CYTOPLASM], cell_mask=True)
    assert ops_decode.cached_neighbours(evidence, 0.0) == {11: {12}, 12: {11}}
    assert ops_decode.cached_neighbours(evidence, 100.0) == {11: {12}, 12: {11}}
    reach = _field([CYTOPLASM], cell_mask=False)
    rule = _rule("nucleus_reach")
    with_cells = ops_decode._rule_with_pixel_neighbours(
        _rule("cell_mask"), {**evidence, "nuc_ids": IDS}, 3.0)
    assert with_cells["neighbours"] == {11: {12}, 12: {11}}
    stripped = {k: v for k, v in evidence.items() if k != "cell_labels"}
    with pytest.raises(ValueError, match="cell_labels"):
        ops_decode.cached_neighbours(stripped, 10.0)
    with pytest.raises(ValueError, match="nuc_ids"):
        ops_decode._rule_with_pixel_neighbours(rule, reach, 3.0)


def test_evidence_and_rule_must_agree_on_the_assignment():
    reach = _field([CYTOPLASM], cell_mask=False)
    cell = _field([CYTOPLASM], cell_mask=True)
    with pytest.raises(ValueError, match="no read_assignment 'cell_mask', no cell_labels"):
        _calls(reach, _rule("cell_mask", ops_decode_pixel_fallback=False))
    with pytest.raises(ValueError, match="set ops_read_assignment 'cell_mask'"):
        _calls(cell, _rule("nucleus_reach", ops_decode_pixel_fallback=False))


def _replay(folder, assignment, **kw):
    return ops_decode.decode_cached_well(
        str(folder), {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                      "ops_footprint": 3.0, "ops_read_assignment": assignment},
        LIBRARY, read_threshold=kw.pop("threshold"), single_read_quality=0.0,
        pixel_quality=0.1, **kw)


def test_a_cell_mask_cache_replays_its_live_calls(tmp_path):
    spots = [CYTOPLASM, IN_NEIGHBOUR, BLEED, LONE]
    evidence = _field(spots, cell_mask=True)
    threshold = 0.25 * float(np.max(evidence["strength"]))
    _cache(tmp_path / "decode_cache", evidence)
    out = _replay(tmp_path / "decode_cache", "cell_mask", threshold=threshold)
    rule = out["rule"]
    assert rule["read_assignment"] == "cell_mask"
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    live = ops_decode.field_calls(
        evidence, {**rule, "neighbours": ops_decode.cached_neighbours(evidence, 10.0)},
        LIBRARY, matcher, decoys, threshold=threshold, footprint=rule["footprint"],
        max_distance=rule["max_distance"], min_fraction=rule["min_fraction"],
        single_read_quality=0.0, crosstalk="none")
    assert repr(out["fields"][0]["rule_a"]["calls"]) == repr(live["calls"])
    assert _guides(live) == {11: GUIDE, 12: OTHER, 13: OTHER}


def _stored_cells(tmp_path, well, site):
    """The SBS cells as the decode stores them: sbs_cell labels in sbs/fov.ome.zarr."""
    from spacr.image_store import PlateWriter, _compact_labels, _merged_store_path

    folder = str(tmp_path / "sbs")
    writer = PlateWriter(folder, "synthetic", store_path=_merged_store_path(folder),
                         shards=True)
    lut = np.concatenate([np.zeros(1, np.int64), IDS])
    writer.write_field(well, site, None, {"sbs_cell": _compact_labels(lut[_cells()])})
    return str(writer.path)


def test_a_nucleus_reach_cache_replays_under_the_stored_cells(tmp_path):
    spots = [CYTOPLASM, IN_NEIGHBOUR, BLEED, LONE]
    reach = _field(spots, cell_mask=False)
    threshold = 0.25 * float(np.max(reach["strength"]))
    folder = tmp_path / "A1" / "decode_cache"
    _cache(folder, reach, site=7)
    store = _stored_cells(tmp_path, "A1", 7)
    replayed = _replay(folder, "cell_mask", threshold=threshold, cell_store=store)
    cell_cache = tmp_path / "cell" / "decode_cache"
    _cache(cell_cache, _field(spots, cell_mask=True), site=7)
    live = _replay(cell_cache, "cell_mask", threshold=threshold)
    got = replayed["fields"][0]["rule_a"]
    assert _guides(got, "primary") == _guides(live["fields"][0]["rule_a"], "primary") \
        == {11: GUIDE, 12: OTHER}
    assert got["reads_owned"] == live["fields"][0]["rule_a"]["reads_owned"]
    as_run = _replay(folder, "nucleus_reach", threshold=threshold)
    assert _guides(as_run["fields"][0]["rule_a"], "primary") == {11: OTHER}


def test_a_cache_without_cell_labels_fails_naming_them(tmp_path):
    reach = _field([CYTOPLASM], cell_mask=False)
    folder = tmp_path / "A1" / "decode_cache"
    _cache(folder, reach, site=7)
    with pytest.raises(FileNotFoundError, match="holds no cell_labels; pass cell_store"):
        _replay(folder, "cell_mask", threshold=1.0)
    store = _stored_cells(tmp_path, "A1", 8)
    with pytest.raises(FileNotFoundError, match="no sbs_cell labels for well A1 site 7"):
        _replay(folder, "cell_mask", threshold=1.0, cell_store=store)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The engine's synthetic well decoded with stored SBS cells, both assignments."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base
    from tests.ops_store_sql import ops_connect
    from tests.test_sbs_cells_1025 import _ProbabilityModel

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setitem(ops_settings.OPS_DEFAULTS, "ops_raster_overlap", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    model = _ProbabilityModel()

    def fake_model(settings, gpu, key="ops_sbs_nucleus_model"):
        return model if key == "ops_sbs_cell_model" else base._ThresholdModel()

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake_model)
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    root = tmp_path_factory.mktemp("cell_mask_1025p2")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "plate": "synthetic", "ops_gpu": False,
                "n_workers": 1, "ops_decode_threshold_mode": "manual",
                "ops_decode_preset": "none"}
    captured = []
    real = ops_engine._decode_field

    def capture(task):
        out = real(task)
        captured.append((task, out))
        return out

    runs = {}

    def run(name, **extra):
        out = root / name
        ops_engine.run_ops({**settings, "dst_root": str(out)}, library=library)
        monkeypatch.setattr(ops_engine, "_decode_field", capture)
        captured.clear()
        result = ops_engine.run_ops({**settings, "dst_root": str(out), **extra},
                                    phases=("decode",), library=library)
        monkeypatch.setattr(ops_engine, "_decode_field", real)
        with ops_connect(result["db"]) as conn:
            rows = conn.execute("SELECT * FROM ops_barcodes ORDER BY object_id").fetchall()
        runs[name] = {"result": result, "rows": rows, "fields": list(captured),
                      "settings": {**settings, "dst_root": str(out), **extra}}

    rule = {"ops_decode_preset": "rule_a_s2", "ops_decode_fdr_calibration": "none",
            "ops_store_field_cache": True}
    run("rule_reach", **rule, ops_sbs_cell_method="watershed")
    run("rule_cell", **rule, ops_sbs_cell_method="watershed",
        ops_read_assignment="cell_mask")
    run("legacy_reach", ops_sbs_cell_method="watershed")
    run("legacy_cell", ops_sbs_cell_method="watershed", ops_read_assignment="cell_mask")
    yield runs, library
    monkeypatch.undo()


def test_the_engine_gives_the_cells_to_the_decode(synthetic):
    runs, _library = synthetic
    for name in ("rule_cell", "legacy_cell"):
        assert runs[name]["rows"]
        report = runs[name]["result"]["wells"]["A1"]["decode"]
        assert report["sbs_cells"]["read_assignment"] == "cell_mask"
        for task, out in runs[name]["fields"]:
            if "skipped" in out:
                continue
            assert task["sbs_cells"]["assignment"] == "cell_mask"
    for name in ("rule_reach", "legacy_reach"):
        report = runs[name]["result"]["wells"]["A1"]["decode"]
        assert "read_assignment" not in report["sbs_cells"]
        assert "read_assignment" not in report.get("decode_rules", {})
    assert runs["rule_cell"]["result"]["wells"]["A1"]["decode"]["decode_rules"][
        "read_assignment"] == "cell_mask"


def test_one_call_per_object_under_cell_mask(synthetic):
    from tests.ops_store_sql import ops_connect

    runs, _library = synthetic
    for name in ("rule_cell", "legacy_cell"):
        with ops_connect(runs[name]["result"]["db"]) as conn:
            counts = conn.execute("SELECT object_id, COUNT(*) FROM ops_barcodes "
                                  "GROUP BY object_id HAVING COUNT(*) > 1").fetchall()
        assert runs[name]["rows"] and not counts


def test_the_stored_run_replays_under_its_cells(synthetic):
    import json
    import os

    runs, library = synthetic
    for name in ("rule_reach", "rule_cell"):
        result = runs[name]["result"]
        decode = result["wells"]["A1"]["decode"]
        folder = os.path.join(os.path.dirname(os.path.abspath(result["db"])), "A1",
                              "decode_cache")
        settings = runs[name]["settings"]
        replay = ops_decode.decode_cached_well(
            folder, {**settings, "ops_read_assignment": "cell_mask"}, library,
            read_threshold=decode["read_threshold"],
            single_read_quality=decode["decode_rules"]["single_read_quality"],
            pixel_quality=decode["decode_rules"]["pixel_quality"],
            cell_store=decode["sbs_cells"]["store"])
        runs[name]["replay"] = replay
        assert replay["fields"]
        json.dumps(replay["split_read_concordance"], default=str)
    live = {r["site"]: r["rule_a"]["calls"]
            for _task, r in runs["rule_cell"]["fields"] if "rule_a" in r}
    assert repr({f["site"]: f["rule_a"]["calls"]
                 for f in runs["rule_cell"]["replay"]["fields"]}) == repr(live)
    primary = [{k: v["guide"] for k, v in f["rule_a"]["calls"].items()
                if v["source"] == "primary"} for f in runs["rule_cell"]["replay"]["fields"]]
    from_reach = [{k: v["guide"] for k, v in f["rule_a"]["calls"].items()
                   if v["source"] == "primary"} for f in runs["rule_reach"]["replay"]["fields"]]
    assert primary == from_reach
