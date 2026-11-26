"""Item 1052 (alpha): ``ops_read_assignment`` 'nucleus_first', the bleed-gated cell-mask fallback.

Every nucleus-reach call stands. Only an object the nucleus reach leaves
uncalled is voted again on the reads of its SBS cell, and that call is kept
only when the bleed gate passes it: the bleed curve (arm 2 of the paper,
ported in ``spacr.ops_decode_qc_panels``) gives each agreeing read the
excess rate at which a read at its distance from a neighbour carrying the
guide, and at its spot position, names the guide of that neighbour. The pixel
fallback is the 1044b path of 'nucleus_reach', byte for byte (item 1052b),
and a cell with a pixel call takes no cell-mask call. The fields
are painted, run through the spaCR -LoG filter, read score, peaks and
``field_evidence``, and given their cells as the live decode gives them
(``ops_engine._with_fallback_cells``).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from scipy import ndimage

from spacr import ops_decode, ops_decode_qc_panels as panels, ops_engine, ops_sbs, ops_settings
from tests.test_ops_cell_mask_assignment_1025p2 import _paint

GUIDE = "ACGTAC"
OTHER = "TGCATG"
LIBRARY = [GUIDE, OTHER, "CATGCA", "GTACGT"]
CYCLES = [1, 2, 3, 4, 5, 6]
SIZE = 64
RADIUS = 5.0
IDS = np.array([11, 12, 13], np.int64)
NUCLEI = np.array([[32.0, 20.0], [32.0, 36.0], [10.0, 52.0]])
A, B, C = (int(i) for i in IDS)

#: Two reads of the guide inside the disc of B: the nucleus call of B.
B_READS = [((32.0, 36.0), GUIDE, 900.0), ((29.0, 38.0), GUIDE, 900.0)]
#: Two reads of the guide of B in the cell of A, 5 px from both discs: beyond the
#: 3 px footprint of either, so the nucleus reach leaves A uncalled.
A_BLEED = [((26.0, 28.0), GUIDE, 900.0), ((38.0, 28.0), GUIDE, 900.0)]
#: One read of another guide in the cytoplasm of C, 3.5 px from its disc, 22 px
#: from B: no neighbour carries it.
C_CLEAN = [((16.0, 58.0), OTHER, 900.0)]
#: The nucleus reads of A and three cytoplasm reads of another guide in its cell.
A_NUCLEUS = [((32.0, 20.0), GUIDE, 900.0), ((29.0, 18.0), GUIDE, 900.0)]
A_CYTOPLASM = [((32.0, 9.0), OTHER, 900.0), ((24.0, 14.0), OTHER, 900.0),
               ((40.0, 14.0), OTHER, 900.0)]
#: A pixel-only amplicon of the guide of B 4 px out of the disc of A, toward B.
A_PIXEL_BLEED = [((38.0, 27.0), GUIDE, 150.0)]
#: A pixel-only amplicon of the guide of B at the centre of A, 11 px from the disc of B.
A_PIXEL_CENTRE = [((32.0, 20.0), GUIDE, 150.0)]
#: A pixel-only amplicon in the disc of C, far from every carrier.
C_PIXEL = [((10.0, 52.0), OTHER, 150.0)]


def _cells():
    """The SBS cells, label ``i + 1`` for object ``i``: A and B touch at x 28 | 29."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cells = np.zeros((SIZE, SIZE), np.int32)
    cells[(np.hypot(yy - 32, xx - 20) <= 13) & (xx <= 28)] = 1
    cells[(np.hypot(yy - 32, xx - 36) <= 13) & (xx >= 29)] = 2
    cells[(np.hypot(yy - 10, xx - 52) <= 12) & (cells == 0)] = 3
    return cells


def _task():
    return {"cycles": CYCLES, "ids": IDS, "centroids": NUCLEI,
            "areas": np.full(len(IDS), np.pi * RADIUS ** 2), "owned": np.ones(len(IDS), bool)}


def _field(spots, seed=3, cells=True):
    """One painted field: nucleus-reach evidence, with its cells as the live decode adds them."""
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
    evidence = ops_decode.field_evidence(stack, filtered, score, peaks, strength, CYCLES,
                                         _task(), edge=3, floor=0.0, pixel_percentile=97.0)
    if cells:
        evidence = ops_engine._with_fallback_cells(
            evidence, {**_task(), "fallback_cells": _cells()})
    return evidence


def _rule(assignment, **extra):
    settings = {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                "ops_footprint": 3.0, "ops_read_assignment": assignment,
                "ops_nucleus_first_bleed_curve": "arm2", **extra}
    return ops_decode.decode_rule(ops_decode.resolve_settings(settings))


def _calls(evidence, rule, threshold=50.0):
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    if ops_decode._read_assignment_of(rule) == "cell_mask":
        evidence = ops_decode._cell_mask_evidence(evidence, evidence["cell_labels"])
    return ops_decode.field_calls(evidence, rule, LIBRARY, matcher, decoys,
                                  threshold=threshold, footprint=3.0, max_distance=0,
                                  min_fraction=0.6, single_read_quality=0.0, crosstalk="none")


def _guides(out, source=None):
    return {int(k): v["guide"] for k, v in out["calls"].items()
            if source is None or v["source"] == source}


def test_the_setting_is_opt_in_and_registered():
    from spacr import settings

    assert ops_settings.OPS_DEFAULTS["ops_read_assignment"] == "nucleus_reach"
    assert "nucleus_first" in ops_decode.READ_ASSIGNMENTS
    plain = ops_decode.decode_rule(ops_decode.resolve_settings({}))
    assert "read_assignment" not in plain and "bleed_gate" not in plain
    rule = _rule("nucleus_first")
    assert rule["read_assignment"] == "nucleus_first"
    assert rule["bleed_gate"] == {"curve": "arm2", "max_share": 0.5, "min_reads": 1.0}
    assert rule["bleed_model"]["excess"] == panels.ARM2_BLEED_MODEL["excess"]
    fit = _rule("nucleus_first", ops_nucleus_first_bleed_curve="fit")
    assert "bleed_model" not in fit
    feature = settings.ALPHA_FEATURES[1052]
    assert set(feature["settings"]) == {"ops_nucleus_first_bleed_curve",
                                        "ops_nucleus_first_max_bleed",
                                        "ops_nucleus_first_min_reads"}
    assert feature["choices"] == {"ops_read_assignment": ("nucleus_first",)}
    for key in feature["settings"]:
        assert key in ops_settings.OPS_DEFAULTS and key in ops_settings.OPS_TOOLTIPS
        assert ops_settings.OPS_DEFAULTS[key] == ops_decode.NUCLEUS_FIRST_DEFAULTS[key]
    assert len(ops_settings.OPS_CATEGORIES["SBS cells α"]) <= 20
    with pytest.raises(ValueError, match="decode rules"):
        ops_decode.decode_rule({"ops_read_assignment": "nucleus_first",
                                "ops_nucleus_first_bleed_curve": "arm2"})
    with pytest.raises(ValueError, match="must lie in"):
        _rule("nucleus_first", ops_nucleus_first_max_bleed=1.5)
    with pytest.raises(FileNotFoundError, match="no bleed curve at"):
        _rule("nucleus_first", ops_nucleus_first_bleed_curve="/no/such/curve.json")


def test_a_live_decode_needs_sbs_cells_manual_thresholds_and_a_stored_curve():
    with pytest.raises(ValueError, match="ops_sbs_cell_method is 'off'"):
        ops_engine._sbs_cell_options({"ops_read_assignment": "nucleus_first"})
    base = {"ops_sbs_cell_method": "watershed", "ops_read_assignment": "nucleus_first"}
    with pytest.raises(ValueError, match="'fit' fits the bleed curve"):
        ops_engine._sbs_cell_options({**base, "ops_decode_threshold_mode": "manual"})
    with pytest.raises(ValueError, match="fixed thresholds"):
        ops_engine._sbs_cell_options({**base, "ops_decode_threshold_mode": "auto",
                                      "ops_nucleus_first_bleed_curve": "arm2"})
    options = ops_engine._sbs_cell_options({**base, "ops_decode_threshold_mode": "manual",
                                            "ops_nucleus_first_bleed_curve": "arm2"})
    assert options["assignment"] == "nucleus_first"


def test_a_nucleus_call_with_conflicting_cytoplasm_reads_is_unchanged():
    evidence = _field(A_NUCLEUS + A_CYTOPLASM)
    reach = _calls(evidence, _rule("nucleus_reach"))
    first = _calls(evidence, _rule("nucleus_first"))
    cell = _calls(evidence, _rule("cell_mask"))
    assert _guides(reach) == {A: GUIDE}
    assert _guides(first) == {A: GUIDE}
    assert "assignment" not in first["calls"][A]
    assert _guides(cell).get(A) != GUIDE
    for key in ("guide", "n_agreeing", "fraction", "quality", "barcode", "n_reads"):
        assert first["calls"][A][key] == reach["calls"][A][key]


def test_an_uncalled_cell_with_clean_cytoplasm_reads_is_called():
    evidence = _field(B_READS + C_CLEAN)
    reads = dict(ops_decode_pixel_fallback=False)
    reach = _calls(evidence, _rule("nucleus_reach", **reads))
    first = _calls(evidence, _rule("nucleus_first", **reads))
    assert _guides(reach) == {B: GUIDE}
    assert _guides(first) == {B: GUIDE, C: OTHER}
    assert first["calls"][C]["assignment"] == "cell_mask"
    assert first["calls"][C]["source"] == "primary"
    assert first["assigned_cell_mask_real"] == 1
    assert first["nucleus_first"]["cell_mask_called"] == 1


def test_cytoplasm_reads_of_a_touching_neighbours_guide_at_bleed_distance_are_refused():
    evidence = _field(B_READS + A_BLEED)
    rule = _rule("nucleus_first")
    first = _calls(evidence, rule)
    cell = _calls(evidence, _rule("cell_mask"))
    assert _guides(cell).get(A) == GUIDE
    assert _guides(first) == {B: GUIDE}
    assert first["nucleus_first"]["read_refused"]["bleed_model"] == 1
    loose = _calls(evidence, _rule("nucleus_first", ops_nucleus_first_max_bleed=1.0,
                                   ops_nucleus_first_min_reads=0.0))
    assert _guides(loose) == {B: GUIDE, A: GUIDE}


def test_the_minimum_agreeing_reads_is_a_second_condition():
    evidence = _field(B_READS + C_CLEAN)
    strict = _calls(evidence, _rule("nucleus_first", ops_nucleus_first_min_reads=2.0,
                                    ops_decode_pixel_fallback=False))
    assert _guides(strict) == {B: GUIDE}
    assert strict["nucleus_first"]["read_refused"]["min_support"] == 1


def _pixel_rows(out):
    """The pixel calls of a result, with every field they hold."""
    return {int(k): v for k, v in out["calls"].items() if v["source"] == "pixel_fallback"}


def test_pixel_calls_are_those_of_nucleus_reach_and_stand():
    for spots, seed in ((B_READS + A_PIXEL_BLEED + C_PIXEL, 3), (B_READS + A_PIXEL_CENTRE, 4),
                        (B_READS + A_BLEED + C_CLEAN + C_PIXEL, 5)):
        evidence = _field(spots, seed=seed)
        reach = _calls(evidence, _rule("nucleus_reach"))
        first = _calls(evidence, _rule("nucleus_first"))
        assert repr(_pixel_rows(first)) == repr(_pixel_rows(reach))
        assert first["pixel_refused"] == reach["pixel_refused"]
        reads = {o: v for o, v in first["calls"].items() if v["source"] == "primary"}
        nucleus = {o: v for o, v in reach["calls"].items() if v["source"] == "primary"}
        assert {o: v for o, v in reads.items() if o in nucleus} == nucleus
        assert all(v.get("assignment") == "cell_mask"
                   for o, v in reads.items() if o not in nucleus)
        assert not set(reads) & set(_pixel_rows(first))
    centre = _field(B_READS + A_PIXEL_CENTRE)
    assert reach_refuses(centre)
    assert A not in _guides(_calls(centre, _rule("nucleus_first")))


def test_a_cell_with_a_pixel_call_takes_no_cell_mask_call(monkeypatch):
    seen = []
    real = ops_decode._bleed_probabilities

    def spy(evidence, points, gap, ids, owner, guide, carriers, sisters, model):
        seen.append(owner)
        return real(evidence, points, gap, ids, owner, guide, carriers, sisters, model)

    monkeypatch.setattr(ops_decode, "_bleed_probabilities", spy)
    evidence = _field(B_READS + C_CLEAN)
    reach = _calls(evidence, _rule("nucleus_reach"))
    assert _guides(reach, "pixel_fallback") == {C: OTHER}
    first = _calls(evidence, _rule("nucleus_first"))
    assert repr(first["calls"]) == repr(reach["calls"])
    assert first["nucleus_first"]["cell_mask_candidates"] == 0 and C not in seen
    without = _calls(evidence, _rule("nucleus_first", ops_decode_pixel_fallback=False))
    assert without["calls"][C]["assignment"] == "cell_mask" and C in seen


def reach_refuses(evidence):
    """The 1044b gate of 'nucleus_reach' refuses the pixel call of A naming the guide of B."""
    out = _calls(evidence, _rule("nucleus_reach"))
    return A not in _guides(out) and out["pixel_refused"]["bleed_read"] == 1


def test_the_bleed_probability_follows_the_curve():
    model = panels.ARM2_BLEED_MODEL
    rates = panels.bleed_rates(model, np.array([5.0, 12.0, 12.0]), np.array([5.0, -1.0, 1.0]))
    assert rates.tolist() == [model["excess"][3][2], model["excess"][5][0],
                              model["excess"][5][1]]
    evidence = _field(B_READS)
    gap = np.array([[5.0, 5.0], [5.0, 5.0]])
    ids = np.array([[A, B], [B, A]])
    points = np.array([[26.0, 28.0], [38.0, 28.0]])
    carriers = {B: frozenset([GUIDE])}
    p = ops_decode._bleed_probabilities(evidence, points, gap, ids, A, GUIDE, carriers, {}, model)
    assert np.allclose(p, model["excess"][3][2])
    assert not ops_decode._bleed_probabilities(evidence, points, gap, ids, A, OTHER, carriers,
                                               {}, model).any()
    assert not ops_decode._bleed_probabilities(evidence, points, gap, ids, A, GUIDE, carriers,
                                               {A: {B}}, model).any()


def test_stored_curves_load_in_every_form(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({"gbin": rng.integers(0, 6, n), "pbin": rng.integers(0, 3, n),
                          "sister": rng.random(n) < 0.2, "has_topj": rng.random(n) < 0.8,
                          "match": rng.random(n) < 0.4, "null_sum": rng.integers(0, 2, n)})
    path = tmp_path / "bleed_pairs.parquet"
    frame.to_parquet(path)
    model = panels.load_bleed_model(str(path))
    keep = frame.has_topj & ~frame.sister
    cell = frame[keep & (frame.gbin == 3) & (frame.pbin == 1)]
    assert model["pairs"][3][1] == len(cell)
    assert np.isclose(model["excess"][3][1],
                      cell.match.mean() - cell.null_sum.mean() / panels.NULL_DRAWS, atol=1e-6)
    stored = tmp_path / "model.json"
    stored.write_text(json.dumps(model))
    assert panels.load_bleed_model(str(stored))["excess"] == model["excess"]
    null = np.zeros((10, int(keep.sum())), bool)
    pairs = frame[keep].drop(columns="null_sum").assign(tile=0)
    summary = panels.bleed_summary(pairs, null, np.zeros(len(pairs), int), 0)
    table = tmp_path / "bleed_panel_A1.parquet"
    summary.to_parquet(table)
    separable = panels.load_bleed_model(str(table))
    assert np.asarray(separable["excess"]).shape == (6, 3)
    arm2 = {"sisters_excluded": {
        "overall": {"bleed_pct": 50.0, "pairs": 100},
        "by_read_to_neighbour_gap": {name: {"bleed_pct": 40.0, "pairs": 100}
                                     for name in panels.GAP_NAMES},
        "by_spot_position": {name: {"bleed_pct": 25.0, "pairs": 100}
                             for name in panels.POSITION_NAMES}}}
    (tmp_path / "arm2_bleed.json").write_text(json.dumps(arm2))
    assert np.allclose(panels.load_bleed_model(str(tmp_path / "arm2_bleed.json"))["excess"],
                       0.4 * 0.25 / 0.5)
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="is not a bleed curve"):
        panels.load_bleed_model(str(tmp_path / "other.json"))


def test_the_arm2_constant_is_the_fit_of_the_measured_pairs():
    path = ("/media/carruthers/mnt1/codex_inbox/specs/1052_nucleus_first/arm2/"
            "bleed_pairs.parquet")
    if not os.path.isfile(path):
        pytest.skip("the arm-2 pairs of the paper are not on this machine")
    fitted = panels.load_bleed_model(path)
    assert np.allclose(fitted["excess"], panels.ARM2_BLEED_MODEL["excess"], atol=1e-6)
    assert fitted["pairs"] == panels.ARM2_BLEED_MODEL["pairs"]


def _save(folder, evidence, site):
    return ops_decode.save_field_cache(str(folder), site, evidence, objects=_task())


def _stored_cells(tmp_path, well, site):
    from spacr.image_store import PlateWriter, _compact_labels, _merged_store_path

    folder = str(tmp_path / "sbs")
    writer = PlateWriter(folder, "synthetic", store_path=_merged_store_path(folder),
                         shards=True)
    lut = np.concatenate([np.zeros(1, np.int64), IDS])
    writer.write_field(well, site, None, {"sbs_cell": _compact_labels(lut[_cells()])})
    return str(writer.path)


def _replay(folder, **kw):
    settings = {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                "ops_footprint": 3.0, "ops_read_assignment": "nucleus_first",
                "ops_nucleus_first_bleed_curve": kw.pop("curve", "arm2")}
    return ops_decode.decode_cached_well(str(folder), settings, LIBRARY, read_threshold=50.0,
                                         single_read_quality=0.0, pixel_quality=0.1, **kw)


def test_replay_from_a_nucleus_reach_cache_and_stored_cells_equals_live(tmp_path):
    spots = B_READS + A_BLEED + C_CLEAN + C_PIXEL
    live_evidence = _field(spots)
    reach_evidence = _field(spots, cells=False)
    folder = tmp_path / "A1" / "decode_cache"
    _save(folder, reach_evidence, 7)
    store = _stored_cells(tmp_path, "A1", 7)
    replayed = _replay(folder, cell_store=store)
    rule = replayed["rule"]
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    live = ops_decode.field_calls(
        live_evidence, rule, LIBRARY, matcher, decoys, threshold=50.0,
        footprint=rule["footprint"], max_distance=rule["max_distance"],
        min_fraction=rule["min_fraction"], single_read_quality=0.0, crosstalk="none")
    got = replayed["fields"][0]["rule_a"]
    assert repr(got["calls"]) == repr(live["calls"])
    assert got["nucleus_first"] == live["nucleus_first"]
    assert _guides(got, "primary") == {B: GUIDE}
    assert _guides(got, "pixel_fallback") == {C: OTHER}
    own = tmp_path / "own" / "decode_cache"
    _save(own, live_evidence, 7)
    assert repr(_replay(own)["fields"][0]["rule_a"]["calls"]) == repr(live["calls"])


def test_a_missing_input_fails_by_name(tmp_path):
    folder = tmp_path / "A1" / "decode_cache"
    _save(folder, _field(B_READS, cells=False), 7)
    with pytest.raises(FileNotFoundError, match="holds no cell_labels; pass cell_store"):
        _replay(folder)
    store = _stored_cells(tmp_path, "A1", 8)
    with pytest.raises(FileNotFoundError, match="no sbs_cell labels for well A1 site 7"):
        _replay(folder, cell_store=store)
    with pytest.raises(ValueError, match="holds no cell_labels"):
        _calls(_field(B_READS, cells=False), _rule("nucleus_first"))
    with pytest.raises(ValueError, match="needs a bleed curve"):
        _calls(_field(B_READS), {k: v for k, v in _rule("nucleus_first").items()
                                 if k != "bleed_model"})
    cell = ops_decode._cell_mask_evidence(_field(B_READS), _cells() * 0 + np.where(
        _cells() > 0, IDS[np.maximum(_cells() - 1, 0)], 0))
    with pytest.raises(ValueError, match="written under 'nucleus_reach'"):
        _calls(cell, _rule("nucleus_first"))


def test_fit_replays_with_the_curve_of_the_run(tmp_path):
    folder = tmp_path / "A1" / "decode_cache"
    for site, seed in ((1, 3), (2, 4), (3, 5)):
        b_other = [((32.0, 36.0), OTHER, 900.0), ((29.0, 38.0), OTHER, 900.0)]
        minority = [((32.0, 30.0), GUIDE, 900.0)]
        _save(folder, _field(A_NUCLEUS + b_other + minority + C_CLEAN, seed=seed), site)
    out = _replay(folder, curve="fit")
    model = out["rule"]["bleed_model"]
    assert model["form"] == panels.BLEED_MODEL_FORM
    assert model["source"].startswith("fit on 3 field caches")
    stripped = tmp_path / "bare" / "A1" / "decode_cache"
    evidence = {k: v for k, v in _field(B_READS).items()
                if k not in ops_decode.CACHE_NUCLEUS_KEYS}
    os.makedirs(stripped)
    np.savez_compressed(stripped / "site0001.npz", **evidence)
    with pytest.raises(ValueError, match="nucleus positions"):
        _replay(stripped, curve="fit")


def _with_nuclei(evidence):
    """The evidence with the nucleus positions a field cache stores."""
    task = _task()
    return {**evidence, "nuc_ids": task["ids"],
            "nuc_centroids": np.asarray(task["centroids"], np.float64),
            "nuc_areas": np.asarray(task["areas"], np.float64)}


def _golden(evidence_sets):
    """A digest of every nucleus-reach result the item touches."""
    digest = hashlib.sha256()
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    for evidence in evidence_sets:
        for ownership in ("read_rule", "read_reach", "nearest_nucleus"):
            for extra in ({}, {"ops_min_top_share": 0.9}, {"ops_read_neighbour_discount": 1.0},
                          {"ops_decode_pixel_fallback": False}):
                rule = ops_decode.decode_rule(ops_decode.resolve_settings({
                    "ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                    "ops_footprint": 3.0, "ops_pixel_ownership": ownership, **extra}))
                out = ops_decode.field_calls(
                    evidence, rule, LIBRARY, matcher, decoys, threshold=50.0, footprint=3.0,
                    max_distance=0, min_fraction=0.6, single_read_quality=0.0,
                    crosstalk="none", evidence_rows=True)
                digest.update(repr(sorted(rule.items(), key=str)).encode())
                digest.update(repr({k: out[k] for k in sorted(out)}).encode())
    return digest.hexdigest()


#: The digest of :func:`_golden` on origin/nightly 567270bacc, before item 1052.
GOLDEN = "f7be7f4e93e94a7d2410f86933ec135dbb6ab9b9cd5ad3b141e4d295ee1e0f39"


def test_nucleus_reach_is_byte_identical():
    sets = [_with_nuclei(_field(spots, seed=seed, cells=False)) for spots, seed in (
        (B_READS + A_BLEED + C_CLEAN + C_PIXEL, 3), (A_NUCLEUS + A_CYTOPLASM, 4),
        (B_READS + A_PIXEL_BLEED + A_PIXEL_CENTRE + C_PIXEL, 5))]
    assert _golden(sets) == GOLDEN


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The synthetic well of the engine decoded with stored SBS cells, under both assignments."""
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
    root = tmp_path_factory.mktemp("nucleus_first_1052")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "plate": "synthetic", "ops_gpu": False,
                "n_workers": 1, "ops_decode_threshold_mode": "manual",
                "ops_decode_preset": "rule_a_s2", "ops_decode_fdr_calibration": "none",
                "ops_store_field_cache": True, "ops_sbs_cell_method": "watershed"}
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

    run("reach")
    run("first", ops_read_assignment="nucleus_first", ops_nucleus_first_bleed_curve="arm2")
    yield runs, library
    monkeypatch.undo()


def _well_replay(runs, name, library, **extra):
    decode = runs[name]["result"]["wells"]["A1"]["decode"]
    folder = os.path.join(os.path.dirname(os.path.abspath(runs[name]["result"]["db"])), "A1",
                          "decode_cache")
    settings = {**runs[name]["settings"], "ops_read_assignment": "nucleus_first",
                "ops_nucleus_first_bleed_curve": "arm2", **extra}
    return ops_decode.decode_cached_well(
        folder, settings, library, read_threshold=decode["read_threshold"],
        single_read_quality=decode["decode_rules"]["single_read_quality"],
        pixel_quality=decode["decode_rules"]["pixel_quality"],
        cell_store=decode["sbs_cells"]["store"] if extra.pop("store", True) else None)


def test_the_live_engine_decode_replays_from_its_cache_and_from_the_reach_cache(synthetic):
    from tests.ops_store_sql import ops_connect

    runs, library = synthetic
    first = runs["first"]["result"]["wells"]["A1"]["decode"]
    assert first["sbs_cells"]["read_assignment"] == "nucleus_first"
    assert first["decode_rules"]["read_assignment"] == "nucleus_first"
    assert runs["first"]["rows"]
    with ops_connect(runs["first"]["result"]["db"]) as conn:
        doubled = conn.execute("SELECT object_id, COUNT(*) FROM ops_barcodes "
                               "GROUP BY object_id HAVING COUNT(*) > 1").fetchall()
    assert not doubled
    live = {r["site"]: r["rule_a"]["calls"] for _task, r in runs["first"]["fields"]
            if "rule_a" in r}
    own = _well_replay(runs, "first", library)
    assert repr({f["site"]: f["rule_a"]["calls"] for f in own["fields"]}) == repr(live)
    from_reach = _well_replay(runs, "reach", library)
    assert repr({f["site"]: f["rule_a"]["calls"] for f in from_reach["fields"]}) == repr(live)
    reach = {r["site"]: r["rule_a"]["calls"] for _task, r in runs["reach"]["fields"]
             if "rule_a" in r}
    for site, calls in reach.items():
        for owner, call in calls.items():
            if call["source"] == "primary":
                assert live[site][owner]["guide"] == call["guide"]


def _pixel_digest(assignment, evidence_sets):
    """A digest of the pixel call rows and the 1044b refusals of every variant (item 1052b)."""
    digest = hashlib.sha256()
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    for evidence in evidence_sets:
        for ownership in ("read_rule", "read_reach", "nearest_nucleus"):
            for extra in ({}, {"ops_min_top_share": 0.9}, {"ops_read_neighbour_discount": 1.0},
                          {"ops_pixel_quality": 0.3}):
                rule = _rule(assignment, ops_pixel_ownership=ownership, **extra)
                out = ops_decode.field_calls(
                    evidence, rule, LIBRARY, matcher, decoys, threshold=50.0, footprint=3.0,
                    max_distance=0, min_fraction=0.6, single_read_quality=0.0,
                    crosstalk="none", evidence_rows=True)
                digest.update(repr(_pixel_rows(out)).encode())
                digest.update(repr(out.get("pixel_refused")).encode())
                digest.update(repr([row for row in out["evidence"]
                                    if row["source"] == "pixel_fallback"]).encode())
    return digest.hexdigest()


def test_the_pixel_rows_under_nucleus_first_hash_as_under_nucleus_reach():
    sets = [_field(spots, seed=seed) for spots, seed in (
        (B_READS + A_BLEED + C_CLEAN + C_PIXEL, 3), (A_NUCLEUS + A_CYTOPLASM + C_PIXEL, 4),
        (B_READS + A_PIXEL_BLEED + A_PIXEL_CENTRE + C_PIXEL, 5), (B_READS + C_CLEAN, 6))]
    assert _pixel_digest("nucleus_first", sets) == _pixel_digest("nucleus_reach", sets)
