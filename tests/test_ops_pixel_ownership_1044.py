"""Item 1044: a fallback pixel goes to the nucleus a read at that place would go to.

The fields are painted as the paper spike-in paints its amplicons
(``truth_spikein/spike_lib.paint``: one Gaussian spot per cycle, on the
channel of that cycle's base, over a noisy background), then run through
spaCR's own -LoG filter, read score, peaks, :func:`field_evidence`, field
cache and :func:`field_calls`.
"""

import numpy as np
import pytest
from scipy import ndimage

from spacr import ops_decode, ops_sbs

GUIDE = "ACGTAC"
OTHER = "TGCATG"
LIBRARY = [GUIDE, OTHER, "CATGCA", "GTACGT"]
CYCLES = [1, 2, 3, 4, 5, 6]
SIZE = 64
RADIUS = 5.0

#: Nucleus A at x 20 holds two amplicons of GUIDE: one inside its disc and
#: one 2 px beyond its boundary, which lies 2.9 px from the disc of the
#: read-less nucleus B (the spike-in receivers' median gap). C, far away,
#: holds a lone amplicon of OTHER that only the pixel fallback calls.
NUCLEI = {1: (32.0, 20.0), 2: (32.0, 34.9), 3: (12.0, 50.0)}
SPOTS = [((30.0, 21.0), GUIDE, 900.0), ((32.0, 27.0), GUIDE, 900.0),
         ((12.0, 50.0), OTHER, 150.0)]


def _paint(layer, barcode, pos, amplitude, sigma=1.3):
    """One amplicon, as spike_lib.paint: a Gaussian per cycle on its base channel."""
    cy, cx = pos
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    for i, base in enumerate(barcode):
        layer[i, ops_sbs.BASES.index(base)] += (amplitude * spot).astype(np.float32)


def _task():
    """The field task the engine builds: ids, centroids, areas, owned, cycles."""
    ids = np.array(sorted(NUCLEI), np.int64)
    return {"cycles": CYCLES, "ids": ids,
            "centroids": np.array([NUCLEI[i] for i in ids], float),
            "areas": np.full(len(ids), np.pi * RADIUS ** 2),
            "owned": np.ones(len(ids), bool)}


def _field(spots=SPOTS, seed=3):
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
                                     _task(), edge=3, floor=0.0, pixel_percentile=97.0)


def _threshold(evidence):
    """A read threshold that keeps the painted amplicons and drops the noise."""
    return 0.25 * float(np.max(evidence["strength"]))


def _rule(ownership, **extra):
    """The item-1023 rule (footprint 3, top share 0.9, reach 10) with the pixel fallback."""
    settings = {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                "ops_footprint": 3.0, "ops_pixel_ownership": ownership, **extra}
    return ops_decode.decode_rule(ops_decode.resolve_settings(settings))


def _with_neighbours(rule):
    """The rule as the engine hands it to a live decode: with the field's neighbours."""
    task = _task()
    return {**rule, "neighbours": ops_sbs._disc_neighbours(
        task["ids"], task["centroids"], task["areas"],
        ops_decode._neighbour_reach(rule, 3.0))}


def _calls(evidence, rule, **kw):
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    return ops_decode.field_calls(evidence, rule, LIBRARY, matcher, decoys,
                                  threshold=kw.pop("threshold", _threshold(evidence)),
                                  footprint=3.0, max_distance=0, min_fraction=0.6,
                                  single_read_quality=0.0, crosstalk="none", **kw)


def _cache(tmp_path, evidence, site=1):
    """The field written as a field cache and read back, as a replay reads it."""
    path = ops_decode.save_field_cache(str(tmp_path), site, evidence, objects=_task())
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def _plain(value):
    """A result with every dict made plain and sorted, so ``repr`` compares it (NaN included)."""
    if isinstance(value, dict):
        return {k: _plain(value[k]) for k in sorted(value, key=repr)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _same(a, b):
    return repr(_plain(a)) == repr(_plain(b))


def _sources(out):
    return {int(o): (h["guide"], h["source"]) for o, h in out["calls"].items()}


def test_the_spike_lies_where_the_spike_in_put_it():
    """The bleed amplicon is 2 px from A's disc and 2.9 px from B's."""
    task = _task()
    gap, index = ops_sbs.nearest_boundaries(np.array([SPOTS[1][0]]), task["centroids"],
                                            task["areas"])
    assert task["ids"][index[0, 0]] == 1 and gap[0, 0] == pytest.approx(2.0)
    assert task["ids"][index[0, 1]] == 2 and gap[0, 1] == pytest.approx(2.9)


def test_a_readless_neighbour_does_not_take_the_amplicon_pixels(tmp_path):
    evidence = _cache(tmp_path, _field())
    old = _calls(evidence, _rule("nearest_nucleus"))
    assert old["calls"][1]["source"] == "primary" and old["calls"][1]["guide"] == GUIDE
    assert _sources(old)[2] == (GUIDE, "pixel_fallback")
    assert old["calls"][2]["n_reads"] == 0
    new = _calls(evidence, _rule("read_rule"))
    assert _sources(new) == {1: (GUIDE, "primary"), 3: (OTHER, "pixel_fallback")}
    assert new["pixel_refused"] == {"bleed_read": 1, "bleed_pixel": 0,
                                   "radius_disagrees": 0}
    assert "pixel_refused" not in old


def test_one_amplicon_cluster_never_genotypes_two_readless_cells(tmp_path):
    """With no read anywhere, the cluster between A and B is called on neither."""
    evidence = _cache(tmp_path, _field(spots=[((32.0, 27.4), GUIDE, 900.0)]))
    threshold = float(np.max(evidence["strength"])) + 1.0
    old = _calls(evidence, _rule("nearest_nucleus"), threshold=threshold)
    assert {o for o, h in old["calls"].items() if h["guide"] == GUIDE} == {1, 2}
    new = _calls(evidence, _rule("read_rule"), threshold=threshold)
    called = {o for o, h in new["calls"].items() if h["guide"] == GUIDE}
    assert len(called) <= 1
    assert new["pixel_refused"]["bleed_pixel"] == 2 - len(called)


def test_the_gate_spares_sisters_the_rule_names(tmp_path):
    evidence = _cache(tmp_path, _field())
    rule = {**_rule("read_rule"), "sisters": {2: {1}, 1: {2}}}
    out = _calls(evidence, rule)
    assert _sources(out)[2] == (GUIDE, "pixel_fallback")
    assert out["pixel_refused"] == {"bleed_read": 0, "bleed_pixel": 0,
                                   "radius_disagrees": 0}


def test_the_gate_never_makes_a_call_the_radius_vote_would_not_make(tmp_path):
    """Within the footprint C's pixels name GUIDE; within ops_pixel_radius, OTHER."""
    spots = [((12.0, 44.0), GUIDE, 300.0), ((12.0, 37.5), OTHER, 900.0),
             ((7.0, 39.0), OTHER, 900.0)]
    evidence = _cache(tmp_path, _field(spots=spots))
    threshold = float(np.max(evidence["strength"])) + 1.0
    assert _sources(_calls(evidence, _rule("nearest_nucleus"), threshold=threshold)) == {
        3: (OTHER, "pixel_fallback")}
    assert _sources(_calls(evidence, _rule("read_reach"), threshold=threshold)) == {
        3: (GUIDE, "pixel_fallback")}
    new = _calls(evidence, _rule("read_rule"), threshold=threshold)
    assert _sources(new) == {}
    assert new["pixel_refused"] == {"bleed_read": 0, "bleed_pixel": 0,
                                    "radius_disagrees": 1}


def test_pixels_follow_the_read_footprint_not_the_pixel_radius():
    """read_reach: a pixel beyond ops_footprint votes for no one, as a read there."""
    evidence = _field(spots=[((12.0, 50.0 + RADIUS + 6.0), OTHER, 900.0)])
    threshold = float(np.max(evidence["strength"])) + 1.0
    old = _calls(evidence, _rule("nearest_nucleus"), threshold=threshold)
    assert _sources(old) == {3: (OTHER, "pixel_fallback")}
    reach = _calls(evidence, _rule("read_reach"), threshold=threshold)
    assert _sources(reach) == {}
    wide = ops_decode.decode_rule(ops_decode.resolve_settings({
        "ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
        "ops_pixel_ownership": "read_reach", "ops_footprint": 10.0}))
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    out = ops_decode.field_calls(evidence, wide, LIBRARY, matcher, decoys,
                                 threshold=threshold, footprint=10.0, max_distance=0,
                                 min_fraction=0.6, single_read_quality=0.0,
                                 crosstalk="none")
    assert _sources(out) == {3: (OTHER, "pixel_fallback")}


def _cell_mask(evidence, cells):
    """The reads given to the cell they lie in (the cell-mask read assignment)."""
    out = dict(evidence)
    peaks = np.asarray(evidence["peaks"], np.int64)
    ids = cells[peaks[:, 0], peaks[:, 1]].astype(np.int64)
    gap = np.full((len(ids), 2), 1e30, np.float32)
    gap[ids > 0, 0] = -1.0
    out.update({"gap": gap, "id0": np.where(ids > 0, ids, -1),
                "id1": np.full(len(ids), -1), "owned0": ids > 0,
                "read_assignment": "cell_mask"})
    return out


def _cells():
    """Cell labels: A's cell reaches x 31 and so holds the bleed amplicon; B's starts at 32."""
    cells = np.zeros((SIZE, SIZE), np.int32)
    cells[20:44, 10:32] = 1
    cells[20:44, 32:46] = 2
    cells[4:20, 42:58] = 3
    return cells


def test_cell_mask_mode_gives_each_pixel_its_cell_label(tmp_path):
    evidence = _cache(tmp_path, _field())
    cells = _cells()
    masked = _cell_mask(evidence, cells)
    with pytest.raises(ValueError, match="cell_labels"):
        _calls(masked, _rule("read_reach"))
    labelled = {**masked, "cell_labels": cells}
    out = _calls(labelled, _rule("read_reach"))
    assert 2 not in out["calls"]
    assert out["calls"][1]["source"] == "primary"
    yx = np.asarray(evidence["pix_yx"], np.int64)
    given = ops_decode._cell_mask_pixels(labelled, cells)
    assert np.array_equal(given["pix_id0"][given["pix_id0"] > 0],
                          cells[yx[:, 0], yx[:, 1]][cells[yx[:, 0], yx[:, 1]] > 0])
    again = _calls({**masked, **{k: given[k] for k in given if k.startswith("pix_")}},
                   _rule("read_reach"))
    assert _sources(again) == _sources(out)
    old = _calls(masked, _rule("nearest_nucleus"))
    assert _sources(old)[2] == (GUIDE, "pixel_fallback")


def test_the_read_path_is_the_same_under_every_pixel_ownership(tmp_path):
    evidence = _cache(tmp_path, _field())
    matcher, _ = ops_decode.matcher_for(LIBRARY, ())
    seen = []
    for ownership in ops_decode.PIXEL_OWNERSHIP:
        votes = ops_decode._field_votes(
            evidence, _rule(ownership), LIBRARY, matcher, threshold=_threshold(evidence),
            footprint=3.0, max_distance=0, min_fraction=0.6, single_read_quality=0.0,
            pixel_qualities=(0.1,), crosstalk="none", evidence_rows=True)
        seen.append(repr(_plain({key: votes[key] for key in (
            "primary", "tallies", "named", "short_share", "split", "objects",
            "ambiguous", "reads", "reads_owned", "reads_near", "reads_ambiguous")})))
    assert seen[0] == seen[1] == seen[2]


def test_the_shared_owner_rule_is_the_read_rule_before_1044():
    rng = np.random.default_rng(5)
    gap = np.sort(rng.uniform(-3, 12, (500, 2)).astype(np.float32), axis=1)
    owned = rng.random(500) > 0.2
    for territory in (False, True):
        rule = {"tie": 1.0, "territory": territory}
        near, ambiguous, own = ops_decode._object_owners(gap, owned, rule, 3.0)
        expect_near = gap[:, 0] <= 3.0
        expect_amb = expect_near & ((gap[:, 1] - gap[:, 0]) < 1.0)
        assert np.array_equal(near, expect_near)
        assert np.array_equal(ambiguous, expect_amb)
        assert np.array_equal(own, expect_near & owned & (territory | ~expect_amb))


#: The calls of the painted field under the rule before item 1044, made by
#: the base of item 1044 (origin/ws-offload-ws-decode-qc-speed-1041,
#: 9971bc42c6) with the same settings and no ``ops_pixel_ownership``.
BEFORE_1044 = {
    "1": ["ACGTAC", "primary", 2, 1.0, "ACGTAC"],
    "2": ["ACGTAC", "pixel_fallback", 0, 1.0, "ACGTAC"],
    "3": ["TGCATG", "pixel_fallback", 0, 1.0, "TGCATG"],
}
#: The SHA-256 of the whole :func:`field_calls` result there, every key's
#: ``repr`` in key order.
BEFORE_1044_SHA = "9f0ab3cb2faefca7c5bea32abc17fdcbf10f642ceda7e0ea8397c54907ec7bce"


def test_the_old_rule_is_the_decode_before_1044(tmp_path):
    evidence = _cache(tmp_path, _field())
    out = _calls(evidence, _rule("nearest_nucleus"))
    got = {str(o): [h["guide"], h["source"], h["n_reads"], round(h["fraction"], 6),
                    h["barcode"]] for o, h in sorted(out["calls"].items())}
    assert got == BEFORE_1044
    import hashlib

    whole = repr(sorted((key, repr(value)) for key, value in out.items()))
    assert hashlib.sha256(whole.encode()).hexdigest() == BEFORE_1044_SHA
    legacy = {key: value for key, value in _rule("nearest_nucleus").items()
              if key != "pixel_ownership"}
    assert _same(_calls(evidence, legacy), out)


def test_a_replay_from_the_field_cache_matches_the_live_decode(tmp_path):
    live_evidence = _field()
    live = _calls(live_evidence, _with_neighbours(_rule("read_rule")))
    replay = _calls(_cache(tmp_path, live_evidence), _rule("read_rule"))
    assert _same(replay, live)
    settings = {"ops_decode_pixel_fallback": True, "ops_assign_by_cell_territory": True,
                "ops_footprint": 3.0, "ops_cell_rule": "mapped",
                "spacr_settings_version": 980}
    well = ops_decode.decode_cached_well(str(tmp_path), settings, LIBRARY,
                                         read_threshold=_threshold(live_evidence),
                                         single_read_quality=0.0, pixel_quality=0.1)
    assert well["rule"]["pixel_ownership"] == "read_rule"
    assert _sources(well["fields"][0]["rule_a"]) == _sources(live)
    old = ops_decode.decode_cached_well(str(tmp_path), {**settings,
                                                        "spacr_settings_version": 979},
                                        LIBRARY, read_threshold=_threshold(live_evidence),
                                        single_read_quality=0.0, pixel_quality=0.1)
    assert old["rule"]["pixel_ownership"] == "nearest_nucleus"
    assert _sources(old["fields"][0]["rule_a"])[2] == (GUIDE, "pixel_fallback")


def test_a_cache_without_nucleus_positions_is_refused_by_name():
    evidence = _field()
    with pytest.raises(ValueError, match="nuc_ids, nuc_centroids, nuc_areas"):
        _calls(evidence, _rule("read_rule"))
    assert _calls(evidence, _rule("read_reach"))["calls"]


def test_the_auto_candidates_refuse_what_the_decode_refuses(tmp_path):
    evidence = _cache(tmp_path, _field())
    rule = _rule("read_rule")
    matcher, decoys = ops_decode.matcher_for(LIBRARY, ())
    floors, qualities = [0.0, 0.3, 2.0], [0.0, 0.1, 0.4]
    candidates = ops_decode._joint_field_candidates(
        evidence, rule, LIBRARY, matcher, decoys, spot_thresholds=[_threshold(evidence)],
        floors=floors, pixel_qualities=qualities, footprint=3.0, max_distance=0,
        min_fraction=0.6, crosstalk="none")["by_spot"][0]
    for i, floor in enumerate(floors):
        for j, q in enumerate(qualities):
            direct = ops_decode.field_calls(
                evidence, {**rule, "pixel_quality": q}, LIBRARY, matcher, decoys,
                threshold=_threshold(evidence), footprint=3.0, max_distance=0,
                min_fraction=0.6, single_read_quality=floor, crosstalk="none")
            rebuilt = ops_decode._candidate_calls(candidates, floor, q, decoys)
            assert _sources(rebuilt) == _sources(direct)
            assert rebuilt["pixel_refused"] == direct["pixel_refused"]
            assert candidates["counts"][i, j, 0] == direct["assigned_real"]


def test_the_pixel_discount_follows_the_read_discount():
    matcher, _ = ops_decode.matcher_for(LIBRARY, ())
    objects = np.array([1] * 5 + [2] * 3)
    codes = [GUIDE] * 3 + [OTHER] * 2 + [OTHER] * 3
    quality = np.ones(8)
    plain = ops_sbs.vote_pixels(objects, codes, quality, matcher, min_fraction=0.7)
    assert 1 not in plain and plain[2]["guide"] == OTHER
    same = ops_sbs.vote_pixels(objects, codes, quality, matcher, min_fraction=0.7,
                               neighbours={1: {2}, 2: {1}}, neighbour_discount=1.0)
    assert same == plain
    discounted = ops_sbs.vote_pixels(objects, codes, quality, matcher, min_fraction=0.7,
                                     neighbours={1: {2}, 2: {1}}, neighbour_discount=0.25)
    assert discounted[1]["guide"] == GUIDE and discounted[1]["n_pixels"] == 3
    assert discounted[1]["fraction"] == pytest.approx(3 / 3.5)


def test_the_setting_defaults_to_the_read_rule_and_old_files_keep_the_nearest_nucleus():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                _fold_older_defaults)

    assert OPS_DEFAULTS["ops_pixel_ownership"] == "read_rule"
    assert OPS_TYPES["ops_pixel_ownership"] is str
    assert "ops_pixel_ownership" in OPS_TOOLTIPS
    assert SETTINGS_SCHEMA_VERSION >= 980
    assert DEFAULT_CHANGES[980] == {"ops_pixel_ownership": "nearest_nucleus"}
    old = {"plate": "P1"}
    applied = _fold_older_defaults(old, 979, known={"plate", "ops_pixel_ownership"})
    assert applied["ops_pixel_ownership"] == ("nearest_nucleus", 980)
    assert old["ops_pixel_ownership"] == "nearest_nucleus"
    assert _fold_older_defaults({"plate": "P1"}, 980,
                                known={"plate", "ops_pixel_ownership"}) == {}
    assert ops_decode.decode_rule({})["pixel_ownership"] == "read_rule"
    with pytest.raises(ValueError, match="ops_pixel_ownership"):
        ops_decode.decode_rule({"ops_pixel_ownership": "nearest"})
    assert ops_decode._rule_pixel_ownership({}) == "nearest_nucleus"
