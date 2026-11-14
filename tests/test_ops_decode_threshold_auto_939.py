"""Item 939: OPS genotyping thresholds, manual or chosen per well from decoys.

``ops_decode_threshold_mode`` 'manual' (the default) keeps the decode as it
was; 'auto' scores every setting of a single-read floor by pixel quality
grid on a random half of each well's fields and keeps the setting with the
most real calls whose decoy FDR (upper 95 % CI) is at most
``ops_decode_fdr_max`` (default 0.1 per 100). The choice is a port of arm
(iii) of the BMC study's S2-it1 iteration (``s2it_analyse.py``); the study
reference below re-implements that script's selection with pandas, as the
script wrote it, and the plate A well A1 check against the study's own
tables is recorded in the item's feature file.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from spacr import ops_decode, ops_engine, ops_settings
from spacr.settings import ALPHA_FEATURES


AUTO = {"ops_decode_preset": "rule_a_s2", "ops_decode_threshold_mode": "auto"}


def _rule():
    return ops_decode.decode_rule(ops_decode.resolve_settings(
        {"ops_decode_preset": "rule_a_s2"}))


def test_manual_is_the_default_and_the_fdr_max_defaults_to_0_1():
    assert ops_settings.OPS_DEFAULTS["ops_decode_threshold_mode"] == "manual"
    assert ops_settings.OPS_DEFAULTS["ops_decode_fdr_max"] == 0.1
    rule = _rule()
    assert ops_decode.threshold_settings({}, rule) == {"mode": "manual"}
    auto = ops_decode.threshold_settings(AUTO, rule)
    assert auto["fdr_max"] == 0.1 and auto["ci"] is True
    assert auto["tune_fraction"] == 0.5 and auto["seed"] == 0
    assert auto["floors"] == list(ops_decode.AUTO_GRID["ops_single_read_quality"])
    assert auto["pixel_qualities"] == list(ops_decode.AUTO_GRID["ops_pixel_quality"])
    defaults = {k: ops_settings.OPS_DEFAULTS[k] for k in ALPHA_FEATURES[939]["settings"]}
    assert ops_decode.threshold_settings({**defaults, **AUTO}, rule) == auto


def test_the_settings_are_alpha_and_validated():
    for key in ALPHA_FEATURES[939]["settings"]:
        assert key in ops_settings.OPS_DEFAULTS and key in ops_settings.OPS_TYPES
        assert ops_settings.OPS_TOOLTIPS[key].endswith("Alpha.")
        assert any(key in keys for keys in ops_settings.OPS_CATEGORIES.values())
    rule = _rule()
    for bad in ({"ops_decode_threshold_mode": "best"},
                {**AUTO, "ops_decode_fdr_max": 0},
                {**AUTO, "ops_decode_auto_tune_fraction": 1.0},
                {**AUTO, "ops_decode_auto_grid": {"ops_read_threshold": [1]}},
                {**AUTO, "ops_decode_auto_grid": "not json"},
                {**AUTO, "ops_decode_auto_grid": {"ops_single_read_quality": [-1]}}):
        with pytest.raises(ValueError):
            ops_decode.threshold_settings(bad, rule)
    with pytest.raises(ValueError, match="ops_decoy_library_k"):
        ops_decode.threshold_settings(AUTO, {**rule, "decoy_k": 0})
    grid = ops_decode.threshold_settings(
        {**AUTO, "ops_decode_auto_grid": '{"ops_single_read_quality": [0.2, 0.1]}'},
        rule)
    assert grid["floors"] == [0.1, 0.2] and grid["pixel_qualities"] == [0.1]
    no_pixels = ops_decode.threshold_settings(AUTO, {**rule, "pixels": False})
    assert no_pixels["pixel_qualities"] == [rule["pixel_quality"]]


def _random_field(seed=3, objects=60, guides=12, cycles=6):
    """A field whose reads and pixels carry a spread of qualities."""
    rng = np.random.default_rng(seed)
    library = sorted({"".join(rng.choice(list("GTAC"), cycles)) for _ in range(guides * 3)})[:guides]
    decoys = sorted({"".join(rng.choice(list("GTAC"), cycles)) for _ in range(guides)}
                    - set(library))[:4]
    pool = library + decoys
    reads, ids, gap = [], [], []
    for obj in range(1, objects + 1):
        truth = pool[rng.integers(len(pool))]
        for _ in range(int(rng.integers(1, 4))):
            code = truth if rng.random() < 0.8 else pool[rng.integers(len(pool))]
            values = rng.uniform(0, 600, (cycles, 4))
            for c, letter in enumerate(code):
                values[c, "GTAC".index(letter)] = rng.uniform(400, 1200)
            reads.append(values)
            ids.append(obj)
            gap.append([rng.uniform(0, 4), rng.uniform(0, 8)])
    pix, pix_ids = [], []
    for obj in range(1, objects + 1):
        truth = pool[rng.integers(len(pool))]
        for _ in range(int(rng.integers(0, 5))):
            values = rng.uniform(0, 600, (cycles, 4))
            for c, letter in enumerate(truth):
                values[c, "GTAC".index(letter)] = rng.uniform(300, 1200)
            pix.append(values)
            pix_ids.append(obj)
    evidence = {
        "strength": rng.uniform(300, 900, len(reads)),
        "values": np.asarray(reads, np.float32), "gap": np.asarray(gap, np.float32),
        "id0": np.asarray(ids), "owned0": rng.random(len(reads)) < 0.9,
        "owned_ids": np.arange(1, objects + 1)[rng.random(objects) < 0.85],
        "pix_values": np.asarray(pix, np.float32),
        "pix_gap": np.column_stack([rng.uniform(0, 12, len(pix)),
                                    np.full(len(pix), 20.0)]).astype(np.float16),
        "pix_id0": np.asarray(pix_ids), "pix_owned0": np.ones(len(pix), bool),
    }
    return library, decoys, evidence


_RULE = {"ops_assign_by_cell_territory": True, "ops_decode_pixel_fallback": True,
         "ops_decode_quality_weighted": True, "ops_decode_runner_up": True,
         "ops_decoy_library_k": 4}
_KW = dict(threshold=315.0, footprint=10.0, max_distance=0, min_fraction=0.6,
           crosstalk="none")


def test_every_grid_setting_rebuilds_the_calls_field_calls_makes():
    library, decoys, evidence = _random_field()
    rule = ops_decode.decode_rule(_RULE)
    matcher, decoy_set = ops_decode.matcher_for(library, decoys)
    floors, qualities = [0.0, 0.05, 0.1, 0.2, 0.4], [0.0, 0.1, 0.2, 0.4]
    votes = ops_decode._field_votes(evidence, rule, library, matcher,
                                    single_read_quality=float("-inf"),
                                    pixel_qualities=qualities, evidence_rows=True, **_KW)
    candidates = ops_decode._field_candidates(votes, floors, qualities, decoy_set, True)
    totals = set()
    for i, floor in enumerate(floors):
        for j, quality in enumerate(qualities):
            expected = ops_decode.field_calls(
                evidence, {**rule, "pixel_quality": quality}, library, matcher,
                decoy_set, single_read_quality=floor, evidence_rows=True, **_KW)
            got = ops_decode._candidate_calls(candidates, floor, quality, decoy_set)
            assert repr(list(got["calls"].items())) == repr(list(expected["calls"].items()))
            assert pd.DataFrame(got["evidence"]).equals(pd.DataFrame(expected["evidence"]))
            for key in ("nuclei", "assigned_real", "assigned_decoy", "reads",
                        "reads_owned", "reads_near", "reads_ambiguous"):
                assert got[key] == expected[key]
            assert tuple(candidates["counts"][i, j]) == (
                expected["assigned_real"], expected["assigned_decoy"])
            totals.add(tuple(candidates["counts"][i, j]))
    assert len(totals) > 3


def _study_choice(counts, nuclei, sites, floors, qualities, seed, fdr_max=0.1):
    """Arm (iii) of s2it_analyse.py, as the script wrote it (pandas, s2it_rule)."""
    def fdr(real, decoy):
        return 100 * 1.268 / 4 * decoy / max(real, 1)

    s = np.array(sorted(sites[nuclei > 0]))
    perm = np.random.default_rng(seed).permutation(len(s))
    halves = (set(s[perm[: len(s) // 2]]), set(s[perm[len(s) // 2:]]))
    row_of = {site: i for i, site in enumerate(sites)}
    grid = []
    for i, floor in enumerate(floors):
        for j, quality in enumerate(qualities):
            st = pd.DataFrame([(site, counts[row_of[site], i, j, 0],
                                counts[row_of[site], i, j, 1])
                               for site in sorted(halves[0])],
                              columns=["site", "real", "decoy"]).set_index("site")
            v = st[["real", "decoy"]].to_numpy()
            rng, out = np.random.default_rng(1), []
            for _ in range(1000):
                k = rng.integers(0, len(v), len(v))
                tot = v[k].sum(0)
                out.append(fdr(tot[0], tot[1]))
            grid.append({"floor": floor, "pix_q": quality, "real": int(st.real.sum()),
                         "fdr_ci": [float(np.percentile(out, 2.5)),
                                    float(np.percentile(out, 97.5))]})
    g = pd.DataFrame(grid)
    ok = g[g.fdr_ci.map(lambda x: x[1]) <= fdr_max]
    best = ok.sort_values("real", ascending=False).iloc[0]
    return (float(best.floor), float(best.pix_q)), halves, best


def _synthetic_well(seed=0, fields=40):
    """Counts where a looser setting genotypes more and has more decoys."""
    rng = np.random.default_rng(seed)
    floors, qualities = [0.06, 0.1, 0.2, 0.4], [0.1, 0.2, 0.4]
    nuclei = rng.integers(800, 1200, fields)
    counts = np.zeros((fields, len(floors), len(qualities), 2), np.int64)
    for i in range(len(floors)):
        for j in range(len(qualities)):
            share = 0.75 - 0.05 * i - 0.03 * j
            real = rng.binomial(nuclei, share)
            counts[:, i, j, 0] = real
            counts[:, i, j, 1] = rng.poisson(real * (0.0016 - 0.0003 * i - 0.00015 * j))
    sites = rng.permutation(np.arange(100, 100 + 3 * fields, 3))
    return counts, nuclei, sites, floors, qualities


def test_auto_reproduces_the_study_selection_and_respects_the_fdr_max():
    counts, nuclei, sites, floors, qualities = _synthetic_well()
    for seed in (20261113, 0, 7):
        study, halves, best = _study_choice(counts, nuclei, sites, floors, qualities, seed)
        got = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                                k=4, scale=1.268, seed=seed)
        assert (got["single_read_quality"], got["pixel_quality"]) == study
        assert set(got["tune_sites"]) == halves[0]
        assert set(got["report_sites"]) == halves[1]
        assert got["tune"]["real"] == best.real
        assert got["tune"]["fdr_ci95"][1] == pytest.approx(best.fdr_ci[1], abs=0)
        assert got["auto_met"] and got["tune"]["fdr_ci95"][1] <= 0.1
        passing = [r for r in got["grid"] if r["passes"]]
        assert all(r["fdr_ci95"][1] <= 0.1 for r in passing)
        assert got["tune"]["real"] == max(r["real"] for r in passing)
    strict = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                               k=4, scale=1.268, fdr_max=0.03)
    loose = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                              k=4, scale=1.268, fdr_max=1.0)
    assert strict["tune"]["real"] <= loose["tune"]["real"]
    assert strict["tune"]["fdr_ci95"][1] <= 0.03 or not strict["auto_met"]
    point = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                              k=4, scale=1.268, ci=False)
    assert point["tune"]["fdr_per_100"] <= 0.1
    assert point["tune"]["real"] >= ops_decode.choose_well_thresholds(
        counts, nuclei, sites, floors, qualities, k=4, scale=1.268)["tune"]["real"]


def test_no_passing_setting_takes_the_strictest_and_says_so():
    counts, nuclei, sites, floors, qualities = _synthetic_well()
    got = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                            k=4, scale=1.268, fdr_max=0.0001)
    assert got["auto_met"] is False
    assert (got["single_read_quality"], got["pixel_quality"]) == (0.4, 0.4)


def test_too_few_decoys_never_give_a_confident_pick():
    """Item 939 acceptance 6: a well too small to show a false call."""
    counts = np.zeros((6, 2, 1, 2), np.int64)
    counts[:, 0, 0, 0], counts[:, 1, 0, 0] = 40, 30
    nuclei = np.full(6, 60)
    got = ops_decode.choose_well_thresholds(counts, nuclei, np.arange(6), [0.06, 0.4],
                                            [0.1], k=4, scale=1.268)
    assert got["auto_met"] is False and got["single_read_quality"] == 0.4
    assert all(r["decoy"] == 0 and not r["passes"] for r in got["grid"])
    one = ops_decode.choose_well_thresholds(counts[:1], nuclei[:1], [0], [0.06, 0.4],
                                            [0.1], k=4, scale=1.268, fdr_max=100)
    assert one["auto_met"] is False and one["grid"] == []
    big = counts * 1000
    ok = ops_decode.choose_well_thresholds(big, nuclei * 1000, np.arange(6), [0.06, 0.4],
                                           [0.1], k=4, scale=1.268)
    assert ok["auto_met"] and ok["single_read_quality"] == 0.06
    with pytest.raises(ValueError, match="decoys"):
        ops_decode.choose_well_thresholds(counts, nuclei, np.arange(6), [0.06, 0.4],
                                          [0.1], k=0, scale=1.268)


def test_the_choice_is_deterministic_and_independent_of_field_order():
    counts, nuclei, sites, floors, qualities = _synthetic_well(seed=5)
    first = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                              k=4, scale=1.268, seed=11)
    again = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                              k=4, scale=1.268, seed=11)
    order = np.random.default_rng(2).permutation(len(sites))
    shuffled = ops_decode.choose_well_thresholds(counts[order], nuclei[order], sites[order],
                                                 floors, qualities, k=4, scale=1.268, seed=11)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert json.dumps(first, sort_keys=True) == json.dumps(shuffled, sort_keys=True)
    other = ops_decode.choose_well_thresholds(counts, nuclei, sites, floors, qualities,
                                              k=4, scale=1.268, seed=12)
    assert other["tune_sites"] != first["tune_sites"]


def test_ties_go_to_the_stricter_setting():
    counts = np.zeros((10, 2, 2, 2), np.int64)
    counts[..., 0], counts[..., 1] = 5000, 1
    got = ops_decode.choose_well_thresholds(counts, np.full(10, 6000), np.arange(10),
                                            [0.06, 0.1], [0.1, 0.2], k=4, scale=1.268)
    assert got["auto_met"]
    assert (got["single_read_quality"], got["pixel_quality"]) == (0.1, 0.2)


def test_a_settings_file_from_before_939_reruns_on_manual_thresholds(tmp_path):
    """No DEFAULT_CHANGES row: 'manual' is today's default and the decode
    before item 939, so old files (unstamped or stamped) stay manual."""
    import csv

    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_VERSION_KEY

    assert not any("ops_decode_threshold_mode" in row for row in DEFAULT_CHANGES.values())
    ops = resolve_module("ops")
    for rows in ([("plate", "P1")], [(SETTINGS_VERSION_KEY, "933"), ("plate", "P1")]):
        path = tmp_path / "old.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Key", "Value"])
            writer.writerows(rows)
        loaded = resolve_settings(ops, str(path))
        assert loaded["ops_decode_threshold_mode"] == "manual"
        rule = ops_decode.decode_rule(ops_decode.resolve_settings(loaded))
        assert ops_decode.threshold_settings(loaded, rule) == {"mode": "manual"}


@pytest.fixture(scope="module")
def auto_run(tmp_path_factory):
    """The synthetic well of item 873's tests, decoded manual and auto."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_auto_939")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_preset": "rule_a_s2"}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    manual = ops_engine.run_ops(settings, library=library)
    auto = ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "auto"},
                              phases=("decode",), library=library)
    loose = ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "auto",
                                "ops_decode_fdr_max": 1000.0,
                                "ops_decode_auto_tune_fraction": 0.5},
                               phases=("decode",), library=library)
    yield manual, auto, loose, settings, library
    monkeypatch.undo()


def test_the_engine_writes_the_chosen_thresholds_per_well(auto_run):
    manual, auto, loose, _settings, _library = auto_run
    assert "thresholds" not in manual["wells"]["A1"]["decode"]
    report = auto["wells"]["A1"]["decode"]["thresholds"]
    assert report["auto_met"] is False
    assert report["single_read_quality"] == 0.4 and report["pixel_quality"] == 0.4
    with open(report["paths"]["thresholds"], encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["well"] == "A1" and len(stored["grid"]) in (0, 45)
    table = pd.read_csv(report["paths"]["plate_table"])
    assert list(table["well"]) == ["A1"]
    assert bool(table["auto_met"][0]) == loose["wells"]["A1"]["decode"]["thresholds"]["auto_met"]
    rules = auto["wells"]["A1"]["decode"]["decode_rules"]
    assert rules["single_read_quality"] == 0.4 and "threshold_choice" not in rules
    chosen = loose["wells"]["A1"]["decode"]
    assert chosen["thresholds"]["auto_met"] is True
    floor = chosen["thresholds"]["single_read_quality"]
    quality = chosen["thresholds"]["pixel_quality"]
    assert chosen["objects_assigned"] >= auto["wells"]["A1"]["decode"]["objects_assigned"]
    assert chosen["decode_rules"]["single_read_quality"] == floor
    assert chosen["decode_rules"]["pixel_quality"] == quality
    assert chosen["fdr"]["assigned"] == chosen["objects_assigned"]
    assert math.isfinite(chosen["fdr"]["pct"])


def test_auto_needs_decoys_and_the_rule_a_decode(auto_run):
    _manual, _auto, _loose, settings, library = auto_run
    with pytest.raises(ValueError, match="ops_decoy_library_k"):
        ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "auto",
                            "ops_decode_preset": "none"}, phases=("decode",),
                           library=library)
    with pytest.raises(ValueError, match="rule_a_s2"):
        ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "auto",
                            "ops_cell_rule": "agree"}, phases=("decode",),
                           library=library)
