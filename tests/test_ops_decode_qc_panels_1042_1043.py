"""Items 1042 and 1043: the opt-in bleed panel and depth-recovery curve of the decode QC."""
import numpy as np
import pytest

from spacr import ops_decode_qc_panels as P
from spacr.ops_decode import _cached_rule, resolve_settings, run_decode_qc
from spacr.ops_settings import OPS_DEFAULTS, ops_defaults

CYCLES = 11


def _library(count, seed=1):
    rng = np.random.default_rng(seed)
    out = set()
    while len(out) < count:
        out.add("".join(rng.choice(list("ACGT"), CYCLES)))
    return sorted(out)


def _values(codes):
    out = np.full((len(codes), CYCLES, 4), 40.0, np.float32)
    for r, code in enumerate(codes):
        for c, base in enumerate(code):
            out[r, c, "GTAC".index(base)] = 1000.0
    return out


def _field(nuclei, reads):
    """A field cache: nuclei ``(id, y, x, area)``, reads ``(code, id0, id1, gap0, gap1, y, x)``."""
    ids = np.array([n[0] for n in nuclei], np.int64)
    n = len(reads)
    return {
        "peaks": np.array([[r[5], r[6]] for r in reads], np.int32).reshape(n, 2),
        "strength": np.full(n, 900.0, np.float32),
        "values": _values([r[0] for r in reads]).reshape(n, CYCLES, 4),
        "gap": np.array([[r[3], r[4]] for r in reads], np.float32).reshape(n, 2),
        "id0": np.array([r[1] for r in reads], np.int64),
        "id1": np.array([r[2] for r in reads], np.int64),
        "owned0": np.ones(n, bool),
        "owned_ids": ids, "owned_areas": np.array([n_[3] for n_ in nuclei], float),
        "cycles": np.arange(1, CYCLES + 1),
        "pix_yx": np.zeros((0, 2), np.int16), "pix_values": np.zeros((0, CYCLES, 4), np.float32),
        "pix_score": np.zeros(0, np.float16), "pix_gap": np.zeros((0, 2), np.float16),
        "pix_id0": np.zeros(0, np.int64), "pix_id1": np.zeros(0, np.int64),
        "pix_owned0": np.zeros(0, bool),
        "nuc_ids": ids, "nuc_centroids": np.array([[n_[1], n_[2]] for n_ in nuclei], float),
        "nuc_areas": np.array([n_[3] for n_ in nuclei], float),
    }


def _bleed_field(library, planted, pairs=24, distinct=None):
    """Pairs of nuclei 14 px apart; each has 3 central reads of its own guide and one minority read."""
    nuclei, reads = [], []
    spare = library[-1]
    for p in range(pairs):
        a, b = 2 * p + 1, 2 * p + 2
        y, x = 60 + 70 * (p // 6), 60 + 70 * (p % 6)
        nuclei += [(a, y, x, 300.0), (b, y, x + 14, 520.0)]
        ga, gb = library[(a - 1) % (distinct or 99)], library[(b - 1) % (distinct or 99)]
        for me, other, guide, dx in ((a, b, ga, 0), (b, a, gb, 14)):
            for _ in range(3):
                reads.append((guide, me, other, 1.0, 8.0, y, x + dx))
            minority = (gb if me == a else ga) if planted else spare
            reads.append((minority, me, other, 2.0, 2.0, y, x + dx))
    return _field(nuclei, reads)


def _exact(z, library):
    names, index, is_decoy = P.universe_index(library, [])
    return P.exact_matches(z, index, {"calibration": False}, library, "none"), is_decoy


def _bleed(planted, seed=5, distinct=None):
    library = _library(60)
    z = _bleed_field(library, planted, distinct=distinct)
    exact, is_decoy = _exact(z, library)
    rng = np.random.default_rng([seed, 0])
    pairs, null = P.bleed_field(z, exact, is_decoy, 315.0, rng, draws=200)
    pairs["tile"] = 0
    return P.bleed_summary(pairs, null, pairs.tile.to_numpy(), seed), pairs, null


def _overall(summary, group="all"):
    return summary[(summary.group == group) & (summary.dimension == "overall")].iloc[0]


def test_planted_bleed_rises_above_the_ring_null():
    planted, pairs, _ = _bleed(True)
    control, _, _ = _bleed(False)
    row, flat = _overall(planted), _overall(control)
    assert row.pairs == len(pairs) > 20
    assert row.observed_pct == pytest.approx(100.0)
    assert row.bleed_pct > 90
    assert row.observed_pct > row.null_hi
    assert flat.observed_pct == pytest.approx(0.0)
    assert flat.bleed_pct <= 0.5


def test_bleed_is_binned_by_distance_and_position_and_sisters_are_split_by_geometry():
    summary, pairs, _ = _bleed(True)
    gap = summary[(summary.group == "all") & (summary.dimension == "read_to_neighbour_gap")]
    assert list(gap["bin"]) == P.GAP_NAMES
    assert gap.set_index("bin").loc["1-3", "pairs"] == len(pairs)
    spot = summary[(summary.group == "all") & (summary.dimension == "spot_position")].set_index("bin")
    assert spot.loc["border", "pairs"] == len(pairs)
    assert not pairs.sister.any()
    assert _overall(summary, "sisters_excluded").pairs == len(pairs)
    assert _overall(summary, "sisters_alone").pairs == 0


def test_close_nuclei_of_one_size_are_sisters_and_are_left_out():
    library = _library(60)
    nuclei = [(1, 100, 100, 300.0), (2, 100, 112, 310.0)]
    reads = [(library[0], 1, 2, 1.0, 8.0, 100, 100)] * 3 + [(library[1], 2, 1, 1.0, 8.0, 100, 112)] * 3
    reads += [(library[1], 1, 2, 2.0, 2.0, 100, 100)]
    z = _field(nuclei, reads)
    exact, is_decoy = _exact(z, library)
    pairs, _ = P.bleed_field(z, exact, is_decoy, 315.0, np.random.default_rng(0), draws=20)
    assert pairs.sister.all() and pairs.match.all()


def test_bleed_is_deterministic_under_the_seed():
    first, _, null = _bleed(True, seed=11, distinct=5)
    again, _, null_again = _bleed(True, seed=11, distinct=5)
    other, _, null_other = _bleed(True, seed=12, distinct=5)
    assert first.equals(again)
    assert (null == null_again).all()
    assert null.mean() > 0
    assert not (null == null_other).all()


def _depth_field(library, cells=8):
    nuclei, reads = [], []
    for c in range(cells):
        y, x = 500, 100 + 100 * c
        nuclei.append((c + 1, y, x, 300.0))
        for _ in range(6):
            reads.append((library[c], c + 1, -1, 1.0, 30.0, y, x))
    return _field(nuclei, reads)


def _depth(seed=3, levels=(1, 2, 3, 4, "all")):
    library = _library(40, seed=2)
    z = _depth_field(library)
    settings = resolve_settings(ops_defaults({}))
    rule, lib, decoys, crosstalk = _cached_rule(
        settings, library, read_threshold=315.0, single_read_quality=1e9, pixel_quality=0.1)
    rule["pixels"] = False
    return P.qc_panels([{"site": 0, "z": z}], rule, lib, decoys, crosstalk, seed=seed,
                       bleed=False, depth=True, tile_shapes={0: (1000.0, 1000.0)},
                       depth_levels=levels, depth_draws=3, depth_null_draws=50)


def test_depth_curve_k1_recovers_less_than_k4_and_the_wrong_rate_is_reported():
    out = _depth()
    assert len(out["silver"]) == 8 and out["silver"].silver.all()
    table = out["depth_summary"].set_index("k")
    assert table.loc["1", "rec_pct"] < table.loc["4", "rec_pct"]
    assert table.loc["4", "rec_pct"] == pytest.approx(100.0)
    assert table.loc["all", "rec_pct"] == pytest.approx(100.0)
    assert {"wrong_pct", "wrong_lo", "wrong_hi", "none_pct"} <= set(table.columns)
    assert table.loc["4", "wrong_pct"] == pytest.approx(0.0)
    assert table.loc["1", "none_pct"] > 0
    assert out["depth_null"]["mean_pct"] < 50


def test_depth_curve_is_deterministic_under_the_seed():
    first, again = _depth(seed=9), _depth(seed=9)
    assert first["depth_cells"].equals(again["depth_cells"])
    assert first["depth_summary"].equals(again["depth_summary"])
    assert first["depth_null"] == again["depth_null"]


def test_a_cell_next_to_another_guide_is_not_silver():
    library = _library(40, seed=2)
    z = _depth_field(library, cells=2)
    z["nuc_centroids"][1] = [500.0, 120.0]
    exact, is_decoy = _exact(z, library)
    calls = {2: int(exact[-1])}
    sets = P.silver_field(z, exact, is_decoy, calls, (1000.0, 1000.0), 315.0)
    assert 1 not in set(sets[sets.silver].object_id)


def test_panels_are_off_by_default_and_a_decode_qc_without_them_writes_nothing_new(tmp_path):
    assert OPS_DEFAULTS["ops_decode_qc_bleed_panel"] is False
    assert OPS_DEFAULTS["ops_decode_qc_depth_curve"] is False
    db = str(tmp_path / "measurements.db")
    report = run_decode_qc(db, "plate", ["A1"], resolve_settings(ops_defaults({})), _library(5))
    assert report == {}
    assert not any("bleed" in f or "depth" in f for f in (p.name for p in (tmp_path / "results").rglob("*")))


def test_the_panels_turn_on_the_field_cache_and_nothing_else_changes_when_off():
    from spacr.ops_decode import decode_rule

    off = decode_rule(resolve_settings(ops_defaults({})))
    assert off["store_cache"] is False
    for key in ("ops_decode_qc_bleed_panel", "ops_decode_qc_depth_curve"):
        on = decode_rule(resolve_settings(ops_defaults({key: True})))
        assert on["store_cache"] is True
        assert {k: v for k, v in on.items() if k != "store_cache"} == {
            k: v for k, v in off.items() if k != "store_cache"}


def test_run_qc_panels_writes_the_tables_and_figures(tmp_path):
    library = _library(60)
    z = _bleed_field(library, True)
    db = str(tmp_path / "measurements.db")
    folder = tmp_path / "results" / "ops_decode_qc"
    folder.mkdir(parents=True)
    settings = resolve_settings(ops_defaults({"ops_decode_qc_bleed_panel": True,
                                              "ops_decode_qc_seed": 4}))
    report = P.run_qc_panels(db, "plate", ["A1"], settings, library, str(folder),
                             fields_of_well={"A1": [{"site": 0, "z": z}]})
    paths = report["A1"]
    assert (folder / "bleed_panel_A1.parquet").is_file() and (folder / "bleed_panel_A1.png").is_file()
    assert "depth_curve" not in paths
