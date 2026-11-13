"""Guide-bootstrap consensus clustering (item 928) and the calibrated cluster test (item 929), alpha.

PHATE, igraph and leidenalg are the optional ``clustering`` extra and are not
installed on CI, so most tests use the stand-ins of the item 916 tests: an
igraph that keeps its edges and a Leiden that returns the connected
components of the edges above zero. Tests that need the real extra skip
without it.

The reference values in ``tests/data/ops_stage5_928_929_fixture.json`` were
computed by the analysis scripts the features reproduce (cal_lib.py's
paired-replicate test and gate loop, a1_margin.py's noise-floor margin,
cs_lib.py's count adjustment and guide bootstrap, and Design F's consensus
cut), by ``spacr_stage5_proof/scripts/make_test_fixture.py``.
"""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import ops_clusters as oc
from spacr.ops_engine import run_ops

from tests.test_ops_leiden_sweep_916 import _install, _references

FIXTURE = json.loads((Path(__file__).resolve().parent / "data" / "ops_stage5_928_929_fixture.json")
                     .read_text(encoding="utf-8"))


def _reference():
    """The fixture's units as :func:`spacr.ops_clusters.calibrated_reference` returns them."""
    ref = FIXTURE["reference"]
    return {"n": ref["n"], "pi": np.array(ref["pi"]), "pj": np.array(ref["pj"]),
            "cx_members": [np.array(m) for m in ref["cx_members"]], "pgenes": np.array(ref["pgenes"]),
            "pair_index": [np.array(m) for m in ref["pair_index"]]}


def _arrays(units):
    """A fixture's stacked units as arrays."""
    return {key: np.array(value) for key, value in units.items()}


def _planted(n_blocks=4, size=6, guides=4, dims=8, noise=0.02, seed=0):
    """Gene rows of planted modules and their noisy guide rows, plus two control rows."""
    rng = np.random.default_rng(seed)
    names, rows, block, g_rows, g_genes = [], [], [], [], []
    for b in range(n_blocks):
        centre = np.zeros(dims)
        centre[b] = 1.0
        for i in range(size):
            name = f"G{b}_{i}"
            members = centre + noise * rng.standard_normal((guides, dims))
            names.append(name)
            rows.append(members.mean(0))
            block.append(b)
            g_rows.append(members)
            g_genes += [name] * guides
    for c in range(2):
        names.append(f"nontargeting_{c}")
        rows.append(rng.standard_normal(dims))
        block.append(-1)
    return (np.array(names), np.vstack(rows), np.array(block), np.vstack(g_rows), np.array(g_genes),
            np.full(len(g_genes), 20.0))


# ----------------------------------------------------------------------------------------------- calibrated test


@pytest.mark.parametrize("case,one_sided,seed", [("two_sided", False, 3), ("one_sided", True, 5)])
def test_the_paired_replicate_test_equals_cal_lib(case, one_sided, seed):
    rows = oc.paired_replicate_test(_arrays(FIXTURE["A"]), _arrays(FIXTURE["B"]), _reference(),
                                    one_sided=one_sided, seed=seed)
    expected = FIXTURE[case]
    assert [r["unit"] for r in rows] == [r["unit"] for r in expected]
    for got, want in zip(rows, expected):
        for key, value in want.items():
            if isinstance(value, float):
                assert abs(got[key] - value) <= 1e-9, (case, key, got[key], value)
            else:
                assert got[key] == value


def test_the_gate_reproduces_cal_libs_false_positive_count():
    gate = FIXTURE["gate"]
    table, share, passed = oc.calibration_gate(_arrays(gate["base"]), [_arrays(c) for c in gate["copies"]],
                                               _reference(), labels={"side": "fixture"})
    assert share == pytest.approx(gate["false_positive_share"], abs=0)
    assert passed is gate["passed"] is False
    want = pd.DataFrame(gate["table"])
    assert (table["p"] < 0.05).sum() == (want["p"] < 0.05).sum()
    for column in ("p", "diff_pts", "ci_low", "ci_high"):
        assert np.abs(table[column].to_numpy() - want[column].to_numpy()).max() <= 1e-9


def test_the_noise_floor_margin_equals_a1_margin():
    gate = FIXTURE["gate"]
    margin = oc.noise_floor_margin(pd.DataFrame(gate["table"]))
    want = pd.DataFrame(gate["margin"]).merge(pd.DataFrame(gate["margin_check"]), on="unit")
    merged = margin.merge(want, on="unit", suffixes=("", "_want"))
    assert len(merged) == 3
    for column in ("delta_pts", "raw_false_positive_share", "margin_rule_false_positive_share_cv"):
        assert np.abs(merged[column] - merged[f"{column}_want"]).max() <= 1e-12
    assert (merged["passed"] == merged["passed_want"]).all()


@pytest.mark.parametrize("n", [3, 6, 9])
def test_the_exact_signflip_p_equals_brute_force_enumeration(n):
    # quarter-integers: every pattern's mean is exact in single precision, so ties are exact
    d = np.random.default_rng(n).integers(-6, 9, n) / 4
    means = np.array([np.mean(np.array(s) * d) for s in itertools.product([1, -1], repeat=n)])
    assert oc.signflip_exact(d) == np.mean(np.abs(means) >= abs(d.mean()))
    assert oc.signflip_exact(d, one_sided=True) == np.mean(means >= d.mean())
    assert oc.signflip_exact(np.zeros(n)) == 1.0


def test_a_planted_effect_above_the_margin_is_called_and_one_below_is_not():
    tests = pd.DataFrame({"unit": ["complexes", "STRING pairs", "STRING pairs, per-gene capped"],
                          "p": [1e-4, 1e-4, 0.5], "diff_pts": [2.0, 0.3, 5.0]})
    called = oc.call_effects(tests, {"complexes": 1.0, "STRING pairs": 1.0}, gated=False)
    assert list(called["call"]) == ["higher", "no detectable effect", "no detectable effect"]
    assert set(called["gate"]) == {"failed: margin rule"}
    plain = oc.call_effects(tests.assign(diff_pts=[-2.0, 0.3, 5.0]))
    assert list(plain["call"]) == ["lower", "higher", "no detectable effect"]
    assert set(plain["gate"]) == {"not gated"}


def test_partition_units_and_the_size_matched_null():
    ref = _reference()
    labels = np.repeat(np.arange(8), 5)
    units = oc.partition_units(labels, ref, nperm=50)
    null = oc.size_matched_null(oc._sizes(labels), ref, 50)
    assert null["pair_prob"] == pytest.approx(8 * 5 * 4 / (40 * 39))
    together = labels[ref["pi"]] == labels[ref["pj"]]
    assert np.allclose(units["pr"], together - null["pair_prob"])
    assert len(units["gm"]) == len(ref["pgenes"]) and len(units["cx"]) == len(ref["cx_members"])
    again = oc.size_matched_null(oc._sizes(labels), ref, 50)
    assert np.array_equal(again["complex_prob"], null["complex_prob"])


def test_the_jitter_copy_has_the_set_relative_size_and_is_fixed_per_copy():
    rows = np.random.default_rng(1).normal(size=(30, 6))
    copy = oc.jitter_rows(rows, 4)
    ratio = np.linalg.norm(copy - rows, axis=1) / np.linalg.norm(rows, axis=1)
    assert np.allclose(ratio, 1.7e-3)
    assert np.array_equal(copy, oc.jitter_rows(rows, 4)) and not np.array_equal(copy, oc.jitter_rows(rows, 5))


# ----------------------------------------------------------------------------------------------- consensus parts


@pytest.mark.parametrize("case", ["too_many", "too_few", "within"])
def test_match_count_equals_the_reference_count_adjustment(case):
    adjust = FIXTURE["adjust"]
    spec = adjust["cases"][case]
    labels, adjusted = oc.match_count(np.array(adjust["X"]), np.array(spec["labels"]), spec["k"],
                                      np.array(adjust["mask"]), 7)
    assert labels.tolist() == spec["expected"] and adjusted is spec["adjusted"]


def test_the_consensus_cut_equals_design_fs_cut():
    cut = FIXTURE["cut"]
    labels, info = oc.consensus_partition(np.array(cut["cooc"], np.float32), cut["k"], np.array(cut["mask"]))
    assert labels.tolist() == cut["labels"] and info["cut"] == cut["cut"]


def test_the_guide_bootstrap_equals_the_reference_resampler():
    boot = FIXTURE["bootstrap"]
    for r, want in boot["draws"].items():
        got = oc.guide_bootstrap_profiles(np.array(boot["rows"]), boot["names"], np.array(boot["guide_rows"]),
                                          boot["guide_genes"], boot["guide_cells"], boot["seed_base"] + int(r))
        assert np.abs(got - np.array(want)).max() == 0.0
    rows = np.array(boot["rows"])
    got = oc.guide_bootstrap_profiles(rows, boot["names"], np.array(boot["guide_rows"]), boot["guide_genes"],
                                      boot["guide_cells"], 1)
    assert np.array_equal(got[-2:], rows[-2:])   # control rows are not resampled
    assert np.array_equal(got[9], rows[9])         # G9 has no guide rows


def test_the_coclustering_matrix_is_a_symmetric_share_with_a_unit_diagonal():
    parts = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([2, 2, 2, 0])]
    cooc = oc.coclustering_matrix(parts)
    assert cooc.dtype == np.float32 and np.allclose(cooc, cooc.T) and np.allclose(np.diag(cooc), 1)
    assert cooc.min() >= 0 and cooc.max() <= 1
    assert cooc[0, 1] == pytest.approx(2 / 3) and cooc[2, 3] == pytest.approx(2 / 3)
    gaps = oc.coclustering_matrix([np.array([0, 0, -1]), np.array([0, 1, 1])])
    assert gaps[0, 2] == 0.0 and gaps[1, 2] == 1.0 and gaps[0, 1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        oc.coclustering_matrix([])


def test_consensus_recovers_planted_modules_and_assigns_every_gene(monkeypatch):
    _install(monkeypatch)
    names, rows, block, g_rows, g_genes, cells = _planted()
    mask = block >= 0
    out = oc.guide_bootstrap_consensus(
        lambda b: oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, cells, 8000 + b), 5, 4, mask,
        rows=rows)
    labels = out["labels"]
    assert len(labels) == len(names) and (labels >= 0).all()
    assert len(np.unique(labels[mask])) == 4
    assert all(len(np.unique(labels[block == b])) == 1 for b in range(4))
    assert np.allclose(out["stability"][mask], 1.0)
    assert [r["seed"] for r in out["runs"]] == [10000, 10001, 10002, 10003, 10004]
    assert np.allclose(out["cooc"], out["cooc"].T) and np.allclose(np.diag(out["cooc"]), 1)


def test_one_bootstrap_gives_the_single_runs_partition(monkeypatch):
    _install(monkeypatch)
    names, rows, block, *_ = _planted()
    mask = block >= 0
    single, _info = oc.single_partition(rows, 4, mask, seed=100)
    consensus = oc.guide_bootstrap_consensus([rows], 1, 4, mask, seed=1, rows=rows)
    assert pd.crosstab(single[mask], consensus["labels"][mask]).gt(0).sum(1).eq(1).all()


def test_unforced_genes_below_the_stability_floor_become_singletons():
    cooc = np.array([[1, .9, .2], [.9, 1, .2], [.2, .2, 1]], np.float32)
    assert np.allclose(oc.consensus_stability(cooc, np.array([0, 0, 0])), [0.55, 0.55, 0.2])
    assert np.allclose(oc.consensus_stability(cooc, np.array([0, 0, 1])), [0.9, 0.9, 1.0])


def test_the_target_count_matches_brieflows_cluster_size():
    assert oc.target_cluster_count(800) == 63 and oc.target_cluster_count(1847) == 146


# ----------------------------------------------------------------------------------------------- real clustering extra


def test_real_consensus_is_deterministic_and_seed_dependent_within_tolerance():
    pytest.importorskip("leidenalg")
    pytest.importorskip("phate")
    names, rows, block, g_rows, g_genes, cells = _planted(n_blocks=5, size=10, noise=0.3, seed=2)
    mask = block >= 0

    def run(seed, draw_seed):
        """One consensus with the given seeds."""
        return oc.guide_bootstrap_consensus(
            lambda b: oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, cells, draw_seed + b), 4, 5, mask,
            seed=seed, rows=rows)

    first, again, other = run(100, 8000), run(100, 8000), run(101, 8100)
    assert np.array_equal(first["labels"], again["labels"]) and np.array_equal(first["cooc"], again["cooc"])
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(first["labels"][mask], other["labels"][mask]) >= 0.8
    assert adjusted_rand_score(block[mask], first["labels"][mask]) >= 0.8


def test_real_consensus_gives_an_ambiguous_gene_lower_stability():
    pytest.importorskip("leidenalg")
    pytest.importorskip("phate")
    names, rows, block, g_rows, g_genes, cells = _planted(n_blocks=4, size=10, noise=0.05, seed=3)
    # an ambiguous gene: half its guides look like module 0, half like module 1
    amb = np.vstack([np.eye(8)[0] + 0.05 * np.random.default_rng(9).standard_normal((3, 8)),
                     np.eye(8)[1] + 0.05 * np.random.default_rng(10).standard_normal((3, 8))])
    names = np.r_[names[:-2], ["AMB"], names[-2:]]
    rows = np.vstack([rows[:-2], amb.mean(0), rows[-2:]])
    block = np.r_[block[:-2], [9], block[-2:]]
    g_rows, g_genes = np.vstack([g_rows, amb]), np.r_[g_genes, ["AMB"] * 6]
    cells = np.r_[cells, np.full(6, 20.0)]
    mask = block >= 0
    out = oc.guide_bootstrap_consensus(
        lambda b: oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, cells, 8000 + b), 10, 4, mask,
        rows=rows)
    stability = out["stability"]
    assert stability[names == "AMB"][0] < np.median(stability[mask & (names != "AMB")])


# ----------------------------------------------------------------------------------------------- phases


def _phase_settings(tmp_path, **extra):
    """Settings of a consensus / bench run on planted rows with a guide table."""
    names, rows, block, g_rows, g_genes, cells = _planted()
    columns = [f"c{j}" for j in range(rows.shape[1])]
    table = pd.DataFrame(rows, columns=columns)
    table.insert(0, "gene", names)
    table.insert(1, "n_cells", 80)
    table.to_parquet(tmp_path / "rows.parquet", index=False)
    guides = pd.DataFrame(g_rows, columns=columns)
    guides.insert(0, "gene", g_genes)
    guides.insert(1, "n_cells", cells)
    guides.to_parquet(tmp_path / "guides.parquet", index=False)
    ref_names = names[block >= 0]
    return {"dst_root": str(tmp_path / "run"), "ops_cluster_rows": str(tmp_path / "rows.parquet"),
            "ops_consensus_guides": str(tmp_path / "guides.parquet"),
            "ops_cluster_reference_dir": str(_references(tmp_path / "ref", ref_names, block[block >= 0])),
            "ops_consensus_bootstraps": 3, "ops_consensus_k": 4, **extra}


def test_the_consensus_phase_writes_its_outputs(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = run_ops(_phase_settings(tmp_path), phases=["cluster_consensus"])
    folder = tmp_path / "run" / "results" / "ops_clusters"
    partition = pd.read_csv(folder / "consensus_partition.csv")
    assert list(partition.columns) == ["gene", "cluster", "stability", "unassigned", "targeting"]
    assert partition.cluster.notna().all() and partition[partition.targeting].cluster.nunique() == 4
    assert len(pd.read_csv(folder / "consensus_runs.csv")) == 3
    saved = np.load(folder / "consensus_cooc.npz")
    assert saved["cooc"].shape == (len(partition), len(partition))
    assert (folder / "consensus.png").stat().st_size > 1000
    assert out["outputs"]["consensus_partition"].endswith("consensus_partition.csv")
    record = (tmp_path / "run" / "ops_profile" / "A" / "runtime.json").read_text()
    assert "ops_consensus_bootstraps" in record


def test_the_bench_phase_tests_and_gates(monkeypatch, tmp_path):
    _install(monkeypatch)
    settings = _phase_settings(tmp_path, ops_bench_replicates=3, ops_bench_gate_copies=2,
                               ops_bench_null_permutations=5, ops_bench_half=None)
    out = run_ops(settings, phases=["cluster_bench_test"])
    folder = tmp_path / "run" / "results" / "ops_clusters"
    stats = pd.read_csv(folder / "test_stats.csv")
    assert list(stats.unit) == ["complexes", "STRING pairs", "STRING pairs, per-gene capped"]
    assert {"diff_pts", "ci_low", "ci_high", "p", "q", "margin_pts", "call", "gate", "calibrated"} <= set(stats)
    assert len(pd.read_csv(folder / "calibration_gate.csv")) == 2 * 3
    assert len(pd.read_csv(folder / "bench_replicates.csv")) == 2 * 3
    assert (folder / "bench_test.png").stat().st_size > 1000
    assert out["outputs"]["test_stats"].endswith("test_stats.csv")


def test_changing_r_without_a_gate_record_is_refused(monkeypatch, tmp_path):
    _install(monkeypatch)
    settings = _phase_settings(tmp_path, ops_bench_replicates=40, ops_bench_gate_copies=0)
    with pytest.raises(ValueError, match="gate record"):
        oc.cluster_bench_test_phase(settings, None, None, str(tmp_path))
    with pytest.raises(ValueError, match="ops_bench_gate_arm"):
        oc.cluster_bench_test_phase({**settings, "ops_bench_gate_arm": "kmeans"}, None, None, str(tmp_path))


def test_a_gene_list_with_unknown_genes_or_missing_guides_is_refused(monkeypatch, tmp_path):
    _install(monkeypatch)
    settings = _phase_settings(tmp_path)
    pd.DataFrame({"gene": ["G0_0", "NOPE"]}).to_csv(tmp_path / "list.csv", index=False)
    with pytest.raises(ValueError, match="not profile rows"):
        oc.cluster_consensus_phase({**settings, "ops_cluster_gene_list": str(tmp_path / "list.csv")}, None, None,
                                   str(tmp_path))
    with pytest.raises(ValueError, match="guide rows"):
        oc.cluster_consensus_phase({**settings, "ops_consensus_guides": None}, None, None, str(tmp_path))


def test_the_default_settings_leave_the_sweep_outputs_identical(monkeypatch, tmp_path):
    """Legacy identity: with the new settings at their defaults, cluster_sweep writes the same files."""
    from spacr.ops_settings import OPS_DEFAULTS

    _install(monkeypatch)
    from spacr.settings import ALPHA_FEATURES

    keys = [*ALPHA_FEATURES[928]["settings"], *ALPHA_FEATURES[929]["settings"]]
    assert len(keys) == 25 and OPS_DEFAULTS["ops_consensus_enabled"] is False
    base = _phase_settings(tmp_path, ops_cluster_knn=4, ops_cluster_seeds="42",
                           ops_cluster_sweep_resolutions={"knn": [1], "phate": [3]}, ops_cluster_target_clusters=4)
    for key in ("ops_consensus_bootstraps", "ops_consensus_k", "ops_consensus_guides"):
        base.pop(key)
    outputs = {}
    for tag, extra in (("legacy", {}), ("defaults", {k: OPS_DEFAULTS[k] for k in keys})):
        settings = {**base, **extra, "dst_root": str(tmp_path / tag)}
        run_ops(settings, phases=["cluster_sweep"])
        folder = tmp_path / tag / "results" / "ops_clusters"
        # gzip headers carry the write time, so compressed files are compared by their content
        outputs[tag] = {p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
                        for p in sorted(folder.iterdir()) if p.suffix != ".png"}
    assert outputs["legacy"] == outputs["defaults"]
    assert not any(name.startswith(("consensus", "bench", "test_stats")) for name in outputs["defaults"])


def test_the_new_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    for item, prefix in ((928, "ops_consensus_"), (929, "ops_bench_")):
        keys = ALPHA_FEATURES[item]["settings"]
        assert keys and all(k.startswith(prefix) for k in keys)
        for key in keys:
            assert key in OPS_DEFAULTS and len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_bench_replicates"] == oc.GATED_REPLICATES == 20
    assert OPS_DEFAULTS["ops_consensus_bootstraps"] == 10 and OPS_DEFAULTS["ops_consensus_method"] == "average"
