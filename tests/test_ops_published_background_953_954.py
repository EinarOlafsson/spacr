"""Published comparator control rows in the Fisher background (item 953) and
the calibrated benchmark's 1,000-permutation null (item 954).

953: ``published_clusters`` keeps a comparator's own control rows, and
``score_published`` counts them in the Fisher background of the CORUM and
KEGG shares only, as the publications (and Brieflow's code) did; they are
never cluster members, enrichment members or STRING pairs. The reference
for the background is a plain Fisher test written out here, with the
background size the publications used (genes plus control rows).

954: ``ops_bench_null_permutations`` defaults to 1,000; the size-matched
null draws its permutations from the same seeded generator, so the first
200 are the 200 of a file saved before item 954, which reruns with 200.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from spacr import ops_clusters as oc
from spacr.ops_engine import run_ops

from tests.test_ops_leiden_sweep_916 import _blocks, _install, _references, _sweep_settings


def _funk_style(tmp_path, names, block, controls=12):
    """A Funk-style table: every gene with a cluster, plus control rows with published clusters."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({"gene_symbol": list(names) + [f"nontargeting_{i}_0" for i in range(controls)],
                         "interphase_cluster": list(block) + [i % 3 for i in range(controls)]})
    path = tmp_path / "funk.csv"
    rows.to_csv(path, index=False)
    return {"name": "Funk published", "table": str(path), "cluster": "interphase_cluster"}


def _share_by_hand(groups, genes, labels, n_background):
    """The share of clusters with a BH-significant Fisher group (every member counted), by hand."""
    from statsmodels.stats.multitest import multipletests

    members = {g: set(d.gene_name) for g, d in groups.groupby("group")}
    hits = []
    for cluster in np.unique(labels):
        in_cluster = set(np.asarray(genes)[labels == cluster])
        p = []
        for name in members:
            a, k, n = len(in_cluster & members[name]), len(members[name]), len(in_cluster)
            p.append(fisher_exact([[a, k - a], [n - a, max(n_background - k - n + a, 0)]])[1])
        hits.append(bool((multipletests(p, method="fdr_bh")[1] < 0.05).any()))
    return round(100 * float(np.mean(hits)), 1)


def _controls_of(published):
    """The control rows of one published clustering and their clusters."""
    names = [g for g in published if str(g).startswith("nontargeting")]
    return names, [published[g] for g in names]


def test_control_rows_are_kept_with_their_published_clusters(tmp_path):
    names, _rows, block = _blocks(n_blocks=4, size=10)
    spec = _funk_style(tmp_path, names, block)
    kept = oc.published_clusters(spec)
    dropped = oc.published_clusters(spec, keep_controls=False)
    assert len(kept) == len(names) + 12 and len(dropped) == len(names)
    assert kept["nontargeting_4_0"] == 1 and all(dropped[g] == kept[g] for g in dropped)
    assert oc.published_clusters(spec, "NONTARGETING", keep_controls=False) == dropped


def test_the_background_grows_by_the_control_rows_and_controls_are_never_members(tmp_path):
    names, _rows, block = _blocks(n_blocks=8, size=6, noise=0.0)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", names, block)))
    labels = block.copy()
    half = np.arange(len(block)) % 6 < 3
    labels[half] = (labels[half] + 1) % 8
    spec = _funk_style(tmp_path / "t", names, labels, controls=200)
    kept = oc.published_clusters(spec)
    with_controls = oc.score_published({"Funk": kept}, names, refs, shuffles=0).iloc[0]
    without = oc.score_published({"Funk": oc.published_clusters(spec, keep_controls=False)}, names, refs,
                                 shuffles=0).iloc[0]
    corum = refs["corum"]
    assert with_controls["CORUM_pct_enriched_may2025"] == _share_by_hand(corum, names, labels, len(names) + 200)
    assert without["CORUM_pct_enriched_may2025"] == _share_by_hand(corum, names, labels, len(names))
    assert with_controls["CORUM_pct_enriched_may2025"] != without["CORUM_pct_enriched_may2025"]
    for key in ("clusters", "genes", "STRING_tp", "STRING_fp", "STRING_fn", "STRING_precision",
                "complexes_recovered_pct", "CORUM_partner_in_cluster_pct", "REPORT_complexes"):
        assert with_controls[key] == without[key], key
    assert with_controls["genes"] == len(names) and with_controls["clusters"] == 8
    controls, control_labels = _controls_of(kept)
    expected = oc.score_clusters("Funk", np.r_[names, controls], np.r_[labels, control_labels],
                                 corum=refs["corum"], kegg=refs["kegg"], positives=refs["positives"])
    p, r = expected.pop("STRING_precision"), expected.pop("STRING_recall")
    expected.pop("STRING_f1")
    assert with_controls["STRING_f1"] == 2 * p * r / (p + r)
    for key, value in expected.items():
        assert with_controls[key] == value, key


def test_the_shuffled_control_permutes_the_scored_genes_only(tmp_path):
    names, _rows, block = _blocks(n_blocks=6, size=8)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", names, block)))
    kept = oc.published_clusters(_funk_style(tmp_path / "t", names, block, controls=30))
    rows = oc.score_published({"Funk": kept}, names, refs, shuffles=2)
    rng = np.random.default_rng(oc.PUBLISHED_CONTROL_SEED)
    controls, control_labels = _controls_of(kept)
    for i in (1, 2):
        shuffled = block[rng.permutation(len(block))]
        want = oc.score_clusters("Funk", np.r_[names, controls], np.r_[shuffled, control_labels],
                                 corum=refs["corum"], kegg=refs["kegg"], positives=refs["positives"])
        assert rows.iloc[i]["CORUM_pct_enriched_may2025"] == want["CORUM_pct_enriched_may2025"]
        assert rows.iloc[i]["STRING_tp"] == want["STRING_tp"] and rows.iloc[i]["genes"] == len(names)


def _phase_files(folder):
    """Every output of a sweep run but the figure, gzip content uncompressed."""
    return {p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            for p in sorted(folder.iterdir()) if p.suffix != ".png"}


def _legacy_score_published(published, genes, refs, shuffles=1, seed=oc.PUBLISHED_CONTROL_SEED,
                            control_prefix="nontargeting"):
    """score_published as it was before item 953."""
    rng = np.random.default_rng(seed)
    out = []
    for name, clusters in published.items():
        scored = np.array([g for g in genes if g in clusters])
        labels = np.array([clusters[g] for g in scored])
        key = {"graph": "published", "seed": np.nan, "resolution": np.nan}
        out.append(dict(oc.score_partition(name, scored, labels, refs), **key, control="real"))
        for _ in range(int(shuffles)):
            out.append(dict(oc.score_partition(name, scored, labels[rng.permutation(len(labels))], refs),
                            **key, control="shuffled gene labels"))
    return pd.DataFrame(out)


def _legacy_published_clusters(spec, control_prefix="nontargeting", keep_controls=True):
    """published_clusters as it was before item 953: control rows always left out."""
    table = pd.read_csv(spec["table"]).dropna(subset=[spec["cluster"]])
    table = table[~table["gene_symbol"].astype(str).str.lower().str.startswith(control_prefix.lower())]
    return dict(zip(table["gene_symbol"], table[spec["cluster"]].astype(int)))


def test_the_phase_keeps_controls_by_default_and_false_is_the_old_output(monkeypatch, tmp_path):
    _install(monkeypatch)
    names, rows, block = _blocks(n_blocks=8, size=12)
    spec = _funk_style(tmp_path / "t", names, block % 4, controls=40)
    outputs, records = {}, {}
    for tag, extra in (("default", {}), ("false", {"ops_cluster_published_controls": False}),
                       ("true", {"ops_cluster_published_controls": True})):
        (tmp_path / tag).mkdir()
        run_ops(_sweep_settings(tmp_path / tag, names, rows, block, ops_cluster_published=[spec], **extra),
                phases=["cluster_sweep"])
        folder = tmp_path / tag / "run" / "results" / "ops_clusters"
        outputs[tag] = _phase_files(folder)
        records[tag] = json.loads((folder / "published_comparators.json").read_text())
    assert outputs["default"] == outputs["true"] != outputs["false"]
    first = records["true"]["comparators"][0]
    assert first["control_rows_in_background"] == 40 and first["genes_scored"] == len(names)
    assert "control_rows_in_background" in records["true"]["legend"]
    assert "control_rows_in_background" not in records["false"]["comparators"][0]
    assert "control_rows_in_background" not in records["false"]["legend"]
    record = oc.published_record
    monkeypatch.setattr(oc, "score_published", _legacy_score_published)
    monkeypatch.setattr(oc, "published_clusters", _legacy_published_clusters)
    monkeypatch.setattr(oc, "published_record", lambda *args: record(*args[:7]))
    (tmp_path / "old").mkdir()
    run_ops(_sweep_settings(tmp_path / "old", names, rows, block, ops_cluster_published=[spec]),
            phases=["cluster_sweep"])
    assert _phase_files(tmp_path / "old" / "run" / "results" / "ops_clusters") == outputs["false"]


def test_settings_files_saved_before_953_and_954_rerun_the_old_way(tmp_path):
    import csv

    from spacr.cli import resolve_module, resolve_settings
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                SETTINGS_VERSION_KEY)

    assert SETTINGS_SCHEMA_VERSION >= 942
    assert DEFAULT_CHANGES[941] == {"ops_cluster_published_controls": False}
    assert DEFAULT_CHANGES[942] == {"ops_bench_null_permutations": 200}
    assert OPS_DEFAULTS["ops_cluster_published_controls"] is True
    assert OPS_DEFAULTS["ops_bench_null_permutations"] == 1000
    assert ALPHA_FEATURES[953]["settings"] == ("ops_cluster_published_controls",)
    ops = resolve_module("ops")
    for stamp, controls, nperm in (("", False, 200), ("940", False, 200), ("941", True, 200),
                                   (str(SETTINGS_SCHEMA_VERSION), True, 1000)):
        path = tmp_path / f"old{stamp}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Key", "Value"])
            writer.writerows(([(SETTINGS_VERSION_KEY, stamp)] if stamp else []) + [("plate", "P1")])
        loaded = resolve_settings(ops, str(path))
        assert loaded["ops_cluster_published_controls"] is controls
        assert loaded["ops_bench_null_permutations"] == nperm
        assert oc._bench_options(loaded).nperm == nperm


def test_the_first_200_permutations_of_the_1000_null_are_the_old_200():
    import hashlib

    rng = np.random.default_rng(3)
    reference = {"cx_members": [rng.choice(60, 4, replace=False) for _ in range(25)]}
    labels = np.repeat(np.arange(12), 5)
    sizes = oc._sizes(labels)
    old, new = oc.size_matched_null(sizes, reference, 200), oc.size_matched_null(sizes, reference, 1000)
    gen = np.random.default_rng(int(hashlib.md5(str(sizes).encode()).hexdigest()[:8], 16))
    base = np.repeat(np.arange(len(sizes)), sizes)
    draws = np.array([oc._recovered(base[gen.permutation(len(base))], reference) for _ in range(1000)])
    assert np.array_equal(draws[:200].mean(0), old["complex_prob"])
    assert np.array_equal(draws.mean(0), new["complex_prob"])
    assert old["pair_prob"] == new["pair_prob"]
    assert oc._bench_options({}).nperm == 1000
    assert oc._bench_options({"ops_bench_null_permutations": 200}).nperm == 200


def test_the_new_setting_is_alpha_with_a_short_tooltip():
    from spacr.ops_settings import OPS_TOOLTIPS, OPS_TYPES

    for key in ("ops_cluster_published_controls", "ops_bench_null_permutations"):
        assert len(OPS_TOOLTIPS[key]) <= 600 and OPS_TOOLTIPS[key].endswith("Alpha.")
    assert OPS_TYPES["ops_cluster_published_controls"] is bool
    assert "item 953" in OPS_TOOLTIPS["ops_cluster_published_controls"]
    assert "Default 1000" in OPS_TOOLTIPS["ops_bench_null_permutations"]
    assert "ops_cluster_published_keep_controls" not in OPS_TOOLTIPS


def test_a_single_seed_read_back_from_a_settings_file_is_one_seed():
    assert oc._sweep_setting_list(42, (1, 2), int) == [42]
    assert oc._sweep_setting_list("42,1", (1, 2), int) == [42, 1]
    assert oc._sweep_setting_list([3], (1, 2), int) == [3]
    assert oc._sweep_setting_list(None, (1, 2), int) == [1, 2]
