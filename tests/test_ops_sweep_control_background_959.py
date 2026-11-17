"""spaCR's own sweep rows count the run's control rows in the Fisher background (item 959).

Since item 953 a published comparator's control rows enter the Fisher
background of its CORUM and KEGG shares. Item 959 gives spaCR's own sweep
rows (``score_partitions``) the same background through the same helper, so
every row of a sweep table uses one background. The control rows are never
cluster members, enrichment members or STRING pairs; the STRING and complex
metrics and the shuffled-label draws do not change. The option is off by
default (maintainer decision 2026-11-17): with
``ops_cluster_sweep_controls_background`` False, the default for new runs
and every settings file, the rows are the rows from before the change.
"""
from __future__ import annotations

import csv
import gzip
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from spacr import ops_clusters as oc
from spacr.ops_engine import run_ops

from tests.test_ops_leiden_sweep_916 import _blocks, _install, _references, _sweep_settings

UNCHANGED = ("clusters", "genes", "STRING_precision", "STRING_recall", "STRING_f1", "STRING_tp",
             "STRING_fp", "STRING_fn", "REPORT_complexes", "complexes_recovered_pct",
             "CORUM_partner_in_cluster_pct")


def _with_controls(names, rows, block, controls, seed=5):
    """Planted rows plus ``controls`` non-targeting rows of noise, in one table."""
    rng = np.random.default_rng(seed)
    extra = 0.05 * rng.standard_normal((controls, rows.shape[1]))
    return (np.r_[names, [f"nontargeting_{i}" for i in range(controls)]], np.vstack([rows, extra]),
            np.r_[block, rng.integers(0, block.max() + 1, controls)])


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


def _legacy_score_partitions(names, partitions, refs, shuffles=1, seed=oc.SWEEP_CONTROL_SEED, *,
                             genes=None, label="spaCR", report=None, **_ignored):
    """score_partitions as it was before item 959: the targeting genes only."""
    names = np.asarray(names).astype(str)
    if genes is None:
        genes = names[~oc._is_control(names, "nontargeting")]
    genes = np.asarray(genes).astype(str)
    position = pd.Series(np.arange(len(names)), index=names).loc[genes].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for (graph, run_seed, resolution), labels in partitions.items():
        scored = np.asarray(labels)[position]
        key = {"graph": graph, "seed": int(run_seed), "resolution": float(resolution)}
        real = dict(oc.score_partition(label, genes, scored, refs), **key, control="real")
        rows.append(real)
        for _ in range(int(shuffles)):
            rows.append(dict(oc.score_partition(label, genes, scored[rng.permutation(len(scored))], refs),
                             **key, control="shuffled gene labels"))
        if report is not None:
            report(f"{graph} seed {run_seed} r {resolution}: {real['clusters']} clusters, "
                   f"CORUM {real['CORUM_pct_enriched_may2025']} %, "
                   f"STRING F1 {round(real['STRING_f1'], 4)}")
    return pd.DataFrame(rows)


def _mixed_labels(block, n_blocks):
    """Half of every block moved to the next cluster, so the enrichment depends on the background."""
    labels = block.copy()
    half = np.arange(len(block)) % 6 < 3
    labels[half] = (labels[half] + 1) % n_blocks
    return labels


def test_the_sweep_background_grows_by_the_run_control_rows_only(tmp_path):
    genes, rows, block = _blocks(n_blocks=8, size=6, noise=0.0)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", genes, block)))
    names, _rows, labels = _with_controls(genes, rows, _mixed_labels(block, 8), controls=200)
    parts = {("knn4", 42, 1.0): labels}
    new = oc.score_partitions(names, parts, refs, shuffles=0, controls_background=True).iloc[0]
    old = oc.score_partitions(names, parts, refs, shuffles=0).iloc[0]
    scored = labels[: len(genes)]
    corum = refs["corum"]
    assert new["CORUM_pct_enriched_may2025"] == _share_by_hand(corum, genes, scored, len(genes) + 200)
    assert old["CORUM_pct_enriched_may2025"] == _share_by_hand(corum, genes, scored, len(genes))
    assert new["CORUM_pct_enriched_may2025"] != old["CORUM_pct_enriched_may2025"]
    for key in UNCHANGED:
        assert new[key] == old[key], key
    assert new["background_rows"] == len(genes) + 200 and new["control_rows_in_background"] == 200
    assert "background_rows" not in old and "control_rows_in_background" not in old


def test_sweep_and_comparator_rows_share_one_helper_and_one_background(tmp_path):
    genes, rows, block = _blocks(n_blocks=6, size=8)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", genes, block)))
    names, _rows, labels = _with_controls(genes, rows, _mixed_labels(block, 6), controls=40)
    sweep = oc.score_partitions(names, {("phate", 1, 2.0): labels}, refs, shuffles=2,
                                controls_background=True)
    published = oc.score_published({"same partition": dict(zip(names, labels))}, genes, refs, shuffles=0,
                                   record_background=True)
    real = sweep[sweep.control == "real"].iloc[0]
    theirs = published.iloc[0]
    for key in UNCHANGED + ("CORUM_pct_enriched_head", "CORUM_pct_enriched_may2025",
                            "KEGG_pct_enriched_head", "KEGG_pct_enriched_may2025",
                            "background_rows", "control_rows_in_background"):
        assert real[key] == theirs[key], key
    assert sweep.background_rows.nunique() == 1 and sweep.control_rows_in_background.eq(40).all()


def test_the_shuffled_controls_draw_the_same_permutations(tmp_path):
    genes, rows, block = _blocks(n_blocks=6, size=8)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", genes, block)))
    names, _rows, labels = _with_controls(genes, rows, block, controls=30)
    parts = {("knn4", 42, 1.0): labels, ("knn4", 42, 2.0): labels}
    new = oc.score_partitions(names, parts, refs, shuffles=3, controls_background=True)
    old = oc.score_partitions(names, parts, refs, shuffles=3)
    pd.testing.assert_frame_equal(new[list(UNCHANGED)], old[list(UNCHANGED)])
    rng = np.random.default_rng(oc.SWEEP_CONTROL_SEED)
    scored = labels[: len(genes)]
    controls = names[len(genes):]
    shuffled = scored[rng.permutation(len(scored))]
    want = oc.score_clusters("spaCR", np.r_[genes, controls], np.r_[shuffled, labels[len(genes):]],
                             corum=refs["corum"], kegg=refs["kegg"], positives=refs["positives"])
    assert new.iloc[1]["CORUM_pct_enriched_may2025"] == want["CORUM_pct_enriched_may2025"]
    assert new.iloc[1]["KEGG_pct_enriched_head"] == want["KEGG_pct_enriched_head"]


def test_off_is_the_scoring_from_before_959_and_a_gene_list_keeps_its_controls(tmp_path):
    genes, rows, block = _blocks(n_blocks=6, size=8)
    refs = oc.sweep_references(str(_references(tmp_path / "ref", genes, block)))
    names, _rows, labels = _with_controls(genes, rows, block, controls=25)
    parts = {("knn4", 42, 1.0): labels, ("phate", 1, 3.0): labels}
    off = oc.score_partitions(names, parts, refs, shuffles=2)
    assert off.to_csv(index=False) == _legacy_score_partitions(names, parts, refs, shuffles=2).to_csv(index=False)
    picked = oc.score_partitions(names, parts, refs, shuffles=0, genes=genes[:20], controls_background=True)
    assert picked.genes.eq(20).all() and picked.control_rows_in_background.eq(25).all()
    assert picked.background_rows.eq(45).all()
    none = oc.score_partitions(genes, {("knn4", 42, 1.0): block}, refs, shuffles=1, controls_background=True)
    plain = oc.score_partitions(genes, {("knn4", 42, 1.0): block}, refs, shuffles=1)
    pd.testing.assert_frame_equal(none.drop(columns=["background_rows", "control_rows_in_background"]), plain)
    assert none.control_rows_in_background.eq(0).all() and none.background_rows.eq(len(genes)).all()


def _phase_files(folder):
    """Every output of a sweep run but the figure, gzip content uncompressed."""
    return {p.name: gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            for p in sorted(folder.iterdir()) if p.suffix != ".png"}


def _funk_style(folder, names, block, controls=12):
    """A Funk-style published table with its own control rows."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({"gene_symbol": list(names) + [f"nontargeting_{i}_0" for i in range(controls)],
                         "interphase_cluster": list(block) + [i % 3 for i in range(controls)]})
    rows.to_csv(folder / "funk.csv", index=False)
    return {"name": "Funk published", "table": str(folder / "funk.csv"), "cluster": "interphase_cluster"}


def test_the_phase_counts_the_controls_when_on_and_the_default_is_the_old_output(monkeypatch, tmp_path):
    _install(monkeypatch)
    genes, rows, block = _blocks(n_blocks=8, size=12)
    names, table_rows, _labels = _with_controls(genes, rows, block, controls=16)
    spec = _funk_style(tmp_path / "t", genes, block % 4, controls=16)
    outputs, records, tables = {}, {}, {}
    for tag, extra in (("default", {}), ("true", {"ops_cluster_sweep_controls_background": True}),
                       ("false", {"ops_cluster_sweep_controls_background": False})):
        (tmp_path / tag).mkdir()
        settings = _sweep_settings(tmp_path / tag, names, table_rows, np.r_[block, np.zeros(16, int)],
                                   ops_cluster_published=[spec], ops_cluster_gene_list=None, **extra)
        run_ops(settings, phases=["cluster_sweep"])
        folder = tmp_path / tag / "run" / "results" / "ops_clusters"
        outputs[tag] = _phase_files(folder)
        records[tag] = json.loads((folder / "published_comparators.json").read_text())
        tables[tag] = pd.read_csv(folder / "sweep_rows_with_published.csv")
    assert outputs["default"] == outputs["false"] != outputs["true"]
    on = tables["true"]
    assert on.control_rows_in_background.eq(16).all()
    assert on.background_rows.nunique() == 1 and on.background_rows.iloc[0] == len(genes) + 16
    assert "background_rows" not in tables["false"].columns
    assert "one background rule" in records["true"]["legend"] and "(16)" in records["true"]["legend"]
    assert "background_rows" not in records["false"]["legend"]
    sweep = pd.read_csv(tmp_path / "true" / "run" / "results" / "ops_clusters" / "sweep_rows.csv")
    old_sweep = pd.read_csv(tmp_path / "false" / "run" / "results" / "ops_clusters" / "sweep_rows.csv")
    pd.testing.assert_frame_equal(sweep[list(UNCHANGED)], old_sweep[list(UNCHANGED)])
    monkeypatch.setattr(oc, "score_partitions", _legacy_score_partitions)
    published = oc.score_published
    monkeypatch.setattr(oc, "score_published", lambda *args, record_background=False, **kw: published(*args, **kw))
    record = oc.published_record
    monkeypatch.setattr(oc, "published_record", lambda *args: record(*args[:8]))
    (tmp_path / "old").mkdir()
    run_ops(_sweep_settings(tmp_path / "old", names, table_rows, np.r_[block, np.zeros(16, int)],
                            ops_cluster_published=[spec], ops_cluster_gene_list=None,
                            ops_cluster_sweep_controls_background=False),
            phases=["cluster_sweep"])
    assert _phase_files(tmp_path / "old" / "run" / "results" / "ops_clusters") == outputs["false"]


def test_the_option_is_off_by_default_for_new_runs_and_every_settings_file(tmp_path):
    from spacr.cli import resolve_module, resolve_settings
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                SETTINGS_VERSION_KEY)

    key = "ops_cluster_sweep_controls_background"
    assert all(key not in row for row in DEFAULT_CHANGES.values())
    assert OPS_DEFAULTS[key] is False and OPS_TYPES[key] is bool
    assert ALPHA_FEATURES[959]["settings"] == (key,)
    assert len(OPS_TOOLTIPS[key]) <= 600 and OPS_TOOLTIPS[key].endswith("Alpha.")
    assert "item 959" in OPS_TOOLTIPS[key] and "Default False" in OPS_TOOLTIPS[key]
    ops = resolve_module("ops")
    for stamp, expected in (("", False), ("940", False), (str(SETTINGS_SCHEMA_VERSION), False)):
        path = tmp_path / f"old{stamp}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Key", "Value"])
            writer.writerows(([(SETTINGS_VERSION_KEY, stamp)] if stamp else []) + [("plate", "P1")])
        assert resolve_settings(ops, str(path))[key] is expected, stamp
