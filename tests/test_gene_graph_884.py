"""Item 884: the gene graph (spacr.gene_graph), on a small synthetic screen, CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import spacr.gene_graph as gg
from tests.gene_graph_synthetic import write_synthetic

FOLDS = [f"fold{k}" for k in range(5)]
ARMS = ("full", "noJUMP", "noOPS", "graph")


def _configs():
    names = [f"{t}__gps__{a}__none__s0" for t in ["time"] + FOLDS for a in ARMS]
    names += [f"{t}__{m}__noJUMP__none__s0" for t in ["time"] + FOLDS for m in ("lr", "cosine")]
    names += [f"{t}__prior__graph__none__s0" for t in ["time"] + FOLDS] + ["time__adamicadar__graph__none__s0"]
    names += ["fold0__lr__spectral__none__s0", "time__lr__spectral__none__s0", "fold0__gps__full__shufX__s0",
              "fold0__sage__full__none__s0", "fold0__gcn__full__none__s0", "time__gcn__full__none__s0"]
    return names


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("gg")
    s = write_synthetic(root)
    s["gene_graph_configs"] = _configs()
    s["gene_graph_figures"] = True
    result = gg.run_gene_graph(s)
    return {"root": root, "settings": s, "result": result, "out": Path(s["gene_graph_src"]) / "gene_graph"}


def _opts(s, **kw):
    s = dict(s)
    s.update({f"gene_graph_{k}": v for k, v in kw.items()})
    return gg.options(s)


def test_defaults_register_and_round_trip_text_values():
    from spacr.settings import expected_types, tooltips

    d = gg.get_gene_graph_default_settings({"gene_graph_hidden": 7})
    assert d["gene_graph_hidden"] == 7 and d["gene_graph_seed"] == 20261108
    assert set(d) == set(gg._DEFAULTS)
    assert all(k in tooltips and k in expected_types for k in gg._DEFAULTS)
    cats = [k for keys in gg._CATEGORIES.values() for k in keys]
    assert sorted(cats) == sorted(gg._DEFAULTS)
    o = gg.options({"gene_graph_arms": "{'a': ['ops']}", "gene_graph_tasks": "['time']"})
    assert o["arms"] == {"a": ["ops"]} and o["tasks"] == ["time"]
    assert gg._literal('{"x": 1}') == {"x": 1}
    with pytest.raises(ValueError):
        gg.paths(gg.options({}))
    assert gg.source_label("STRINGnew>=0.7") == "STRING" and gg.source_label("BioGRID2022") == "BioGRID"
    assert gg.known_key("STRING>=0.4") == "STRING04"


def test_hadamard_lr_identity():
    torch.manual_seed(0)
    dims = {"a": 3, "b": 2}
    m = gg.GeneGraphModel("lr", dims, n_cat=4)
    with torch.no_grad():
        m.r.copy_(torch.randn_like(m.r))
    X = {"a": torch.randn(6, 3), "b": torch.randn(6, 2)}
    H = {"a": torch.ones(6, dtype=torch.bool), "b": torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)}
    z, zc = m.embed(X, H, {"N": 6})
    assert zc is None and z.shape == (6, 3 + 1 + 2 + 1)
    x = torch.cat([X["a"], torch.ones(6, 1), X["b"] * H["b"][:, None], H["b"][:, None].float()], 1)
    expected = x @ torch.diag(m.r) @ x.T * m.scale + m.b
    assert torch.allclose(m.pair_matrix(z, z), expected, atol=1e-6)
    assert torch.allclose(m.pair_logit(z[:2], z[2:4]), expected[[0, 1], [2, 3]], atol=1e-6)
    assert m.cat_logits(z, None).shape == (6, 4)
    with pytest.raises(KeyError):
        gg.GeneGraphModel("mlp", dims, 0)


def test_rwse_matches_numpy_with_an_isolated_node():
    e = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]])
    ei = torch.as_tensor(np.concatenate([e, e[:, ::-1]]).T.copy())
    pe, ld = gg.rwse(6, ei, k=5)
    A = np.zeros((6, 6))
    A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1
    P = A / np.maximum(A.sum(1, keepdims=True), 1)
    expect = np.stack([np.diag(np.linalg.matrix_power(P, k)) for k in range(1, 6)], 1)
    assert np.allclose(pe.numpy(), expect, atol=1e-6)
    assert np.all(pe.numpy()[5] == 0) and ld[5, 0] == 0
    assert np.isclose(ld[2, 0].item(), np.log1p(3))


def test_rewire_keeps_degrees_without_self_loops_or_duplicates():
    rng = np.random.default_rng(1)
    E = np.array(sorted({tuple(sorted(rng.choice(30, 2, replace=False))) for _ in range(80)}))
    R = gg.rewire_pairs(E, np.random.default_rng(2), 2 * len(E))
    assert np.array_equal(np.bincount(E.ravel(), minlength=30), np.bincount(R.ravel(), minlength=30))
    assert (R[:, 0] != R[:, 1]).all() and len(set(map(tuple, R))) == len(R) and (R[:, 0] < R[:, 1]).all()
    assert len(set(map(tuple, R)) - set(map(tuple, E))) > 0
    B = np.array([(g, c) for g in range(20) for c in range(4) if (g + c) % 3 == 0])
    RB = gg.rewire_pairs(B, np.random.default_rng(3), 2 * len(B), bipartite=True)
    for col in (0, 1):
        assert np.array_equal(np.bincount(B[:, col]), np.bincount(RB[:, col], minlength=B[:, col].max() + 1))
    assert len(set(map(tuple, RB))) == len(RB)
    assert gg.rewire_pairs(E[:1], rng, 5).tolist() == E[:1].tolist()


def test_qvalues_monotone_decoy_bound_and_minimum():
    rng = np.random.default_rng(0)
    cand, dec = rng.random(200), rng.random(50) * 0.5
    q, qn, qc = gg.qvalues(cand, dec)
    o = np.argsort(-cand)
    assert (np.diff(qn[o]) >= -1e-12).all() and (np.diff(q[o]) >= -1e-12).all() and (np.diff(qc[o]) >= -1e-12).all()
    assert np.allclose(q, np.maximum(qn, qc)) or (q >= np.maximum(qn, qc) - 1e-12).all()
    n0 = int((cand > dec.max()).sum())
    assert np.isclose(qn.min(), 200 / (50 * n0)) and np.isclose(qn[o[0]], qn.min())
    q2, qn2, _ = gg.qvalues(np.full(10, 0.1), np.full(5, 0.9))
    assert np.all(qn2 == 1) and np.all(q2 == 1)


def test_build_assembles_nodes_blocks_edges_and_folds(run):
    data = run["out"] / "data"
    summ = json.loads((data / "data_summary.json").read_text())
    assert summ["nodes"] == 66 and summ["real_genes"] == 60 and summ["nt_pseudogenes"] == 6
    assert summ["genes_missing_from_gene_folds"] == 2 and summ["classes"] == 5
    assert summ["blocks"]["ops"]["dims"] == 9 and summ["blocks"]["jcr"]["dims"] == 4
    assert summ["blocks"]["ops_tvn"]["dims"] == 3
    z = np.load(data / "graph.npz")
    nodes = pd.read_parquet(data / "nodes.parquet")
    assert "G_low" not in set(nodes.gene) and nodes.is_nt.sum() == 6
    assert set(nodes.fold) <= set(range(5))
    train = set().union(*[set(map(tuple, z[f"E_train_{k}"])) for k in ("STRING", "BioGRID", "CORUM")])
    for k in ("STRING", "BioGRID", "CORUM"):
        assert not set(map(tuple, z[f"E_new_{k}"])) & train
        assert (z[f"E_train_{k}"][:, 0] < z[f"E_train_{k}"][:, 1]).all()
    assert summ["train_union"] == len(train)
    assert nodes.go_terms[nodes.gene == "G00"].item() == 0 and nodes.go_terms[nodes.gene == "G04"].item() == 4
    assert nodes.string04_degree_genome[nodes.gene == "G59"].item() >= 1
    H = z["H_jcr"]
    assert abs(H[:60].mean() - summ["blocks"]["jcr"]["coverage_real"]) < 1e-12 and H[60:].any()
    assert np.isclose(np.sqrt((z["X_ops"][:60, :8] ** 2).mean()), 1.0)


def test_training_never_supervises_or_passes_messages_through_held_out_genes(run, monkeypatch):
    o = _opts(run["settings"], max_epochs=20)
    D = gg.GeneGraphData(run["out"] / "data")
    seen = {"msg": [], "pos": [], "neg": []}
    real_build, real_neg = gg.build_graph, gg.sample_negatives

    def build(D_, S, pairs, bits, gc, dev, kind, pe_k=16):
        """Record every message graph."""
        seen["msg"].append((pairs.copy(), gc.copy()))
        return real_build(D_, S, pairs, bits, gc, dev, kind, pe_k)

    def neg(rng, pos, pool, deg, n, keyset, N):
        """Record positives and negatives."""
        out = real_neg(rng, pos, pool, deg, n, keyset, N)
        seen["pos"].append(pos.copy())
        seen["neg"].append((out.copy(), keyset))
        return out

    monkeypatch.setattr(gg, "build_graph", build)
    monkeypatch.setattr(gg, "sample_negatives", neg)
    for control in ("none", "shufY", "rewire"):
        for v in seen.values():
            v.clear()
        name = f"fold2__gps__full__{control}__s0"
        S = gg.prepare(D, "fold2", ["ops", "jcr"], control, 0, o)
        gg.run_config(D, name, o, torch.device("cpu"), run["root"] / "leak_runs", prepared=S)
        test = S["test"]
        assert test.sum() > 5
        for pairs, gc in seen["msg"]:
            assert not test[pairs].any() and not test[gc[:, 0]].any()
        banned = test | S["val_gene"] | D.is_nt
        for pos in seen["pos"]:
            assert not banned[pos].any()
        known = set(map(tuple, S["U_all"]))
        for out, keyset in seen["neg"]:
            assert not banned[out].any()
            assert not set(map(tuple, out)) & known
            assert np.array_equal(keyset, np.sort(S["U_all"][:, 0] * D.N + S["U_all"][:, 1]))


def test_controls_permute_features_and_labels(run):
    o = _opts(run["settings"])
    D = gg.GeneGraphData(run["out"] / "data")
    base = gg.prepare(D, "fold0", ["ops", "jcr"], "none", 0, o)
    sx = gg.prepare(D, "fold0", ["ops", "jcr"], "shufX", 0, o)
    sy = gg.prepare(D, "fold0", ["ops", "jcr"], "shufY", 0, o)
    so = gg.prepare(D, "fold0", ["ops", "jcr"], "shufOPS", 0, o)
    ss = gg.prepare(D, "fold0", ["ops", "jcr"], "shufScreen", 0, o, screen_blocks=("ops", "jcr"))
    nt = D.is_nt
    assert np.array_equal(sx["X"]["ops"][nt], base["X"]["ops"][nt])
    assert not np.array_equal(sx["X"]["ops"], base["X"]["ops"])
    assert sorted(map(tuple, sx["X"]["ops"].round(6))) == sorted(map(tuple, base["X"]["ops"].round(6)))
    assert np.array_equal(sy["X"]["ops"], base["X"]["ops"]) and not np.array_equal(sy["U_all"], base["U_all"])
    assert np.array_equal(so["X"]["jcr"], base["X"]["jcr"]) and not np.array_equal(so["X"]["ops"], base["X"]["ops"])
    perm_ops = [np.flatnonzero((base["X"]["ops"] == r).all(1))[0] for r in ss["X"]["ops"][D.real]]
    perm_jcr = [i for i, r in zip(perm_ops, ss["X"]["jcr"][D.real]) if np.array_equal(base["X"]["jcr"][i], r)]
    assert len(perm_jcr) == D.real.sum()
    tm = gg.prepare(D, "time", ["ops"], "none", 0, o)
    assert tm["val_edge"].any() and not tm["use_cat"] and len(tm["GC"]) == 0
    sp = gg.prepare(D, "fold1", [], "none", 0, o, spectral=True)
    assert sp["X"]["spec"].shape[1] == 4 and not sp["H"]["spec"][sp["test"]].any()
    with pytest.raises(KeyError):
        D.blocks(["nope"])


def test_yield_resumes_to_identical_scores(run):
    o = _opts(run["settings"], max_epochs=20)
    D = gg.GeneGraphData(run["out"] / "data")
    a, b = run["root"] / "runs_a", run["root"] / "runs_b"
    name = "fold1__gps__full__none__s0"
    gg.run_config(D, name, o, torch.device("cpu"), a)
    calls = {"n": 0}

    def once():
        """Ask for a yield after the first epoch only."""
        calls["n"] += 1
        return calls["n"] == 1

    with pytest.raises(gg.GpuYield):
        gg.run_config(D, name, o, torch.device("cpu"), b, over_limit=once)
    assert (b / name / "resume.pt").exists() and not (b / name / "done.json").exists()
    assert gg.run_config(D, name, o, torch.device("cpu"), b, over_limit=once) == "ran"
    assert gg.run_config(D, name, o, torch.device("cpu"), b) == "done"
    za, zb = np.load(a / name / "scores.npz"), np.load(b / name / "scores.npz")
    for k in ("rows", "S", "C"):
        assert np.array_equal(za[k], zb[k])
    assert not (b / name / "resume.pt").exists()
    status = gg.train_runs(D, ["fold3__lr__full__none__s0"], o, run["root"] / "runs_c", over_limit=lambda: True)
    assert status["status"] == "yielded" and status["ran"] == []


def test_cross_fitted_isotonic_never_sees_its_own_fold(run, monkeypatch):
    D = gg.GeneGraphData(run["out"] / "data")
    R = gg.Reference(D)
    oof = gg.load_oof(run["out"] / "runs", "gps", "noJUMP", 5)
    oof = {f: {k: v.copy() for k, v in z.items()} for f, z in oof.items()}
    oof[0]["C"][:] = 1e6
    oof[0]["S"][:] = 1e6
    fits = []
    real = gg._isotonic

    def record(x, y):
        """Record what each fit sees."""
        fits.append(np.asarray(x).copy())
        return real(x, y)

    monkeypatch.setattr(gg, "_isotonic", record)
    Pc, rows = gg.cat_calibrated(R, oof, np.random.default_rng(0))
    assert not (fits[0] == 1e6).any() and all((f == 1e6).any() for f in fits[1:5])
    assert len(rows) == 5 and np.isfinite(Pc[D.real]).all()
    fits.clear()
    Pp, prow = gg.pair_calibrated(R, oof, np.random.default_rng(0))
    assert not (fits[0] == 1e6).any() and all((f == 1e6).any() for f in fits[1:5])
    assert {r["task"] for r in prow} == {"pair"}
    assert gg.load_oof(run["out"] / "runs", "gps", "missing_arm", 5) is None


def test_planted_modules_are_found_and_controls_fall_to_base(tmp_path):
    s = write_synthetic(tmp_path)
    s.update({"gene_graph_max_epochs": 200, "gene_graph_min_epochs": 60, "gene_graph_patience": 4,
              "gene_graph_hidden": 32, "gene_graph_lr_graph": 3e-3, "gene_graph_sensitivity_blocks": {}})
    s["gene_graph_configs"] = [f"fold0__{m}__noJUMP__{c}__s0" for m in ("gps", "sage", "cosine")
                               for c in ("none", "shufX", "shufY")]
    gg.build_gene_graph(s)
    gg.train_gene_graph(s)
    F = gg.evaluate_gene_graph(s)["heldout_metrics_per_fold"].set_index("name")
    ratio = F.link_AUPRC_2022 / F.link_base_2022
    for m in ("gps", "sage", "cosine"):
        assert ratio[f"fold0__{m}__noJUMP__none__s0"] > 3
        assert 0.5 < ratio[f"fold0__{m}__noJUMP__shufX__s0"] < 1.5
    for m in ("gps", "sage"):
        r = F.loc[f"fold0__{m}__noJUMP__shufY__s0"]
        assert r.cat_micro_AUPRC < 2 * r.cat_base and r.cat_micro_AUPRC < 0.5 * F.loc[f"fold0__{m}__noJUMP__none__s0"].cat_micro_AUPRC
        assert ratio[f"fold0__{m}__noJUMP__shufY__s0"] < 0.6 * ratio[f"fold0__{m}__noJUMP__none__s0"]
    assert (F.NT_cat_enrichment_top1pct == 0).all()


def test_run_writes_every_table_with_its_columns(run):
    t = run["out"] / "tables"
    res = run["result"]
    assert res["train"]["status"] == "done" and res["train"]["device"] == "cpu"
    for name in ("time_split_metrics", "time_split_bootstrap", "heldout_metrics_per_fold", "heldout_summary",
                 "ablation_table", "heldout_paired_deltas", "calibration", "run_configs",
                 "fdr_summary_gps_noJUMP", "fdr_summary_timesplit_gps_noJUMP",
                 "predictions_gene_category_gps_noJUMP", "predictions_gene_gene_gps_noJUMP",
                 "predictions_gene_gene_timesplit_gps_noJUMP"):
        assert (t / f"{name}.csv").exists(), name
    assert (t / "ablation_table.md").read_text().startswith("| model, arm |")
    T = pd.read_csv(t / "time_split_metrics.csv")
    assert {"AUPRC_any", "AUPRC_STRING", "P@10", "R@100", "P@10_degmatched", "AUPRC_any_lo", "NT_enrichment_top1000",
            "AUPRC_any_fit0"} <= set(T.columns)
    assert list(T.columns[:7]) == ["task", "model", "arm", "control", "seed", "name", "candidates"]
    F = pd.read_csv(t / "heldout_metrics_per_fold.csv")
    assert {"link_AUPRC_2022", "link_AUPRC_current", "link_geneMAP_2022", "cat_micro_AUPRC", "cat_macro_AUPRC",
            "cat_recall@10", "link_top5_precision_degmatched", "NT_cat_enrichment_top1pct"} <= set(F.columns)
    S = pd.read_csv(t / "heldout_summary.csv")
    assert (S.loc[(S.model == "gps") & (S.arm == "noJUMP"), "folds"] == 5).all()
    D = pd.read_csv(t / "heldout_paired_deltas.csv")
    assert list(D.columns) == ["metric", "A", "B", "A_mean", "B_mean", "delta", "delta_min", "delta_max",
                               "folds_A_better", "n"]
    row = D[(D.metric == "link_AUPRC_2022") & (D.A == "gps/noJUMP") & (D.B == "gps/graph")].iloc[0]
    g = F[(F.model == "gps") & (F.control == "none")].set_index(["arm", "task"]).link_AUPRC_2022
    deltas = g["noJUMP"] - g["graph"]
    assert np.isclose(row.delta, deltas.mean()) and row.folds_A_better == (deltas > 0).sum() and row.n == 5
    C = pd.read_csv(t / "calibration.csv")
    assert list(C.columns) == ["task", "fold", "ECE", "Brier", "mean_p", "base", "model", "arm"]
    assert set(C.arm) == set(ARMS)
    P = pd.read_csv(t / "predictions_gene_category_gps_noJUMP.csv")
    assert len(P) and {"gene", "class_id", "class", "source", "class_size", "p_calibrated", "q_NT", "q_cal",
                       "q_combined", "q", "string04_degree", "go_terms", "fold", "p_full", "p_graph",
                       "JUMP_CRISPR_cos", "JUMP_CRISPR_p", "JUMP_CRISPR_members", "delta_OPS_over_graph",
                       "delta_JUMP", "delta_OPS_given_JUMP", "drivers", "spec_flag_gene"} <= set(P.columns)
    assert len(P) and (P.q < 1.01).all() and not P.gene.str.startswith("ntpseudo").any()
    Pt = pd.read_csv(t / "predictions_gene_gene_timesplit_gps_noJUMP.csv")
    assert {"gene", "partner", "score", "gene_2022_degree", "q_NT", "spec_flag_partner"} <= set(Pt.columns)
    rule = json.loads((t / "poorly_characterised_rule.json").read_text())
    assert rule["poorly_characterised_genes"] > 0
    figs = run["out"] / "figures"
    for fn in ("fig1_heldout_category", "fig2_heldout_link", "fig3_time_split_auprc", "fig5_calibration_ece"):
        assert (figs / f"{fn}.png").exists() and (figs / f"{fn}.svg").exists()
    done = json.loads((run["out"] / "runs" / "fold0__gps__full__none__s0" / "done.json").read_text())
    assert done["device"] == "cpu" and done["hparams"]["h"] == 16 and done["blocks"] == ["ops", "jcr"]


def test_rerun_from_saved_settings_file_is_byte_identical(run, tmp_path):
    from spacr.cli import load_settings_file

    saved = run["out"] / "settings" / "gene_graph_settings.csv"
    s = load_settings_file(str(saved))
    assert s["gene_graph_arms"] == run["settings"]["gene_graph_arms"]
    s["gene_graph_src"] = str(tmp_path / "again")
    s["gene_graph_figures"] = False
    gg.run_gene_graph(s)
    a, b = run["out"] / "tables", tmp_path / "again" / "gene_graph" / "tables"
    names = sorted(p.name for p in a.glob("*.csv") if not p.name.startswith(("fig", "run_configs")))
    assert len(names) >= 12
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    for d in sorted((run["out"] / "runs").iterdir()):
        za, zb = np.load(d / "scores.npz"), np.load(tmp_path / "again" / "gene_graph" / "runs" / d.name / "scores.npz")
        assert all(np.array_equal(za[k], zb[k]) for k in za.files), d.name


def test_default_configs_and_evaluation_on_stored_runs(run):
    o = gg.options({**run["settings"], "gene_graph_configs": [], "gene_graph_tasks": ["time", "fold0"],
                    "gene_graph_seeds": [0, 1]})
    names = gg.default_configs(o)
    assert "time__adamicadar__graph__none__s0" in names and "fold0__adamicadar__graph__none__s0" not in names
    assert "fold0__lr__spectral__none__s1" in names and "fold0__cosine__graph__none__s0" not in names
    assert "fold0__gps__full__rewire__s1" in names and "fold0__gps__noJUMP__shufY__s0" in names
    assert "fold0__cosine__noJUMP__shufX__s0" in names and "fold0__cosine__noJUMP__rewire__s0" not in names
    assert len(names) == len(set(names))
    s = dict(run["settings"])
    s.update({"gene_graph_src": "", "gene_graph_data_dir": str(run["out"] / "data"),
              "gene_graph_runs_dir": str(run["out"] / "runs")})
    p = gg.paths(gg.options(s))
    assert p["root"] == run["out"] and p["data"] == run["out"] / "data"
    ev = gg.evaluate_gene_graph(s)
    ref = pd.read_csv(run["out"] / "tables" / "heldout_metrics_per_fold.csv")
    assert np.allclose(ev["heldout_metrics_per_fold"].select_dtypes("number").to_numpy(float),
                       ref.select_dtypes("number").to_numpy(float), equal_nan=True)


def test_helpers_admit_log_device_flags_and_cli(run, tmp_path, monkeypatch):
    seq = iter([18.0, 10.0, 10.0, 10.0])
    waits = []
    u = gg.admit(3.5, 19.0, 20, poll_s=10, used=lambda: next(seq), sleep=waits.append)
    assert u == 10.0 and len(waits) == 3
    log = tmp_path / "gpu" / "log"
    gg.gpu_log(log, "lab", "START x")
    assert "[lab] START x" in log.read_text()
    assert gg._device({"device": "cuda"}).type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert gg._device({"device": "cuda"}).type == "cpu"
    assert gg.specificity_flag(None, "G00") == "not_in_table"
    tab = pd.DataFrame({"gene": ["A"], "flag_multi_copy": [True], "flag_near_paralog": [True],
                        "flag_adjacent_stain_target": [False]}).set_index("gene")
    assert gg.specificity_flag(tab, "A&B") == "multi_copy+near_paralog" and gg.specificity_flag(tab, "Z") == "not_in_table"
    assert gg.ece(np.array([0.1, 0.9]), np.array([0, 1]), bins=2) == pytest.approx(0.1)
    assert np.isnan(gg.wap(np.arange(3), np.zeros(3)))
    assert gg.jump_support_cat(np.ones((3, 2)), np.array([False, True, True]), 0, [1, 2], np.random.default_rng(0))[2] == 0
    assert np.isnan(gg.jump_support_pair(np.ones((3, 2)), np.array([True, False, True]), 0, 1)[0])
    with pytest.raises(KeyError):
        gg.run_baseline(gg.GeneGraphData(run["out"] / "data"), "time__knn__graph__none__s0",
                        gg.options(run["settings"]), tmp_path / "r")
    T = gg.drivers(pd.DataFrame({"p_calibrated": [0.8, 0.4], "p_full": [0.9, 0.4], "p_noJUMP": [0.8, 0.4],
                                 "p_noOPS": [0.1, 0.4], "p_graph": [0.1, 0.3]}))
    assert list(T.drivers) == ["OPS+JUMP", "OPS+graph"]
    s = dict(run["settings"])
    s.update({"gene_graph_src": str(tmp_path / "cli"), "gene_graph_configs": ["fold0__cosine__noJUMP__none__s0"],
              "gene_graph_figures": False})
    from spacr.cli import write_settings_file

    csv = write_settings_file(s, tmp_path / "cli.csv")[0]
    assert gg.main(["--settings", csv]) == 0
    assert (tmp_path / "cli" / "gene_graph" / "tables" / "heldout_metrics_per_fold.csv").exists()
