"""Item 886: multimodal gene-function prediction (spacr.gene_function) on the synthetic gene graph, CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spacr.gene_function as gf
import spacr.gene_graph as gg
from tests.gene_graph_synthetic import write_knowledge, write_synthetic

FOLDS = [f"fold{k}" for k in range(5)]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("gf")
    s = write_synthetic(root)
    s.update({"gene_graph_max_epochs": 40, "gene_graph_min_epochs": 20, "gene_graph_sensitivity_blocks": {},
              "gene_graph_configs": [f"{t}__{m}__{a}__{c}__s0" for t in FOLDS for m, a, c in (
                  ("gps", "noJUMP", "none"), ("gps", "graph", "none"), ("gps", "full", "shufX"),
                  ("cosine", "noJUMP", "none"), ("cosine", "noJUMP", "shufX"))]})
    gg.build_gene_graph(s)
    write_knowledge(Path(s["gene_graph_src"]) / "gene_graph" / "data", root / "know")
    gg.run_gene_graph({**s, "gene_graph_data_dir": str(Path(s["gene_graph_src"]) / "gene_graph" / "data")})
    fs = {"gene_function_src": str(root / "fn"), "gene_function_gene_graph_src": s["gene_graph_src"],
          "gene_function_knowledge_src": str(root / "know"),
          "gene_function_gps_runs": ["gps:f", "gps:c", "gps:ep"], "gene_function_gps_controls": ["c:shufOPS", "ep:shufScreen"],
          "gene_function_late_pairs": {"late_opsf": ["gene_graph:gps__noJUMP__none", "self:gps__f__none"],
                                       "late_opsf_kNNe": ["self:gps__f__none", "self:cosine__k_e__none"]},
          "gene_function_zscore_null": 50,
          "gene_function_uniprot_function_cache": str(root / "cache.tsv"),
          "gene_function_specificity_table": str(root / "specificity.csv")}
    pd.DataFrame({"gene": ["G50", "G51"], "function": ["Part of class 2 machinery.", "Unknown."]}).to_csv(
        root / "cache.tsv", sep="\t", index=False)
    (root / "entry.list").write_text("ENTRY_AC\tENTRY_TYPE\tENTRY_NAME\nPF00001\tFamily\tKinase\n")
    fs["gene_function_interpro_entry_list"] = str(root / "entry.list")
    res = gf.run_gene_function(fs)
    return {"root": root, "gs": s, "fs": fs, "res": res, "out": root / "fn" / "gene_function"}


def test_defaults_and_folders():
    d = gf.get_gene_function_default_settings()
    assert len(d["gene_function_stack_arms"]) == 22 and d["gene_function_seed"] == 20261109
    with pytest.raises(ValueError):
        gf.folders(gf.options({}))
    assert gf.arm_complexity("late", "x", gf.options({})) == 4.5
    assert gf.arm_complexity("cosine", "k_e", gf.options({})) == 5.5


def test_stack_with_one_block_equals_the_gene_graph_cosine_baseline(world):
    a = np.load(world["out"] / "runs" / "fold0__cosine__k_ops__none__s0" / "scores.npz")
    b = np.load(Path(world["gs"]["gene_graph_src"]) / "gene_graph" / "runs" / "fold0__cosine__noJUMP__none__s0" / "scores.npz")
    for k in ("rows", "S", "C"):
        assert np.array_equal(a[k], b[k])


def test_late_fusion_of_a_run_with_itself_keeps_its_ranking():
    rng = np.random.default_rng(0)
    z = {"rows": np.arange(4), "S": rng.normal(size=(4, 9)) * 3 + 2, "C": rng.normal(size=(4, 5))}
    f = gf.late_fusion([z, z])
    assert np.allclose(f["S"], gf.zscore_matrix(z["S"]), atol=1e-6)
    assert np.array_equal(np.argsort(-f["S"].ravel(), kind="stable"), np.argsort(-z["S"].ravel(), kind="stable"))
    other = {"rows": np.arange(4), "S": z["S"] * 100, "C": z["C"]}
    assert np.allclose(gf.late_fusion([z, other])["S"], gf.zscore_matrix(z["S"]), atol=1e-5)
    with pytest.raises(ValueError):
        gf.late_fusion([z, {**z, "rows": np.arange(1, 5)}])


def test_screen_controls_permute_only_the_screen_blocks(world):
    o = gf.options(world["fs"])
    f = gf.folders(o)
    D = gf.function_data(f)
    go = gf.trainer_options(o, f)
    base = gf.prepare_run(D, "fold1__cosine__k_ep__none__s0", go, o)
    so = gf.prepare_run(D, "fold1__cosine__k_ep__shufOPS__s0", go, o)
    ss = gf.prepare_run(D, "fold1__cosine__k_ep__shufScreen__s0", go, o)
    for b in ("esm", "dom", "txt", "s3di", "struct_spec"):
        assert np.array_equal(base["X"][b], so["X"][b]) and np.array_equal(base["X"][b], ss["X"][b])
    assert np.array_equal(base["X"]["img_dino"], so["X"]["img_dino"])
    assert not np.array_equal(base["X"]["ops"], so["X"]["ops"])
    assert np.array_equal(base["X"]["txt"], D.z.get("X_txt_f1", D.extra[0]["X_txt_f1"]))
    real = D.real
    perm = [int(np.flatnonzero((base["X"]["ops"][real] == r).all(1))[0]) for r in ss["X"]["ops"][real]]
    for b in ("img_dino", "img_rn50"):
        assert np.array_equal(ss["X"][b][real], base["X"][b][real][perm])
    assert np.array_equal(so["X"]["ops"], ss["X"]["ops"])


def test_paralog_excluded_metrics(world):
    D = gf.function_data(gf.folders(gf.options(world["fs"])))
    R = gg.Reference(D)
    z = np.load(world["out"] / "runs" / "fold2__cosine__k_c__none__s0" / "scores.npz")
    plain = gg.fold_metrics(R, z, "x")
    none = gf.fold_metrics_paralog(R, z, np.zeros((D.N, D.N), bool))
    assert np.isclose(none["link_AUPRC_2022_noparalog"], plain["link_AUPRC_2022"])
    assert np.isclose(none["cat_micro_AUPRC_noparalog"], plain["cat_micro_AUPRC"])
    assert none["link_pairs_dropped_paralog_frac"] == 0 and none["cat_cells_dropped_paralog_frac"] == 0
    P = np.zeros((D.N, D.N), bool)
    rows = z["rows"][R.real[z["rows"]]]
    test = np.zeros(D.N, bool)
    test[z["rows"]] = True
    cols = np.flatnonzero(R.real & ~test)
    P[rows[0], cols[:4]] = P[cols[:4], rows[0]] = True
    some = gf.fold_metrics_paralog(R, z, P)
    assert np.isclose(some["link_pairs_dropped_paralog_frac"], 4 / (len(rows) * len(cols)))
    Ytr = R.Y & ~test[:, None] & R.real[:, None]
    bad = (P[rows].astype(float) @ Ytr.astype(float)) > 0
    has = R.Y[rows].sum(1) > 0
    assert np.isclose(some["cat_cells_dropped_paralog_frac"], bad[has].mean())


def test_winner_rule_matches_a_hand_computed_choice():
    rows = []

    def add(src, model, arm, values, control="none"):
        """Five folds of one run."""
        for k, v in enumerate(values):
            rows.append({"src": src, "model": model, "arm": arm, "control": control, "task": f"fold{k}",
                         "link_AUPRC_2022": v, "cat_micro_AUPRC": v})

    add("gene_graph", "gps", "noJUMP", [0.10] * 5)
    add("gene_graph", "cosine", "noJUMP", [0.10] * 5)
    add("mm", "gps", "f", [0.12] * 5)
    add("mm", "gps", "c", [0.21, 0.21, 0.21, 0.19, 0.19])
    add("mm", "gps", "c", [0.05] * 5, "shufOPS")
    add("mm", "gps", "ep", [0.204, 0.204, 0.204, 0.204, 0.204])
    add("mm", "gps", "ep", [0.05] * 5, "shufScreen")
    add("mm", "gps", "ep_noOPS", [0.11] * 5)
    add("mm", "gps", "e", [0.30] * 5)
    add("mm", "gps", "e", [0.31] * 5, "shufScreen")
    o = gf.options({"gene_function_relations": {"gps:c": ["f", None, "shufOPS"], "gps:ep": ["ep_noOPS", "f", "shufScreen"],
                                                "gps:e": [None, "f", "shufScreen"]}})
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        res = gf.results_and_winners(pd.DataFrame(rows), pd.DataFrame(), o, Path(d))
        Q = pd.read_csv(Path(d) / "winner_candidates_link_AUPRC_2022.csv")
    assert res["winners"]["link_AUPRC_2022"]["arm"] == "c"
    q = Q.set_index("arm")
    assert not q.loc["e", "qualifies"] and q.loc["e", "beats_reference"] and not q.loc["e", "screen_dependent"]
    assert q.loc["ep", "qualifies"] and q.loc["c", "qualifies"] and q.loc["c", "blocks"] < q.loc["ep", "blocks"]
    o2 = {**o, "winner_simpler_within": 0.001}
    with tempfile.TemporaryDirectory() as d:
        assert gf.results_and_winners(pd.DataFrame(rows), pd.DataFrame(), o2, Path(d))["winners"]["link_AUPRC_2022"]["arm"] == "ep"
    o3 = {**o, "winner_min_folds": 6}
    with tempfile.TemporaryDirectory() as d:
        assert gf.results_and_winners(pd.DataFrame(rows), pd.DataFrame(), o3, Path(d))["winners"]["cat_micro_AUPRC"]["model"] == "reference"


def test_explanation_uses_training_members_and_never_the_gene_itself(world):
    import scipy.sparse as sp

    o = gf.options(world["fs"])
    f = gf.folders(o)
    D = gf.function_data(f)
    R = gg.Reference(D)
    g, c = 3, 0
    mem = np.flatnonzero(R.Y[:, c] & (R.fold != R.fold[g]) & R.real)
    assert g not in mem and not (R.fold[mem] == R.fold[g]).any()
    blocks = {b: gf._block(D, b) for b in ("ops", "img_dino", "esm")}
    B = sp.load_npz(f["knowledge"] / "data" / "dom_membership_current.npz").tocsr()
    vocab = json.loads((f["knowledge"] / "data" / "dom_vocab_current.json").read_text())
    args = (blocks, B, vocab, {}, set(map(tuple, D.extra_edges)), R.nodes, R)
    a = gf.explain_call(g, mem, *args, np.random.default_rng(0), 30)
    R.Y[g] = ~R.Y[g]
    R.train[g] = ~R.train[g]
    R.cur[g] = ~R.cur[g]
    b = gf.explain_call(g, mem, *args, np.random.default_rng(0), 30)
    assert a == b and a["explanation"].endswith("own UniProt text hidden (held-out)")


def test_planted_screen_module_beats_knowledge_and_control_and_knowledge_module_is_not_screen_specific(world):
    F = pd.read_csv(world["out"] / "tables" / "heldout_metrics_per_fold.csv")
    mm = F[F.src == "mm"].set_index(["model", "arm", "control", "task"]).cat_micro_AUPRC
    for t in FOLDS:
        assert mm[("cosine", "k_c", "none", t)] > mm[("cosine", "k_f", "none", t)]
        assert mm[("cosine", "k_c", "none", t)] > mm[("cosine", "k_c", "shufOPS", t)]
    H = pd.read_csv(world["out"] / "tables" / "hypotheses_ranked.csv")
    assert {"gene", "rank_in_gene", "class_id", "p_calibrated", "q_NT", "p_knowledge_only", "p_screen_only",
            "p_winner_shuffled", "delta_screen", "OPS_z", "ESM_nearest_member", "shared_domains", "struct_hit_members",
            "explanation", "top_new_partners_link_winner", "posthoc_uniprot_mentions_class", "drivers",
            "screen_specific", "call_type", "gene_graph_q_NT", "spec_flag"} <= set(H.columns)
    assert (H.rank_in_gene <= 4).all()
    crafted = pd.DataFrame({"gene": ["G31", "G01"], "rank_in_gene": [1, 1], "class_id": ["C2", "C0"],
                            "class": ["c2", "c0"], "p_calibrated": [0.9, 0.8], "p_knowledge_only": [0.88, 0.1],
                            "p_screen_only": [0.2, 0.7], "p_winner_shuffled": [0.87, 0.1], "delta_screen": [0.02, 0.7],
                            "delta_knowledge": [0.7, 0.1], "delta_vs_shuffled": [0.03, 0.7],
                            "posthoc_uniprot_mentions_class": [False, False], "explanation": ["", ""],
                            "top_new_partners_link_winner": ["", ""]})
    f = gf.folders(gf.options(world["fs"]))
    f = {**f, "tables": world["root"] / "crafted"}
    f["tables"].mkdir(exist_ok=True)
    ranked = gf.rank_hypotheses(crafted, f, gf.options(world["fs"])).set_index("gene")
    assert not ranked.loc["G31", "screen_specific"] and ranked.loc["G31", "call_type"].startswith("knowledge")
    assert ranked.loc["G01", "screen_specific"] and ranked.index[0] == "G01"
    top = (f["tables"] / "hypotheses_top.md").read_text()
    assert top.startswith("rows 2") and "## Screen-specific calls" in top


def test_run_writes_every_table(world):
    t = world["out"] / "tables"
    for name in ("heldout_metrics_per_fold.csv", "time_split_metrics.csv", "results_table.csv", "paired_deltas.csv",
                 "winners.json", "winner_candidates_cat_micro_AUPRC.csv", "winner_candidates_link_AUPRC_2022_noparalog.csv",
                 "calibration_cosine_k_ep.csv", "fdr_summary_cosine_k_ep.json", "hypotheses_cosine_k_ep.csv",
                 "fig1_heldout_link_points.csv", "fig3_screen_contribution.csv"):
        assert (t / name).exists(), name
    F = pd.read_csv(t / "heldout_metrics_per_fold.csv")
    assert {"key", "src", "link_AUPRC_2022_noparalog", "cat_micro_AUPRC_noparalog", "paralog_with_structure"} <= set(F.columns)
    assert set(F.src) == {"mm", "gene_graph"}
    assert (world["out"] / "runs" / "fold0__late__late_opsf__shufScreen__s0" / "done.json").exists()
    D = pd.read_csv(t / "paired_deltas.csv")
    assert {"vs noOPS", "vs shuffled", "vs knowledge-only f", "vs OPS-only GPS (gene_graph)"} <= set(D.comparison)
    w = json.loads((t / "winners.json").read_text())
    assert set(w) == {"link_AUPRC_2022", "cat_micro_AUPRC", "link_AUPRC_2022_noparalog", "cat_micro_AUPRC_noparalog"}
    for fig in ("fig1_heldout_link", "fig2_heldout_category", "fig3_screen_contribution"):
        assert (world["out"] / "figures" / f"{fig}.png").exists()


def test_rerun_from_saved_settings_is_byte_identical(world, tmp_path):
    from spacr.cli import load_settings_file, write_settings_file

    s = load_settings_file(str(world["out"] / "settings" / "gene_function_settings.csv"))
    s["gene_function_src"] = str(tmp_path / "again")
    s["gene_function_figures"] = False
    csv = write_settings_file(s, tmp_path / "s.csv")[0]
    assert gf.main(["--settings", csv]) == 0
    a, b = world["out"] / "tables", tmp_path / "again" / "gene_function" / "tables"
    names = sorted(p.name for p in a.iterdir() if not p.name.startswith("fig"))
    assert len(names) >= 14
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
