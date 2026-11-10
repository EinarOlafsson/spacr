"""Item 889: gene classes, gene-grouped folds, the per-class baseline with decoy heads, and its evaluation.

Tiny synthetic annotation files on the CPU. The plate A identity checks
(class tables, folds and the logistic-regression runs of record) ran against
the reference outputs; see the item record.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spacr import gene_classes as gc

N = 120


def _sources(tmp_path, seed=0):
    """Screen, kinase, CORUM, KEGG, GO-slim and design files for 120 genes."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(N)]
    p1 = np.where(np.arange(N) < 40, 1e-6, rng.uniform(0.2, 1, N))
    p2 = np.where(np.arange(N) < 30, 1e-6, rng.uniform(0.2, 1, N))
    p2[5] = -1.0
    screen = pd.DataFrame({"gene_symbol": genes, "gene_id": np.arange(1000, 1000 + N),
                           "uniprot_accession": [f"['P{i:05d}']" if i % 2 == 0 else "" for i in range(N)],
                           "area_pvalue": p1, "area_score": np.where(np.arange(N) % 2, 1.0, -1.0),
                           "dapi_pvalue": p2, "dapi_score": 1.0,
                           "interphase_cluster": np.arange(N) % 7, "mitotic_cluster": np.arange(N) % 5})
    screen.to_csv(tmp_path / "screen.csv", index=False)
    kin = pd.DataFrame({"xName": ["K1", "K2", "K3", "K4", "K5", "K6", "K7"],
                        "Manning Name": ["a"] * 7, "HGNC Name": ["G100", "G101", "G102", "G103", "G104", "nope", "G106"],
                        "Kinase Name": ["k"] * 7, "Group": ["TK"] * 7, "Family": ["Abl"] * 6 + [np.nan],
                        "SubFamily": [np.nan] * 7, "UniprotID": ["P00100", "x", "P00102", "y", "P00104", "z", "w"]})
    kin.to_csv(tmp_path / "kinhub.csv", index=False)
    corum = pd.DataFrame({"group": ["cA"] * 8 + ["cB"] * 8 + ["cSmall"] * 4,
                          "gene_name": genes[0:8] + genes[1:9] + genes[50:54]})
    corum.to_csv(tmp_path / "corum.tsv", sep="\t", index=False)
    link = pd.DataFrame({"g": [f"hsa:{1000 + i}" for i in range(20, 32)] + [f"hsa:{1000 + i}" for i in range(60, 70)]
                         + ["hsa:1000", "hsa:1001", "hsa:1002", "hsa:1003", "hsa:1004", "hsa:1005"],
                         "p": ["path:hsa00010"] * 12 + ["path:hsa05200"] * 10 + ["path:hsa01100"] * 6})
    link.to_csv(tmp_path / "link.tsv", sep="\t", index=False, header=False)
    names = pd.DataFrame({"p": ["hsa00010", "hsa05200", "hsa01100"],
                          "name": ["Glycolysis - Homo sapiens (human)", "Cancer - Homo sapiens (human)", "Metabolic"]})
    names.to_csv(tmp_path / "list.tsv", sep="\t", index=False, header=False)
    (tmp_path / "br08901.keg").write_text("AMetabolism\nBCarbohydrate\nC    00010  Glycolysis\n"
                                          "AHuman Diseases\nBCancer\nC    05200  Cancer\n")
    go = pd.DataFrame({"ens": ["E"] * 14, "name": genes[70:76] + ["nobody"] + genes[80:86] + ["G090"],
                       "entrez": [str(1070 + i) + ".0" for i in range(6)] + ["9"] + [""] * 7,
                       "go": ["GO:0000001"] * 7 + ["GO:0000002"] * 6 + [np.nan],
                       "desc": ["process one"] * 7 + ["component two"] * 6 + ["none"]})
    go.to_csv(tmp_path / "goslim.tsv", sep="\t", index=False)
    (tmp_path / "goslim.obo").write_text("[Term]\nid: GO:0000001\nnamespace: biological_process\n\n"
                                         "[Term]\nid: GO:0000002\nnamespace: cellular_component\n")
    design = pd.DataFrame({"gene_symbol": ["nontargeting"] * 24 + genes[:4],
                           "sgRNA": [f"nt{i:02d}" for i in range(24)] + ["a", "b", "c", "d"]})
    design.to_csv(tmp_path / "design.csv", index=False)
    return {"screen": str(tmp_path / "screen.csv"), "kinases": str(tmp_path / "kinhub.csv"),
            "corum": str(tmp_path / "corum.tsv"), "kegg_link": str(tmp_path / "link.tsv"),
            "kegg_list": str(tmp_path / "list.tsv"), "kegg_hierarchy": str(tmp_path / "br08901.keg"),
            "goslim_annotations": str(tmp_path / "goslim.tsv"), "goslim_obo": str(tmp_path / "goslim.obo"),
            "design": str(tmp_path / "design.csv")}


def test_bh_matches_statsmodels_and_keeps_missing_values():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.01, 0.04, np.nan, 0.03, 0.5, 0.002])
    q = gc.bh(p)
    finite = np.isfinite(p)
    assert np.isnan(q[2])
    assert np.allclose(q[finite], multipletests(p[finite], method="fdr_bh")[1])
    assert np.isnan(gc.bh([np.nan, np.nan])).all()


def test_grouped_multilabel_folds_put_each_gene_in_one_fold_and_spread_every_class():
    rng = np.random.default_rng(0)
    Y = (rng.random((300, 12)) < 0.08).astype(np.int8)
    Y[:10] = 0
    fold = gc.grouped_multilabel_folds(Y, k=5, seed=0)
    assert fold.shape == (300,) and set(fold) == {0, 1, 2, 3, 4}
    assert (np.bincount(fold) >= 50).all()
    for j in range(Y.shape[1]):
        per_fold = np.bincount(fold[Y[:, j] > 0], minlength=5)
        if Y[:, j].sum() >= 10:
            assert per_fold.min() >= 1 and per_fold.max() - per_fold.min() <= 2
    assert np.array_equal(fold, gc.grouped_multilabel_folds(Y, k=5, seed=0))
    assert not np.array_equal(fold, gc.grouped_multilabel_folds(Y, k=5, seed=1))


def test_build_gene_classes_merges_filters_and_folds(tmp_path):
    sources = _sources(tmp_path)
    out = tmp_path / "tables"
    summary = gc.build_gene_classes(sources, {"train_folds": 3}, str(out))
    table = pd.read_csv(out / "class_table.csv")
    assert summary["kept"] == len(table) and summary["kegg_filter"] and summary["merge_mode"] == "merge (union)"
    names = set(table.name)
    assert "cA" in names and "cB" not in names
    merged = pd.read_csv(out / "merge_table.csv")
    absorbed = merged.set_index("absorbed_name")
    assert absorbed.at["cB", "into_name"] == "cA" and absorbed.at["cB", "into_class_id"].startswith("C")
    assert "cSmall" not in names
    dropped = pd.read_csv(out / "kegg_dropped_pathways.csv")
    assert list(dropped.pathway) == ["hsa05200 Cancer"] and dropped.brite_A.iloc[0] == "Human Diseases"
    assert "hsa00010 Glycolysis" in names and not any("hsa01100" in n for n in names)
    assert "GO:0000001 process one" in names and not any("GO:0000002" in n for n in names)
    assert "family TK/Abl" in names
    assert {"all kinases", "group TK"} <= set(merged.absorbed_name)
    kin = pd.read_csv(out / "kinase_screened.csv")
    assert set(kin.gene) == {"G100", "G101", "G102", "G103", "G104", "G106"}
    assert {"high area", "low area", "high dapi"} <= names
    members = pd.read_csv(out / "class_members.csv")
    assert members.trainable.all()
    folds = pd.read_csv(out / "gene_folds.csv")
    assert folds.gene.is_unique and set(folds.fold) == {0, 1, 2} and len(folds) == N
    nt = pd.read_csv(out / "nt_pseudogenes.csv")
    assert len(nt) == 24 and nt.groupby("pseudogene").size().eq(4).all()
    assert nt.groupby("pseudogene").fold.nunique().eq(1).all()
    record = json.loads((out / "sources.json").read_text())
    assert set(record) == set(gc.SOURCE_KEYS) and len(record["screen"]["md5"]) == 32
    universe = pd.read_csv(out / "gene_universe.csv")
    assert universe.phenotype_funk.sum() == 29 and not universe.phenotype_funk.iloc[5]
    assert pd.read_csv(out / "classes_per_gene.csv").columns.tolist() == ["gene", "n_classes"]
    gated = gc.build_gene_classes(sources, {"train_folds": 3, "gene_class_gating": True, "gene_class_merge": False,
                                            "gene_class_merge_jaccard": 0.7, "gene_class_kegg_filter": False},
                                  str(tmp_path / "v1"))
    assert gated["phenotype_rule"] == "phenotype_funk" and gated["merge_mode"] == "drop"
    v1 = pd.read_csv(tmp_path / "v1" / "class_table_all_candidates.csv")
    assert v1.status.str.startswith("duplicate_of").any() and not (tmp_path / "v1" / "merge_table.csv").exists()
    with pytest.raises(KeyError, match="design"):
        gc.build_gene_classes({k: v for k, v in sources.items() if k != "design"}, {}, str(tmp_path / "x"))


def test_gate_gene_classes_keeps_phenotype_positive_members(tmp_path):
    sources = _sources(tmp_path)
    gc.build_gene_classes(sources, {"train_folds": 3}, str(tmp_path / "v3"))
    summary = gc.gate_gene_classes(str(tmp_path / "v3"), sources["screen"], str(tmp_path / "gated"))
    members = pd.read_csv(tmp_path / "gated" / "class_members.csv")
    table = pd.read_csv(tmp_path / "gated" / "class_table.csv")
    assert summary["kept"] == len(table) and (table.n_trainable >= 5).all()
    assert set(members[members.trainable].gene) <= {f"G{i:03d}" for i in range(30)}
    assert (tmp_path / "gated" / "gene_folds.csv").read_bytes() == (tmp_path / "v3" / "gene_folds.csv").read_bytes()


def _class_run(tmp_path, signal=True, seed=0):
    """Class tables for 400 genes in 6 classes, and a feature table with or without the class signal."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(400)]
    tables = tmp_path / "tables"
    tables.mkdir()
    classes = [f"C{j:03d}" for j in range(6)]
    Y = np.zeros((400, 6), np.int8)
    for j in range(6):
        Y[j * 30:(j + 1) * 30, j] = 1
    fold = gc.grouped_multilabel_folds(Y, 5, 0)
    pd.DataFrame({"class_id": classes, "source": ["corum", "corum", "kegg", "kegg", "goslim", "funk"],
                  "name": [f"class {j}" for j in range(6)]}).to_csv(tables / "class_table.csv", index=False)
    rows = [(classes[j], genes[i], True) for j in range(6) for i in np.flatnonzero(Y[:, j])]
    rows.append((classes[0], genes[399], False))
    pd.DataFrame(rows, columns=["class_id", "gene", "trainable"]).to_csv(tables / "class_members.csv", index=False)
    pd.DataFrame({"gene": genes, "fold": fold}).to_csv(tables / "gene_folds.csv", index=False)
    nt = pd.DataFrame({"sgRNA": [f"nt{i}" for i in range(20)], "pseudogene": [f"NT_{i // 4:03d}" for i in range(20)],
                       "fold": [i // 4 for i in range(20)]})
    nt.to_csv(tables / "nt_pseudogenes.csv", index=False)
    pd.DataFrame({"gene": genes, "phenotype_funk": np.arange(400) % 3 == 0}).to_csv(tables / "gene_universe.csv",
                                                                                   index=False)
    X = rng.normal(0, 1, (405, 8))
    if signal:
        for j in range(6):
            X[j * 30:(j + 1) * 30, j] += 3.0
    index = genes + [f"NT_{i:03d}" for i in range(5)]
    features = pd.DataFrame(X, columns=[f"f{k}_score" for k in range(8)])
    features.insert(0, "gene_symbol", index)
    features["note"] = "x"
    features.to_csv(tmp_path / "features.csv", index=False)
    return tables


def test_the_baseline_learns_planted_classes_and_the_controls_do_not(tmp_path):
    tables = _class_run(tmp_path)
    settings = {"train_gene_classes": str(tables), "train_features": str(tmp_path / "features.csv"),
                "train_feature_suffix": "_score", "train_exclude_sources": "funk", "train_decoy_heads": 120,
                "gene_class_eval_boot": 50, "gene_class_dst": str(tmp_path / "run")}
    lines = []
    out = gc.run_gene_classes(settings, report=lines.append)
    summary = out["summary"]
    assert summary["classes_scored"] == 5 and summary["median_auroc"] > 0.9
    assert summary["decoy_median_auroc"] < summary["median_auroc"] - 0.2
    assert summary["discovery_null"] == "gene-wise (>= 100 matched decoys)"
    per = pd.read_csv(tmp_path / "run" / "eval" / "per_class.csv")
    assert "C005" not in set(per.class_id) and per.class_id.str.startswith("DECOY").sum() == 120
    assert (per[~per.class_id.str.startswith("DECOY")].auroc_lo > 0.5).all()
    for name in ("discovery.csv", "discovery_a.csv", "discovery_b.csv", "summary.json", "gene_class_scores.parquet"):
        assert (tmp_path / "run" / "eval" / name).is_file()
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["n_features"] == 8 and len(config["classes"]) == 125
    fold0 = pd.read_parquet(tmp_path / "run" / "fold_00" / "gene_scores.parquet")
    assert (fold0.n_cells == 999).all()
    assert any(line.startswith("fold 0") for line in lines)
    shuffled = gc.run_gene_classes({**settings, "train_shuffle_control": True, "train_decoy_heads": 0,
                                    "gene_class_dst": str(tmp_path / "shuffled")})["summary"]
    assert shuffled["median_auroc"] == pytest.approx(0.5, abs=0.12)
    assert shuffled["discovery_null"].startswith("pooled") and shuffled["decoy_median_auroc"] is None


def test_the_decoy_null_makes_no_calls_on_random_features(tmp_path):
    tables = _class_run(tmp_path, signal=False)
    settings = {"train_gene_classes": str(tables), "train_features": str(tmp_path / "features.csv"),
                "train_decoy_heads": 150, "gene_class_eval_boot": 20, "gene_class_dst": str(tmp_path / "run")}
    summary = gc.run_gene_classes(settings)["summary"]
    assert summary["median_auroc"] == pytest.approx(0.5, abs=0.12)
    assert summary["discovery_a_member_new_phenotype_pairs"] == 0
    assert summary["discovery_b_new_function_pairs"] == 0
    assert summary["decoy_frac_auroc_lo_gt_0.5"] <= 0.1


def test_the_folds_never_share_a_gene(tmp_path):
    tables = _class_run(tmp_path)
    gc.train_gene_class_baseline(gc._feature_table(tmp_path / "features.csv", ""), str(tables), str(tmp_path / "r"))
    seen = [set(pd.read_parquet(tmp_path / "r" / f"fold_{k:02d}" / "gene_scores.parquet").index) for k in range(5)]
    for a in range(5):
        for b in range(a + 1, 5):
            assert not seen[a] & seen[b]
    assert set().union(*seen) == set(gc._feature_table(tmp_path / "features.csv", "").index)


def test_run_gene_classes_refuses_what_it_cannot_run(tmp_path):
    with pytest.raises(ValueError, match="train_label_source"):
        gc.run_gene_classes({"train_label_source": "annotation"})
    with pytest.raises(ValueError, match="grouped_multilabel"):
        gc.run_gene_classes({"train_split": "random"})
    with pytest.raises(ValueError, match="logistic_regression"):
        gc.run_gene_classes({"ml_model": "resnet18"})
    with pytest.raises(ValueError, match="train_gene_classes"):
        gc.run_gene_classes({})
    with pytest.raises(ValueError, match="train_features"):
        gc.run_gene_classes({"train_gene_classes": str(tmp_path)})


def test_run_gene_classes_builds_and_gates_from_settings(tmp_path):
    sources = _sources(tmp_path)
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    lines = []
    with pytest.raises(ValueError, match="train_features"):
        gc.run_gene_classes({"gene_class_sources": str(tmp_path / "sources.json"), "train_folds": 3,
                             "train_gene_classes": str(tmp_path / "built")}, report=lines.append)
    assert (tmp_path / "built" / "class_table.csv").is_file() and lines[0].startswith("gene classes:")
    with pytest.raises(ValueError, match="train_features"):
        gc.run_gene_classes({"gene_class_sources": str(tmp_path / "sources.json"),
                             "gene_class_gated_from": str(tmp_path / "built"),
                             "train_gene_classes": str(tmp_path / "gated")})
    assert json.loads((tmp_path / "gated" / "class_summary.json").read_text())["arm"] == "gated (supplement)"


def test_the_defaults_are_registered():
    from spacr import settings as S

    defaults = gc.gene_class_defaults({"train_folds": 3})
    assert defaults["train_folds"] == 3 and defaults["train_decoy_heads"] == 300
    assert S.has_registered_defaults(gc.APP_KEY)
    assert set(S.ALPHA_FEATURES[889]["module_settings"][gc.APP_KEY]) == set(gc.gene_class_defaults())


class _ReferenceFixedCNN(nn.Module):
    """The benchmark's fixed spaCR CNN arm (spacr_cnn_fixed.py), layer for layer."""

    def __init__(self, in_channels, num_classes):
        """Build the four layers."""
        super().__init__()
        self.fusion = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        self.dilated_conv1 = nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1)
        self.spatial_attention = nn.Conv2d(64, 64, kernel_size=1)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        """Raw logits."""
        x = self.fusion(x)
        x = self.spatial_attention(F.relu(self.dilated_conv1(x), inplace=True))
        return self.fc1(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))


def test_the_spacr_cnn_is_the_benchmark_fixed_arm():
    from spacr.utils import CustomCellClassifier

    torch.manual_seed(0)
    ours = CustomCellClassifier(num_classes=5, in_channels=4).eval()
    ref = _ReferenceFixedCNN(4, 5).eval()
    mapping = {"early_fusion.conv1": "fusion", "multi_scale_block_1.dilated_conv1": "dilated_conv1",
               "multi_scale_block_1.spatial_attention": "spatial_attention", "fc1": "fc1"}
    state = {}
    for key, value in ours.state_dict().items():
        prefix, leaf = key.rsplit(".", 1)
        state[f"{mapping[prefix]}.{leaf}"] = value
    ref.load_state_dict(state)
    x = torch.rand(3, 4, 24, 24)
    with torch.no_grad():
        a, b = ours(x), ref(x)
    assert torch.equal(a, b)
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in ref.parameters())
