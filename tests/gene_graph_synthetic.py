"""A small synthetic screen for the gene graph tests (items 884-886).

60 genes in three planted modules of 15 plus 15 background genes, six
non-targeting pseudo-genes (four guides each), five categories, a screen
block and a JUMP-like block with partial coverage. Module genes share a
module-specific profile; the 2022 edges and the categories follow the
modules, and the "new" edges are further module pairs.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

N_GENES = 60
MODULES = (range(0, 15), range(15, 30), range(30, 45))


def genes():
    return [f"G{i:02d}" for i in range(N_GENES)]


def module_of(i):
    for m, r in enumerate(MODULES):
        if i in r:
            return m
    return -1


def write_synthetic(root: Path, seed: int = 0, noise: float = 0.35) -> Dict[str, Any]:
    """Write every input table under ``root`` and return the gene graph settings."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    g = genes()
    centres = rng.normal(size=(4, 8))
    centres[3] = 0
    X = np.array([centres[module_of(i)] for i in range(N_GENES)]) + noise * rng.normal(size=(N_GENES, 8))
    cols = [f"w_{k}" for k in range(8)]
    gene_tab = pd.DataFrame(X, columns=cols)
    gene_tab.insert(0, "gene", g)
    gene_tab.insert(1, "n_cells", rng.integers(30, 300, N_GENES))
    gene_tab.loc[len(gene_tab)] = ["G_low", 5] + list(rng.normal(size=8))
    gene_tab.to_parquet(root / "genes.parquet")
    guides = []
    for i, name in enumerate(g):
        for k in range(2):
            guides.append([f"{name}_sg{k}", name, int(rng.integers(10, 150))]
                          + list(X[i] + 0.1 * rng.normal(size=8)))
    nt = []
    for k in range(24):
        guides.append([f"nontargeting_{k:03d}", f"nontargeting_{k:03d}", int(rng.integers(10, 150))]
                      + list(noise * rng.normal(size=8)))
        nt.append({"pseudogene": f"ntpseudo_{k // 4}", "sgRNA": f"nontargeting_{k:03d}", "fold": (k // 4) % 5})
    nt.append({"pseudogene": "ntpseudo_9", "sgRNA": "nontargeting_missing", "fold": 0})
    pd.DataFrame(guides, columns=["guide", "gene", "n_cells"] + cols).to_parquet(root / "guides.parquet")
    pd.DataFrame(nt).to_csv(root / "nt_pseudogenes.csv", index=False)
    cover = rng.random(N_GENES) < 0.7
    J = np.array([centres[module_of(i)][:6] for i in range(N_GENES)]) + noise * rng.normal(size=(N_GENES, 6))
    pd.DataFrame(J[cover], index=np.array(g)[cover], columns=[f"j{k}" for k in range(6)]).to_parquet(root / "jcr.parquet")
    pd.DataFrame(noise * rng.normal(size=(10, 6)), columns=[f"j{k}" for k in range(6)]).to_parquet(root / "jcr_nt.parquet")
    tvn = pd.DataFrame(X[:, :6] + 0.05 * rng.normal(size=(N_GENES, 6)), columns=[f"t{k}" for k in range(6)])
    tvn.insert(0, "gene", g)
    tvn.insert(1, "n_cells", 50)
    tvn.to_parquet(root / "tvn_genes.parquet")
    tg = pd.DataFrame([[r[0], r[1]] + list(rng.normal(size=6)) for r in guides], columns=["guide", "gene"] + list(tvn.columns[2:]))
    tg.to_parquet(root / "tvn_guides.parquet")

    pairs = [(a, b) for a in range(N_GENES) for b in range(a + 1, N_GENES) if module_of(a) == module_of(b) >= 0]
    draw = rng.random(len(pairs))
    old = [p for p, d in zip(pairs, draw) if d < 0.45]
    new = [p for p, d in zip(pairs, draw) if 0.45 <= d < 0.7]
    background = [(a, b) for a in range(N_GENES) for b in range(a + 1, N_GENES)
                  if module_of(a) != module_of(b) and rng.random() < 0.01]

    def table(ps):
        return pd.DataFrame({"a": [g[a] for a, b in ps], "b": [g[b] for a, b in ps]})

    split = rng.random(len(old))
    table([p for p, s in zip(old, split) if s < 0.5] + background).to_parquet(root / "STRING.parquet")
    table([p for p, s in zip(old, split) if s >= 0.5]).to_parquet(root / "BioGRID.parquet")
    table([p for p in old if module_of(p[0]) == 0][:20]).to_parquet(root / "CORUM.parquet")
    nsplit = rng.random(len(new))
    table([p for p, s in zip(new, nsplit) if s < 0.5]).to_parquet(root / "STRINGnew.parquet")
    table([p for p, s in zip(new, nsplit) if s >= 0.5]).to_parquet(root / "BioGRIDnew.parquet")
    table([]).to_parquet(root / "CORUMnew.parquet")
    table(old + new + background).to_parquet(root / "current.parquet")
    table(old + new + background + [(0, 59)]).to_parquet(root / "string04.parquet")
    classes = pd.DataFrame({"class_id": [f"C{k}" for k in range(5)], "name": [f"class {k}" for k in range(5)],
                            "source": ["CORUM", "GO", "GO", "KEGG", "GO"]})
    classes.to_csv(root / "class_table.csv", index=False)
    members = [(f"C{module_of(i)}", g[i]) for i in range(45)]
    members += [("C3", g[i]) for i in range(0, 8)] + [("C4", g[i]) for i in (50, 51, 52, 53, 3)]
    pd.DataFrame(members, columns=["class_id", "gene"]).to_csv(root / "class_members.csv", index=False)
    pd.DataFrame({"gene": g[2:], "fold": [i % 5 for i in range(2, N_GENES)]}).to_csv(root / "gene_folds.csv", index=False)
    with gzip.open(root / "goa.gaf.gz", "wt") as fh:
        fh.write("!gaf-version: 2.2\n")
        for i in range(0, N_GENES, 2):
            for t in range(i % 7):
                fh.write(f"UniProtKB\tP{i}\t{g[i]}\t\tGO:{1000 + t:07d}\tPMID:1\tIDA\n")
        fh.write(f"UniProtKB\tP0\t{g[0]}\t\tGO:0005515\tPMID:1\tIPI\n")
        fh.write(f"UniProtKB\tP0\t{g[0]}\tNOT\tGO:0009999\tPMID:1\tIDA\n")
        fh.write("short\tline\n")
    pd.DataFrame({"gene": g, "flag_multi_copy": [i % 11 == 0 for i in range(N_GENES)],
                  "flag_near_paralog": [i % 13 == 0 for i in range(N_GENES)],
                  "flag_adjacent_stain_target": False}).to_csv(root / "specificity.csv", index=False)
    return {
        "gene_graph_src": str(root / "out"),
        "gene_graph_ops_gene_table": str(root / "genes.parquet"),
        "gene_graph_ops_guide_table": str(root / "guides.parquet"),
        "gene_graph_nt_grouping": str(root / "nt_pseudogenes.csv"),
        "gene_graph_extra_blocks": {"jcr": str(root / "jcr.parquet")},
        "gene_graph_extra_block_nt": {"jcr": str(root / "jcr_nt.parquet")},
        "gene_graph_extra_block_pca": {"jcr": 4},
        "gene_graph_sensitivity_blocks": {"ops_tvn": [str(root / "tvn_genes.parquet"), str(root / "tvn_guides.parquet")]},
        "gene_graph_sensitivity_dims": 3,
        "gene_graph_train_edges": [str(root / f"{k}.parquet") for k in ("STRING", "BioGRID", "CORUM")],
        "gene_graph_new_edges": [str(root / f"{k}new.parquet") for k in ("STRING", "BioGRID", "CORUM")],
        "gene_graph_current_edges": [str(root / "current.parquet")] * 3,
        "gene_graph_known_extra": [str(root / "string04.parquet")],
        "gene_graph_go_annotations": str(root / "goa.gaf.gz"),
        "gene_graph_class_table": str(root / "class_table.csv"),
        "gene_graph_class_members": str(root / "class_members.csv"),
        "gene_graph_gene_folds": str(root / "gene_folds.csv"),
        "gene_graph_specificity_table": str(root / "specificity.csv"),
        "gene_graph_arms": {"full": ["ops", "jcr"], "noOPS": ["jcr"], "noJUMP": ["ops"], "graph": []},
        "gene_graph_jump_blocks": {"CRISPR": "jcr"},
        "gene_graph_jump_null": 50,
        "gene_graph_device": "cpu",
        "gene_graph_hidden": 16, "gene_graph_layers": 2, "gene_graph_heads": 2,
        "gene_graph_rwse_steps": 4, "gene_graph_max_epochs": 30, "gene_graph_eval_every": 10,
        "gene_graph_min_epochs": 10, "gene_graph_patience": 2, "gene_graph_spectral_dims": 4,
        "gene_graph_bootstrap_reps": 3, "gene_graph_topk": [10, 100],
        "gene_graph_lr_pos_minibatch": 50, "gene_graph_fdr": 1.01,
        "gene_graph_poorly_characterised": {"string04_degree_q": 0.6, "go_terms_q": 0.6},
        "gene_graph_figures": False,
    }


def write_knowledge(gene_graph_data: Path, root: Path, seed: int = 1, knowledge_module: int = 2) -> Path:
    """Knowledge blocks on a synthetic graph's nodes: ESM, domains and text carry one module only.

    The screen block of that module's genes is replaced by noise in graph.npz,
    so the module is knowledge-only and the other two are screen-only.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    nodes = pd.read_parquet(gene_graph_data / "nodes.parquet")
    N = len(nodes)
    mods = np.array([module_of(i) if i < N_GENES else -1 for i in range(N)])
    z = dict(np.load(gene_graph_data / "graph.npz"))
    k = mods == knowledge_module
    z["X_ops"][k, :8] = 0.35 * rng.normal(size=(k.sum(), 8))
    np.savez_compressed(gene_graph_data / "graph.npz", **z)
    sig = np.where(k[:, None], 1.5, 0.0) + 0.3 * rng.normal(size=(N, 6))
    out = root / "gene_knowledge"
    (out / "data").mkdir(parents=True, exist_ok=True)
    blocks = {"esm": sig.astype(np.float32), "dom": (sig[:, :4] + 0.2 * rng.normal(size=(N, 4))).astype(np.float32)}
    for f in range(5):
        h = nodes.fold.to_numpy() != f
        blocks[f"txt_f{f}"] = (np.where(h[:, None], sig[:, :3], 0) + 0.1 * rng.normal(size=(N, 3))).astype(np.float32)
    blocks["txt_time"] = blocks["txt_f0"]
    blocks["img_dino"] = rng.normal(size=(N, 4)).astype(np.float32)
    blocks["img_rn50"] = rng.normal(size=(N, 4)).astype(np.float32)
    arrs = {}
    for b, X in blocks.items():
        arrs[f"X_{b}"] = X
        arrs[f"H_{b}"] = (np.ones(N, bool) if not b.startswith("txt_f")
                          else nodes.fold.to_numpy() != int(b[-1]))
    np.savez_compressed(out / "blocks.npz", **arrs)
    mem = np.flatnonzero(k)
    E = np.array(sorted({(min(a, b), max(a, b)) for a in mem for b in mem if a != b and rng.random() < 0.3}), np.int32)
    para = E[:3]
    s3 = (sig[:, :5] + 0.3 * rng.normal(size=(N, 5))).astype(np.float32)
    spec = rng.normal(size=(N, 3)).astype(np.float32)
    np.savez_compressed(out / "structure.npz", E_struct=E, E_paralog_fident30=para, X_s3di=s3, H_s3di=np.ones(N, bool),
                        X_struct_spec=spec, H_struct_spec=np.ones(N, bool))
    B = np.zeros((N, 3))
    B[mem[:6], 0] = 1
    B[:N_GENES:7, 1] = 1
    B[mem[:2], 2] = 1
    sp.save_npz(out / "data" / "dom_membership_current.npz", sp.csr_matrix(B))
    (out / "data" / "dom_vocab_current.json").write_text('["PF00001", "PF00002", "IPR000001"]')
    return out
