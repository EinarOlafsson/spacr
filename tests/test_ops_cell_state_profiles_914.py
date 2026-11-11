"""Item 914: cell-state stratified OPS gene profiles and the per-gene mitotic index test.

A synthetic screen has interphase and mitotic cells told apart by one DNA
feature. One gene changes only its mitotic cells, another only its
interphase cells, and one gene arrests cells in mitosis. Each state run
through the whole recipe must show its own gene and not the other's; the
index test must call the arrest gene and not the control guides.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from spacr import ops_mitotic as M
from spacr import ops_tvn as T
from spacr.ops_engine import run_ops

FEATURES = ["cell_area", "cell_dapi_mean_intensity", "nucleus_gfp_mean_intensity", "cell_a594_mean_intensity",
            "cytoplasm_af750_mean_intensity", "nucleus_area", "cell_texture_1", "cell_texture_2"]
GENES = [f"G{g}" for g in range(10)]


def _cells(folder, seed=0):
    """Two wells; G0 moves mitotic cells only, G1 interphase cells only, G2 arrests in mitosis."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    shift = np.zeros(len(FEATURES))
    shift[[0, 2, 3, 6]] = 3.0
    for well in ("A1", "A2"):
        rows = []
        for gene in GENES + ["nontargeting"] * 3:
            for k in range(3):
                guide = f"{gene}_{k}" if gene != "nontargeting" else f"nt{len(rows)}_{k}"
                for _ in range(60):
                    mitotic = rng.random() < (0.6 if gene == "G2" else 0.3)
                    x = rng.standard_normal(len(FEATURES))
                    x[1] += 5.0 if mitotic else 0.0
                    if (gene == "G0" and mitotic) or (gene == "G1" and not mitotic):
                        x += shift
                    rows.append((gene, guide, mitotic, x))
        frame = pd.DataFrame(np.array([r[3] for r in rows]), columns=FEATURES)
        frame.insert(0, "well", well)
        frame.insert(1, "sgRNA", [r[1] for r in rows])
        frame.insert(2, "gene", [r[0] for r in rows])
        frame.insert(3, "mitotic", [bool(r[2]) for r in rows])
        frame.to_parquet(folder / f"raw_{well}.parquet", index=False)
    return folder


def _classifier(path, cells):
    """A logistic classifier on the well-normalised DNA feature, saved as a bundle."""
    table = pd.concat([pd.read_parquet(p) for p in sorted(cells.glob("raw_*.parquet"))], ignore_index=True)
    x = M._well_robust_z(table, ["cell_dapi_mean_intensity"], "well")
    model = LogisticRegression().fit(x, table.mitotic.to_numpy())
    joblib.dump({"format": M.MODEL_FORMAT, "model": model, "features": ["cell_dapi_mean_intensity"],
                 "normalise_by": "well", "threshold": 0.5, "summary": {}, "folds": []}, path)
    return str(path)


def _distance(genes, name):
    """Distance of a gene row from the mean control row."""
    cols = [c for c in genes.columns if c.startswith(("PC_", "w_"))]
    control = genes[genes.gene.str.lower().str.startswith(("nontargeting", "ntpseudo"))][cols].mean()
    return float(np.linalg.norm(genes.set_index("gene").loc[name, cols] - control))


def test_each_state_shows_its_own_gene_effect_and_the_index_test_calls_the_arrest_gene(tmp_path):
    cells = _cells(tmp_path / "cells")
    settings = {"dst_root": str(tmp_path / "run"), "plate": "P1", "ops_profile_cells": str(cells),
                "ops_tvn_whiten_epsilon": 0, "ops_profile_min_cells": 10,
                "ops_profile_cell_state": ["interphase_classifier", "mitotic_classifier", "all"],
                "ops_mitotic_model": _classifier(tmp_path / "clf.joblib", cells),
                "ops_mitotic_index_test": True, "ops_mitotic_index_min_cells": 5}
    result = run_ops(settings, phases=["profile"])
    folder = result["folder"]
    for state in ("interphase_classifier", "mitotic_classifier", "all"):
        assert os.path.isfile(os.path.join(folder, f"genes_{state}.parquet"))
        assert os.path.isfile(os.path.join(folder, f"guides_{state}.parquet"))
    mitotic = pd.read_parquet(os.path.join(folder, "genes_mitotic_classifier.parquet"))
    interphase = pd.read_parquet(os.path.join(folder, "genes_interphase_classifier.parquet"))
    assert _distance(mitotic, "G0") > 2 * _distance(mitotic, "G1")
    assert _distance(interphase, "G1") > 2 * _distance(interphase, "G0")
    per_gene = pd.read_csv(os.path.join(folder, "mitotic_index_per_gene.csv")).set_index("gene")
    assert per_gene.loc["G2", "q"] < 0.05 and per_gene.loc["G2", "z"] > 0
    assert (per_gene.drop(index="G2").q > 0.05).mean() > 0.8
    summary = json.load(open(os.path.join(folder, "mitotic_index_summary.json")))
    assert summary["nt_guides"] == 9 and summary["nt_guides_p05_pct"] < 50
    assert "same pooled control rate" in summary["matching"]
    runtime = json.load(open(os.path.join(folder, "runtime.json")))
    assert set(runtime["states"]) == {"interphase_classifier", "mitotic_classifier", "all"}
    assert "ops_mitotic_model" in runtime["settings"]


def test_default_state_keeps_the_unsuffixed_outputs(tmp_path):
    cells = _cells(tmp_path / "cells", seed=1)
    settings = {"dst_root": str(tmp_path / "run"), "plate": "P1", "ops_profile_cells": str(cells),
                "ops_tvn_whiten_epsilon": 0, "ops_profile_min_cells": 10}
    folder = run_ops(settings, phases=["profile"])["folder"]
    assert sorted(n for n in os.listdir(folder) if n.endswith(".parquet")) == ["genes.parquet", "guides.parquet"]
    runtime = json.load(open(os.path.join(folder, "runtime.json")))
    assert not any(k.startswith("ops_mitotic_") for k in runtime["settings"])


def test_weighted_rows_and_gene_universe(tmp_path):
    cells = _cells(tmp_path / "cells", seed=2)
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps(list(reversed(GENES))))
    settings = {"dst_root": str(tmp_path / "run"), "plate": "P1", "ops_profile_cells": str(cells),
                "ops_tvn_whiten_epsilon": 0, "ops_profile_min_cells": 10, "ops_profile_cell_state": "all",
                "ops_profile_recipe": {"nt": "barcode"}, "ops_profile_gene_universe": str(universe),
                "ops_profile_weighting": "contrastive_wiener", "ops_guide_efficacy": "jacks_blend"}
    folder = run_ops(settings, phases=["profile"])["folder"]
    rows = np.load(os.path.join(folder, "rows_all.npz"))
    names = rows["names"].tolist()
    assert names[:len(GENES)] == list(reversed(GENES))
    assert all(n.startswith("nontargeting") for n in names[len(GENES):])
    assert rows["pcs"].dtype == np.float32 and len(rows["pcs"]) == len(names)
    with pytest.raises(ValueError, match="weighted space"):
        run_ops(dict(settings, ops_profile_weighting=None, dst_root=str(tmp_path / "r2")), phases=["profile"])


def test_states_and_classifier_are_checked(tmp_path):
    assert T.cell_states(None) == ["interphase_rule"]
    assert T.cell_states("all, mitotic_classifier") == ["all", "mitotic_classifier"]
    with pytest.raises(ValueError, match="ops_profile_cell_state"):
        T.cell_states(["g2"])
    cells = pd.DataFrame({"mitotic": [True, False]})
    assert T._state_keep(cells, "all", None).tolist() == [True, True]
    with pytest.raises(ValueError, match="ops_mitotic_model"):
        T._state_keep(cells, "mitotic_classifier", None)
    with pytest.raises(ValueError, match="ops_mitotic_model"):
        run_ops({"dst_root": str(tmp_path), "ops_profile_cells": str(_cells(tmp_path / "c")),
                 "ops_tvn_whiten_epsilon": 0, "ops_mitotic_index_test": True}, phases=["profile"])
    assert T._json_list(["a", 1]) == ["a", "1"]


def test_mitotic_index_table_and_published_index():
    rng = np.random.default_rng(0)
    rows = []
    for gene, rate in (("UP", 0.5), ("DOWN", 0.0), ("FLAT", 0.1)):
        for i in range(400):
            called = (i % 10 == 0) if gene == "FLAT" else rng.random() < rate
            rows.append((gene, f"{gene}_{i % 4}", called, rng.random() < 0.1))
    for i in range(4000):
        rows.append(("nontargeting", f"nt{i % 10}", (i // 10) % 10 == 0, rng.random() < 0.1))
    cells = pd.DataFrame(rows, columns=["gene", "sgRNA", "clf", "rule"])
    published = pd.DataFrame({"gene_symbol": ["UP", "DOWN", "FLAT", "UP"],
                              **{f"sgRNA_{k}_mitotic_cells": [50, 0, 10, 1] for k in range(1, 5)},
                              **{f"sgRNA_{k}_interphase_cells": [50, 100, 90, 1] for k in range(1, 5)}})
    reference = M.published_mitotic_index(published)
    assert reference["UP"] == 0.5 and reference["DOWN"] == 0.0
    genes, guides, summary = M.mitotic_index_table(cells, reference=reference, min_cells=10, shuffles=20)
    assert list(genes.index[:1]) == ["UP"] and genes.loc["UP", "q"] < 1e-10 and genes.loc["DOWN", "z"] < 0
    assert summary["genes_q05_up"] == 1 and summary["genes_q05_down"] >= 1 and genes.loc["DOWN", "q"] < 0.05
    assert len(guides) == 10 and "q" not in guides and summary["genes_rho"] == 3
    assert summary["rho_mitotic_index_vs_reference"] == pytest.approx(1.0)
    plain, _g, plain_summary = M.mitotic_index_table(cells, rule=None)
    assert "rule_index" not in plain and "genes_rho" not in plain_summary
    assert M._bh(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.04, 0.04])
