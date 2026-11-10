"""Item 879: one settings-driven profile path, stored cell tables -> gene profiles -> metrics."""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_reference as R
from spacr import ops_tvn as T
from spacr.ops_engine import run_ops


@pytest.fixture(autouse=True)
def _fresh_cache():
    R.clear_cache()
    yield
    R.clear_cache()


FEATURES = ["cell_area", "cell_dapi_mean_intensity", "nucleus_gfp_mean_intensity", "cell_a594_mean_intensity",
            "cytoplasm_af750_mean_intensity", "nucleus_area", "cell_texture_1", "cell_texture_2", "cell_gappy"]


def _cell_tables(folder, wells=("A1", "A2"), seed=0, genes=12, guides=3, controls=9, cells=14):
    """Stored cell tables: genes in four groups with a shared shift, control guides, mitotic cells, gaps."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    shift = {g: rng.standard_normal(len(FEATURES)) * 2 for g in range(4)}
    for w in wells:
        rows = []
        for g in range(genes):
            for k in range(guides):
                for _ in range(cells):
                    rows.append((f"G{g}", f"g{g}_{k}", shift[g % 4] + rng.standard_normal(len(FEATURES))))
        for k in range(controls):
            for _ in range(cells):
                rows.append(("nontargeting", f"nt{k}", rng.standard_normal(len(FEATURES))))
        values = np.array([r[2] for r in rows])
        frame = pd.DataFrame(values, columns=FEATURES)
        frame.insert(0, "well", w)
        frame.insert(1, "sgRNA", [r[1] for r in rows])
        frame.insert(2, "gene", [r[0] for r in rows])
        frame.insert(3, "mitotic", rng.random(len(rows)) < 0.05)
        frame["cell_centroid_x"] = 1.0
        frame.loc[rng.random(len(rows)) < 0.2, "cell_gappy"] = np.nan
        frame.loc[3, "cell_area"] = np.inf
        frame.loc[5, ["cell_texture_1", "cell_texture_2"]] = np.nan
        frame.to_parquet(folder / f"raw_{w}.parquet", index=False)
    return folder


def _references(folder):
    groups = {f"cx{g}": [f"G{x}" for x in range(12) if x % 4 == g] for g in range(4)}
    rows = [(m, c) for c, ms in groups.items() for m in ms]
    pd.DataFrame(rows, columns=["gene_name", "group"]).to_csv(folder / "corum_group_benchmark.tsv", sep="\t",
                                                               index=False)
    pd.DataFrame(rows, columns=["gene_name", "group"]).to_csv(folder / "kegg_group_benchmark.tsv", sep="\t",
                                                               index=False)
    pairs = sorted(R._group_pairs(groups))
    pd.DataFrame([(g, k) for k, p in enumerate(pairs) for g in p], columns=["gene_name", "pair"]).to_csv(
        folder / "string_pair_benchmark.tsv", sep="\t", index=False)
    frame = pd.DataFrame(pairs, columns=["a", "b"])
    frame["combined_score"] = 900
    frame.to_parquet(folder / "string_v12_ge400.parquet", index=False)
    frame[["a", "b"]].to_parquet(folder / "biogrid_physical.parquet", index=False)


def test_recipe_defaults_and_checks():
    assert T.profile_recipe() == T.PROFILE_RECIPE
    assert T.profile_recipe({"agg": "gene"})["agg"] == "gene"
    with pytest.raises(ValueError, match="unknown keys"):
        T.profile_recipe({"lof": 0, "scale": 1})
    with pytest.raises(ValueError, match="must be one of"):
        T.profile_recipe({"pca": "umap"})


def test_pseudo_genes_pool_four_control_guides_with_a_fixed_permutation():
    pool = [f"nt{k}" for k in range(9)]
    names = T.nt_pseudo_genes(pool + ["ntX", "g1"], ["nontargeting"] * 10 + ["G1"], pool)
    order = np.argsort(np.random.default_rng(0).permutation(9))
    want = {g: "NTpseudo_%02d" % (k // 4) for g, k in zip(sorted(pool), order)}
    assert names[:9].tolist() == [want[g] for g in pool]
    assert names[9] == "nontargeting_drop" and names[10] == "G1"
    assert sorted(set(names[:9])) == ["NTpseudo_00", "NTpseudo_01", "NTpseudo_02"]


def test_profile_tables_from_stored_cells(tmp_path):
    cells = _cell_tables(tmp_path / "cells")
    settings = {"ops_profile_cells": str(cells), "plate": "P1", "ops_tvn_whiten_epsilon": 0,
                "ops_profile_min_cells": 10}
    genes, guides, info = T.profile_cell_tables(settings, report=print)
    assert set(genes.gene) == {f"G{g}" for g in range(12)} | {"NTpseudo_00", "NTpseudo_01", "NTpseudo_02"}
    pcs = [c for c in genes.columns if c.startswith("PC_")]
    assert list(genes.columns[:2]) == ["gene", "n_cells"] and len(pcs) == info["tvn_dims"] == info["pcs"]
    assert list(guides.columns[:3]) == ["guide", "gene", "n_cells"]
    assert genes.set_index("gene").n_cells.sum() == guides.n_cells.sum() == info["cells"]
    g0 = guides[guides.gene == "G0"]
    assert np.allclose(genes.set_index("gene").loc["G0", pcs].to_numpy(float), g0[pcs].median().to_numpy())
    assert info["features"] == 8 and info["recipe"] == T.PROFILE_RECIPE
    assert info["runtime"]["plates"] == ["P1"]
    shifted, _, _ = T.profile_cell_tables({**settings, "ops_tvn_whiten_epsilon": 3.0})
    assert not np.allclose(shifted[pcs].to_numpy(float), genes[pcs].to_numpy(float))


def test_every_recipe_choice_runs(tmp_path):
    cells = _cell_tables(tmp_path / "cells", wells=("A1",), cells=12)
    base = {"ops_profile_cells": {"Q": str(cells)}, "ops_tvn_whiten_epsilon": 3, "ops_profile_min_cells": 5}
    recipe = {"lof": 1, "impute": "knn", "pca": "brieflow", "agg": "gene", "nt": "barcode"}
    genes, guides, info = T.profile_cell_tables({**base, "ops_profile_recipe": recipe})
    assert info["runtime"]["plates"] == ["Q"]
    assert {f"nontargeting_nt{k}" for k in range(9)} <= set(genes.gene)
    assert info["cells"] < 12 * (12 * 3 + 9)
    hits = pd.DataFrame({"gene": ["G0", "G1", "G2"], "a_fdr": [0.01, 0.2, 0.3], "b_fdr": [0.5, 0.04, 0.9]})
    hits.to_csv(tmp_path / "hits.csv", index=False)
    active, kept, _ = T.profile_cell_tables({**base, "ops_profile_active_filter": {
        "hits_table": str(tmp_path / "hits.csv"), "fdr": 0.05}})
    assert {g for g in active.gene if not g.startswith("NT")} == {"G0", "G1"}
    assert set(kept.gene) == set(active.gene)


def test_bad_inputs_are_refused(tmp_path):
    cells = _cell_tables(tmp_path / "cells", wells=("A1",))
    with pytest.raises(ValueError, match="not available"):
        T.profile_cell_tables({"ops_profile_cells": str(cells), "ops_tvn_whiten_epsilon": 0,
                               "ops_profile_shrinkage": "eb_nt"})
    with pytest.raises(ValueError, match="is not a folder"):
        T.profile_cell_tables({"ops_profile_cells": str(tmp_path / "none"), "ops_tvn_whiten_epsilon": 0})
    with pytest.raises(ValueError, match="no stored cell table"):
        T.profile_cell_tables({"ops_profile_cells": str(cells), "ops_profile_wells": ["Z9"],
                               "ops_tvn_whiten_epsilon": 0})
    with pytest.raises(ValueError, match="dst_root"):
        run_ops({"ops_profile_cells": str(cells)}, phases=("profile",))
    with pytest.raises(ValueError, match="cosine"):
        run_ops({"dst_root": str(tmp_path), "ops_profile_similarity": "euclidean"}, phases=("profile",))
    with pytest.raises(ValueError, match="on their own"):
        run_ops({"dst_root": str(tmp_path)}, phases=("aggregate", "profile"))
    with pytest.raises(ValueError, match="run the profile phase first"):
        run_ops({"dst_root": str(tmp_path / "empty")}, phases=("evaluate",))


def test_settings_run_writes_profiles_metrics_and_their_record(tmp_path):
    cells = _cell_tables(tmp_path / "cells", genes=12, controls=9)
    refs = tmp_path / "refs"
    refs.mkdir()
    _references(refs)
    settings = {"dst_root": str(tmp_path / "out"), "plate": "P1", "ops_profile_cells": str(cells),
                "ops_tvn_whiten_epsilon": 0, "ops_reference_dir": str(refs), "ops_eval_null_size": 100,
                "ops_eval_references": ["CORUM", "STRING>=0.7", "KEGG", "BioGRID"], "ops_eval_decoys": 2}
    out = run_ops(settings, phases=("profile", "evaluate"))
    folder = tmp_path / "out" / "ops_profile" / "P1"
    assert out["folder"] == str(folder)
    for name in ("genes.parquet", "guides.parquet", "metrics.csv", "metrics.extra.csv", "per_gene.csv.gz",
                 "runtime.json"):
        assert (folder / name).is_file(), name
    metrics = pd.read_csv(folder / "metrics.csv")
    corum = metrics[(metrics.reference == "CORUM") & metrics.control.isna()].iloc[0]
    assert corum["queries"] == 12 and corum["mAP"] > corum["expected_mAP_random"]
    record = json.loads((folder / "runtime.json").read_text())
    assert record["settings"]["ops_tvn_whiten_epsilon"] == 0 and record["phases"] == ["profile", "evaluate"]
    assert set(record["references"]) >= {"CORUM", "CORUM@SELECT"} and "metrics_s" in record["runtime"]
    genes = pd.read_parquet(folder / "genes.parquet")
    again = run_ops({**settings, "ops_eval_covariates": None}, phases=("evaluate",))
    assert pd.read_csv(folder / "metrics.csv").equals(metrics)
    assert "genes" not in again["outputs"]
    cov = pd.DataFrame({"gene": genes.gene, "cells_per_guide": genes.n_cells / 3.0})
    cov.to_csv(tmp_path / "cov.csv", index=False)
    fitness = T._profile_covariates({"ops_eval_covariates": str(tmp_path / "cov.csv")}, None)
    assert fitness["G0"] == pytest.approx(genes.set_index("gene").n_cells["G0"] / 3.0)
    pooled = run_ops({"dst_root": str(tmp_path / "pool"), "ops_profile_cells": {"P1": str(cells), "P2": str(cells)},
                      "ops_tvn_whiten_epsilon": 0}, phases=("profile",))
    assert pooled["folder"].endswith("P1P2")


def test_knn_imputation_fills_gaps_from_neighbours():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": rng.standard_normal(40), "b": rng.integers(0, 5, 40), "c": rng.standard_normal(40)})
    frame.loc[[2, 7], "a"] = np.nan
    frame["b"] = frame["b"].astype("Int64")
    frame.loc[3, "b"] = pd.NA
    filled = T._knn_impute(frame.copy(), batch_size=1, sample_size=10)
    assert not filled.isna().any().any() and filled["b"].dtype == "float64"
    assert filled.loc[0, "a"] == frame.loc[0, "a"]
    clean = pd.DataFrame({"a": [1.0, 2.0]})
    assert T._knn_impute(clean) is clean


def test_a_named_run_goes_to_its_own_folder(tmp_path):
    cells = _cell_tables(tmp_path / "cells", wells=("A1",))
    out = run_ops({"dst_root": str(tmp_path), "ops_profile_cells": str(cells), "ops_profile_run": "named",
                   "ops_tvn_whiten_epsilon": 0, "ops_profile_min_cells": 5}, phases=("profile",))
    assert out["folder"] == str(tmp_path / "ops_profile" / "named") and "genes" in out["outputs"]
