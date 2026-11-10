"""Item 906: gene profiles from frozen single-cell embeddings, and fusion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import embedding_profiles, fusion_profiles


def _export(seed=0, cells=3000, dims=16):
    """An fp16 embedding export with a 20 % wider row set than the cells,
    two wells with a well offset, and gene GA shifted along one axis."""
    rng = np.random.default_rng(seed)
    total = int(cells * 1.2)
    embedding = rng.normal(0, 1, (total, dims)).astype(np.float16)
    rows = np.sort(rng.choice(total, cells, replace=False))
    genes = rng.choice(["GA", "GB", "GC", "nontargeting"], cells, p=[0.2, 0.2, 0.2, 0.4])
    guide = [f"{g}_{k}" if g != "nontargeting" else f"nt{k}"
             for g, k in zip(genes, rng.integers(0, 4, cells))]
    wells = np.where(np.arange(cells) < cells // 2, "A1", "A2")
    embedding[rows[wells == "A2"]] += np.float16(2.0)
    embedding[rows[genes == "GA"], 0] += np.float16(3.0)
    low = pd.Series(guide).str.endswith(("0", "1")).to_numpy()
    names = np.where(genes == "nontargeting",
                     np.where(low, "NTpseudo_00", "NTpseudo_01"), genes)
    table = pd.DataFrame({"sgRNA": guide, "gene": names, "well": wells})
    return embedding, rows, table, table["gene"].str.startswith("NTpseudo")


@pytest.mark.parametrize("norm", ["tvn", "robustz"])
def test_embedding_profiles_find_the_shifted_gene(norm):
    embedding, rows, table, control = _export()
    genes, info = embedding_profiles(embedding, table, control=control, rows=rows,
                                     norm=norm, sample=1500, chunk=700)
    assert info["cells"] == len(table) and info["components"] >= 2
    profile = genes.set_index("gene").filter(like="emb_")
    norms = np.linalg.norm(profile.to_numpy(), axis=1)
    assert profile.index[np.argmax(norms)] == "GA"
    assert list(genes.columns[:3]) == ["gene", "n_guides", "n_cells"]


def test_dropped_cells_stay_in_the_statistics_but_not_the_profiles():
    embedding, rows, table, control = _export(seed=1)
    table.loc[table.index[:50], "gene"] = "nontargeting_drop"
    genes, info = embedding_profiles(embedding, table, control=control, rows=rows,
                                     sample=1000)
    assert "nontargeting_drop" not in set(genes["gene"])
    assert info["cells"] == len(table) - 50
    with pytest.raises(ValueError, match="norm must be one of"):
        embedding_profiles(embedding, table, control=control, rows=rows, norm="zca")


def test_fusion_gives_each_block_equal_weight():
    rng = np.random.default_rng(2)
    names = [f"G{k:02d}" for k in range(40)]
    small = pd.DataFrame({"gene": names, "n_cells": np.arange(40) + 10,
                          **{f"PC_{d}": rng.normal(0, 1, 40) for d in range(3)}})
    wide = pd.DataFrame({"gene": names[::-1][:35], "n_cells": 1,
                         **{f"emb_{d}": rng.normal(0, 5, 35) for d in range(30)}})
    fused = fusion_profiles({"meas": small, "emb": wide})
    assert len(fused) == 35 and fused["gene"].isin(names[5:]).all()
    meas = fused.filter(like="f_meas").to_numpy()
    emb = fused.filter(like="f_emb").to_numpy()
    assert (fused.set_index("gene")["n_cells"] == small.set_index("gene")["n_cells"]
            .reindex(fused["gene"]).to_numpy()).all()
    assert (meas ** 2).sum(1).mean() == pytest.approx((emb ** 2).sum(1).mean(), rel=0.35)


def _export_folder(tmp_path, wells=("A1", "A2", "B1"), n=1800, dims=12):
    """An embedding export of the profile plate's cells in the frozen-export
    layout: embeddings.npy and index.parquet."""
    rng = np.random.default_rng(7)
    folder = tmp_path / "export"
    folder.mkdir()
    index = pd.DataFrame({"well": np.repeat(list(wells), n),
                          "site": np.tile(np.arange(1, n + 1) % 6, len(wells)),
                          "label": np.tile(np.arange(1, n + 1), len(wells))})
    index = index.sample(frac=1.0, random_state=0).reset_index(drop=True)
    index.insert(0, "row", np.arange(len(index)))
    np.save(folder / "embeddings.npy", rng.normal(0, 1, (len(index), dims)).astype(np.float16))
    index.to_parquet(folder / "index.parquet")
    return folder


def test_the_aggregate_phase_writes_embedding_and_fusion_profiles(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    folder = _export_folder(tmp_path)
    report = _run(tmp_path, ops_profile_space=f"fusion:{folder}",
                  ops_fusion_blocks="meas,emb", ops_profile_cell_cycle="G1")
    options = report["profile_options"]
    assert options["embedding"]["cells_matched"] < 3 * 1800
    embedded = pd.read_parquet(options["outputs"]["genes_embedding_parquet"])
    fused = pd.read_parquet(options["outputs"]["genes_fusion_parquet"])
    assert embedded.columns[3] == "emb_0"
    assert fused.columns[2].startswith("f_meas")
    assert set(options["embedding"]["fusion_dims"]) == {"meas", "emb"}


def test_the_profile_space_is_checked(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    folder = _export_folder(tmp_path)
    with pytest.raises(ValueError, match="ops_profile_space must be"):
        _run(tmp_path, ops_profile_space="umap")
    with pytest.raises(ValueError, match="ops_fusion_blocks names"):
        _run(tmp_path, ops_profile_space=f"fusion:{folder}", ops_fusion_blocks="meas,dino")
