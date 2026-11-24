"""Cell-cycle nuisance sources and the stratified profile (spec G3 and G4)."""
import numpy as np
import pandas as pd
import pytest

from spacr import ops_tvn as T
from tests.test_ops_profile_nuisance import GENES, _corr, _prepared, _profile

STAGES = ["G1", "S", "G2", "prometaphase", "metaphase", "ana_telo_cyto"]
CLASSES = ["interphase", "g2_prophase", "prometaphase", "metaphase", "ana_telo_cyto",
           "telophase_daughters", "artefact"]
MITOTIC = ["prometaphase", "metaphase", "ana_telo_cyto"]


def _table(well, rng, n_per=120):
    rows = []
    for gene in GENES + ["nontargeting"] * 3:
        for k in range(2):
            guide = f"{gene}_{k}" if gene != "nontargeting" else f"nt{len(rows)}_{k}"
            rows += [(gene, guide)] * n_per
    n = len(rows)
    stage = rng.choice(STAGES, n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    frame = pd.DataFrame({"well": well, "sgRNA": [r[1] for r in rows], "gene": [r[0] for r in rows],
                          "mitotic": False, "site": rng.integers(1, 4, n),
                          "cell_cycle_stage": stage})
    probs = rng.dirichlet(np.ones(len(CLASSES)), n)
    for j, c in enumerate(CLASSES):
        frame[f"p_{c}"] = probs[:, j]
    frame["posterior_S"] = rng.random(n) * 0.5
    frame["posterior_G2"] = rng.random(n) * 0.5
    frame["dna_log2_fc"] = np.where(stage == "G1", 0.0, rng.uniform(0, 1, n))
    mit = np.isin(stage, MITOTIC).astype(float)
    x = rng.standard_normal((n, 5)) * 0.4
    x[:, 0] += 3 * mit
    x[:, 1] += 2 * (stage == "S") + 3 * (stage == "G2")
    x[:, 2] += 4 * frame["p_g2_prophase"].to_numpy()
    x[:, 3] += 3 * frame["dna_log2_fc"].to_numpy() ** 2
    x[:, 4] += 3 * (frame.gene.to_numpy() == "G1")
    for j, name in enumerate(["cell_dapi_mean_intensity", "cell_a", "cell_b", "cell_c", "nucleus_d"]):
        frame[name] = x[:, j]
    return frame


def _cells(folder, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for well in ("A1", "A2"):
        _table(well, rng).to_parquet(folder / f"raw_{well}.parquet", index=False)
    return folder


def _removed(cells, kind):
    kinds = T.nuisance_kinds([kind])
    metas, values, columns = _prepared(cells, kinds)
    names = [c for c in metas[0].columns if c.startswith("nz_")]
    control = [m.gene_symbol_0.str.startswith("nontargeting").to_numpy() for m in metas]
    covariates = [m[names].to_numpy(float) for m in metas]
    before = np.vstack(values)
    T.regress_nuisance(values, covariates, control)
    on = np.concatenate(control)
    return columns, before[on], np.vstack(values)[on], names, np.vstack(covariates)[on]


@pytest.mark.parametrize("kind,feature,column", [
    ("mitotic_binary", "cell_dapi_mean_intensity", "nz_mitotic"),
    ("phase_onehot", "cell_a", None),
    ("stage_probs", "cell_b", "nz_cc_p_g2_prophase"),
    ("dna_content", "cell_c", "nz_cc_dna_sq")])
def test_each_source_removes_its_planted_effect(tmp_path, kind, feature, column):
    columns, before, after, names, z = _removed(_cells(tmp_path / "cells"), kind)
    j = columns.index(feature)
    bound = 3 / np.sqrt(len(after))
    if column is None:
        assert names == ["nz_phase_S", "nz_phase_G2", "nz_phase_M"]
        design = np.nan_to_num(z)
        for c in range(3):
            assert _corr(after[:, j], design[:, c]) < bound
        assert max(_corr(before[:, j], design[:, c]) for c in range(3)) > 0.3
    else:
        assert _corr(before[:, j], z[:, names.index(column)]) > 0.3
        assert _corr(after[:, j], z[:, names.index(column)]) < bound


def test_cell_cycle_is_the_union_of_stage_probs_and_dna_content(tmp_path):
    cells = pd.read_parquet(_cells(tmp_path / "cells") / "raw_A1.parquet")
    union = T.nuisance_covariates(cells, ["cell_cycle"])
    parts = pd.concat([T.nuisance_covariates(cells, ["stage_probs"]),
                       T.nuisance_covariates(cells, ["dna_content"])], axis=1)
    assert set(union) == set(parts)
    pd.testing.assert_frame_equal(union, parts[list(union.columns)])


def test_mitotic_binary_marks_the_mitotic_cells_and_keeps_them(tmp_path):
    cells = pd.read_parquet(_cells(tmp_path / "cells") / "raw_A1.parquet")
    z = T.nuisance_covariates(cells, ["mitotic_binary"])["nz_mitotic"]
    assert (z.to_numpy() == cells["cell_cycle_stage"].isin(MITOTIC).to_numpy()).all()
    assert 0 < z.mean() < 1


def test_a_missing_source_says_what_to_join():
    with pytest.raises(ValueError, match="cell_cycle_stage"):
        T.nuisance_source_columns(["site"], ["phase_onehot"])


def test_stratified_profile_equals_a_manual_subset(tmp_path):
    cells = _cells(tmp_path / "cells")
    manual = tmp_path / "manual"
    manual.mkdir()
    for path in sorted(cells.glob("raw_*.parquet")):
        frame = pd.read_parquet(path)
        frame[frame["cell_cycle_stage"] == "G1"].to_parquet(manual / path.name, index=False)
    genes, guides, info = _profile(cells, ops_profile_phase_filter="G1")
    want, want_guides, _ = _profile(manual)
    pd.testing.assert_frame_equal(genes, want)
    pd.testing.assert_frame_equal(guides, want_guides)
    assert info["phase_filter"]["phase"] == "G1"
    assert info["phase_filter"]["cells_per_gene"]["G1"] == int(
        genes.set_index("gene").loc["G1", "n_cells"])
    assert not genes.equals(_profile(cells)[0])


def test_the_default_leaves_profiles_unchanged(tmp_path):
    cells = _cells(tmp_path / "cells")
    base = _profile(cells)
    pd.testing.assert_frame_equal(_profile(cells, ops_profile_phase_filter=None)[0], base[0])
    assert "phase_filter" not in base[2]
