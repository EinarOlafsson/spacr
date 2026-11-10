"""Item 888: the OPS module atlas, with the profile columns used as they are for weighted profiles.

PHATE and Leiden are the optional ``clustering`` extra, so these tests put a
deterministic stand-in in their place: a row whose profile points along the
first column is cluster 0, every other row the cluster of its largest
column. The plate A identity check (byte-identical tables, identical
partitions) ran against the pilot's stored outputs; see the item record.
"""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_clusters, ops_modules as om

N_GENES = 3000
DIMS = 20
MODULE = [f"G{i:04d}" for i in range(8)]
CORE = MODULE[:5]


def _fake_phate_leiden(components, resolutions, *, seed=42, **kwargs):
    """Cluster 0 for rows along the first column, else 1 + the largest other column."""
    x = np.asarray(components, float)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    labels = np.where(unit[:, 0] > 0.9, 0, 1 + np.argmax(np.abs(x[:, 1:]), axis=1))
    return {r: labels.copy() for r in resolutions}


@pytest.fixture
def fake_clustering(monkeypatch):
    """Replace PHATE + Leiden by the stand-in and count the calls."""
    calls = []

    def fake(components, resolutions, **kwargs):
        """Record the call, then cluster."""
        calls.append(np.asarray(components).copy())
        return _fake_phate_leiden(components, resolutions, **kwargs)

    monkeypatch.setattr(ops_clusters, "phate_leiden", fake)
    return calls


def _tables(seed=0):
    """Genes, guides and robust-z tables with one planted module along the first column."""
    rng = np.random.default_rng(seed)
    cols = [f"PC_{k}" for k in range(DIMS)]
    rows = []
    for i in range(N_GENES):
        gene = f"G{i:04d}"
        if gene in MODULE:
            centre = np.zeros(DIMS)
            centre[0] = 5.0
        else:
            centre = rng.normal(0, 1, DIMS)
            centre[0] = 0.0
        for k in range(4):
            rows.append({"guide": f"{gene}_{k}", "gene": gene, "n_cells": int(5 + (i * 997) % N_GENES),
                         **dict(zip(cols, centre + rng.normal(0, 0.05, DIMS)))})
    for k in range(32):
        v = rng.normal(0, 1, DIMS)
        v[0] = 0.0
        rows.append({"guide": f"nt_{k}", "gene": f"nontargeting_{k:02d}", "n_cells": 30,
                     **dict(zip(cols, v))})
    guides = pd.DataFrame(rows)
    genes = guides.groupby("gene", sort=True).agg(n_cells=("n_cells", "sum")).reset_index()
    for c in cols:
        genes[c] = guides.groupby("gene", sort=True)[c].median().to_numpy()
    z_cols = [f"feat_{k}" for k in range(10)]
    z_genes = pd.DataFrame(rng.normal(0, 1, (len(genes), 10)), columns=z_cols)
    z_genes.insert(0, "gene", genes.gene.to_numpy())
    z_genes["n_cells"] = genes.n_cells.to_numpy()
    z_guides = pd.DataFrame(rng.normal(0, 1, (len(guides), 10)), columns=z_cols)
    z_guides.insert(0, "gene", guides.gene.to_numpy())
    return genes, guides, z_genes, z_guides


def _references():
    """CORUM / KEGG / GO-slim groups, STRING, BioGRID and the specificity table."""
    corum = pd.DataFrame({"group": ["complex_A"] * 5 + ["complex_B"] * 4,
                          "gene_name": CORE + ["G0100", "G0101", "G0102", "G0103"]})
    kegg = pd.DataFrame({"group": ["pathway_X"] * 3, "gene_name": ["G0200", "G0201", "G0202"]})
    go = pd.DataFrame({"group": ["tiny"] * 2, "gene_name": ["G0300", "G0301"]})
    string = pd.DataFrame({"a": ["G0000", "G0001"], "b": ["G0007", "G0002"], "combined_score": [450, 900]})
    biogrid = pd.DataFrame({"a": ["G0500"], "b": ["G0501"]})
    spec = pd.DataFrame({"gene": ["G0006&G0999", "G0400"], "specificity_flag": [True, False]})
    return corum, kegg, go, string, biogrid, spec


def test_precomputed_embedding_keeps_the_columns_and_profile_columns_picks_pcs():
    frame = pd.DataFrame({"gene": ["a", "b"], "PC_0": [1.0, 2.0], "PC_1": [3.0, -1.0], "n_cells": [5, 6]})
    values, n, used = om.precomputed_embedding(frame, ["PC_0", "PC_1"])
    assert np.array_equal(values, [[1.0, 3.0], [2.0, -1.0]]) and n == used == 2
    assert om.profile_columns(frame) == ["PC_0", "PC_1"]
    plain = pd.DataFrame({"gene": ["a"], "guide": ["x"], "n_cells": [3], "area": [1.0], "label": ["q"]})
    assert om.profile_columns(plain) == ["area"]


def test_reference_partitions_embed_or_keep_the_columns(fake_clustering):
    genes, guides, _, _ = _tables()
    cols = om.profile_columns(guides)
    rows = ops_clusters.consensus_rows(genes, guides, cols)
    kept = om.reference_partitions(rows, cols, precomputed=True)
    assert kept["pcs"].dtype == np.float32
    assert np.allclose(kept["pcs"], rows[cols].to_numpy(float).astype(np.float32))
    assert kept["info"]["pcs"] == DIMS and kept["info"]["rows"] == len(rows) == N_GENES + 8
    assert set(kept["labels"]) == set(om.MODULE_RESOLUTIONS.values())
    module_rows = rows.gene.isin(MODULE).to_numpy()
    assert (kept["labels"][3][module_rows] == 0).all()
    embedded = om.reference_partitions(rows, cols, precomputed=False)
    assert embedded["info"]["pcs"] <= DIMS
    assert not np.allclose(embedded["pcs"][:, :2], kept["pcs"][:, :2])


def test_consensus_partitions_pass_the_columns_through_and_resume(fake_clustering, tmp_path):
    genes, guides, _, _ = _tables()
    cols = om.profile_columns(guides)
    lines = []
    first = om.consensus_partitions(genes, guides, cols, runs=3, precomputed=True,
                                    folder=str(tmp_path), report=lines.append)
    assert first["labels"].shape == (3, N_GENES + 8)
    assert [r["resolution"] for r in first["runs"]] == list(om.CONSENSUS_SWEEP[:3])
    assert ((first["labels"] == -1).sum(axis=1) == (N_GENES + 8) - int(0.9 * (N_GENES + 8))).all()
    assert len(lines) == 3 and json.loads(lines[0])["run"] == 0
    assert len(fake_clustering) == 3 and fake_clustering[0].shape[1] == DIMS
    again = om.consensus_partitions(genes, guides, cols, runs=3, precomputed=True, folder=str(tmp_path))
    assert len(fake_clustering) == 3
    assert np.array_equal(again["labels"], first["labels"]) and again["runs"] == first["runs"]
    standard = om.consensus_partitions(genes, guides, cols, runs=1)
    assert standard["labels"].shape == (1, N_GENES + 8)
    assert om.consensus_partitions(genes, guides, cols, runs=0)["labels"].shape == (0, 0)


def test_anchor_clusters_find_the_enriched_group_as_the_core():
    names = np.array([f"g{i}" for i in range(40)] + ["NTpseudo_00"])
    clusters = np.array([0] * 6 + [1] * 17 + [2] * 17 + [0])
    controls = np.array([False] * 40 + [True])
    groups = {"CORUM: c": {"g0", "g1", "g2", "g3"}, "KEGG: split": {"g6", "g7", "g23", "g24"}}
    out = om.anchor_clusters(clusters, names, controls, groups, 40)
    assert list(out) == [0]
    assert out[0]["anchor"] == "CORUM: c" and out[0]["core"] == ["g0", "g1", "g2", "g3"]
    assert out[0]["members"] == [f"g{i}" for i in range(6)] and out[0]["anchor_q"] < 0.05
    assert om.anchor_clusters(np.array([0, 0, 1]), np.array(["g0", "g1", "g2"]),
                              np.zeros(3, bool), groups, 3) == {}


def test_module_atlas_calls_the_planted_member_and_no_control(fake_clustering):
    genes, guides, z_genes, z_guides = _tables()
    cols = om.profile_columns(guides)
    rows = ops_clusters.consensus_rows(genes, guides, cols)
    ref = om.reference_partitions(rows, cols, precomputed=True)
    labels = om.consensus_partitions(genes, guides, cols, runs=6, precomputed=True)["labels"]
    corum, kegg, go, string, biogrid, spec = _references()
    background = set(rows.gene[~rows.gene.str.startswith("NTpseudo")])
    groups = om.module_groups(background, [("CORUM", corum), ("KEGG", kegg), ("GOslim", go)])
    assert set(groups) == {"CORUM: complex_A", "CORUM: complex_B", "KEGG: pathway_X"}
    out = om.module_atlas(rows, ref["pcs"], ref["labels"], labels, guides, z_genes, z_guides,
                          groups, string, biogrid, spec)
    members, atlas, summary = out["members"], out["atlas"], out["summary"]
    assert set(atlas.anchor) == {"CORUM: complex_A"} and len(atlas) == 3
    assert (atlas.known_core == " ".join(CORE)).all()
    assert set(members.gene) >= {"G0005", "G0006"}
    assert "G0007" not in set(members.gene)
    called = members[members.stable_new_member]
    assert set(called.gene) == {"G0005"}
    six = members[members.gene == "G0006"].iloc[0]
    assert bool(six.specificity_flag) and not bool(six.stable_new_member)
    assert not members[members.is_nt_pseudo].stable_new_member.any()
    assert summary["stable_new_member_genes"] == 1
    assert summary["anchored_modules"] == {"~70": 1, "~250": 1, "~500": 1}
    assert summary["nt_pseudo_genes_called (control)"] == 0
    assert summary["empirical_FDR_vs_NT_pseudo_genes"] == 0.0
    assert len(summary["shuffled_labels (control)"]) == 3
    assert summary["gates"] == om.GATES_TEXT
    assert (atlas.new_members == "G0005").all() and (atlas.n_new == 1).all()
    assert atlas.driving_features_core.str.count("feat_").eq(8).all()
    repeat = om.module_atlas(rows, ref["pcs"], ref["labels"], labels, guides, z_genes, z_guides,
                             groups, string, biogrid, spec)
    pd.testing.assert_frame_equal(repeat["members"], members)
    alone = om.module_atlas(rows, ref["pcs"], ref["labels"], labels, guides, z_genes, z_guides,
                            {"CORUM: complex_B": groups["CORUM: complex_B"]}, string, biogrid, spec,
                            shuffles=0)
    assert alone["members"].empty and alone["summary"]["empirical_FDR_vs_shuffled_labels"] is None
    assert alone["summary"]["anchored_modules"] == {"~70": 0, "~250": 0, "~500": 0}


def _write_inputs(tmp_path):
    """The profile folder and the reference folder of a settings run."""
    genes, guides, z_genes, z_guides = _tables()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    genes.to_parquet(profiles / "genes.parquet", index=False)
    guides.to_parquet(profiles / "guides.parquet", index=False)
    drivers = tmp_path / "drivers"
    drivers.mkdir()
    z_genes.to_parquet(drivers / "genes.parquet", index=False)
    z_guides.to_parquet(drivers / "guides.parquet", index=False)
    refs = tmp_path / "refs"
    refs.mkdir()
    corum, kegg, go, string, biogrid, spec = _references()
    corum.to_csv(refs / "corum_group_benchmark.tsv", sep="\t", index=False)
    kegg.to_csv(refs / "kegg_group_benchmark.tsv", sep="\t", index=False)
    go.to_parquet(refs / "go_slim_groups.parquet", index=False)
    string.to_parquet(refs / "string_v12_ge400.parquet", index=False)
    biogrid.to_parquet(refs / "biogrid_physical.parquet", index=False)
    spec.to_csv(refs / "gene_specificity.csv", index=False)
    return profiles, drivers, refs


def test_the_modules_phase_runs_from_settings_and_resumes(fake_clustering, tmp_path):
    from spacr.ops_engine import run_ops

    profiles, drivers, refs = _write_inputs(tmp_path)
    settings = {"ops_modules_profiles": str(profiles), "ops_modules_driver_profiles": str(drivers),
                "ops_reference_dir": str(refs), "ops_modules_runs": 4,
                "ops_modules_precomputed_embedding": True, "plate": "A"}
    out = run_ops(settings, phases=("modules",))["modules"]
    atlas_dir = profiles / "modules" / "atlas"
    consensus_dir = profiles / "modules" / "consensus"
    for name in ("module_members.csv", "module_atlas.csv", "module_summary.json"):
        assert (atlas_dir / name).is_file()
    for name in ("rows_reference.parquet", "reference.npz", "reference.json", "run_003.npz",
                 "consensus_partitions.npz", "consensus_runs.json"):
        assert (consensus_dir / name).is_file()
    summary = json.loads((atlas_dir / "module_summary.json").read_text())
    assert summary["stable_new_member_genes"] == out["summary"]["stable_new_member_genes"] == 1
    assert json.loads((consensus_dir / "reference.json").read_text())["pcs"] == DIMS
    with np.load(consensus_dir / "consensus_partitions.npz", allow_pickle=True) as stored:
        assert stored["labels"].shape == (4, N_GENES + 8)
    calls = len(fake_clustering)
    listed = {**settings, "ops_modules_driver_profiles": [str(drivers / "genes.parquet"),
                                                          str(drivers / "guides.parquet")]}
    again = run_ops(listed, phases=("modules",))["modules"]
    assert len(fake_clustering) == calls
    assert again["summary"] == out["summary"]


def test_the_modules_phase_finds_the_profile_run_folder(fake_clustering, tmp_path):
    from spacr.ops_modules import run_module_phase

    profiles, drivers, refs = _write_inputs(tmp_path)
    run = tmp_path / "dst" / "ops_profile" / "AB"
    run.mkdir(parents=True)
    for name in ("genes.parquet", "guides.parquet"):
        (run / name).write_bytes((profiles / name).read_bytes())
    base = {"dst_root": str(tmp_path / "dst"), "ops_modules_driver_profiles": str(drivers),
            "ops_reference_dir": str(refs), "ops_modules_runs": 2, "ops_modules_precomputed_embedding": True}
    out = run_module_phase({**base, "ops_profile_cells": {"B": "x", "A": "y"}})
    assert out["folder"] == str(run / "modules")
    named = tmp_path / "dst" / "ops_profile" / "named"
    named.mkdir()
    with pytest.raises(FileNotFoundError, match="genes.parquet"):
        run_module_phase({**base, "ops_profile_run": "named"})
    assert om._profile_folder({"dst_root": "d", "plate": "C"}).endswith("ops_profile/C")


def test_the_modules_phase_refuses_bad_settings(fake_clustering, tmp_path):
    from spacr.ops_modules import run_module_phase

    profiles, drivers, refs = _write_inputs(tmp_path)
    good = {"ops_modules_profiles": str(profiles), "ops_modules_driver_profiles": str(drivers),
            "ops_reference_dir": str(refs), "ops_modules_runs": 1, "ops_modules_precomputed_embedding": True}
    with pytest.raises(ValueError, match="ops_modules_runs"):
        run_module_phase({**good, "ops_modules_runs": 0})
    with pytest.raises(ValueError, match="dst_root"):
        run_module_phase({**good, "ops_modules_profiles": None})
    with pytest.raises(ValueError, match="paths"):
        run_module_phase({**good, "ops_modules_driver_profiles": [str(drivers)]})
    (refs / "gene_specificity.csv").unlink()
    with pytest.raises(FileNotFoundError, match="gene_specificity.csv"):
        run_module_phase(good)
