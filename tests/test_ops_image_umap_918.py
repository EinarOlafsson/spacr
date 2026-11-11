"""Item 918: the ``ops_profiles`` source of Image UMAP.

leidenalg and igraph are the optional ``clustering`` extra and are not in the
test environment, so the graph is clustered by stand-ins (an igraph that keeps
its edges and a Leiden that returns connected components above a cosine
floor); the planted gene groups are tight blocks, so the right partition is
known.
"""
from __future__ import annotations

import filecmp
import glob
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import ops_image_umap as oiu


def _components(n, edges, weights, floor):
    """Connected components over the edges whose weight exceeds ``floor``."""
    parent = list(range(n))

    def root(i):
        """The component root of ``i``."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), w in zip(edges, weights):
        if w > floor:
            parent[root(a)] = root(b)
    roots = [root(i) for i in range(n)]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [order[r] for r in roots]


@pytest.fixture
def leiden(monkeypatch):
    """Stand-in igraph and leidenalg modules; returns the calls made."""
    calls = []

    class Graph:
        """Keeps its edges."""

        def __init__(self, n, edges, directed):
            self.n, self.edges, self.es = n, list(edges), {}

    def find_partition(graph, kind, *, weights, n_iterations, seed, resolution_parameter):
        """Connected components above a cosine of 0.5."""
        calls.append((seed, resolution_parameter))
        return types.SimpleNamespace(membership=_components(graph.n, graph.edges,
                                                            graph.es[weights], 0.5))

    monkeypatch.setitem(sys.modules, "igraph", types.SimpleNamespace(Graph=Graph))
    monkeypatch.setitem(sys.modules, "leidenalg", types.SimpleNamespace(
        find_partition=find_partition, RBConfigurationVertexPartition=object))
    return calls


def _blocks(n_blocks=6, size=15, dims=12, noise=0.02, seed=0, controls=6):
    """Planted gene groups along separate axes; gene 0 of each is the centre."""
    rng = np.random.default_rng(seed)
    rows, names, block = [], [], []
    for b in range(n_blocks):
        centre = np.zeros(dims)
        centre[b] = 1.0
        for i in range(size):
            rows.append(centre + (0 if i == 0 else noise) * rng.standard_normal(dims))
            names.append(f"G{b}_{i:02d}")
            block.append(b)
    for i in range(controls):
        rows.append(np.full(dims, 0.3) + noise * rng.standard_normal(dims))
        names.append(f"nontargeting_{i}")
        block.append(-1)
    return np.array(names), np.vstack(rows), np.array(block)


def _inputs(tmp_path, names, rows, block):
    """Profile npz, a planted gene-set file and a cell table with fields."""
    np.savez(tmp_path / "genes.npz", names=names, pcs=rows)
    sets = pd.DataFrame({"gene_name": [n for n, b in zip(names, block) if b == 0]
                         + ["G1_01", "G2_01", "G3_01"], "group": ["planted"] * 15 + ["scattered"] * 3})
    sets.to_csv(tmp_path / "planted.tsv", sep="\t", index=False)
    rng = np.random.default_rng(3)
    field = rng.integers(100, 4000, size=(400, 400, 2)).astype(np.uint16)
    np.save(tmp_path / "field.npy", field)
    cells = []
    k = 0
    for name, row, b in zip(names, rows, block):
        if b < 0:
            continue
        for j in range(4):
            k += 1
            cells.append({"cell_id": f"c{k:04d}", "gene": name, "field_path": str(tmp_path / "field.npy"),
                          "centroid_y": 50 + (k * 37) % 300, "centroid_x": 50 + (k * 53) % 300,
                          "cell_px": 500, **{f"e{d}": v for d, v in enumerate(
                              row + 0.05 * (j + 1) * rng.standard_normal(row.size))}})
    table = pd.DataFrame(cells)
    centre_cells = table.gene == "G0_00"
    table.loc[table.index[centre_cells][0], "centroid_y"] = 5
    table.loc[table.index[centre_cells][1], "cell_px"] = 10
    table.to_parquet(tmp_path / "cells.parquet")
    return table


def _settings(tmp_path, dst, **extra):
    """Item 918 settings over the planted inputs."""
    base = {"umap_source": "ops_profiles", "ops_umap_profiles": str(tmp_path / "genes.npz"),
            "ops_umap_label_sources": [f"PLANTED={tmp_path / 'planted.tsv'}", "GO_BP"],
            "ops_umap_reference_dir": str(tmp_path), "ops_umap_hierarchy_levels": [3, 6],
            "ops_umap_cell_embedding": str(tmp_path / "cells.parquet"),
            "ops_umap_crop_channels": {"0": "#1f4e79", "1": "#2ec4b6"}, "ops_umap_crop_px": 32,
            "ops_umap_stability_seeds": [42, 1], "ops_umap_dst": str(tmp_path / dst),
            "ops_umap_crop_level": "cluster", "black_background": False}
    base.update(extra)
    return base


def test_profile_clusters_recover_planted_groups_that_a_distorted_layout_does_not(leiden):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    names, rows, block = _blocks()
    labels, record = oiu.profile_clusters(names, rows, oiu.ops_umap_settings({}))
    assert adjusted_rand_score(block, labels) == 1.0
    assert record["source"] == "leiden" and leiden == [(42, 12.0)]
    distorted = np.random.default_rng(0).random((len(rows), 2))
    on_layout = KMeans(n_clusters=7, n_init=10, random_state=0).fit_predict(distorted)
    assert adjusted_rand_score(block, on_layout) < 0.2


def test_a_partition_file_is_used_as_given(tmp_path):
    names, rows, block = _blocks(n_blocks=2, size=4, controls=0)
    pd.DataFrame({"gene": names, "s42_r12": block + 5}).to_csv(tmp_path / "p.csv", index=False)
    labels, record = oiu.profile_clusters(names, rows,
                                          {"ops_umap_partition": f"{tmp_path / 'p.csv'}:s42_r12"})
    assert list(labels) == list(block + 5) and record["column"] == "s42_r12"
    with pytest.raises(ValueError, match="no 's42_r12'"):
        oiu.profile_clusters(np.append(names, "X"), rows,
                             {"ops_umap_partition": f"{tmp_path / 'p.csv'}:s42_r12"})


def test_the_planted_complex_is_labelled_and_the_shuffled_control_is_not():
    names, rows, block = _blocks()
    sets = pd.DataFrame({"source": "S", "term": "planted",
                         "gene": [n for n, b in zip(names, block) if b == 0]})
    table, chosen = oiu.enrich_clusters(names, block, sets)
    assert chosen[0][0] == "planted" and chosen[0][1] < 0.05
    assert chosen[1] == ("unlabelled", chosen[1][1]) and np.isnan(chosen[1][1])
    assert table.loc[table.label, "cluster"].tolist() == [0]
    assert -1 not in chosen
    assert oiu.shuffled_label_control(names, block, sets, seed=7) == 0
    fisher, chosen_f = oiu.enrich_clusters(names, block, sets, test="fisher_two_sided")
    from scipy.stats import fisher_exact, hypergeom

    assert chosen_f[0][0] == "planted"
    assert fisher.p.iloc[0] == fisher_exact([[15, 0], [0, 75]])[1]
    assert table.p.iloc[0] == hypergeom.sf(14, 90, 15, 15)
    with pytest.raises(ValueError, match="enrichment_test"):
        oiu.enrich_clusters(names, block, sets, test="t")


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(2).random(40) ** 3
    assert np.allclose(oiu._bh(p), multipletests(p, method="fdr_bh")[1])
    assert oiu._bh(np.array([])).size == 0
    assert np.allclose(oiu._bh(p, 100), multipletests(np.r_[p, np.ones(60)], method="fdr_bh")[1][:40])


def test_medoid_selection_returns_the_planted_central_gene():
    names, rows, block = _blocks(controls=0)
    for b in range(6):
        assert oiu.medoid_gene(names[block == b], rows[block == b]) == f"G{b}_00"
    assert oiu.medoid_gene(["b", "a"], np.ones((2, 3))) == "a"


def test_hierarchy_cuts_centroids_into_the_asked_levels():
    names, rows, block = _blocks(controls=0)
    rows = rows.copy()
    rows[block == 1] += np.eye(12)[0] * 0.9
    table, z = oiu.cluster_hierarchy(rows, block, levels=(5, 6))
    assert set(table.level) == {5, 6} and len(z) == 5
    five = table[table.level == 5].set_index("cluster").parent
    assert five[0] == five[1] and five.nunique() == 5
    assert table[table.level == 6].parent.nunique() == 6
    one, z1 = oiu.cluster_hierarchy(rows[block == 0], block[block == 0], levels=(3,))
    assert len(z1) == 0 and one.parent.tolist() == [1]


def test_ring_slots_follow_the_angle_of_each_group():
    points = {"e": (1, 0), "n": (0, 1), "w": (-1, 0), "s": (0, -1)}
    assert oiu.layout_crop_ring(points, (0, 0), n_slots=4) == {"e": 0, "n": 1, "w": 2, "s": 3}
    eight = oiu.layout_crop_ring(points, (0, 0), n_slots=8)
    assert eight == {"e": 0, "n": 2, "w": 4, "s": 6}
    assert oiu.layout_crop_ring({}, (0, 0)) == {}


def test_display_range_is_pooled_and_fixed_values_win():
    crops = [np.full((4, 4, 2), 10), np.full((4, 4, 2), 30)]
    ranges = oiu.shared_display_range(crops, [0, 1], percentiles=(0, 100), fixed={"1": (0, 5)})
    assert ranges == {0: (10.0, 30.0), 1: (0.0, 5.0)}
    rgb = oiu.composite(np.stack([np.full((2, 2), 20), np.zeros((2, 2))], -1),
                        {0: "#ffffff"}, {0: (10.0, 30.0)})
    assert np.allclose(rgb, 0.5)


def test_crops_are_raw_and_skipped_when_cut_by_the_field_edge(tmp_path):
    field = np.arange(100 * 100 * 2, dtype=np.uint16).reshape(100, 100, 2)
    np.save(tmp_path / "f.npy", field)
    crop = oiu.read_crop({"field_path": str(tmp_path / "f.npy"), "centroid_y": 50,
                          "centroid_x": 40}, 10)
    assert crop.dtype == np.uint16 and np.array_equal(crop, field[45:55, 35:45])
    assert oiu.read_crop({"field_path": str(tmp_path / "f.npy"), "centroid_y": 3,
                          "centroid_x": 40}, 10) is None
    np.save(tmp_path / "c.npy", field[:20, :20, 0])
    assert oiu.read_crop({"crop_path": str(tmp_path / "c.npy")}, 10).shape == (10, 10, 1)
    assert oiu.read_crop({"crop_path": str(tmp_path / "c.npy")}, 30) is None
    from matplotlib.image import imsave

    imsave(tmp_path / "p.png", np.zeros((12, 12, 3)))
    assert oiu.read_crop({"png_path": str(tmp_path / "p.png"), "crop_path": None}, 8).shape[:2] == (8, 8)


def test_the_run_writes_every_output_and_reruns_byte_identically(tmp_path, leiden):
    names, rows, block = _blocks()
    _inputs(tmp_path, names, rows, block)
    first = oiu.run_ops_image_umap(_settings(tmp_path, "a"))
    oiu.run_ops_image_umap(_settings(tmp_path, "b"))
    a, b = tmp_path / "a", tmp_path / "b"
    for name in ("umap_coords.csv", "clusters.csv", "cluster_hierarchy.csv",
                 "cluster_enrichment.csv", "cluster_medoids.csv", "crops_display.json",
                 "enrichment_control.json", "layout_stability.csv", "ops_umap.png"):
        assert filecmp.cmp(a / name, b / name, shallow=False), name
    for path in glob.glob(str(a / "crops" / "*")):
        assert filecmp.cmp(path, b / "crops" / os.path.basename(path), shallow=False)
    for ext in ("pdf", "svg"):
        assert os.path.getsize(a / f"ops_umap.{ext}") > 0
    assert first["clusters"] == 6 and first["labelled"] == 1 and first["labelled_shuffled"] == 0
    medoids = pd.read_csv(a / "cluster_medoids.csv")
    row0 = medoids[medoids.group == 0].iloc[0]
    assert row0.medoid_gene == "G0_00" and len(row0.cell_ids.split(";")) == 2
    centre_cells = pd.read_parquet(tmp_path / "cells.parquet").query("gene == 'G0_00'").cell_id
    assert set(row0.cell_ids.split(";")) == set(centre_cells.iloc[2:])
    display = json.loads((a / "crops_display.json").read_text())
    pooled = [np.load(p) for p in sorted(glob.glob(str(a / "crops" / "*.npy")))]
    for c in (0, 1):
        lo, hi = np.percentile(np.concatenate([p[..., c].ravel() for p in pooled]).astype(float),
                               [0.5, 99.5])
        assert display["channels"][str(c)]["low"] == lo and display["channels"][str(c)]["high"] == hi
    provenance = json.loads((a / "umap_provenance.json").read_text())
    assert provenance["versions"]["umap"] and provenance["n_jobs"] == 1
    assert provenance["label_sets"]["GO_BP"] == "missing"
    assert set(provenance["input_md5"]) == {"ops_umap_profiles", "ops_umap_cell_embedding"}
    assert "layouts differ" in provenance["note"]
    coords = pd.read_csv(a / "umap_coords.csv")
    assert coords.control.sum() == 6 and set(coords.kind) == {"gene"}


def test_changing_only_the_umap_seed_moves_the_layout_and_nothing_else(tmp_path, leiden):
    names, rows, block = _blocks()
    _inputs(tmp_path, names, rows, block)
    oiu.run_ops_image_umap(_settings(tmp_path, "a", ops_umap_stability_seeds=[42]))
    oiu.run_ops_image_umap(_settings(tmp_path, "b", random_seed=5, ops_umap_stability_seeds=[42],
                                     ops_umap_crop_level="top", black_background=True))
    a, b = tmp_path / "a", tmp_path / "b"
    assert not filecmp.cmp(a / "umap_coords.csv", b / "umap_coords.csv", shallow=False)
    for name in ("clusters.csv", "cluster_hierarchy.csv", "cluster_enrichment.csv"):
        assert filecmp.cmp(a / name, b / name, shallow=False), name
    top = pd.read_csv(b / "cluster_medoids.csv")
    hierarchy = pd.read_csv(b / "cluster_hierarchy.csv")
    parent = hierarchy[hierarchy.level == 3].set_index("cluster").parent
    assert len(top) == 3
    for row in top.itertuples():
        members = np.isin(block, parent.index[parent == row.group])
        assert row.medoid_gene == oiu.medoid_gene(names[members], rows[members])


def test_generate_image_umap_routes_the_ops_source_and_checks_its_inputs(tmp_path, leiden):
    from spacr.core import generate_image_umap
    from spacr.settings import set_default_umap_image_settings

    names, rows, block = _blocks(n_blocks=3, size=8, controls=2)
    np.savez(tmp_path / "genes.npz", names=names, pcs=rows)
    guides = pd.DataFrame(rows[:6] + 0.01, columns=[f"f{i}" for i in range(12)])
    guides.insert(0, "guide", [f"sg{i}" for i in range(6)])
    guides.to_parquet(tmp_path / "guides.parquet")
    defaults = set_default_umap_image_settings({"umap_source": "ops_profiles"})
    assert (defaults["n_neighbors"], defaults["min_dist"], defaults["metric"]) == (15, 0.3, "cosine")
    assert set_default_umap_image_settings({})["n_neighbors"] == 1000
    out = generate_image_umap({"umap_source": "ops_profiles",
                               "ops_umap_profiles": str(tmp_path / "genes.npz"),
                               "ops_umap_guide_profiles": str(tmp_path / "guides.parquet"),
                               "ops_umap_guides": "project", "ops_umap_label_sources": [],
                               "ops_umap_hierarchy_levels": [2], "ops_umap_stability_seeds": [],
                               "n_neighbors": 5})
    assert out["clusters"] == 3 and os.path.isfile(out["dst"] + "/ops_umap.png")
    coords = pd.read_csv(os.path.join(out["dst"], "umap_coords.csv"))
    assert (coords.kind == "guide").sum() == 6
    with pytest.raises(ValueError, match="needs ops_umap_profiles"):
        oiu.run_ops_image_umap({"umap_source": "ops_profiles"})
    with pytest.raises(ValueError, match="crop_channels"):
        oiu.run_ops_image_umap({"ops_umap_profiles": "x.npz", "ops_umap_cell_embedding": "c"})


def test_layouts_and_readers_refuse_what_they_cannot_use(tmp_path):
    names, rows, _ = _blocks(n_blocks=2, size=10, controls=0)
    with pytest.raises(ValueError, match="ops_umap_guides"):
        oiu.umap_layout(rows, mode="sideways")
    genes, guides = oiu.umap_layout(rows[:14], n_neighbors=5, extra=rows[14:], mode="joint")
    assert genes.shape == (14, 2) and guides.shape == (6, 2) and genes.dtype == np.float32
    pd.DataFrame(rows).assign(name=names).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match="no 'gene'"):
        oiu.read_profiles(str(tmp_path / "t.csv"))
    pd.DataFrame({"gene": ["a"]}).to_csv(tmp_path / "cells.csv", index=False)
    with pytest.raises(ValueError, match="cell_id"):
        oiu.read_cell_table(str(tmp_path / "cells.csv"))
    cells = pd.DataFrame({"cell_id": ["1"], "gene": [names[0]], "e0": [0.0]})
    with pytest.raises(ValueError, match="dimensions"):
        oiu.medoid_cells({0: list(names)}, names, rows, cells, np.zeros((1, 1)))
    empty = oiu.medoid_cells({0: ["nope"]}, names, rows, cells, np.zeros((1, 12)))
    assert empty.empty


def test_label_sets_read_corum_halves_named_files_and_report_missing(tmp_path):
    from spacr.ops_reference import clear_cache, group_halves

    groups = {f"complex{i}": [f"G{i}", f"H{i}"] for i in range(6)}
    pd.DataFrame([(m, g) for g, ms in groups.items() for m in ms],
                 columns=["gene_name", "group"]).to_csv(tmp_path / "corum_group_benchmark.tsv",
                                                        sep="\t", index=False)
    pd.DataFrame({"gene_name": ["A", "B"], "group": ["go1", "go1"]}).to_csv(
        tmp_path / "GO_BP.tsv", sep="\t", index=False)
    clear_cache()
    sets, record = oiu.label_sets(["CORUM@REPORT", "CORUM", "GO_BP", "KEGG"], str(tmp_path))
    halves = group_halves(groups)
    report = {g for g, h in halves.items() if h == 1}
    assert set(sets[sets.source == "CORUM@REPORT"].term) == report
    assert set(sets[sets.source == "CORUM"].term) == set(groups)
    assert record["GO_BP"]["terms"] == 1 and len(record["GO_BP"]["md5"]) == 32
    assert record["KEGG"] == "missing"
    none, missing = oiu.label_sets(["CORUM"], None)
    assert none.empty and missing == {"CORUM": "missing"}
    clear_cache()


def test_stability_reports_trustworthiness_per_seed():
    names, rows, block = _blocks(n_blocks=3, size=10, controls=0)
    s = oiu.ops_umap_settings({"n_neighbors": 5})
    table = oiu.layout_stability(rows, block, s, [42, 1], k=5)
    assert table.seed.tolist() == [42, 1]
    assert table.trustworthiness.between(0, 1).all()
    assert (table.same_cluster_profile == 1.0).all()


def test_library_versions_name_missing_packages_as_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "leidenalg", None)
    versions = oiu.library_versions()
    assert versions["leidenalg"] is None and versions["numpy"] == np.__version__


def test_preflight_checks_the_profile_files_not_src(tmp_path):
    from spacr.validate import validate_settings

    np.savez(tmp_path / "g.npz", names=np.array(["a"]), pcs=np.zeros((1, 2)))
    ok = validate_settings({"umap_source": "ops_profiles", "ops_umap_profiles": str(tmp_path / "g.npz")},
                           "umap")
    assert not [p for p in ok if p.severity == "error"]
    bad = validate_settings({"umap_source": "ops_profiles",
                             "ops_umap_cell_embedding": str(tmp_path / "nope.parquet")}, "umap")
    errors = {p.setting for p in bad if p.severity == "error"}
    assert errors == {"ops_umap_profiles", "ops_umap_cell_embedding"}
