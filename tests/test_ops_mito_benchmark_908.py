"""Item 908: the MitoCarta and external-module benchmark of OPS clusterings.

A small screen with a planted mitochondrial cluster: its clusterings, profile
cosines and consensus atlas are written to disk, and the mitocarta phase
is run from settings. The plate A identity check against the improved
analysis' mito_*.csv tables ran on the stored arms; see the item record.
"""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_clusters as oc

MITO = [f"M{i:02d}" for i in range(12)]
OTHER = [f"G{i:03d}" for i in range(48)]
GENES = MITO + OTHER


def _mitocarta(tmp_path):
    """A MitoCarta-shaped table: 12 mitochondrial genes in three sub-compartments."""
    paths = (["OXPHOS > Complex I"] * 4 + ["Mitochondrial ribosome"] * 4
             + ["OXPHOS > Complex V"] * 2 + ["Mitochondrial dynamics"] * 2)
    table = pd.DataFrame({"Symbol": MITO, "MitoCarta3.0_MitoPathways": paths})
    path = tmp_path / "mitocarta.csv"
    table.to_csv(path, index=False)
    return path


def _arm(tmp_path, name, rng, mixed=False):
    """Assignments, profiles and a consensus folder for one arm."""
    clusters = np.array([0] * 12 + list(1 + np.arange(48) % 6))
    if mixed:
        clusters = rng.permutation(clusters)
    names = GENES + ["nontargeting_1", "nontargeting_2"]
    table = pd.DataFrame({"gene": names, "leiden_r3": list(clusters % 3) + [0, 1],
                          "leiden_r10": list(clusters) + [1, 2]})
    assignments = tmp_path / f"{name}_assignments.csv"
    table.to_csv(assignments, index=False)
    pcs = rng.normal(0, 1, (len(names), 8))
    if not mixed:
        pcs[:4] += [6, 0, 0, 0, 0, 0, 0, 0]
        pcs[4:8] += [0, 6, 0, 0, 0, 0, 0, 0]
    profiles = tmp_path / f"{name}.npz"
    np.savez(profiles, names=np.array(names), pcs=pcs)
    consensus = tmp_path / f"{name}_consensus"
    consensus.mkdir()
    rows = pd.DataFrame({"gene": GENES + ["NTpseudo_00"], "n_cells": list(20 + (np.arange(61) * 37 % 61) * 3)})
    rows.to_parquet(consensus / "rows_reference.parquet", index=False)
    runs = np.vstack([np.where(rng.random(61) < 0.1, -1, np.append(clusters, 9)) for _ in range(20)])
    np.savez_compressed(consensus / "consensus_partitions.npz", labels=runs, genes=rows.gene.to_numpy())
    np.savez(consensus / "reference.npz", r10=np.append(clusters, 9))
    return {"label": name, "assignments": str(assignments), "profiles": str(profiles), "atlas": str(consensus)}


def _external(tmp_path):
    """External modules, a published clustering and one named cluster."""
    modules = pd.DataFrame({"cluster_id": [5] * 6 + [6] * 4 + [7] * 2,
                            "gene_symbol": MITO[:6] + ["G000", "G001", "G002", "G003"] + ["M00", "G999"]})
    modules.to_csv(tmp_path / "modules.csv", index=False)
    published = pd.DataFrame({"gene_symbol": GENES, "cluster_id": [0] * 12 + [1] * 48})
    published.to_csv(tmp_path / "published.csv", index=False)
    named = pd.DataFrame({"cluster_id": [147] * 5 + [3], "gene_symbol": ["M00", "M01", "M02", "G010", "G011", "G020"],
                          "in_mitocarta": [True, True, True, False, False, False]})
    named.to_csv(tmp_path / "named.csv", index=False)
    spec = {"target_clusters": 12,
            "modules": {"name": "Ext", "table": str(tmp_path / "modules.csv"), "clusters": [5, 6, 7]},
            "published": [{"label": "published", "table": str(tmp_path / "published.csv")},
                          {"label": "own modules", "table": str(tmp_path / "published.csv"), "score_modules": False}],
            "named_clusters": [{"key": "c147", "title": "Cluster 147", "short": "147",
                                "table": str(tmp_path / "named.csv"), "id": 147, "members_flag": "in_mitocarta",
                                "members_name": "OXPHOS", "partners": ["G010", "G011"],
                                "partners_name": "partner pair", "probe": "M00"}]}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(spec))
    return path


def test_read_mitocarta_splits_the_subcompartments(tmp_path):
    genes, parts = oc.read_mitocarta(str(_mitocarta(tmp_path)))
    assert genes == set(MITO)
    assert parts["OXPHOS"] == set(MITO[:4] + MITO[8:10])
    assert parts["mitoribosome"] == set(MITO[4:8])
    assert parts["ATP synthase (complex V)"] == set(MITO[8:10])
    assert parts["membrane organisation (dynamics, cristae, phospholipid)"] == set(MITO[10:12])


def test_enrichment_finds_the_planted_mitochondrial_cluster():
    assignments = dict(zip(GENES, [0] * 12 + list(1 + np.arange(48) % 6)))
    out = oc.mitocarta_enrichment(assignments, set(MITO))
    assert out["mito_enriched_clusters"] == 1 and out["mito_genes_in_enriched"] == 12
    assert out["purity_in_enriched"] == 1.0 and out["share_of_mitocarta_recovered"] == 1.0
    assert out["genes"] == 60 and out["clusters"] == 7
    rng = np.random.default_rng(0)
    scattered = dict(zip(GENES, rng.permutation([0] * 12 + list(1 + np.arange(48) % 6))))
    assert oc.mitocarta_enrichment(scattered, set(MITO))["mito_enriched_clusters"] == 0


def test_subcompartment_retrieval_beats_its_random_expectation():
    rng = np.random.default_rng(1)
    names = np.array(GENES + ["nontargeting_1"])
    x = rng.normal(0, 1, (len(names), 6))
    x[:4] += [8, 0, 0, 0, 0, 0]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    parts = {"OXPHOS": set(MITO[:4]), "mitoribosome": set(MITO[4:8]), "none": {"G000"}}
    rows = oc.subcompartment_retrieval("arm", names, (x @ x.T).astype(np.float32), set(MITO), parts)
    by = {r["sub_compartment"]: r for r in rows}
    assert set(by) == {"OXPHOS", "mitoribosome"}
    assert by["OXPHOS"]["mAP"] == pytest.approx(1.0) and by["OXPHOS"]["queries"] == 4
    assert by["OXPHOS"]["expected_random"] < 0.5 and by["OXPHOS"]["candidates (MitoCarta genes)"] == 12
    assert by["mitoribosome"]["mAP"] < by["OXPHOS"]["mAP"]


def test_module_coherence_against_fitness_matched_sets(tmp_path):
    rng = np.random.default_rng(2)
    arm = _arm(tmp_path, "tight", rng)
    out = oc.module_coherence(arm["atlas"], set(MITO[:6]), np.random.default_rng(0), n_null=50)
    assert out["present"] == 6 and out["comembership"] == 1.0
    assert out["recovered (above null p99)"] and out["percentile"] == 1.0
    assert out["largest_fraction_in_one_~250_cluster"] == 1.0
    assert oc.module_coherence(arm["atlas"], {"M00", "nothere"}, rng) == {"present": 1}
    assert oc.module_coherence(str(tmp_path / "none"), set(MITO), rng) is None
    again = oc.module_coherence(arm["atlas"], list(reversed(MITO[:6])), np.random.default_rng(0), n_null=50)
    assert again == out


def test_the_mitocarta_phase_runs_from_settings(tmp_path):
    from spacr.ops_engine import run_ops

    rng = np.random.default_rng(3)
    arms = [_arm(tmp_path, "tight", rng), dict(_arm(tmp_path, "mixed", rng, mixed=True), own_genes_only=True),
            {"label": "empty"}]
    (tmp_path / "arms.json").write_text(json.dumps(arms))
    settings = {"dst_root": str(tmp_path / "dst"), "ops_eval_mitocarta": str(_mitocarta(tmp_path)),
                "ops_eval_external_modules": str(_external(tmp_path)), "ops_eval_arms": str(tmp_path / "arms.json")}
    out = run_ops(settings, phases=("mitocarta",))["mitocarta"]
    folder = tmp_path / "dst" / "ops_eval" / "mitocarta"
    for name in ("mito_A_mitocarta_recovery.csv", "mito_B_subcompartment_retrieval.csv", "mito_C_submodules.csv",
                 "mito_D_c147.csv", "mito_comparison.csv"):
        assert (folder / name).is_file()
    a = out["tables"]["A"]
    assert list(a.set) == ["published", "published", "own modules", "own modules", "tight", "tight", "mixed"]
    assert a.scope.iloc[1] == "common genes (60)" and a.leiden_resolution.iloc[4] == 10
    tight = a[(a.set == "tight") & (a.scope == "own genes")].iloc[0]
    assert tight.mito_enriched_clusters == 1 and tight.mito_genes_in_enriched == 12
    c = out["tables"]["C"]
    assert list(c.columns[:3]) == ["arm", "ext_cluster", "module_genes"]
    row = c[(c.arm == "tight") & (c.ext_cluster == 5)].iloc[0]
    assert row["atlas recovered (above null p99)"] and row["present_in_arm"] == 6
    assert set(c.arm) == {"tight", "mixed", "empty", "published"}
    assert c[(c.arm == "tight") & (c.ext_cluster == 7)].iloc[0]["atlas present"] == 1
    d = out["tables"]["D_c147"]
    assert list(d.arm) == ["tight", "mixed", "empty"]
    first = d.iloc[0]
    assert first["c147_oxphos_members"] == 3 and first["partner pair clusters (~12)"] == 2
    assert first["147 OXPHOS members sharing a cluster with the partner pair"] == 0
    assert first["M00 cluster size"] == 12 and first["M00 cluster members from Cluster 147"] == "M00, M01, M02"
    assert "cosine partner pair vs 147 OXPHOS" in d and "random p99" in d
    comp = out["tables"]["comparison"]
    assert list(comp.set) == ["published", "own modules", "tight", "mixed"]
    assert comp.set_index("set").at["tight", "Ext sub-modules coherent in atlas (of 3)"] >= 1
    assert "sub-compartment mAP: OXPHOS" in comp
    again = run_ops(settings, phases=("mitocarta",))["mitocarta"]["tables"]
    pd.testing.assert_frame_equal(again["C"], c)


def test_the_mitocarta_phase_refuses_missing_settings(tmp_path):
    with pytest.raises(ValueError, match="ops_eval_mitocarta"):
        oc.run_mitocarta_benchmark({"dst_root": str(tmp_path)})
    bad = tmp_path / "x.csv"
    pd.DataFrame({"gene": ["a"], "cluster": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="gene_symbol"):
        oc._external_clusters({"table": str(bad)})
    with pytest.raises(ValueError, match="leiden_r"):
        oc._arm_assignments(str(bad), 250)


@pytest.mark.parametrize("pin, expected", [(True, "threadpoolctl"), (False, None)])
def test_the_atlas_phases_honour_ops_deterministic_blas(monkeypatch, tmp_path, pin, expected):
    from threadpoolctl import threadpool_info

    from spacr import ops_modules
    from spacr.ops_engine import run_ops

    seen = {}

    def fake(name):
        """A stand-in phase that records the BLAS pool sizes it ran under."""
        def run(settings, report=None):
            """Record the pool sizes and return the phase name."""
            seen[name] = {pool["num_threads"] for pool in threadpool_info()}
            return {"phase": name}
        return run

    monkeypatch.setattr(ops_modules, "run_module_phase", fake("modules"))
    monkeypatch.setattr(oc, "run_mitocarta_benchmark", fake("mitocarta"))
    out = run_ops({"dst_root": str(tmp_path), "ops_deterministic_blas": pin},
                  phases=("modules", "mitocarta"))
    assert out["modules"] == {"phase": "modules"} and out["mitocarta"] == {"phase": "mitocarta"}
    assert out["atlas_blas_threads"] == expected
    if pin:
        assert max(seen["modules"]) <= 2 and max(seen["mitocarta"]) <= 2
