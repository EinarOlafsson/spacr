"""The power module writes a simulated screen straight into the store.

The in-memory simulation keeps its outputs for a seed (a recorded hash of a
nominal calibration over four designs and two seeds); the streamed writer
puts the same model into :func:`spacr.store.writer` well by well, with the
recorded dtype map, a fixed seed and the plate, well, field, gene, guide and
key structure of an OPS run.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import duckdb

from spacr import power_ops, store

#: sha256 of the screens below, recorded before the model was split into the
#: helpers the streamed writer shares with it.
_GOLDEN = "cf164e5b302ec51d1eb70b958f981a2836642d60852b4efc64eb932ca23468c7"


def _golden_calibration():
    rng = np.random.default_rng(5)
    dims = 12
    a = rng.normal(size=(dims, dims))
    cov = a @ a.T / dims + np.eye(dims)
    guides = np.array([f"nontargeting_{i % 6}" for i in range(3000)])
    return {"dims": dims, "nt_mean": np.zeros(dims).tolist(), "nt_cov": cov.tolist(),
            "between_guide_var": (np.ones(dims) * 0.05).tolist(),
            "between_well_var": (np.ones(dims) * 0.03).tolist(),
            "between_site_var": (np.ones(dims) * 0.02).tolist(), "t_dof": 8.0, "t_scale_factor": 0.9,
            "n_genes": 40, "guides_per_gene": {"median": 4}, "nt_guides": 6, "cells_per_gene": {"q50": 60},
            "call_rate": {"value": 0.8}, "fdr_pct": {"value": 2.0}, "wells": ["A1", "A2", "B1"],
            "guide_alpha": 50.0, "nt_ratio": 0.05, "real_etest": {"hit_norm2": [1.0, 2.0, 3.5]},
            "complexes": {"sizes": [2, 3, 4]},
            "_pool": (rng.normal(size=(3000, dims)) @ np.linalg.cholesky(cov).T, guides)}


def test_the_in_memory_simulation_keeps_its_outputs_for_a_seed():
    calibration = _golden_calibration()
    digest = hashlib.sha256()
    for design in ({}, {"mode": "parametric", "fields_per_well": 4}, {"effect_type": "variance", "fdr_pct": 5.0},
                   {"effect_type": "subpopulation", "effect_size": "calibrated", "ko_penetrance": 0.8,
                    "ko_penetrance_sd": 0.1, "inactive_guide_fraction": 0.1, "effect_block": "new"}):
        for seed in (0, 3):
            screen = power_ops.simulate_ops_screen(calibration, design, seed)
            for key in ("cells", "labels", "guides", "wells"):
                digest.update(np.ascontiguousarray(screen[key]).tobytes())
            digest.update(screen["truth"].to_csv().encode())
    assert digest.hexdigest() == _GOLDEN


def _screen(project, **kwargs):
    calibration = power_ops.nominal_calibration(dims=8, n_genes=30, nt_guides=4, pool_cells=2000, seed=1)
    shape = dict(plates=2, wells=3, fields=3, cells_per_field=40, n_features=60)
    shape.update(kwargs)
    made = power_ops.write_ops_screen(calibration, project, {"cells_per_gene": 20}, 7, **shape)
    store.register(project)
    return made


def _digests(project):
    conn = duckdb.connect(os.path.join(project, store.CATALOG), read_only=True)
    try:
        return conn.execute("SELECT path, content_digest FROM spacr_files ORDER BY path").fetchall()
    finally:
        conn.close()


def test_a_streamed_screen_has_the_plate_well_field_gene_guide_key_structure(tmp_path):
    project = str(tmp_path / "p")
    made = _screen(project)
    assert made["cells"] == 2 * 3 * 3 * 40 and made["features"] == 60
    with store.open(project) as s:
        assert s.count("cell") == s.count("nucleus") == s.count("cytoplasm") == 720
        cell = s.query(table="cell").sort_values(["plate", "well", "object_key"], ignore_index=True)
        nucleus = s.query(table="nucleus").sort_values(["plate", "well", "object_key"], ignore_index=True)
        assert (cell["object_key"] == nucleus["object_key"]).all()
        assert (nucleus["parent_cell_key"] == nucleus["object_key"]).all()
        assert set(cell["plate"]) == {"plate01", "plate02"} and set(cell["well"]) == {"A01", "A02", "A03"}
        assert set(cell["fieldID"]) == {1, 2, 3} and cell["object_label"].max() == 40
        assert cell.groupby("plate")["object_key"].apply(lambda k: k.is_unique).all()
        assert cell["gene"].str.startswith(("G", "nontargeting")).all()
        assert cell["guide"].str.match(r"^(G\d{5}_\d|NT\d+)$").all()
        columns = s.columns("cell").set_index("name")
        assert {"halffloat", "float"} <= set(columns["stored_type"])
        assert all(dtype != np.float16 for dtype in cell.dtypes)


def test_a_seed_gives_the_same_bytes_and_a_well_subset_the_same_wells(tmp_path):
    first, second, subset = (str(tmp_path / n) for n in ("a", "b", "c"))
    _screen(first)
    _screen(second)
    assert _digests(first) == _digests(second)
    made = _screen(subset, only=["plate02/A02"])
    assert made["wells"] == [["plate02", "A02"]]
    whole = dict(_digests(first))
    part = dict(_digests(subset))
    assert part and all(whole[path] == digest for path, digest in part.items())


def test_batches_bound_the_memory_and_keep_the_keys_ordered(tmp_path):
    project = str(tmp_path / "p")
    _screen(project, plates=1, wells=1, fields=30, cells_per_field=100, batch_cells=1000)
    with store.open(project) as s:
        keys = s.query(table="cell")["object_key"].to_numpy()
    assert len(keys) == 3000 and (np.diff(keys) > 0).all()


def test_a_nominal_calibration_has_every_key_the_simulation_reads():
    calibration = power_ops.nominal_calibration(dims=6, n_genes=10, nt_guides=3, pool_cells=500)
    screen = power_ops.simulate_ops_screen(calibration, {"cells_per_gene": 5}, 0)
    assert screen["cells"].shape[1] == 6 and len(screen["truth"]) == 10
