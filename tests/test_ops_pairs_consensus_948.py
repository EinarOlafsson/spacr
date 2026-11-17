"""The gene-pairs consensus mode of item 948 (alpha): SNF of the profile with interphase and mitosis views, then
PHATE and Leiden, over guide bootstraps.

The reference values in ``tests/data/ops_pairs_948_fixture.json`` were computed by the analysis scripts the feature
reproduces (Design F's ``f_cells.snf_fuse`` and ``f_cells.phase_view``, and the speed card's single-precision fusion
``ps_lib.snf_fuse32``) by ``spacr_948_pairs_proof/scripts/make_fixture948.py``. PHATE, igraph and leidenalg are the
optional ``clustering`` extra; tests that need them use the item 916 stand-ins or skip without the real extra.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import ops_clusters as oc
from spacr.ops_settings import ops_defaults

from tests.test_ops_leiden_sweep_916 import CALLS, _install

FIXTURE = json.loads((Path(__file__).resolve().parent / "data" / "ops_pairs_948_fixture.json")
                     .read_text(encoding="utf-8"))


def _fixture_cells():
    """The fixture's toy cell table with its five value columns as PC_0 .. PC_4."""
    cells = pd.DataFrame(FIXTURE["cells"])
    values = np.asarray(FIXTURE["cell_values"], np.float32)
    return pd.concat([cells, pd.DataFrame(values, columns=[f"PC_{i}" for i in range(values.shape[1])])], axis=1)


def _fixture_phases():
    """The phase medians and guides :func:`spacr.ops_clusters._pairs_inputs` builds from the fixture cells."""
    cells = _fixture_cells()
    data = {"order": np.array(FIXTURE["rows"]), "prefix": "nontargeting"}
    return cells, data


def test_double_precision_fusion_equals_the_published_script():
    views = [np.array(v) for v in FIXTURE["views"]]
    assert np.array_equal(oc.snf_fuse(views), np.array(FIXTURE["fused64"]))


def test_single_precision_fusion_equals_the_speed_card_and_stays_close():
    views = [np.array(v) for v in FIXTURE["views"]]
    fused = oc.snf_fuse(views, single=True)
    assert np.array_equal(fused, np.array(FIXTURE["fused32"]))
    reference = np.array(FIXTURE["fused64"])
    assert np.abs(fused - reference).max() <= 1e-5 * np.abs(reference).max()


def test_the_fused_network_is_symmetric_and_double_precision():
    views = [np.array(v) for v in FIXTURE["views"]]
    for single in (False, True):
        fused = oc.snf_fuse(views, single=single)
        assert fused.dtype == np.float64 and np.array_equal(fused, fused.T)


def test_phase_views_equal_the_published_script(tmp_path):
    cells, data = _fixture_phases()
    path = tmp_path / "cells.parquet"
    cells.to_parquet(path, index=False)
    phases = oc._pairs_inputs({"ops_consensus_cells": str(path)}, data, 5)
    mask = np.array(FIXTURE["mask"])
    for phase, key in (("I", "view_I"), ("M", "view_M")):
        got = oc.phase_view(data["order"], mask, phases["guides"], phases["medians"][phase], FIXTURE["view_seed"], 5)
        assert np.array_equal(got, np.array(FIXTURE[key]))


def test_a_row_without_cells_in_a_phase_is_zero_and_controls_centre_the_view():
    medians = {"A|a1": (np.array([2.0, 4.0]), 3), "nontargeting_x|x": (np.array([1.0, 1.0]), 2),
               "nontargeting_y|y": (np.array([3.0, 5.0]), 2)}
    names = np.array(["A", "B", "nontargeting_x", "nontargeting_y"])
    guides = {"A": ["a1"], "B": ["b1"], "nontargeting_x": ["x"], "nontargeting_y": ["y"]}
    view = oc.phase_view(names, np.array([True, True, False, False]), guides, medians, None, 2)
    assert np.array_equal(view[1], [0.0, 0.0])
    assert np.allclose(np.median(view[2:], 0), 0.0)


def test_the_cells_table_is_required_and_checked(tmp_path):
    data = {"order": np.array(["A"]), "prefix": "nontargeting"}
    with pytest.raises(ValueError, match="ops_consensus_cells"):
        oc._pairs_inputs({}, data, 2)
    pd.DataFrame({"gene": ["A"], "sgRNA": ["a"], "mitotic": [False], "PC_0": [1.0]}).to_parquet(tmp_path / "c.parquet")
    with pytest.raises(ValueError, match="PC_1"):
        oc._pairs_inputs({"ops_consensus_cells": str(tmp_path / "c.parquet")}, data, 2)


def test_the_settings_default_to_the_928_consensus():
    settings = ops_defaults({})
    assert settings["ops_consensus_mode"] == "profile" and settings["ops_consensus_pairs_search"] == "fast"
    assert settings["ops_consensus_workers"] == 1 and settings["ops_consensus_cells"] is None
    options = oc._consensus_options(settings, 100)
    assert options.mode == "profile"
    with pytest.raises(ValueError, match="ops_consensus_mode"):
        oc._consensus_options({**settings, "ops_consensus_mode": "fusion"}, 100)
    with pytest.raises(ValueError, match="ops_consensus_pairs_search"):
        oc._consensus_options({**settings, "ops_consensus_pairs_search": "quick"}, 100)


def _blocks(n_blocks=3, size=6, guides=3, dims=6, seed=0):
    """Planted modules: gene rows, guide rows, the per-cell table and the control rows."""
    rng = np.random.default_rng(seed)
    names, rows, g_rows, g_genes, cells, block = [], [], [], [], [], []
    for b in range(n_blocks):
        centre = np.zeros(dims)
        centre[b] = 1.0
        for i in range(size):
            gene = f"G{b}_{i}"
            members = centre + 0.02 * rng.standard_normal((guides, dims))
            names.append(gene)
            rows.append(members.mean(0))
            block.append(b)
            for j, guide in enumerate(members):
                g_rows.append(guide)
                g_genes.append(gene)
                for _ in range(4):
                    cells.append({"gene": gene, "sgRNA": f"{gene}_{j}", "mitotic": bool(rng.random() < 0.3),
                                  **{f"PC_{d}": float(v) for d, v in enumerate(guide + 0.01 * rng.standard_normal(dims))}})
    for c in range(2):
        name = f"nontargeting_c{c}"
        names.append(name)
        rows.append(0.01 * rng.standard_normal(dims))
        block.append(-1)
        for _ in range(4):
            cells.append({"gene": "nontargeting", "sgRNA": f"c{c}", "mitotic": bool(rng.random() < 0.3),
                          **{f"PC_{d}": float(v) for d, v in enumerate(0.01 * rng.standard_normal(dims))}})
    return (np.array(names), np.array(rows), np.array(g_rows), np.array(g_genes), np.full(len(g_genes), 4.0),
            pd.DataFrame(cells), np.array(block))


def _pairs_run(tmp_path, monkeypatch, search, workers, n_boot=4):
    """One pairs consensus on the planted modules with the stand-in extra."""
    _install(monkeypatch)
    names, rows, g_rows, g_genes, g_cells, cells, block = _blocks()
    path = tmp_path / "cells.parquet"
    cells.to_parquet(path, index=False)
    mask = block >= 0
    phases = oc._pairs_inputs({"ops_consensus_cells": str(path)}, {"order": names, "prefix": "nontargeting"}, 6)
    views = partial(oc._pairs_views, order=names, mask=mask, phases=phases, draw_seed=8000)
    return oc.pairs_bootstrap_consensus(
        lambda b: oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, g_cells, 8000 + b), views, n_boot, 3,
        mask, seed=100, rows=rows, search=search, workers=workers), block


def test_the_fast_search_searches_once_then_runs_short_passes(tmp_path, monkeypatch):
    out, block = _pairs_run(tmp_path, monkeypatch, "fast", 1)
    passes = [call["n_iterations"] for call in CALLS]
    assert passes[-3:] == [oc._PAIRS_FAST_PASSES] * 3
    assert -1 in passes and len({r["resolution"] for r in out["runs"]}) == 1
    assert len(np.unique(out["labels"][block >= 0])) == 3


def test_the_exact_search_runs_every_bootstrap_to_the_end(tmp_path, monkeypatch):
    out, block = _pairs_run(tmp_path, monkeypatch, "exact", 1)
    assert {call["n_iterations"] for call in CALLS} == {-1}
    assert len(out["runs"]) == 4 and len(out["labels"]) == len(block)


def test_worker_processes_give_byte_identical_results(tmp_path, monkeypatch):
    serial, _ = _pairs_run(tmp_path, monkeypatch, "fast", 1)
    pooled, _ = _pairs_run(tmp_path, monkeypatch, "fast", 3)
    assert np.array_equal(serial["labels"], pooled["labels"])
    assert serial["cooc"].tobytes() == pooled["cooc"].tobytes()


def test_the_short_pass_search_ends_with_one_full_run(monkeypatch):
    _install(monkeypatch)
    graph = oc.gene_graph(np.eye(4), "knn", k=1)
    oc.match_resolution(graph, 2, 5, np.ones(4, bool), passes=2)
    assert [call["n_iterations"] for call in CALLS][-1] == -1
    assert all(call["n_iterations"] == 2 for call in CALLS[:-1])
    CALLS.clear()
    oc.match_resolution(graph, 2, 5, np.ones(4, bool))
    assert {call["n_iterations"] for call in CALLS} == {-1}


def test_the_phase_runs_the_pairs_mode_and_writes_the_928_outputs(tmp_path, monkeypatch):
    _install(monkeypatch)
    names, rows, g_rows, g_genes, g_cells, cells, block = _blocks()
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    columns = [f"PC_{d}" for d in range(rows.shape[1])]
    pd.concat([pd.DataFrame({"gene": names, "n_cells": 12}), pd.DataFrame(rows, columns=columns)], axis=1).to_csv(
        tmp_path / "rows.csv", index=False)
    pd.concat([pd.DataFrame({"gene": g_genes, "n_cells": g_cells}), pd.DataFrame(g_rows, columns=columns)],
              axis=1).to_csv(tmp_path / "guides.csv", index=False)
    settings = ops_defaults({"ops_cluster_rows": str(tmp_path / "rows.csv"),
                             "ops_consensus_guides": str(tmp_path / "guides.csv"), "ops_consensus_mode": "pairs",
                             "ops_consensus_cells": str(tmp_path / "cells.parquet"), "ops_consensus_view_columns": 6,
                             "ops_consensus_bootstraps": 3, "ops_consensus_k": 3})
    out = oc.cluster_consensus_phase(settings, None, None, str(tmp_path))
    partition = pd.read_csv(out["outputs"]["consensus_partition"])
    assert len(partition) == len(names) and not partition["unassigned"].any()
    summary = json.loads(Path(out["outputs"]["consensus_summary"]).read_text(encoding="utf-8"))
    assert summary["options"]["mode"] == "pairs" and out["clusters"] == 3


def test_the_default_profile_summary_has_no_pairs_keys(tmp_path, monkeypatch):
    """In the default 'profile' mode the summary keeps the 928 options, so its bytes match the 928 output."""
    _install(monkeypatch)
    names, rows, g_rows, g_genes, g_cells, _, _ = _blocks()
    columns = [f"PC_{d}" for d in range(rows.shape[1])]
    pd.concat([pd.DataFrame({"gene": names, "n_cells": 12}), pd.DataFrame(rows, columns=columns)], axis=1).to_csv(
        tmp_path / "rows.csv", index=False)
    pd.concat([pd.DataFrame({"gene": g_genes, "n_cells": g_cells}), pd.DataFrame(g_rows, columns=columns)],
              axis=1).to_csv(tmp_path / "guides.csv", index=False)
    settings = ops_defaults({"ops_cluster_rows": str(tmp_path / "rows.csv"),
                             "ops_consensus_guides": str(tmp_path / "guides.csv"),
                             "ops_consensus_bootstraps": 3, "ops_consensus_k": 3})
    out = oc.cluster_consensus_phase(settings, None, None, str(tmp_path))
    summary = json.loads(Path(out["outputs"]["consensus_summary"]).read_text(encoding="utf-8"))
    assert not {"mode", "pairs_search", "workers", "view_columns"} & summary["options"].keys()
    assert {"n_boot", "seed", "draw_seed", "knn", "method", "k", "match"} <= summary["options"].keys()


def test_real_pairs_consensus_is_deterministic_and_recovers_modules(tmp_path):
    pytest.importorskip("leidenalg")
    pytest.importorskip("phate")
    names, rows, g_rows, g_genes, g_cells, cells, block = _blocks(n_blocks=4, size=8, seed=4)
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    mask = block >= 0
    phases = oc._pairs_inputs({"ops_consensus_cells": str(tmp_path / "cells.parquet")},
                              {"order": names, "prefix": "nontargeting"}, 6)
    views = partial(oc._pairs_views, order=names, mask=mask, phases=phases, draw_seed=8000)

    def run(search, workers=1):
        """One real pairs consensus."""
        return oc.pairs_bootstrap_consensus(
            lambda b: oc.guide_bootstrap_profiles(rows, names, g_rows, g_genes, g_cells, 8000 + b), views, 4, 4,
            mask, seed=100, rows=rows, search=search, workers=workers)

    from sklearn.metrics import adjusted_rand_score

    fast, again, pooled, exact = run("fast"), run("fast"), run("fast", 2), run("exact")
    assert np.array_equal(fast["labels"], again["labels"]) and np.array_equal(fast["labels"], pooled["labels"])
    for out in (fast, exact):
        assert adjusted_rand_score(block[mask], out["labels"][mask]) >= 0.8
