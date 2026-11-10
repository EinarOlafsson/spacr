"""Item 891: penetrance, escapers, phase tests and neighbour effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (
    ashman_d, disk_offsets, escaper_classes, escaper_statistics, field_folds,
    mask_neighbours, neighbour_effects, penetrance_tests, phase_tests,
    storey_penetrance)

DIMS = 4


def _screen(seed=0, sites=20, per_site=120):
    """Two wells of fields; gene HALF has a +4 shift in half of its cells,
    gene ALL in every cell, gene NONE in none; eight control guides."""
    rng = np.random.default_rng(seed)
    rows = []
    nt = [f"NT{k}" for k in range(8)]
    for well in ("A1", "A2"):
        for site in range(sites):
            for _ in range(per_site):
                kind = rng.choice(["nt", "HALF", "ALL", "NONE"], p=[0.55, 0.15, 0.15, 0.15])
                guide = rng.choice(nt) if kind == "nt" else f"{kind}_{rng.integers(0, 3)}"
                x = rng.normal(0, 1, DIMS)
                if kind == "ALL" or (kind == "HALF" and rng.random() < 0.5):
                    x[0] += 4.0
                rows.append({"well": well, "site": site, "sgRNA": guide,
                             "gene": "nontargeting" if kind == "nt" else kind,
                             **{f"PC_{d}": x[d] for d in range(DIMS)}})
    return pd.DataFrame(rows)


def test_fields_keep_their_fold_and_storey_reads_the_affected_share():
    folds = field_folds(["A1", "A1", "B2"], [3, 3, 4], folds=5)
    assert folds[0] == folds[1] and set(folds) <= set(range(5))
    rng = np.random.default_rng(1)
    assert abs(storey_penetrance(rng.random(20000))) < 0.03
    mixed = np.r_[rng.random(5000), np.full(5000, 0.99)]
    assert storey_penetrance(mixed) == pytest.approx(0.5, abs=0.03)


def test_escapers_need_a_separated_responder_mode():
    rng = np.random.default_rng(2)
    split = np.r_[rng.normal(0, 1, 1500), rng.normal(4, 0.7, 1500)]
    shifted = rng.normal(1.0, 1.2, 3000)
    hit, posterior = escaper_statistics(split)
    miss, _p = escaper_statistics(shifted)
    assert hit["escaper_raw"] and not miss["escaper_raw"]
    assert hit["mix_p"] == pytest.approx(0.5, abs=0.05)
    assert posterior[-1500:].mean() > 0.9
    assert not escaper_statistics(split, min_separation=5.0)[0]["escaper_raw"]
    assert ashman_d(4.0, 0.0) == pytest.approx(4 * np.sqrt(2))


def test_penetrance_separates_partial_full_and_null_genes_with_controls():
    cells = _screen()
    control = cells["gene"] == "nontargeting"
    result = penetrance_tests(cells, [f"PC_{d}" for d in range(DIMS)],
                              ["HALF", "ALL", "NONE", "ABSENT"], control=control,
                              control_pool=1500, reps=50, simulations=3,
                              shuffled=1)
    genes = result["genes"].set_index("gene")
    assert genes.loc["HALF", "penetrance"] == pytest.approx(0.5, abs=0.12)
    assert genes.loc["ALL", "penetrance"] > 0.85
    assert abs(genes.loc["NONE", "penetrance"]) < 0.15
    assert genes.loc["HALF", "pen_lo"] < genes.loc["HALF", "penetrance"] < genes.loc["HALF", "pen_hi"]
    assert "ABSENT" not in genes.index
    controls = result["controls"]
    pseudo = controls[controls.kind == "control: NT pseudo-gene"]
    assert len(pseudo) == 2 and (pseudo.penetrance.abs() < 0.2).all()
    shuffled = controls[controls.kind == "control: shuffled labels"]
    assert len(shuffled) == 1 and 0 < shuffled.real_gene_share.iloc[0] < 1
    assert set(result["guides"]["gene"]) >= {"HALF", "ALL", "NONE"}
    assert result["cells"]["u"].between(0, 1).all()


def test_phase_tests_find_a_gene_that_acts_in_mitosis_only():
    rng = np.random.default_rng(3)
    rows = []
    genes = [f"G{k:02d}" for k in range(30)]
    for well in ("A1", "A2"):
        for g in genes + [f"NTpseudo_{k:02d}" for k in range(8)]:
            for guide in range(4):
                for phase, count in (("G1", 40), ("SG2", 20), ("M", 10)):
                    x = rng.normal(0, 1, (count, DIMS))
                    if g == "G00" and phase == "M":
                        x[:, 0] += 3.0
                    for row in x:
                        rows.append({"well": well, "sgRNA": f"{g}_{guide}", "gene": g,
                                     "phase": phase,
                                     **{f"PC_{d}": row[d] for d in range(DIMS)}})
    cells = pd.DataFrame(rows)
    control = cells["gene"].str.startswith("NTpseudo")
    out = phase_tests(cells, [f"PC_{d}" for d in range(DIMS)], control=control,
                      null_sets=300, fresh_sets=50)
    calls = out["calls"].set_index("gene")
    assert calls.loc["G00", "q_M"] < 0.1
    assert calls.loc["G00", "specific_M"]
    assert not calls.drop(index="G00")["specific_M"].any()
    assert "control_fresh_random_gene_sets_p_lt_0.05_by_profile" in out["summary"]
    nt = phase_tests(cells, [f"PC_{d}" for d in range(DIMS)], control=control,
                     null="nt", null_sets=200)
    assert not nt["tests"]["gene"].str.startswith("RANDSET").any()
    with pytest.raises(ValueError, match="null must be one of"):
        phase_tests(cells, ["PC_0"], control=control, null="other")


def test_mask_neighbours_reads_the_cells_inside_the_disk():
    mask = np.zeros((60, 60), np.int32)
    mask[10:20, 10:20] = 1
    mask[10:20, 22:32] = 2
    mask[50:58, 50:58] = 3
    pairs, receivers = mask_neighbours(mask, [1, 3], [(15, 15), (54, 54)],
                                       radius=12, grid=2)
    assert set(map(tuple, pairs[["r_label", "n_label"]].to_numpy())) == {(1, 2)}
    row = receivers.set_index("label")
    assert row.loc[1, "n_nbr"] == 1 and row.loc[3, "n_nbr"] == 0
    assert row.loc[3, "edge_dist"] == pytest.approx(5.0)
    assert len(disk_offsets(50, 2)[0]) == 1961


def _neighbour_well(rng, well, effect):
    """A field grid: receivers next to gene X cells move by ``effect``."""
    rows, recv, pairs = [], [], []
    label = 0
    for site in range(30):
        for k in range(40):
            label += 1
            gene = rng.choice(["nontargeting", "X", "Y", "Z"], p=[0.6, 0.15, 0.15, 0.1])
            guide = f"{gene}_{rng.integers(0, 2)}" if gene != "nontargeting" else f"nt{rng.integers(0, 8)}"
            rows.append({"site": site, "label": label, "sgRNA": guide, "gene": gene})
        for k in range(40):
            r = label - 39 + k
            neighbour = label - 39 + (k + 1) % 40
            pairs.append({"site": site, "r_label": r, "n_label": neighbour, "n_px": 20})
            recv.append({"site": site, "label": r, "cy": 500.0 + k, "cx": 700.0,
                         "edge_dist": 100.0, "disk_occupied_frac": rng.random(),
                         "n_nbr": 1})
    cells = pd.DataFrame(rows)
    x = rng.normal(0, 1, (len(cells), DIMS))
    pair_frame = pd.DataFrame(pairs)
    neighbour_gene = cells.set_index(["site", "label"]).loc[
        list(zip(pair_frame.site, pair_frame.n_label)), "gene"].to_numpy()
    x[:, 0] += np.where(neighbour_gene == "X", effect, 0.0)
    for d in range(DIMS):
        cells[f"PC_{d}"] = x[:, d]
    return cells, pd.DataFrame(recv), pair_frame


def test_neighbour_effects_flag_the_gene_whose_neighbours_change():
    rng = np.random.default_rng(5)
    wells = {w: _neighbour_well(rng, w, 1.5) for w in ("A1", "A2")}
    out = neighbour_effects(wells, [f"PC_{d}" for d in range(DIMS)], active=[],
                            null_reps=2)
    tests = out["tests"].set_index("gene")
    assert tests.loc["X", "T"] > 3 * max(tests.loc["Y", "T"], tests.loc["Z", "T"])
    genes_rows = tests[tests.kind == "gene"]
    assert genes_rows["p_shuffle"].between(0, 1).all()
    assert genes_rows["q_clone"].notna().all()
    assert tests.loc["NT", "kind"] == "reference"
    assert set(out["nulls"]["null"]) == {"shuffle", "clone"}
    strict = neighbour_effects(wells, [f"PC_{d}" for d in range(DIMS)],
                               active=["X"], receivers="nt", null_reps=1)
    assert strict["tests"].set_index("gene").loc["X", "n_exposed"] > 0
    with pytest.raises(ValueError, match="receivers must be one of"):
        neighbour_effects(wells, ["PC_0"], active=[], receivers="all")


def test_escaper_classes_name_why_an_escaper_splits():
    rng = np.random.default_rng(6)
    genes = pd.DataFrame({
        "gene": ["E1", "E2", "C1", "C2"], "escaper": [True, True, False, False],
        "penetrance": 0.4, "pen_lo": 0.3, "pen_hi": 0.5, "mix_p": 0.4,
        "mix_mu1": [4.0, 0.5, 3.0, 3.0], "mix_s1": 1.0, "dBIC": 50.0,
        "guide_het_p": [1e-6, 0.5, 0.5, 0.5], "guide_resp_span": [0.6, 0.1, 0.1, 0.1],
        "guide_driven": [True, False, False, False]})
    rows = []
    for g in genes.gene:
        for k in range(200):
            responder = k < 80
            rows.append({"gene": g, "p_responder": 0.9 if responder else 0.1,
                         "e": rng.random() * 0.1, "n_exact": 4, "runner_margin": 1.0,
                         "source": "primary", "n_reads": 1 + (k % 3),
                         "phase": rng.choice(["G1", "SG2", "M"])})
    table, summary = escaper_classes(genes, pd.DataFrame(rows), controls=1)
    classes = table.set_index("gene")["escaper_class"]
    assert classes["E1"] == "guide-driven"
    assert classes["E2"] == "weakly bimodal (heavy-tailed shift)"
    assert summary["escapers"] == 2 and summary["controls"] == 1


def test_the_aggregate_phase_runs_penetrance_and_phase_tests(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_penetrance=True, ops_escapers=True,
                  ops_penetrance_nt_pool=600, ops_penetrance_folds=3,
                  ops_penetrance_lambda=0.5, ops_escaper_dbic=10.0,
                  ops_escaper_min_sep=2.0, ops_phase_tests=True,
                  ops_phase_null="targeting")
    options = report["profile_options"]
    genes = pd.read_parquet(options["outputs"]["penetrance_genes_parquet"])
    assert len(genes) == 40 and genes["penetrance"].median() > 0.3
    controls = pd.read_parquet(options["outputs"]["penetrance_controls_parquet"])
    pseudo = controls[controls["kind"] == "control: NT pseudo-gene"]
    assert len(pseudo) == 3 and pseudo["penetrance"].abs().max() < 0.25
    assert "escapers" in options["penetrance"]
    calls = pd.read_parquet(options["outputs"]["phase_calls_parquet"])
    assert calls["q_pooled"].notna().sum() >= 40
    assert "pooled_sig" in options["phase_tests"]


def test_escapers_need_the_penetrance_fit(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    with pytest.raises(ValueError, match="ops_penetrance too"):
        _run(tmp_path, ops_escapers=True)
