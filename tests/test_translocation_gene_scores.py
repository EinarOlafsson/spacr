"""Per-gene translocation scores: planted defects are found, a null is not.

Synthetic cells only, fixed seeds, CPU. The per-cell readout stands in for the
nuclear DNA-marker correlation of Feldman et al. (2019): non-targeting cells
and unaffected genes share one distribution, defect genes are shifted down.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (_integrated_cdf_difference,
                                            translocation_gene_scores)


def _cells(*, planted=(), null_genes=40, guides=3, cells_per_guide=50,
           control_cells=1500, shift=-0.25, seed=1):
    """Cells for ``planted`` defect genes, unaffected genes and NT controls."""
    rng = np.random.default_rng(seed)
    rows = [{"gene": "nontargeting", "guide": f"nt_{i % 30}",
             "readout": value}
            for i, value in enumerate(rng.normal(0.5, 0.2, control_cells))]
    names = [(name, shift) for name in planted]
    names += [(f"null_{i:03d}", 0.0) for i in range(null_genes)]
    for name, offset in names:
        for g in range(guides):
            for value in rng.normal(0.5 + offset, 0.2, cells_per_guide):
                rows.append({"gene": name, "guide": f"{name}_sg{g}",
                             "readout": value})
    return pd.DataFrame(rows)


def test_the_integrated_cdf_difference_is_the_mean_shift():
    rng = np.random.default_rng(3)
    guide, control = rng.normal(0.2, 1, 80), rng.normal(0.5, 1, 400)
    assert _integrated_cdf_difference(guide, control) == pytest.approx(
        guide.mean() - control.mean(), abs=1e-12)


def test_planted_defects_score_negative_and_pass_the_fdr():
    cells = _cells(planted=("RELA", "TRADD", "IKBKB"))
    result = translocation_gene_scores(cells, "readout", permutations=400,
                                       seed=0)
    genes = result["genes"].set_index("gene")
    for name in ("RELA", "TRADD", "IKBKB"):
        assert genes.loc[name, "score"] < -0.15
        assert genes.loc[name, "q"] < 0.1
        assert genes.loc[name, "mannwhitney_q"] < 0.1
        assert genes.loc[name, "n_guides"] == 3
    null_hits = genes.filter(like="null_", axis=0)["q"] < 0.1
    assert null_hits.sum() <= 2
    assert set(result["guides"].columns) == {"group", "gene", "guide",
                                             "n_cells", "score"}
    assert len(result["guides"]) == 43 * 3


def test_a_null_screen_calls_hits_at_about_the_nominal_rate():
    # Every gene shares one non-targeting pool, so the P values of one screen
    # are correlated and a single screen's rate swings (0.8-10.8 % at p < 0.05
    # over 20 seeds). Calibration is judged over several screens.
    rates, fdr_hits = [], 0
    for screen in range(6):
        genes = translocation_gene_scores(
            _cells(null_genes=60, seed=100 + screen), "readout",
            permutations=300, seed=screen)["genes"]
        rates.append(float((genes["p"] < 0.05).mean()))
        fdr_hits += int((genes["q"] < 0.1).sum())
    assert 0.01 <= np.mean(rates) <= 0.10
    assert fdr_hits <= 3


def test_a_rerun_with_the_same_seed_is_identical():
    cells = _cells(planted=("RELA",), null_genes=5)
    first = translocation_gene_scores(cells, "readout", permutations=200, seed=5)
    second = translocation_gene_scores(cells, "readout", permutations=200, seed=5)
    pd.testing.assert_frame_equal(first["genes"], second["genes"])
    pd.testing.assert_frame_equal(first["guides"], second["guides"])


def test_groups_are_scored_and_corrected_separately():
    tnf = _cells(planted=("TRADD",), null_genes=10, seed=2).assign(stimulus="TNF")
    il1 = _cells(planted=("MYD88",), null_genes=10, seed=4).assign(stimulus="IL1")
    genes = translocation_gene_scores(pd.concat([tnf, il1], ignore_index=True),
                                      "readout", group="stimulus",
                                      permutations=300, seed=0)["genes"]
    by = genes.set_index(["group", "gene"])
    assert by.loc[("TNF", "TRADD"), "q"] < 0.1
    assert by.loc[("IL1", "MYD88"), "q"] < 0.1
    assert ("TNF", "MYD88") not in by.index


def test_small_guides_are_left_out_and_bad_inputs_are_named():
    cells = _cells(planted=("RELA",), null_genes=2, cells_per_guide=10)
    result = translocation_gene_scores(cells, "readout", min_cells=20,
                                       permutations=50)
    assert result["genes"].empty and result["guides"].empty
    with pytest.raises(KeyError, match="readout_missing"):
        translocation_gene_scores(cells, "readout_missing")
    with pytest.raises(ValueError, match="non-targeting"):
        translocation_gene_scores(cells[cells.gene != "nontargeting"],
                                  "readout")
    with pytest.raises(ValueError, match="at least 1"):
        translocation_gene_scores(cells, "readout", permutations=0)
