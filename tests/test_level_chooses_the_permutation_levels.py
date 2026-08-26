"""`level` says which levels a run reports, on the permutation side too.

The permutation path gated its gene pass on `guide_permutation_gene_level`,
a key that appears in no settings category and so had no control anywhere.
Meanwhile `level` -- the control that asks exactly this question on the
fitted side -- was greyed out whenever regression_type was 'mixed', which is
a parametric answer to a question the permutation test does not ask, since it
fits no model at all.

Between the two, a reader running the permutation test had no way to ask for
genes. Both halves are covered here: what the run writes, and whether the
control is offered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _screen(n_genes=6, guides_per_gene=3, wells=90, seed=0):
    """A long table shaped like spaCR's saved regression_data.csv."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    guides = [(g, f"{g}_{k}") for g in genes for k in range(guides_per_gene)]
    rows = []
    for well in range(wells):
        plate = f"plate{well % 3 + 1}"
        prc = f"{plate}_r{well // 12 + 1}_c{well % 12 + 1}"
        chosen = rng.choice(len(guides), size=4, replace=False)
        share = rng.dirichlet(np.ones(4))
        # One gene really does move the phenotype, so the gene pass has
        # something to find and the test is not measuring an empty frame.
        hit = sum(share[i] for i, c in enumerate(chosen)
                  if guides[c][0] == "G0")
        pred = float(np.clip(0.2 + 0.6 * hit + rng.normal(0, 0.05), 0.01, 0.99))
        for i, c in enumerate(chosen):
            gene, guide = guides[c]
            rows.append({"prc": prc, "grna": guide, "gene": gene,
                         "fraction": float(share[i]), "pred": pred,
                         "cell_count": 120, "plateID": plate,
                         "rowID": f"r{well // 12 + 1}",
                         "columnID": f"c{well % 12 + 1}"})
    return pd.DataFrame(rows)


def _settings(level, **extra):
    base = dict(guide_min_wells=[1], guide_primary_min_wells=1,
                guide_permutations=60, guide_permutation_seed=0,
                guide_permutation_block="plateID", guide_nuisance_columns=[],
                multiple_testing_method="fdr_bh", fdr_alpha=0.05,
                guide_presence_threshold=0.0,
                guide_permutation_batch_size=30, grna_statistic="pearson",
                analysis_unit="well", agg_type="mean", regression_type="beta",
                level=level)
    base.update(extra)
    return base


def _run(tmp_path, level, **extra):
    from spacr.ml import _run_guide_permutation_analysis

    dst = tmp_path / f"run_{level}_{len(extra)}"
    dst.mkdir()
    _run_guide_permutation_analysis(_screen(), "pred", str(dst),
                                    _settings(level, **extra))
    return pd.read_csv(dst / "results.csv"), dst


@pytest.mark.parametrize("level,expected", [
    ("grna", {"grna"}),
    ("gene", {"gene"}),
    ("both", {"grna", "gene"}),
])
def test_the_primary_table_reports_the_level_asked_for(tmp_path, level,
                                                       expected):
    frame, _ = _run(tmp_path, level)
    assert set(frame["level"].dropna().unique()) == expected
    assert len(frame) > 0


def test_grna_skips_the_gene_pass_entirely(tmp_path):
    """The gene permutation is the expensive half; asking for guides only
    must not pay for it."""
    _, dst = _run(tmp_path, "grna")
    genes = pd.read_csv(dst / "results_gene.csv")
    assert genes.empty


def test_gene_still_runs_the_guide_pass(tmp_path):
    """A gene's regressor IS the sum of its guides' fractions, so there is no
    gene answer without the guide table -- it is reported, not skipped."""
    _, dst = _run(tmp_path, "gene")
    guides = pd.read_csv(dst / "results_grna.csv")
    assert len(guides) > 0


def test_both_is_the_two_tables_stacked(tmp_path):
    frame, dst = _run(tmp_path, "both")
    guides = pd.read_csv(dst / "results_grna.csv")
    genes = pd.read_csv(dst / "results_gene.csv")
    assert len(frame) == len(guides) + len(genes)


def test_the_gene_family_is_corrected_on_its_own(tmp_path):
    """Never pooled with the guides: same wells, and the gene regressor is
    literally the sum of the guide regressors."""
    frame, _ = _run(tmp_path, "both")
    genes = frame[frame["level"] == "gene"]
    guides = frame[frame["level"] == "grna"]
    assert genes["tested_genes_in_family"].dropna().nunique() == 1
    assert int(genes["tested_genes_in_family"].dropna().iloc[0]) == len(genes)
    assert int(guides["tested_guides_in_family"].dropna().iloc[0]) == len(guides)


def test_the_explicit_key_still_wins(tmp_path):
    """A saved settings file naming guide_permutation_gene_level keeps
    meaning what it said."""
    frame, _ = _run(tmp_path, "both", guide_permutation_gene_level=False)
    assert set(frame["level"].dropna().unique()) == {"grna"}


def test_the_control_is_offered_under_nonparametric():
    """It was greyed by a mixed regression_type -- which the permutation test
    never reads, because it fits no model."""
    from spacr.settings import get_setting_dependencies

    predicate = get_setting_dependencies()["level"]["predicate"]
    for family in ("mixed", "ols", "beta"):
        assert predicate({"inference": "nonparametric",
                          "regression_type": family,
                          "random_row_column_effects": True}, None)


def test_a_fitted_mixed_model_still_greys_it():
    """The parametric reason is real and must survive: a mixed model nests
    guides inside genes and answers both levels at once."""
    from spacr.settings import get_setting_dependencies

    rule = get_setting_dependencies()["level"]
    settings = {"inference": "parametric", "regression_type": "mixed",
                "random_row_column_effects": False}
    assert not rule["predicate"](settings, None)
    assert "mixed" in rule["reason"](settings, None)


def test_inference_is_one_of_the_sources():
    """Without it the panel never re-evaluates the rule when inference
    changes, so the control would stay greyed until something else moved."""
    from spacr.settings import get_setting_dependencies

    assert "inference" in get_setting_dependencies()["level"]["sources"]
