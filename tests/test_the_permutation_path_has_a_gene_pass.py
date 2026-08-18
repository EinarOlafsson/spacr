"""Instruction 132: the nonparametric path gets a gene pass, with its own family.

The guide permutation test answers "does this guide move the phenotype". The
parametric path answers that AND "does this gene move the phenotype", from two
separate fits. The permutation path only ever answered the first, so choosing
``inference='nonparametric'`` silently lost the gene level.

TWO THINGS THIS FILE PINS DOWN.

* The gene's regressor is the SUM of its guides' fractions -- exactly the
  ``gene_fraction`` that :func:`spacr.ml.check_and_clean_data` builds and the
  parametric gene fit regresses on. Measured against the real function on the
  maintainer's TSG101 frame: max |difference| = 0.0.

* The gene P value is a test of the SET, taken against the same Freedman--Lane
  null as the guides, and is NOT a combination of the guides' P values.
  Fisher's and Stouffer's methods assume independence; two guides scored in the
  same wells share that well's phenotype, plate and cells, so combining them
  would claim a confidence the design cannot support.

And the correction: two families, corrected separately. Pooling would be wrong
twice -- not independent, and twice the family for no protection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)


def _long_table(seed=0, n_plates=3, n_rows=4, n_cols=4):
    """spaCR's long regression table: one row per (well, guide)."""
    rng = np.random.default_rng(seed)
    guides = {
        "geneA": ["geneA_1", "geneA_2", "geneA_3"],
        "geneB": ["geneB_1", "geneB_2"],
        "geneC": ["geneC_1"],
        "control": ["control_1", "control_2"],
    }
    recs = []
    for plate in range(n_plates):
        for row in range(n_rows):
            for col in range(n_cols):
                well = f"plate{plate + 1}_r{row + 1}_c{col + 1}"
                effect = 0.0
                for gene, gene_guides in guides.items():
                    for guide in gene_guides:
                        if rng.random() < 0.45:
                            continue
                        fraction = float(rng.uniform(0.05, 0.5))
                        if gene == "geneA":
                            effect += 1.4 * fraction
                        recs.append({
                            "prc": well,
                            "plateID": f"plate{plate + 1}",
                            "rowID": f"r{row + 1}",
                            "columnID": f"c{col + 1}",
                            "grna": guide,
                            "gene": gene,
                            "fraction": fraction,
                        })
                for record in recs:
                    if record["prc"] == well:
                        record["pred"] = effect + 0.05 * plate
    frame = pd.DataFrame(recs)
    noise = rng.normal(0, 0.03, frame["prc"].nunique())
    per_well = dict(zip(sorted(frame["prc"].unique()), noise))
    frame["pred"] = frame["pred"] + frame["prc"].map(per_well)
    return frame


# --------------------------------------------------------------------------- #
#  The gene regressor IS the summed guide fraction
# --------------------------------------------------------------------------- #

def test_the_gene_matrix_is_the_parametric_gene_fraction():
    """Both paths must test the same regressor, or they answer different
    questions and their disagreement means nothing."""
    from spacr.guide_permutation import prepare_long_gene_data
    from spacr.ml import check_and_clean_data

    frame = _long_table()
    gene_fractions, _outcomes, _metadata = prepare_long_gene_data(frame, "pred")

    clean = check_and_clean_data(frame.copy(), "pred")
    reference = (clean.groupby(["prc", "gene"], observed=True)["gene_fraction"]
                 .first().unstack(fill_value=0.0))
    reference = reference.reindex(index=gene_fractions.index,
                                  columns=gene_fractions.columns,
                                  fill_value=0.0)

    difference = np.abs(gene_fractions.to_numpy(dtype=float)
                        - reference.to_numpy(dtype=float)).max()
    assert difference == 0.0, difference


def test_the_gene_column_is_the_sum_of_its_own_guide_columns():
    from spacr.guide_permutation import (gene_fraction_matrix,
                                         prepare_long_guide_data)

    frame = _long_table(seed=1)
    fractions, _outcomes, _metadata = prepare_long_guide_data(frame, "pred")
    mapping = dict(zip(frame["grna"].astype(str), frame["gene"].astype(str)))

    genes = gene_fraction_matrix(fractions, mapping)

    for gene in genes.columns:
        own = [name for name in fractions.columns if mapping[str(name)] == gene]
        assert np.allclose(genes[gene].to_numpy(),
                           fractions[own].to_numpy().sum(axis=1), atol=0), gene


def test_a_guide_with_no_gene_is_refused_rather_than_dropped():
    from spacr.guide_permutation import (gene_fraction_matrix,
                                         prepare_long_guide_data)

    frame = _long_table(seed=2)
    fractions, _outcomes, _metadata = prepare_long_guide_data(frame, "pred")
    mapping = dict(zip(frame["grna"].astype(str), frame["gene"].astype(str)))
    mapping.pop("geneA_1")

    with pytest.raises(ValueError, match="have none"):
        gene_fraction_matrix(fractions, mapping)


def test_a_guide_naming_two_genes_is_refused():
    """Summing one guide into two genes would count its wells twice."""
    from spacr.guide_permutation import prepare_long_gene_data

    frame = _long_table(seed=3)
    frame.loc[frame.index[0], "gene"] = "somethingelse"

    with pytest.raises(ValueError, match="more than one gene"):
        prepare_long_gene_data(frame, "pred")


def test_a_frame_without_a_gene_column_says_so():
    from spacr.guide_permutation import prepare_long_gene_data

    frame = _long_table(seed=4).drop(columns=["gene"])

    with pytest.raises(ValueError, match="needs a 'gene' column"):
        prepare_long_gene_data(frame, "pred")


# --------------------------------------------------------------------------- #
#  TWO FAMILIES, CORRECTED SEPARATELY
# --------------------------------------------------------------------------- #

def test_the_gene_pass_is_corrected_over_genes_and_the_guide_pass_over_guides():
    from spacr.guide_permutation import (analyse_long_gene_table,
                                         analyse_long_guide_table)
    from spacr.multiple_testing import adjust_p_values

    frame = _long_table(seed=5)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[2],
                                    n_permutations=999, random_state=0)
    guides = analyse_long_guide_table(frame, "pred", min_wells=[2],
                                      n_permutations=999, random_state=0)

    assert len(genes) < len(guides), (len(genes), len(guides))
    assert set(genes["level"]) == {"gene"}
    assert int(genes["tested_genes_in_family"].iloc[0]) == len(genes)
    assert int(guides["tested_guides_in_family"].iloc[0]) == len(guides)

    for table, column in ((genes, "tested_genes_in_family"),
                          (guides, "tested_guides_in_family")):
        expected, _rejected = adjust_p_values(
            table["permutation_p_value"].to_numpy(dtype=float),
            method="fdr_bh", alpha=0.05)
        assert np.allclose(table["adjusted_p_value"].to_numpy(dtype=float),
                           expected), column

    # ...and NOT one correction over both stacked.
    pooled, _ = adjust_p_values(
        np.concatenate([genes["permutation_p_value"].to_numpy(dtype=float),
                        guides["permutation_p_value"].to_numpy(dtype=float)]),
        method="fdr_bh", alpha=0.05)
    separate = np.concatenate([genes["adjusted_p_value"].to_numpy(dtype=float),
                               guides["adjusted_p_value"].to_numpy(dtype=float)])
    assert not np.allclose(pooled, separate), (
        "the two permutation families were pooled into one correction")


def test_the_gene_pass_is_not_a_combination_of_its_guides_p_values():
    """Fisher and Stouffer assume independence; guides sharing a well are not.

    The gene P value must come from testing the SUMMED regressor against the
    permutation null, so it must not equal either combination.
    """
    from scipy import stats

    from spacr.guide_permutation import (analyse_long_gene_table,
                                         analyse_long_guide_table)

    frame = _long_table(seed=6)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[1],
                                    n_permutations=999, random_state=0)
    guides = analyse_long_guide_table(frame, "pred", min_wells=[1],
                                      n_permutations=999, random_state=0)
    gene_of = dict(zip(frame["grna"].astype(str), frame["gene"].astype(str)))
    guides = guides.assign(gene=guides["guide"].astype(str).map(gene_of))

    checked = 0
    for gene, rows in guides.groupby("gene"):
        own = genes.loc[genes["gene"] == gene]
        if own.empty or len(rows) < 2:
            continue
        checked += 1
        observed = float(own["permutation_p_value"].iloc[0])
        parts = rows["permutation_p_value"].to_numpy(dtype=float)
        fisher = float(stats.combine_pvalues(parts, method="fisher")[1])
        stouffer = float(stats.combine_pvalues(parts, method="stouffer")[1])
        assert not np.isclose(observed, fisher), (gene, observed, fisher)
        assert not np.isclose(observed, stouffer), (gene, observed, stouffer)
    assert checked >= 2, "no multi-guide gene was actually compared"


def test_the_gene_pass_uses_the_same_permutation_null_as_the_guide_pass():
    """Same seed, same outcome, same nuisance design: the same permutations.

    A single-guide gene's summed regressor IS that guide's regressor, so its
    empirical P value must come out identical -- which is only true if both
    passes permuted the same residual vectors in the same order.
    """
    from spacr.guide_permutation import (analyse_long_gene_table,
                                         analyse_long_guide_table)

    frame = _long_table(seed=7)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[1],
                                    n_permutations=499, random_state=0)
    guides = analyse_long_guide_table(frame, "pred", min_wells=[1],
                                      n_permutations=499, random_state=0)

    solo = genes.loc[genes["gene"] == "geneC"]
    its_guide = guides.loc[guides["guide"] == "geneC_1"]
    assert len(solo) == 1 and len(its_guide) == 1

    assert float(solo["standardized_marginal_effect"].iloc[0]) == pytest.approx(
        float(its_guide["standardized_marginal_effect"].iloc[0]))
    assert (int(solo["permutation_exceedances"].iloc[0])
            == int(its_guide["permutation_exceedances"].iloc[0])), (
        "the two passes did not permute the same residuals")


def test_every_gene_row_records_how_many_guides_it_rests_on():
    """A one-degree-of-freedom set test cancels when guides disagree, so the
    number of guides behind a gene has to be on the row that reports it."""
    from spacr.guide_permutation import analyse_long_gene_table

    frame = _long_table(seed=8)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[1],
                                    n_permutations=199, random_state=0)

    counts = frame.groupby("gene")["grna"].nunique().to_dict()
    assert "guides_in_gene" in genes.columns
    for _index, row in genes.iterrows():
        assert int(row["guides_in_gene"]) == counts[row["gene"]], row["gene"]
    assert int(genes.loc[genes["gene"] == "geneC",
                         "guides_in_gene"].iloc[0]) == 1


def test_the_gene_pass_finds_the_planted_gene():
    """It has to actually work, not merely run."""
    from spacr.guide_permutation import analyse_long_gene_table

    frame = _long_table(seed=9, n_plates=4, n_rows=6, n_cols=6)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[2],
                                    n_permutations=4999, random_state=0)

    ranked = genes.sort_values("permutation_p_value")
    assert ranked["gene"].iloc[0] == "geneA", ranked[
        ["gene", "standardized_marginal_effect", "permutation_p_value"]]
    assert float(ranked["standardized_marginal_effect"].iloc[0]) > 0
    assert bool(ranked["significant"].iloc[0])


# --------------------------------------------------------------------------- #
#  The run itself writes both tables
# --------------------------------------------------------------------------- #

def _permutation_settings(**over):
    base = {
        "guide_min_wells": [2],
        "guide_permutations": 499,
        "guide_permutation_seed": 0,
        "guide_permutation_block": "plateID",
        "multiple_testing_method": "fdr_bh",
        "fdr_alpha": 0.05,
        "analysis_unit": "well",
        "guide_permutation_plot": False,
        "threshold_method": "std",
        "threshold_multiplier": 3.0,
        "controls": ["control_1", "control_2"],
        "negative_control": "control",
    }
    base.update(over)
    return base


def test_a_permutation_run_writes_results_gene_csv_beside_results_grna(tmp_path):
    """It never wrote one at all: the gene level was simply missing."""
    from spacr.ml import _run_guide_permutation_analysis

    frame = _long_table(seed=11, n_plates=3, n_rows=5, n_cols=5)
    output = _run_guide_permutation_analysis(
        frame, "pred", str(tmp_path), _permutation_settings())

    guide_csv = tmp_path / "results_grna.csv"
    gene_csv = tmp_path / "results_gene.csv"
    assert guide_csv.is_file() and gene_csv.is_file()

    guides = pd.read_csv(guide_csv)
    genes = pd.read_csv(gene_csv)
    assert len(genes) > 0 and len(guides) > 0
    assert set(genes["level"]) == {"gene"}
    assert genes["feature"].str.startswith("gene_fraction:gene[").all()
    assert guides["feature"].str.startswith("fraction:grna[").all()

    # TWO FAMILIES: the gene q values are BH over the genes alone.
    from spacr.multiple_testing import adjust_p_values

    expected, _rejected = adjust_p_values(
        genes["p_value"].to_numpy(dtype=float), method="fdr_bh", alpha=0.05)
    assert np.allclose(genes["q_value"].to_numpy(dtype=float), expected)
    assert len(genes) != len(guides)

    assert output["gene_results"] is not None
    assert len(output["gene_results"]) == len(genes)


def test_the_gene_pass_can_be_declined(tmp_path):
    """It costs a second full permutation run, so a sweep must be able to
    decline it -- and then the file is empty rather than absent."""
    from spacr.ml import _run_guide_permutation_analysis

    frame = _long_table(seed=12, n_plates=3, n_rows=5, n_cols=5)
    output = _run_guide_permutation_analysis(
        frame, "pred", str(tmp_path),
        _permutation_settings(guide_permutation_gene_level=False))

    assert output["gene_results"] is None
    gene_csv = tmp_path / "results_gene.csv"
    assert gene_csv.is_file()
    assert len(pd.read_csv(gene_csv)) == 0
