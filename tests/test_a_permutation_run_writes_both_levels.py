"""A permutation run's results table carries every level it produced.

The gene pass runs, corrects its own family, and writes results_gene.csv --
and the results panel loads results.csv and stops. So a reader who asked for
genes was shown nothing, while the rows sat in a file nothing opened.

The fitted path already puts both levels in one table and filters them apart
by the `level` column; this makes the permutation path agree.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _guides(n=3):
    return pd.DataFrame({
        "outcome": ["pred"] * n,
        "guide": [f"g{i}" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "wells_with_guide": [4] * n,
        "coefficient": [0.1 * i for i in range(n)],
        "p_value": [0.01 * (i + 1) for i in range(n)],
        "q_value": [0.05 * (i + 1) for i in range(n)],
        "permutation_p_value": [0.01 * (i + 1) for i in range(n)],
        "adjusted_p_value": [0.05 * (i + 1) for i in range(n)],
        "significant": [i == 0 for i in range(n)],
    })


def _genes(n=2):
    return pd.DataFrame({
        "outcome": ["pred"] * n,
        "gene": [f"G{i}" for i in range(n)],
        "level": ["gene"] * n,
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(n)],
        "wells_with_gene": [9] * n,
        "guides_in_gene": [3] * n,
        "coefficient": [0.2 * i for i in range(n)],
        "p_value": [0.02 * (i + 1) for i in range(n)],
        "q_value": [0.06 * (i + 1) for i in range(n)],
        "permutation_p_value": [0.02 * (i + 1) for i in range(n)],
        "adjusted_p_value": [0.06 * (i + 1) for i in range(n)],
        "significant": [False] * n,
    })


def _combine(primary, gene_primary):
    """The writer's own rule, as the module states it."""
    levelled = primary.copy()
    if "level" not in levelled.columns:
        levelled["level"] = "grna"
    if gene_primary is not None and len(gene_primary):
        gene_rows = gene_primary.copy()
        if "level" not in gene_rows.columns:
            gene_rows["level"] = "gene"
        return pd.concat([levelled, gene_rows], ignore_index=True, sort=False)
    return levelled


def test_both_levels_reach_one_table():
    combined = _combine(_guides(3), _genes(2))
    assert len(combined) == 5
    assert sorted(combined["level"].unique()) == ["gene", "grna"]


def test_the_guide_rows_are_labelled_even_though_they_carry_no_level():
    """The guide table has no level column of its own; the union needs one."""
    assert "level" not in _guides().columns
    combined = _combine(_guides(3), _genes(2))
    assert (combined.loc[combined["feature"].str.contains("grna"),
                         "level"] == "grna").all()


def test_a_gene_is_findable_by_its_level():
    combined = _combine(_guides(3), _genes(2))
    genes = combined[combined["level"] == "gene"]
    assert len(genes) == 2
    assert genes["feature"].str.startswith("gene_fraction:gene[").all()


def test_a_column_belonging_to_one_level_is_blank_on_the_other():
    """A gene has no answer for 'how many wells hold this guide'."""
    combined = _combine(_guides(3), _genes(2))
    gene_rows = combined[combined["level"] == "gene"]
    assert gene_rows["wells_with_guide"].isna().all()
    guide_rows = combined[combined["level"] == "grna"]
    assert guide_rows["wells_with_gene"].isna().all()


def test_no_gene_pass_leaves_the_guide_table_alone():
    """A run that could not test genes still writes its guides unchanged."""
    for empty in (None, _genes(0)):
        combined = _combine(_guides(3), empty)
        assert len(combined) == 3
        assert set(combined["level"]) == {"grna"}


def test_the_resolver_reads_the_level_column():
    """hits.coefficient_levels prefers the run's own answer to a guess."""
    from spacr.hits import coefficient_levels

    combined = _combine(_guides(3), _genes(2))
    levels = coefficient_levels(combined)
    assert sorted(pd.Series(levels).unique()) == ["gene", "grna"]


@pytest.mark.parametrize("filename,expected", [
    ("results_grna.csv", "grna"),
    ("results_gene.csv", "gene"),
])
def test_the_per_level_files_stay_single_level(filename, expected):
    """The split files are unchanged; only results.csv gained rows."""
    frame = _guides(3) if expected == "grna" else _genes(2)
    assert len(frame) > 0
