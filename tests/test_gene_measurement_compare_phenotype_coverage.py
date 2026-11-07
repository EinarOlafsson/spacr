"""Per-guide normalisation and the bootstrap null where there is nothing to
compare against."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (aggregate_genes, aggregate_guides,
                                            bootstrap_gene_pvalues,
                                            robust_z_to_controls)


def test_a_well_without_controls_stays_nan_and_does_not_borrow_another():
    cells = pd.DataFrame({
        "well": ["A1", "A1", "A1", "B1", "B1"],
        "area": [10.0, 12.0, 20.0, 100.0, 200.0]})
    control = pd.Series([True, True, False, False, False])
    z = robust_z_to_controls(cells, ["area"], control=control)
    assert z.loc[3:, "area"].isna().all()
    # A1: control median 11, MAD 1 -> spread 1.4826.
    assert np.allclose(z.loc[:2, "area"].to_numpy(),
                       np.array([-1.0, 1.0, 9.0]) / 1.4826)


def test_guides_and_genes_without_a_mitotic_column_carry_no_mitotic_fields():
    cells = pd.DataFrame({"guide": ["g1", "g1", "g2"],
                          "gene": ["A", "A", "A"],
                          "area": [1.0, 3.0, 5.0]})
    guides = aggregate_guides(cells, ["area"])
    assert list(guides.columns) == ["guide", "gene", "n_cells", "area"]
    genes = aggregate_genes(guides, ["area"])
    assert list(genes.columns) == ["gene", "n_guides", "n_cells", "area"]
    assert genes.loc[0, "n_cells"] == 3 and genes.loc[0, "area"] == 3.5


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_a_feature_with_no_control_values_has_no_pvalue_and_no_fdr():
    guides = pd.DataFrame({"guide": ["g1", "g2"], "gene": ["A", "B"],
                           "n_cells": [5, 5], "area": [1.0, np.nan],
                           "size": [2.0, 3.0]})
    controls = pd.DataFrame({"area": [np.nan, np.nan],
                             "size": [0.0, 0.1]})
    out = bootstrap_gene_pvalues(controls, guides, ["area", "size"],
                                 n_boot=20, seed=1,
                                 null="cell_bootstrap").set_index("gene")
    assert out["area_pvalue"].isna().all()
    assert out["area_fdr"].isna().all()
    assert out["size_pvalue"].between(0, 1).all()
    assert out["size_fdr"].notna().all()


def test_no_genes_gives_an_empty_table_without_feature_columns():
    guides = pd.DataFrame({"guide": [], "gene": [], "n_cells": [],
                           "area": []})
    for options in ({"null": "cell_bootstrap"},
                    {"control_guides": pd.DataFrame({
                        "n_cells": [5, 5], "area": [0.0, 1.0]})}):
        out = bootstrap_gene_pvalues(pd.DataFrame({"area": [1.0, 2.0]}),
                                     guides, ["area"], n_boot=5, **options)
        assert out.empty
        assert "area_pvalue" not in out and "area_fdr" not in out
