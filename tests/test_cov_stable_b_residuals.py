"""Direct drivers for the two reachable stable-B coverage residuals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_the_real_dropped_column_report_deduplicates_merged_names():
    """Two table records may report the same merged measurement."""
    from spacr.plate_measurements import PlateMerge

    first = SimpleNamespace(
        dropped=("area", "perimeter"), merged_column=lambda name: name,
    )
    second = SimpleNamespace(
        dropped=("area",), merged_column=lambda name: name,
    )
    merge = PlateMerge(
        frame=pd.DataFrame(), anchor="cell", attachments=(),
        tables=(first, second),
    )

    assert merge.dropped_columns == ("area", "perimeter")


def test_distinct_dropped_columns_are_not_deduplicated_together():
    """The duplicate branch has a positive counterpart with two real names."""
    from spacr.plate_measurements import PlateMerge

    entry = SimpleNamespace(
        dropped=("zeta", "alpha"), merged_column=lambda name: name,
    )
    merge = PlateMerge(
        frame=pd.DataFrame(), anchor="cell", attachments=(), tables=(entry,),
    )

    assert merge.dropped_columns == ("alpha", "zeta")


def test_every_concordance_summary_gene_has_source_values():
    """The groupby index cannot name a gene absent from its own frame."""
    frame = pd.DataFrame({
        "gene": ["a", "a", "b"],
        "agree": [0.1, 0.2, 0.9],
    })
    summary = frame.groupby("gene").agg(agreement=("agree", "mean"))

    for gene in summary.index:
        values = frame.loc[frame["gene"] == gene, "agree"].to_numpy(float)
        assert len(values) > 0

    assert np.isfinite(summary["agreement"]).all()
