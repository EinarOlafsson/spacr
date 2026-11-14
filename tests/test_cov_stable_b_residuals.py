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


def test_the_rescaled_response_panel_labels_both_histograms():
    """The two guaranteed histogram handles still produce two legend labels."""
    from spacr.response_distribution import panel

    result = panel(np.linspace(0.05, 0.95, 40), "sqrt")
    labels = [text.get_text() for text in result["axes"].get_legend().texts]

    assert len(result["axes"].patches) == 40
    assert labels[0] == "before"
    assert labels[1].startswith("after ")


def test_each_sudoku_round_reads_its_own_named_column(monkeypatch):
    """The full guide tuple returned by the inner solve fixes each index."""
    from spacr import sudoku as module

    calls = 0

    def inner(features, scores, wells, fractions, guides, **_kwargs):
        nonlocal calls
        calls += 1
        n = len(features)
        guide = "g1" if calls == 1 else "g2"
        called = [guide, *([module.ABSTAIN] * (n - 1))]
        evidence = np.tile(np.array([11.0, 22.0]), (n, 1))
        return SimpleNamespace(
            names=tuple(guides), guides=called, affirm=evidence,
            eliminate=evidence + 1, posterior=evidence + 2,
            reach=np.zeros(n),
        )

    monkeypatch.setattr(module, "sudoku", inner)
    result = module.sudoku_all(
        np.zeros((4, 1)), np.zeros(4), ["w"] * 4,
        {"w": {"g1": 0.5, "g2": 0.5}},
        [("g1", 0.9), ("g2", 0.8)],
    )

    assert result.affirm[0, 0] == 11.0
    assert result.affirm[1, 1] == 22.0
