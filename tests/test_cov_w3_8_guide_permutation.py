"""How the guide permutation analysis labels a volcano, and when it refuses one.

The input guards and arithmetic of the permutation test itself live in
``test_guide_permutation_guards.py``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import guide_permutation as gp


# ---------------------------------------------------------------------------
# The volcano
# ---------------------------------------------------------------------------

def _results(guides=("g1", "g2", "g3", "g4")):
    return pd.DataFrame({
        "guide": list(guides),
        "outcome": "pathogen_rate",
        "minimum_wells_threshold": 1,
        "standardized_marginal_effect": [0.4, -0.3, 0.2, -0.1],
        "permutation_p_value": [0.001, 0.002, 0.4, 0.5],
        "adjusted_p_value": [0.004, 0.008, 0.6, 0.7],
        "significant": [True, True, False, False],
        "multiple_testing_method": "fdr_bh",
        "alpha": 0.05,
    })


def test_labelled_guides_are_annotated_and_absent_ones_are_skipped(tmp_path):
    """Four labels, three of which exist: the fourth must not raise."""
    import matplotlib

    matplotlib.use("Agg")
    written = gp.plot_guide_permutation_volcano(
        _results(), outcome="pathogen_rate", minimum_wells=1,
        save_path=tmp_path / "volcano.png",
        label_guides={"g1": "GENE_A g1", "g2": "GENE_B g2",
                      "g3": "GENE_C g3", "g9": "not in the table"},
        effect_threshold=0.25,
    )
    assert written.exists()
    assert written.stat().st_size > 0


def test_a_volcano_for_a_family_with_no_rows_says_so(tmp_path):
    with pytest.raises(ValueError, match="No rows for outcome"):
        gp.plot_guide_permutation_volcano(
            _results(), outcome="pathogen_rate", minimum_wells=7,
            save_path=tmp_path / "volcano.png")
