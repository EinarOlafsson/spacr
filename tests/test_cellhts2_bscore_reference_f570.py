"""Golden public-screen scores from unchanged cellHTS2 R routines, not a port."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import score_screen, screen_wells

_EVIDENCE = (Path(__file__).resolve().parents[1] / "features" / "data"
             / "570_cellhts2_reference_2026-10-01")


@pytest.mark.parametrize("feature", ["cell_count_proxy", "nuclear_area"])
@pytest.mark.parametrize("scope", ["plate", "pooled"])
def test_public_screen_scores_calls_and_rankings_match_original_cellhts2(feature, scope):
    screen = pd.read_csv(_EVIDENCE / "screen.csv")
    reference = pd.read_csv(_EVIDENCE / "reference.csv")
    wells = screen_wells(screen, feature, plate_column="plateID",
                         control_column="well_type", negative_levels=["negcon"])
    result = score_screen(wells, scope=scope, rank_by="b_score")
    actual = result.wells.sort_values(["plateID", "well"]).reset_index(drop=True)
    expected = reference.sort_values(["plateID", "well"]).reset_index(drop=True)
    assert len(actual) == 1536
    assert actual[["plateID", "well"]].equals(expected[["plateID", "well"]])
    scores = actual["b_score"].to_numpy()
    golden = expected[f"{feature}_{scope}"].to_numpy()
    np.testing.assert_allclose(scores, golden, rtol=1e-10, atol=1e-10)
    samples = actual["role"].eq("sample").to_numpy()
    np.testing.assert_array_equal(actual["hit_b_score"], samples & (np.abs(golden) >= 3))
    # Original R and NumPy may assign tiny roundoff residuals to exact ties.
    # No distinguishable reference scores may be reversed by spaCR's ranking.
    ranked = actual.loc[samples].sort_values("rank").index.to_numpy()
    assert np.all(np.diff(np.abs(golden[ranked])) <= 1e-10)
    assert sorted(actual.loc[samples, "rank"]) == list(range(1, samples.sum() + 1))
