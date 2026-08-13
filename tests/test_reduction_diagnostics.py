"""Is the projection about what the user thinks it is about?

Instruction 49, taking two ideas from starplast, which built the same picker
for a different table:

  * a group can be named as an input and carry almost nothing -- starplast
    caught one at 1.1% -- and nobody notices, because a projection always
    produces a picture;
  * a projection can separate objects on WHETHER THEY WERE MEASURED rather
    than on what was measured, and that reads as a phenotype.

The second is worse in spaCR than it was there. A cell with no pathogen has
NaN for every pathogen measurement, and reduce_dimensions median-fills rather
than dropping the row -- deliberately, because dropping loses every
measurement the object DID have. But a median fill puts every uninfected cell
at the same point on those axes, so the embedding can split infected from
uninfected on missingness alone. That split is real, reproducible, and not a
phenotype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import (
    group_variance_share,
    missingness_leak,
    reduce_dimensions,
)


def _table(n=400, seed=0, infected_fraction=0.5, noise=1.0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "cell_area": rng.normal(100, 20 * noise, n),
        "cell_perimeter": rng.normal(40, 5 * noise, n),
        "cell_channel_1_mean_intensity": rng.normal(1000, 300 * noise, n),
        "cell_channel_2_mean_intensity": rng.normal(900, 250 * noise, n),
    })
    infected = rng.random(n) < infected_fraction
    frame["pathogen_area"] = np.where(infected, rng.normal(30, 6, n), np.nan)
    frame["pathogen_channel_1_mean_intensity"] = np.where(
        infected, rng.normal(500, 90, n), np.nan)
    return frame, infected


# ---------------------------------------------------------------------------
# Which group is the picture actually about
# ---------------------------------------------------------------------------

def test_a_group_that_carries_nothing_is_visible_as_carrying_nothing():
    """The defect starplast caught at 1.1%: named as an input, contributing
    nothing, and invisible because a projection always draws something."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "loud_a": rng.normal(0, 50, 300),
        "loud_b": rng.normal(0, 50, 300),
        "flat_a": np.full(300, 7.0) + rng.normal(0, 1e-9, 300),
        "flat_b": np.full(300, 3.0) + rng.normal(0, 1e-9, 300),
    })
    share = group_variance_share(
        frame, {"loud": ["loud_a", "loud_b"], "flat": ["flat_a", "flat_b"]},
        scale=False)
    assert share.loc["loud", "share"] > 0.99
    assert share.loc["flat", "share"] < 0.01


def test_shares_are_ordered_largest_first():
    frame, _ = _table()
    share = group_variance_share(frame, {
        "morphology": ["cell_area", "cell_perimeter"],
        "intensity": [c for c in frame.columns if "intensity" in c]})
    assert list(share["share"]) == sorted(share["share"], reverse=True)


def test_scaling_off_lets_the_larger_numbers_win_and_that_is_the_point():
    """The same trap reduce_dimensions' own `scale` exists for."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "small": rng.normal(1, 0.1, 300),
        "big": rng.normal(1000, 100, 300),
    })
    unscaled = group_variance_share(frame, {"s": ["small"], "b": ["big"]},
                                    scale=False)
    scaled = group_variance_share(frame, {"s": ["small"], "b": ["big"]},
                                  scale=True)
    assert unscaled.loc["b", "share"] > 0.99
    assert 0.4 < scaled.loc["b", "share"] < 0.6


def test_a_column_in_two_groups_counts_in_both_and_the_frame_says_so():
    frame, _ = _table()
    share = group_variance_share(frame, {
        "cell": ["cell_area", "cell_channel_1_mean_intensity"],
        "intensity": ["cell_channel_1_mean_intensity",
                      "cell_channel_2_mean_intensity"]})
    assert share.attrs["overlapping"] is True


def test_column_counts_reflect_what_survived_the_coverage_filter():
    frame, _ = _table()
    frame["mostly_empty"] = np.nan
    frame.loc[frame.index[:5], "mostly_empty"] = 1.0
    share = group_variance_share(frame, {
        "morphology": ["cell_area", "cell_perimeter", "mostly_empty"]})
    assert share.loc["morphology", "columns"] == 2


def test_too_few_usable_columns_returns_an_empty_frame_with_its_columns():
    """A caller must be able to sort the result without a KeyError."""
    frame = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
    share = group_variance_share(frame, {"g": ["only"]})
    assert share.empty
    assert list(share.columns) == ["share", "columns"]


# ---------------------------------------------------------------------------
# Missingness, which is the one that produces a wrong conclusion
# ---------------------------------------------------------------------------

def test_a_projection_split_on_missingness_is_reported():
    """Uninfected cells all land on the median of every pathogen column, so
    the embedding separates them -- on the fact of measurement."""
    rng = np.random.default_rng(3)
    n = 400
    infected = rng.random(n) < 0.65
    frame = pd.DataFrame({"cell_area": rng.normal(100, 1, n)})
    # Many pathogen columns, so missingness dominates the matrix.
    for index in range(12):
        frame[f"pathogen_m{index}"] = np.where(
            infected, rng.normal(30, 5, n), np.nan)
    components = reduce_dimensions(frame, list(frame.columns), method="pca")
    leak = missingness_leak(components, frame, list(frame.columns))
    assert not leak.empty
    worst = leak.iloc[0]
    assert worst["column"].startswith("pathogen_")
    # THE COLLAPSE, not the displacement. A median fill puts the uninfected
    # cells in the MIDDLE of the infected ones, so the centroids barely move
    # (0.06 radii measured) while the missing group loses its spread
    # entirely (0.11). Asserting on the centroid alone would have passed a
    # detector that misses spaCR's actual artefact.
    assert worst["dispersion_ratio"] < 0.3, leak
    assert worst["severity"] > 0.5, leak


def test_a_table_with_nothing_missing_reports_nothing():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({f"m{i}": rng.normal(0, 1, 200) for i in range(6)})
    components = reduce_dimensions(frame, list(frame.columns), method="pca")
    leak = missingness_leak(components, frame, list(frame.columns))
    assert leak.empty
    assert list(leak.columns) == ["column", "missing_fraction",
                                  "centroid_gap", "dispersion_ratio",
                                  "severity"]


def test_a_column_with_too_few_on_either_side_is_skipped_not_reported():
    """A gap computed from four objects is noise, and reporting it would
    bury the real ones."""
    rng = np.random.default_rng(5)
    n = 300
    frame = pd.DataFrame({f"m{i}": rng.normal(0, 1, n) for i in range(4)})
    frame["rarely_missing"] = rng.normal(0, 1, n)
    frame.loc[frame.index[:4], "rarely_missing"] = np.nan
    components = reduce_dimensions(frame, list(frame.columns), method="pca")
    leak = missingness_leak(components, frame, list(frame.columns))
    assert "rarely_missing" not in set(leak["column"])


def test_the_gap_is_in_map_radii_so_it_compares_across_runs():
    """Scaling every coordinate must not change the answer."""
    rng = np.random.default_rng(6)
    n = 300
    infected = rng.random(n) < 0.65
    frame = pd.DataFrame({"a": rng.normal(0, 1, n)})
    for index in range(8):
        frame[f"p{index}"] = np.where(infected, rng.normal(5, 1, n), np.nan)
    components = reduce_dimensions(frame, list(frame.columns), method="pca")
    near = missingness_leak(components, frame, list(frame.columns))
    far = missingness_leak(components * 1000.0, frame, list(frame.columns))
    assert np.allclose(near["centroid_gap"].to_numpy(),
                       far["centroid_gap"].to_numpy())


def test_components_that_are_all_nan_are_not_a_crash():
    frame, _ = _table(n=100)
    components = pd.DataFrame({"PC1": np.nan, "PC2": np.nan},
                              index=frame.index)
    assert missingness_leak(components, frame, list(frame.columns)).empty


def test_a_column_absent_from_the_frame_is_ignored():
    frame, _ = _table(n=100)
    components = reduce_dimensions(frame, list(frame.columns), method="pca")
    leak = missingness_leak(components, frame, ["not_a_column"])
    assert leak.empty


def test_the_diagnostic_matches_the_matrix_the_reducer_actually_built():
    """A diagnostic computed on differently-prepared data describes a
    projection nobody ran."""
    frame, _ = _table()
    frame["barely_there"] = np.nan
    frame.loc[frame.index[:3], "barely_there"] = 1.0
    share = group_variance_share(frame, {"all": list(frame.columns)})
    # reduce_dimensions drops it at min_coverage; so must this.
    assert share.loc["all", "columns"] == len(frame.columns) - 1


def test_displacement_is_caught_too_even_when_nothing_collapses():
    """The other artefact: missing objects sitting somewhere else entirely.

    Constructed rather than fitted -- the components are handed in directly,
    so this pins the statistic and not a reducer's behaviour.
    """
    rng = np.random.default_rng(7)
    n = 400
    missing = np.zeros(n, bool)
    missing[: n // 2] = True
    coords = rng.normal(0, 1, (n, 2))
    coords[missing] += 20.0            # moved, but just as spread out
    components = pd.DataFrame(coords, columns=["PC1", "PC2"])
    frame = pd.DataFrame({"m": np.where(missing, np.nan, 1.0)},
                         index=components.index)
    leak = missingness_leak(components, frame, ["m"])
    row = leak.iloc[0]
    assert row["centroid_gap"] > 1.5
    assert 0.5 < row["dispersion_ratio"] < 2.0     # nothing collapsed
    assert row["severity"] == pytest.approx(row["centroid_gap"])


def test_a_clean_projection_scores_near_zero_on_both():
    rng = np.random.default_rng(8)
    n = 400
    missing = rng.random(n) < 0.5
    components = pd.DataFrame(rng.normal(0, 1, (n, 2)), columns=["PC1", "PC2"])
    frame = pd.DataFrame({"m": np.where(missing, np.nan, 1.0)},
                         index=components.index)
    row = missingness_leak(components, frame, ["m"]).iloc[0]
    assert row["centroid_gap"] < 0.3
    assert row["severity"] < 0.3
