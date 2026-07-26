"""Replication assay (Toxoplasma endodyogeny): ``spacr.submodules.analyze_replication``.

*Toxoplasma gondii* replicates by endodyogeny — two daughters inside one
mother — so a parasitophorous vacuole holds 1, 2, 4, 8 or 16 parasites. The
assay's job is to report the **distribution** of parasites-per-vacuole per
well, with everything off the power-of-two ladder kept in its own visible
bucket instead of being rounded into a neighbour.

Every fixture below is a hand-built ``measurements.db`` whose answer is known
by construction: the parasite centroids are placed on a grid so that the
vacuole membership, and therefore every bucket fraction, can be written down
before the code runs.

The single most important test in this file is
``test_two_vacuoles_in_one_host_cell_count_as_two`` — the counting unit is the
vacuole, and grouping on the host cell instead silently produces a plausible
but meaningless number.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# analyze_replication imports spacr.io / spacr.plot / spacr.settings lazily
# inside the call; pull the heavy chain in at collection time so it is not
# charged to whichever test happens to run first.
import spacr.io  # noqa: E402,F401
import spacr.plot  # noqa: E402,F401
import spacr.settings  # noqa: E402,F401
import spacr.sp_stats  # noqa: E402,F401
import spacr.submodules  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    """Never let a figure window open, never let figures accumulate."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


PARASITE_AREA = 100.0          # ~11.3 px equivalent diameter
PARASITE_DIAMETER = 2.0 * np.sqrt(PARASITE_AREA / np.pi)
# Default link factor is 1.5, so two centroids closer than ~16.9 px are one
# vacuole. Rosette members sit 8 px apart; separate vacuoles 400 px apart.
ROSETTE_SPACING = 8.0
VACUOLE_SPACING = 400.0


def _rosette(n, origin_x, origin_y, spacing=ROSETTE_SPACING):
    """Return ``n`` centroids in a compact chain, each within one link hop."""
    return [(origin_y, origin_x + i * spacing) for i in range(n)]


def write_db(root, vacuole_spec, extra_cell_wells=()):
    """Write ``<root>/measurements/measurements.db`` from an explicit vacuole spec.

    ``vacuole_spec`` is a list of dicts::

        {'row': 'r1', 'column': 'c1', 'field': 'f1',
         'cell': 1, 'n': 4}

    Every entry becomes one rosette of ``n`` parasites. Rosettes sharing a
    (field, cell) are placed ``VACUOLE_SPACING`` apart so they stay distinct
    vacuoles inside one host cell.

    ``extra_cell_wells`` is a list of ``(row, column)`` pairs that get host
    cells but no parasites at all — an uninfected well.
    """
    measurements = os.path.join(str(root), "measurements")
    os.makedirs(measurements, exist_ok=True)
    db = os.path.join(measurements, "measurements.db")

    pathogen_rows, cell_rows = [], []
    label = 0
    cells_seen = set()
    # How many rosettes have already been placed in each (field, cell).
    placed = {}

    for spec in vacuole_spec:
        row, column = spec["row"], spec["column"]
        field = spec.get("field", "f1")
        cell = spec["cell"]
        prcf = f"plate1_{row}_{column}_{field}"
        key = (prcf, cell)
        index = placed.get(key, 0)
        placed[key] = index + 1

        origin_x = 50.0 + index * VACUOLE_SPACING
        origin_y = 50.0 + index * VACUOLE_SPACING
        for (cy, cx) in _rosette(spec["n"], origin_x, origin_y):
            label += 1
            pathogen_rows.append({
                "object_label": label,
                "cell_id": cell,
                "plateID": "plate1", "rowID": row, "columnID": column,
                "fieldID": field, "prcf": prcf,
                "pathogen_area": PARASITE_AREA,
                "pathogen_equivalent_diameter_area": PARASITE_DIAMETER,
                "pathogen_channel_0_centroid_weighted-0": cy,
                "pathogen_channel_0_centroid_weighted-1": cx,
                "pathogen_channel_0_mean_intensity": 50.0,
            })

        if key not in cells_seen:
            cells_seen.add(key)
            cell_rows.append({
                "object_label": cell,
                "plateID": "plate1", "rowID": row, "columnID": column,
                "fieldID": field, "prcf": prcf,
                "cell_area": 20000.0,
                "cell_channel_0_mean_intensity": 100.0,
            })

    for i, (row, column) in enumerate(extra_cell_wells):
        cell_rows.append({
            "object_label": 900 + i,
            "plateID": "plate1", "rowID": row, "columnID": column,
            "fieldID": "f1", "prcf": f"plate1_{row}_{column}_f1",
            "cell_area": 20000.0,
            "cell_channel_0_mean_intensity": 100.0,
        })

    with sqlite3.connect(db) as con:
        pd.DataFrame(pathogen_rows).to_sql("pathogen", con, index=False,
                                           if_exists="replace")
        pd.DataFrame(cell_rows).to_sql("cell", con, index=False,
                                       if_exists="replace")
    return str(root)


def settings_for(src, **overrides):
    """Baseline settings: two conditions in columns c1/c2, nothing saved."""
    settings = {
        "src": src,
        "cell_types": None,
        "cell_plate_metadata": None,
        "pathogen_types": ["dmso", "drug"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "save": False,
        "verbose": False,
    }
    settings.update(overrides)
    return settings


# ---------------------------------------------------------------------------
# Bucketing primitives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16"),
    (32, ">16"), (64, ">16"),
    (3, "non_power_of_two"), (5, "non_power_of_two"),
    (6, "non_power_of_two"), (7, "non_power_of_two"),
    (9, "non_power_of_two"), (0, "non_power_of_two"),
])
def test_replication_bucket_only_powers_of_two_get_their_own_bucket(n, expected):
    from spacr.submodules import _replication_bucket
    assert _replication_bucket(n) == expected


def test_bucket_order_puts_non_power_of_two_last():
    """The ladder is ordinal; the QC bucket sits off the end of it, not at
    the top of it."""
    from spacr.submodules import _replication_bucket_order
    assert _replication_bucket_order(16) == [
        "1", "2", "4", "8", "16", ">16", "non_power_of_two"]
    assert _replication_bucket_order(8) == [
        "1", "2", "4", "8", ">8", "non_power_of_two"]


def test_link_distance_is_derived_from_the_parasites_themselves():
    from spacr.submodules import _derive_vacuole_link_distance
    df = pd.DataFrame({"pathogen_equivalent_diameter_area": [10.0, 12.0, 14.0]})
    assert _derive_vacuole_link_distance(df, "pathogen", 1.5) == pytest.approx(18.0)

    # No diameter column: fall back to the diameter of a disc with that area.
    df = pd.DataFrame({"pathogen_area": [100.0, 100.0]})
    expected = 2.0 * np.sqrt(100.0 / np.pi) * 2.0
    assert _derive_vacuole_link_distance(df, "pathogen", 2.0) == pytest.approx(expected)


def test_link_distance_without_size_columns_says_what_to_set():
    from spacr.submodules import _derive_vacuole_link_distance
    with pytest.raises(ValueError, match="vacuole_link_distance"):
        _derive_vacuole_link_distance(pd.DataFrame({"x": [1]}), "pathogen", 1.5)


def test_chi_pairwise_safety_check_rejects_what_scipy_rejects():
    """The shared proportion-bar helper slices two groups at a time; scipy
    refuses a slice with an all-zero column, and the helper divides by the
    number of comparisons. Both are checked before delegating."""
    from spacr.submodules import _chi_pairwise_is_safe

    dense = pd.DataFrame([[3, 4], [5, 2]], index=["a", "b"], columns=["1", "2"])
    assert _chi_pairwise_is_safe(dense) is True
    # A pair may be lopsided, as long as no whole column of the pair is empty.
    assert _chi_pairwise_is_safe(pd.DataFrame([[1, 0], [0, 1]])) is True

    # Three wells, one bucket each: the (a, b) slice leaves column '4' empty
    # and scipy rejects it.
    identity = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            index=["a", "b", "c"], columns=["1", "2", "4"])
    assert _chi_pairwise_is_safe(identity) is False

    # Only one group -> no pair to compare at all.
    assert _chi_pairwise_is_safe(pd.DataFrame([[3, 4]], index=["a"])) is False
    # An empty group is an all-zero row.
    assert _chi_pairwise_is_safe(pd.DataFrame([[3, 4], [0, 0]])) is False
    # Degenerate shapes.
    assert _chi_pairwise_is_safe(np.zeros((2, 0))) is False
    assert _chi_pairwise_is_safe(np.array([1, 2, 3])) is False


def test_sparse_per_well_table_still_draws_bars_with_empty_stats(tmp_path):
    """Three wells whose single vacuoles land in three different buckets
    cannot be chi-squared pairwise — every pair leaves the third bucket empty.
    The figures are still drawn and the omnibus statistics come back NaN,
    instead of the run dying inside the shared plot helper."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 1},
        {"row": "r1", "column": "c2", "cell": 1, "n": 2},
        {"row": "r1", "column": "c3", "cell": 1, "n": 4},
    ])
    out = analyze_replication(settings_for(
        src,
        pathogen_types=["dmso", "low", "high"],
        pathogen_plate_metadata=[["c1"], ["c2"], ["c3"]],
    ))

    figure = out["figures"]["per_well"]
    labels = sorted(t.get_text() for t in figure.axes[0].get_xticklabels())
    assert labels == ["plate1_r1_c1", "plate1_r1_c2", "plate1_r1_c3"]
    assert figure.axes[0].get_ylim() == (0.0, 1.0)
    assert np.isnan(out["chi_squared"]["p_value"].iloc[0])
    assert len(out["chi_squared_pairwise"]) == 0
    # The ordered test does not need that contingency table and still runs.
    assert len(out["comparisons"]) == 3
    assert set(out["comparisons"]["n1_power_of_two"]) == {1}


def test_centroid_columns_prefer_morphology_then_lowest_channel():
    from spacr.submodules import _find_centroid_columns
    plain = pd.DataFrame(columns=["pathogen_centroid-0", "pathogen_centroid-1",
                                  "pathogen_channel_0_centroid_weighted-0",
                                  "pathogen_channel_0_centroid_weighted-1"])
    assert _find_centroid_columns(plain) == ("pathogen_centroid-0",
                                             "pathogen_centroid-1")

    channels = pd.DataFrame(columns=[
        "pathogen_channel_2_centroid_weighted-0",
        "pathogen_channel_2_centroid_weighted-1",
        "pathogen_channel_1_centroid_weighted-0",
        "pathogen_channel_1_centroid_weighted-1",
    ])
    assert _find_centroid_columns(channels) == (
        "pathogen_channel_1_centroid_weighted-0",
        "pathogen_channel_1_centroid_weighted-1")

    assert _find_centroid_columns(pd.DataFrame(columns=["a"])) is None


# ---------------------------------------------------------------------------
# The exact-answer field: vacuoles of 1, 2, 4, 8 parasites
# ---------------------------------------------------------------------------

@pytest.fixture
def ladder_src(tmp_path):
    """One well (c1), four host cells, holding 1 / 2 / 4 / 8 parasites."""
    root = tmp_path / "plate1"
    spec = [
        {"row": "r1", "column": "c1", "cell": 1, "n": 1},
        {"row": "r1", "column": "c1", "cell": 2, "n": 2},
        {"row": "r1", "column": "c1", "cell": 3, "n": 4},
        {"row": "r1", "column": "c1", "cell": 4, "n": 8},
    ]
    return write_db(root, spec)


def test_ladder_field_gives_exact_bucket_fractions(ladder_src):
    """Four vacuoles of 1, 2, 4 and 8 parasites -> 25% in each of those
    buckets, 0% anywhere else, and 15 parasites in total."""
    from spacr.submodules import analyze_replication

    out = analyze_replication(settings_for(ladder_src))

    vacuoles = out["vacuoles"]
    assert len(vacuoles) == 4
    assert sorted(vacuoles["n_parasites"]) == [1, 2, 4, 8]
    assert sorted(vacuoles["replication_bucket"].astype(str)) == ["1", "2", "4", "8"]
    assert vacuoles["is_power_of_two"].all()
    assert sorted(vacuoles["doublings"]) == [0.0, 1.0, 2.0, 3.0]

    wells = out["wells"]
    well = wells[wells["prc"] == "plate1_r1_c1"].iloc[0]
    assert well["n_vacuoles"] == 4
    assert well["n_parasites"] == 15
    for bucket in ("1", "2", "4", "8"):
        assert well[f"frac_{bucket}"] == pytest.approx(0.25)
    assert well["frac_16"] == 0.0
    assert well["frac_gt16"] == 0.0
    assert well["frac_non_power_of_two"] == 0.0
    assert bool(well["qc_flag_non_power_of_two"]) is False

    # Median respects the discreteness: median of [1, 2, 4, 8] is 3.0, which
    # is *not* a legal vacuole size -- that is the point of also reporting the
    # full distribution and the median doubling index.
    assert well["median_parasites_per_vacuole"] == pytest.approx(3.0)
    assert well["median_doublings"] == pytest.approx(1.5)
    # The mean is reported with the fraction of vacuoles it came from.
    assert well["mean_parasites_per_vacuole"] == pytest.approx(15 / 4)
    assert well["mean_fraction_of_vacuoles"] == pytest.approx(1.0)


def test_vacuole_table_carries_plate_row_column_field_identity(ladder_src):
    from spacr.submodules import analyze_replication

    vacuoles = analyze_replication(settings_for(ladder_src))["vacuoles"]
    for column in ("plateID", "rowID", "columnID", "fieldID", "prc", "prcf",
                   "vacuole_id", "n_parasites"):
        assert column in vacuoles.columns
    assert set(vacuoles["plateID"]) == {"plate1"}
    assert set(vacuoles["rowID"]) == {"r1"}
    assert set(vacuoles["columnID"]) == {"c1"}
    assert set(vacuoles["fieldID"]) == {"f1"}
    assert vacuoles["vacuole_id"].nunique() == len(vacuoles)


# ---------------------------------------------------------------------------
# THE GROUPING KEY TEST — two vacuoles in one host cell
# ---------------------------------------------------------------------------

def test_two_vacuoles_in_one_host_cell_count_as_two(tmp_path):
    """One host cell holding a 2-parasite and a 4-parasite vacuole must be
    reported as TWO vacuoles of 2 and 4, never as one vacuole of 6.

    Grouping on the host cell is the single failure mode that produces a
    plausible-looking but meaningless replication number: 6 is not a power of
    two, so the merged answer would also silently inflate the
    ``non_power_of_two`` QC bucket and hide the real, healthy distribution.
    """
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 2},
        {"row": "r1", "column": "c1", "cell": 1, "n": 4},   # SAME host cell
    ])

    out = analyze_replication(settings_for(src))
    vacuoles = out["vacuoles"]

    assert out["vacuole_key"] == "spatial"
    assert len(vacuoles) == 2, "two vacuoles in one host cell were merged"
    assert sorted(vacuoles["n_parasites"]) == [2, 4]
    assert 6 not in set(vacuoles["n_parasites"])
    # Both vacuoles are on the ladder, so nothing lands in the QC bucket.
    assert sorted(vacuoles["replication_bucket"].astype(str)) == ["2", "4"]
    assert vacuoles["is_power_of_two"].all()
    # ... and they really do share a host cell.
    assert set(vacuoles["cell_id"]) == {1}


def test_cell_id_grouping_is_the_documented_wrong_answer(tmp_path):
    """``vacuole_key='cell_id'`` is offered for low-MOI screens where a host
    cell holds one vacuole. Asked to group the two-vacuole host cell above, it
    returns the merged count -- and says so via ``vacuole_key``."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 2},
        {"row": "r1", "column": "c1", "cell": 1, "n": 4},
    ])

    out = analyze_replication(settings_for(src, vacuole_key="cell_id"))
    assert out["vacuole_key"] == "cell_id"
    assert list(out["vacuoles"]["n_parasites"]) == [6]
    # And the merge is visible in the QC bucket rather than hidden.
    assert list(out["vacuoles"]["replication_bucket"].astype(str)) == \
        ["non_power_of_two"]


def test_vacuoles_in_different_fields_never_merge(tmp_path):
    """Same host-cell label in two different fields is two different cells."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "field": "f1", "cell": 1, "n": 2},
        {"row": "r1", "column": "c1", "field": "f2", "cell": 1, "n": 2},
    ])
    vacuoles = analyze_replication(settings_for(src))["vacuoles"]
    assert len(vacuoles) == 2
    assert set(vacuoles["fieldID"]) == {"f1", "f2"}
    assert list(vacuoles["n_parasites"]) == [2, 2]


# ---------------------------------------------------------------------------
# non_power_of_two is reported, never folded away
# ---------------------------------------------------------------------------

def test_three_parasite_vacuole_lands_in_non_power_of_two(tmp_path):
    """A rosette of 3 is a segmentation error or an asynchronous vacuole. It
    goes in its own bucket and is visible in every output table."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 3},
        {"row": "r1", "column": "c1", "cell": 2, "n": 4},
        {"row": "r1", "column": "c1", "cell": 3, "n": 4},
        {"row": "r1", "column": "c1", "cell": 4, "n": 4},
    ])

    out = analyze_replication(settings_for(src))

    vacuoles = out["vacuoles"]
    odd = vacuoles[vacuoles["n_parasites"] == 3]
    assert len(odd) == 1
    assert odd["replication_bucket"].astype(str).iloc[0] == "non_power_of_two"
    assert bool(odd["is_power_of_two"].iloc[0]) is False
    assert np.isnan(odd["doublings"].iloc[0])
    # It was NOT rounded into the 2 or 4 bucket.
    assert int((vacuoles["replication_bucket"].astype(str) == "4").sum()) == 3
    assert int((vacuoles["replication_bucket"].astype(str) == "2").sum()) == 0

    well = out["wells"].iloc[0]
    assert well["n_non_power_of_two"] == 1
    assert well["frac_non_power_of_two"] == pytest.approx(0.25)
    assert well["non_power_of_two_fraction"] == pytest.approx(0.25)
    # 25% > the 20% default threshold -> the well is flagged.
    assert bool(well["qc_flag_non_power_of_two"]) is True
    # The mean is computed only from the three trustworthy vacuoles, and the
    # fraction it came from is reported next to it.
    assert well["mean_parasites_per_vacuole"] == pytest.approx(4.0)
    assert well["mean_fraction_of_vacuoles"] == pytest.approx(0.75)
    assert well["n_power_of_two"] == 3

    summary = out["summary"].iloc[0]
    assert summary["frac_non_power_of_two"] == pytest.approx(0.25)
    assert bool(summary["qc_flag_non_power_of_two"]) is True


def test_qc_flag_stays_down_below_the_threshold(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    spec = [{"row": "r1", "column": "c1", "cell": i, "n": 4} for i in range(1, 10)]
    spec.append({"row": "r1", "column": "c1", "cell": 10, "n": 3})
    src = write_db(root, spec)

    well = analyze_replication(settings_for(src))["wells"].iloc[0]
    assert well["frac_non_power_of_two"] == pytest.approx(0.1)
    assert bool(well["qc_flag_non_power_of_two"]) is False


def test_vacuoles_above_the_named_ladder_get_their_own_bucket(tmp_path):
    """A 32-parasite vacuole is still a power of two: it keeps its ordinal
    place in the '>16' bucket rather than being called a segmentation error."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 32},
        {"row": "r1", "column": "c1", "cell": 2, "n": 16},
    ])
    vacuoles = analyze_replication(settings_for(src))["vacuoles"]
    big = vacuoles[vacuoles["n_parasites"] == 32].iloc[0]
    assert big["replication_bucket"] == ">16"
    assert bool(big["is_power_of_two"]) is True
    assert big["doublings"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Degenerate wells
# ---------------------------------------------------------------------------

def test_uninfected_well_reports_zeros_not_nan(tmp_path):
    """A well with host cells but no parasites must appear in the per-well
    table with zeros — not vanish, not divide by zero, not produce NaN."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(
        root,
        [{"row": "r1", "column": "c1", "cell": 1, "n": 4}],
        extra_cell_wells=[("r1", "c2")],
    )

    out = analyze_replication(settings_for(src))
    wells = out["wells"]

    empty = wells[wells["prc"] == "plate1_r1_c2"]
    assert len(empty) == 1, "the uninfected well disappeared from the output"
    empty = empty.iloc[0]
    assert empty["n_vacuoles"] == 0
    assert empty["n_parasites"] == 0
    for bucket in ("1", "2", "4", "8", "16", "gt16", "non_power_of_two"):
        assert empty[f"n_{bucket}"] == 0
        assert empty[f"frac_{bucket}"] == 0.0
    assert empty["median_parasites_per_vacuole"] == 0.0
    assert empty["median_doublings"] == 0.0
    assert empty["mean_parasites_per_vacuole"] == 0.0
    assert empty["mean_fraction_of_vacuoles"] == 0.0
    assert bool(empty["qc_flag_non_power_of_two"]) is False

    numeric = empty[[c for c in wells.columns
                     if pd.api.types.is_numeric_dtype(wells[c])]]
    assert not numeric.isna().any(), f"NaN in an uninfected well: {numeric}"


def test_single_vacuole_well_does_not_crash_the_stats_path(tmp_path):
    """One well, one condition, one vacuole. The shared proportion-bar helper
    divides by the number of pairwise comparisons, so this is exactly the
    input that used to raise ZeroDivisionError."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])

    out = analyze_replication(settings_for(src))

    assert len(out["vacuoles"]) == 1
    assert out["vacuoles"]["n_parasites"].iloc[0] == 4
    assert len(out["wells"]) == 1
    assert out["wells"]["frac_4"].iloc[0] == pytest.approx(1.0)
    assert len(out["summary"]) == 1
    assert out["summary"]["median_parasites_per_vacuole"].iloc[0] == 4.0
    # One group -> no pair to compare, and an empty (but fully typed) table.
    assert len(out["comparisons"]) == 0
    assert "p_value_adj" in out["comparisons"].columns
    assert len(out["figures"]) == 2


def test_extracellular_parasites_are_excluded_by_default(tmp_path):
    """cell_id == 0 means the object overlapped no host cell. It has no
    vacuole, so it cannot enter a per-vacuole count."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 4},
        {"row": "r1", "column": "c1", "cell": 0, "n": 2},   # extracellular
    ])

    kept = analyze_replication(settings_for(src))["vacuoles"]
    assert list(kept["n_parasites"]) == [4]

    loosened = analyze_replication(
        settings_for(src, require_host_cell=False))["vacuoles"]
    assert sorted(loosened["n_parasites"]) == [2, 4]


def test_all_objects_filtered_away_fails_loudly(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    with pytest.raises(ValueError, match="No parasite objects left"):
        analyze_replication(settings_for(src, min_parasite_area=10 ** 9))


def test_area_filters_drop_debris_and_clumps(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    # Every synthetic parasite has area 100; a max below that removes them all.
    with pytest.raises(ValueError, match="No parasite objects left"):
        analyze_replication(settings_for(src, max_parasite_area=50))
    # A window that contains 100 keeps everything.
    out = analyze_replication(
        settings_for(src, min_parasite_area=50, max_parasite_area=200))
    assert out["vacuoles"]["n_parasites"].iloc[0] == 4


def test_wells_with_no_condition_are_dropped_loudly(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c9", "cell": 1, "n": 4}])
    with pytest.raises(ValueError, match="empty 'condition'"):
        analyze_replication(settings_for(src))


def test_missing_group_column_reports_available_columns(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    with pytest.raises(KeyError, match="Available columns"):
        analyze_replication(settings_for(src, group_column="nope"))


def test_unknown_vacuole_key_reports_available_columns(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    with pytest.raises(KeyError, match="Available columns"):
        analyze_replication(settings_for(src, vacuole_key="not_a_column"))


def test_empty_settings_fail_loudly():
    from spacr.submodules import analyze_replication
    with pytest.raises(Exception):
        analyze_replication({})


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------

def _two_condition_src(tmp_path, counts_c1, counts_c2, name="plate1"):
    root = tmp_path / name
    spec = []
    for i, n in enumerate(counts_c1, start=1):
        spec.append({"row": "r1", "column": "c1", "cell": i, "n": n})
    for i, n in enumerate(counts_c2, start=1):
        spec.append({"row": "r1", "column": "c2", "cell": i, "n": n})
    return write_db(root, spec)


def test_clearly_different_distributions_are_rejected(tmp_path):
    """DMSO replicating to 8/16, drug stalled at 1/2: the ordered test must
    reject, and the direction must show in the rank-biserial effect size."""
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [8] * 10 + [16] * 10, [1] * 10 + [2] * 10)
    out = analyze_replication(settings_for(src))

    comparisons = out["comparisons"]
    assert len(comparisons) == 1
    result = comparisons.iloc[0]
    assert result["test"].startswith("Mann-Whitney U")
    assert result["p_value"] < 0.001
    assert result["p_value_adj"] < 0.001
    # Group order follows first appearance: dmso (c1) then drug (c2).
    assert result["group1"] == "dmso" and result["group2"] == "drug"
    assert result["median_doublings_1"] == pytest.approx(3.5)
    assert result["median_doublings_2"] == pytest.approx(0.5)
    # Complete separation upward -> rank-biserial pinned at +1.
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["n1_power_of_two"] == 20
    assert result["n2_power_of_two"] == 20
    assert result["chi_squared_p_value"] < 0.001


def test_identical_distributions_are_not_rejected(tmp_path):
    from spacr.submodules import analyze_replication

    counts = [1, 2, 2, 4, 4, 4, 8, 8, 8, 8]
    src = _two_condition_src(tmp_path, counts, counts)
    out = analyze_replication(settings_for(src))

    result = out["comparisons"].iloc[0]
    assert result["p_value"] == pytest.approx(1.0)
    assert result["p_value_adj"] == pytest.approx(1.0)
    assert result["rank_biserial"] == pytest.approx(0.0)
    assert result["median_doublings_1"] == result["median_doublings_2"]
    assert result["chi_squared_p_value"] == pytest.approx(1.0)


def test_comparison_matches_scipy_run_on_the_same_doubling_indices(tmp_path):
    """The reported statistic is a Mann-Whitney U on log2(parasites), not on
    the raw counts and not a t-test."""
    from scipy.stats import mannwhitneyu
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [1, 2, 4, 8, 8], [1, 1, 2, 2, 4])
    out = analyze_replication(settings_for(src))
    vacuoles = out["vacuoles"]

    left = vacuoles[(vacuoles["condition"] == "dmso")
                    & vacuoles["is_power_of_two"]]["doublings"]
    right = vacuoles[(vacuoles["condition"] == "drug")
                     & vacuoles["is_power_of_two"]]["doublings"]
    statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")

    result = out["comparisons"].iloc[0]
    assert result["u_statistic"] == pytest.approx(statistic)
    assert result["p_value"] == pytest.approx(p_value)


def test_non_power_of_two_vacuoles_are_kept_out_of_the_ordered_test(tmp_path):
    """A vacuole of 3 has no doubling index, so it cannot enter the rank test
    — but it is still counted in n1/n2 and in the chi-squared table."""
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 4, 3], [4, 4, 4, 4])
    out = analyze_replication(settings_for(src))

    result = out["comparisons"].iloc[0]
    assert result["n1"] == 4
    assert result["n1_power_of_two"] == 3
    assert result["non_power_of_two_fraction_1"] == pytest.approx(0.25)
    assert result["non_power_of_two_fraction_2"] == 0.0
    # Ranks on [2,2,2] vs [2,2,2,2] cannot separate.
    assert result["p_value"] == pytest.approx(1.0)


def test_a_condition_with_no_power_of_two_vacuoles_yields_nan_not_a_crash(tmp_path):
    """Total segmentation failure in one arm: the ordered test has nothing to
    rank, so it reports NaN rather than inventing a p-value."""
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4] * 8, [3, 3, 3, 5, 5, 5, 7, 7])
    out = analyze_replication(settings_for(src))

    result = out["comparisons"].iloc[0]
    assert result["n2"] == 8
    assert result["n2_power_of_two"] == 0
    assert np.isnan(result["p_value"])
    assert np.isnan(result["rank_biserial"])
    assert np.isnan(result["p_value_adj"])
    # The omnibus chi-squared still has a table to work on.
    assert result["chi_squared_p_value"] < 0.05
    assert result["non_power_of_two_fraction_2"] == pytest.approx(1.0)


def test_summary_reports_one_row_per_condition(tmp_path):
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 8, 8], [1, 1, 2, 2])
    summary = analyze_replication(settings_for(src))["summary"]

    assert set(summary["condition"]) == {"dmso", "drug"}
    dmso = summary[summary["condition"] == "dmso"].iloc[0]
    drug = summary[summary["condition"] == "drug"].iloc[0]
    assert dmso["n_vacuoles"] == 4 and drug["n_vacuoles"] == 4
    assert dmso["frac_4"] == pytest.approx(0.5)
    assert dmso["frac_8"] == pytest.approx(0.5)
    assert drug["frac_1"] == pytest.approx(0.5)
    assert drug["frac_2"] == pytest.approx(0.5)
    assert dmso["median_doublings"] == pytest.approx(2.5)
    assert drug["median_doublings"] == pytest.approx(0.5)
    assert dmso["n_wells"] == 1


# ---------------------------------------------------------------------------
# Output files and figures
# ---------------------------------------------------------------------------

def test_save_writes_the_csvs_next_to_the_other_submodule_outputs(tmp_path):
    """Sibling submodules write to <src>/results/<function name>/; so does
    this one."""
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 8], [1, 2, 2])
    out = analyze_replication(settings_for(src, save=True))

    output_dir = os.path.join(src, "results", "analyze_replication")
    assert os.path.isdir(output_dir)

    vacuoles = pd.read_csv(os.path.join(output_dir, "vacuole_counts.csv"))
    assert len(vacuoles) == len(out["vacuoles"])
    for column in ("plateID", "rowID", "columnID", "fieldID", "vacuole_id",
                   "n_parasites", "replication_bucket", "doublings",
                   "is_power_of_two", "condition"):
        assert column in vacuoles.columns
    assert sorted(vacuoles["n_parasites"]) == [1, 2, 2, 4, 4, 8]

    wells = pd.read_csv(os.path.join(output_dir, "well_distribution.csv"))
    for column in ("prc", "n_vacuoles", "frac_1", "frac_2", "frac_4", "frac_8",
                   "frac_16", "frac_gt16", "frac_non_power_of_two",
                   "median_parasites_per_vacuole", "median_doublings",
                   "mean_parasites_per_vacuole", "mean_fraction_of_vacuoles",
                   "qc_flag_non_power_of_two"):
        assert column in wells.columns, column
    assert len(wells) == 2

    summary = pd.read_csv(os.path.join(output_dir, "condition_summary.csv"))
    assert set(summary["condition"]) == {"dmso", "drug"}

    comparisons = pd.read_csv(os.path.join(output_dir,
                                           "condition_comparisons.csv"))
    assert len(comparisons) == 1
    assert "p_value" in comparisons.columns and "rank_biserial" in comparisons.columns

    for name in ("chi_squared_results.csv", "chi_squared_pairwise_results.csv"):
        assert os.path.getsize(os.path.join(output_dir, name)) > 0

    for name in ("parasites_per_vacuole_per_well.pdf",
                 "parasites_per_vacuole_by_condition.pdf"):
        path = os.path.join(output_dir, name)
        assert os.path.getsize(path) > 0
        with open(path, "rb") as handle:
            assert handle.read(4) == b"%PDF"

    # The settings snapshot lands where every other pipeline puts it.
    assert os.path.isfile(os.path.join(src, "settings",
                                       "analyze_replication.csv"))


def test_figures_are_created_and_closed(tmp_path):
    """Two figures come back in the output dict and neither is left open in
    pyplot's registry — a batch over 40 plates must not leak 80 figures."""
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 8], [1, 2, 2])
    plt.close("all")
    assert plt.get_fignums() == []

    out = analyze_replication(settings_for(src))

    assert set(out["figures"]) == {"per_well", "by_condition"}
    assert plt.get_fignums() == [], "analyze_replication leaked a figure"

    for figure in out["figures"].values():
        axes = figure.axes[0]
        assert axes.get_ylim() == (0.0, 1.0)
        assert axes.get_legend().get_title().get_text() == "Parasites per vacuole"
        # Stacked proportion bars: one bar container per bucket present.
        assert len(axes.containers) >= 1

    assert out["figures"]["per_well"].axes[0].get_title() == \
        "Parasites per vacuole — per well"
    assert out["figures"]["by_condition"].axes[0].get_title() == \
        "Parasites per vacuole — by condition"


def test_per_well_bars_have_one_bar_per_well(tmp_path):
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 8], [1, 2, 2])
    out = analyze_replication(settings_for(src))
    labels = [t.get_text()
              for t in out["figures"]["per_well"].axes[0].get_xticklabels()]
    assert sorted(labels) == ["plate1_r1_c1", "plate1_r1_c2"]


def test_level_well_aggregates_conditions_across_wells(tmp_path):
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [4, 4, 8], [1, 2, 2])
    out = analyze_replication(settings_for(src, level="well"))
    assert len(out["vacuoles"]) == 6
    assert out["chi_squared"]["p_value"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# Alternative layouts
# ---------------------------------------------------------------------------

def test_parasite_count_column_uses_a_precomputed_per_vacuole_count(tmp_path):
    """When the table is already one row per vacuole with a parasite count
    (e.g. an organelle-in-pathogen summary), that column is used verbatim."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 1},
        {"row": "r1", "column": "c1", "cell": 2, "n": 1},
    ])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame["organelle_summary_organelle_count"] = [4, 8]
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_replication(settings_for(
        src, parasite_count_column="organelle_summary_organelle_count"))
    assert sorted(out["vacuoles"]["n_parasites"]) == [4, 8]
    assert sorted(out["vacuoles"]["replication_bucket"].astype(str)) == ["4", "8"]


def test_missing_parasite_count_column_is_named_in_the_error(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    with pytest.raises(KeyError, match="ghost_count"):
        analyze_replication(settings_for(src, parasite_count_column="ghost_count"))


def test_explicit_vacuole_column_wins_over_clustering(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        # Split the one physical rosette into two declared vacuoles.
        frame["vacuole_id"] = ["a", "a", "b", "b"]
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_replication(settings_for(src))
    assert out["vacuole_key"] == "vacuole_id"
    assert sorted(out["vacuoles"]["n_parasites"]) == [2, 2]


def test_without_centroids_the_fallback_is_announced(tmp_path, capsys):
    """No centroid columns -> the host cell is the only grouping left, and the
    warning says exactly what that costs."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 2},
        {"row": "r1", "column": "c1", "cell": 1, "n": 2},
    ])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame = frame.drop(columns=[c for c in frame.columns if "centroid" in c])
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_replication(settings_for(src))
    assert out["vacuole_key"] == "cell_id"
    assert list(out["vacuoles"]["n_parasites"]) == [4]
    assert "combined parasite count" in capsys.readouterr().out


def test_no_host_cell_column_falls_back_to_one_vacuole_per_object(tmp_path,
                                                                 capsys):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 3}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame = frame.drop(columns=["cell_id"])
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_replication(settings_for(src))
    assert out["vacuole_key"] == "object"
    assert list(out["vacuoles"]["n_parasites"]) == [1, 1, 1]
    assert "its own vacuole" in capsys.readouterr().out


def test_spatial_without_centroids_says_what_to_do(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 2}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame = frame.drop(columns=[c for c in frame.columns if "centroid" in c])
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    with pytest.raises(KeyError, match="vacuole_key='cell_id'"):
        analyze_replication(settings_for(src, vacuole_key="spatial"))


def test_link_distance_can_be_set_explicitly(tmp_path):
    """Shrinking the link distance below the rosette spacing splits one
    vacuole into singletons — the knob really is what joins parasites."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])

    joined = analyze_replication(settings_for(src))
    assert list(joined["vacuoles"]["n_parasites"]) == [4]
    assert joined["vacuole_link_distance"] == pytest.approx(
        PARASITE_DIAMETER * 1.5)

    split = analyze_replication(settings_for(src, vacuole_link_distance=1.0))
    assert sorted(split["vacuoles"]["n_parasites"]) == [1, 1, 1, 1]
    assert split["vacuole_link_distance"] == pytest.approx(1.0)


def test_non_finite_centroids_become_their_own_vacuoles(tmp_path):
    """A parasite whose centroid could not be computed cannot be clustered;
    it must not silently join every other unplaceable object."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame.loc[2:3, "pathogen_channel_0_centroid_weighted-0"] = np.nan
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    vacuoles = analyze_replication(settings_for(src))["vacuoles"]
    # Two placeable parasites form one vacuole; the two NaN ones stand alone.
    assert sorted(vacuoles["n_parasites"]) == [1, 1, 2]


def test_max_parasites_per_vacuole_renames_the_top_bucket(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [
        {"row": "r1", "column": "c1", "cell": 1, "n": 8},
        {"row": "r1", "column": "c1", "cell": 2, "n": 2},
    ])
    out = analyze_replication(settings_for(src, max_parasites_per_vacuole=4))
    buckets = list(out["vacuoles"]["replication_bucket"].cat.categories)
    assert buckets == ["1", "2", "4", ">4", "non_power_of_two"]
    assert set(out["vacuoles"]["replication_bucket"].astype(str)) == {">4", "2"}
    assert "frac_gt4" in out["wells"].columns


def test_change_plate_relabels_each_source(tmp_path):
    from spacr.submodules import analyze_replication

    src_a = _two_condition_src(tmp_path, [4], [2], name="plateA")
    src_b = _two_condition_src(tmp_path, [8], [1], name="plateB")
    out = analyze_replication(settings_for([src_a, src_b], change_plate=True))
    assert set(out["vacuoles"]["plateID"]) == {"plate1", "plate2"}
    assert len(out["vacuoles"]) == 4


def test_verbose_reports_the_grouping_and_the_flagged_wells(tmp_path, capsys):
    from spacr.submodules import analyze_replication

    src = _two_condition_src(tmp_path, [3, 3, 4], [4, 4, 4])
    analyze_replication(settings_for(src, verbose=True))
    captured = capsys.readouterr().out
    assert "vacuole_key='spatial'" in captured
    assert "non_power_of_two threshold" in captured
    assert "Parasites-per-vacuole comparisons" in captured


def test_a_database_without_a_cell_table_still_runs(tmp_path):
    """Well seeding is best-effort: a measurements.db holding only the
    pathogen table loses the uninfected-well rows, not the whole assay."""
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        con.execute("DROP TABLE cell")

    out = analyze_replication(settings_for(src))
    assert list(out["vacuoles"]["n_parasites"]) == [4]
    assert list(out["wells"]["prc"]) == ["plate1_r1_c1"]


def test_a_table_without_the_well_columns_is_rejected(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 4}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame = frame.drop(columns=["fieldID"])
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    with pytest.raises(ValueError, match="has no 'fieldID' column"):
        analyze_replication(settings_for(src))


def test_settings_module_defaults_take_over_when_registered(tmp_path,
                                                            monkeypatch):
    """spacr.settings owns every pipeline's defaults. Until it registers this
    one the local copy fills in; once it does, its values win."""
    import spacr.settings as spacr_settings
    from spacr.submodules import analyze_replication

    seen = {}

    def fake_defaults(settings):
        seen["called"] = True
        # A value the local fallback would set differently.
        settings.setdefault("max_parasites_per_vacuole", 4)
        return settings

    monkeypatch.setattr(spacr_settings, "set_analyze_replication_defaults",
                        fake_defaults, raising=False)

    root = tmp_path / "plate1"
    src = write_db(root, [{"row": "r1", "column": "c1", "cell": 1, "n": 8}])
    out = analyze_replication(settings_for(src))

    assert seen.get("called") is True
    # settings.py's 4 beat the local fallback's 16, so 8 is above the ladder.
    assert list(out["vacuoles"]["replication_bucket"].astype(str)) == [">4"]


def test_well_distribution_helper_handles_an_empty_vacuole_table():
    """Called directly with no vacuoles at all, every seeded well is reported
    with zeros rather than dividing by zero."""
    from spacr.submodules import (_replication_bucket_order,
                                  _replication_well_distribution)

    buckets = _replication_bucket_order(16)
    empty = pd.DataFrame(columns=["plateID", "rowID", "columnID", "prc",
                                  "condition", "n_parasites",
                                  "replication_bucket", "is_power_of_two",
                                  "doublings"])
    seed = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                          "columnID": "c1", "prc": "plate1_r1_c1",
                          "condition": "dmso"}])

    wells = _replication_well_distribution(empty, "condition", buckets,
                                           wells=seed)
    assert len(wells) == 1
    assert wells["n_vacuoles"].iloc[0] == 0
    assert wells["frac_4"].iloc[0] == 0.0
    assert wells["median_parasites_per_vacuole"].iloc[0] == 0.0


def test_seeding_wells_from_cells_can_be_switched_off(tmp_path):
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    src = write_db(
        root,
        [{"row": "r1", "column": "c1", "cell": 1, "n": 4}],
        extra_cell_wells=[("r1", "c2")],
    )
    wells = analyze_replication(
        settings_for(src, seed_wells_from_cells=False))["wells"]
    assert set(wells["prc"]) == {"plate1_r1_c1"}
