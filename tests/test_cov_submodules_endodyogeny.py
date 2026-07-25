"""Coverage for ``spacr.submodules.analyze_endodyogeny``.

``analyze_endodyogeny`` is the replication assay: it reads a real
``measurements.db``, converts a compartment *area* into a pseudo *volume*
(``area ** 1.5``), buckets every object into log2 volume-doubling bins and
then chi-squared tests the bin proportions between experimental groups.

Everything here drives the *real* entry point over a hand-built sqlite
``measurements.db`` (cell + pathogen + png_list tables) so the whole
``_read_db`` -> ``_split_data`` -> merge -> ``annotate_conditions`` ->
binning -> ``plot_proportion_stacked_bars`` chain is exercised on CPU with
no network and no GPU.

The synthetic DB is deliberately tiny (24 objects) and every area is a
round number so the expected volume bin of every single row can be
recomputed independently in the assertions.
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

# analyze_endodyogeny imports spacr.io / spacr.plot / spacr.settings lazily
# *inside* the call, which would otherwise charge the whole heavy
# cellpose/torch/seaborn import chain to whichever test happens to run first.
# Pull them in at collection time instead. The symbol under test is still
# imported per-test below.
import spacr.io  # noqa: E402,F401
import spacr.plot  # noqa: E402,F401
import spacr.settings  # noqa: E402,F401
import spacr.sp_stats  # noqa: E402,F401
import spacr.submodules  # noqa: E402,F401
from statsmodels.stats.multitest import multipletests  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    """Never let a figure window open, never let figures accumulate."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


# Areas chosen so that (area ** 1.5) lands in four distinct doubling bins
# relative to min_volume_bin = 500 ** 1.5 = 11180.34:
#   ~600  -> bin 1   [11180.34,  22360.68)
#   ~1000 -> bin 2   [22360.68,  44721.36)
#   ~1600 -> bin 3   [44721.36,  89442.72)
#   ~2600 -> bin 4   [89442.72, 178885.44)
AREAS_C1 = [600.0, 620.0, 640.0, 1000.0, 1050.0, 1100.0,
            1600.0, 1650.0, 1700.0, 2600.0, 2700.0, 2800.0]
AREAS_C2 = [610.0, 1010.0, 1020.0, 1030.0, 1610.0, 1620.0,
            1630.0, 1640.0, 2610.0, 2620.0, 2630.0, 2640.0]

MIN_VOLUME_BIN = 500 ** 1.5


def _write_endodyogeny_db(root, areas_by_column):
    """Write ``<root>/measurements/measurements.db`` with the tables the
    merge layer of :func:`spacr.io._read_and_merge_data` expects.

    One pathogen per cell (so ``pathogen_limit=1`` keeps everything and the
    per-object summed ``pathogen_area`` is the object's own area), plus a
    ``png_list`` table carrying an integer ``predictions`` class column.
    """
    meas = os.path.join(str(root), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")

    cell_rows, pathogen_rows, png_rows = [], [], []
    obj = 0
    for column, areas in areas_by_column.items():
        for k, area in enumerate(areas):
            obj += 1
            row = f"r{(k % 3) + 1}"
            field = f"f{(k % 2) + 1}"
            base = {
                "plateID": "plate1", "rowID": row, "columnID": column,
                "fieldID": field, "prcf": f"plate1_{row}_{column}_{field}",
            }
            cell_rows.append({**base, "object_label": obj,
                              "cell_area": 20000.0 + float(area),
                              "cell_channel_0_mean_intensity": 100.0 + k})
            # cell_id is the INTEGER parent label here — spacr prefixes 'o'.
            pathogen_rows.append({**base, "object_label": obj, "cell_id": obj,
                                  "pathogen_area": float(area),
                                  "pathogen_channel_0_mean_intensity": 50.0 + k})
            # png_list stores the already-prefixed 'o<N>' string form.
            png_rows.append({**base, "cell_id": f"o{obj}",
                             "png_path": f"/x/plate1_{row}_{column}_{field}_o{obj}.png",
                             "predictions": k % 2})

    with sqlite3.connect(db) as con:
        pd.DataFrame(cell_rows).to_sql("cell", con, index=False, if_exists="replace")
        pd.DataFrame(pathogen_rows).to_sql("pathogen", con, index=False,
                                           if_exists="replace")
        pd.DataFrame(png_rows).to_sql("png_list", con, index=False,
                                      if_exists="replace")
    return db


@pytest.fixture
def endo_src(tmp_path):
    """A src directory holding the standard 24-object endodyogeny DB."""
    root = tmp_path / "plate1"
    _write_endodyogeny_db(root, {"c1": AREAS_C1, "c2": AREAS_C2})
    return str(root)


def _settings(src, **overrides):
    """Baseline settings: pixel units, object level, nothing saved."""
    settings = {
        "src": src,
        "tables": ["cell", "pathogen"],
        "cell_types": ["Hela"],
        "cell_plate_metadata": None,
        "pathogen_types": ["nc", "pc"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "um_per_px": None,
        "min_area_bin": 500,
        "max_area": 10 ** 9,
        "level": "object",
        "nuclei_limit": 10,
        "pathogen_limit": 1,
        "verbose": False,
        "save": False,
    }
    settings.update(overrides)
    return settings


def _expected_bin_index(volume, min_volume_bin=MIN_VOLUME_BIN):
    """Independent re-implementation of the doubling-bin assignment."""
    return int(np.floor(np.log2(volume / min_volume_bin))) + 1


# ---------------------------------------------------------------------------
# Happy path — object level, pixel units
# ---------------------------------------------------------------------------

def test_endodyogeny_bins_volumes_and_runs_chi_squared(endo_src):
    """Every object is binned by log2 volume doubling and the two
    pathogen groups are compared with a chi-squared test."""
    from spacr.submodules import analyze_endodyogeny
    from scipy.stats import chi2_contingency

    out = analyze_endodyogeny(_settings(endo_src, verbose=True))

    assert set(out) == {"data", "chi_squared"}
    data = out["data"]
    # Nothing was lost: 24 objects in, 24 objects out.
    assert len(data) == len(AREAS_C1) + len(AREAS_C2)

    # volume == area ** 1.5 (pixel units: um_per_px is None so no scaling).
    assert np.allclose(data["pathogen_volume"], data["pathogen_area"] ** 1.5)
    assert set(data["pathogen_area"]) == set(AREAS_C1) | set(AREAS_C2)

    # Every row's bin_index matches an independent computation.
    expected = [_expected_bin_index(v) for v in data["pathogen_volume"]]
    assert list(data["bin_index"]) == expected
    assert sorted(set(expected)) == [1, 2, 3, 4]

    # The bin column is an ORDERED categorical whose categories run from the
    # smallest volume range to the largest.
    binned = data["pathogen_volume_bin"]
    assert isinstance(binned.dtype, pd.CategoricalDtype)
    assert binned.cat.ordered is True
    assert not binned.isna().any()
    lower_edges = [float(c.split("-")[0]) for c in binned.cat.categories]
    assert lower_edges == sorted(lower_edges)
    assert lower_edges[0] == pytest.approx(MIN_VOLUME_BIN, rel=1e-6)
    # Categories double.
    assert lower_edges[1] == pytest.approx(lower_edges[0] * 2, rel=1e-9)

    # Two conditions (nc in column c1, pc in column c2).
    assert set(data["condition"]) == {"Hela_nc", "Hela_pc"}

    # The chi-squared result matches scipy run on the same contingency table.
    counts = data.groupby(["condition", "pathogen_volume_bin"],
                          observed=True).size().unstack(fill_value=0)
    chi2, p, dof, _ = chi2_contingency(counts)
    res = out["chi_squared"]
    assert list(res.columns) == ["chi_squared_stat", "p_value",
                                 "degrees_of_freedom"]
    assert res["chi_squared_stat"].iloc[0] == pytest.approx(chi2)
    assert res["p_value"].iloc[0] == pytest.approx(p)
    assert int(res["degrees_of_freedom"].iloc[0]) == dof == 3


def test_endodyogeny_pixel_units_legend_title(endo_src):
    """With ``um_per_px=None`` the legend is labelled in px^3 and carries one
    entry per (1-indexed) volume bin."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings(endo_src))
    legend = plt.gca().get_legend()
    assert legend.get_title().get_text() == "Volume Range (px³)"
    labels = [t.get_text() for t in legend.get_texts()]
    n_bins = len(out["data"]["pathogen_volume_bin"].cat.categories)
    assert len(labels) == n_bins
    assert labels[0].startswith("1: ")
    assert labels[-1].startswith(f"{n_bins}: ")
    # y-axis is clamped to a proportion.
    assert plt.gca().get_ylim() == (0.0, 1.0)


# ---------------------------------------------------------------------------
# um_per_px scaling branch
# ---------------------------------------------------------------------------

def test_endodyogeny_um_per_px_rescales_areas_and_legend(endo_src):
    """``um_per_px`` converts px^2 areas to um^2 (and the min_area_bin with
    them), so binning is unchanged but the units in the legend switch."""
    from spacr.submodules import analyze_endodyogeny

    um = 0.1
    out = analyze_endodyogeny(_settings(endo_src, um_per_px=um))
    data = out["data"]

    # Areas were multiplied by um_per_px ** 2 ...
    assert np.allclose(sorted(data["pathogen_area"]),
                       sorted(np.array(AREAS_C1 + AREAS_C2) * um ** 2))
    # ... and the volume follows from the scaled area.
    assert np.allclose(data["pathogen_volume"], data["pathogen_area"] ** 1.5)
    # min_area_bin was scaled by the same factor, so the *bins* are identical
    # to the pixel-unit run.
    px_bins = [_expected_bin_index(a ** 1.5) for a in AREAS_C1 + AREAS_C2]
    assert sorted(data["bin_index"]) == sorted(px_bins)

    assert plt.gca().get_legend().get_title().get_text() == \
        "Volume Range (µm³)"


# ---------------------------------------------------------------------------
# Area filters
# ---------------------------------------------------------------------------

def test_endodyogeny_min_area_bin_and_max_area_filter_objects(endo_src):
    """Objects smaller than ``min_area_bin`` or larger than ``max_area`` are
    dropped before binning."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(
        _settings(endo_src, min_area_bin=1000, max_area=1700))
    areas = out["data"]["pathogen_area"]
    assert areas.min() >= 1000
    assert areas.max() <= 1700
    kept = [a for a in AREAS_C1 + AREAS_C2 if 1000 <= a <= 1700]
    assert len(areas) == len(kept)
    assert sorted(areas) == sorted(kept)
    # min_volume_bin moved with min_area_bin: two doublings survive.
    assert sorted(set(out["data"]["bin_index"])) == [1, 2]
    assert int(out["chi_squared"]["degrees_of_freedom"].iloc[0]) == 1


def test_endodyogeny_raises_when_no_object_exceeds_the_first_bin(tmp_path):
    """If the largest volume is not greater than the first bin edge there is
    nothing to bin, and the helper says so instead of emitting empty bins."""
    from spacr.submodules import analyze_endodyogeny

    root = tmp_path / "flat"
    # Every object sits exactly on min_area_bin -> max_volume == min_volume_bin.
    _write_endodyogeny_db(root, {"c1": [500.0] * 4, "c2": [500.0] * 4})
    with pytest.raises(ValueError, match="is not greater than"):
        analyze_endodyogeny(_settings(str(root), min_area_bin=500))


# ---------------------------------------------------------------------------
# Bin-edge extension: max volume lands exactly on a doubling edge
# ---------------------------------------------------------------------------

def test_endodyogeny_extends_last_edge_so_max_volume_is_not_clipped(tmp_path):
    """``2000 ** 1.5`` is exactly ``8 * 500 ** 1.5``, i.e. the largest volume
    falls exactly on the last computed bin edge. The extra edge appended by
    the binner keeps that object inside the top bin instead of dropping it
    as out-of-range."""
    from spacr.submodules import analyze_endodyogeny

    assert 2000.0 ** 1.5 == MIN_VOLUME_BIN * 8  # the premise of this test

    root = tmp_path / "poweroftwo"
    _write_endodyogeny_db(
        root,
        {"c1": [600.0, 1000.0, 1600.0, 2000.0],
         "c2": [620.0, 1010.0, 1610.0, 2000.0]},
    )
    out = analyze_endodyogeny(_settings(str(root), min_area_bin=500))
    data = out["data"]

    # Nothing dropped, no NaN bins.
    assert len(data) == 8
    assert not data["pathogen_volume_bin"].isna().any()

    top = data[data["pathogen_area"] == 2000.0]
    assert len(top) == 2
    expected_label = f"{MIN_VOLUME_BIN * 8:.2f}-{MIN_VOLUME_BIN * 16:.2f}"
    assert set(top["pathogen_volume_bin"].astype(str)) == {expected_label}
    assert set(top["bin_index"]) == {4}
    # The appended edge shows up as the last category.
    assert data["pathogen_volume_bin"].cat.categories[-1] == expected_label


# ---------------------------------------------------------------------------
# max_bins capping
# ---------------------------------------------------------------------------

def test_endodyogeny_max_bins_collapses_the_tail_into_one_bin(endo_src):
    """``max_bins`` caps the number of bins; everything above the cap is
    merged into a single open-ended '>edge' bin."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings(endo_src, max_bins=2))
    data = out["data"]

    assert len(data) == len(AREAS_C1) + len(AREAS_C2)
    assert data["bin_index"].max() == 2
    cats = list(data["pathogen_volume_bin"].cat.categories)
    assert len(cats) == 2
    assert cats[-1].startswith(">")
    assert cats[-1] == f">{MIN_VOLUME_BIN * 2:.2f}"
    # Every object whose true bin was >= 2 now sits in the capped bin.
    n_capped = sum(1 for a in AREAS_C1 + AREAS_C2
                   if _expected_bin_index(a ** 1.5) >= 2)
    assert int((data["bin_index"] == 2).sum()) == n_capped
    assert int(out["chi_squared"]["degrees_of_freedom"].iloc[0]) == 1


def test_endodyogeny_max_bins_larger_than_bin_count_is_a_no_op(endo_src):
    """A ``max_bins`` above the number of natural bins leaves the labels
    untouched (no '>' bin is created)."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings(endo_src, max_bins=99))
    cats = list(out["data"]["pathogen_volume_bin"].cat.categories)
    assert len(cats) == 4
    assert not any(c.startswith(">") for c in cats)


# ---------------------------------------------------------------------------
# group_by_class
# ---------------------------------------------------------------------------

def test_endodyogeny_group_by_class_splits_condition_by_class_column(endo_src):
    """``group_by_class`` re-groups on condition+class instead of condition."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings(
        endo_src, group_by_class=True, class_column="predictions"))
    data = out["data"]

    assert "new_condition" in data.columns
    expected = data["condition"].astype(str) + data["predictions"].astype(str)
    assert list(data["new_condition"]) == list(expected)
    # Two conditions x two predicted classes.
    assert len(set(data["new_condition"])) == 4
    # dof grows with the extra groups: (4 - 1) * (n_bins - 1).
    n_bins = len(data["pathogen_volume_bin"].cat.categories)
    assert int(out["chi_squared"]["degrees_of_freedom"].iloc[0]) == 3 * (n_bins - 1)


# ---------------------------------------------------------------------------
# group_column NaN handling
# ---------------------------------------------------------------------------

def test_endodyogeny_drops_objects_with_no_condition(tmp_path):
    """Wells that match no metadata entry get a NaN condition and are dropped
    before binning."""
    from spacr.submodules import analyze_endodyogeny

    root = tmp_path / "unmapped"
    _write_endodyogeny_db(
        root,
        {"c1": AREAS_C1, "c2": AREAS_C2, "c3": AREAS_C1},
    )
    # cell_types=None so no host-cell string is glued onto the condition:
    # column c3 maps to no pathogen, so its condition is NA.
    out = analyze_endodyogeny(_settings(str(root), cell_types=None))
    data = out["data"]
    assert set(data["condition"]) == {"nc", "pc"}
    assert "c3" not in set(data["columnID"])
    assert len(data) == len(AREAS_C1) + len(AREAS_C2)


# ---------------------------------------------------------------------------
# level switch
# ---------------------------------------------------------------------------

def test_endodyogeny_plate_level_uses_the_same_raw_counts(endo_src):
    """``level='plate'`` switches the per-well column to 'plate' but the
    chi-squared test is still run on the raw object counts."""
    from spacr.submodules import analyze_endodyogeny

    object_level = analyze_endodyogeny(_settings(endo_src, level="object"))
    plt.close("all")
    plate_level = analyze_endodyogeny(_settings(endo_src, level="plate"))

    pd.testing.assert_frame_equal(object_level["chi_squared"],
                                  plate_level["chi_squared"])
    assert len(plate_level["data"]) == len(object_level["data"])


# ---------------------------------------------------------------------------
# save branch
# ---------------------------------------------------------------------------

def test_endodyogeny_save_writes_figure_and_three_csvs(endo_src):
    """``save=True`` writes the figure plus the data / chi-squared /
    pairwise tables under ``<src>/results/analyze_endodyogeny``."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings(endo_src, save=True))
    out_dir = os.path.join(endo_src, "results", "analyze_endodyogeny")

    pdf = os.path.join(out_dir, "chi_squared_results.pdf")
    assert os.path.getsize(pdf) > 0
    with open(pdf, "rb") as fh:
        assert fh.read(4) == b"%PDF"

    saved_data = pd.read_csv(os.path.join(out_dir, "data.csv"))
    assert len(saved_data) == len(out["data"])
    assert np.allclose(sorted(saved_data["pathogen_volume"]),
                       sorted(out["data"]["pathogen_volume"]))

    saved_chi = pd.read_csv(os.path.join(out_dir, "chi_squared_results.csv"))
    assert saved_chi["chi_squared_stat"].iloc[0] == pytest.approx(
        out["chi_squared"]["chi_squared_stat"].iloc[0])

    pairwise = pd.read_csv(
        os.path.join(out_dir, "chi_squared_pairwise_results.csv"))
    # One row per pair of conditions -> exactly one pair here.
    assert len(pairwise) == 1


def test_endodyogeny_accepts_a_list_of_sources(endo_src):
    """``src`` may be a list of plate directories; the results land under the
    first one."""
    from spacr.submodules import analyze_endodyogeny

    out = analyze_endodyogeny(_settings([endo_src], save=True))
    assert len(out["data"]) == len(AREAS_C1) + len(AREAS_C2)
    assert os.path.isfile(os.path.join(
        endo_src, "results", "analyze_endodyogeny", "data.csv"))


# ---------------------------------------------------------------------------
# Missing group column — the informative KeyError is dead code today
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="BUG: the 'group_column not in df.columns' check sits AFTER "
           "df.dropna(subset=[group_column]), so pandas raises a bare "
           "KeyError first and the informative 'Available columns' message "
           "is unreachable",
)
def test_endodyogeny_missing_group_column_reports_available_columns(endo_src):
    from spacr.submodules import analyze_endodyogeny

    with pytest.raises(KeyError) as excinfo:
        analyze_endodyogeny(_settings(endo_src, group_column="does_not_exist"))
    assert "Available columns" in str(excinfo.value)
