"""CPU-only coverage for ``spacr.submodules.analyze_class_proportion``.

``analyze_class_proportion`` is the screen-level entry point that answers
"does the classifier call a different mix of classes in each experimental
group?".  It merges the per-object measurement tables with ``png_list``,
annotates each row with a condition, runs a chi-squared test on the class
column, draws the stacked proportion bars plus a per-plate heatmap, and
finishes with normality / Levene / group-comparison / post-hoc tests.

What is exercised here:

* the full pipeline against a **real** synthetic ``measurements.db`` so
  ``io._read_and_merge_data`` executes for real, with ``save=True`` so
  every CSV/PDF write branch runs and the emitted files are inspected,
* the ``save=False`` branch (nothing must be written to disk),
* the ``level='object'`` bar-plot branch,
* the NaN -> 0 fill of the class column,
* the "group column missing" diagnostic branch,
* the src-listification and the ``png_list`` table auto-append, both
  observed through a recording fake of ``_read_and_merge_data``.

No network, no CUDA, no TensorFlow.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Imported up front so the slow torch/skimage import chain behind spacr is
# paid at collection time rather than charged to the first test.
import spacr.io  # noqa: E402,F401
import spacr.submodules  # noqa: E402,F401


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------

_ROWS = ("r1", "r2", "r3")
_COLUMNS = ("c1", "c2", "c3")
_FIELDS = ("f1", "f2")
_OBJECTS = (1, 2, 3, 4)

_N_OBJECTS = len(_ROWS) * len(_COLUMNS) * len(_FIELDS) * len(_OBJECTS)

# Class-label mix per plate column. Skewed per column so the contingency
# table carries real structure (and so every class appears in every group,
# which chi2_contingency needs for non-zero expected frequencies).
_CLASS_WEIGHTS = {
    "c1": (0.70, 0.20, 0.10),
    "c2": (0.20, 0.60, 0.20),
    "c3": (0.10, 0.25, 0.65),
}

# columnID -> pathogen, via the pathogen_plate_metadata used in _settings().
_EXPECTED_CONDITIONS = {"HeLa_nc", "HeLa_pc", "HeLa_tc"}


def _synthetic_tables(seed=0):
    """Return ``(cell_df, png_list_df)`` shaped exactly like spacr writes them.

    ``cell`` carries an integer ``object_label`` plus the prcf/prc keys the
    merge layer needs; ``png_list`` carries the parent label in its ``'o<N>'``
    string form under ``cell_id`` (that is what ``_read_and_merge_data``
    joins on) and the classifier output in the ``test`` column.
    """
    rng = np.random.default_rng(seed)
    cell_rows, png_rows = [], []
    for r in _ROWS:
        for c in _COLUMNS:
            for f in _FIELDS:
                for obj in _OBJECTS:
                    prcf = f"plate1_{r}_{c}_{f}"
                    cell_rows.append({
                        "plateID": "plate1", "rowID": r, "columnID": c,
                        "fieldID": f, "object_label": int(obj),
                        "prcf": prcf, "prc": f"plate1_{r}_{c}",
                        "cell_area": float(rng.integers(300, 4000)),
                        "cell_channel_0_mean_intensity": float(rng.normal(1200, 150)),
                    })
                    png_rows.append({
                        "plateID": "plate1", "rowID": r, "columnID": c,
                        "fieldID": f, "cell_id": f"o{obj}",
                        "png_path": f"/pngs/{prcf}_o{obj}.png",
                        "test": int(rng.choice([0, 1, 2], p=_CLASS_WEIGHTS[c])),
                    })
    return pd.DataFrame(cell_rows), pd.DataFrame(png_rows)


def _build_measurements_db(src, seed=0):
    """Write ``<src>/measurements/measurements.db`` and return its path."""
    meas = os.path.join(str(src), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")
    cell, png = _synthetic_tables(seed)
    con = sqlite3.connect(db)
    try:
        cell.to_sql("cell", con, index=False)
        png.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def _fake_merged_frame(seed=0):
    """A DataFrame shaped like the output of ``io._read_and_merge_data``.

    Same content the real merge produces for the synthetic DB above:
    prcfo index, 'o<N>' object labels, the class column as floats.
    """
    cell, png = _synthetic_tables(seed)
    merged = cell.copy()
    merged["object_label"] = ["o%d" % o for o in cell["object_label"]]
    merged["test"] = png["test"].astype(float).to_numpy()
    merged["png_path"] = png["png_path"].to_numpy()
    merged.index = pd.Index(merged["prcf"] + "_" + merged["object_label"],
                            name="prcfo")
    return merged


def _install_fake_merge(monkeypatch, df, recorder=None):
    """Patch ``spacr.io._read_and_merge_data`` with a recording fake."""
    def _fake(locs, tables, verbose=False, nuclei_limit=10, pathogen_limit=10,
              **kwargs):
        if recorder is not None:
            recorder.update({"locs": list(locs), "tables": list(tables),
                             "verbose": verbose, "nuclei_limit": nuclei_limit,
                             "pathogen_limit": pathogen_limit})
        return df.copy(), []

    monkeypatch.setattr(spacr.io, "_read_and_merge_data", _fake)


def _settings(src, **over):
    """Settings for a 3-condition (nc / pc / tc) class-proportion run."""
    s = {
        "src": [str(p) for p in src] if isinstance(src, list) else str(src),
        "tables": ["cell"],
        "cell_types": ["HeLa"], "cell_plate_metadata": None,
        "pathogen_types": ["nc", "pc", "tc"],
        "pathogen_plate_metadata": [["c1"], ["c2"], ["c3"]],
        "treatments": None, "treatment_plate_metadata": None,
        "group_column": "condition", "class_column": "test",
        "nuclei_limit": 1000, "pathogen_limit": 1000,
        "level": "well", "save": False, "verbose": False,
    }
    s.update(over)
    return s


# ===========================================================================
# end-to-end against a real synthetic measurements.db, save=True
# ===========================================================================

def test_real_db_run_returns_annotated_data_and_chi_squared(tmp_path):
    """The full pipeline over a real DB annotates conditions and returns a
    one-row chi-squared table describing the class x condition contingency."""
    from spacr.submodules import analyze_class_proportion

    _build_measurements_db(tmp_path)
    out = analyze_class_proportion(_settings(tmp_path))

    assert set(out) == {"data", "chi_squared"}

    df = out["data"]
    # One row per measured object, all of them annotated.
    assert len(df) == _N_OBJECTS
    assert set(df["condition"].unique()) == _EXPECTED_CONDITIONS
    assert set(df["pathogen"].unique()) == {"nc", "pc", "tc"}
    # The class column survived the merge as a numeric column with no gaps.
    assert not df["test"].isna().any()
    assert set(np.unique(df["test"])) <= {0.0, 1.0, 2.0}

    chi = out["chi_squared"]
    # The historical three columns are still there and still row 0.
    # Instruction 80 added the columns naming the test and its unit, plus
    # rows for the level-appropriate test and the mixed model: a chi-squared
    # over objects and a t-test over wells answer different questions and
    # were previously reported as one number with one n.
    for column in ("chi_squared_stat", "p_value", "degrees_of_freedom",
                   "test", "unit", "n"):
        assert column in chi.columns, column
    assert chi.loc[0, "unit"] == "object"
    # One row per test now, not one row total: the object chi-squared, then
    # the per-unit proportion test and the clustered model, one of each per
    # bin. The object row must still be first, because every published
    # figure came from it.
    assert len(chi) >= 1
    assert chi.loc[0, "test"] == "chi-squared on object counts"
    # 3 conditions x 3 classes -> (3-1)*(3-1) = 4 degrees of freedom, and the
    # deliberately skewed class mix must come out significant.
    assert int(chi["degrees_of_freedom"].iloc[0]) == 4
    assert chi["chi_squared_stat"].iloc[0] > 0
    assert 0.0 <= chi["p_value"].iloc[0] < 0.05


def test_real_db_run_with_save_writes_every_artifact(tmp_path):
    """save=True writes the chi-squared / pairwise / data CSVs, both PDFs and
    the four follow-up statistics tables under results/analyze_class_proportion."""
    from spacr.submodules import analyze_class_proportion

    _build_measurements_db(tmp_path)
    out = analyze_class_proportion(_settings(tmp_path, save=True))

    outdir = tmp_path / "results" / "analyze_class_proportion"
    expected = [
        "class_chi_squared_results.csv",
        "class_frequency_test.csv",
        "class_chi_squared_data.csv",
        "class_chi_squared.pdf",
        "class_heatmap.pdf",
        "normality_results.csv",
        "variance_results.csv",
        "statistical_test_results.csv",
        "posthoc_results.csv",
    ]
    for name in expected:
        assert (outdir / name).is_file(), f"missing {name}"

    # Both figures are real PDFs, not empty placeholder files.
    for pdf in ("class_chi_squared.pdf", "class_heatmap.pdf"):
        blob = (outdir / pdf).read_bytes()
        assert blob.startswith(b"%PDF"), pdf
        assert len(blob) > 1000, pdf

    # The saved data CSV is the returned frame.
    saved_data = pd.read_csv(outdir / "class_chi_squared_data.csv")
    assert len(saved_data) == len(out["data"]) == _N_OBJECTS
    assert set(saved_data["condition"].unique()) == _EXPECTED_CONDITIONS

    # Chi-squared CSV round-trips the returned table.
    saved_chi = pd.read_csv(outdir / "class_chi_squared_results.csv")
    assert saved_chi["chi_squared_stat"].iloc[0] == pytest.approx(
        out["chi_squared"]["chi_squared_stat"].iloc[0]
    )

    # 3 groups -> 3 pairwise comparisons, each with a raw and adjusted p.
    pairwise = pd.read_csv(outdir / "class_frequency_test.csv")
    assert len(pairwise) == 3
    assert {"Group 1", "Group 2", "p-value", "p-value_adj"} <= set(pairwise.columns)
    assert ((pairwise["p-value"] >= 0) & (pairwise["p-value"] <= 1)).all()

    # One normality result per condition for the single class column.
    normality = pd.read_csv(outdir / "normality_results.csv")
    assert len(normality) == len(_EXPECTED_CONDITIONS)
    assert set(normality["Column"]) == {"test"}

    # Levene's test: exactly one row, named.
    variance = pd.read_csv(outdir / "variance_results.csv")
    assert len(variance) == 1
    assert variance["Test Name"].iloc[0] == "Levene's Test"
    assert 0.0 <= variance["p-value"].iloc[0] <= 1.0

    # One group-comparison row for the one class column, across 3 groups.
    stats_df = pd.read_csv(outdir / "statistical_test_results.csv")
    assert len(stats_df) == 1
    assert stats_df["Groups"].iloc[0] == 3
    assert stats_df["Test Name"].iloc[0] in ("One-way ANOVA", "Kruskal-Wallis test")

    # Post-hoc: the 3 unique pairs formed from the 3 conditions.
    posthoc = pd.read_csv(outdir / "posthoc_results.csv")
    assert set(posthoc["Comparison"]) == {
        "HeLa_nc vs HeLa_pc", "HeLa_nc vs HeLa_tc", "HeLa_pc vs HeLa_tc",
    }
    assert ((posthoc["Adjusted p-value"] >= 0)
            & (posthoc["Adjusted p-value"] <= 1)).all()
    # Integer class labels are not normal, so the rank-based branch runs.
    assert set(posthoc["Test Name"]) == {"Dunn's Post-hoc"}
    assert set(posthoc["Adjusted Method"]) == {"holm"}


# ===========================================================================
# save=False — nothing may be written
# ===========================================================================

def test_save_false_writes_no_results_directory(tmp_path, monkeypatch):
    """With save=False the results folder is never created and no CSV/PDF
    is emitted, but the analysis still returns its two payloads."""
    from spacr.submodules import analyze_class_proportion

    _install_fake_merge(monkeypatch, _fake_merged_frame())
    out = analyze_class_proportion(_settings(tmp_path, save=False))

    assert not (tmp_path / "results").exists()
    # save_settings still snapshots the resolved settings; that is the only
    # thing on disk besides nothing.
    assert set(out) == {"data", "chi_squared"}
    assert len(out["data"]) == _N_OBJECTS
    assert out["chi_squared"]["p_value"].iloc[0] < 0.05


# ===========================================================================
# level='object' vs level='well'
# ===========================================================================

def test_object_level_matches_well_level_chi_squared(tmp_path, monkeypatch):
    """`level` only changes how the bars are aggregated for the plot — the
    chi-squared statistic is computed on the raw object counts either way."""
    from spacr.submodules import analyze_class_proportion

    df = _fake_merged_frame()
    _install_fake_merge(monkeypatch, df)
    well = analyze_class_proportion(_settings(tmp_path, level="well"))
    plt.close("all")
    _install_fake_merge(monkeypatch, df)
    obj = analyze_class_proportion(_settings(tmp_path, level="object"))

    assert obj["chi_squared"]["chi_squared_stat"].iloc[0] == pytest.approx(
        well["chi_squared"]["chi_squared_stat"].iloc[0]
    )
    assert obj["chi_squared"]["degrees_of_freedom"].iloc[0] == 4

    # Two figures come out of the object-level run: the stacked bars first,
    # then the plate heatmap. The bar axes carry one stacked bar per
    # condition on a 0-1 proportion axis.
    fignums = plt.get_fignums()
    assert len(fignums) == 2
    bars_ax = plt.figure(fignums[0]).axes[0]
    assert bars_ax.get_ylim() == (0.0, 1.0)
    assert [t.get_text() for t in bars_ax.get_xticklabels()] == sorted(
        _EXPECTED_CONDITIONS
    )
    # 3 classes stacked -> 3 bar containers of 3 bars each.
    assert len(bars_ax.containers) == 3
    assert all(len(c) == len(_EXPECTED_CONDITIONS) for c in bars_ax.containers)


# ===========================================================================
# the class column NaN -> 0 fill
# ===========================================================================

def test_nan_class_values_are_filled_with_zero(tmp_path, monkeypatch):
    """Objects with no classifier call (NaN) are counted as class 0 rather
    than being dropped from the contingency table."""
    from spacr.submodules import analyze_class_proportion

    df = _fake_merged_frame()
    # Blank out every class-2 call; they must reappear as class 0.
    n_missing = int((df["test"] == 2.0).sum())
    n_zero_before = int((df["test"] == 0.0).sum())
    assert n_missing > 0 and n_zero_before > 0
    df.loc[df["test"] == 2.0, "test"] = np.nan
    assert df["test"].isna().sum() == n_missing

    _install_fake_merge(monkeypatch, df)
    out = analyze_class_proportion(_settings(tmp_path))

    data = out["data"]
    assert not data["test"].isna().any()
    assert int((data["test"] == 0.0).sum()) == n_zero_before + n_missing
    assert len(data) == _N_OBJECTS
    # Only two classes remain -> (3-1)*(2-1) = 2 degrees of freedom.
    assert int(out["chi_squared"]["degrees_of_freedom"].iloc[0]) == 2


# ===========================================================================
# missing group column diagnostic
# ===========================================================================

def test_missing_group_column_lists_available_columns(tmp_path, monkeypatch,
                                                      capsys):
    """When the requested group column is absent, the function prints the
    column it wanted plus every column that *is* available before the
    downstream group-by fails."""
    from spacr.submodules import analyze_class_proportion

    df = _fake_merged_frame()
    _install_fake_merge(monkeypatch, df)
    # treatments=None means annotate_conditions never creates 'treatment'.
    settings = _settings(tmp_path, group_column="treatment")

    with pytest.raises(KeyError):
        analyze_class_proportion(settings)

    printed = capsys.readouterr().out
    assert "treatment not found in DataFrame, please choose from:" in printed
    # The diagnostic enumerates the real columns so the user can pick one.
    for col in ("condition", "columnID", "prc", "test"):
        assert f"\n{col}\n" in printed


# ===========================================================================
# input normalisation: src listification + png_list auto-append
# ===========================================================================

def test_string_src_is_listified_into_one_db_location(tmp_path, monkeypatch):
    """A bare string src becomes a one-element list and is turned into the
    canonical measurements.db path handed to the merge layer."""
    from spacr.submodules import analyze_class_proportion

    rec = {}
    _install_fake_merge(monkeypatch, _fake_merged_frame(), rec)
    settings = _settings(tmp_path, verbose=True, nuclei_limit=7,
                         pathogen_limit=3)
    analyze_class_proportion(settings)

    assert settings["src"] == [str(tmp_path)]
    assert rec["locs"] == [os.path.join(str(tmp_path),
                                        "measurements/measurements.db")]
    # verbose / limit settings are forwarded verbatim.
    assert rec["verbose"] is True
    assert rec["nuclei_limit"] == 7
    assert rec["pathogen_limit"] == 3


def test_png_list_table_is_appended_exactly_once(tmp_path, monkeypatch):
    """png_list is appended when missing and left alone when already
    requested — it must never appear twice (that would duplicate columns)."""
    from spacr.submodules import analyze_class_proportion

    rec = {}
    _install_fake_merge(monkeypatch, _fake_merged_frame(), rec)
    analyze_class_proportion(_settings(tmp_path, tables=["cell"]))
    assert rec["tables"] == ["cell", "png_list"]

    plt.close("all")
    rec2 = {}
    _install_fake_merge(monkeypatch, _fake_merged_frame(), rec2)
    analyze_class_proportion(_settings(tmp_path, tables=["png_list", "cell"]))
    assert rec2["tables"] == ["png_list", "cell"]
    assert rec2["tables"].count("png_list") == 1


def test_multi_src_list_produces_one_location_per_source(tmp_path, monkeypatch):
    """A list src maps to one measurements.db per entry, in order, and the
    first entry is the one the results folder is written under."""
    from spacr.submodules import analyze_class_proportion

    src_a = tmp_path / "plateA"
    src_b = tmp_path / "plateB"
    src_a.mkdir()
    src_b.mkdir()

    rec = {}
    _install_fake_merge(monkeypatch, _fake_merged_frame(), rec)
    analyze_class_proportion(_settings([src_a, src_b], save=True))

    assert rec["locs"] == [
        os.path.join(str(src_a), "measurements/measurements.db"),
        os.path.join(str(src_b), "measurements/measurements.db"),
    ]
    assert (src_a / "results" / "analyze_class_proportion"
            / "class_chi_squared_results.csv").is_file()
    assert not (src_b / "results").exists()


# ===========================================================================
# the fill is a CHOICE, so it has to be visible
# ===========================================================================

def test_the_nan_fill_reports_how_many_it_filled(tmp_path, monkeypatch,
                                                 capsys):
    """The fill above is right for a classifier column and wrong for an
    annotation column, and the code cannot tell which it was given.

    Filling a classifier's uncalled objects with the negative class is
    correct. Filling an ANNOTATION column's NaNs is not: annotate 500 of
    40,000 cells and the other 39,500 become a class-0 majority that decides
    the chi-squared on its own, with nothing on screen to say so.

    So the count is printed. This does not change any number -- it makes the
    one case where the number is wrong announce itself.
    """
    from spacr.submodules import analyze_class_proportion

    df = _fake_merged_frame()
    n_missing = int((df["test"] == 2.0).sum())
    assert n_missing > 0
    df.loc[df["test"] == 2.0, "test"] = np.nan

    _install_fake_merge(monkeypatch, df)
    analyze_class_proportion(_settings(tmp_path))

    printed = capsys.readouterr().out
    assert f"{n_missing} of {len(df)} objects have no value" in printed
    assert "counted as class 0" in printed
    assert "annotation rather than a classifier call" in printed, (
        "the message does not tell the user which situation they are in")


def test_nothing_is_printed_when_every_object_has_a_class(tmp_path,
                                                          monkeypatch, capsys):
    """A warning that fires on a clean run is a warning nobody reads."""
    from spacr.submodules import analyze_class_proportion

    df = _fake_merged_frame()
    assert not df["test"].isna().any()

    _install_fake_merge(monkeypatch, df)
    analyze_class_proportion(_settings(tmp_path))

    assert "have no value in" not in capsys.readouterr().out
