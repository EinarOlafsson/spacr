"""Branch coverage for :func:`spacr.ml.perform_regression`.

``perform_regression`` is the ~750-line pooled-CRISPR screen entry point: it
reads per-well score CSVs and per-well sgRNA count CSVs, aligns them on
plate/row/column, fits a regression of phenotype on gRNA/gene abundance and
emits results + hit tables + (optionally) toxo reports.

These tests drive it end to end on tiny synthetic CSVs.  Only the purely
visual helpers are stubbed (``plot_plates``, ``plot_histogram``,
``plot_data_from_csv``, the toxo plots) plus ``minimum_cell_simulation``
(a Monte-Carlo resampler that contributes nothing to this function's own
branches); everything else -- metadata correction, plate recovery, control
filtering, ``process_scores`` / ``process_reads``, the statsmodels /
sklearn fits, the coefficient bookkeeping and every CSV that gets written --
runs for real.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic screen builders
# ---------------------------------------------------------------------------

GENES = ("000000", "233460", "239740", "111111")
N_GRNA_PER_GENE = 3
ROWS = ("r1", "r2", "r3")
COLS = ("c1", "c2", "c3", "c4", "c5", "c6")   # c1..c3 are dropped as controls
KEPT_COLS = ("c4", "c5", "c6")
CONTROLS = [f"000000_{i}" for i in (1, 2, 3)]


def _grnas():
    """Full gRNA name list in the org_gene_grna form spacr splits on."""
    return [f"TGGT1_{g}_{i}" for g in GENES for i in range(1, N_GRNA_PER_GENE + 1)]


def results_dir(count_csv, regression_type="ols"):
    """Where a run writes: ``<count data folder>/results/<regression type>``.

    Asked for on 2026-08-16 -- "just store everything in the same location as
    the first count data ... then the type so for me .../claude/results/ols".

    Every expectation in this file used to spell the path that replaced:
    ``<src>/results/<score file stem>/<type>/list``. When the layout changed
    the tests kept reading a folder nothing writes to any more, so
    twenty-three of them failed on a missing CSV while the run that wrote it
    was fine -- a suite reporting the wrong defect, which is worse than a
    silent one because it sends the reader to the wrong file.

    One helper rather than a literal per test, so the next layout change
    costs one line here instead of another twenty-three red marks.
    """
    return os.path.join(os.path.dirname(count_csv), "results", regression_type)


def _score_records(plate, seed, n_cells=6, with_path=False, plate_token=None,
                   n_bad_paths=0, rows=ROWS, cols=COLS):
    rng = np.random.default_rng(seed)
    recs = []
    made = 0
    for r in rows:
        for c in cols:
            base = float(rng.uniform(0.2, 0.8))
            for k in range(n_cells):
                rec = {
                    "plateID": plate,
                    "rowID": r,
                    "columnID": c,
                    "fieldID": "f1",
                    "pred": float(np.clip(base + rng.normal(0, 0.1), 0.02, 0.98)),
                    "recruitment": float(rng.normal(50.0, 12.0)),
                }
                if with_path:
                    letter = chr(ord("A") + int(r[1:]) - 1)
                    token = plate_token or plate.upper()
                    # spacr anchors its well regexes at the start of the
                    # string, so these are bare basenames.
                    if made < n_bad_paths:
                        rec["path"] = f"no_plate_prefix_{made}.png"
                    else:
                        rec["path"] = (f"{token}_{letter}{int(c[1:]):02d}"
                                       f"_1_1_{k}.png")
                recs.append(rec)
                made += 1
    return recs


def write_scores(path, plate="plate1", seed=0, n_cells=6, with_path=False,
                 plate_token=None, n_bad_paths=0, drop=(), row_prefix=None,
                 rows=ROWS, cols=COLS):
    """Write a per-object score CSV and return its path."""
    df = pd.DataFrame(_score_records(plate, seed, n_cells, with_path,
                                     plate_token, n_bad_paths, rows, cols))
    if row_prefix:
        df["rowID"] = row_prefix + "_" + df["rowID"]
    for col in drop:
        df = df.drop(columns=[col])
    df.to_csv(path, index=False)
    return str(path)


def write_counts(path, plate="plate1", seed=1, grnas=None, sparse_grna=None,
                 sparse_well=("r1", "c4"), row_prefix=None, rows=ROWS,
                 cols=COLS, drop=()):
    """Write a per-well sgRNA count CSV and return its path.

    ``sparse_grna`` is present in exactly one (kept) well, which makes it an
    outlier for ``get_outlier_reference_values``.
    """
    rng = np.random.default_rng(seed)
    grnas = list(grnas if grnas is not None else _grnas())
    recs = []
    for r in rows:
        for c in cols:
            for g in grnas:
                if sparse_grna is not None and g == sparse_grna:
                    if (r, c) != tuple(sparse_well):
                        continue
                recs.append({"plateID": plate, "rowID": r, "columnID": c,
                             "grna": g, "count": int(rng.integers(20, 400))})
    df = pd.DataFrame(recs)
    if row_prefix:
        df["rowID"] = row_prefix + "_" + df["rowID"]
    for col in drop:
        df = df.drop(columns=[col])
    df.to_csv(path, index=False)
    return str(path)


def write_metadata(path):
    """Write a gene-metadata CSV shaped like the toxo TGGT1/TGME49 summaries."""
    rows = []
    for g in GENES:
        rows.append({
            "Gene ID": f"TGGT1_{g}",
            "Gene Name": f"gene_{g}",
            "T.gondii GT1 CRISPR Phenotype - Mean Phenotype": 0.5,
            "T.gondii GT1 CRISPR Phenotype - Standard Error": 0.1,
            "sense - Tachyzoites": 1.0,
            "sense - Tissue cysts": 2.0,
            "sense - EES1": 3.0,
            "sense - EES2": 4.0,
            "sense - EES3": 5.0,
            "sense - EES4": 6.0,
            "sense - EES5": 7.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    plt.close("all")


@pytest.fixture
def screen(tmp_path):
    """One-plate screen: score CSV, count CSV, metadata CSV + expected paths."""
    sdir = tmp_path / "scores"
    cdir = tmp_path / "counts"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "xgb_scores.csv")
    count = write_counts(cdir / "counts.csv")
    meta = write_metadata(tmp_path / "TGME49_Summary.csv")
    return {
        "root": tmp_path,
        "score": score,
        "count": count,
        "meta": meta,
        "res": results_dir(count),
        "count_dir": str(cdir),
    }


def base_settings(screen, **over):
    """Settings dict exactly as a dispatcher builds it, plus this suite's choices.

    The dict is finished by the SAME defaults builder all three entry points
    use -- ``gui_core.setup_settings_panel`` (Tk),
    ``qt.screens.settings_model.resolve_default_settings`` (Qt) and
    ``cli.module_defaults`` (``spacr-run regression``) -- so anything the
    builder fails to supply is missing here too, and the tests below fail the
    way a user's run fails.

    This fixture used to hand-write ``score_column``, ``tolerance``,
    ``verbose``, ``invert_dependent_variable`` and ``y_lims`` on top of a
    literal dict. ``get_perform_regression_default_settings`` supplied none of
    them while ``perform_regression`` indexed all of them, so every test in
    this file passed against a dict no entry point could produce, and the real
    pipeline died on ``KeyError: 'verbose'`` at ml.py:1409 -- after both input
    CSVs had been read and ``settings/regression.csv`` had been written.

    Only keys that are a deliberate *test* choice belong in the literal below:
    the tiny synthetic wells (``min_cell_count``), a fixed threshold instead of
    the sweep (``fraction_threshold``), the toxo reports off by default. Adding
    a key here that the builder is supposed to supply hides the next such bug.
    """
    from spacr.settings import get_perform_regression_default_settings

    settings = {
        "score_data": [screen["score"]],
        "count_data": [screen["count"]],
        "dependent_variable": "pred",
        "regression_type": "ols",
        "min_cell_count": 3,
        "fraction_threshold": 0.005,
        "metadata_files": [screen["meta"], screen["meta"]],
        "toxo": False,
        "controls": list(CONTROLS),
        "outlier_detection": False,
        "alpha": 1.0,
    }
    settings.update(over)
    # Last, so a test that picks its own dependent_variable also gets the
    # score_column that follows it.
    return get_perform_regression_default_settings(settings)


@pytest.fixture
def heavy_stubs(monkeypatch):
    """Stub the slow visual helpers; record every call for assertions."""
    import spacr.plot as P
    import spacr.ml as ML
    # perform_regression imports these lazily on every call; warm them here so
    # the one-off import cost is not charged to whichever test runs first.
    import spacr.toxo  # noqa: F401
    import spacr.sequencing  # noqa: F401
    import spacr.settings  # noqa: F401

    rec = {"plates": [], "histograms": [], "sim": [], "house_plates": []}

    def fake_plot_plates(df, **kwargs):
        rec["plates"].append({"n_rows": len(df), "kwargs": dict(kwargs),
                              "columns": list(df.columns)})
        return None

    def fake_show_plates(df, variable, dst):
        # THE HOUSE-STYLE PLATE PANEL IS THE ONE A RUN DRAWS NOW.
        # `plot_plates` is still there as the fallback, so both are recorded
        # and the test can say which path was taken rather than only that
        # something was called.
        rec["house_plates"].append({"n_rows": len(df), "variable": variable,
                                    "dst": dst,
                                    "columns": list(df.columns)})
        return True

    def fake_plot_histogram(df, column, dst=None):
        rec["histograms"].append((column, dst))
        return None

    def fake_sim(settings, **kwargs):
        rec["sim"].append(dict(kwargs))
        return 3

    monkeypatch.setattr(P, "plot_plates", fake_plot_plates)
    monkeypatch.setattr(P, "plot_histogram", fake_plot_histogram)
    monkeypatch.setattr(ML, "_show_plates", fake_show_plates)
    monkeypatch.setattr(ML, "minimum_cell_simulation", fake_sim)
    return rec


@pytest.fixture
def stubs(monkeypatch, heavy_stubs):
    """heavy_stubs plus a recording stand-in for plot_data_from_csv."""
    import spacr.plot as P

    heavy_stubs["csv_plots"] = []

    def fake_plot_data_from_csv(settings):
        heavy_stubs["csv_plots"].append(dict(settings))
        return None, None

    monkeypatch.setattr(P, "plot_data_from_csv", fake_plot_data_from_csv)
    return heavy_stubs


@pytest.fixture
def toxo_stubs(monkeypatch, stubs):
    """Stub the toxo reporting plots and let the test choose the gene list."""
    import spacr.toxo as T

    stubs["volcano"] = []
    stubs["phenotypes"] = []
    stubs["heatmaps"] = []
    stubs["gene_list"] = ["TGGT1_239740"]

    def fake_volcano(data, metadata_path, **kwargs):
        stubs["volcano"].append({"n_rows": len(data),
                                 "metadata_path": metadata_path,
                                 "kwargs": dict(kwargs)})
        return stubs["gene_list"]

    def fake_phenotypes(data, gene_list, **kwargs):
        stubs["phenotypes"].append({"gene_list": gene_list,
                                    "kwargs": dict(kwargs)})
        return None

    def fake_heatmaps(data, gene_list, columns, **kwargs):
        stubs["heatmaps"].append({"gene_list": gene_list,
                                  "columns": list(columns)})
        return None

    monkeypatch.setattr(T, "custom_volcano_plot", fake_volcano)
    monkeypatch.setattr(T, "plot_gene_phenotypes", fake_phenotypes)
    monkeypatch.setattr(T, "plot_gene_heatmaps", fake_heatmaps)
    return stubs


# ---------------------------------------------------------------------------
# _perform_regression_read_data
# ---------------------------------------------------------------------------

def test_scalar_score_and_count_paths_are_wrapped_in_lists(screen, stubs):
    """A bare string for score_data/count_data is normalised to a 1-list."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, score_data=screen["score"],
                             count_data=screen["count"])
    out = perform_regression(settings)

    assert settings["score_data"] == [screen["score"]]
    assert settings["count_data"] == [screen["count"]]
    # The coefficient table and the hits, plus what is needed to judge the
    # fit: without the model there is no R-squared and no residual to test,
    # and without the design there is no way to count what reached it.
    assert {"results", "significant"} <= set(out)
    assert {"model", "model_data"} <= set(out), \
        "the fit and its design must come back, or no diagnostic can be computed"
    assert os.path.isfile(os.path.join(screen["res"], "results.csv"))


def test_legacy_score_list_longer_than_count_list_migrates(screen):
    """An unpaired tail remains legal because the final join is by well."""
    from spacr.ml import normalize_regression_input_pairs

    settings = base_settings(
        screen, score_data=[screen["score"], screen["score"]])
    pairs, migrated = normalize_regression_input_pairs(settings)
    assert migrated
    assert pairs[1] == {"score": screen["score"], "count": None,
                        "plate": None}


def test_legacy_count_list_longer_than_score_list_migrates(screen):
    """The opposite unpaired tail is retained too."""
    from spacr.ml import normalize_regression_input_pairs

    settings = base_settings(
        screen, count_data=[screen["count"], screen["count"]])
    pairs, migrated = normalize_regression_input_pairs(settings)
    assert migrated
    assert pairs[1] == {"score": None, "count": screen["count"],
                        "plate": None}


def test_paired_input_copies_plate_identity_from_its_partner(tmp_path, stubs):
    """A score file without plateID inherits the count file's declared plate."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv", plate="ignored",
                         drop=("plateID",))
    count = write_counts(cdir / "counts.csv", plate="plate2")
    meta = write_metadata(tmp_path / "md.csv")
    scr = {"score": score, "count": count, "meta": meta}

    settings = base_settings(
        scr, paired_data=[{"score": score, "count": count}])
    out = perform_regression(settings)

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    assert set(data["plateID"].unique()) == {"plate2"}
    assert data["prc"].str.startswith("plate2_").all()
    assert len(out["results"]) > 0


def test_missing_dependent_variable_raises(screen, stubs):
    """A dependent_variable absent from the score CSV is rejected."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, dependent_variable="not_a_column")
    # The message names the setting and the columns actually present, and --
    # when the score slot holds a count table -- says the inputs look swapped,
    # which is how this error is usually reached.
    with pytest.raises(ValueError, match="'not_a_column' is not a column"):
        perform_regression(settings)


def test_shortest_distance_dependent_variable_is_exempt_from_the_check(screen, stubs):
    """'pathogen_nucleus_shortest_distance' skips the column check ...

    ... and therefore only fails later, in process_scores, when the column is
    genuinely absent.
    """
    from spacr.ml import perform_regression

    settings = base_settings(
        screen, dependent_variable="pathogen_nucleus_shortest_distance")
    with pytest.raises(KeyError):
        perform_regression(settings)


def test_unsupported_regression_type_raises(screen, stubs):
    """regression_type is validated against the supported list."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="banana")
    with pytest.raises(ValueError, match="Unsupported regression type banana"):
        perform_regression(settings)


# ---------------------------------------------------------------------------
# plate / row / column recovery
# ---------------------------------------------------------------------------

def test_rowid_with_plate_prefix_is_split(tmp_path, stubs):
    """A count-CSV rowID of the 'plate1_r2' form is reduced to 'r2'."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv")
    count = write_counts(cdir / "counts.csv", row_prefix="plate1")
    meta = write_metadata(tmp_path / "md.csv")

    settings = base_settings({"score": score, "count": count, "meta": meta})
    out = perform_regression(settings)

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    assert set(data["rowID"].unique()) <= set(ROWS)
    assert len(out["results"]) > 0


def test_one_multiplate_score_file_can_be_paired_to_plate_count_files(tmp_path):
    """A consolidated score export may be reused without duplicating plates."""
    from spacr.ml import load_regression_input_pairs

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score_path = sdir / "scores.csv"
    pd.concat([
        pd.DataFrame(_score_records("plate1", 3)),
        pd.DataFrame(_score_records("plate2", 4)),
    ], ignore_index=True).to_csv(score_path, index=False)
    c1 = write_counts(cdir / "c1.csv", plate="plate1", seed=5)
    c2 = write_counts(cdir / "c2.csv", plate="plate2", seed=6)

    counts, scores, audit = load_regression_input_pairs([
        {"score": str(score_path), "count": c1},
        {"score": str(score_path), "count": c2},
    ])
    assert set(scores["plateID"]) == {"plate1", "plate2"}
    assert len(scores) == 2 * len(_score_records("plate1", 3))
    assert set(counts["plateID"]) == {"plate1", "plate2"}
    assert all("subset" in row["rule"] for row in audit)


def test_path_is_not_used_to_guess_missing_well_columns(tmp_path, stubs):
    """Filename parsing is a picker hint, not runtime metadata authority."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    s1 = write_scores(sdir / "s1.csv", plate="p", seed=3, with_path=True,
                      plate_token="PLATE1", drop=("rowID", "columnID"))
    c1 = write_counts(cdir / "c1.csv", plate="p", seed=5)
    meta = write_metadata(tmp_path / "md.csv")

    settings = base_settings({"score": s1, "count": c1, "meta": meta})
    with pytest.raises(ValueError, match="rowID.*columnID"):
        perform_regression(settings)


def test_declared_plate_is_kept_without_inspecting_image_paths(
        tmp_path, stubs, capsys):
    """Runtime identity comes from CSV columns even when paths are malformed."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv", plate="plate1", with_path=True,
                         plate_token="PLATE1", n_bad_paths=6)
    count = write_counts(cdir / "counts.csv", plate="plate1")
    meta = write_metadata(tmp_path / "md.csv")

    settings = base_settings({"score": score, "count": count, "meta": meta},
                             verbose=True)
    out = perform_regression(settings)
    printed = capsys.readouterr().out
    assert "PLATEn_ prefix" not in printed

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    assert set(data["plateID"].unique()) == {"plate1"}
    assert len(out["results"]) > 0


def test_missing_plate_column_defaults_to_file_position(tmp_path, stubs):
    """CSVs with no plateID column at all fall back to 'plate{i+1}'."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv", drop=("plateID",))
    count = write_counts(cdir / "counts.csv", drop=("plateID",))
    meta = write_metadata(tmp_path / "md.csv")

    assert "plateID" not in pd.read_csv(score).columns
    assert "plateID" not in pd.read_csv(count).columns

    settings = base_settings({"score": score, "count": count, "meta": meta})
    out = perform_regression(settings)

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    assert set(data["plateID"].unique()) == {"plate1"}
    assert len(out["results"]) > 0


def test_non_list_filter_value_disables_control_well_removal(screen, stubs):
    """A scalar filter_value clears the count-side filter, keeping every well."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_value="c4")
    perform_regression(settings)

    data = pd.read_csv(os.path.join(screen["res"], "regression_data.csv"))
    # nothing was dropped: all six columns (including 'c4') survive
    assert set(data["columnID"].unique()) == set(COLS)
    assert len(data) == len(ROWS) * len(COLS) * len(GENES) * N_GRNA_PER_GENE


# ---------------------------------------------------------------------------
# threshold / simulation / sequencing-stats wiring
# ---------------------------------------------------------------------------

def test_min_cell_count_none_is_filled_from_the_simulation(screen, stubs):
    """min_cell_count=None takes the elbow point returned by the simulation."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, min_cell_count=None)
    perform_regression(settings)

    assert settings["min_cell_count"] == 3
    assert stubs["sim"] == [{"tolerance": 0.02}]


def test_fraction_threshold_none_is_filled_from_graph_sequencing_stats(
        screen, stubs, monkeypatch):
    """fraction_threshold=None delegates the cutoff to graph_sequencing_stats."""
    from spacr.ml import perform_regression
    seen = []

    def fake_stats(settings):
        seen.append(settings["count_data"])
        return 0.004

    # Patch the callable's actual global. Package lazy-loader tests can replace
    # ``spacr.sequencing`` in sys.modules while this already-imported function
    # still resolves names from its original module object.
    monkeypatch.setitem(
        perform_regression.__globals__, "_graph_sequencing_stats", fake_stats,
    )

    settings = base_settings(screen, fraction_threshold=None)
    perform_regression(settings)

    assert settings["fraction_threshold"] == 0.004
    assert seen == [[screen["count"]]]


def test_missing_gene_column_raises_keyerror(tmp_path, stubs):
    """gRNA names without the org_gene_grna form leave no 'gene' column."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv")
    count = write_counts(cdir / "counts.csv",
                         grnas=[f"guide{i}" for i in range(1, 8)])
    meta = write_metadata(tmp_path / "md.csv")

    settings = base_settings({"score": score, "count": count, "meta": meta})
    with pytest.raises(KeyError, match="Column 'gene' not found in independent_df"):
        perform_regression(settings)


# ---------------------------------------------------------------------------
# the QC block (regression_data.csv / grna_well.csv / well_grna.csv)
# ---------------------------------------------------------------------------

def test_qc_block_writes_the_three_well_level_tables(screen, stubs):
    """The QC block saves regression data + gRNA/well metrics and plots each."""
    from spacr.ml import perform_regression

    perform_regression(base_settings(screen))

    res = screen["res"]
    data = pd.read_csv(os.path.join(res, "regression_data.csv"))
    grna_well = pd.read_csv(os.path.join(res, "grna_well.csv"))
    well_grna = pd.read_csv(os.path.join(res, "well_grna.csv"))

    assert set(data.columns) >= {"prc", "grna", "gene", "fraction", "pred",
                                 "cell_count", "plateID", "rowID", "columnID"}
    assert set(grna_well.columns) == {"grna", "plateID", "grna_well_count",
                                      "gene_well_count"}
    # 12 gRNAs on one plate, each seen in all 9 non-control wells.
    assert len(grna_well) == len(GENES) * N_GRNA_PER_GENE
    assert (grna_well["grna_well_count"] == len(ROWS) * len(KEPT_COLS)).all()
    assert (grna_well["gene_well_count"] == len(ROWS) * len(KEPT_COLS)).all()
    # one row per well, each holding all 4 genes
    assert len(well_grna) == len(ROWS) * len(KEPT_COLS)
    assert (well_grna["gene_count"] == len(GENES)).all()

    names = [c["graph_name"] for c in stubs["csv_plots"]]
    assert names == ["cell_count", "wells_per_gene", "gene_per_well"]
    assert stubs["csv_plots"][1]["src"].endswith("grna_well.csv")
    assert stubs["csv_plots"][2]["src"].endswith("well_grna.csv")
    # plot_plates got the merged frame and the *original* dependent variable.
    # The house-style panel got the merged frame and the ORIGINAL dependent
    # variable -- not the transformed one, because a plate heatmap of
    # log(pred) is a heatmap of a different quantity than the screen measured.
    assert stubs["house_plates"], (
        "no plate panel was drawn; neither the house-style path nor the "
        "fallback ran")
    assert stubs["house_plates"][0]["variable"] == "pred"
    assert stubs["house_plates"][0]["dst"] == res
    assert not stubs["plates"], (
        "the legacy plot_plates ran as well; a run should draw its plates "
        "once, in one idiom")


def test_batch_correction_runs_before_regression_and_writes_report(
        screen, stubs):
    """Regression consumes corrected scores and persists its diagnostics."""
    from spacr.ml import perform_regression

    perform_regression(base_settings(
        screen,
        batch_correction="center",
        batch_column="plateID",
    ))

    path = os.path.join(screen["res"], "batch_correction.json")
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    assert report["method"] == "center"
    assert report["batch_column"] == "plateID"
    assert report["rows"] > 0
    assert report["warnings"] == [
        "Only 1 batch was present; correction was a no-op.",
    ]


def test_outlier_detection_drops_sparsely_covered_grnas(tmp_path, stubs):
    """outlier_detection removes gRNAs whose well coverage is an IQR outlier."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    sparse = "TGGT1_111111_3"
    s1 = write_scores(sdir / "s1.csv", plate="plate1", seed=3)
    s2 = write_scores(sdir / "s2.csv", plate="plate2", seed=4)
    c1 = write_counts(cdir / "c1.csv", plate="plate1", seed=5,
                      sparse_grna=sparse, sparse_well=("r1", "c4"))
    c2 = write_counts(cdir / "c2.csv", plate="plate2", seed=6)
    meta = write_metadata(tmp_path / "md.csv")

    settings = base_settings({"score": s1, "count": c1, "meta": meta},
                             score_data=[s1, s2], count_data=[c1, c2],
                             outlier_detection=True)
    perform_regression(settings)

    res = results_dir(c1)
    grna_well = pd.read_csv(os.path.join(res, "grna_well.csv"))
    data = pd.read_csv(os.path.join(res, "regression_data.csv"))

    assert "111111_3" not in set(grna_well["grna"])
    assert "111111_3" not in set(data["grna"])
    # the outlier gRNA is removed from *both* plates
    assert len(grna_well) == 2 * (len(GENES) * N_GRNA_PER_GENE - 1)


def test_qc_plot_failure_does_not_cost_the_qc_tables(screen, heavy_stubs,
                                                     monkeypatch, capsys):
    """A failing QC *plot* is reported; the QC *tables* are still written.

    The tables are data outputs interleaved with the plots inside one big
    try/except, so a single bad plot used to take out every write that came
    after it.
    """
    from spacr.ml import perform_regression
    import spacr.plot as P

    def boom(settings):
        raise RuntimeError("qc plot exploded")

    monkeypatch.setattr(P, "plot_data_from_csv", boom)

    out = perform_regression(base_settings(screen))
    printed = capsys.readouterr().out
    assert "qc plot exploded" in printed
    assert "Skipping QC plot 'cell_count'" in printed
    # every QC table still made it to disk ...
    assert os.path.isfile(os.path.join(screen["res"], "grna_well.csv"))
    assert os.path.isfile(os.path.join(screen["res"], "well_grna.csv"))
    # ... and the regression itself still completed.
    assert os.path.isfile(os.path.join(screen["res"], "results.csv"))
    assert len(out["results"]) > 0


def test_qc_block_failure_is_swallowed_and_the_run_continues(screen, heavy_stubs,
                                                             monkeypatch, capsys):
    """A non-plot failure inside the QC block is printed, not propagated."""
    from spacr.ml import perform_regression

    real_to_csv = pd.DataFrame.to_csv

    def exploding_to_csv(self, path_or_buf=None, *args, **kwargs):
        if isinstance(path_or_buf, str) and path_or_buf.endswith("grna_well.csv"):
            raise RuntimeError("qc table exploded")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", exploding_to_csv)

    out = perform_regression(base_settings(screen))
    assert "qc table exploded" in capsys.readouterr().out
    # the QC tables produced after the failing write were never written ...
    assert not os.path.exists(os.path.join(screen["res"], "well_grna.csv"))
    # ... but the regression itself still completed.
    assert os.path.isfile(os.path.join(screen["res"], "results.csv"))
    assert len(out["results"]) > 0


# ---------------------------------------------------------------------------
# control-derived significance threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["var", "variance", "std",
                                    "standard_deveation"])
def test_threshold_methods_are_supported(screen, stubs, method):
    """Both the variance- and std-based control thresholds run to completion."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, threshold_method=method, verbose=True)
    out = perform_regression(settings)

    sig = pd.read_csv(os.path.join(screen["res"], "results_significant.csv"))
    assert list(sig.columns) == list(out["significant"].columns)
    assert len(sig) == len(out["significant"])
    assert not sig["feature"].str.contains("row|column").any()


def test_unsupported_threshold_method_raises(screen, stubs):
    """An unknown threshold_method is rejected with the supported list."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, threshold_method="mad")
    with pytest.raises(ValueError, match="Unsupported threshold method mad"):
        perform_regression(settings)


def test_controls_none_skips_the_threshold_block(screen, stubs):
    """With controls=None every q<alpha coefficient survives unfiltered.

    This used to assert against the RAW p value, and passed only because the
    Intercept -- p = 1.7e-18, and not a hypothesis about any guide -- was
    being counted as a screen hit. The correction is now applied across the
    tested coefficients only, so the intercept and the row/column nuisance
    terms are excluded from both the family and the hit list.
    """
    from spacr.ml import perform_regression

    settings = base_settings(screen, controls=None,
                             multiple_testing_method="none")
    out = perform_regression(settings)

    results = out["results"]
    tested = ~results["feature"].astype(str).str.contains(
        "row|column|Intercept", case=False, regex=True)
    sig = out["significant"]
    assert len(sig) == int((results.loc[tested, "q_value"] < 0.05).sum())
    assert (sig["q_value"] < 0.05).all()
    # No nuisance term can reach the hit list, whatever its p value.
    assert not sig["feature"].astype(str).str.contains(
        "Intercept", case=False).any()


def test_the_correction_is_actually_applied_to_the_parametric_fit():
    """multiple_testing_method must change the parametric hit list.

    It existed as a setting, was offered in the panel and named in Methods
    sections, while this branch called hits on the raw OLS p value. On the
    real screen that is 56 uncorrected hits against 10 under
    Benjamini-Hochberg -- the defect behind a published volcano that drew a
    P = 0.05 line while its Methods claimed BH q < 0.05.
    """
    import numpy as np
    from spacr.multiple_testing import adjust_p_values

    # 1,200 coefficients, 60 of which beat 0.05 by chance alone.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1200)
    raw = int((p <= 0.05).sum())
    corrected, _ = adjust_p_values(p, "fdr_bh", 0.05)
    assert raw > 40, "the premise is that uncorrected testing finds noise"
    assert int((corrected < 0.05).sum()) == 0


def test_min_n_filters_the_significant_hits(screen, stubs):
    """results_significant_filtered.csv keeps only well-covered features."""
    from spacr.ml import perform_regression

    # alpha=1 makes every tested coefficient a hit, so there is something for
    # min_n to filter. The fixture's only sub-0.05 p value belonged to the
    # Intercept, which is now correctly excluded from the tested family --
    # so a filter test can no longer borrow it, and asks for hits explicitly
    # instead of depending on one arriving by accident.
    settings = base_settings(screen, min_n=1000,
                             multiple_testing_method="none",
                             fdr_alpha=0.999)
    out = perform_regression(settings)

    sig = pd.read_csv(os.path.join(screen["res"], "results_significant.csv"))
    filt = pd.read_csv(
        os.path.join(screen["res"], "results_significant_filtered.csv"))
    # hits exist, but none has >1000 wells of gRNA or gene support
    assert len(sig) == len(out["significant"]) > 0
    assert len(filt) == 0
    # And no nuisance term reached the hit list even at alpha=0.999.
    assert not sig["feature"].astype(str).str.contains(
        "Intercept|row|column", case=False).any()


# ---------------------------------------------------------------------------
# regression back-ends
# ---------------------------------------------------------------------------

def test_ols_results_tables_carry_grna_and_gene_annotations(screen, stubs):
    """results/gene/grna CSVs split the patsy feature names back apart."""
    from spacr.ml import perform_regression

    out = perform_regression(base_settings(screen))

    res = screen["res"]
    ids = {"gene": str, "grna": str}
    results = pd.read_csv(os.path.join(res, "results.csv"), dtype=ids)
    gene = pd.read_csv(os.path.join(res, "results_gene.csv"), dtype=ids)
    grna = pd.read_csv(os.path.join(res, "results_grna.csv"), dtype=ids)

    assert {"feature", "coefficient", "p_value", "grna", "gene",
            "n_grna", "n_gene"} <= set(results.columns)
    # every gene-level row is a gene_fraction term with a non-null count
    assert gene["feature"].str.startswith("gene_fraction:gene[").all()
    assert gene["n_gene"].notna().all()
    assert set(gene["gene"]) == set(GENES)
    assert grna["feature"].str.startswith("fraction:grna[").all()
    assert grna["n_grna"].notna().all()
    assert len(grna) == len(GENES) * N_GRNA_PER_GENE
    assert len(results) == len(gene) + len(grna) + 1   # + Intercept


def test_verbose_ols_writes_the_model_summary(screen, stubs):
    """verbose + ols dumps the statsmodels summary next to the results."""
    from spacr.ml import perform_regression

    perform_regression(base_settings(screen, verbose=True))

    summary = os.path.join(screen["res"], "mode_summary.csv")
    assert os.path.isfile(summary)
    with open(summary) as fh:
        text = fh.read()
    assert "OLS Regression Results" in text


def test_ridge_regression_backend(screen, stubs):
    """regression_type='ridge' produces coefficients for every design column."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="ridge", alpha=1.0)
    out = perform_regression(settings)

    res = results_dir(screen["count"], "ridge")
    assert os.path.isfile(os.path.join(res, "results.csv"))
    assert out["results"]["coefficient"].notna().all()
    assert (out["results"]["feature"].str.contains("grna\\[").sum()
            == len(GENES) * N_GRNA_PER_GENE)


def test_regression_type_none_uses_the_auto_results_folder(screen, stubs, capsys):
    """regression_type=None auto-detects the model and writes under 'auto'."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type=None)
    out = perform_regression(settings)

    printed = capsys.readouterr().out
    # a per-well mean score strictly inside (0, 1) -> beta regression
    assert "Using regression type: beta" in printed
    res = results_dir(screen["count"], "auto")
    assert os.path.isfile(os.path.join(res, "results.csv"))
    assert {"std_err", "wald_stat"} <= set(out["results"].columns)
    assert len(out["results"]) > 0


def test_regression_type_quantile_fits_the_requested_quantile(screen, stubs):
    """'quantile' fits end to end instead of dying at the last statement.

    It used to pass the entry-point whitelist, get its own agg_type handling
    in get_perform_regression_default_settings and its own volcano-filename
    rule, and then raise "Unsupported regression type quantile" from
    regression_model - after both input CSVs had been read, the QC plots drawn
    and regression_data.csv written.
    """
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="quantile", quantile=0.75)
    out = perform_regression(settings)

    # agg_type is forced to None for quantile, so the fit is on objects.
    assert settings["agg_type"] is None
    res = results_dir(screen["count"], "quantile")
    assert os.path.isfile(os.path.join(res, "results.csv"))
    assert out["results"]["coefficient"].notna().all()
    assert out["results"]["p_value"].notna().all()
    assert (out["results"]["feature"].str.contains("grna\\[").sum()
            == len(GENES) * N_GRNA_PER_GENE)


def test_quantile_regression_refuses_the_old_alpha_spelling(screen, stubs):
    """alpha used to double as the quantile; the overload is refused, not ignored.

    A settings CSV written before the split says alpha=0.75 and means "the
    75th percentile". Silently dropping it would fit the median and label the
    output folder as a quantile run.
    """
    from spacr.ml import perform_regression

    with pytest.raises(ValueError, match=r"does not use alpha"):
        perform_regression(base_settings(screen, regression_type="quantile",
                                         alpha=0.75))


def test_lasso_uses_bootstrap_selection_frequencies(screen, stubs):
    """Lasso hits are ranked by bootstrap selection frequency, not p-values."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="lasso", alpha=0.0005,
                             lasso_n_boot=3, lasso_selection_threshold=0.5)
    out = perform_regression(settings)

    results = out["results"]
    assert {"selection_frequency", "mean_coefficient"} <= set(results.columns)
    freq = results["selection_frequency"].dropna()
    assert len(freq) > 0
    assert ((freq >= 0) & (freq <= 1)).all()
    sig = out["significant"]
    assert (sig["coefficient"] != 0).all()
    assert (sig["selection_frequency"] >= 0.5).all()
    # sorted by |coefficient| descending
    assert list(sig["coefficient"].abs()) == sorted(
        sig["coefficient"].abs(), reverse=True)


def test_lasso_with_auto_alpha_cross_validates_each_resample(screen, stubs):
    """alpha='auto' switches both the fit and the bootstrap to LassoCV."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="lasso", alpha="auto",
                             lasso_n_boot=2, lasso_selection_threshold=0.0)
    out = perform_regression(settings)

    results = out["results"]
    assert "selection_frequency" in results.columns
    freq = results["selection_frequency"].dropna()
    assert len(freq) == len(results)
    # every resample succeeded, so the frequency is a multiple of 1/2
    assert set(np.unique(np.round(freq.values * 2, 6))) <= {0.0, 1.0, 2.0}
    assert results["mean_coefficient"].notna().all()


def test_lasso_bootstrap_raises_when_every_resample_fails(screen, stubs,
                                                          monkeypatch):
    """If no bootstrap resample can be designed the helper refuses to guess."""
    import spacr.ml as ML
    from spacr.ml import perform_regression

    real = ML.dmatrices
    state = {"n": 0}

    def flaky(formula, data=None, return_type=None, **kwargs):
        state["n"] += 1
        if state["n"] > 2:
            raise ValueError("factor level vanished from resample")
        return real(formula, data=data, return_type=return_type, **kwargs)

    settings = base_settings(screen, regression_type="lasso", alpha=0.0005,
                             lasso_n_boot=4)
    monkeypatch.setattr(ML, "dmatrices", flaky)
    with pytest.raises(RuntimeError, match="All bootstrap resamples failed"):
        perform_regression(settings)
    assert state["n"] > 2


# ---------------------------------------------------------------------------
# metadata merge + toxo reporting block
# ---------------------------------------------------------------------------

def test_metadata_file_string_is_wrapped_and_merged(screen, stubs):
    """A single metadata_files string is wrapped and merged into every table."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, metadata_files=screen["meta"])
    perform_regression(settings)

    assert settings["metadata_files"] == [screen["meta"]]
    name = os.path.splitext(os.path.basename(screen["meta"]))[0]
    for stem in ("results", "results_gene", "results_grna",
                 "results_significant"):
        merged = os.path.join(screen["res"], f"{stem}{name}.csv")
        assert os.path.isfile(merged), merged
    merged_df = pd.read_csv(os.path.join(screen["res"], f"results{name}.csv"),
                            dtype={"gene": str})
    assert "Gene Name" in merged_df.columns
    assert merged_df.loc[merged_df["gene"] == "239740", "Gene Name"].iloc[0] \
        == "gene_239740"


@pytest.mark.parametrize("volcano", ["all", "gene", "grna"])
def test_toxo_block_renders_the_requested_volcano(screen, toxo_stubs, volcano):
    """Each volcano mode feeds a different merged table to the toxo plot."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, toxo=True, volcano=volcano)
    perform_regression(settings)

    assert len(toxo_stubs["volcano"]) == 1
    call = toxo_stubs["volcano"][0]
    assert call["metadata_path"].endswith(os.path.join("resources", "data",
                                                       "lopit.csv"))
    assert call["kwargs"]["save_path"].endswith("volcano_plot.pdf")
    assert call["kwargs"]["metadata_column"] == "tagm_location"
    # the duplicated tail block calls the phenotype/heatmap plots twice
    # Once, not twice. The phenotype/heatmap block used to be duplicated
    # verbatim, so every report was built twice and the second copy was
    # unguarded -- a run with fewer than two metadata files died in it after
    # the volcano had already been drawn.
    assert len(toxo_stubs["phenotypes"]) == 1
    assert len(toxo_stubs["heatmaps"]) == 1
    assert toxo_stubs["heatmaps"][0]["columns"][0] == "sense - Tachyzoites"


def test_toxo_block_skips_unknown_volcano_mode(screen, toxo_stubs, capsys):
    """An unrecognised volcano setting skips the plot but still reports."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, toxo=True, volcano="none",
                             controls=None)
    perform_regression(settings)

    printed = capsys.readouterr().out
    assert "Skipping volcano plot" in printed
    assert "No gene_list produced" in printed
    assert toxo_stubs["volcano"] == []
    # No volcano means no gene list, so nothing downstream is drawn. The
    # duplicate block used to fire anyway with gene_list=None.
    assert toxo_stubs["phenotypes"] == []


def test_toxo_block_with_empty_gene_list(screen, toxo_stubs, capsys):
    """An empty gene list from the volcano plot skips the phenotype figures."""
    from spacr.ml import perform_regression

    toxo_stubs["gene_list"] = []
    settings = base_settings(screen, toxo=True, volcano="gene")
    perform_regression(settings)

    printed = capsys.readouterr().out
    assert "No gene_list produced" in printed
    # An empty gene list draws nothing; there is no second unguarded copy.
    assert toxo_stubs["phenotypes"] == []


# ---------------------------------------------------------------------------
# filter_column shapes
# ---------------------------------------------------------------------------

def test_filter_column_may_be_a_list(screen, stubs):
    """process_reads accepts a list of filter columns; so should the caller.

    The list used to reach clean_controls' `column in df.columns` membership
    test and raise "unhashable type: 'list'".
    """
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_column=["columnID"])
    out = perform_regression(settings)
    assert len(out["results"]) > 0


def test_filter_column_may_be_none(screen, stubs, capsys):
    """filter_column=None means 'drop no control wells'.

    The local `filter_column` was only bound in the `isinstance(..., str)`
    branch, so None fell through and the process_reads call below raised
    UnboundLocalError before any filtering decision was made.
    """
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_column=None)
    out = perform_regression(settings)

    assert len(out["results"]) > 0
    # clean_controls announces every value it drops; None must drop nothing.
    assert "Removed data from" not in capsys.readouterr().out


def test_qc_tables_are_written_with_the_real_plot_helper(screen, heavy_stubs):
    """The gRNA-coverage QC tables must survive a real plot_data_from_csv."""
    from spacr.ml import perform_regression

    perform_regression(base_settings(screen))
    assert os.path.isfile(os.path.join(screen["res"], "grna_well.csv"))
    assert os.path.isfile(os.path.join(screen["res"], "well_grna.csv"))


def test_toxo_volcano_without_controls(screen, toxo_stubs):
    """A screen with no control gRNAs should still be able to plot a volcano."""
    from spacr.ml import perform_regression

    settings = base_settings(screen, toxo=True, volcano="gene", controls=None)
    perform_regression(settings)
    assert len(toxo_stubs["volcano"]) == 1
