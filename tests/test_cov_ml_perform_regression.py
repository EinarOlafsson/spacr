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


def parametric_settings(screen, **over):
    """`base_settings` for a run that actually FITS the model it names.

    The suite's default inference became `nonparametric` on 2026-08-19, and
    under `analysis_mode='guide_permutation'` the permutation test IS the
    analysis: `regression_type` is never read, no model is fitted, and none
    of the parametric artefacts -- the statsmodels summary, bootstrap
    selection frequencies, the effect-size cut on coefficients, the toxo
    volcanoes -- are produced at all.

    Twenty tests in this file are about those artefacts and were reading
    their absence as a defect. A test of the parametric path has to ASK for
    the parametric path; that is what this is. It is not a workaround for a
    bug -- the permutation default is deliberate and is what a user gets.
    """
    over.setdefault("inference", "parametric")
    over.setdefault("analysis_mode", "regression")
    return base_settings(screen, **over)


def results_dir(count_csv, regression_type="ols", settings=None):
    """Where a run writes: ``<count data folder>/results/<what named it>``.

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

    AND IT HAPPENED AGAIN, for the reason the paragraph above describes: the
    leaf stopped being the regression type. Under
    `analysis_mode='guide_permutation'` -- the default these tests run with --
    the run names the folder after the INFERENCE, because the regression type
    is never read and a folder called `ridge` would name something the run
    did not do. 39 tests in this file went red on missing CSVs that had been
    written all along.

    So the rule is no longer re-derived here. `spacr.ml.results_folder_kind`
    is the one copy and this asks it, which is the only version of this
    helper that cannot go stale a third time.
    """
    from spacr.ml import results_folder_kind
    from spacr.settings import get_perform_regression_default_settings

    if settings is None:
        # THE SAME DEFAULTS THE RUN GETS, for the same reason `base_settings`
        # finishes its dict with this builder: a helper that assumed
        # parametric inference would answer for a run nobody performs.
        settings = get_perform_regression_default_settings({})
        settings["regression_type"] = regression_type
    kind = results_folder_kind(settings)
    return os.path.join(os.path.dirname(count_csv), "results", kind)


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
    the tiny synthetic wells (``min_cells_per_well``), a fixed threshold instead of
    the sweep (``fraction_threshold``), the toxo reports off by default. Adding
    a key here that the builder is supposed to supply hides the next such bug.
    """
    from spacr.settings import get_perform_regression_default_settings

    settings = {
        "score_data": [screen["score"]],
        "count_data": [screen["count"]],
        "dependent_variable": "pred",
        "regression_type": "ols",
        "min_cells_per_well": 3,
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

def case_scalar_score_and_count_paths_are_wrapped_in_lists(screen, stubs):
    """A bare string for score_data/count_data is normalised to a 1-list."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, score_data=screen["score"],
                             count_data=screen["count"])
    out = perform_regression(settings)

    _seen.append(bool(
        settings["score_data"] == [screen["score"]]))
    _seen.append(bool(
        settings["count_data"] == [screen["count"]]))
    # The coefficient table and the hits, plus what is needed to judge the
    # fit: without the model there is no R-squared and no residual to test,
    # and without the design there is no way to count what reached it.
    _seen.append(bool(
        {"results", "significant"} <= set(out)))
    _seen.append(bool(
        {"model", "model_data"} <= set(out)))
    _seen.append(bool(
        os.path.isfile(os.path.join(results_dir(screen["count"], settings=settings), "results.csv"))))
    return _seen


@pytest.mark.parametrize("level", ["grna", "gene", "both"])
@pytest.mark.parametrize("layout", ["long", "wide"])
def test_regression_returns_prepared_data_separate_from_annotated_coefficients(
        screen, stubs, level, layout):
    import json
    from spacr.ml import perform_regression
    from spacr.regression_summary import build_run_summary
    from spacr.trial_metrics import design_summary

    settings = parametric_settings(
        screen, regression_type="ridge", level=level, model_data_layout=layout,
        regression_qc=False, model_plate_position=False)
    output = perform_regression(settings)
    prepared = output["model_data"]
    saved = pd.read_csv(os.path.join(output["res_folder"], "regression_data.csv"),
                        dtype={"gene": str, "grna": str, "prc": str})
    pd.testing.assert_frame_equal(prepared.reset_index(drop=True), saved,
                                  check_dtype=False)
    assert {"prc", "grna", "gene", "fraction"} <= set(prepared.columns)
    assert "coefficient" not in prepared
    assert "coefficient" in output["results"]
    expected_levels = {"grna", "gene"} if level == "both" else {level}
    assert set(output["fit_designs"]) == expected_levels
    expected_rows = len(prepared) if layout == "long" else prepared["prc"].nunique()
    for counts in output["fit_designs"].values():
        assert counts["n_rows_fitted"] == expected_rows
        assert counts["n_wells"] == prepared["prc"].nunique()
    summary = design_summary(output)
    assert summary["n_rows_prepared"] == len(prepared)
    assert summary["n_rows_fitted"] == expected_rows
    with open(os.path.join(output["res_folder"], "regression_fit_designs.json")) as handle:
        assert json.load(handle) == output["fit_designs"]
    reopened = build_run_summary(res_folder=output["res_folder"], settings=settings)
    assert reopened.field("n_rows_prepared").value.startswith(f"{len(prepared):,} rows")
    for one in expected_levels:
        assert f"{one}: {expected_rows:,}" in reopened.field("n_rows_fitted").value


def case_legacy_score_list_longer_than_count_list_migrates(screen):
    """An unpaired tail remains legal because the final join is by well."""
    _seen = []
    from spacr.ml import normalize_regression_input_pairs

    settings = base_settings(
        screen, score_data=[screen["score"], screen["score"]])
    pairs, migrated = normalize_regression_input_pairs(settings)
    _seen.append(bool(
        migrated))
    _seen.append(bool(
        pairs[1] == {"score": screen["score"], "count": None,
                        "plate": None}))
    return _seen


def case_legacy_count_list_longer_than_score_list_migrates(screen):
    """The opposite unpaired tail is retained too."""
    _seen = []
    from spacr.ml import normalize_regression_input_pairs

    settings = base_settings(
        screen, count_data=[screen["count"], screen["count"]])
    pairs, migrated = normalize_regression_input_pairs(settings)
    _seen.append(bool(
        migrated))
    _seen.append(bool(
        pairs[1] == {"score": None, "count": screen["count"],
                        "plate": None}))
    return _seen


def case_paired_input_copies_plate_identity_from_its_partner(tmp_path, stubs):
    """A score file without plateID inherits the count file's declared plate."""
    _seen = []
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
    _seen.append(keep(set(data["plateID"].unique())))
    _seen.append(bool(
        data["prc"].str.startswith("plate2_").all()))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_missing_dependent_variable_raises(screen, stubs):
    """A dependent_variable absent from the score CSV is rejected."""
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, dependent_variable="not_a_column")
    # The message names the setting and the columns actually present, and --
    # when the score slot holds a count table -- says the inputs look swapped,
    # which is how this error is usually reached.
    with pytest.raises(ValueError, match="'not_a_column' is not a column") as _exc1:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_missing_pred_names_the_file_and_says_how_to_get_it(screen, stubs):
    """The error names the score file and points at Classify's output."""
    _seen = []
    from spacr.ml import perform_regression

    table = pd.read_csv(screen["score"]).rename(columns={"pred": "measured"})
    table.to_csv(screen["score"], index=False)
    settings = base_settings(screen, dependent_variable="pred")
    with pytest.raises(ValueError) as caught:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    message = str(caught.value)
    _seen.append(bool(
        os.path.basename(screen["score"]) in message))
    _seen.append(('Run Classify' if 'Run Classify' in (message) else ("missing", 'Run Classify')))
    _seen.append(('_result.csv' if '_result.csv' in (message) else ("missing", '_result.csv')))
    return _seen


def case_other_missing_response_names_the_file_without_the_pred_advice(
        screen, stubs):
    """A misspelt response names the file but does not talk about Classify."""
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, dependent_variable="not_a_column")
    with pytest.raises(ValueError) as caught:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    _seen.append(bool(
        os.path.basename(screen["score"]) in str(caught.value)))
    _seen.append((("absent", 'Classify') if 'Classify' not in (str(caught.value)) else ("present", 'Classify')))
    return _seen


def case_shortest_distance_dependent_variable_is_exempt_from_the_check(screen, stubs):
    """'pathogen_nucleus_shortest_distance' skips the column check ...

    ... and therefore only fails later, in process_scores, when the column is
    genuinely absent.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(
        screen, dependent_variable="pathogen_nucleus_shortest_distance")
    with pytest.raises(KeyError) as _exc1:
        perform_regression(settings)
    _seen.append(('KeyError' if isinstance(_exc1.value, KeyError) else type(_exc1.value).__name__))
    return _seen


def case_unsupported_regression_type_raises(screen, stubs):
    """regression_type is validated against the supported list."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, regression_type="banana")
    with pytest.raises(ValueError, match="Unsupported regression type banana") as _exc1:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# plate / row / column recovery
# ---------------------------------------------------------------------------

def case_rowid_with_plate_prefix_is_split(tmp_path, stubs):
    """A count-CSV rowID of the 'plate1_r2' form is reduced to 'r2'."""
    _seen = []
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
    _seen.append(bool(
        set(data["rowID"].unique()) <= set(ROWS)))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_one_multiplate_score_file_can_be_paired_to_plate_count_files(tmp_path):
    """A consolidated score export may be reused without duplicating plates."""
    _seen = []
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
    _seen.append(keep(set(scores["plateID"])))
    _seen.append(bool(
        len(scores) == 2 * len(_score_records("plate1", 3))))
    _seen.append(keep(set(counts["plateID"])))
    _seen.append(bool(
        all("subset" in row["rule"] for row in audit)))
    return _seen


def case_path_is_not_used_to_guess_missing_well_columns(tmp_path, stubs):
    """Filename parsing is a picker hint, not runtime metadata authority."""
    _seen = []
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
    with pytest.raises(ValueError, match="rowID.*columnID") as _exc1:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_declared_plate_is_kept_without_inspecting_image_paths(
        tmp_path, stubs, capsys):
    """Runtime identity comes from CSV columns even when paths are malformed."""
    _seen = []
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
    _seen.append((("absent", 'PLATEn_ prefix') if 'PLATEn_ prefix' not in (printed) else ("present", 'PLATEn_ prefix')))

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    _seen.append(keep(set(data["plateID"].unique())))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_missing_plate_column_defaults_to_file_position(tmp_path, stubs):
    """CSVs with no plateID column at all fall back to 'plate{i+1}'."""
    _seen = []
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv", drop=("plateID",))
    count = write_counts(cdir / "counts.csv", drop=("plateID",))
    meta = write_metadata(tmp_path / "md.csv")

    _seen.append((("absent", 'plateID') if 'plateID' not in (pd.read_csv(score).columns) else ("present", 'plateID')))
    _seen.append((("absent", 'plateID') if 'plateID' not in (pd.read_csv(count).columns) else ("present", 'plateID')))

    settings = base_settings({"score": score, "count": count, "meta": meta})
    out = perform_regression(settings)

    data = pd.read_csv(os.path.join(results_dir(count), "regression_data.csv"))
    _seen.append(keep(set(data["plateID"].unique())))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_non_list_filter_value_disables_control_well_removal(screen, stubs):
    """A scalar filter_value clears the count-side filter, keeping every well."""
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_value="c4")
    perform_regression(settings)

    data = pd.read_csv(os.path.join(screen["res"], "regression_data.csv"))
    # nothing was dropped: all six columns (including 'c4') survive
    _seen.append(bool(
        set(data["columnID"].unique()) == set(COLS)))
    _seen.append(bool(
        len(data) == len(ROWS) * len(COLS) * len(GENES) * N_GRNA_PER_GENE))
    return _seen


# ---------------------------------------------------------------------------
# threshold / simulation / sequencing-stats wiring
# ---------------------------------------------------------------------------

def case_min_cells_per_well_none_is_filled_from_the_simulation(screen, stubs):
    """min_cells_per_well=None takes the elbow point returned by the simulation."""
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, min_cells_per_well=None)
    perform_regression(settings)

    _seen.append(keep(settings["min_cells_per_well"]))
    # `dst` joined `tolerance` on 2026-08-17. It is not decoration: the curve
    # used to be drawn into <count folder>/results/, one path shared by every
    # run of the screen, and `run_sweep_parallel` fits n_jobs trials of that
    # one screen at once -- so the workers overwrote each other's figure and
    # the copy-back could file a neighbour's curve as this trial's. The run
    # folder is unique per run, so naming it removes the collision rather
    # than racing it. Pinned as an exact dict, not a subset, so a future
    # caller cannot quietly send the figure somewhere else again.
    _seen.append(bool(
        stubs["sim"] == [{"tolerance": 0.02, "dst": screen["res"]}]))
    return _seen


def case_fraction_threshold_none_is_filled_from_graph_sequencing_stats(
        screen, stubs, monkeypatch):
    """fraction_threshold=None delegates the cutoff to graph_sequencing_stats."""
    _seen = []
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

    _seen.append(keep(settings["fraction_threshold"]))
    _seen.append(bool(
        seen == [[screen["count"]]]))
    return _seen


def case_missing_gene_column_raises_keyerror(tmp_path, stubs):
    """gRNA names without the org_gene_grna form leave no 'gene' column."""
    _seen = []
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
    with pytest.raises(KeyError, match="Column 'gene' not found in independent_df") as _exc1:
        perform_regression(settings)
    _seen.append(('KeyError' if isinstance(_exc1.value, KeyError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# the QC block (regression_data.csv / grna_well.csv / well_grna.csv)
# ---------------------------------------------------------------------------

def case_qc_block_writes_the_three_well_level_tables(screen, stubs):
    """The QC block saves regression data + gRNA/well metrics and plots each."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen)
    perform_regression(settings)

    res = results_dir(screen["count"], settings=settings)
    data = pd.read_csv(os.path.join(res, "regression_data.csv"))
    grna_well = pd.read_csv(os.path.join(res, "grna_well.csv"))
    well_grna = pd.read_csv(os.path.join(res, "well_grna.csv"))

    _seen.append(bool(
        set(data.columns) >= {"prc", "grna", "gene", "fraction", "pred",
                                 "cell_count", "plateID", "rowID", "columnID"}))
    _seen.append(keep(set(grna_well.columns)))
    # 12 gRNAs on one plate, each seen in all 9 non-control wells.
    _seen.append(bool(
        len(grna_well) == len(GENES) * N_GRNA_PER_GENE))
    _seen.append(bool(
        (grna_well["grna_well_count"] == len(ROWS) * len(KEPT_COLS)).all()))
    _seen.append(bool(
        (grna_well["gene_well_count"] == len(ROWS) * len(KEPT_COLS)).all()))
    # one row per well, each holding all 4 genes
    _seen.append(bool(
        len(well_grna) == len(ROWS) * len(KEPT_COLS)))
    _seen.append(bool(
        (well_grna["gene_count"] == len(GENES)).all()))

    names = [c["graph_name"] for c in stubs["csv_plots"]]
    _seen.append(keep(names))
    _seen.append(bool(
        stubs["csv_plots"][1]["src"].endswith("grna_well.csv")))
    _seen.append(bool(
        stubs["csv_plots"][2]["src"].endswith("well_grna.csv")))
    # plot_plates got the merged frame and the *original* dependent variable.
    # The house-style panel got the merged frame and the ORIGINAL dependent
    # variable -- not the transformed one, because a plate heatmap of
    # log(pred) is a heatmap of a different quantity than the screen measured.
    _seen.append(bool(
        stubs["house_plates"]))
    _seen.append(keep(stubs["house_plates"][0]["variable"]))
    _seen.append(bool(
        stubs["house_plates"][0]["dst"] == res))
    _seen.append(bool(
        not stubs["plates"]))
    return _seen


def case_batch_correction_runs_before_regression_and_writes_report(
        screen, stubs):
    """Regression consumes corrected scores and persists its diagnostics."""
    _seen = []
    from spacr.ml import perform_regression

    perform_regression(base_settings(
        screen,
        batch_correction="center",
        batch_column="plateID",
    ))

    path = os.path.join(screen["res"], "batch_correction.json")
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    _seen.append(keep(report["method"]))
    _seen.append(keep(report["batch_column"]))
    _seen.append(bool(
        report["rows"] > 0))
    _seen.append(keep(report["warnings"]))
    return _seen


def case_outlier_detection_drops_sparsely_covered_grnas(tmp_path, stubs,
                                                         capsys):
    """outlier_detection removes gRNAs whose well coverage is an IQR outlier.

    And leaves the two QC tables on disk while doing it. Dropping the outlier
    rebinds ``merged_df`` to a SLICE, and ``grna_metricks`` assigns
    plate/row/column onto whatever it is handed, so the second pass wrote to a
    copy: SettingWithCopyWarning, promoted to an error by this suite's
    pytest.ini and caught by the blanket handler at the end of the QC block.
    ``grna_well.csv`` and ``well_grna.csv`` were then never written, and the
    only trace was the warning's text printed on a line of its own -- the run
    itself completed and produced every other output, which is why it read as
    a broken test rather than a broken run.
    """
    _seen = []
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
    printed = capsys.readouterr().out

    res = results_dir(c1)
    grna_well = pd.read_csv(os.path.join(res, "grna_well.csv"))
    data = pd.read_csv(os.path.join(res, "regression_data.csv"))

    _seen.append((("absent", '111111_3') if '111111_3' not in (set(grna_well["grna"])) else ("present", '111111_3')))
    _seen.append((("absent", '111111_3') if '111111_3' not in (set(data["grna"])) else ("present", '111111_3')))
    # the outlier gRNA is removed from *both* plates
    _seen.append(bool(
        len(grna_well) == 2 * (len(GENES) * N_GRNA_PER_GENE - 1)))
    # Both QC tables, and the run said so. The file check alone passes if the
    # write is skipped and a stale file is left behind by an earlier run.
    _seen.append(bool(
        os.path.isfile(os.path.join(res, "well_grna.csv"))))
    _seen.append(('Saved grna per well data to' if 'Saved grna per well data to' in (printed) else ("missing", 'Saved grna per well data to')))
    _seen.append(('Saved well per grna data to' if 'Saved well per grna data to' in (printed) else ("missing", 'Saved well per grna data to')))
    # Nothing was swallowed on the way. The block that writes these tables
    # catches everything and prints it, so a bare warning text on its own
    # line IS the failure report.
    _seen.append((("absent", 'A value is trying to be set on a copy of a slice') if 'A value is trying to be set on a copy of a slice' not in (printed) else ("present", 'A value is trying to be set on a copy of a slice')))
    return _seen


def case_qc_plot_failure_does_not_cost_the_qc_tables(screen, heavy_stubs,
                                                     monkeypatch, capsys):
    """A failing QC *plot* is reported; the QC *tables* are still written.

    The tables are data outputs interleaved with the plots inside one big
    try/except, so a single bad plot used to take out every write that came
    after it.
    """
    _seen = []
    from spacr.ml import perform_regression
    import spacr.plot as P

    def boom(settings):
        raise RuntimeError("qc plot exploded")

    monkeypatch.setattr(P, "plot_data_from_csv", boom)

    out = perform_regression(base_settings(screen))
    printed = capsys.readouterr().out
    _seen.append(('qc plot exploded' if 'qc plot exploded' in (printed) else ("missing", 'qc plot exploded')))
    _seen.append(("Skipping QC plot 'cell_count'" if "Skipping QC plot 'cell_count'" in (printed) else ("missing", "Skipping QC plot 'cell_count'")))
    # every QC table still made it to disk ...
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "grna_well.csv"))))
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "well_grna.csv"))))
    # ... and the regression itself still completed.
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "results.csv"))))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_qc_block_failure_is_swallowed_and_the_run_continues(screen, heavy_stubs,
                                                             monkeypatch, capsys):
    """A non-plot failure inside the QC block is printed, not propagated."""
    _seen = []
    from spacr.ml import perform_regression

    real_to_csv = pd.DataFrame.to_csv

    def exploding_to_csv(self, path_or_buf=None, *args, **kwargs):
        if isinstance(path_or_buf, str) and path_or_buf.endswith("grna_well.csv"):
            raise RuntimeError("qc table exploded")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", exploding_to_csv)

    out = perform_regression(base_settings(screen))
    _seen.append(('qc table exploded' if 'qc table exploded' in (capsys.readouterr().out) else ("missing", 'qc table exploded')))
    # the QC tables produced after the failing write were never written ...
    _seen.append(bool(
        not os.path.exists(os.path.join(screen["res"], "well_grna.csv"))))
    # ... but the regression itself still completed.
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "results.csv"))))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


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


def case_unsupported_threshold_method_raises(screen, stubs):
    """An unknown threshold_method is rejected, and the message names the list.

    This asked for ``"mad"`` until 2026-08-17, when the arithmetic moved into
    ``spacr.thresholds`` and grew from two methods to seven -- the maintainer
    asked for "at least 4 more" -- so ``mad`` became a SUPPORTED method and
    the test was asserting a refusal the code had deliberately stopped making.
    It failed at HEAD for that reason and for no other.

    An unsupported spelling is what it was always about, so it asks for one
    that is genuinely not offered, and pins the half that matters: a refusal
    that does not say what IS available leaves the user guessing, which is why
    ``canonical`` raises instead of falling back to a default nobody chose.
    """
    _seen = []
    from spacr.ml import perform_regression
    from spacr.thresholds import METHODS

    _seen.append(('mad' if 'mad' in (METHODS) else ("missing", 'mad')))
    settings = base_settings(screen, threshold_method="banana")
    with pytest.raises(ValueError, match="Unsupported threshold method") as err:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(err.value, ValueError) else type(err.value).__name__))
    for name in METHODS:
        assert name in str(err.value), name
    return _seen


def case_controls_none_skips_the_threshold_block(screen, stubs, capsys):
    """With controls=None every q<alpha coefficient survives unfiltered.

    This used to assert against the RAW p value, and passed only because the
    Intercept -- p = 1.7e-18, and not a hypothesis about any guide -- was
    being counted as a screen hit. The correction is now applied across the
    tested coefficients only, so the intercept and the row/column nuisance
    terms are excluded from both the family and the hit list.

    And the run SAYS there is no cut. A screen with controls prints its cut
    and the rule behind it; a screen without them printed nothing at all, so
    "no effect-size cut was applied" was indistinguishable from that line
    having scrolled past -- and a hit list called on the corrected P value
    alone is a different claim from one that also cleared a width.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, controls=None,
                             multiple_testing_method="none")
    out = perform_regression(settings)
    # CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17): level='both' fits
    # two models and each measures its OWN cut on its own controls, so the
    # line has to say WHICH fit it is about. Both of them must say it -- a
    # run that announced the guide fit's absent cut and stayed quiet about the
    # gene fit's would be exactly the silence this test was written for.
    printed = capsys.readouterr().out
    for level in ("grna", "gene"):
        assert f"Effect-size cut ({level}): no control gRNAs were named" in \
            printed

    results = out["results"]
    tested = ~results["feature"].astype(str).str.contains(
        "row|column|Intercept", case=False, regex=True)
    sig = out["significant"]
    _seen.append(bool(
        len(sig) == int((results.loc[tested, "q_value"] < 0.05).sum())))
    _seen.append(bool(
        (sig["q_value"] < 0.05).all()))
    # No nuisance term can reach the hit list, whatever its p value.
    _seen.append(bool(
        not sig["feature"].astype(str).str.contains(
        "Intercept", case=False).any()))
    return _seen


def case_the_correction_is_actually_applied_to_the_parametric_fit():
    """multiple_testing_method must change the parametric hit list.

    It existed as a setting, was offered in the panel and named in Methods
    sections, while this branch called hits on the raw OLS p value. On the
    real screen that is 56 uncorrected hits against 10 under
    Benjamini-Hochberg -- the defect behind a published volcano that drew a
    P = 0.05 line while its Methods claimed BH q < 0.05.
    """
    _seen = []
    import numpy as np
    from spacr.multiple_testing import adjust_p_values

    # 1,200 coefficients, 60 of which beat 0.05 by chance alone.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1200)
    raw = int((p <= 0.05).sum())
    corrected, _ = adjust_p_values(p, "fdr_bh", 0.05)
    _seen.append(bool(
        raw > 40))
    _seen.append(keep(int((corrected < 0.05).sum())))
    return _seen


def case_min_observations_per_hit_filters_the_significant_hits(screen, stubs):
    """results_significant_filtered.csv keeps only well-covered features."""
    _seen = []
    from spacr.ml import perform_regression

    # alpha=1 makes every tested coefficient a hit, so there is something for
    # min_n to filter. The fixture's only sub-0.05 p value belonged to the
    # Intercept, which is now correctly excluded from the tested family --
    # so a filter test can no longer borrow it, and asks for hits explicitly
    # instead of depending on one arriving by accident.
    #
    # threshold_method='none' was added on 2026-08-17, when the effect-size
    # cut started actually cutting. This screen's three controls have a
    # standard deviation of 0.259 and the default multiplier is 3, so the cut
    # is 0.57 -- wider than every coefficient the fixture produces (the
    # largest is 0.370). Leaving it on would empty the hit list and this test
    # would then pass for the wrong reason, asserting that a filter kept
    # nothing out of nothing. The cut has its own tests below; this one is
    # about min_n, so it says out loud that it does not want a cut.
    settings = parametric_settings(screen, min_observations_per_hit=1000,
                             multiple_testing_method="none",
                             fdr_alpha=0.999,
                             threshold_method="none")
    out = perform_regression(settings)

    sig = pd.read_csv(os.path.join(results_dir(screen["count"], settings=settings), "results_significant.csv"))
    filt = pd.read_csv(
        os.path.join(results_dir(screen["count"], settings=settings), "results_significant_filtered.csv"))
    # hits exist, but none has >1000 wells of gRNA or gene support
    _seen.append(bool(
        len(sig) == len(out["significant"]) > 0))
    _seen.append(keep(len(filt)))
    # And no nuisance term reached the hit list even at alpha=0.999.
    _seen.append(bool(
        not sig["feature"].astype(str).str.contains(
        "Intercept|row|column", case=False).any()))
    return _seen


# ---------------------------------------------------------------------------
# the effect-size cut, on the parametric path
#
# It was a pair of one-sided masks whose union was every row, so the cut
# filtered nothing at all and results_significant.csv disagreed with the
# volcano drawn from the same number. Fixed 2026-08-17; these pin the three
# ways it went wrong, each measured on this fixture before the change.
# ---------------------------------------------------------------------------

def _measured_cut(screen, multiplier, method="std", settings=None):
    """The cut this run's own controls produce, from the module that sets it.

    Recomputed from ``results_grna.csv`` through the same
    ``spacr.thresholds.coefficient_threshold`` the run calls, rather than
    hard-coded: the point of the assertions below is that the hit list and the
    volcano select on ONE number, so the test has to ask for that number the
    way both of them do.

    :param settings: the settings the run used, so the right results folder is
        read. ``screen["res"]`` is computed from the SUITE defaults, which are
        nonparametric -- a parametric run writes somewhere else entirely, and
        reading the default folder finds nothing.
    """
    from spacr.thresholds import coefficient_threshold

    folder = (results_dir(screen["count"], settings=settings)
              if settings is not None else screen["res"])
    grna = pd.read_csv(os.path.join(folder, "results_grna.csv"),
                       dtype={"grna": str})
    controls = grna.loc[grna["grna"].isin(CONTROLS), "coefficient"]
    threshold, _rule = coefficient_threshold(controls, method=method,
                                             multiplier=multiplier,
                                             centre=None)
    return threshold


def case_the_effect_size_cut_is_symmetric_about_zero(screen, stubs):
    """A guide that pushes the phenotype DOWN past the cut is a hit too.

    The cut was applied as ``coefficient >= t`` OR ``coefficient <= t``, and
    ``t`` is ``|median| + k x spread`` and so never negative -- so the union
    was every row and the cut removed nothing. ``custom_volcano_plot``, handed
    the same ``t``, marks hits with ``|coefficient| >= t``, so the figure and
    results_significant.csv described different experiments.

    Whichever direction the cut had been written one-sided, it would call half
    a screen: on this fixture the strongest effect of all is NEGATIVE.
    """
    _seen = []
    from spacr.ml import perform_regression

    # multiplier=0 puts the cut at the controls' median (0.104) rather than
    # three spreads out (0.570), which is wider than anything this small
    # fixture produces. That is the only reason for the 0: it lands the cut in
    # the middle of the coefficients so both sides of it have rows.
    settings = parametric_settings(screen, multiple_testing_method="none",
                             fdr_alpha=0.999, threshold_method="std",
                             threshold_multiplier=0)
    out = perform_regression(settings)

    cut = _measured_cut(screen, multiplier=0, settings=settings)
    results = out["results"]
    # CHANGED BY INSTRUCTION 132: `_measured_cut` is built from the control
    # GUIDES, so it is the GUIDE fit's cut. The gene fit measures its own on
    # the control gene, and mixing the two would test one fit's rows against
    # the other fit's number.
    guides = results.loc[results["level"] == "grna"]
    tested = ~guides["feature"].astype(str).str.contains(
        "row|column|Intercept", case=False, regex=True)
    coefficients = guides.loc[tested, "coefficient"]

    # THE FIXTURE HAS TO BE ABLE TO FAIL THIS, and the bug being caught says
    # exactly how. `t` is `|median| + k x spread` and so is never negative,
    # so `coefficient >= t OR coefficient <= t` is EVERY row: the one-sided
    # writing selects the whole screen and removes nothing. Two things
    # therefore have to hold, and the first is what the bug broke:
    #
    #   * something is BELOW the cut -- otherwise the rule is selecting
    #     everything and passes any assertion about the hits;
    #   * the hits carry BOTH SIGNS -- otherwise a rule written one-sided in
    #     the other direction would satisfy them.
    #
    # The stronger form -- both signs on both SIDES -- was demanded here
    # until 2026-08-19 and is a property of the fit, not of the cut: on this
    # 12-guide fixture only one coefficient now falls below the cut, and it
    # happens to be negative. Asserting it made the test fail for the fit
    # having changed rather than for the rule being wrong.
    hits = coefficients[coefficients.abs() >= cut]
    _seen.append(bool(
        len(coefficients[coefficients.abs() < cut])))
    _seen.append(bool(
        (hits > 0).any()))
    _seen.append(bool(
        (hits < 0).any()))
    # THE LARGEST EFFECT IN THE SCREEN IS A HIT, whichever way it points.
    # Its SIGN was asserted here ("the strongest effect of all is NEGATIVE")
    # until 2026-08-19; that was a fact about one fit on one fixture, and it
    # flipped to +0.41 when this test started running the parametric path it
    # is about. What the cut has to guarantee is that the biggest effect
    # survives it -- a rule that dropped the strongest result in the screen
    # would be wrong in either direction, and the both-signs check above
    # already stops a one-sided rule from passing.
    strongest = coefficients.loc[coefficients.abs().idxmax()]
    _seen.append(bool(
        abs(strongest) >= cut))

    sig = pd.read_csv(os.path.join(results_dir(screen["count"], settings=settings), "results_significant.csv"))
    _seen.append(bool(
        (sig.loc[sig["level"] == "grna", "coefficient"].abs()
            >= cut).all()))
    # EVERY hit clears ITS OWN fit's cut, and the cut it cleared is recorded
    # on the row -- a hit list that mixes two families with two cuts is
    # unreadable unless each row says which number it passed.
    own = sig["effect_size_threshold"].fillna(0.0).to_numpy(dtype=float)
    _seen.append(bool(
        (sig["coefficient"].abs().to_numpy(dtype=float) >= own).all()))
    _seen.append(bool(
        sig.loc[sig["level"] == "grna",
                   "effect_size_threshold"].unique().tolist() == [cut]))
    guide_hits = sig.loc[sig["level"] == "grna"]
    _seen.append(bool(
        set(guide_hits["feature"]) == set(
        guides.loc[tested & (guides["coefficient"].abs() >= cut), "feature"])))
    # And it really cut: every tested coefficient passed correction at
    # alpha=0.999, so the old union kept all of them.
    _seen.append(bool(
        len(guide_hits) < int(tested.sum())))
    _seen.append(bool(
        len(sig) == len(out["significant"])))
    return _seen


def case_the_run_says_how_many_the_effect_size_cut_removed(screen, stubs,
                                                            capsys):
    """A hit list shorter than the corrected list must account for itself.

    Every other number this module drops is announced, and a coefficient that
    passed multiple-testing correction and is then removed on effect size is
    the one a reader is most likely to go looking for.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, multiple_testing_method="none",
                             fdr_alpha=0.999, threshold_method="std",
                             threshold_multiplier=0)
    out = perform_regression(settings)
    printed = capsys.readouterr().out

    # CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17): each fit is its own
    # family with its own cut, so the sentence is per fit and the counts are
    # that fit's. Measured on the guide fit, which is the one `_measured_cut`
    # describes -- it is built from the control GUIDES.
    results = out["results"]
    guides = results.loc[results["level"] == "grna"]
    tested = ~guides["feature"].astype(str).str.contains(
        "row|column|Intercept", case=False, regex=True)
    called = int(tested.sum())          # every one of them, at alpha=0.999
    hits = out["significant"]
    removed = called - int((hits["level"] == "grna").sum())
    cut = _measured_cut(screen, multiplier=0, settings=settings)
    _seen.append(bool(
        removed > 0))
    _seen.append(bool(
        f"Effect-size cut (grna) removed {removed} of {called}" in printed))
    _seen.append(bool(
        f"{cut:.3g}" in printed))
    return _seen


def case_asking_for_no_effect_size_cut_keeps_the_hits_it_had(screen, stubs):
    """threshold_method='none' means no cut -- not no hits.

    ``coefficient_threshold`` answers None when no cut can be made, and that
    None reached ``significant['coefficient'] >= reg_threshold``. pandas 2.x
    evaluates a comparison against None as all-False instead of raising, so
    BOTH one-sided masks were empty and the run wrote an empty
    results_significant.csv: measured on this fixture, 16 of 16 corrected hits
    lost, with nothing in the log to say a hit list had been thrown away.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, multiple_testing_method="none",
                             fdr_alpha=0.999, threshold_method="none")
    out = perform_regression(settings)

    results = out["results"]
    tested = ~results["feature"].astype(str).str.contains(
        "row|column|Intercept", case=False, regex=True)
    _seen.append(keep(int(tested.sum())))
    sig = pd.read_csv(os.path.join(results_dir(screen["count"], settings=settings), "results_significant.csv"))
    _seen.append(bool(
        len(sig) == len(out["significant"]) == 16))
    return _seen


def case_one_control_guide_is_too_few_to_measure_a_spread_and_costs_no_hits(
        screen, stubs, capsys):
    """A screen with a single named control loses its whole hit list.

    The same None as above, reached the way a real screen reaches it rather
    than by asking for it: one control coefficient is not a spread, so
    ``coefficient_threshold`` refuses to invent one. Refusing is right;
    turning the refusal into an empty hit list is not.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, controls=[CONTROLS[0]],
                             multiple_testing_method="none", fdr_alpha=0.999)
    out = perform_regression(settings)
    printed = capsys.readouterr().out

    _seen.append(('is not enough to measure a spread' if 'is not enough to measure a spread' in (printed) else ("missing", 'is not enough to measure a spread')))
    _seen.append(keep(len(out["significant"])))
    return _seen


def case_no_cut_still_hands_the_volcano_a_number(screen, stubs, monkeypatch):
    """The toxo volcano does abs(threshold), and None has no abs().

    Reached after the fit, every results CSV and every QC panel were written,
    so the run was reported as a failure while its coefficients were already
    on disk. ``custom_volcano_plot`` is deliberately NOT stubbed here: the
    TypeError came from inside it, and a stand-in would test the stand-in.
    """
    _seen = []
    import spacr.toxo as T
    from spacr.ml import perform_regression

    # The two OPTIONAL reports built from the returned gene list are stubbed;
    # they are not on the path this test is about and they read curated
    # Toxoplasma tables the fixture only imitates.
    monkeypatch.setattr(T, "plot_gene_phenotypes", lambda *a, **k: None)
    monkeypatch.setattr(T, "plot_gene_heatmaps", lambda *a, **k: None)

    settings = base_settings(screen, toxo=True, threshold_method="none")
    out = perform_regression(settings)

    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "results.csv"))))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


# ---------------------------------------------------------------------------
# regression back-ends
# ---------------------------------------------------------------------------

def case_ols_results_tables_carry_grna_and_gene_annotations(screen, stubs):
    """results/gene/grna CSVs split the patsy feature names back apart."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen)
    out = perform_regression(settings)

    res = results_dir(screen["count"], settings=settings)
    ids = {"gene": str, "grna": str}
    results = pd.read_csv(os.path.join(res, "results.csv"), dtype=ids)
    gene = pd.read_csv(os.path.join(res, "results_gene.csv"), dtype=ids)
    grna = pd.read_csv(os.path.join(res, "results_grna.csv"), dtype=ids)

    _seen.append(bool(
        {"feature", "coefficient", "p_value", "grna", "gene",
            "n_grna", "n_gene"} <= set(results.columns)))
    # every gene-level row is a gene_fraction term with a non-null count
    _seen.append(bool(
        gene["feature"].str.startswith("gene_fraction:gene[").all()))
    _seen.append(bool(
        gene["n_gene"].notna().all()))
    _seen.append(bool(
        set(gene["gene"]) == set(GENES)))
    _seen.append(bool(
        grna["feature"].str.startswith("fraction:grna[").all()))
    _seen.append(bool(
        grna["n_grna"].notna().all()))
    _seen.append(bool(
        len(grna) == len(GENES) * N_GRNA_PER_GENE))
    # CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17): level='both' fits
    # TWO models, so results.csv carries TWO intercepts -- one per fit -- not
    # one. It used to be a single design containing both levels, which is the
    # collinear model that instruction removed (`gene_fraction` is the sum of
    # the gene's guide fractions, so the gene block was an exact linear
    # combination of the guide block).
    _seen.append(bool(
        len(results) == len(gene) + len(grna) + 2))
    _seen.append(keep((results["feature"] == "Intercept").sum()))
    _seen.append(keep(set(results["level"])))
    _seen.append(keep(set(gene["level"])))
    _seen.append(keep(set(grna["level"])))
    return _seen


def case_verbose_ols_writes_the_model_summary(screen, stubs):
    """verbose + ols dumps the statsmodels summary next to the results.

    RUN PARAMETRIC, EXPLICITLY. The suite's default inference is
    `guide_permutation`, which fits no model at all -- so this test was
    asking a permutation run for a statsmodels summary and reading its
    perfectly correct "the permutation test fits no model" as a failure.
    A test of the statsmodels path has to ask for the statsmodels path.
    """
    _seen = []
    from spacr.ml import perform_regression

    # `inference='parametric'` AS WELL AS the mode: with `inference='auto'`
    # (the default) `resolve_auto_inference` re-picks the permutation test
    # from the design and quietly overrides `analysis_mode` -- correctly, for
    # a screen with more guides than wells, which this fixture is.
    settings = base_settings(screen, verbose=True, inference="parametric",
                             analysis_mode="regression",
                             regression_type="ols")
    perform_regression(settings)

    # THE NAME MOVED AND THIS TEST DID NOT. `mode_summary.csv` was wrong
    # twice over -- "mode" is a typo for "model" and the content is the
    # statsmodels TEXT summary, never CSV -- so runs write
    # `model_summary.txt` now. Asked for by name from `spacr.ml` rather than
    # spelled again here, so the next rename cannot leave this behind.
    from spacr.ml import SUMMARY_FILENAME, SUMMARY_FILENAMES

    res = results_dir(screen["count"], settings=settings)
    summary = os.path.join(res, SUMMARY_FILENAME)
    _seen.append(bool(
        os.path.isfile(summary)))
    with open(summary) as fh:
        text = fh.read()
    _seen.append(('OLS Regression Results' if 'OLS Regression Results' in (text) else ("missing", 'OLS Regression Results')))

    # And the old names are still READ, so a run finished last month keeps
    # its summary.
    _seen.append(('mode_summary.csv' if 'mode_summary.csv' in (SUMMARY_FILENAMES) else ("missing", 'mode_summary.csv')))
    return _seen


def case_ridge_regression_backend(screen, stubs):
    """regression_type='ridge' produces coefficients for every design column."""
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, regression_type="ridge", alpha=1.0)
    out = perform_regression(settings)

    res = results_dir(screen["count"], "ridge")
    _seen.append(bool(
        os.path.isfile(os.path.join(res, "results.csv"))))
    _seen.append(bool(
        out["results"]["coefficient"].notna().all()))
    _seen.append(bool(
        out["results"]["feature"].str.contains("grna\\[").sum()
            == len(GENES) * N_GRNA_PER_GENE))
    return _seen


def case_regression_type_none_uses_the_auto_results_folder(screen, stubs, capsys):
    """regression_type=None auto-detects the model and writes under 'auto'."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, regression_type=None)
    out = perform_regression(settings)

    printed = capsys.readouterr().out
    # a per-well mean score strictly inside (0, 1) -> beta regression
    _seen.append(('Using regression type: beta' if 'Using regression type: beta' in (printed) else ("missing", 'Using regression type: beta')))
    res = results_dir(screen["count"], settings=settings)
    _seen.append(bool(
        os.path.isfile(os.path.join(res, "results.csv"))))
    _seen.append(bool(
        {"std_err", "wald_stat"} <= set(out["results"].columns)))
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_regression_type_quantile_fits_the_requested_quantile(screen, stubs):
    """'quantile' fits end to end instead of dying at the last statement.

    It used to pass the entry-point whitelist, get its own agg_type handling
    in get_perform_regression_default_settings and its own volcano-filename
    rule, and then raise "Unsupported regression type quantile" from
    regression_model - after both input CSVs had been read, the QC plots drawn
    and regression_data.csv written.
    """
    _seen = []
    from spacr.ml import perform_regression

    # PARAMETRIC, because that is the path this test is about. Under the
    # default `inference='nonparametric'` the permutation test IS the
    # analysis and regression_type is never read -- so there is no quantile
    # fit to check, and agg_type deliberately stays 'mean' there.
    settings = base_settings(screen, regression_type="quantile", quantile=0.75,
                             inference="parametric",
                             analysis_mode="regression")
    out = perform_regression(settings)

    # agg_type is forced to None for quantile, so the fit is on objects.
    _seen.append(bool(
        settings["agg_type"] is None))
    res = results_dir(screen["count"], settings=settings)
    _seen.append(bool(
        os.path.isfile(os.path.join(res, "results.csv"))))
    _seen.append(bool(
        out["results"]["coefficient"].notna().all()))
    _seen.append(bool(
        out["results"]["p_value"].notna().all()))
    _seen.append(bool(
        out["results"]["feature"].str.contains("grna\\[").sum()
            == len(GENES) * N_GRNA_PER_GENE))
    return _seen


def case_quantile_regression_refuses_the_old_alpha_spelling(screen, stubs):
    """alpha used to double as the quantile; the overload is refused, not ignored.

    A settings CSV written before the split says alpha=0.75 and means "the
    75th percentile". Silently dropping it would fit the median and label the
    output folder as a quantile run.
    """
    _seen = []
    from spacr.ml import perform_regression

    with pytest.raises(ValueError, match=r"does not use alpha") as _exc1:
        perform_regression(base_settings(screen, regression_type="quantile",
                                         alpha=0.75))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_lasso_uses_bootstrap_selection_frequencies(screen, stubs):
    """Lasso hits are ranked by bootstrap selection frequency, not p-values."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, regression_type="lasso", alpha=1e-7,
                             lasso_n_boot=3, lasso_selection_threshold=0.5)
    out = perform_regression(settings)

    results = out["results"]
    _seen.append(bool(
        {"selection_frequency", "mean_coefficient"} <= set(results.columns)))
    freq = results["selection_frequency"].dropna()
    _seen.append(bool(
        len(freq) > 0))
    _seen.append(bool(
        ((freq >= 0) & (freq <= 1)).all()))
    sig = out["significant"]
    _seen.append(bool(
        (sig["coefficient"] != 0).all()))
    _seen.append(bool(
        (sig["selection_frequency"] >= 0.5).all()))
    # SORTED WITHIN EACH LEVEL, not across the two of them. Since the guide
    # and gene fits became separate families (128/132) the hit list is their
    # concatenation: each part is ordered by |coefficient| descending, and
    # the join between them is a step back up. Asserting one global ordering
    # tested that the two fits were one fit.
    for level, block in sig.groupby("level", sort=False):
        assert list(block["coefficient"].abs()) == sorted(
            block["coefficient"].abs(), reverse=True), level
    return _seen


def case_lasso_with_auto_alpha_refuses_the_empty_model_it_cross_validated_to(
        screen, stubs):
    """alpha='auto' switches the fit to LassoCV -- and says what CV found.

    THIS FIXTURE IS A NULL SCREEN FOR CROSS-VALIDATION. 12 guides over a
    handful of synthetic wells: no gRNA predicts held-out wells better than
    their mean does, so LassoCV picks a penalty that zeroes everything. That
    is not a defect in the fixture and not one in the fit.

    What matters is what spaCR does with it. Writing the result out would
    report "0 significant gRNAs", which a reader cannot tell apart from a
    screen that was tested and found nothing -- so the fit is REFUSED, by
    name, with the two things a user can act on: check the dependent
    variable, or fit an unpenalised model to see what the penalty is
    shrinking away.

    The refusal is also the proof that CV ran: the message names the
    cross-validated empty model, which nothing else could have discovered.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, regression_type="lasso",
                                   alpha="auto", lasso_n_boot=2,
                                   lasso_selection_threshold=0.0)

    with pytest.raises(ValueError) as excinfo:
        perform_regression(settings)
    _seen.append(('ValueError' if isinstance(excinfo.value, ValueError) else type(excinfo.value).__name__))

    message = str(excinfo.value)
    _seen.append(('cross-validated' if 'cross-validated' in (message) else ("missing", 'cross-validated')))
    _seen.append(('null screen' if 'null screen' in (message) else ("missing", 'null screen')))
    _seen.append(('0 significant gRNAs' if '0 significant gRNAs' in (message) else ("missing", '0 significant gRNAs')))
    _seen.append(("'ols'" if "'ols'" in (message) else ("missing", "'ols'")))
    return _seen


def case_lasso_bootstrap_raises_when_every_resample_fails(screen, stubs,
                                                          monkeypatch):
    """If no bootstrap resample can be designed the helper refuses to guess."""
    _seen = []
    import spacr.ml as ML
    from spacr.ml import perform_regression

    # CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17): the RESAMPLES have
    # to be the calls that fail, and a run builds more designs before them
    # than it used to. level='both' fits the guide model and the gene model
    # separately, so there are two main designs, and then
    # `bootstrap_selection_frequencies` builds its own reference design before
    # resampling anything. Failing from call 3 would kill that reference
    # design instead and the run would die of a different error -- the test
    # would still be red, but for the wrong reason, which is the failure mode
    # a stale count produces.
    #
    # ALLOWED = the two level fits plus the bootstrap's own reference design;
    # every call after that is a resample and every one of them fails.
    ALLOWED = 3
    real = ML.dmatrices
    state = {"n": 0}

    def flaky(formula, data=None, return_type=None, **kwargs):
        state["n"] += 1
        if state["n"] > ALLOWED:
            raise ValueError("factor level vanished from resample")
        return real(formula, data=data, return_type=return_type, **kwargs)

    settings = parametric_settings(screen, regression_type="lasso", alpha=1e-7,
                             lasso_n_boot=4)
    monkeypatch.setattr(ML, "dmatrices", flaky)
    with pytest.raises(RuntimeError, match="All bootstrap resamples failed") as _exc1:
        perform_regression(settings)
    _seen.append(('RuntimeError' if isinstance(_exc1.value, RuntimeError) else type(_exc1.value).__name__))
    _seen.append(bool(
        state["n"] > ALLOWED))
    return _seen


# ---------------------------------------------------------------------------
# metadata merge + toxo reporting block
# ---------------------------------------------------------------------------

def case_metadata_file_string_is_wrapped_and_merged(screen, stubs):
    """A single metadata_files string is wrapped and merged into every table."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, metadata_files=screen["meta"])
    perform_regression(settings)

    _seen.append(bool(
        settings["metadata_files"] == [screen["meta"]]))
    name = os.path.splitext(os.path.basename(screen["meta"]))[0]
    for stem in ("results", "results_gene", "results_grna",
                 "results_significant"):
        merged = os.path.join(results_dir(screen["count"], settings=settings), f"{stem}{name}.csv")
        assert os.path.isfile(merged), merged
    merged_df = pd.read_csv(os.path.join(results_dir(screen["count"], settings=settings), f"results{name}.csv"),
                            dtype={"gene": str})
    _seen.append(('Gene Name' if 'Gene Name' in (merged_df.columns) else ("missing", 'Gene Name')))
    _seen.append(keep(merged_df.loc[merged_df["gene"] == "239740", "Gene Name"].iloc[0]))
    return _seen


@pytest.mark.parametrize("volcano", ["all", "gene", "grna"])
def test_toxo_block_renders_the_requested_volcano(screen, toxo_stubs, volcano):
    """Each volcano mode feeds a different merged table to the toxo plot."""
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, toxo=True, volcano=volcano)
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


# `test_toxo_block_skips_unknown_volcano_mode` lived here. It asserted that an
# unrecognised `volcano` setting skipped the plot -- and the setting was
# removed on 2026-08-17 as redundant, because the interactive volcano filters
# genes/guides by right-click on the same fit. There is no unrecognised mode
# to skip any more: the toxo block always draws the gene table.
# tests/test_the_volcano_setting_is_retired.py holds what replaced it.
def case_toxo_block_with_empty_gene_list(screen, toxo_stubs, capsys):
    """An empty gene list from the volcano plot skips the phenotype figures."""
    _seen = []
    from spacr.ml import perform_regression

    toxo_stubs["gene_list"] = []
    settings = parametric_settings(screen, toxo=True)
    perform_regression(settings)

    printed = capsys.readouterr().out
    _seen.append(('No gene_list produced' if 'No gene_list produced' in (printed) else ("missing", 'No gene_list produced')))
    # An empty gene list draws nothing; there is no second unguarded copy.
    _seen.append(keep(toxo_stubs["phenotypes"]))
    return _seen


# ---------------------------------------------------------------------------
# filter_column shapes
# ---------------------------------------------------------------------------

def case_filter_column_may_be_a_list(screen, stubs):
    """process_reads accepts a list of filter columns; so should the caller.

    The list used to reach clean_controls' `column in df.columns` membership
    test and raise "unhashable type: 'list'".
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_column=["columnID"])
    out = perform_regression(settings)
    _seen.append(bool(
        len(out["results"]) > 0))
    return _seen


def case_filter_column_may_be_none(screen, stubs, capsys):
    """filter_column=None means 'drop no control wells'.

    The local `filter_column` was only bound in the `isinstance(..., str)`
    branch, so None fell through and the process_reads call below raised
    UnboundLocalError before any filtering decision was made.
    """
    _seen = []
    from spacr.ml import perform_regression

    settings = base_settings(screen, filter_column=None)
    out = perform_regression(settings)

    _seen.append(bool(
        len(out["results"]) > 0))
    # clean_controls announces every value it drops; None must drop nothing.
    _seen.append((("absent", 'Removed data from') if 'Removed data from' not in (capsys.readouterr().out) else ("present", 'Removed data from')))
    return _seen


def case_qc_tables_are_written_with_the_real_plot_helper(screen, heavy_stubs):
    """The gRNA-coverage QC tables must survive a real plot_data_from_csv."""
    _seen = []
    from spacr.ml import perform_regression

    perform_regression(base_settings(screen))
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "grna_well.csv"))))
    _seen.append(bool(
        os.path.isfile(os.path.join(screen["res"], "well_grna.csv"))))
    return _seen


def case_toxo_volcano_without_controls(screen, toxo_stubs):
    """A screen with no control gRNAs should still be able to plot a volcano."""
    _seen = []
    from spacr.ml import perform_regression

    settings = parametric_settings(screen, toxo=True, controls=None)
    perform_regression(settings)
    _seen.append(keep(len(toxo_stubs["volcano"])))
    return _seen


# ---------------------------------------------------------------------------
# manifest-driven publication packages (instruction 343)
# ---------------------------------------------------------------------------

def _finished_panel_run(tmp_path, analysis_mode):
    folder = tmp_path / analysis_mode
    folder.mkdir()
    paths = {}
    for level in ("grna", "gene"):
        path = folder / f"results_{level}.csv"
        pd.DataFrame(
            {level: [f"item_{level}"], "coefficient": [0.25]}
        ).to_csv(path, index=False)
        paths[f"results_{level}"] = str(path)
    outcome = {"res_folder": str(folder), "analysis_mode": analysis_mode}
    if analysis_mode == "guide_permutation":
        outcome["paths"] = paths
    return outcome, paths


def _panel_settings(dependent_variable="pred", **overrides):
    """Return the smallest settings mapping that requests panel packaging."""
    settings = {
        "regression_panel_manifest": {"figure_id": "Figure_7", "panels": [{}]},
        "dependent_variable": dependent_variable,
    }
    settings.update(overrides)
    return settings


def case_publication_panels_require_a_mapping_outcome_and_results_folder(
    tmp_path,
):
    """Validation names both malformed return shapes before reading a table."""
    _seen = []
    import spacr.ml as ML

    settings = _panel_settings()
    with pytest.raises(TypeError, match="mapping outcome") as _exc1:
        ML._write_regression_panel_packages(object(), settings)
    _seen.append(('TypeError' if isinstance(_exc1.value, TypeError) else type(_exc1.value).__name__))
    _seen.append(keep(settings["_regression_stage"]))

    with pytest.raises(ValueError, match="results folder") as _exc2:
        ML._write_regression_panel_packages({}, _panel_settings())
    _seen.append(('ValueError' if isinstance(_exc2.value, ValueError) else type(_exc2.value).__name__))
    return _seen


def case_publication_panels_name_every_missing_result_file(tmp_path):
    """An incomplete run is refused before a partial panel package is built."""
    _seen = []
    import spacr.ml as ML

    folder = tmp_path / "incomplete"
    folder.mkdir()
    with pytest.raises(FileNotFoundError) as raised:
        ML._write_regression_panel_packages(
            {"res_folder": str(folder), "paths": None},
            _panel_settings(),
        )
    _seen.append(('FileNotFoundError' if isinstance(raised.value, FileNotFoundError) else type(raised.value).__name__))

    message = str(raised.value)
    _seen.append(bool(
        str(folder / "results_grna.csv") in message))
    _seen.append(bool(
        str(folder / "results_gene.csv") in message))
    return _seen


@pytest.mark.parametrize("dependent_variable", [None, [], ["pred", " "]])
def test_publication_panels_require_named_phenotypes(
    tmp_path, dependent_variable,
):
    """Missing, empty and partly blank phenotype requests share one clear error."""
    import spacr.ml as ML

    outcome, _paths = _finished_panel_run(tmp_path, "regression")
    with pytest.raises(ValueError, match="one or more named"):
        ML._write_regression_panel_packages(
            outcome,
            _panel_settings(dependent_variable),
        )


def case_multi_phenotype_panels_require_an_outcome_column(tmp_path):
    """Two requested phenotypes cannot be inferred from an unlabelled table."""
    _seen = []
    import spacr.ml as ML

    outcome, _paths = _finished_panel_run(tmp_path, "regression")
    with pytest.raises(ValueError, match="has no 'outcome' column") as _exc1:
        ML._write_regression_panel_packages(
            outcome,
            _panel_settings(["first", "second"]),
        )
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_multi_phenotype_panels_require_rows_for_every_request(tmp_path):
    """A named but absent phenotype is an error, not an empty publication panel."""
    _seen = []
    import spacr.ml as ML

    outcome, paths = _finished_panel_run(tmp_path, "guide_permutation")
    for path in paths.values():
        pd.DataFrame({
            "outcome": ["present"],
            "coefficient": [0.25],
        }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no rows for phenotype 'missing'") as _exc1:
        ML._write_regression_panel_packages(
            outcome,
            _panel_settings(["present", "missing"]),
        )
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_multi_phenotype_panels_use_the_saved_folder_and_default_fdr(
    tmp_path, monkeypatch,
):
    """The settings-folder fallback builds one filtered artifact per level/value."""
    _seen = []
    import spacr.ml as ML
    import spacr.regression_panels as RP

    folder = tmp_path / "saved-result-folder"
    folder.mkdir()
    for level in ("grna", "gene"):
        pd.DataFrame({
            "outcome": ["first", "second"],
            level: [f"first_{level}", f"second_{level}"],
            "coefficient": [0.25, -0.5],
        }).to_csv(folder / f"results_{level}.csv", index=False)

    captured = {}

    def fake_builder(manifest, artifacts, destination):
        captured.update(artifacts)
        assert manifest["figure_id"] == "Figure_7"
        assert destination == str(folder / "publication_panels")
        return "built"

    monkeypatch.setattr(RP, "build_manifest_packages", fake_builder)
    settings = _panel_settings(
        [" first ", "second"],
        _regression_folder=str(folder),
    )

    _seen.append(keep(ML._write_regression_panel_packages({}, settings)))
    _seen.append(bool(
        settings["_regression_folder"] == str(folder)))
    _seen.append(keep(set(captured)))
    for key, artifact in captured.items():
        assert len(artifact["data"]) == 1, key
        assert artifact["data"].iloc[0]["outcome"] == artifact["phenotype"]
        assert artifact["fdr_alpha"] == pytest.approx(0.05)
    return _seen


@pytest.mark.parametrize("analysis_mode", ["regression", "guide_permutation"])
def test_public_regression_builds_panels_after_each_result_path_exists(
    tmp_path, monkeypatch, analysis_mode
):
    """Both OLS and Freedman-Lane returns reach one post-success writer."""
    import spacr.ml as ML
    import spacr.regression_panels as RP

    events = []
    returned = {"figure_pdf": tmp_path / "publication" / "Figure_7.pdf"}

    def fake_perform(_settings):
        events.append("performing")
        outcome, paths = _finished_panel_run(tmp_path, analysis_mode)
        assert all(os.path.isfile(path) for path in paths.values())
        events.append("results-written")
        return outcome

    def fake_builder(manifest, artifacts, destination):
        events.append("writer")
        assert manifest == {"figure_id": "Figure_7", "panels": [{}]}
        assert set(artifacts) == {"pred_grna", "pred_gene"}
        assert os.fspath(destination) == os.path.join(
            os.fspath(tmp_path / analysis_mode), "publication_panels"
        )
        for level in ("grna", "gene"):
            artifact = artifacts[f"pred_{level}"]
            assert artifact["level"] == level
            assert artifact["phenotype"] == "pred"
            assert artifact["dependent_variable"] == "pred"
            assert artifact["fdr_alpha"] == 0.025
            assert artifact["res_folder"] == os.fspath(tmp_path / analysis_mode)
            assert artifact["run_artifact_path"] == os.path.join(
                os.fspath(tmp_path / analysis_mode), f"results_{level}.csv"
            )
            assert os.path.isfile(artifact["run_artifact_path"])
            assert len(artifact["data"]) == 1
        return returned

    monkeypatch.setattr(ML, "_perform_regression", fake_perform)
    monkeypatch.setattr(ML, "_write_fit_resources", lambda *_args: "")
    monkeypatch.setattr(RP, "build_manifest_packages", fake_builder)
    settings = {
        "regression_panel_manifest": {"figure_id": "Figure_7", "panels": [{}]},
        "dependent_variable": "pred",
        "fdr_alpha": 0.025,
    }

    outcome = ML.perform_regression(settings)

    assert events == ["performing", "results-written", "writer"]
    assert outcome["publication_panels"] is returned


def case_public_regression_without_a_manifest_preserves_the_outcome(
    tmp_path, monkeypatch,
):
    """The shipped None default neither imports nor calls the panel writer."""
    _seen = []
    import spacr.ml as ML
    import spacr.regression_panels as RP

    expected = {"results": "unchanged", "res_folder": str(tmp_path / "untouched")}
    monkeypatch.setattr(ML, "_perform_regression", lambda _settings: expected)
    monkeypatch.setattr(ML, "_write_fit_resources", lambda *_args: "")
    monkeypatch.setattr(
        RP,
        "build_manifest_packages",
        lambda *_args, **_kwargs: pytest.fail("no manifest must not call writer"),
    )

    outcome = ML.perform_regression(
        {"regression_panel_manifest": None, "dependent_variable": "pred"}
    )

    _seen.append(bool(
        outcome is expected))
    _seen.append(bool(
        outcome == {
        "results": "unchanged", "res_folder": str(tmp_path / "untouched")
    }))
    return _seen


def case_publication_writer_failure_uses_the_regression_failure_report(
    tmp_path, monkeypatch,
):
    """An explicit package is part of the run and uses its existing reporter."""
    _seen = []
    import spacr.ml as ML
    import spacr.regression_failure as RF
    import spacr.regression_panels as RP

    outcome, _paths = _finished_panel_run(tmp_path, "regression")
    failure = RuntimeError("panel composition failed")
    reports = []
    monkeypatch.setattr(ML, "_perform_regression", lambda _settings: outcome)
    monkeypatch.setattr(
        RP,
        "build_manifest_packages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(RF, "describe_failure", lambda *_args, **_kwargs: "report")

    def fake_report(folder, error, **details):
        reports.append((folder, error, details))
        return ""

    monkeypatch.setattr(RF, "write_failure_report", fake_report)
    settings = {
        "regression_panel_manifest": {"figure_id": "Figure_7", "panels": [{}]},
        "dependent_variable": "pred",
        "fdr_alpha": 0.05,
    }

    with pytest.raises(RuntimeError, match="panel composition failed") as raised:
        ML.perform_regression(settings)
    _seen.append(('RuntimeError' if isinstance(raised.value, RuntimeError) else type(raised.value).__name__))

    _seen.append(bool(
        raised.value is failure))
    _seen.append(keep(len(reports)))
    folder, error, details = reports[0]
    _seen.append(bool(
        folder == os.fspath(tmp_path / "regression")))
    _seen.append(bool(
        error is failure))
    _seen.append(keep(details["stage"]))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_scalar_score_and_count_paths_are_wrapped_in_lists, [True, True, True, True, True], id="scalar_score_and_count_paths_are_wrapped_in_lists"),
    pytest.param(case_paired_input_copies_plate_identity_from_its_partner, [{'plate2'}, True, True], id="paired_input_copies_plate_identity_from_its_partner"),
    pytest.param(case_missing_dependent_variable_raises, ['ValueError'], id="missing_dependent_variable_raises"),
    pytest.param(case_missing_pred_names_the_file_and_says_how_to_get_it, ['ValueError', True, 'Run Classify', '_result.csv'], id="missing_pred_names_the_file_and_says_how_to_get_it"),
    pytest.param(case_other_missing_response_names_the_file_without_the_pred_advice, ['ValueError', True, ("absent", 'Classify')], id="other_missing_response_names_the_file_without_the_pred_advice"),
    pytest.param(case_shortest_distance_dependent_variable_is_exempt_from_the_check, ['KeyError'], id="shortest_distance_dependent_variable_is_exempt_from_the_check"),
    pytest.param(case_unsupported_regression_type_raises, ['ValueError'], id="unsupported_regression_type_raises"),
    pytest.param(case_rowid_with_plate_prefix_is_split, [True, True], id="rowid_with_plate_prefix_is_split"),
    pytest.param(case_path_is_not_used_to_guess_missing_well_columns, ['ValueError'], id="path_is_not_used_to_guess_missing_well_columns"),
    pytest.param(case_declared_plate_is_kept_without_inspecting_image_paths, [("absent", 'PLATEn_ prefix'), {'plate1'}, True], id="declared_plate_is_kept_without_inspecting_image_paths"),
    pytest.param(case_missing_plate_column_defaults_to_file_position, [("absent", 'plateID'), ("absent", 'plateID'), {'plate1'}, True], id="missing_plate_column_defaults_to_file_position"),
    pytest.param(case_non_list_filter_value_disables_control_well_removal, [True, True], id="non_list_filter_value_disables_control_well_removal"),
    pytest.param(case_min_cells_per_well_none_is_filled_from_the_simulation, [3, True], id="min_cells_per_well_none_is_filled_from_the_simulation"),
    pytest.param(case_fraction_threshold_none_is_filled_from_graph_sequencing_stats, [0.004, True], id="fraction_threshold_none_is_filled_from_graph_sequencing_stats"),
    pytest.param(case_missing_gene_column_raises_keyerror, ['KeyError'], id="missing_gene_column_raises_keyerror"),
    pytest.param(case_qc_block_writes_the_three_well_level_tables, [True, {'grna', 'plateID', 'grna_well_count', 'gene_well_count'}, True, True, True, True, True, ['cell_count', 'wells_per_gene', 'gene_per_well'], True, True, True, 'pred', True, True], id="qc_block_writes_the_three_well_level_tables"),
    pytest.param(case_batch_correction_runs_before_regression_and_writes_report, ['center', 'plateID', True, ['Only 1 batch was present; correction was a no-op.']], id="batch_correction_runs_before_regression_and_writes_report"),
    pytest.param(case_outlier_detection_drops_sparsely_covered_grnas, [("absent", '111111_3'), ("absent", '111111_3'), True, True, 'Saved grna per well data to', 'Saved well per grna data to', ("absent", 'A value is trying to be set on a copy of a slice')], id="outlier_detection_drops_sparsely_covered_grnas"),
    pytest.param(case_qc_plot_failure_does_not_cost_the_qc_tables, ['qc plot exploded', "Skipping QC plot 'cell_count'", True, True, True, True], id="qc_plot_failure_does_not_cost_the_qc_tables"),
    pytest.param(case_qc_block_failure_is_swallowed_and_the_run_continues, ['qc table exploded', True, True, True], id="qc_block_failure_is_swallowed_and_the_run_continues"),
    pytest.param(case_unsupported_threshold_method_raises, ['mad', 'ValueError'], id="unsupported_threshold_method_raises"),
    pytest.param(case_controls_none_skips_the_threshold_block, [True, True, True], id="controls_none_skips_the_threshold_block"),
    pytest.param(case_the_effect_size_cut_is_symmetric_about_zero, [True, True, True, True, True, True, True, True, True, True], id="the_effect_size_cut_is_symmetric_about_zero"),
    pytest.param(case_the_run_says_how_many_the_effect_size_cut_removed, [True, True, True], id="the_run_says_how_many_the_effect_size_cut_removed"),
    pytest.param(case_asking_for_no_effect_size_cut_keeps_the_hits_it_had, [16, True], id="asking_for_no_effect_size_cut_keeps_the_hits_it_had"),
    pytest.param(case_one_control_guide_is_too_few_to_measure_a_spread_and_costs_no_hits, ['is not enough to measure a spread', 16], id="one_control_guide_is_too_few_to_measure_a_spread_and_costs_no_hits"),
    pytest.param(case_no_cut_still_hands_the_volcano_a_number, [True, True], id="no_cut_still_hands_the_volcano_a_number"),
    pytest.param(case_ols_results_tables_carry_grna_and_gene_annotations, [True, True, True, True, True, True, True, True, 2, {'grna', 'gene'}, {'gene'}, {'grna'}], id="ols_results_tables_carry_grna_and_gene_annotations"),
    pytest.param(case_verbose_ols_writes_the_model_summary, [True, 'OLS Regression Results', 'mode_summary.csv'], id="verbose_ols_writes_the_model_summary"),
    pytest.param(case_ridge_regression_backend, [True, True, True], id="ridge_regression_backend"),
    pytest.param(case_regression_type_none_uses_the_auto_results_folder, ['Using regression type: beta', True, True, True], id="regression_type_none_uses_the_auto_results_folder"),
    pytest.param(case_regression_type_quantile_fits_the_requested_quantile, [True, True, True, True, True], id="regression_type_quantile_fits_the_requested_quantile"),
    pytest.param(case_quantile_regression_refuses_the_old_alpha_spelling, ['ValueError'], id="quantile_regression_refuses_the_old_alpha_spelling"),
    pytest.param(case_lasso_uses_bootstrap_selection_frequencies, [True, True, True, True, True], id="lasso_uses_bootstrap_selection_frequencies"),
    pytest.param(case_lasso_with_auto_alpha_refuses_the_empty_model_it_cross_validated_to, ['ValueError', 'cross-validated', 'null screen', '0 significant gRNAs', "'ols'"], id="lasso_with_auto_alpha_refuses_the_empty_model_it_cross_validated_to"),
    pytest.param(case_lasso_bootstrap_raises_when_every_resample_fails, ['RuntimeError', True], id="lasso_bootstrap_raises_when_every_resample_fails"),
    pytest.param(case_toxo_block_with_empty_gene_list, ['No gene_list produced', []], id="toxo_block_with_empty_gene_list"),
    pytest.param(case_filter_column_may_be_a_list, [True], id="filter_column_may_be_a_list"),
    pytest.param(case_filter_column_may_be_none, [True, ("absent", 'Removed data from')], id="filter_column_may_be_none"),
    pytest.param(case_qc_tables_are_written_with_the_real_plot_helper, [True, True], id="qc_tables_are_written_with_the_real_plot_helper"),
    pytest.param(case_toxo_volcano_without_controls, [1], id="toxo_volcano_without_controls"),
    pytest.param(case_public_regression_without_a_manifest_preserves_the_outcome, [True, True], id="public_regression_without_a_manifest_preserves_the_outcome"),
    pytest.param(case_publication_writer_failure_uses_the_regression_failure_report, ['RuntimeError', True, 1, True, True, 'writing publication panel packages'], id="publication_writer_failure_uses_the_regression_failure_report"),
    pytest.param(case_publication_panels_require_a_mapping_outcome_and_results_folder, ['TypeError', 'writing publication panel packages', 'ValueError'], id="publication_panels_require_a_mapping_outcome_and_results_folder"),
    pytest.param(case_publication_panels_name_every_missing_result_file, ['FileNotFoundError', True, True], id="publication_panels_name_every_missing_result_file"),
    pytest.param(case_multi_phenotype_panels_require_an_outcome_column, ['ValueError'], id="multi_phenotype_panels_require_an_outcome_column"),
    pytest.param(case_multi_phenotype_panels_require_rows_for_every_request, ['ValueError'], id="multi_phenotype_panels_require_rows_for_every_request"),
    pytest.param(case_multi_phenotype_panels_use_the_saved_folder_and_default_fdr, ['built', True, {'first_grna', 'second_grna', 'first_gene', 'second_gene'}], id="multi_phenotype_panels_use_the_saved_folder_and_default_fdr"),
    pytest.param(case_legacy_score_list_longer_than_count_list_migrates, [True, True], id="legacy_score_list_longer_than_count_list_migrates"),
    pytest.param(case_legacy_count_list_longer_than_score_list_migrates, [True, True], id="legacy_count_list_longer_than_score_list_migrates"),
    pytest.param(case_one_multiplate_score_file_can_be_paired_to_plate_count_files, [{'plate1', 'plate2'}, True, {'plate1', 'plate2'}, True], id="one_multiplate_score_file_can_be_paired_to_plate_count_files"),
    pytest.param(case_the_correction_is_actually_applied_to_the_parametric_fit, [True, 0], id="the_correction_is_actually_applied_to_the_parametric_fit"),
    pytest.param(case_min_observations_per_hit_filters_the_significant_hits, [True, 0, True], id="min_observations_per_hit_filters_the_significant_hits"),
    pytest.param(case_metadata_file_string_is_wrapped_and_merged, [True, 'Gene Name', 'gene_239740'], id="metadata_file_string_is_wrapped_and_merged"),
])
def test_ml_perform_regression_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected



