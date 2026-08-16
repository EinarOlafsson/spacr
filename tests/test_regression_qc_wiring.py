"""The regression QC suite is produced by an ordinary run, not only by a test.

:func:`spacr.regression_qc.regression_qc_report` and its twenty-three panels
were fully implemented and fully tested but had **no production caller**: the
diagnostics existed and no run ever wrote them. These tests pin the hook in
:func:`spacr.ml.regression` and they assert on FILES ON DISK -- not on "a mock
was called" -- because the defect being guarded against is precisely that the
code ran and nothing landed anywhere.

Variance homogeneity (the scale-location / Brown-Forsythe panel) is asserted by
name: the maintainer asked for it explicitly, so it gets its own test rather
than being counted among "23 panels".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr.regression_qc import QC_DIRNAME


NC = "233460"
PC = "220950"
GENES = [NC, PC, "gene3", "gene4"]


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _wells_df(seed=0, n_plates=1, n_rows=4, n_cols=6, dep="predictions"):
    """Long-format score/count table: one row per (well, gRNA)."""
    rng = np.random.default_rng(seed)
    grnas = {g: [f"{g}_a", f"{g}_b"] for g in GENES}
    records = []
    for plate_index in range(n_plates):
        plate = f"plate{plate_index + 1}"
        for row_index in range(n_rows):
            for column_index in range(n_cols):
                row_id = f"r{row_index + 1:02d}"
                column_id = f"c{column_index + 1:02d}"
                raw = rng.random(len(GENES) * 2) + 0.2
                fractions = raw / raw.sum()
                score = float(rng.normal(0.0, 1.0))
                position = 0
                for gene in GENES:
                    for grna in grnas[gene]:
                        records.append({
                            "plateID": plate, "rowID": row_id,
                            "columnID": column_id,
                            "prc": f"{plate}_{row_id}_{column_id}",
                            "gene": gene, "grna": grna,
                            "fraction": float(fractions[position]),
                            "cell_count": int(rng.integers(30, 200)),
                            dep: score,
                        })
                        position += 1
    return pd.DataFrame(records)


def _qc_dir(dst):
    return os.path.join(str(dst), QC_DIRNAME)


def _run(tmp_path, **kwargs):
    from spacr.ml import regression

    options = dict(dependent_variable="predictions", regression_type="ols",
                   nc=NC, pc=PC, dst=str(tmp_path))
    options.update(kwargs)
    frame = options.pop("df", None)
    if frame is None:
        frame = _wells_df(seed=options.pop("seed", 0))
    else:
        options.pop("seed", None)
    return regression(frame, str(tmp_path / "scores.csv"), **options)


# ---------------------------------------------------------------------------
# The files land
# ---------------------------------------------------------------------------


def test_an_ordinary_regression_run_writes_the_qc_panels_to_disk(tmp_path):
    """The whole point: run the pipeline function, find the figures afterwards."""
    _run(tmp_path)

    out = _qc_dir(tmp_path)
    assert os.path.isdir(out), (
        f"regression() finished but wrote no {QC_DIRNAME}/ folder; the QC "
        f"report is orphaned again. {os.listdir(tmp_path)}")

    written = sorted(os.listdir(out))
    panels = [name for name in written if name.endswith(".pdf")]
    # Twenty-three panels are registered; every one that this OLS fit supports
    # must be a real file with real bytes in it.
    assert len(panels) >= 15, f"only {len(panels)} panel files: {written}"
    for name in panels:
        assert os.path.getsize(os.path.join(out, name)) > 0, f"{name} is empty"

    # The combined page and the grep-able text report.
    assert "regression_qc_report.pdf" in written
    assert "regression_qc_report.txt" in written


def test_variance_homogeneity_is_among_the_saved_figures(tmp_path):
    """Named explicitly by the maintainer, so it is asserted explicitly."""
    _run(tmp_path)
    out = _qc_dir(tmp_path)

    # scale_location IS the variance-homogeneity panel: sqrt|std resid| vs
    # fitted, with Brown-Forsythe across quartiles of the fit.
    scale_location = os.path.join(out, "scale_location.pdf")
    assert os.path.isfile(scale_location), (
        f"the variance homogeneity panel was not written; "
        f"got {sorted(os.listdir(out))}")
    assert os.path.getsize(scale_location) > 0

    report = open(os.path.join(out, "regression_qc_report.txt"),
                  encoding="utf-8").read()
    assert "Scale-location" in report
    # The panel's verdict and its two statistics are the part a reviewer
    # quotes, so they must be in the text report, not only in the picture.
    assert "levene_p" in report
    assert "quartile_sd_ratio" in report
    assert "[WRITTEN]" in report


def test_the_saved_report_covers_the_rest_of_the_standard_set(tmp_path):
    """residuals-vs-fitted, QQ, leverage, Cook's distance, VIF, p-values."""
    # Two plates, so the between-plate panel has something to compare; on a
    # one-plate screen it skips itself, correctly, with that as the reason.
    _run(tmp_path, df=_wells_df(seed=7, n_plates=2, n_rows=3, n_cols=4))
    out = _qc_dir(tmp_path)
    written = set(os.listdir(out))

    for panel in ("residuals_vs_fitted", "qq_residuals", "influence",
                  "cooks_distance", "dffits", "vif", "condition_number",
                  "predictor_correlation", "p_value_histogram",
                  "observed_vs_predicted", "plate_effects", "row_effects",
                  "column_effects", "cell_count_vs_effect"):
        assert f"{panel}.pdf" in written, (
            f"{panel} was not written; got {sorted(written)}")


def test_no_panel_fails_on_the_design_the_pipeline_builds(tmp_path):
    """A FAILED panel means the hook is feeding the report something wrong."""
    _run(tmp_path)
    report = open(os.path.join(_qc_dir(tmp_path), "regression_qc_report.txt"),
                  encoding="utf-8").read()
    assert "FAILED" not in report, report


def test_wells_are_labelled_by_prc_so_metadata_reached_the_report(tmp_path):
    """The plate/row/column panels only exist if the metadata was aligned."""
    frame = _wells_df(seed=3)
    _run(tmp_path, df=frame)
    report = open(os.path.join(_qc_dir(tmp_path), "regression_qc_report.txt"),
                  encoding="utf-8").read()

    # Cook's distance names its worst well; that name comes from the metadata
    # frame regression() assembles, so finding a real prc in the report proves
    # the alignment happened rather than silently skipping.
    assert any(prc in report for prc in frame["prc"].unique()), report


# ---------------------------------------------------------------------------
# Preferences, guards and blast radius
# ---------------------------------------------------------------------------


def test_the_panels_follow_the_figure_format_preference(tmp_path, monkeypatch):
    """A user who set PNG gets PNG panels, like every other spaCR figure."""
    import spacr.plot as plot_module

    monkeypatch.setattr(plot_module, "figure_output_preferences",
                        lambda: ("png", 300))
    _run(tmp_path)

    written = sorted(os.listdir(_qc_dir(tmp_path)))
    assert "scale_location.png" in written, written
    assert "residuals_vs_fitted.png" in written, written
    assert not any(name.endswith(".pdf") and name != "regression_qc_report.pdf"
                   for name in written), written


def test_the_mixed_model_branch_is_guarded_and_does_not_raise(tmp_path):
    """fit_mixed_model builds no design, so X and y do not exist there."""
    frame = _wells_df(seed=2, n_plates=2, n_rows=2, n_cols=3)
    model, coef_df, regression_type = _run(
        tmp_path, df=frame, random_row_column_effects=True)

    # The fit still succeeds -- a NameError here was the risk.
    assert regression_type == "mixed"
    assert not coef_df.empty
    # And it writes no QC folder rather than a half-built one.
    assert not os.path.isdir(_qc_dir(tmp_path))


def test_without_a_destination_nothing_is_written_and_nothing_raises(tmp_path):
    """regression_qc_report raises on a falsy dst on purpose; do not call it."""
    model, coef_df, regression_type = _run(tmp_path, dst=None)

    assert regression_type == "ols"
    assert not coef_df.empty
    assert not os.path.isdir(_qc_dir(tmp_path))


def test_qc_can_be_turned_off(tmp_path):
    _run(tmp_path, qc=False)
    assert not os.path.isdir(_qc_dir(tmp_path))


def test_a_broken_qc_report_does_not_destroy_a_successful_fit(tmp_path, capsys):
    """An hour-long fit must not be lost to a diagnostic."""
    import spacr.regression_qc as qc_module

    def _explode(*args, **kwargs):
        raise RuntimeError("qc is having a bad day")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(qc_module, "regression_qc_report", _explode)
    try:
        model, coef_df, regression_type = _run(tmp_path)
    finally:
        monkeypatch.undo()

    assert regression_type == "ols"
    assert not coef_df.empty
    assert "qc is having a bad day" in capsys.readouterr().out


def test_unusable_metadata_downgrades_to_unlabelled_rather_than_failing(tmp_path):
    """A duplicated index cannot be aligned; the fit still gets its report."""
    frame = _wells_df(seed=4)
    frame.index = np.zeros(len(frame), dtype=int)     # every label identical

    _run(tmp_path, df=frame)

    # The report is still written; the well-labelled panels skip themselves.
    assert os.path.isfile(
        os.path.join(_qc_dir(tmp_path), "regression_qc_report.txt"))
    assert os.path.isfile(os.path.join(_qc_dir(tmp_path), "scale_location.pdf"))


# ---------------------------------------------------------------------------
# The figure leak in regression_diagnostics._finish
# ---------------------------------------------------------------------------


def test_diagnostic_plots_close_their_figure_when_not_saving():
    """save_path=None used to return before plt.close, leaking one per call.

    All three public plot_*_diagnostics functions default to save_path=None --
    the way a caller asks for the report dict without keeping the picture --
    and they build their figures with plt.subplots, so pyplot held every one
    of them alive.
    """
    from spacr.regression_diagnostics import (plot_design_diagnostics,
                                              plot_inference_diagnostics,
                                              plot_residual_diagnostics)

    rng = np.random.default_rng(0)
    fractions = pd.DataFrame(rng.random((40, 6)),
                             columns=[f"g{i}" for i in range(6)])
    observed = rng.normal(size=40)
    fitted = observed + rng.normal(scale=0.2, size=40)
    p_values = rng.random(60)

    plt.close("all")
    before = len(plt.get_fignums())

    for _ in range(5):
        plot_design_diagnostics(fractions)
        plot_residual_diagnostics(observed, fitted)
        plot_inference_diagnostics(p_values)

    assert len(plt.get_fignums()) == before, (
        f"{len(plt.get_fignums()) - before} figure(s) left open by fifteen "
        f"diagnostic calls that were told not to save")


def test_diagnostic_plots_still_close_their_figure_when_saving(tmp_path):
    from spacr.regression_diagnostics import plot_design_diagnostics

    rng = np.random.default_rng(1)
    fractions = pd.DataFrame(rng.random((30, 4)),
                             columns=[f"g{i}" for i in range(4)])

    plt.close("all")
    path, report = plot_design_diagnostics(
        fractions, save_path=str(tmp_path / "design.png"))

    assert os.path.isfile(path)
    assert report
    assert plt.get_fignums() == []
