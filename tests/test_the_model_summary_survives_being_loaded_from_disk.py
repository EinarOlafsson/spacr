"""Instruction 153 — the model summary survives being loaded from disk.

Reported 2026-08-18, on a run that had just finished:

    "i get: 'No summary: this panel was opened from a results table on disk
     rather than from a run, so the fitted model is not here. Re-run to see
     it.' dor the run, also eaven if this was loaded from disk i should still
     have access to the summary."

Two separate defects, and the file names both:

* THE SUMMARY WAS ALREADY ON DISK AND NOTHING READ IT BACK. Every ols/beta
  run writes ``model.summary().as_text()`` into its own results folder, so a
  panel opened from a results table has it one ``open()`` away. "Re-run to
  see it" asked for GPU minutes to recover a file sitting beside the table.
* "OPENED FROM DISK" WAS SOMETIMES A FALSE EXPLANATION. ``self._model`` was
  assigned AFTER the diagnostics were computed, so a fit whose QC context
  could not be built had its model thrown away -- and the tab then explained
  the absence with a cause that had not happened.

Every path here ends in either a summary or a TRUE sentence about why not.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sm = pytest.importorskip("statsmodels.api")

from spacr.qt.widgets.regression_results import (  # noqa: E402
    SUMMARY_FROM_DISK, find_summary_file, summary_text,
)


def _fit():
    rng = np.random.default_rng(0)
    X = sm.add_constant(rng.normal(size=(60, 2)))
    y = X @ [1.0, 0.4, -0.2] + rng.normal(0, 0.3, 60)
    return sm.OLS(y, X).fit()


def _results_table(folder):
    """The one file the panel searches for, so ``folder`` reads as a run."""
    import pandas as pd

    frame = pd.DataFrame({"feature": ["a", "b"], "coefficient": [0.4, -0.2],
                          "p_value": [0.01, 0.5]})
    path = os.path.join(folder, "results.csv")
    frame.to_csv(path, index=False)
    return frame, path


# --------------------------------------------------------------------------- #
#  The file a run writes: its name, and that it is written at all
# --------------------------------------------------------------------------- #

def test_the_run_writes_model_summary_txt_not_mode_summary_csv(tmp_path):
    """"mode" was a typo for "model" and the content was never CSV."""
    from spacr.ml import SUMMARY_FILENAME, save_summary_to_file

    assert SUMMARY_FILENAME == "model_summary.txt"
    written = save_summary_to_file(_fit(),
                                   file_path=str(tmp_path / SUMMARY_FILENAME))
    assert os.path.isfile(written)
    assert "OLS Regression Results" in open(written).read()


def test_the_default_name_is_the_new_one():
    """A default of ``summary.csv`` is a path nobody can open."""
    from spacr.ml import SUMMARY_FILENAME, save_summary_to_file

    default = inspect.signature(save_summary_to_file).parameters["file_path"]
    assert default.default == SUMMARY_FILENAME


def test_the_old_name_is_still_read():
    """Old runs on disk keep working -- migrate, do not break."""
    from spacr.ml import SUMMARY_FILENAMES

    assert "mode_summary.csv" in SUMMARY_FILENAMES
    assert SUMMARY_FILENAMES[0] == "model_summary.txt"


def test_a_backend_without_a_summary_is_not_a_crashed_run(tmp_path):
    """It is called after every table is written; it must not raise."""
    from spacr.ml import save_summary_to_file

    class NoSummary:
        pass

    assert save_summary_to_file(NoSummary(),
                                file_path=str(tmp_path / "x.txt")) is None
    assert not os.path.exists(tmp_path / "x.txt")


def test_a_summary_that_raises_does_not_take_the_run_down(tmp_path):
    from spacr.ml import save_summary_to_file

    class Broken:
        def summary(self):
            raise RuntimeError("no design matrix")

    assert save_summary_to_file(Broken(),
                                file_path=str(tmp_path / "x.txt")) is None


def test_the_summary_is_written_whether_or_not_the_run_is_verbose():
    """It used to sit inside the `verbose` branch beside the print.

    A quiet run -- the normal case -- therefore left NO summary on disk at
    all, and the panel re-opened from that folder had nothing to read back.
    Printing is a console preference; the summary is part of the output.
    """
    from spacr import ml

    lines = inspect.getsource(ml.perform_regression).splitlines()
    call = next(i for i, line in enumerate(lines)
                if "save_summary_to_file(" in line)
    verbose = max(i for i in range(call)
                  if lines[i].strip() == "if settings['verbose']:")

    def indent(line):
        return len(line) - len(line.lstrip())

    # At or outside the `if`'s own indentation is outside its body. Inside it
    # the call was indented one level further, and a quiet run wrote nothing.
    assert indent(lines[call]) <= indent(lines[verbose]), lines[verbose:call + 1]
    assert "mode_summary.csv" not in "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Finding it again
# --------------------------------------------------------------------------- #

def test_a_run_folder_finds_its_own_summary(tmp_path):
    _results_table(tmp_path)
    (tmp_path / "model_summary.txt").write_text("OLS Regression Results\n")

    assert find_summary_file(str(tmp_path)) == str(
        tmp_path / "model_summary.txt")


def test_the_results_csv_finds_the_summary_beside_it(tmp_path):
    _frame, path = _results_table(tmp_path)
    (tmp_path / "model_summary.txt").write_text("OLS Regression Results\n")

    assert find_summary_file(path) == str(tmp_path / "model_summary.txt")


def test_an_old_run_with_the_typo_name_still_answers(tmp_path):
    _results_table(tmp_path)
    (tmp_path / "mode_summary.csv").write_text("OLS Regression Results\n")

    assert find_summary_file(str(tmp_path)) == str(
        tmp_path / "mode_summary.csv")


def test_the_new_name_wins_when_a_folder_holds_both(tmp_path):
    """A re-run of an old folder leaves both. The new one is this run's."""
    _results_table(tmp_path)
    (tmp_path / "mode_summary.csv").write_text("old\n")
    (tmp_path / "model_summary.txt").write_text("new\n")

    assert find_summary_file(str(tmp_path)).endswith("model_summary.txt")


def test_a_parent_of_a_run_folder_finds_the_runs_summary(tmp_path):
    """`load` accepts a parent, so the summary is looked for where the table
    that was actually chosen is -- not only in the folder that was typed."""
    run = tmp_path / "results" / "ols_1"
    run.mkdir(parents=True)
    _results_table(run)
    (run / "model_summary.txt").write_text("OLS Regression Results\n")

    assert find_summary_file(str(tmp_path)) == str(run / "model_summary.txt")


def test_a_folder_with_no_summary_is_none_not_a_traceback(tmp_path):
    _results_table(tmp_path)

    assert find_summary_file(str(tmp_path)) is None
    assert find_summary_file("") is None
    assert find_summary_file(None) is None


# --------------------------------------------------------------------------- #
#  What the tab shows
# --------------------------------------------------------------------------- #

def test_the_summary_is_read_back_rather_than_re_fitted(tmp_path):
    body = _fit().summary().as_text()
    _results_table(tmp_path)
    (tmp_path / "model_summary.txt").write_text(body)

    text = summary_text(None, path=str(tmp_path))

    assert body.strip() in text
    assert not text.startswith("No summary")
    assert "Re-run" not in text


def test_it_says_the_summary_is_the_runs_own_and_names_the_file(tmp_path):
    """A reader pasting it into a methods section is entitled to know it was
    read from the run rather than recomputed here."""
    _results_table(tmp_path)
    path = tmp_path / "model_summary.txt"
    path.write_text(_fit().summary().as_text())

    text = summary_text(None, path=str(tmp_path))

    assert text.startswith(SUMMARY_FROM_DISK.format(path=str(path)))


def test_an_older_run_with_no_summary_file_says_exactly_that(tmp_path):
    _results_table(tmp_path)

    text = summary_text(None, path=str(tmp_path))

    assert text.startswith("No summary")
    assert str(tmp_path) in text                 # names the folder looked in
    assert "model_summary.txt" in text           # and what it looked for
    assert "mode_summary.csv" in text


def test_no_path_does_not_claim_a_disk_that_was_never_read():
    text = summary_text(None)

    assert "results table on disk" not in text
    assert text.startswith("No summary")


def test_an_unreadable_summary_file_is_reported_not_raised(tmp_path):
    _results_table(tmp_path)
    path = tmp_path / "model_summary.txt"
    path.write_text("something")
    os.chmod(path, 0o000)
    try:
        if os.access(path, os.R_OK):             # running as root
            pytest.skip("cannot make a file unreadable here")
        text = summary_text(None, path=str(tmp_path))
    finally:
        os.chmod(path, 0o644)

    assert text.startswith("No summary")
    assert "could not be read" in text


def test_an_empty_summary_file_is_not_an_empty_tab(tmp_path):
    _results_table(tmp_path)
    (tmp_path / "model_summary.txt").write_text("   \n")

    text = summary_text(None, path=str(tmp_path))

    assert text.startswith("No summary")
    assert "empty" in text


# --------------------------------------------------------------------------- #
#  The panel
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_a_table_opened_from_disk_shows_the_runs_summary(qtbot, tmp_path):
    """The whole of the request: close the panel, re-open the table, and the
    Summary tab holds the same text it held during the run."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    body = _fit().summary().as_text()
    _results_table(tmp_path)
    (tmp_path / "model_summary.txt").write_text(body)

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    assert panel.load(str(tmp_path))

    shown = panel._summary.toPlainText()
    assert body.strip() in shown
    assert "Re-run to see it" not in shown


@pytest.mark.qt
def test_a_new_table_does_not_leave_the_previous_runs_summary_up(qtbot,
                                                                 tmp_path):
    """A stale summary under a new table is authoritative and about nothing."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    first = tmp_path / "run_a"
    first.mkdir()
    frame, _path = _results_table(first)
    (first / "model_summary.txt").write_text("SUMMARY OF RUN A\n")
    second = tmp_path / "run_b"
    second.mkdir()
    _results_table(second)

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    assert panel.load(str(first))
    assert "SUMMARY OF RUN A" in panel._summary.toPlainText()

    assert panel.load(str(second))
    assert "SUMMARY OF RUN A" not in panel._summary.toPlainText()
    assert str(second) in panel._summary.toPlainText()


@pytest.mark.qt
def test_a_fit_whose_diagnostics_fail_keeps_its_model(qtbot):
    """A failure in the VIEW must not destroy the thing being viewed."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets import regression_results as RR
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    model = _fit()

    import spacr.regression_qc as QC

    def explode(*_args, **_kwargs):
        raise ValueError("the design matrix is not on this fit")

    original = QC.context_from_model
    QC.context_from_model = explode
    try:
        assert panel.set_diagnostics(model, regression_type="ols") is False
    finally:
        QC.context_from_model = original

    assert panel._model is model, "the fit was thrown away by its own view"
    assert RR.summary_text is not None
    # And the Summary tab renders it, live, with no re-fit.
    assert panel.set_summary(None, regression_type="ols") is True
    assert "OLS Regression Results" in panel._summary.toPlainText()


@pytest.mark.qt
def test_the_diagnostics_say_which_error_stopped_them(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    import spacr.regression_qc as QC

    def explode(*_args, **_kwargs):
        raise ValueError("the design matrix is not on this fit")

    original = QC.context_from_model
    QC.context_from_model = explode
    try:
        panel.set_diagnostics(_fit(), regression_type="ols")
    finally:
        QC.context_from_model = original

    # Read off the plot's own status line -- the sentence a user sees where
    # the diagnostics would have been -- and off the constant-spread verdict,
    # which is the other place the absence shows.
    for plot in panel.diagnostic_plots():
        assert "the design matrix is not on this fit" in plot._headline, (
            plot._headline)
    assert "the design matrix is not on this fit" in (
        panel.homogeneity_verdict())


@pytest.mark.qt
def test_a_live_run_with_no_model_does_not_blame_the_disk(qtbot, tmp_path):
    """The run finished and returned no model. That is the true sentence."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    frame, _path = _results_table(tmp_path)
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    assert panel.set_frame(frame, source=str(tmp_path))
    panel.set_run_settings({"regression_type": "ols"})
    panel.set_summary(None, regression_type="ols")

    shown = panel._summary.toPlainText()
    assert "opened from a results table on disk" not in shown
    assert "came back without a fitted model" in shown
