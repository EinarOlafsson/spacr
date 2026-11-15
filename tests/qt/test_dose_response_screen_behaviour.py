"""The Dose-Response screen: loading a table, the pickers, refused fits, the figure, closing.

A CSV loads with no table picker; a SQLite database offers its tables and reads
the chosen one; a file that is not a database is named with the reason.  A fit
is refused (as text, never raised) with no frame or an impossible spec, and an
empty result prints its report.  The dose picker guesses by name only.  The
figure draws every curve, and the selected row's EC50 as a dashed line with a
band only when both interval ends exist, or as a dotted bound at the edge of
the tested doses when the doses never bracket it.  Closing stops the queued
draw, with or without a canvas that can cancel one.  Real files and a real
inline fit; only the interval fields of a result are edited.
"""
from __future__ import annotations

import dataclasses
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen   # noqa: E402
from spacr.qt.widgets.dose_response import (                     # noqa: E402
    CI_WALD, DoseResponseSet, DoseResponseSpec, four_parameter_logistic)

pytestmark = pytest.mark.qt

#: An eight-point three-fold dilution series, 27 uM down to about 12 nM.
DOSES = 27.0 / 3.0 ** np.arange(8)
NO_DOSE = ("No column of this table has at least four distinct positive "
           "values, so none of them can be a dilution series. A dose–response "
           "needs the concentration itself, not a log dose and not a plate "
           "coordinate.")


def _series(ec50, seed):
    """A clean inhibition curve through ``DOSES``, in triplicate, with noise."""
    dose = np.repeat(DOSES, 3)
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(ec50), -1.0)
    return dose, clean + np.random.default_rng(seed).normal(0.0, 1.0, dose.size)


@pytest.fixture()
def make(qtbot):
    """Build an unthreaded screen."""
    def _build():
        widget = DoseResponseScreen(threaded=False)
        qtbot.addWidget(widget)
        return widget
    return _build


@pytest.fixture()
def files(tmp_path):
    """A CSV, a two-table SQLite database, and a file that is not a database."""
    csv = tmp_path / "plate.csv"
    pd.DataFrame({"conc_uM": [0.1, 1.0, 10.0] * 3,
                  "signal": [10.0, 50.0, 90.0] * 3,
                  "gene": ["a"] * 9}).to_csv(csv, index=False)
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE object (conc_uM REAL, signal REAL)")
        connection.executemany("INSERT INTO object VALUES (?, ?)",
                               [(0.1, 10.0), (1.0, 50.0), (10.0, 90.0)])
        connection.execute("CREATE TABLE well (conc_uM REAL, other REAL)")
        connection.execute("INSERT INTO well VALUES (0.5, 1.0)")
    broken = tmp_path / "notes.db"
    broken.write_bytes(b"this is not a database")
    return csv, db, broken


@pytest.fixture()
def fitted(make):
    """A screen that has fitted two curves (geneA EC50 ~1, geneB ~0.3)."""
    parts = []
    for gene, ec50, seed in (("geneA", 1.0, 1), ("geneB", 0.3, 2)):
        dose, response = _series(ec50, seed)
        parts.append(pd.DataFrame({"gene": gene, "conc_uM": dose,
                                   "signal": response}))
    widget = make()
    widget.set_frame(pd.concat(parts, ignore_index=True), label="synthetic")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.unit_edit.setText("µM")
    widget.fit()
    assert [fit.group for fit in widget.result_set().fits] == ["geneA", "geneB"]
    return widget


def _offered(picker):
    return [picker.itemText(index) for index in range(picker.count())]


def test_load_path_reads_a_csv_or_the_chosen_database_table(make, files,
                                                            subtests):
    """The source line names the file, table and shape, or why it was refused."""
    csv, db, broken = files

    with subtests.test(msg="a csv has no table picker"):
        screen = make()
        screen.load_path(str(csv))
        assert screen._table_picker.isVisible() is False
        assert screen._table_picker.count() == 0
        assert screen._source.text() == "plate.csv · 9 rows × 3 columns"
        assert len(screen._frame) == 9
        # Three doses cannot be a dilution series.
        assert screen.report.toPlainText() == NO_DOSE
        assert screen.fit_button.isEnabled() is False

    with subtests.test(msg="a database offers its tables and reads the first"):
        screen = make()
        screen.load_path(str(db))
        assert _offered(screen._table_picker) == ["object", "well"]
        assert list(screen._frame.columns) == ["conc_uM", "signal"]
        assert screen._source.text() == (
            "measurements.db · object · 3 rows × 2 columns")
        screen._on_table_picked("well")
        assert list(screen._frame.columns) == ["conc_uM", "other"]
        assert screen._source.text() == (
            "measurements.db · well · 1 rows × 2 columns")

    with subtests.test(msg="the named table is the one read and shown"):
        screen = make()
        screen.load_path(str(db), table="well")
        assert screen._table_picker.currentText() == "well"
        assert list(screen._frame.columns) == ["conc_uM", "other"]

    with subtests.test(msg="picking no table reloads nothing"):
        screen = make()
        screen.load_path(str(db))
        frame = screen._frame
        screen._on_table_picked("")
        assert screen._frame is frame
        assert screen._source.text() == (
            "measurements.db · object · 3 rows × 2 columns")

    with subtests.test(msg="a table picked before any path loads nothing"):
        screen = make()
        screen._on_table_picked("well")
        assert screen._frame is None

    with subtests.test(msg="a file that is not a database says which and why"):
        screen = make()
        screen.load_path(str(broken))
        assert screen._source.text() == (
            "could not read notes.db: file is not a database")
        assert screen._frame is None


def test_a_fit_that_cannot_run_reports_instead_of_drawing(make, files,
                                                          subtests):
    """No frame, a refused spec, and an empty result each leave text, not a curve."""
    csv, _db, _broken = files

    with subtests.test(msg="no table loaded"):
        screen = make()
        screen.fit()
        assert screen.report.toPlainText() == ""
        assert screen.result_set() is None

    with subtests.test(msg="a ci method from a newer build"):
        screen = make()
        screen.load_path(str(csv))
        screen.ci_picker.addItem("From a newer build", "bootstrap")
        screen.ci_picker.setCurrentIndex(screen.ci_picker.count() - 1)
        screen.fit()
        assert screen.report.toPlainText() == (
            "unknown ci_method 'bootstrap'; it is 'profile' (profile "
            "likelihood, the default — it can report that the interval does "
            "not close) or 'wald' (asymptotic, always finite whether or not "
            "the data supports it)")
        assert screen.result_set() is None

    with subtests.test(msg="a fit with no groups prints its report"):
        screen = make()
        screen.load_path(str(csv))
        empty = DoseResponseSet(fits=(), spec=DoseResponseSpec(
            concentration="conc_uM", response="signal", ci_method=CI_WALD))
        screen._on_fitted(empty)
        assert screen.table.rowCount() == 0
        assert screen.result_set() is empty
        assert screen.report.toPlainText() == (
            "0 of 0 curve(s) give a bounded EC50.\n  4PL · signal vs conc_uM "
            "· 95% Wald CI on log10(EC50) · direction auto\n")
        assert list(screen._figure.axes[0].lines) == []

    with subtests.test(msg="a row selected before any fit draws nothing"):
        screen = make()
        screen.load_path(str(csv))
        report = screen.report.toPlainText()
        screen.table.setRowCount(1)
        screen.table.selectRow(0)
        screen._on_row_selected()
        assert screen.result_set() is None
        assert screen.report.toPlainText() == report


def test_the_dose_picker_guesses_by_name_and_keeps_the_users_choice(make,
                                                                    subtests):
    """The first dose-shaped name wins; with none the first column shows."""
    dose, response = _series(1.0, seed=7)
    cases = [
        # (case id, frame, offered doses, chosen dose)
        ("walks past a numeric column that is not a dose",
         pd.DataFrame({"assay_level": dose * 2.0, "conc_uM": dose,
                       "signal": response}),
         ["assay_level", "conc_uM", "signal"], "conc_uM"),
        ("no dose-shaped name leaves the first column",
         pd.DataFrame({"x": dose, "y": response}), ["x", "y"], "x"),
    ]
    for case_id, frame, offered, chosen in cases:
        with subtests.test(msg=case_id):
            screen = make()
            screen.set_frame(frame)
            assert _offered(screen.concentration_picker) == offered
            assert screen.concentration_picker.currentText() == chosen
            assert screen.fit_button.isEnabled() is True
            assert screen.report.toPlainText() == ""
            assert screen._source.text() == (
                f"24 rows × {len(frame.columns)} columns")

    with subtests.test(msg="a reload keeps the column the user had chosen"):
        frame = pd.DataFrame({"conc_uM": [0.01, 0.1, 1.0, 10.0, 100.0] * 3,
                              "dose_nM": [10.0, 100.0, 1e3, 1e4, 1e5] * 3,
                              "signal": [5.0, 20.0, 50.0, 80.0, 95.0] * 3})
        screen = make()
        screen.set_frame(frame, label="first")
        assert screen.concentration_picker.currentText() == "conc_uM"
        screen.concentration_picker.setCurrentText("dose_nM")
        screen.set_frame(frame, label="again")
        assert screen.concentration_picker.currentText() == "dose_nM"
        assert screen._source.text() == "again"


def _axes(widget):
    return widget._figure.axes[0]


def _styles(axes):
    return [line.get_linestyle() for line in axes.lines]


def _with_first_result(widget, **changes):
    """Rebuild the loaded set with the first curve's result fields overridden."""
    original = widget.result_set()
    first = original.fits[0]
    edited = dataclasses.replace(first, result=dataclasses.replace(
        first.result, **changes))
    widget._set = dataclasses.replace(original,
                                      fits=(edited,) + original.fits[1:])
    return widget._set.fits[0].result


CURVES = ["None", "-", "None", "-"]


def test_the_figure_marks_only_the_selected_rows_ec50(fitted, subtests):
    """Every curve is drawn; the marker belongs to the selected, in-range row."""
    fitted._draw(None)
    plain_xlim = _axes(fitted).get_xlim()
    fits = fitted.result_set().fits

    with subtests.test(msg="no row selected"):
        axes = _axes(fitted)
        assert _styles(axes) == CURVES
        assert list(axes.texts) == []
        assert len(axes.patches) == 0
        assert {line.get_alpha() for line in axes.lines} == {0.9, 1.0}
        assert axes.get_xscale() == "log"
        assert axes.get_xlabel() == "conc_uM (µM)"
        assert axes.get_ylabel() == "signal"
        assert plain_xlim[0] < float(DOSES.min())
        assert plain_xlim[1] > float(DOSES.max())

    for selected in (0, 1):
        with subtests.test(msg=f"row {selected} selected"):
            fitted._draw(selected)
            axes = _axes(fitted)
            result = fits[selected].result
            assert _styles(axes) == CURVES + ["--"]
            assert axes.lines[-1].get_xdata()[0] == pytest.approx(result.ec50)
            assert [text.get_text() for text in axes.texts] == [
                f"EC50 {result.ec50:.3g}"]
            assert len(axes.patches) == 1
            # The other curve fades to 0.25 (points) / 0.3 (line).
            assert {line.get_alpha() for line in axes.lines[:4]} == {
                0.9, 1.0, 0.25, 0.3}
            assert axes.get_xlim() == pytest.approx(plain_xlim)

    with subtests.test(msg="a selection the last fit no longer has"):
        fitted._draw(len(fits) + 3)
        axes = _axes(fitted)
        assert _styles(axes) == CURVES
        assert list(axes.texts) == []
        assert len(axes.patches) == 0
        assert {line.get_alpha() for line in axes.lines} == {0.25, 0.3}


def test_an_ec50_interval_or_bound_is_drawn_as_the_numbers_say(fitted,
                                                               subtests):
    """An open interval end drops the band; an unbracketed EC50 is a dotted bound."""
    cases = [
        # (case id, result changes, last line style, band count, label format)
        ("open at the bottom", {"ec50_low": None}, "--", 0, "EC50 {ec50:.3g}"),
        ("open at the top", {"ec50_high": None}, "--", 0, "EC50 {ec50:.3g}"),
        ("above the doses",
         {"ec50": None, "ec50_bounded": False, "bound_direction": "above"},
         ":", 0, "EC50 > {dose_max:.3g}"),
        ("below the doses",
         {"ec50": None, "ec50_bounded": False, "bound_direction": "below"},
         ":", 0, "EC50 < {dose_min:.3g}"),
        ("bounded but missing its value",
         {"ec50": None, "ec50_bounded": True, "bound_direction": "above"},
         ":", 0, "EC50 > {dose_max:.3g}"),
    ]
    original = fitted.result_set()
    for case_id, changes, style, bands, label in cases:
        with subtests.test(msg=case_id):
            fitted._set = original
            result = _with_first_result(fitted, **changes)
            fitted._draw(0)
            axes = _axes(fitted)
            assert _styles(axes) == CURVES + [style]
            assert len(axes.patches) == bands
            assert [text.get_text() for text in axes.texts] == [
                label.format(ec50=result.ec50 or 0.0,
                             dose_max=float(result.dose_max),
                             dose_min=float(result.dose_min))]
            if style == ":":
                edge = (result.dose_min if changes["bound_direction"] == "below"
                        else result.dose_max)
                assert axes.lines[-1].get_xdata()[0] == pytest.approx(edge)
            else:
                assert axes.lines[-1].get_xdata()[0] == pytest.approx(
                    result.ec50)


class _CanvasWithoutATimer:
    """A canvas that never queued a draw, so it has nothing to cancel."""


def test_closing_the_screen_stops_its_queued_draw_and_workers(make, fitted,
                                                              subtests):
    """A close cancels the canvas's pending draw, and survives a canvas with no hook."""
    frame = fitted._frame

    with subtests.test(msg="the queued draw is cancelled"):
        screen = make()
        screen.set_frame(frame, label="synthetic")
        assert screen.canvas._draw_pending is True
        assert screen.close() is True
        assert screen.canvas._draw_pending is False
        assert screen.canvas._spacr_draw_timer.isActive() is False
        assert screen.is_busy() is False

    with subtests.test(msg="a canvas with no cancel hook"):
        screen = make()
        screen.set_frame(frame, label="synthetic")
        screen.canvas = _CanvasWithoutATimer()
        assert screen.close() is True
        assert screen.isVisible() is False
        assert screen.active_jobs() == 0
        assert screen.is_busy() is False
