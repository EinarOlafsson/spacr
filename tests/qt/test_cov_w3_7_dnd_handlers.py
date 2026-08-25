"""Drop paths through the handlers that the other dnd files do not reach.

``tests/qt/test_dnd_handlers_full.py``, ``test_cov_dnd_handlers.py`` and
``tests/test_cov_w3_8_dnd.py`` cover the registry, the dropzone and the bulk of
the handlers between them. What is left, and is asserted here, is the small
set of paths a drop takes when the widget on the other side refuses -- a
source control that raises while adding, one that will not say what it holds
-- and Regression's two-sided drop, where the same target has to take a
measurements database OR the parameter sweep's CSVs.

Handlers are given stand-in screens: each is one or two attributes, and what
these tests are about is exactly what happens when one of them is missing or
throws.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import dnd_handlers as dh


class _Console:
    """The one method ``_log`` calls."""

    def __init__(self):
        self.lines = []

    def append_stdout(self, text):
        self.lines.append(text)


class _Model:
    def __init__(self, widgets):
        self._widgets = dict(widgets)


class _Screen:
    """A screen with a settings model, a console, and nothing else."""

    def __init__(self, widgets=None, app_key="measure"):
        self._settings_model = _Model(widgets or {})
        self._console = _Console()
        self.app_key = app_key


# ---------------------------------------------------------------------------
# Adding to a multi-source control
# ---------------------------------------------------------------------------

def test_a_source_control_that_refuses_the_add_hands_back_to_the_caller():
    """``None`` means "not handled here", so the single-source path runs."""
    class Refuses:
        def add_sources(self, values):
            raise RuntimeError("the list is read-only")

        def sources(self):
            return []

    screen = _Screen({"src": Refuses()})
    assert dh._add_to_source_set(screen, "/plate") is None


def test_a_control_that_will_not_say_what_it_holds_counts_as_added():
    """The add succeeded; only the read-back failed, and it is a check."""
    class Deaf:
        def __init__(self):
            self.added = []

        def add_sources(self, values):
            self.added.extend(values)
            return len(values)

        def sources(self):
            raise RuntimeError("the model is being rebuilt")

    widget = Deaf()
    screen = _Screen({"src": widget})
    assert dh._add_to_source_set(screen, ["/plate1", "/plate2"]) is True
    assert widget.added == ["/plate1", "/plate2"]


def test_a_screen_with_no_source_control_is_not_this_functions_business():
    assert dh._add_to_source_set(_Screen({}), "/plate") is None
    assert dh._add_to_source_set(object(), "/plate") is None


# ---------------------------------------------------------------------------
# A measurements database onto a plate row
# ---------------------------------------------------------------------------

def _plate_with_database(tmp_path):
    root = tmp_path / "plate1"
    (root / "measurements").mkdir(parents=True)
    (root / "measurements" / "measurements.db").write_bytes(b"SQLite format 3")
    return root


def test_a_dropped_database_goes_onto_the_plate_row_not_into_src(tmp_path):
    """``src`` is not where a paired-input screen's measurements live."""
    class PairedTable:
        def __init__(self):
            self.attached = []

        def attach_database(self, path):
            self.attached.append(path)
            return f"attached {Path(path).name} to plate1"

    table = PairedTable()
    screen = _Screen({"paired_data": table, "src": object()},
                     app_key="regression")
    root = _plate_with_database(tmp_path)

    dh.MeasurementsDropHandler().apply(root, screen)
    assert table.attached == [str(root / "measurements" / "measurements.db")]
    assert screen._console.lines == [
        "[drop] attached measurements.db to plate1\n"]


def test_a_bare_database_file_is_its_own_answer(tmp_path):
    database = tmp_path / "measurements.db"
    database.write_bytes(b"SQLite format 3")
    assert dh._measurement_database(database) == database
    assert dh.MeasurementsDropHandler.database_file(database) == database


# ---------------------------------------------------------------------------
# The sweep's CSVs
# ---------------------------------------------------------------------------

def test_one_csv_is_one_table(tmp_path):
    csv = tmp_path / "scores.csv"
    csv.write_text("prc,pred\nplate1_r1_c1,0.5\n")
    assert dh.SweepInputsDropHandler._tables(csv) == [csv]
    assert dh.SweepInputsDropHandler._tables(tmp_path / "gone.csv") == []


# ---------------------------------------------------------------------------
# Regression's two-sided drop
# ---------------------------------------------------------------------------

class _List:
    def __init__(self):
        self.paths = []

    def add_paths(self, values):
        self.paths.extend(values)


class _SweepCard:
    def __init__(self):
        self.score_data = _List()
        self.count_data = _List()


def _regression_screen(tmp_path=None):
    screen = _Screen({}, app_key="regression")
    screen._sweep = _SweepCard()
    return screen


def test_regression_takes_a_database_or_the_sweeps_csvs(tmp_path):
    handler = dh.RegressionDropHandler()
    assert handler.accepts_multiple() is True

    root = _plate_with_database(tmp_path)
    assert handler.can_accept(root) is True

    csv = tmp_path / "counts.csv"
    csv.write_text("prc,grna,count\nplate1_r1_c1,A,7\n")
    assert handler.can_accept(csv) is True

    nothing = tmp_path / "notes.txt"
    nothing.write_text("hello")
    assert handler.can_accept(nothing) is False
    assert "measurements/measurements.db" in handler.error_message(nothing)
    assert "score / gRNA count CSVs" in handler.error_message(nothing)


def test_the_database_half_wins_when_the_path_is_a_plate(tmp_path):
    class PairedTable:
        def __init__(self):
            self.attached = []

        def attach_database(self, path):
            self.attached.append(path)
            return "attached"

    screen = _regression_screen()
    table = PairedTable()
    screen._settings_model._widgets["paired_data"] = table

    dh.RegressionDropHandler().apply(_plate_with_database(tmp_path), screen)
    assert table.attached
    assert screen._sweep.score_data.paths == []
    assert screen._sweep.count_data.paths == []


def test_each_csv_lands_on_the_side_its_header_says(tmp_path):
    """The header decides, not the filename: a count filed as a score is a
    wrong sweep rather than an error the user ever sees."""
    counts = tmp_path / "table_one.csv"
    counts.write_text("prc,grna,count\nplate1_r1_c1,A,7\n")
    scores = tmp_path / "table_two.csv"
    scores.write_text("prc,pred\nplate1_r1_c1,0.5\n")
    screen = _regression_screen()

    dh.RegressionDropHandler().apply(counts, screen)
    dh.RegressionDropHandler().apply(scores, screen)
    assert screen._sweep.count_data.paths == [str(counts)]
    assert screen._sweep.score_data.paths == [str(scores)]
    assert "parameter_sweep count" in screen._console.lines[0]
    assert "parameter_sweep score" in screen._console.lines[1]


def test_a_regression_screen_with_no_sweep_card_says_so(tmp_path):
    csv = tmp_path / "counts.csv"
    csv.write_text("prc,grna,count\nplate1_r1_c1,A,7\n")
    screen = _Screen({}, app_key="regression")

    with pytest.raises(TypeError, match="no parameter-sweep inputs"):
        dh.RegressionDropHandler().apply(csv, screen)
