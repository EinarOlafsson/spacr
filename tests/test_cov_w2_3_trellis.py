"""Loading a table into Small Multiples, and saying so when it cannot be read.

The screen reads on a worker thread, so a second load has to supersede the
first rather than let two reads race to deliver their frames. Listing the
tables stays inline, which makes an unreadable file the one failure the
screen has to report by itself.
"""
from __future__ import annotations


import pandas as pd
import pytest

from spacr.qt.screens import trellis as T


@pytest.fixture
def screen(qtbot):
    """An unthreaded screen, so a load lands before the call returns."""
    widget = T.TrellisScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def measurement_csv(tmp_path):
    """A small measurement table with a grouping column to facet on."""
    frame = pd.DataFrame({
        "plate": ["p1", "p1", "p2", "p2"],
        "area": [10.0, 20.0, 30.0, 40.0],
        "perimeter": [4.0, 8.0, 12.0, 16.0],
    })
    path = tmp_path / "measurements.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_a_read_that_fails_on_the_worker_is_reported_against_the_path(screen):
    """The failure signal names the file the user chose, not the worker."""
    screen._path = "/data/plate7/measurements.db"
    screen._on_load_failed("database is locked")
    assert screen._source.text() == (
        "could not read measurements.db: database is locked")


def test_choosing_no_file_leaves_the_screen_as_it_was(screen, monkeypatch):
    """A cancelled file dialog is not a request to load anything."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_table()
    assert screen._frame is None
    assert screen._path is None


def test_choosing_a_file_loads_it(screen, monkeypatch, measurement_csv):
    """The path the dialog returns is the path that gets read."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (measurement_csv, "")))
    screen.choose_table()
    assert screen._path == measurement_csv
    assert screen._frame is not None


def test_nothing_is_pushed_before_a_table_has_been_loaded(screen):
    """With no frame there is nothing for the grid or the filter to hold."""
    assert screen.formulas.computed_frame() is None
    screen._push_frame()
    assert screen._frame is None


def test_the_spec_is_the_panels_own(screen, measurement_csv):
    """The screen forwards the grid's specification rather than copying it."""
    screen.load_path(measurement_csv)
    spec = screen.spec
    updated = spec.replace(x="area") if hasattr(spec, "replace") else spec
    screen.set_spec(updated)
    assert screen.spec == screen.panel.spec


def test_the_screen_reports_whether_a_read_is_still_running(screen):
    """The host asks the screen, which asks its runner."""
    assert screen.active_jobs() == 0
    assert screen.is_busy() is False


def test_the_table_signal_before_a_file_is_loaded_does_nothing(screen):
    """The combo is filled programmatically and fires while it is being filled.

    ``addItems`` emits ``currentTextChanged``, so the handler runs before any
    file has been chosen. Reloading on that would call ``load_path`` with an
    empty path -- a read of "" reported to the user as a file that could not
    be read, on a screen they have not touched.
    """
    screen._on_table_picked("well")

    assert screen._frame is None
    assert "could not read" not in screen._source.text()
