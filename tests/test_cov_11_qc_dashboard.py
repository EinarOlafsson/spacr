"""The QC dashboard re-reads whenever it cannot prove nothing changed.

The screen caches a whole project's verdicts behind a fingerprint of the
files they came from. Any part of that fingerprint being unavailable has to
mean "read again", never "the cache is still good" -- a dashboard showing
last week's verdicts for a plate that has since been re-masked is worse than
one that is merely slow.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.qc_dashboard import QCDashboardScreen
from spacr.qt.widgets.qc_summary import Dashboard


def _screen(qtbot, reader=None):
    screen = QCDashboardScreen(threaded=False, reader=reader)
    qtbot.addWidget(screen)
    return screen


def _counting_reader():
    calls = []

    def read(src):
        calls.append(src)
        return Dashboard(root=str(src), verdict="pass",
                         headline="nothing to report")

    read.calls = calls
    return read


def test_a_scorecard_scan_that_fails_forces_a_fresh_read(qtbot, tmp_path,
                                                         monkeypatch):
    """No fingerprint means no cache, so the next visit reads the disk again.

    Trusting a cache whose key could not be computed is how a re-masked
    plate keeps showing the verdicts of the masks it replaced.
    """
    from spacr import seg_qc

    def refuse(_src):
        raise RuntimeError("the qc folder cannot be listed")

    monkeypatch.setattr(seg_qc, "find_scorecards", refuse)
    reader = _counting_reader()
    screen = _screen(qtbot, reader)

    screen.set_source(str(tmp_path))
    screen.refresh()

    assert screen._fingerprint(str(tmp_path)) is None
    assert len(reader.calls) == 2


def test_a_scorecard_that_vanishes_between_listing_and_reading_forces_a_read(
        qtbot, tmp_path, monkeypatch):
    """A path that no longer exists cannot contribute to the fingerprint.

    Skipping it instead would produce a key that matches the cache while the
    set of files on disk has changed underneath it.
    """
    from spacr import seg_qc

    monkeypatch.setattr(seg_qc, "find_scorecards",
                        lambda _src: (str(tmp_path / "qc" / "gone.csv"),))
    reader = _counting_reader()
    screen = _screen(qtbot, reader)

    screen.set_source(str(tmp_path))
    screen.refresh()

    assert screen._fingerprint(str(tmp_path)) is None
    assert len(reader.calls) == 2


def test_a_fingerprint_that_can_be_taken_stops_the_second_read(qtbot,
                                                               tmp_path):
    """The control: with a usable fingerprint the cache is honoured.

    Without this, the two tests above would pass against a screen that never
    caches anything at all.
    """
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"x")
    reader = _counting_reader()
    screen = _screen(qtbot, reader)

    screen.set_source(str(tmp_path))
    assert screen.refresh() is False
    assert len(reader.calls) == 1
    assert "nothing on disk has changed" in screen.status_text().lower()


def test_a_read_that_produced_nothing_leaves_the_last_dashboard_alone(
        qtbot, tmp_path):
    """A failed read must not blank a screen that is showing real verdicts."""
    reader = _counting_reader()
    screen = _screen(qtbot, reader)
    screen.set_source(str(tmp_path))
    before = screen.as_text()

    screen._on_read(None)

    assert screen.dashboard() is not None
    assert screen.as_text() == before


def test_the_text_export_says_so_when_nothing_has_been_read(qtbot):
    """Copying from an empty screen gives a sentence, not an empty string."""
    screen = _screen(qtbot)

    assert screen.as_text() == "No folder read yet."


def test_browsing_to_a_folder_points_the_screen_at_it(qtbot, tmp_path,
                                                      monkeypatch):
    """The browse button is the same call as `set_source`."""
    from spacr.qt.screens import qc_dashboard

    reader = _counting_reader()
    screen = _screen(qtbot, reader)
    monkeypatch.setattr(qc_dashboard.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))

    screen._on_browse()

    assert screen.source() == str(tmp_path)
    assert reader.calls == [str(tmp_path)]


def test_cancelling_the_browse_dialog_leaves_the_folder_alone(qtbot, tmp_path,
                                                              monkeypatch):
    """An empty answer is a cancelled dialog, not a request for the root."""
    from spacr.qt.screens import qc_dashboard

    reader = _counting_reader()
    screen = _screen(qtbot, reader)
    screen.set_source(str(tmp_path))
    monkeypatch.setattr(qc_dashboard.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    screen._on_browse()

    assert screen.source() == str(tmp_path)
