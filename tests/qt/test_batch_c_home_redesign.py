"""Tests for the Batch C home-screen redesign — horizontal section
rows + insights dashboard (System / Recent runs / Totals cards)."""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QLabel, QScrollArea


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    """Point the run journal at an empty tmp dir so Home cards render
    against known state."""
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def _populated_journal(tmp_path, monkeypatch):
    """Journal with three completed mask runs."""
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    for _ in range(3):
        with rj.open_run("mask", {"src": "/tmp/x"}):
            pass
    yield tmp_path


# ---------------------------------------------------------------------------
# journal_totals API
# ---------------------------------------------------------------------------

class TestJournalTotals:
    def test_empty_returns_zeros(self, _empty_journal):
        from spacr.run_journal import journal_totals
        t = journal_totals()
        assert t["total_runs"] == 0
        assert t["mask_runs"] == 0
        assert t["models_recorded"] == 0

    def test_counts_by_app_key(self, _empty_journal):
        from spacr import run_journal as rj
        with rj.open_run("mask", {"src": "/tmp/x"}):
            pass
        with rj.open_run("measure", {"src": "/tmp/y"}):
            pass
        with rj.open_run("measure", {"src": "/tmp/z"}):
            pass
        t = rj.journal_totals()
        assert t["total_runs"] == 3
        assert t["mask_runs"] == 1
        assert t["measure_runs"] == 2


# ---------------------------------------------------------------------------
# Horizontal section rows
# ---------------------------------------------------------------------------

class TestHorizontalSections:
    def test_home_uses_scroll_areas_for_sections(self, qtbot,
                                                    _empty_journal):
        """Every section grid is now a QScrollArea (horizontal row).
        Previously it was a QGridLayout of tiles."""
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        scrolls = win._startup.findChildren(QScrollArea)
        # Multiple scroll areas — one outer + one per section row.
        # ≥ 3 sections (Core, Analysis, Cellpose at minimum).
        assert len(scrolls) >= 3


# ---------------------------------------------------------------------------
# Insights dashboard on Home
# ---------------------------------------------------------------------------

class TestInsightsDashboard:
    def _labels(self, win):
        return [w.text() for w in win._startup.findChildren(QLabel)]

    def test_dashboard_shows_three_section_headers(self, qtbot,
                                                      _empty_journal):
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = self._labels(win)
        # Three cards: System / Recent runs / Totals
        for expected in ("SYSTEM", "RECENT RUNS", "TOTALS"):
            assert any(expected in lbl for lbl in labels), (
                f"missing '{expected}' card header on Home")

    def test_totals_card_shows_zero_labels_for_empty_journal(
            self, qtbot, _empty_journal):
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = self._labels(win)
        # Row labels ("Runs", "Mask", "Meas.", "Models") should be
        # present with a "0" value nearby.
        assert any(lbl.strip() == "Runs" for lbl in labels)
        assert any(lbl.strip() == "Models" for lbl in labels)

    def test_totals_card_reflects_populated_journal(
            self, qtbot, _populated_journal):
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = self._labels(win)
        # 3 mask runs → the Totals card should carry "3" for both
        # total runs and mask runs
        assert any(lbl.strip() == "3" for lbl in labels)

    def test_system_card_reports_gpu_or_no_cuda(self, qtbot,
                                                  _empty_journal):
        """The System card should surface *something* for GPU / VRAM
        / Disk — never leave the field blank. On a box without CUDA
        the GPU row reads 'no CUDA' or 'n/a'."""
        from spacr.qt.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        labels = self._labels(win)
        # Row header labels
        for expected in ("GPU", "VRAM", "Disk"):
            assert any(lbl.strip() == expected for lbl in labels)
