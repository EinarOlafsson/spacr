"""Volcano Explorer screen — finding the table, trimming it, and the reader.

The screen is a reader with three jobs and this file drives all three
against real files on disk rather than around them:

* :func:`find_results_table` is handed a CSV, a non-CSV file, a path that
  does not exist, a folder holding several candidate names at once, and a
  parent one level above the folder that actually holds the table;
* :func:`load_results` is given a stacked permutation table -- every
  minimum-support family and two fitted responses in one file -- and has to
  come back with one family, one response and a fresh index, because
  plotting the stack draws each guide two to four times and that reads as
  extra hits;
* the widget itself opens a folder, refuses one with nothing in it, and
  accepts a dropped directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QFileDialog

from spacr.qt.screens import volcano


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _results_frame(n=24, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "guide": [f"g{i}" for i in range(n)],
        "gene": [f"TGGT1_{200000 + i // 3}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(0, 0.08, n),
        "adjusted_p_value": np.clip(rng.beta(0.6, 6, n), 1e-8, 1),
        "wells_with_guide": rng.integers(1, 12, n),
        "significant": False,
        "alpha": 0.05,
    })


@pytest.fixture
def results_folder(tmp_path):
    """A regression output folder holding a plain ``results.csv``."""
    folder = tmp_path / "run_2024"
    folder.mkdir()
    _results_frame().to_csv(folder / "results.csv", index=False)
    return folder


# ---------------------------------------------------------------------------
# find_results_table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qtbot):
    widget = volcano._make_screen(app_key=volcano.APP_KEY, host=None)
    qtbot.addWidget(widget)
    return widget


def test_the_open_button_loads_the_folder_the_dialog_returned(
        screen, results_folder, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(results_folder)))
    screen._open()
    assert screen._path_label.text().endswith("results.csv")


def test_cancelling_the_open_dialog_leaves_the_screen_alone(screen,
                                                            monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._open()
    assert "No results loaded" in screen._path_label.text()


def test_importing_the_module_puts_no_row_in_the_registry():
    """The explorer is a fold, so importing it must add no tile.

    This asserted the import-time ``register()`` had run and that a
    second call was a no-op. Both the call and the function are gone:
    the explorer is "Publication figure…" on the Regression volcano and
    a button on that masthead, and the failure worth catching now is the
    row coming back.
    """
    from spacr.qt.app import APPS

    assert not any(row[0] == volcano.APP_KEY for row in APPS)
    assert not hasattr(volcano, "register")


def test_the_gui_only_sentence_is_written_where_the_cli_reads_it():
    """The sentence survived the row, in both of its copies.

    It travelled into ``cli.INTERACTIVE_ONLY`` as the row's
    ``cli_note=`` and ``unregister_app`` takes a pushed entry back out
    with the row, so without a hand-written copy ``spacr-run
    volcano_explorer`` would stop naming the renderer to call and start
    guessing at a typo. ``spacr.cli`` answers ``--list`` where PySide6 is
    not installed, so it cannot read the sentence from here; the two
    copies are pinned equal instead.
    """
    from spacr import cli

    assert cli.INTERACTIVE_ONLY[volcano.APP_KEY] == volcano.APP_CLI_NOTE
    assert volcano.APP_KEY not in cli.MODULES
