"""What the PCA screen does when the table, the filter or the disk says no.

Every one of these paths ends in a sentence on the source label rather than
in a dialog or a traceback: the screen is used from a headless-ish offscreen
session as often as from a desktop, and a modal nobody can dismiss is how
such a session hangs. The tests read that label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog                       # noqa: E402

from spacr.qt.linked_selection import LinkedSelection          # noqa: E402
from spacr.qt.screens import pca as pca_screen                  # noqa: E402
from spacr.selection import CategoryFilter, DataFilter          # noqa: E402


@pytest.fixture
def link():
    """A private link, never the process-wide one every open view listens to."""
    return LinkedSelection()


@pytest.fixture
def screen(qtbot, link):
    """A PCA screen whose reads and fits run inline, so a call settles it."""
    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def features():
    """Four correlated numeric features and a group column, 60 rows."""
    rng = np.random.default_rng(7)
    shift = np.repeat([0.0, 1.0], 30)
    return pd.DataFrame({
        "plateID": ["p1"] * 60,
        "rowID": ["r1"] * 60,
        "columnID": [f"c{i % 4 + 1}" for i in range(60)],
        "fieldID": ["f1"] * 60,
        "object_label": list(range(60)),
        "f1": 10.0 + 4.0 * shift + rng.normal(scale=0.4, size=60),
        "f2": 20.0 - 3.0 * shift + rng.normal(scale=0.4, size=60),
        "f3": 5.0 + 2.0 * shift + rng.normal(scale=0.4, size=60),
        "f4": rng.normal(scale=1.0, size=60),
        "gene": ["control"] * 30 + ["treated"] * 30,
    })


# ---------------------------------------------------------------------------
# the filter is upstream of the maths
# ---------------------------------------------------------------------------

def test_with_no_table_there_is_nothing_to_filter(screen):
    """``_filtered`` answers None rather than inventing an empty frame."""
    assert screen._filtered() is None


def test_recomputing_with_no_table_does_nothing(screen):
    """A filter change before a load is not an error."""
    screen._recompute_filtered()               # must not raise

    assert screen.pca.result is None


def test_a_filter_naming_a_missing_column_is_reported_not_swallowed(
        screen, link, features, caplog):
    """The whole frame is decomposed, and the mismatch is logged."""
    screen.set_frame(features)
    link.set_filter(DataFilter([CategoryFilter("no_such_column", ("x",))]))

    with caplog.at_level("INFO", logger="spacr.qt.screens.pca"):
        got = screen._filtered()

    assert len(got) == len(features)
    assert "does not apply to this table" in caplog.text


def test_a_filter_change_with_no_table_starts_no_timer(screen, link):
    """Nothing to recompute means nothing is scheduled."""
    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))

    assert not screen._refilter.isActive()


# ---------------------------------------------------------------------------
# loading a table
# ---------------------------------------------------------------------------

def test_choosing_a_table_loads_the_path_the_dialog_returned(
        screen, monkeypatch, synth_sqlite_db):
    """The file dialog's answer goes straight into the read."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(synth_sqlite_db), "")))

    screen.choose_table()

    assert screen._path == str(synth_sqlite_db)
    assert screen._table_picker.isVisibleTo(screen)
    assert "cell" in [screen._table_picker.itemText(i)
                      for i in range(screen._table_picker.count())]
    assert "measurements.db" in screen._source.text()


def test_cancelling_the_file_dialog_loads_nothing(screen, monkeypatch):
    """An empty path is a cancel, not a read of the empty string."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_table()

    assert screen._path is None
    assert screen._source.text() == "no table loaded"


def test_picking_a_table_before_any_path_does_nothing(screen):
    """Populating the picker must not fire a read of nothing."""
    screen._on_table_picked("cell")

    assert screen._frame is None


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------------

def test_exporting_before_a_pca_says_to_run_one(screen):
    """There is nothing to write, and the label says which thing is missing."""
    screen.export_csv()

    assert screen._source.text() == "Nothing to export — run a PCA first."


# ---------------------------------------------------------------------------
# handing a brushed cluster on
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------


def test_the_factory_builds_a_screen(qtbot):
    """The registry's factory takes the app key and returns the screen."""
    made = pca_screen.make_pca_screen("pca")
    qtbot.addWidget(made)

    assert isinstance(made, pca_screen.PCAScreen)
    made.close()


def test_a_link_that_refuses_to_disconnect_does_not_stop_the_close(qtbot,
                                                                   link):
    """Closing must free the workers even if the signal cannot be dropped."""
    import types

    made = pca_screen.PCAScreen(link=link, threaded=False)
    qtbot.addWidget(made)

    def refuse(_slot):
        raise TypeError("that slot is not connected")

    made._link = types.SimpleNamespace(
        filter_changed=types.SimpleNamespace(disconnect=refuse))
    made._refilter.start()

    made.close()                                   # must not raise

    assert not made._refilter.isActive()
    assert made.active_jobs() == 0


def test_the_screen_puts_no_row_in_the_registry():
    """PCA is a fold, so importing this module must add no tile.

    This asserted the module's one row was registered and that a second
    ``register()`` did not duplicate it. There is no row and no
    ``register()`` now: PCA is a button on Image UMAP's masthead, and
    the failure worth catching is the row coming back.
    """
    from spacr.qt.app import APPS

    assert not any(row[0] == pca_screen.APP_KEY for row in APPS)
    assert not hasattr(pca_screen, "register")
