"""Identity, filtering and the annotation write in the Image UMAP explorer.

A lasso here has to name the same rows a selection in the database browser
names, which is entirely a question of whether each point could be given an
object key. When it could not, the explorer has to say so -- a filter it
cannot honour must be reported as ignored, never applied by drawing
everything. The write to ``png_list`` runs on a thread for the same reason:
a screen frozen on SQLite is a screen nobody trusts with a plate.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

from spacr import schema  # noqa: E402
from spacr.qt.widgets import umap_explorer as UE  # noqa: E402
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer  # noqa: E402


def _payload(tmp_path, *, identified=True, count=4):
    """A four-point payload with real crops and a real ``png_list`` table."""
    database = tmp_path / "measurements.db"
    paths = []
    for index in range(count):
        path = tmp_path / f"object_{index}.png"
        Image.new("RGB", (16, 16), (index * 40, 80, 200)).save(path)
        paths.append(path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS png_list "
            "(png_path TEXT PRIMARY KEY, plateID TEXT)")
        connection.execute("DELETE FROM png_list")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, 'plate1')",
            [(str(p),) for p in paths])

    records = []
    for index, path in enumerate(paths):
        record = {"image": path, "display_name": path.name,
                  "db_path": database, "db_png_path": str(path)}
        if identified:
            record[schema.PRCFO_KEY] = f"plate1_A_01_f1_o{index + 1}"
        records.append(record)
    embedding = np.array([[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1]])
    return {
        "embedding": embedding[:count],
        "labels": np.array([0, 0, 1, 1])[:count],
        "records": records,
    }, database


@pytest.fixture
def explorer(qtbot):
    from spacr.qt.linked_selection import LinkedSelection

    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)
    # A private link, so a lasso drawn here does not reach the views another
    # test left subscribed to the process-wide one.
    widget.link_selection("umap", link=LinkedSelection())
    return widget


# ---------------------------------------------------------------------------
# giving a point an identity
# ---------------------------------------------------------------------------

def test_a_gap_is_not_an_identity_token():
    """``None``, NaN and blanks all become the literal key ``'nan'`` if kept."""
    assert UE._usable(None) is False
    assert UE._usable(float("nan")) is False
    assert UE._usable("   ") is False
    assert UE._usable("plate1") is True


def test_a_value_pandas_cannot_judge_is_not_usable():
    """``pd.isna`` raises on a container; that is not an identity either."""
    assert UE._usable(["plate1", "plate2"]) is False
    assert UE._usable(np.array(["plate1", "plate2"])) is False


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def test_a_redraw_keeps_the_selection_and_the_pick_highlighted(explorer,
                                                               tmp_path):
    """Changing a display setting rebuilds the artists; both marks come back."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer._on_lasso([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    explorer.show_point(1)
    assert explorer._selected.tolist() == [0, 1]

    assert explorer.apply_display({"point_size": 44}) is True
    assert explorer._selection_artist.get_offsets().shape[0] == 2
    assert explorer._picked_artist.get_offsets().shape[0] == 1


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------


def test_a_point_index_outside_the_records_previews_nothing(explorer,
                                                            tmp_path):
    """An index from a stale figure must not walk off the record list."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer.show_point(99)
    assert explorer._picked is None


# ---------------------------------------------------------------------------
# writing annotations
# ---------------------------------------------------------------------------


def test_a_failed_write_is_reported_against_the_database(explorer):
    """The database's message reaches the status line rather than a log only."""
    explorer._on_write_done(0, 4, "attempt to write a readonly database")
    assert explorer._status.text() == (
        "Database write failed: attempt to write a readonly database")


def test_the_worker_reports_a_write_that_raised(qtbot, monkeypatch):
    """A worker that raised must still emit, or the screen waits forever."""
    monkeypatch.setattr(
        UE, "write_umap_annotations",
        lambda records, values, column: (_ for _ in ()).throw(
            sqlite3.OperationalError("no such table: png_list")))
    worker = UE._AnnotationWorker([{"a": 1}, {"a": 2}], [1, 2], "col")
    seen = []
    worker.finished_result.connect(
        lambda updated, skipped, error: seen.append((updated, skipped, error)))
    worker.run()
    assert seen == [(0, 2, "no such table: png_list")]


def test_the_worker_reports_what_it_wrote(qtbot, monkeypatch):
    """The success path carries both counts back to the GUI thread."""
    monkeypatch.setattr(UE, "write_umap_annotations",
                        lambda records, values, column: (3, 1))
    worker = UE._AnnotationWorker([{"a": 1}], [1], "col")
    seen = []
    worker.finished_result.connect(
        lambda updated, skipped, error: seen.append((updated, skipped, error)))
    worker.run()
    assert seen == [(3, 1, "")]


# ---------------------------------------------------------------------------
# drawing before there is anything to draw
# ---------------------------------------------------------------------------

def test_opacity_and_highlights_are_no_ops_before_a_payload(explorer):
    """Every redraw path is safe to call on an explorer with no figure yet."""
    assert explorer._scatter is None
    explorer._apply_point_alpha()
    explorer._draw_linked_points()


def test_a_display_change_that_only_the_next_run_can_use_is_not_applied(
        explorer, tmp_path):
    """Silently ignoring it would be worse than saying it needs a re-run."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer.apply_display({"figuresize": 12}) is False
    assert "figuresize" not in explorer.display_settings()
    assert explorer.apply_display({}) is False
    assert explorer.apply_display({"point_size": None}) is False


# ---------------------------------------------------------------------------
# the display-settings window
# ---------------------------------------------------------------------------

def test_a_cancelled_display_dialog_changes_nothing(explorer, tmp_path,
                                                    monkeypatch):
    """Nothing is applied and nothing is propagated when the user backs out."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    before = explorer.display_settings()
    monkeypatch.setattr(UE.UmapDisplaySettings, "exec", lambda self: 0)
    explorer.open_display_settings()
    assert explorer.display_settings() == before


def test_the_display_dialog_applies_the_live_half_and_propagates_all_of_it(
        explorer, tmp_path, monkeypatch):
    """The propagate seam is what saves the not-live half with the run."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)

    monkeypatch.setattr(UE.UmapDisplaySettings, "exec", lambda self: 1)
    monkeypatch.setattr(UE.UmapDisplaySettings, "live_values",
                        lambda self: {"point_size": 40})
    monkeypatch.setattr(UE.UmapDisplaySettings, "values",
                        lambda self: {"point_size": 40, "figuresize": 9})
    pushed = []
    explorer.set_propagate_callback(pushed.append)
    explorer.open_display_settings()
    assert explorer.display_settings()["point_size"] == 40
    assert pushed == [{"dot_size": 40, "figuresize": 9}]
    assert explorer._status.text() == "Display updated."


def test_a_settings_getter_that_raises_does_not_stop_the_dialog(
        explorer, tmp_path, monkeypatch):
    """The seed is a convenience; losing it must not cost the window."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer._settings_getter = lambda: (_ for _ in ()).throw(
        RuntimeError("the settings panel is gone"))
    explorer.set_propagate_callback(
        lambda values: (_ for _ in ()).throw(RuntimeError("and so is that")))
    monkeypatch.setattr(UE.UmapDisplaySettings, "exec", lambda self: 1)
    monkeypatch.setattr(UE.UmapDisplaySettings, "live_values",
                        lambda self: {})
    explorer.open_display_settings()
    assert "next run" in explorer._status.text()
