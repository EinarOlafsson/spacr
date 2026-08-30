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
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

from spacr import schema  # noqa: E402
from spacr.qt.widgets import umap_explorer as UE  # noqa: E402
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer  # noqa: E402
from spacr.selection import OBJECT_KEY_COLUMNS  # noqa: E402


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


def test_key_columns_spelled_out_on_the_record_are_trusted_first():
    """When the record says it outright there is nothing to parse."""
    record = {c: f"v{i}" for i, c in enumerate(OBJECT_KEY_COLUMNS)}
    record[schema.PRCFO_KEY] = "something_else_entirely"
    assert UE._record_identity(record) == {
        c: f"v{i}" for i, c in enumerate(OBJECT_KEY_COLUMNS)}


def test_a_record_with_neither_columns_nor_a_prcfo_has_no_identity():
    """Reported as unidentifiable rather than keyed on ``'nan'``."""
    assert UE._record_identity({"image": "/tmp/x.png"}) is None
    assert UE._record_identity({schema.PRCFO_KEY: "  "}) is None


def test_a_prcfo_that_cannot_be_parsed_has_no_identity():
    """A malformed prcfo is a gap, not a key with odd parts."""
    assert UE._record_identity({schema.PRCFO_KEY: "not-a-prcfo"}) is None


def test_a_prcfo_is_rebuilt_into_the_object_key_columns():
    """``o7`` in a prcfo is ``7`` in every object table."""
    identity = UE._record_identity(
        {schema.PRCFO_KEY: "plate1_A_01_f1_o7"})
    assert identity is not None
    assert identity[schema.OBJECT_LABEL_KEY] == "7"
    assert identity[schema.PLATE_KEY] == "plate1"


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def test_an_embedding_that_is_not_two_dimensional_is_refused(explorer):
    """The explorer plots x against y; anything else is not an embedding."""
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        explorer.set_payload({"embedding": [[1.0, 2.0, 3.0]],
                              "labels": [0], "records": [{}]})


def test_arrays_of_different_lengths_are_refused(explorer, tmp_path):
    """One label per point, one record per point, one frame row per point."""
    payload, _db = _payload(tmp_path)
    payload["labels"] = payload["labels"][:2]
    with pytest.raises(ValueError, match="equal lengths"):
        explorer.set_payload(payload)

    payload, _db = _payload(tmp_path)
    payload["frame"] = pd.DataFrame({"area": [1.0, 2.0]})
    with pytest.raises(ValueError, match="equal lengths"):
        explorer.set_payload(payload)


def test_an_empty_payload_leaves_nothing_to_key_on(explorer):
    """No points means no point frame and no keys."""
    explorer.set_payload({"embedding": np.empty((0, 2)), "labels": [],
                          "records": []})
    assert explorer.point_keys() is None
    assert explorer._point_frame is None


def test_points_with_no_identity_cannot_be_keyed(explorer, tmp_path):
    """Without keys the explorer still draws, but publishes nothing."""
    payload, _db = _payload(tmp_path, identified=False)
    explorer.set_payload(payload)
    assert explorer.point_keys() is None


def test_identified_points_get_one_key_each(explorer, tmp_path):
    """This is what makes a lasso here the same selection as one elsewhere."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    keys = explorer.point_keys()
    assert keys is not None and len(keys) == 4


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


def test_a_new_payload_starts_with_nothing_selected(explorer, tmp_path):
    """A pick from the previous embedding names a different object now."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer._on_lasso([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    explorer.set_payload(payload)
    assert explorer._selected.tolist() == []
    assert explorer._picked is None


# ---------------------------------------------------------------------------
# the shared filter and the shared selection
# ---------------------------------------------------------------------------

def test_a_filter_this_payload_cannot_honour_is_reported_not_applied(
        explorer, tmp_path):
    """Drawing everything as if the filter had applied would be a lie."""
    from spacr.selection import DataFilter, RangeFilter

    payload, _db = _payload(tmp_path, identified=False)
    explorer.set_payload(payload)
    explorer.link.set_filter(
        DataFilter().add(RangeFilter("area", low=10.0)))
    explorer._recompute_visible_points()
    assert explorer._point_visible.all()
    assert explorer._filter_note


def test_no_points_means_no_filtering_to_do(explorer):
    """An empty explorer has nothing to dim."""
    explorer.set_payload({"embedding": np.empty((0, 2)), "labels": [],
                          "records": []})
    explorer._recompute_visible_points()
    assert len(explorer._point_visible) == 0
    assert explorer._filter_note == ""


def test_a_link_whose_filter_cannot_be_read_leaves_every_point_visible(
        explorer, tmp_path, monkeypatch):
    """A dead link is not a filter that excludes everything."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)

    class DeadLink:
        @property
        def filter(self):
            raise RuntimeError("the link's C++ side is gone")

        @property
        def selection(self):
            raise RuntimeError("the link's C++ side is gone")

    monkeypatch.setattr(type(explorer), "link",
                        property(lambda self: DeadLink()))
    explorer._recompute_visible_points()
    assert explorer._point_visible.all()
    explorer._recompute_linked_points()
    assert len(explorer._linked_points) == 0


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------

def test_a_click_outside_the_axes_picks_nothing(explorer, tmp_path):
    """Only a click on the scatter is a request to preview a point."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)

    class Event:
        inaxes = None
        xdata = None
        ydata = None
        button = "up"

    explorer._on_click(Event())
    assert explorer._picked is None
    explorer._on_scroll(Event())


def test_a_click_on_the_scatter_previews_the_nearest_point(explorer, tmp_path):
    """Nearest in scaled space, so a stretched axis does not decide it."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)

    class Event:
        def __init__(self, x, y, axes):
            self.inaxes = axes
            self.xdata = x
            self.ydata = y
            self.button = "up"

    explorer._on_click(Event(2.1, 2.05, explorer._axes))
    assert explorer._picked in (2, 3)
    assert not explorer._preview.pixmap().isNull()
    assert explorer._preview.source_pixmap().width() > 0


def test_a_point_index_outside_the_records_previews_nothing(explorer,
                                                            tmp_path):
    """An index from a stale figure must not walk off the record list."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer.show_point(99)
    assert explorer._picked is None


def test_scrolling_over_the_scatter_zooms_around_the_pointer(explorer,
                                                             tmp_path):
    """Zoom is about the cursor, so the point under it stays under it."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    before = explorer._axes.get_xlim()

    class Event:
        def __init__(self, axes):
            self.inaxes = axes
            self.xdata = 1.0
            self.ydata = 1.0
            self.button = "up"

    explorer._on_scroll(Event(explorer._axes))
    after = explorer._axes.get_xlim()
    assert (after[1] - after[0]) < (before[1] - before[0])


def test_choosing_no_cluster_selects_nothing(explorer, tmp_path):
    """The "all clusters" entry carries no label to select on."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer._cluster_box.setCurrentIndex(0)
    explorer._select_cluster(0)
    assert len(explorer._selected) == 0


# ---------------------------------------------------------------------------
# writing annotations
# ---------------------------------------------------------------------------

def test_writing_with_nothing_selected_asks_for_a_selection(explorer,
                                                            tmp_path):
    """The status line says what to do, rather than writing zero rows."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    explorer._write_selected()
    assert "Draw a lasso or select a cluster first." in explorer._status.text()


def test_a_second_write_while_one_is_running_is_refused(explorer, tmp_path):
    """Two writers on one sqlite file is how a plate gets half-annotated."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)

    class RunningWorker:
        @staticmethod
        def isRunning():
            return True

    explorer._worker = RunningWorker()
    try:
        explorer._start_write(payload["records"], [1] * 4, "manual selection")
        assert "already running" in explorer._status.text()
    finally:
        explorer._worker = None


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


def test_closing_with_a_worker_running_waits_for_it(explorer, tmp_path):
    """Qt aborts the process if a running QThread is destroyed."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    stopped = []

    class SlowWorker:
        @staticmethod
        def requestInterruption():
            stopped.append("interrupted")

        @staticmethod
        def wait():
            stopped.append("waited")

    explorer._worker = SlowWorker()
    explorer.close()
    assert stopped == ["interrupted", "waited"]
    assert explorer._worker is None


# ---------------------------------------------------------------------------
# points that are only half identified
# ---------------------------------------------------------------------------

def test_a_payload_that_names_only_some_of_its_points_names_none(
        explorer, tmp_path):
    """Half a lasso published as the whole of it is the worse answer."""
    payload, _db = _payload(tmp_path)
    payload["records"][0].pop(schema.PRCFO_KEY)
    explorer.set_payload(payload)
    assert explorer.point_keys() is None


def test_an_attached_frame_without_key_columns_can_still_be_filtered(
        explorer, tmp_path):
    """The frame is kept for the filter even when no key can be built."""
    payload, _db = _payload(tmp_path, identified=False)
    payload["frame"] = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0]})
    explorer.set_payload(payload)
    assert explorer._point_frame is not None
    assert "area" in explorer._point_frame.columns
    assert explorer.point_keys() is None


def test_keys_that_cannot_be_built_leave_the_points_unidentified(
        explorer, tmp_path, monkeypatch):
    """A key builder that raises is "unidentifiable", not a crash on load."""
    monkeypatch.setattr(UE, "object_keys",
                        lambda table: (_ for _ in ()).throw(
                            ValueError("a key column holds a list")))
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer.point_keys() is None
    assert explorer._point_frame is not None


# ---------------------------------------------------------------------------
# drawing before there is anything to draw
# ---------------------------------------------------------------------------

def test_opacity_and_highlights_are_no_ops_before_a_payload(explorer):
    """Every redraw path is safe to call on an explorer with no figure yet."""
    assert explorer._scatter is None
    explorer._apply_point_alpha()
    explorer._draw_linked_points()


def test_a_colour_that_is_not_a_colour_falls_back_to_the_cluster_map(
        explorer, tmp_path):
    """An unusable colour must not leave the scatter unpainted."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer.apply_display({"point_color": "not-a-colour"}) is True
    assert explorer._scatter is not None
    assert explorer._scatter.get_array() is not None


def test_a_display_change_that_only_the_next_run_can_use_is_not_applied(
        explorer, tmp_path):
    """Silently ignoring it would be worse than saying it needs a re-run."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer.apply_display({"figuresize": 12}) is False
    assert "figuresize" not in explorer.display_settings()
    assert explorer.apply_display({}) is False
    assert explorer.apply_display({"point_size": None}) is False


def test_the_deferred_draw_does_nothing_when_nothing_is_pending(explorer,
                                                                tmp_path):
    """The owned timer can fire after the draw it was queued for happened."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    canvas = explorer._canvas
    canvas._draw_pending = False
    canvas._spacr_draw()
    assert canvas._draw_pending is False


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


# ---------------------------------------------------------------------------
# publishing, and shutting down
# ---------------------------------------------------------------------------

def test_a_publish_that_fails_does_not_take_the_lasso_with_it(
        explorer, tmp_path, monkeypatch):
    """The lasso still selected what it selected, whatever the link did."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    monkeypatch.setattr(type(explorer), "publish_selection",
                        lambda self, keys: (_ for _ in ()).throw(
                            RuntimeError("the link's C++ side is gone")))
    explorer._on_lasso([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    assert explorer._selected.tolist() == [0, 1]


def test_closing_after_the_link_has_gone_still_closes(explorer, tmp_path,
                                                      monkeypatch):
    """At interpreter teardown the link's C++ side may already be deleted.

    The unlink is the FIRST thing the close does, and the rest of the
    teardown is what matters: a close that stopped there would leave the
    lasso connected to a canvas Qt is about to delete, which is the crash
    this teardown exists to prevent and which raises nothing here.
    """
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer._lasso is not None

    monkeypatch.setattr(type(explorer), "unlink_selection",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("wrapped C/C++ object deleted")))
    explorer.close()
    assert explorer._lasso is None
    assert explorer.isVisible() is False


def test_a_real_colour_paints_every_point_the_same(explorer, tmp_path):
    """A named colour is honoured rather than falling back to the cluster map."""
    payload, _db = _payload(tmp_path)
    explorer.set_payload(payload)
    assert explorer.apply_display({"point_color": "#ff0000"}) is True
    assert explorer._scatter.get_array() is None


def test_a_payload_may_carry_the_display_settings_it_was_drawn_with(
        explorer, tmp_path):
    """A saved run reopens at the sizes it was read at."""
    payload, _db = _payload(tmp_path)
    payload["display"] = {"point_size": 55, "canvas_width": 700,
                          "sidebar_width": 300, "point_alpha": None}
    explorer.set_payload(payload)
    settings = explorer.display_settings()
    assert settings["point_size"] == 55
    assert settings["canvas_width"] == 700
    assert settings["point_alpha"] == 0.65
