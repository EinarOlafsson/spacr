"""Identity, filtering, picking, writing and display in the Image UMAP explorer.

A lasso here has to name the same rows a selection in the database browser
names, which is a question of whether each point could be given an object
key; when it could not, a filter must be reported as ignored, never applied by
drawing everything.  Every case drives a real explorer with real crops and a
real ``png_list`` table, walking a case table with pytest's ``subtests``.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("matplotlib")
pytest.importorskip("PySide6")

from spacr import schema  # noqa: E402
from spacr.qt.widgets import umap_explorer as UE  # noqa: E402
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer  # noqa: E402
from spacr.selection import OBJECT_KEY_COLUMNS  # noqa: E402

_LASSO = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


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
    return {"embedding": embedding[:count],
            "labels": np.array([0, 0, 1, 1])[:count],
            "records": records}


def _explorer(qtbot, payload=None) -> ImageUmapExplorer:
    from spacr.qt.linked_selection import LinkedSelection

    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)
    # A private link, so a lasso here reaches no other test's views.
    widget.link_selection("umap", link=LinkedSelection())
    if payload is not None:
        widget.set_payload(payload)
    return widget


class _Event:
    def __init__(self, x, y, axes, button="up"):
        self.inaxes, self.xdata, self.ydata, self.button = axes, x, y, button


def _empty_payload():
    return {"embedding": np.empty((0, 2)), "labels": [], "records": []}


# ---------------------------------------------------------------------------
# a point's identity
# ---------------------------------------------------------------------------

def test_a_record_identity_comes_from_its_key_columns_or_its_prcfo(subtests):
    """Spelled-out columns win; a prcfo is parsed; anything else is None."""
    spelled = {c: f"v{i}" for i, c in enumerate(OBJECT_KEY_COLUMNS)}
    cases = [
        ("key columns trusted first",
         {**spelled, schema.PRCFO_KEY: "something_else_entirely"}, spelled),
        ("a prcfo rebuilt into key columns",
         {schema.PRCFO_KEY: "plate1_A_01_f1_o7"},
         {schema.PLATE_KEY: "plate1", schema.ROW_KEY: "A",
          schema.COLUMN_KEY: "01", schema.FIELD_KEY: "f1",
          schema.OBJECT_LABEL_KEY: "7"}),
        ("neither columns nor prcfo", {"image": "/tmp/x.png"}, None),
        ("a blank prcfo", {schema.PRCFO_KEY: "  "}, None),
        ("a prcfo that cannot be parsed", {schema.PRCFO_KEY: "not-a-prcfo"},
         None),
    ]
    for case_id, record, expected in cases:
        with subtests.test(msg=case_id):
            assert UE._record_identity(record) == expected


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def test_a_payload_is_checked_keyed_and_starts_unselected(qtbot, tmp_path,
                                                          monkeypatch,
                                                          subtests):
    """Shapes are refused by message; keys are all-or-nothing."""
    with subtests.test(msg="an embedding that is not two-dimensional"):
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            _explorer(qtbot).set_payload({"embedding": [[1.0, 2.0, 3.0]],
                                          "labels": [0], "records": [{}]})

    for case_id, edit in [
            ("labels of another length",
             lambda p: p.update(labels=p["labels"][:2])),
            ("a frame of another length",
             lambda p: p.update(frame=pd.DataFrame({"area": [1.0, 2.0]})))]:
        with subtests.test(msg=case_id):
            payload = _payload(tmp_path)
            edit(payload)
            with pytest.raises(ValueError, match="equal lengths"):
                _explorer(qtbot).set_payload(payload)

    with subtests.test(msg="an empty payload has nothing to key on"):
        explorer = _explorer(qtbot, _empty_payload())
        assert explorer.point_keys() is None
        assert explorer._point_frame is None

    with subtests.test(msg="identified points get one key each"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        keys = explorer.point_keys()
        assert len(keys) == 4
        assert len(set(map(str, keys))) == 4

    def _half_named(payload):
        payload["records"][0].pop(schema.PRCFO_KEY)

    for case_id, identified, edit in [
            ("points with no identity", False, None),
            ("only some points named", True, _half_named)]:
        with subtests.test(msg=f"{case_id} are not keyed"):
            payload = _payload(tmp_path, identified=identified)
            if edit:
                edit(payload)
            assert _explorer(qtbot, payload).point_keys() is None

    with subtests.test(msg="a frame without key columns is still kept"):
        payload = _payload(tmp_path, identified=False)
        payload["frame"] = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0]})
        explorer = _explorer(qtbot, payload)
        assert list(explorer._point_frame["area"]) == [1.0, 2.0, 3.0, 4.0]
        assert explorer.point_keys() is None

    with subtests.test(msg="a key builder that raises leaves no keys"):
        monkeypatch.setattr(UE, "object_keys",
                            lambda table: (_ for _ in ()).throw(
                                ValueError("a key column holds a list")))
        explorer = _explorer(qtbot, _payload(tmp_path))
        assert explorer.point_keys() is None
        assert len(explorer._point_frame) == 4
        monkeypatch.undo()

    with subtests.test(msg="a new payload starts with nothing selected"):
        payload = _payload(tmp_path)
        explorer = _explorer(qtbot, payload)
        explorer._on_lasso(_LASSO)
        assert explorer._selected.tolist() == [0, 1]
        explorer.set_payload(payload)
        assert explorer._selected.tolist() == []
        assert explorer._picked is None


# ---------------------------------------------------------------------------
# the shared filter and the shared selection
# ---------------------------------------------------------------------------

def test_a_filter_is_applied_reported_or_ignored(qtbot, tmp_path, monkeypatch,
                                                 subtests):
    """A filter it cannot honour is named as ignored; a dead link hides none."""
    from spacr.selection import DataFilter, RangeFilter

    with subtests.test(msg="no identities: the filter is ignored and said"):
        explorer = _explorer(qtbot, _payload(tmp_path, identified=False))
        data_filter = DataFilter().add(RangeFilter("area", low=10.0))
        explorer.link.set_filter(data_filter)
        explorer._recompute_visible_points()
        assert explorer._point_visible.tolist() == [True] * 4
        assert explorer._filter_note == (
            f" · filter ignored ({data_filter.describe()}): this embedding "
            "carries no identities")

    with subtests.test(msg="no points means nothing to filter"):
        explorer = _explorer(qtbot, _empty_payload())
        explorer._recompute_visible_points()
        assert len(explorer._point_visible) == 0
        assert explorer._filter_note == ""

    with subtests.test(msg="a dead link leaves every point visible"):
        explorer = _explorer(qtbot, _payload(tmp_path))

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
        assert explorer._point_visible.tolist() == [True] * 4
        explorer._recompute_linked_points()
        assert len(explorer._linked_points) == 0
        monkeypatch.undo()

    with subtests.test(msg="a publish that fails keeps the lasso"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        monkeypatch.setattr(type(explorer), "publish_selection",
                            lambda self, keys: (_ for _ in ()).throw(
                                RuntimeError("the link's C++ side is gone")))
        explorer._on_lasso(_LASSO)
        assert explorer._selected.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------

def test_the_canvas_previews_zooms_and_selects_clusters(qtbot, tmp_path,
                                                        subtests):
    """Only events on the scatter count; zoom is about the pointer."""
    with subtests.test(msg="a click or scroll outside the axes does nothing"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        before = explorer._axes.get_xlim()
        explorer._on_click(_Event(None, None, None))
        explorer._on_scroll(_Event(None, None, None))
        assert explorer._picked is None
        assert explorer._axes.get_xlim() == before

    with subtests.test(msg="a click previews the nearest point"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        explorer._on_click(_Event(2.2, 2.1, explorer._axes))
        assert explorer._picked == 3
        assert explorer._preview.source_pixmap().width() == 16

    with subtests.test(msg="scrolling up zooms in by 0.8 about the pointer"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        x0, x1 = explorer._axes.get_xlim()
        explorer._on_scroll(_Event(1.0, 1.0, explorer._axes))
        assert explorer._axes.get_xlim() == pytest.approx(
            (1.0 - (1.0 - x0) * 0.8, 1.0 + (x1 - 1.0) * 0.8))

    with subtests.test(msg="choosing no cluster selects nothing"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        explorer._cluster_box.setCurrentIndex(0)
        explorer._select_cluster(0)
        assert len(explorer._selected) == 0

    with subtests.test(msg="the deferred draw does nothing when not pending"):
        canvas = _explorer(qtbot, _payload(tmp_path))._canvas
        canvas._draw_pending = False
        canvas._spacr_draw()
        assert canvas._draw_pending is False


# ---------------------------------------------------------------------------
# writing annotations and shutting down
# ---------------------------------------------------------------------------

def test_writing_and_closing_never_leave_a_worker_or_lasso_behind(
        qtbot, tmp_path, monkeypatch, subtests):
    """A refused write says why; close interrupts, waits and disconnects."""
    with subtests.test(msg="writing with nothing selected"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        explorer._write_selected()
        assert explorer._status.text() == (
            "Draw a lasso or select a cluster first.")

    with subtests.test(msg="a second write while one runs is refused"):
        payload = _payload(tmp_path)
        explorer = _explorer(qtbot, payload)

        class RunningWorker:
            @staticmethod
            def isRunning():
                return True

        explorer._worker = RunningWorker()
        try:
            explorer._start_write(payload["records"], [1] * 4,
                                  "manual selection")
            assert explorer._status.text() == (
                "An annotation write is already running.")
        finally:
            explorer._worker = None

    with subtests.test(msg="closing with a worker running waits for it"):
        explorer = _explorer(qtbot, _payload(tmp_path))
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

    with subtests.test(msg="a second unlink after a close is a no-op"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        assert explorer.is_linked is True
        explorer.close()
        assert explorer.is_linked is False
        explorer.unlink_selection()
        assert explorer.is_linked is False

    with subtests.test(msg="closing after the link has gone still closes"):
        explorer = _explorer(qtbot, _payload(tmp_path))
        assert explorer._lasso is not None
        monkeypatch.setattr(type(explorer), "unlink_selection",
                            lambda self: (_ for _ in ()).throw(
                                RuntimeError("wrapped C/C++ object deleted")))
        explorer.close()
        assert explorer._lasso is None
        assert explorer.isVisible() is False


# ---------------------------------------------------------------------------
# display settings
# ---------------------------------------------------------------------------

def test_display_settings_colour_the_points_and_ride_with_the_payload(
        qtbot, tmp_path, subtests):
    """A real colour paints flat, junk falls back, saved sizes are reopened."""
    for case_id, colour, cluster_mapped in [
            ("a real colour paints every point the same", "#ff0000", False),
            ("a colour that is not one falls back to clusters",
             "not-a-colour", True)]:
        with subtests.test(msg=case_id):
            explorer = _explorer(qtbot, _payload(tmp_path))
            assert explorer.apply_display({"point_color": colour}) is True
            array = explorer._scatter.get_array()
            if cluster_mapped:
                assert list(array) == [0, 0, 1, 1]
            else:
                assert array is None

    with subtests.test(msg="a payload carries its display settings"):
        payload = _payload(tmp_path)
        payload["display"] = {"point_size": 55, "canvas_width": 700,
                              "sidebar_width": 300, "point_alpha": None}
        settings = _explorer(qtbot, payload).display_settings()
        assert (settings["point_size"], settings["canvas_width"],
                settings["sidebar_width"], settings["point_alpha"]) == (
            55, 700, 300, 0.65)
