"""CPU coverage for the database / persistence half of
``spacr.gui_elements.AnnotateApp``.

Everything here runs headless: instead of building a real Tk window we
allocate the instance with ``AnnotateApp.__new__`` and populate exactly the
attributes the database methods touch, stubbing out the two UI callbacks
(``load_images`` / ``prefilter_paths_annotations``) with recorders. The
SQLite side is real -- every assertion is made against rows that were
actually written to a temp ``png_list`` table.

Covered here: the background writer thread (``update_database_worker``),
``shutdown``, the three paging helpers, ``train_and_classify``,
``_get_png_list_columns``, ``_parse_field_value``,
``convert_settings_dict_for_gui``, ``build_multi_annotation`` and
``ensure_multi_annot_from_selection``.
"""
from __future__ import annotations

import os
import queue
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a matplotlib window survive a test in this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# tiny fakes standing in for the Tk widgets the DB code pokes at
# ---------------------------------------------------------------------------

class _FakeLabel:
    """Stand-in for the status ``tk.Label``."""

    def __init__(self):
        self.configs = []

    def config(self, **kwargs):
        self.configs.append(kwargs)


class _FakeRoot:
    """Stand-in for the Tk root/Toplevel used by the DB helpers."""

    def __init__(self, quit_exc=None, destroy_exc=None):
        self.updates = 0
        self.quit_calls = 0
        self.destroy_calls = 0
        self._quit_exc = quit_exc
        self._destroy_exc = destroy_exc

    def update(self):
        self.updates += 1

    def quit(self):
        self.quit_calls += 1
        if self._quit_exc:
            raise self._quit_exc

    def destroy(self):
        self.destroy_calls += 1
        if self._destroy_exc:
            raise self._destroy_exc


class _FakeQueue:
    """A Queue whose ``join`` returns immediately (no worker required)."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def join(self):
        return None


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _build_png_db(tmp_path, paths, annotations=None, extra_columns=(),
                  name="measurements.db"):
    """Create a temp sqlite DB holding a ``png_list`` table."""
    db = tmp_path / name
    cols = ['"png_path" TEXT', '"annotate" INTEGER'] + list(extra_columns)
    con = sqlite3.connect(str(db))
    try:
        con.execute(f'CREATE TABLE "png_list" ({", ".join(cols)})')
        if annotations is None:
            annotations = [None] * len(paths)
        con.executemany(
            'INSERT INTO "png_list" ("png_path", "annotate") VALUES (?, ?)',
            list(zip(paths, annotations)),
        )
        con.commit()
    finally:
        con.close()
    return str(db)


def _fetch(db_path, sql, params=()):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def _app(db_path, **overrides):
    """Allocate an AnnotateApp with only the DB-relevant state populated."""
    from spacr.gui_elements import AnnotateApp

    app = AnnotateApp.__new__(AnnotateApp)
    app.db_path = str(db_path)
    app.src = os.path.dirname(str(db_path))
    app.index = 0
    app.SENTINEL = object()
    app.annotation_column = "annotate"
    app.orig_annotation_columns = "annotate"
    app.image_type = None
    app.channels = None
    app.image_size = (50, 50)
    app.percentiles = (1, 99)
    app.grid_rows = 1
    app.grid_cols = 3
    app.measurement = None
    app.threshold = None
    app.threshold_direction = "higher"
    app.pending_updates = {}
    app.images = {}
    app.labels = []
    app.adjusted_to_original_paths = {}
    app.terminate = False
    app.update_queue = queue.Queue()
    app.worker_busy = False
    app._unsaved_batches = 0
    app._batch_lock = threading.Lock()
    app._last_save_ts = None
    app.filtered_paths_annotations = []
    app._total_filtered = 0
    app.root = _FakeRoot()
    app.status_label = _FakeLabel()

    app.calls = []
    app.load_images = lambda: app.calls.append("load_images")
    app.prefilter_paths_annotations = lambda: app.calls.append("prefilter")
    app.update_gui_text = lambda text: app.calls.append(("gui_text", text))

    for key, value in overrides.items():
        setattr(app, key, value)
    return app


def _paths(tmp_path, n, prefix="cell"):
    return [str(tmp_path / f"{prefix}_{i}.png") for i in range(n)]


# ===========================================================================
# update_database_worker
# ===========================================================================

def test_worker_writes_set_and_null_batches(tmp_path):
    """One queued batch is committed: ints are written, None nulls the cell."""
    paths = _paths(tmp_path, 3)
    db = _build_png_db(tmp_path, paths, annotations=[7, 7, 7])
    app = _app(db)
    app._unsaved_batches = 1
    app.update_queue.put({paths[0]: 1, paths[1]: 2, paths[2]: None})
    app.update_queue.put(app.SENTINEL)

    app.update_database_worker()

    rows = dict(_fetch(db, 'SELECT png_path, annotate FROM png_list'))
    assert rows[paths[0]] == 1
    assert rows[paths[1]] == 2
    assert rows[paths[2]] is None
    assert app.worker_busy is False
    assert app._unsaved_batches == 0
    assert isinstance(app._last_save_ts, float) and app._last_save_ts > 0


def test_worker_coalesces_multiple_batches_into_one_commit(tmp_path):
    """Two queued dicts are merged; each coalesced batch decrements the counter."""
    paths = _paths(tmp_path, 3)
    db = _build_png_db(tmp_path, paths)
    app = _app(db, terminate=True)
    app._unsaved_batches = 3
    app.update_queue.put({paths[0]: 1})
    app.update_queue.put({paths[1]: 2})
    app.update_queue.put({paths[2]: 1})

    app.update_database_worker()

    rows = dict(_fetch(db, 'SELECT png_path, annotate FROM png_list'))
    assert [rows[p] for p in paths] == [1, 2, 1]
    # 2 coalesced decrements + 1 in the finally block
    assert app._unsaved_batches == 0


def test_worker_skips_empty_batch_without_touching_db(tmp_path):
    """An empty dict is acked and discarded; no UPDATE is issued."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(tmp_path, paths, annotations=[5, 6])
    app = _app(db, terminate=True)
    app._unsaved_batches = 4
    app.update_queue.put({})

    app.update_database_worker()

    assert [r[0] for r in _fetch(db, 'SELECT annotate FROM png_list')] == [5, 6]
    # the `continue` path never reaches the finally-block decrement
    assert app._unsaved_batches == 4
    assert app.worker_busy is False


def test_worker_quotes_annotation_column_with_embedded_quote(tmp_path):
    """A column name containing a double quote is escaped, not injected."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(tmp_path, paths)
    con = sqlite3.connect(db)
    try:
        con.execute('ALTER TABLE "png_list" ADD COLUMN "we""ird" INTEGER')
        con.commit()
    finally:
        con.close()

    app = _app(db, annotation_column='we"ird', terminate=True)
    app.update_queue.put({paths[0]: 3})
    app.update_database_worker()

    got = _fetch(db, 'SELECT "we""ird" FROM png_list WHERE png_path = ?', (paths[0],))
    assert got == [(3,)]


def test_worker_swallows_cursor_close_failure(tmp_path, monkeypatch):
    """A cursor whose close() raises must not break worker teardown."""
    paths = _paths(tmp_path, 1)
    db = _build_png_db(tmp_path, paths)

    made = []

    class _Cur:
        def __init__(self, cur):
            self._cur = cur

        def __getattr__(self, name):
            return getattr(self._cur, name)

        def close(self):
            raise sqlite3.ProgrammingError("cursor close boom")

    class _Conn:
        def __init__(self, conn):
            self._conn = conn
            self.closed = False

        def cursor(self):
            return _Cur(self._conn.cursor())

        def commit(self):
            self._conn.commit()

        def close(self):
            self.closed = True
            self._conn.close()

    real_connect = sqlite3.connect

    def fake_connect(path, *args, **kwargs):
        proxy = _Conn(real_connect(path, *args, **kwargs))
        made.append(proxy)
        return proxy

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    app = _app(db, terminate=True)
    app.update_queue.put({paths[0]: 1})

    app.update_database_worker()  # must not raise

    monkeypatch.undo()
    assert made and made[0].closed is True
    assert _fetch(db, 'SELECT annotate FROM png_list') == [(1,)]


# ===========================================================================
# shutdown
# ===========================================================================

def test_shutdown_aborts_when_user_declines(tmp_path, monkeypatch):
    """Answering 'no' to the unsaved-work prompt leaves the app running."""
    import tkinter.messagebox as messagebox

    db = _build_png_db(tmp_path, _paths(tmp_path, 1))
    app = _app(db)
    app._unsaved_batches = 2
    app.db_update_thread = None

    asked = []
    monkeypatch.setattr(messagebox, "askyesno",
                        lambda *a, **k: (asked.append(a), False)[1])

    assert app.shutdown() is None
    assert asked, "the confirmation dialog was never shown"
    assert app.terminate is False
    assert app.root.quit_calls == 0
    assert app.update_queue.qsize() == 0


def test_shutdown_flushes_pending_and_joins_worker(tmp_path, monkeypatch):
    """Confirmed shutdown flushes pending edits through a live worker thread."""
    import tkinter.messagebox as messagebox

    paths = _paths(tmp_path, 2)
    db = _build_png_db(tmp_path, paths)
    app = _app(db)
    app.pending_updates = {paths[0]: 1, paths[1]: 2}
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)

    app.db_update_thread = threading.Thread(target=app.update_database_worker,
                                            daemon=True)
    app.db_update_thread.start()

    app.shutdown()

    assert app.terminate is True
    assert app.pending_updates == {}
    assert not app.db_update_thread.is_alive()
    assert app.root.quit_calls == 1
    assert app.root.destroy_calls == 1
    rows = dict(_fetch(db, 'SELECT png_path, annotate FROM png_list'))
    assert [rows[p] for p in paths] == [1, 2]


def test_shutdown_tolerates_broken_thread_and_destroy(tmp_path, capsys):
    """A thread without join() and a raising destroy() are both swallowed."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 1))
    app = _app(db)
    app.update_queue = _FakeQueue()
    app.db_update_thread = object()          # no .join -> AttributeError
    app.root = _FakeRoot(destroy_exc=RuntimeError("already destroyed"))

    app.shutdown()

    assert app.terminate is True
    assert app.update_queue.items == [app.SENTINEL]
    assert app.root.quit_calls == 1
    assert app.root.destroy_calls == 1
    assert "Quit application" in capsys.readouterr().out


# ===========================================================================
# skip_to_last_annotated
# ===========================================================================

def test_skip_to_last_annotated_jumps_to_containing_page(tmp_path):
    """The page holding the highest annotated row index becomes current."""
    paths = _paths(tmp_path, 10)
    anns = [None] * 10
    anns[3] = 2
    anns[5] = 0          # zero must NOT count as annotated
    anns[7] = 1
    db = _build_png_db(tmp_path, paths, annotations=anns)
    app = _app(db, grid_rows=1, grid_cols=3)

    app.skip_to_last_annotated()

    assert app.index == 6                      # (7 // 3) * 3
    assert [r[0] for r in app.filtered_paths_annotations] == paths[6:9]
    assert app.filtered_paths_annotations[1][1] == 1
    assert app.calls[-1] == "load_images"


def test_skip_to_last_annotated_honours_image_type_filter(tmp_path):
    """image_type restricts both the scan and the re-fetched page."""
    cells = _paths(tmp_path, 4, prefix="cell")
    nuclei = _paths(tmp_path, 4, prefix="nucleus")
    paths, anns = [], []
    for i in range(4):
        paths += [cells[i], nuclei[i]]
        anns += [None, 1]        # only nuclei annotated
    anns[2 * 3] = 2              # cell_3 annotated too (last cell row)
    db = _build_png_db(tmp_path, paths, annotations=anns)
    app = _app(db, grid_rows=1, grid_cols=2, image_type="cell")

    app.skip_to_last_annotated()

    # among the 4 'cell' rows only index 3 is annotated -> page (3//2)*2 = 2
    assert app.index == 2
    assert [r[0] for r in app.filtered_paths_annotations] == cells[2:4]
    assert all("cell" in p for p, _ in app.filtered_paths_annotations)


def test_skip_to_last_annotated_reports_when_nothing_annotated(tmp_path):
    """With no annotations the user is told and the grid is left alone."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 4))
    app = _app(db)

    app.skip_to_last_annotated()

    assert ("gui_text", "No annotated images found.") in app.calls
    assert "load_images" not in app.calls
    assert app.index == 0


def test_skip_to_last_annotated_flushes_pending_updates(tmp_path):
    """Pending edits are queued (and counted) before the scan runs."""
    paths = _paths(tmp_path, 4)
    db = _build_png_db(tmp_path, paths, annotations=[None, None, 1, None])
    app = _app(db, grid_rows=1, grid_cols=2)
    app.pending_updates = {paths[0]: 1}

    app.skip_to_last_annotated()

    assert app._unsaved_batches == 1
    assert app.update_queue.get_nowait() == {paths[0]: 1}
    assert app.pending_updates == {}
    assert app.index == 2


def test_skip_to_last_annotated_keeps_prefiltered_rows_in_measurement_mode(tmp_path):
    """With a measurement threshold active the page is not re-queried."""
    paths = _paths(tmp_path, 6)
    db = _build_png_db(tmp_path, paths, annotations=[None] * 5 + [1])
    preset = [("sentinel.png", None)]
    app = _app(db, grid_rows=1, grid_cols=2,
               measurement="cell_area", threshold=10,
               filtered_paths_annotations=preset)

    app.skip_to_last_annotated()

    assert app.index == 4                       # (5 // 2) * 2
    assert app.filtered_paths_annotations is preset
    assert app.calls[-1] == "load_images"


# ===========================================================================
# next_page / previous_page
# ===========================================================================

def test_next_page_advances_and_refetches(tmp_path):
    """next_page moves one page forward and pulls that page from the DB."""
    paths = _paths(tmp_path, 10)
    db = _build_png_db(tmp_path, paths, annotations=list(range(10)))
    app = _app(db, grid_rows=1, grid_cols=3, _total_filtered=10)

    app.next_page()

    assert app.index == 3
    assert [r[0] for r in app.filtered_paths_annotations] == paths[3:6]
    assert [r[1] for r in app.filtered_paths_annotations] == [3, 4, 5]
    assert app.calls == ["load_images"]


def test_next_page_clamps_at_last_page(tmp_path):
    """Paging past the end keeps the current index."""
    paths = _paths(tmp_path, 5)
    db = _build_png_db(tmp_path, paths)
    app = _app(db, grid_rows=1, grid_cols=3, index=3, _total_filtered=5)

    app.next_page()

    assert app.index == 3
    assert [r[0] for r in app.filtered_paths_annotations] == paths[3:5]


def test_next_page_applies_image_type_and_flushes_pending(tmp_path):
    """image_type narrows the fetched page and pending edits are queued."""
    cells = _paths(tmp_path, 4, prefix="cell")
    nuclei = _paths(tmp_path, 4, prefix="nucleus")
    db = _build_png_db(tmp_path, cells + nuclei)
    app = _app(db, grid_rows=1, grid_cols=2, image_type="cell",
               _total_filtered=4)
    app.pending_updates = {cells[0]: 2}

    app.next_page()

    assert app._unsaved_batches == 1
    assert app.update_queue.get_nowait() == {cells[0]: 2}
    assert app.pending_updates == {}
    assert app.index == 2
    assert [r[0] for r in app.filtered_paths_annotations] == cells[2:4]


def test_next_page_in_measurement_mode_only_moves_the_index(tmp_path):
    """With a threshold active the in-memory list is sliced, not re-queried."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 9))
    preset = [(f"p{i}", None) for i in range(9)]
    app = _app(db, grid_rows=1, grid_cols=3, measurement="cell_area",
               threshold=1, filtered_paths_annotations=preset,
               _total_filtered=9)

    app.next_page()

    assert app.index == 3
    assert app.filtered_paths_annotations is preset


def test_previous_page_steps_back_and_refetches(tmp_path):
    """previous_page rewinds one page and re-reads it from the DB."""
    paths = _paths(tmp_path, 10)
    db = _build_png_db(tmp_path, paths, annotations=list(range(10)))
    app = _app(db, grid_rows=1, grid_cols=3, index=6)

    app.previous_page()

    assert app.index == 3
    assert [r[0] for r in app.filtered_paths_annotations] == paths[3:6]
    assert app.calls == ["load_images"]


def test_previous_page_floors_at_zero_and_flushes_pending(tmp_path):
    """Going back from page 0 stays at 0; pending edits still get queued."""
    paths = _paths(tmp_path, 4)
    db = _build_png_db(tmp_path, paths)
    app = _app(db, grid_rows=1, grid_cols=3, index=1)
    app.pending_updates = {paths[1]: 1}

    app.previous_page()

    assert app.index == 0
    assert app._unsaved_batches == 1
    assert app.update_queue.get_nowait() == {paths[1]: 1}
    assert [r[0] for r in app.filtered_paths_annotations] == paths[0:3]


def test_previous_page_with_image_type_and_measurement_branches(tmp_path):
    """image_type is honoured on the way back; measurement mode skips the query."""
    cells = _paths(tmp_path, 6, prefix="cell")
    others = _paths(tmp_path, 6, prefix="blob")
    db = _build_png_db(tmp_path, cells + others)

    app = _app(db, grid_rows=1, grid_cols=2, image_type="cell", index=4)
    app.previous_page()
    assert app.index == 2
    assert [r[0] for r in app.filtered_paths_annotations] == cells[2:4]

    preset = [("kept", None)]
    app2 = _app(db, grid_rows=1, grid_cols=2, index=4, measurement=["a"],
                threshold=[1], filtered_paths_annotations=preset)
    app2.previous_page()
    assert app2.index == 2
    assert app2.filtered_paths_annotations is preset


# ===========================================================================
# update_gui_text
# ===========================================================================

def test_update_gui_text_sets_label_and_pumps_the_event_loop(tmp_path):
    """The status label gets the text and the root is refreshed once."""
    from spacr.gui_elements import AnnotateApp

    db = _build_png_db(tmp_path, _paths(tmp_path, 1))
    app = _app(db)

    AnnotateApp.update_gui_text(app, "Merging data...")

    assert app.status_label.configs == [{"text": "Merging data..."}]
    assert app.root.updates == 1


# ===========================================================================
# train_and_classify
# ===========================================================================

def _merged_frame(paths, feature="separable", labels=None, n_features=3,
                  extra_nan_path=False):
    """Build the frame the patched _read_and_merge_data hands back."""
    n = len(paths)
    if feature == "constant":
        data = {f"feat_{j}": np.ones(n, dtype=float) for j in range(n_features)}
    else:
        rs = np.random.RandomState(0)
        base = np.asarray(labels, dtype=float)
        data = {"feat_0": base * 100.0 + rs.rand(n)}
        for j in range(1, n_features):
            data[f"feat_{j}"] = rs.rand(n)
    df = pd.DataFrame(data)
    df["png_path"] = list(paths)
    df["some_text"] = ["x"] * n          # non-numeric: must be ignored
    if extra_nan_path:
        row = {c: 0.0 for c in df.columns if c.startswith("feat_")}
        row["png_path"] = np.nan
        row["some_text"] = "x"
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def _patch_merge(monkeypatch, df):
    import spacr.io as spacr_io
    monkeypatch.setattr(spacr_io, "_read_and_merge_data",
                        lambda locs, tables, verbose=False, **kw: (df.copy(), []))


def test_train_and_classify_writes_scores_and_annotations(tmp_path, monkeypatch):
    """A separable two-class problem yields confident 1/2 labels in png_list."""
    n = 60
    paths = _paths(tmp_path, n)
    labels = [i % 2 for i in range(n)]          # 0/1 alternating
    db = _build_png_db(
        tmp_path, paths,
        annotations=[1 if lab else 2 for lab in labels],   # 2 maps to class 0
    )
    _patch_merge(monkeypatch, _merged_frame(paths, "separable", labels,
                                            extra_nan_path=True))
    app = _app(db)

    app.train_and_classify()

    cols = [r[1] for r in _fetch(db, 'PRAGMA table_info("png_list")')]
    assert "XGboost_score" in cols and "XGboost_annotation" in cols
    rows = _fetch(db, 'SELECT png_path, XGboost_score, XGboost_annotation '
                      'FROM png_list')
    scores = {p: s for p, s, _ in rows}
    annos = {p: a for p, _, a in rows}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    # class-1 rows should score high and be labelled 1, class-0 rows labelled 2
    assert annos[paths[1]] == 1 and scores[paths[1]] > 0.9
    assert annos[paths[0]] == 2 and scores[paths[0]] < 0.1
    assert app.annotation_column == "XGboost_annotation"
    assert ("gui_text", "Training XGBoost model...") in app.calls


def test_train_and_classify_leaves_uncertain_rows_null(tmp_path, monkeypatch):
    """Constant features give p=0.5 everywhere, so annotations stay NULL."""
    n = 40
    paths = _paths(tmp_path, n)
    labels = [i % 2 for i in range(n)]
    db = _build_png_db(tmp_path, paths,
                       annotations=[1 if lab else 2 for lab in labels])
    _patch_merge(monkeypatch, _merged_frame(paths, "constant", labels))
    app = _app(db)

    app.train_and_classify()

    rows = _fetch(db, 'SELECT XGboost_score, XGboost_annotation FROM png_list')
    assert all(a is None for _, a in rows)
    assert all(abs(s - 0.5) < 1e-6 for s, _ in rows)


def test_train_and_classify_fabricates_missing_class(tmp_path, monkeypatch, capsys):
    """With only one manual class, unlabeled rows are sampled as the other."""
    n = 40
    paths = _paths(tmp_path, n)
    anns = [1 if i < 12 else None for i in range(n)]
    db = _build_png_db(tmp_path, paths, annotations=anns)
    labels = [1 if i < 12 else 0 for i in range(n)]
    _patch_merge(monkeypatch, _merged_frame(paths, "separable", labels))
    app = _app(db)

    app.train_and_classify()

    out = capsys.readouterr().out
    assert "Only one class was present" in out
    assert "randomly labeled 12 unannotated rows as 0" in out
    assert app.annotation_column == "XGboost_annotation"
    cols = [r[1] for r in _fetch(db, 'PRAGMA table_info("png_list")')]
    assert "XGboost_score" in cols


def test_train_and_classify_bails_when_no_unannotated_rows(tmp_path, monkeypatch, capsys):
    """Single class + nothing left to sample => refuse to train."""
    paths = _paths(tmp_path, 6)
    db = _build_png_db(tmp_path, paths, annotations=[1] * 6)
    _patch_merge(monkeypatch, _merged_frame(paths, "separable", [1] * 6))
    app = _app(db)

    app.train_and_classify()

    assert "No unannotated rows to sample" in capsys.readouterr().out
    assert ("gui_text", "Not enough data to train (no second class).") in app.calls
    assert app.annotation_column == "annotate"
    cols = [r[1] for r in _fetch(db, 'PRAGMA table_info("png_list")')]
    assert "XGboost_score" not in cols


def test_train_and_classify_bails_without_annotations(tmp_path, monkeypatch, capsys):
    """Zero manual labels => the 'need at least 2' guard fires."""
    paths = _paths(tmp_path, 5)
    db = _build_png_db(tmp_path, paths)
    _patch_merge(monkeypatch, _merged_frame(paths, "separable", [0] * 5))
    app = _app(db)

    app.train_and_classify()

    assert "Not enough annotated data to train" in capsys.readouterr().out
    assert ("gui_text", "Not enough data to train.") in app.calls
    assert app.annotation_column == "annotate"


def test_train_and_classify_reuses_existing_xgboost_columns(tmp_path, monkeypatch):
    """Re-running over a DB that already has the XGboost columns just updates."""
    n = 40
    paths = _paths(tmp_path, n)
    labels = [i % 2 for i in range(n)]
    db = _build_png_db(
        tmp_path, paths,
        annotations=[1 if lab else 2 for lab in labels],
        extra_columns=['"XGboost_annotation" INTEGER', '"XGboost_score" FLOAT'],
    )
    _patch_merge(monkeypatch, _merged_frame(paths, "constant", labels))
    app = _app(db)

    app.train_and_classify()

    rows = _fetch(db, 'SELECT XGboost_score FROM png_list')
    assert len(rows) == n
    assert all(r[0] is not None for r in rows)


# ===========================================================================
# _get_png_list_columns
# ===========================================================================

def test_get_png_list_columns_reports_names_and_upper_types(tmp_path):
    """Declared types come back uppercased; untyped columns come back as ''."""
    from spacr.gui_elements import AnnotateApp

    db = tmp_path / "cols.db"
    con = sqlite3.connect(str(db))
    try:
        con.execute('CREATE TABLE "png_list" ("png_path" text, "annotate" integer, "loose")')
        con.commit()
    finally:
        con.close()

    app = _app(db)
    cols = AnnotateApp._get_png_list_columns(app)

    assert cols == [("png_path", "TEXT"), ("annotate", "INTEGER"), ("loose", "")]


# ===========================================================================
# _parse_field_value
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("true", True), ("T", True), ("Yes", True), ("y", True), ("on", True),
    ("1", True),
    ("false", False), ("f", False), ("NO", False), ("n", False), ("off", False),
    ("0", False),
    ("42", 42),
    ("-7", -7),
    ("3.5", 3.5),
    ("1e3", 1000.0),
    ("plain text", "plain text"),
])
def test_parse_field_value_scalars(tmp_path, raw, expected):
    """Bools win over numbers; numbers win over free text."""
    from spacr.gui_elements import AnnotateApp

    app = _app(_build_png_db(tmp_path, _paths(tmp_path, 1)))
    got = AnnotateApp._parse_field_value(app, "some_key", raw)
    assert got == expected
    assert type(got) is type(expected)


def test_parse_field_value_lists(tmp_path):
    """Comma strings and known list-keys become lists with coerced members."""
    from spacr.gui_elements import AnnotateApp

    app = _app(_build_png_db(tmp_path, _paths(tmp_path, 1)))
    parse = lambda k, v: AnnotateApp._parse_field_value(app, k, v)

    assert parse("anything", "1,2.5,abc") == [1, 2.5, "abc"]
    assert parse("anything", "a,,b") == ["a", "b"]          # empty token skipped
    assert parse("classes", "nc") == ["nc"]                 # listy key, no comma
    assert parse("train_channels", "r,g,b") == ["r", "g", "b"]
    assert parse("tables", "1e2,3") == [100.0, 3]


# ===========================================================================
# convert_settings_dict_for_gui
#
# ``AnnotateApp`` used to carry its own copy of this function. It had no
# callers -- gui_core and spacr.qt.screens.settings_model both import the
# gui_utils one -- and the copy had drifted stale (timelapse_mode without
# trackastra/ultrack, optimizer_type with only a subset of the torch optimizers,
# loss_type with 2 of the 6 build_loss aliases). It was deleted rather than
# resynced; these tests now exercise the single live implementation.
# ===========================================================================

def test_convert_settings_dict_classifies_widget_kinds():
    """bool -> check, numbers/lists/strings -> entry, known keys -> combo."""
    from spacr import gui_utils as GU

    out = GU.convert_settings_dict_for_gui({
        "verbose": True,
        "epochs": 10,
        "lr": 0.001,
        "channels_list": [1, 2, 3],
        "name": "abc",
        "nothing": None,
        "metadata_type": "cq1",
        "channels": "[0,1]",
    })

    assert out["verbose"] == ("check", None, True)
    assert out["epochs"] == ("entry", None, 10)
    assert out["lr"] == ("entry", None, 0.001)
    assert out["channels_list"] == ("entry", None, "[1, 2, 3]")
    assert out["name"] == ("entry", None, "abc")
    assert out["nothing"] == ("entry", None, None)
    # special cases ignore the supplied value and use the canned spec
    assert out["metadata_type"] == ("combo",
                                    ["cellvoyager", "cq1", "auto", "custom"],
                                    "cellvoyager")
    kind, options, initial = out["channels"]
    assert kind == "combo" and initial == "[0,1,2,3]" and "[0,1]" in options


def test_annotate_app_no_longer_shadows_convert_settings_dict():
    """The stale duplicate is gone, so the GUI cannot serve two option sets."""
    from spacr.gui_elements import AnnotateApp
    assert not hasattr(AnnotateApp, "convert_settings_dict_for_gui")


def test_convert_settings_dict_uses_real_torchvision_model_list():
    """model_type options extend to the full zoo once torchvision is loaded."""
    pytest.importorskip("torchvision.models")
    from spacr import gui_utils as GU

    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"model_type": "resnet50"}
    )["model_type"]
    assert kind == "combo" and initial == "resnet50"
    assert "resnet50" in options
    assert options == sorted(options)
    # the curated fallback is short; torchvision exposes far more
    assert len(options) > 5


def test_convert_settings_dict_falls_back_when_torchvision_unloaded(monkeypatch):
    """With torchvision absent from sys.modules the curated list is used.

    gui_utils deliberately never *imports* torchvision here -- enumerating the
    zoo costs ~5 s and made the first module open sluggish -- it only reads
    sys.modules. So the fallback is exercised by hiding the module, not by
    blocking the import.
    """
    import sys
    from spacr import gui_utils as GU

    monkeypatch.delitem(sys.modules, "torchvision.models", raising=False)
    out = GU.convert_settings_dict_for_gui({"model_type": "resnet50"})

    kind, options, initial = out["model_type"]
    assert kind == "combo" and initial == "resnet50"
    assert options == list(GU._TORCHVISION_MODELS_CURATED)


# --- the combos must offer exactly what the pipeline accepts ---------------

def test_dataset_mode_combo_matches_the_modes_io_dispatches_on():
    """'recruitment' used to be offered here and is not a real mode.

    io.generate_training_dataset dispatches on metadata|annotation|measurement
    and returns (None, None) for anything else -- so picking 'recruitment' in
    the Tk GUI silently produced no dataset at all.
    """
    from spacr import gui_utils as GU
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"dataset_mode": "metadata"})["dataset_mode"]
    assert kind == "combo"
    assert set(options) == {"metadata", "annotation", "measurement"}
    assert initial == "metadata"


def test_class_balance_combo_matches_io_class_balance_modes():
    from spacr import gui_utils as GU
    from spacr.io import CLASS_BALANCE_MODES
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"class_balance": "none"})["class_balance"]
    assert kind == "combo"
    assert set(options) == set(CLASS_BALANCE_MODES)
    assert initial == "none"


def test_cv_group_by_combo_matches_io_cv_group_levels():
    from spacr import gui_utils as GU
    from spacr.io import CV_GROUP_LEVELS
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"cv_group_by": "well"})["cv_group_by"]
    assert kind == "combo"
    assert set(options) == set(CV_GROUP_LEVELS)
    # well is the safe default: crops from one well are not independent
    assert initial == "well"


def test_seg_qc_combo_matches_seg_qc_modes():
    from spacr import gui_utils as GU
    from spacr.seg_qc import MODES
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"seg_qc": "report"})["seg_qc"]
    assert kind == "combo"
    assert set(options) == set(MODES)
    # report, not flag: surface the problem, do not silently filter fields
    assert initial == "report"


def test_optimizer_and_loss_combos_are_all_accepted_by_the_pipeline():
    """Every offered option must survive the real dispatch, not just look right."""
    from spacr import gui_utils as GU
    specs = GU.convert_settings_dict_for_gui(
        {"optimizer_type": "adamw", "loss_type": "auto"})

    _, opt_options, _ = specs["optimizer_type"]
    assert set(opt_options) == {
        "adamw", "adam", "adamax", "adagrad", "adadelta", "asgd",
        "sgd", "rmsprop", "nadam", "radam",
    }

    torch = pytest.importorskip("torch")
    from spacr.utils import build_loss
    _, loss_options, _ = specs["loss_type"]
    counts = torch.tensor([80.0, 20.0])
    for name in loss_options:
        n_classes = 1 if name == "binary_cross_entropy_with_logits" else 2
        fn = build_loss(name, num_classes=n_classes, class_counts=counts)
        assert callable(fn), name


# ===========================================================================
# build_multi_annotation
# ===========================================================================

@pytest.mark.parametrize("bad", [None, [], (), "col_a", 0])
def test_build_multi_annotation_rejects_bad_sources(tmp_path, bad):
    """Anything that is not a non-empty list/tuple is a ValueError."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 1))
    app = _app(db)
    with pytest.raises(ValueError, match="non-empty list"):
        app.build_multi_annotation(bad)


def test_build_multi_annotation_encodes_base3_digits(tmp_path):
    """Each source column contributes a base-3 digit; all-zero stores NULL."""
    paths = _paths(tmp_path, 7)
    db = _build_png_db(tmp_path, paths,
                       extra_columns=['"col_a" INTEGER', '"col_b" INTEGER'])
    combos = [(1, None), (2, None), (None, 1), (1, 1), (2, 2), (None, None), (0, 0)]
    con = sqlite3.connect(db)
    try:
        for p, (a, b) in zip(paths, combos):
            con.execute('UPDATE "png_list" SET "col_a"=?, "col_b"=? WHERE png_path=?',
                        (a, b, p))
        con.commit()
    finally:
        con.close()

    app = _app(db)
    app.build_multi_annotation(["col_a", "col_b"], target_column="multi_annot")

    got = dict(_fetch(db, 'SELECT png_path, multi_annot FROM png_list'))
    expected = [2, 3, 4, 5, 9, None, None]     # 1 + d_a*1 + d_b*3
    assert [got[p] for p in paths] == expected
    assert app.annotation_column == "multi_annot"
    assert app.calls == ["prefilter", "load_images"]


def test_build_multi_annotation_creates_missing_source_columns(tmp_path):
    """Absent source columns are added as NULL INTEGER before the UPDATE."""
    paths = _paths(tmp_path, 3)
    db = _build_png_db(tmp_path, paths, extra_columns=['"col_a" INTEGER'])
    con = sqlite3.connect(db)
    try:
        con.execute('UPDATE "png_list" SET "col_a"=2 WHERE png_path=?', (paths[0],))
        con.commit()
    finally:
        con.close()

    app = _app(db)
    app.build_multi_annotation(["col_a", "never_seen", "third"],
                               target_column="combo_col")

    info = {r[1]: r[2] for r in _fetch(db, 'PRAGMA table_info("png_list")')}
    assert info["never_seen"] == "INTEGER" and info["third"] == "INTEGER"
    got = dict(_fetch(db, 'SELECT png_path, combo_col FROM png_list'))
    assert got[paths[0]] == 3          # 1 + 2*1 + 0*3 + 0*9
    assert got[paths[1]] is None


def test_build_multi_annotation_uses_powers_of_three_for_three_sources(tmp_path):
    """The third source column is weighted by 3**2 = 9."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(
        tmp_path, paths,
        extra_columns=['"a" INTEGER', '"b" INTEGER', '"c" INTEGER'],
    )
    con = sqlite3.connect(db)
    try:
        con.execute('UPDATE "png_list" SET a=1, b=2, c=2 WHERE png_path=?', (paths[0],))
        con.execute('UPDATE "png_list" SET a=NULL, b=NULL, c=1 WHERE png_path=?', (paths[1],))
        con.commit()
    finally:
        con.close()

    app = _app(db)
    app.build_multi_annotation(["a", "b", "c"], target_column="mc")

    got = dict(_fetch(db, 'SELECT png_path, mc FROM png_list'))
    assert got[paths[0]] == 1 + 1 * 1 + 2 * 3 + 2 * 9
    assert got[paths[1]] == 1 + 0 + 0 + 1 * 9


# ===========================================================================
# ensure_multi_annot_from_selection
# ===========================================================================

@pytest.mark.parametrize("bad", [None, [], "col_a"])
def test_ensure_multi_annot_rejects_bad_sources(tmp_path, bad):
    """The selection must be a non-empty list/tuple."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 1))
    app = _app(db)
    with pytest.raises(ValueError, match="non-empty list"):
        app.ensure_multi_annot_from_selection(bad)


def test_ensure_multi_annot_single_column_is_used_directly(tmp_path):
    """One selected column becomes the annotation column with no new SQL column."""
    db = _build_png_db(tmp_path, _paths(tmp_path, 2))
    app = _app(db)

    result = app.ensure_multi_annot_from_selection(["only_col"])

    assert result == "only_col"
    assert app.annotation_column == "only_col"
    assert app.calls == ["prefilter", "load_images"]
    cols = [r[1] for r in _fetch(db, 'PRAGMA table_info("png_list")')]
    assert "only_col" in cols          # _ensure_annotation_column created it
    assert "class_column" not in cols


def test_ensure_multi_annot_bumps_colliding_target_name(tmp_path):
    """An occupied target name is auto-bumped until a free one is found."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(
        tmp_path, paths,
        extra_columns=['"a" INTEGER', '"b" INTEGER',
                       '"class_column" INTEGER', '"class_column_1" INTEGER'],
    )
    con = sqlite3.connect(db)
    try:
        con.execute('UPDATE "png_list" SET a=2, b=1 WHERE png_path=?', (paths[0],))
        con.commit()
    finally:
        con.close()

    app = _app(db)
    result = app.ensure_multi_annot_from_selection(["a", "b"])

    assert result == "class_column_2"
    assert app.annotation_column == "class_column_2"
    got = dict(_fetch(db, 'SELECT png_path, class_column_2 FROM png_list'))
    assert got[paths[0]] == 1 + 2 * 1 + 1 * 3
    assert got[paths[1]] is None
    # build_multi_annotation refreshes once, then the tail refreshes again
    assert app.calls == ["prefilter", "load_images", "prefilter", "load_images"]


def test_ensure_multi_annot_without_force_rebuild_still_builds(tmp_path):
    """force_rebuild=False on an already-current column takes the else branch."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(tmp_path, paths,
                       extra_columns=['"a" INTEGER', '"b" INTEGER'])
    con = sqlite3.connect(db)
    try:
        con.execute('UPDATE "png_list" SET a=1, b=2 WHERE png_path=?', (paths[0],))
        con.commit()
    finally:
        con.close()

    app = _app(db, annotation_column="class_column")
    result = app.ensure_multi_annot_from_selection(
        ["a", "b"], target_column="class_column", force_rebuild=False
    )

    assert result == "class_column"
    got = dict(_fetch(db, 'SELECT png_path, class_column FROM png_list'))
    assert got[paths[0]] == 1 + 1 * 1 + 2 * 3
    assert got[paths[1]] is None


def test_ensure_multi_annot_blank_target_falls_back_to_class_column(tmp_path):
    """A whitespace-only target name defaults to 'class_column'."""
    paths = _paths(tmp_path, 2)
    db = _build_png_db(tmp_path, paths,
                       extra_columns=['"a" INTEGER', '"b" INTEGER'])
    con = sqlite3.connect(db)
    try:
        con.execute('UPDATE "png_list" SET a=2, b=2 WHERE png_path=?', (paths[1],))
        con.commit()
    finally:
        con.close()

    app = _app(db)
    result = app.ensure_multi_annot_from_selection(["a", "b"], target_column="   ")

    assert result == "class_column"
    got = dict(_fetch(db, 'SELECT png_path, class_column FROM png_list'))
    assert got[paths[1]] == 1 + 2 * 1 + 2 * 3
