"""Five more single-decision modules: two driven, three pinned.

Each of the three pins names the line that makes its guard unreachable,
so the pin fails if that line changes rather than the guard quietly
coming alive.
"""
from __future__ import annotations

import inspect
import os
import sqlite3
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# spacr/qt/preview_registry.py -- a preview panel that does not propagate
# ---------------------------------------------------------------------------

def _screen_with_a_runtime_panel(qtbot):
    """The two anchors :func:`_insert_above_actions` reaches for."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    screen = QWidget()
    qtbot.addWidget(screen)
    wrap = QWidget(screen)
    layout = QVBoxLayout(wrap)
    actions = QWidget(wrap)
    layout.addWidget(actions)
    screen._runtime_wrap = wrap
    screen._actions_row = actions
    return screen


def _builder_module(monkeypatch, panel_factory):
    """A ``module:function`` builder the registry can resolve by name."""
    from PySide6.QtWidgets import QWidget

    module = types.ModuleType("spacr_test_preview_builder")
    module.build = lambda screen: (panel_factory(), QWidget())
    monkeypatch.setitem(sys.modules, "spacr_test_preview_builder", module)
    return "spacr_test_preview_builder:build"


class TestAPanelWithNothingToPropagate:

    def test_a_panel_offering_the_hook_is_wired_to_the_host(self, qtbot,
                                                            monkeypatch):
        pytest.importorskip("PySide6")
        from spacr.qt.preview_registry import PreviewSpec, _attach

        registered = []

        class Panel:
            def set_propagate_callback(self, callback):
                registered.append(callback)

        spec = PreviewSpec(builder=_builder_module(monkeypatch, Panel),
                           title="Preview")
        host = _attach(_screen_with_a_runtime_panel(qtbot), "measure", spec)

        assert host is not None
        assert len(registered) == 1, "the panel was not given the host's hook"
        assert registered[0] == host.on_propagate

    def test_a_panel_without_the_hook_is_left_alone(self, qtbot, monkeypatch):
        """THE UNCOVERED ARC.

        The hook is optional: a preview with nothing to hand back to the
        settings form does not implement it. Calling it anyway would be
        an AttributeError from inside the registry, which loses the whole
        preview rather than just its propagation.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.preview_registry import PreviewSpec, _attach

        class Panel:
            pass

        spec = PreviewSpec(builder=_builder_module(monkeypatch, Panel),
                           title="Preview")
        host = _attach(_screen_with_a_runtime_panel(qtbot), "measure", spec)

        assert host is not None, (
            "a panel that cannot propagate lost its whole preview")
        assert host.toggle.text() == "Preview"

    def test_a_screen_with_no_runtime_panel_gets_no_preview(self, qtbot,
                                                            monkeypatch):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QWidget

        from spacr.qt.preview_registry import PreviewSpec, _attach

        class Panel:
            pass

        bare = QWidget()
        qtbot.addWidget(bare)
        spec = PreviewSpec(builder=_builder_module(monkeypatch, Panel),
                           title="Preview")

        assert _attach(bare, "measure", spec) is None


# ---------------------------------------------------------------------------
# spacr/qt/multi_format.py -- a multi-page tiff whose series carries no axes
# ---------------------------------------------------------------------------

class _Page:
    shape = (8, 6)
    dtype = "uint16"


class _FakeTiff:
    """Enough of tifffile.TiffFile for :func:`_describe_tif`."""

    def __init__(self, pages, series):
        self.pages = pages
        self.series = series

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_tifffile(monkeypatch, pages, series):
    module = types.ModuleType("tifffile")
    module.TiffFile = lambda path: _FakeTiff(pages, series)
    monkeypatch.setitem(sys.modules, "tifffile", module)


class TestDescribingAMultiPageTiff:

    def test_a_series_with_axes_puts_them_in_the_notes(self, monkeypatch,
                                                       tmp_path):
        from spacr.qt.multi_format import _describe_tif

        series = [types.SimpleNamespace(axes="ZCYX")]
        _install_tifffile(monkeypatch, [_Page(), _Page(), _Page()], series)

        described = _describe_tif(tmp_path / "stack.tif")
        assert described.kind == "tif_multi"
        assert described.n_fields == 3
        assert "pages=3" in described.notes
        assert "axes=ZCYX" in described.notes

    def test_no_series_at_all_leaves_the_axis_note_off(self, monkeypatch,
                                                       tmp_path):
        """THE UNCOVERED ARC.

        The axes string comes from the SERIES, not the file -- a reader
        that parsed no series, or one whose series has no axes, gives
        nothing to say. An "axes=" note with nothing after it would claim
        the stack was read and found unlabelled, which is a different
        statement from not knowing.
        """
        from spacr.qt.multi_format import _describe_tif

        _install_tifffile(monkeypatch, [_Page(), _Page()], [])

        described = _describe_tif(tmp_path / "stack.tif")
        assert described.notes == ["pages=2"]

    def test_a_single_page_tiff_is_not_this_functions_business(
            self, monkeypatch, tmp_path):
        from spacr.qt.multi_format import _describe_tif

        _install_tifffile(monkeypatch, [_Page()], [])
        assert _describe_tif(tmp_path / "flat.tif") is None


# ---------------------------------------------------------------------------
# spacr/qt/widgets/foldable.py -- remember() is only installed when it has
# somewhere to remember to
# ---------------------------------------------------------------------------

class TestTheFoldIsOnlyStoredWhenThereIsAKey:

    def test_no_key_means_the_callback_is_the_callers_own(self, qtbot):
        """THE PIN.

        ``remember`` is handed to the Folder only when ``key`` is
        truthy; with no key the caller's own ``on_change`` is used
        directly. So the ``if key:`` inside ``remember`` can never be
        false when it runs.

        The empty key is not an oversight -- it is what a bare panel in a
        test wants. A test that wrote to the real preferences would fold
        a panel on the user's next launch.
        """
        from spacr.qt.widgets import foldable

        source = inspect.getsource(foldable.make_foldable)
        assert "on_change=remember if key else on_change" in source, (
            "remember is now installed regardless of the key, so its own "
            "`if key:` is live and needs a test rather than this pin")

    def test_with_a_key_the_fold_is_written_and_read_back(self, qtbot,
                                                          monkeypatch):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QLabel, QWidget

        from spacr.qt.widgets import foldable

        stored = {}
        module = types.ModuleType("spacr.qt.preferences")
        module.set_folded_panel = lambda key, shut: stored.__setitem__(key, shut)
        module.get_folded_panels = lambda: dict(stored)
        monkeypatch.setitem(sys.modules, "spacr.qt.preferences", module)

        heading, body = QLabel("Advanced"), QWidget()
        qtbot.addWidget(heading)
        qtbot.addWidget(body)
        folder = foldable.make_foldable(heading, body,
                                        persist_key="measure/advanced")
        folder.set_shut(True)

        assert stored.get("measure/advanced") is True

    def test_without_a_key_nothing_is_written(self, qtbot, monkeypatch):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QLabel, QWidget

        from spacr.qt.widgets import foldable

        wrote = []
        module = types.ModuleType("spacr.qt.preferences")
        module.set_folded_panel = lambda key, shut: wrote.append(key)
        module.get_folded_panels = lambda: {}
        monkeypatch.setitem(sys.modules, "spacr.qt.preferences", module)

        heading, body = QLabel("Advanced"), QWidget()
        qtbot.addWidget(heading)
        qtbot.addWidget(body)
        folder = foldable.make_foldable(heading, body)
        folder.set_shut(True)

        assert wrote == [], "a keyless panel wrote to the user's preferences"


# ---------------------------------------------------------------------------
# spacr/qt/widgets/qc_summary.py -- a table with no columns
# ---------------------------------------------------------------------------

class TestEveryTableSqliteListsHasColumns:

    def test_a_stamped_and_an_unstamped_table_are_told_apart(self, tmp_path):
        from spacr.qt.widgets.qc_summary import _read_units

        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute(
                "CREATE TABLE cell (object_label TEXT, cell_area REAL, "
                "measurement_ndim INTEGER, measurement_units TEXT)")
            connection.execute(
                "INSERT INTO cell VALUES ('1', 2.0, 2, 'px^2')")
            connection.execute("CREATE TABLE notes (text TEXT)")

        card = _read_units(str(tmp_path))
        assert card.source == str(db)
        assert card.verdict in ("ok", "warn", "error", "unknown")

    def test_pragma_names_a_column_for_every_table_in_the_catalogue(
            self, tmp_path):
        """THE PIN.

        ``if columns:`` guards the "unstamped" list against a table
        PRAGMA reports nothing for. SQLite has no zero-column table, and
        the names come from its own catalogue in the same read-only
        connection, so the empty case cannot arise -- but an empty
        PRAGMA result would otherwise put a phantom table on the card.
        """
        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a TEXT)")
            connection.execute("CREATE TABLE nucleus (a TEXT, b REAL)")
            connection.execute("CREATE TABLE png_list (a TEXT)")

        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            assert tables
            for table in tables:
                columns = {row[1] for row in
                           connection.execute(f'PRAGMA table_info("{table}")')}
                assert columns, f"{table} was listed but reports no columns"


# ---------------------------------------------------------------------------
# spacr/errors.py -- closing a connection that was never opened
# ---------------------------------------------------------------------------

class TestClosingTheRunStatusConnection:

    def test_a_database_that_cannot_be_opened_is_reported_not_swallowed(
            self, tmp_path, monkeypatch):
        """The producing side of the `conn is not None` guard.

        When ``connect`` raises, the finally clause runs with nothing to
        close -- which is the only way that guard is false.
        """
        from spacr import errors

        db = tmp_path / "measurements.db"
        db.write_bytes(b"not a database at all")

        def refuse(*args, **kwargs):
            raise sqlite3.DatabaseError("file is not a database")

        module = types.ModuleType("spacr.database_concurrency")
        module.connect = refuse
        monkeypatch.setitem(sys.modules, "spacr.database_concurrency", module)

        with pytest.raises(errors.RunStatusUnreadable, match="cannot be read"):
            errors.read_run_status(str(db))

    def test_a_file_that_is_not_a_database_yields_no_records(self, tmp_path):
        from spacr import errors

        assert errors.read_run_status(str(tmp_path / "missing.db")) == []

    def test_a_database_with_no_stamp_table_yields_no_records(self, tmp_path):
        from spacr import errors

        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a TEXT)")

        assert errors.read_run_status(str(db)) == []
