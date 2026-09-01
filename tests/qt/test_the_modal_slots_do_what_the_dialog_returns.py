"""The slots behind the file dialogs, driven by stubbing the dialog.

Instruction 288. Seven slots were marked ``# pragma: no cover - modal``.
Modal is a reason not to OPEN one in a test; it is not a reason to leave
the slot untested, and everything interesting in these happens after the
dialog returns -- the cancel that must do nothing, and the path that must
be used exactly as given.

The same technique the ``ask_for_the_path`` tests use: replace the Qt
static so the dialog never opens and the slot runs unchanged.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def hits(qtbot):
    from spacr.qt.screens.hit_list import HitListScreen

    screen = HitListScreen()
    qtbot.addWidget(screen)
    # A LIST TO EXPORT. `_ask_and_export` refuses before it opens a
    # dialog when `_shown` is None, so without this the export slots
    # never reach the dialog at all -- and a test asserting "nothing was
    # exported" passes for the wrong reason.
    screen._shown = object()
    return screen


# ---------------------------------------------------------------------------
# The three exports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot,fmt,suffix", [
    ("_on_export_csv", "csv", ".csv"),
    ("_on_export_markdown", "markdown", ".md"),
    ("_on_export_html", "html", ".html"),
])
def test_each_export_asks_for_a_path_and_uses_it(hits, monkeypatch, tmp_path,
                                                 slot, fmt, suffix):
    """THE ARM. The format is the part that must not get crossed over --
    three slots differing only in a string are exactly where a copied
    line goes unnoticed.
    """
    from PySide6.QtWidgets import QFileDialog

    chosen = tmp_path / f"hits{suffix}"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))

    exported = []
    monkeypatch.setattr(type(hits), "export",
                        lambda self, path, kind: exported.append((path, kind)))

    getattr(hits, slot)()

    assert exported == [(str(chosen), fmt)], (
        f"{slot} exported {exported}, expected {fmt} to {chosen}")


@pytest.mark.parametrize("slot", ["_on_export_csv", "_on_export_markdown",
                                  "_on_export_html"])
def test_cancelling_an_export_writes_nothing(hits, monkeypatch, slot):
    """The cancel. A dialog that returns "" must not become a file
    called "" -- which is what an unguarded export would attempt."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    exported = []
    monkeypatch.setattr(type(hits), "export",
                        lambda self, path, kind: exported.append(path))

    asked = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (asked.append(True), ("", ""))[1]))

    getattr(hits, slot)()

    assert asked == [True], "the dialog was never reached"
    assert exported == [], "a cancelled export still wrote something"


# ---------------------------------------------------------------------------
# Browse, and the metadata picker
# ---------------------------------------------------------------------------

def test_browsing_loads_the_folder_that_was_chosen(hits, monkeypatch,
                                                   tmp_path):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    loaded = []
    monkeypatch.setattr(type(hits), "load_folder",
                        lambda self, folder: loaded.append(folder))

    hits._on_browse()

    assert loaded == [str(tmp_path)]


def test_cancelling_browse_loads_nothing(hits, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    loaded = []
    monkeypatch.setattr(type(hits), "load_folder",
                        lambda self, folder: loaded.append(folder))

    hits._on_browse()

    assert loaded == [], "a cancelled browse loaded a folder anyway"


def test_picking_metadata_passes_every_file_through(hits, monkeypatch,
                                                    tmp_path):
    """getOpenFileNames is plural, and all of them have to arrive."""
    from PySide6.QtWidgets import QFileDialog

    picked = [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (picked, "")))
    given = []
    monkeypatch.setattr(type(hits), "set_metadata_files",
                        lambda self, paths: given.append(list(paths)))

    hits._on_pick_metadata()

    assert given == [picked]


def test_cancelling_the_metadata_picker_changes_nothing(hits, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))
    given = []
    monkeypatch.setattr(type(hits), "set_metadata_files",
                        lambda self, paths: given.append(paths))

    hits._on_pick_metadata()

    assert given == []


def test_exporting_with_no_hit_list_refuses_before_any_dialog(qtbot,
                                                              monkeypatch):
    """The earlier refusal, pinned so the fixture's `_shown` is visibly
    necessary rather than decoration.

    This is also why the cancel tests above set it: without a list, they
    would assert "nothing was exported" against a slot that never opened
    a dialog, which is true for the wrong reason.
    """
    from PySide6.QtWidgets import QFileDialog

    from spacr.qt.screens.hit_list import HitListScreen

    screen = HitListScreen()
    qtbot.addWidget(screen)
    assert screen._shown is None

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: pytest.fail("a dialog was opened")))

    said = []
    screen._set_summary = lambda text, *, problem: said.append((text, problem))

    screen._on_export_csv()

    assert said and "no hit list to export" in said[0][0]
    assert said[0][1] is True


# ---------------------------------------------------------------------------
# The same two slots in the methods export screen
# ---------------------------------------------------------------------------

@pytest.fixture
def methods(qtbot):
    from spacr.qt.screens.methods_export import MethodsExportScreen

    screen = MethodsExportScreen()
    qtbot.addWidget(screen)
    return screen


def test_browsing_for_a_folder_fills_the_field(methods, monkeypatch,
                                               tmp_path):
    """THE ARM, folder half. `is_folder` picks the dialog, and picking
    the wrong one is a file picker that cannot choose a directory."""
    from PySide6.QtWidgets import QFileDialog

    key = next(iter(methods._fields))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: pytest.fail("the file picker was used")))

    methods._on_browse(key, True)

    assert methods._fields[key].text() == str(tmp_path)


def test_browsing_for_a_file_uses_the_file_picker(methods, monkeypatch,
                                                  tmp_path):
    """The other half, so `is_folder` is proved to select between them."""
    from PySide6.QtWidgets import QFileDialog

    key = next(iter(methods._fields))
    chosen = tmp_path / "settings.csv"
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: pytest.fail("the folder picker was used")))

    methods._on_browse(key, False)

    assert methods._fields[key].text() == str(chosen)


def test_cancelling_a_browse_leaves_the_field_alone(methods, monkeypatch):
    """Not "sets it to empty" -- a cancel must not erase what is there."""
    from PySide6.QtWidgets import QFileDialog

    key = next(iter(methods._fields))
    methods._fields[key].setText("/somewhere/already/chosen")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    methods._on_browse(key, True)

    assert methods._fields[key].text() == "/somewhere/already/chosen"


def test_exporting_writes_to_the_path_that_was_chosen(methods, monkeypatch,
                                                      tmp_path):
    from PySide6.QtWidgets import QFileDialog

    chosen = tmp_path / "methods.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))
    written = []
    monkeypatch.setattr(type(methods), "export",
                        lambda self, path: written.append(path))

    methods._on_export()

    assert written == [str(chosen)]


def test_cancelling_the_export_writes_nothing(methods, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    asked = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (asked.append(True), ("", ""))[1]))
    written = []
    monkeypatch.setattr(type(methods), "export",
                        lambda self, path: written.append(path))

    methods._on_export()

    assert asked == [True], "the dialog was never reached"
    assert written == []
