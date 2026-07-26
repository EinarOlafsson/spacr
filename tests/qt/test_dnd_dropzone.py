"""End-to-end drag-and-drop tests for :mod:`spacr.qt.dnd`.

Real ``QDragEnterEvent`` / ``QDragMoveEvent`` / ``QDropEvent`` objects carrying
real ``QMimeData`` are dispatched at a real widget that has had
:func:`install_dropzone` applied, and the assertions are on what the handler
and the screen actually received.

Qt only routes DragMove/Drop to a widget once that widget has *accepted* a
DragEnter, so :func:`_drop` replays the full enter → move → drop sequence the
window system produces — which is also what makes the rejection path (a drag
the dropzone refuses) observable.

Modal dialogs are driven by :func:`_auto_modal`, a timer that accepts/rejects
whatever modal is up; that keeps the real dialog code in the loop without
blocking the suite.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QTimer, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QListWidget, QWidget

from spacr.qt import dnd as dnd_mod
from spacr.qt.dnd import (
    DropHandler, has_images_in, find_image_folders_nearby, install_dropzone,
    sample_image_names, suggest_alternatives_dialog, _apply_settings_csv,
    _mime_has_local_paths, _mime_local_paths,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mkimg(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"II*\x00\x08\x00\x00\x00")
    return p


def _mime(paths, remote=()):
    m = QMimeData()
    urls = [QUrl.fromLocalFile(str(p)) for p in paths]
    urls += [QUrl(u) for u in remote]
    m.setUrls(urls)
    return m


def _drop(widget, paths, remote=(), enter_paths=None):
    """Replay enter → move → drop on ``widget``; return the QDropEvent.

    ``enter_paths`` defaults to ``paths``; pass it explicitly to make the
    DragEnter carry different content from the Drop.
    """
    enter_mime = _mime(enter_paths if enter_paths is not None else paths)
    e1 = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, enter_mime,
                         Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, e1)
    move_mime = _mime(enter_paths if enter_paths is not None else paths)
    e2 = QDragMoveEvent(QPoint(5, 5), Qt.CopyAction, move_mime,
                        Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, e2)
    drop_mime = _mime(paths, remote)
    e3 = QDropEvent(QPointF(5, 5), Qt.CopyAction, drop_mime,
                    Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, e3)
    return e3


@contextlib.contextmanager
def _auto_modal(action="accept", select_row=None):
    """Accept (or reject) the next modal dialog that appears."""
    state = {"seen": 0}
    t = QTimer()
    t.setInterval(5)

    def tick():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            return
        if select_row is not None:
            for lw in dlg.findChildren(QListWidget):
                lw.setCurrentRow(select_row)
        state["seen"] += 1
        getattr(dlg, action)()

    t.timeout.connect(tick)
    t.start()
    try:
        yield state
    finally:
        t.stop()


class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class FakeScreen:
    """Minimal stand-in for an AppScreen: records applied settings + logs."""

    def __init__(self):
        self.applied = []
        self._console = _Console()

    def apply_settings_dict(self, settings):
        self.applied.append(dict(settings))
        return len(settings)


class ScreenWidget(QWidget, FakeScreen):
    """Same, but a real QWidget so it can parent a real modal dialog."""

    def __init__(self):
        QWidget.__init__(self)
        FakeScreen.__init__(self)


class RecordingHandler(DropHandler):
    """Configurable handler that records every call."""

    def __init__(self, accept=True, alternatives=None, multiple=False,
                 raise_on_apply=False):
        self._accept = accept
        self._alts = alternatives or []
        self._multiple = multiple
        self._raise = raise_on_apply
        self.applied = []
        self.asked = []

    def accepts_multiple(self):
        return self._multiple

    def can_accept(self, path):
        self.asked.append(path)
        return self._accept

    def suggest_alternatives(self, path):
        return list(self._alts)

    def error_message(self, path):
        return f"cannot use {path.name}"

    def apply(self, path, screen):
        if self._raise:
            raise RuntimeError("boom: disk on fire")
        self.applied.append(path)


class FakeMessageBox:
    """Stands in for QMessageBox — records calls instead of blocking."""

    calls = []

    @staticmethod
    def warning(parent, title, text):
        FakeMessageBox.calls.append(("warning", title, text))

    @staticmethod
    def information(parent, title, text):
        FakeMessageBox.calls.append(("information", title, text))


@pytest.fixture
def msgbox(monkeypatch):
    FakeMessageBox.calls = []
    monkeypatch.setattr(dnd_mod, "QMessageBox", FakeMessageBox)
    return FakeMessageBox


@pytest.fixture
def zone(qtbot, qt_theme_applied):
    """Factory: build a dropzone widget wired to ``handler``."""
    made = []

    def _make(handler, screen=None):
        w = ScreenWidget()
        qtbot.addWidget(w)
        w.resize(120, 120)
        w.show()
        install_dropzone(w, handler, screen if screen is not None else w)
        made.append(w)
        return w

    yield _make
    for w in made:
        w.hide()


# ---------------------------------------------------------------------------
# DropHandler base-class defaults
# ---------------------------------------------------------------------------

def test_drophandler_is_abstract():
    with pytest.raises(TypeError):
        DropHandler()


def test_drophandler_defaults(tmp_path):
    class Minimal(DropHandler):
        def can_accept(self, path):
            return True

        def apply(self, path, screen):
            pass

    h = Minimal()
    assert h.suggest_alternatives(tmp_path) == []
    assert h.error_message(tmp_path / "plate01") == \
        "This module can't use 'plate01'."
    assert h.accepts_multiple() is False


# ---------------------------------------------------------------------------
# Mime helpers
# ---------------------------------------------------------------------------

def test_mime_has_local_paths_false_without_urls():
    m = QMimeData()
    m.setText("just some text")
    assert _mime_has_local_paths(m) is False


def test_mime_has_local_paths_false_for_remote_only():
    m = QMimeData()
    m.setUrls([QUrl("https://example.com/plate.tif")])
    assert _mime_has_local_paths(m) is False


def test_mime_has_local_paths_true_when_one_is_local(tmp_path):
    m = QMimeData()
    m.setUrls([QUrl("https://example.com/x.tif"),
               QUrl.fromLocalFile(str(tmp_path))])
    assert _mime_has_local_paths(m) is True


def test_mime_local_paths_drops_remote_urls(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(a)),
               QUrl("ftp://host/remote.tif"),
               QUrl.fromLocalFile(str(b))])
    assert _mime_local_paths(m) == [a, b]


# ---------------------------------------------------------------------------
# install_dropzone + the event filter
# ---------------------------------------------------------------------------

def test_dragenter_with_local_file_is_accepted(zone, tmp_path):
    w = zone(RecordingHandler())
    m = _mime([tmp_path])
    ev = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, m,
                         Qt.LeftButton, Qt.NoModifier)
    assert QApplication.sendEvent(w, ev) is True
    assert ev.isAccepted() is True


def test_dragenter_with_only_remote_urls_is_not_accepted(zone):
    w = zone(RecordingHandler())
    m = QMimeData()
    m.setUrls([QUrl("https://example.com/plate.tif")])
    ev = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, m,
                         Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(w, ev)
    assert ev.isAccepted() is False


def test_dragmove_is_accepted(zone, tmp_path):
    w = zone(RecordingHandler())
    m = _mime([tmp_path])
    QApplication.sendEvent(w, QDragEnterEvent(
        QPoint(5, 5), Qt.CopyAction, m, Qt.LeftButton, Qt.NoModifier))
    ev = QDragMoveEvent(QPoint(5, 5), Qt.CopyAction, m,
                        Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(w, ev)
    assert ev.isAccepted() is True


def test_filter_ignores_unrelated_event_types(zone, tmp_path):
    """The filter must not swallow ordinary events — it returns False."""
    w = zone(RecordingHandler())
    filt = [c for c in w.children()
            if type(c).__name__ == "_DropzoneFilter"][0]
    assert filt.eventFilter(w, QEvent(QEvent.User)) is False


def test_filter_ignores_events_for_other_objects(zone, tmp_path, qtbot):
    w = zone(RecordingHandler())
    other = QWidget()
    qtbot.addWidget(other)
    filt = [c for c in w.children()
            if type(c).__name__ == "_DropzoneFilter"][0]
    m = _mime([tmp_path])
    ev = QDropEvent(QPointF(1, 1), Qt.CopyAction, m,
                    Qt.LeftButton, Qt.NoModifier)
    assert filt.eventFilter(other, ev) is False
    assert ev.isAccepted() is False


# ---------------------------------------------------------------------------
# Drop routing
# ---------------------------------------------------------------------------

def test_accepted_folder_drop_calls_apply_and_accepts_event(zone, tmp_path,
                                                            msgbox):
    folder = tmp_path / "plate01"
    _mkimg(folder / "img.tif")
    h = RecordingHandler(accept=True)
    w = zone(h)
    ev = _drop(w, [folder])
    assert h.applied == [folder]
    assert ev.isAccepted() is True
    assert msgbox.calls == []


def test_drop_passes_the_screen_not_the_target(zone, tmp_path, qtbot):
    """install_dropzone can point at a different owner than the widget."""
    seen = {}

    class H(RecordingHandler):
        def apply(self, path, screen):
            seen["screen"] = screen

    owner = FakeScreen()
    w = zone(H(), screen=owner)
    _drop(w, [tmp_path])
    assert seen["screen"] is owner


def test_drop_with_no_local_paths_is_a_no_op(zone, tmp_path, msgbox):
    """A drag that entered with a file but drops only remote URLs."""
    h = RecordingHandler()
    w = zone(h)
    ev = _drop(w, [], remote=["https://example.com/plate.tif"],
               enter_paths=[tmp_path])
    assert h.asked == [] and h.applied == []
    assert ev.isAccepted() is False


def test_multi_folder_drop_is_truncated_when_handler_is_single(zone, tmp_path,
                                                                msgbox):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for d in (a, b, c):
        _mkimg(d / "img.tif")
    h = RecordingHandler(accept=True, multiple=False)
    w = zone(h)
    _drop(w, [a, b, c])
    assert h.applied == [a]


def test_multi_folder_drop_applies_all_when_handler_accepts_multiple(
        zone, tmp_path, msgbox):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for d in (a, b, c):
        _mkimg(d / "img.tif")
    h = RecordingHandler(accept=True, multiple=True)
    w = zone(h)
    _drop(w, [a, b, c])
    assert h.applied == [a, b, c]


def test_dropped_path_that_does_not_exist_is_reported(zone, tmp_path, msgbox):
    missing = tmp_path / "was_moved" / "gone"
    h = RecordingHandler(accept=False, alternatives=[])
    w = zone(h)
    _drop(w, [missing])
    assert h.applied == []
    assert msgbox.calls == [("information", "Nothing to drop into",
                             "cannot use gone")]


def test_rejected_drop_without_alternatives_shows_error_message(zone, tmp_path,
                                                                 msgbox):
    folder = tmp_path / "empty_plate"
    folder.mkdir()
    h = RecordingHandler(accept=False, alternatives=[])
    w = zone(h)
    _drop(w, [folder])
    assert msgbox.calls == [("information", "Nothing to drop into",
                             "cannot use empty_plate")]


def test_handler_apply_exception_becomes_a_warning(zone, tmp_path, msgbox):
    folder = tmp_path / "plate"
    _mkimg(folder / "img.tif")
    h = RecordingHandler(accept=True, raise_on_apply=True)
    w = zone(h)
    _drop(w, [folder])
    assert msgbox.calls == [("warning", "Drop failed", "boom: disk on fire")]


def test_rejected_drop_with_alternatives_applies_the_users_pick(
        zone, tmp_path, msgbox):
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    right = tmp_path / "right"
    _mkimg(right / "img.tif")
    other = tmp_path / "other"
    _mkimg(other / "img.tif")
    h = RecordingHandler(accept=False, alternatives=[right, other])
    w = zone(h)
    with _auto_modal("accept", select_row=1) as st:
        _drop(w, [wrong])
    assert st["seen"] >= 1
    assert h.applied == [other]            # row 1 of the alternatives
    assert msgbox.calls == []


def test_rejected_drop_with_alternatives_cancelled_applies_nothing(
        zone, tmp_path, msgbox):
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    right = tmp_path / "right"
    _mkimg(right / "img.tif")
    h = RecordingHandler(accept=False, alternatives=[right])
    w = zone(h)
    with _auto_modal("reject") as st:
        _drop(w, [wrong])
    assert st["seen"] >= 1
    assert h.applied == []
    assert msgbox.calls == []


def test_alternative_pick_that_fails_to_apply_warns(zone, tmp_path, msgbox):
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    right = tmp_path / "right"
    _mkimg(right / "img.tif")
    h = RecordingHandler(accept=False, alternatives=[right],
                         raise_on_apply=True)
    w = zone(h)
    with _auto_modal("accept"):
        _drop(w, [wrong])
    assert msgbox.calls == [("warning", "Drop failed", "boom: disk on fire")]


# ---------------------------------------------------------------------------
# CSV settings import through a real drop
# ---------------------------------------------------------------------------

def _write_settings_csv(path, rows, key_col="Key", val_col="Value"):
    lines = [f"{key_col},{val_col}"]
    lines += [f"{k},{v}" for k, v in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dropping_a_settings_csv_imports_it_and_accepts_the_drop(
        zone, tmp_path, msgbox):
    csv = _write_settings_csv(tmp_path / "settings.csv",
                              [("cell_diameter", "30"), ("verbose", "True")])
    h = RecordingHandler(accept=True)
    w = zone(h)
    ev = _drop(w, [csv])
    assert h.applied == []                       # CSVs never reach the handler
    assert w.applied == [{"cell_diameter": 30, "verbose": True}]
    assert "[drop] imported 2 settings from settings.csv" in w._console.text
    # Regression: a CSV-only drop used to leave the action unaccepted, so the
    # OS reported the (successful) import back to the user as a rejected drop.
    assert ev.isAccepted() is True


def test_dropping_csv_and_folder_together_does_both(zone, tmp_path, msgbox):
    csv = _write_settings_csv(tmp_path / "s.csv", [("nucleus_channel", "0")])
    folder = tmp_path / "plate"
    _mkimg(folder / "img.tif")
    h = RecordingHandler(accept=True)
    w = zone(h)
    _drop(w, [csv, folder])
    assert w.applied == [{"nucleus_channel": 0}]
    assert h.applied == [folder]


def test_a_directory_named_dot_csv_goes_to_the_handler(zone, tmp_path, msgbox):
    """`.csv` only routes to the settings importer when it's a real file."""
    fake = tmp_path / "exports.csv"
    fake.mkdir()
    h = RecordingHandler(accept=True)
    w = zone(h)
    _drop(w, [fake])
    assert h.applied == [fake]
    assert w.applied == []


# ---------------------------------------------------------------------------
# _apply_settings_csv directly
# ---------------------------------------------------------------------------

def test_apply_settings_csv_noop_when_screen_has_no_importer(tmp_path, msgbox):
    csv = _write_settings_csv(tmp_path / "s.csv", [("a", "1")])

    class Bare:
        pass

    _apply_settings_csv(csv, Bare())         # must not raise
    assert msgbox.calls == []


def test_apply_settings_csv_falls_back_to_setting_key_columns(tmp_path,
                                                               msgbox):
    """Regression: load_settings RAISES on a column mismatch, so the
    setting_key/setting_value fallback was unreachable and every CSV in that
    (equally valid) layout was reported as a failed import."""
    csv = _write_settings_csv(tmp_path / "old.csv",
                              [("cell_diameter", "42")],
                              key_col="setting_key", val_col="setting_value")
    screen = FakeScreen()
    _apply_settings_csv(csv, screen)
    assert screen.applied == [{"cell_diameter": 42}]
    assert msgbox.calls == []


def test_apply_settings_csv_warns_when_columns_are_unusable(tmp_path, msgbox):
    csv = tmp_path / "wrong.csv"
    csv.write_text("alpha,beta\n1,2\n")
    screen = FakeScreen()
    _apply_settings_csv(csv, screen)
    assert screen.applied == []
    assert len(msgbox.calls) == 1
    kind, title, text = msgbox.calls[0]
    assert (kind, title) == ("warning", "CSV import failed")
    assert "setting_key" in text


def test_apply_settings_csv_works_without_a_console(tmp_path, msgbox):
    csv = _write_settings_csv(tmp_path / "s.csv", [("a", "1"), ("b", "2")])

    class NoConsole:
        def __init__(self):
            self.applied = []

        def apply_settings_dict(self, d):
            self.applied.append(d)
            return len(d)

    screen = NoConsole()
    _apply_settings_csv(csv, screen)
    assert screen.applied == [{"a": 1, "b": 2}]
    assert msgbox.calls == []


# ---------------------------------------------------------------------------
# suggest_alternatives_dialog — the real dialog
# ---------------------------------------------------------------------------

def test_suggest_dialog_returns_first_row_by_default(qtbot, qt_theme_applied,
                                                     tmp_path):
    alts = [tmp_path / "one", tmp_path / "two"]
    with _auto_modal("accept"):
        pick = suggest_alternatives_dialog(None, tmp_path / "orig", alts,
                                           why="no images at top level")
    assert pick == alts[0]


def test_suggest_dialog_returns_the_selected_row(qtbot, qt_theme_applied,
                                                 tmp_path):
    alts = [tmp_path / "one", tmp_path / "two", tmp_path / "three"]
    with _auto_modal("accept", select_row=2):
        pick = suggest_alternatives_dialog(None, tmp_path / "orig", alts)
    assert pick == alts[2]


def test_suggest_dialog_returns_none_on_cancel(qtbot, qt_theme_applied,
                                                tmp_path):
    alts = [tmp_path / "one"]
    with _auto_modal("reject"):
        assert suggest_alternatives_dialog(None, tmp_path / "o", alts) is None


def test_suggest_dialog_returns_none_when_nothing_is_selected(
        qtbot, qt_theme_applied, tmp_path):
    """An empty alternatives list leaves currentRow() at -1."""
    with _auto_modal("accept"):
        assert suggest_alternatives_dialog(None, tmp_path / "o", []) is None


def test_suggest_dialog_lists_every_alternative(qtbot, qt_theme_applied,
                                                 tmp_path):
    alts = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    captured = {}

    t = QTimer()
    t.setInterval(5)

    def tick():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            return
        lw = dlg.findChildren(QListWidget)[0]
        captured["items"] = [lw.item(i).text() for i in range(lw.count())]
        captured["title"] = dlg.windowTitle()
        dlg.reject()

    t.timeout.connect(tick)
    t.start()
    try:
        suggest_alternatives_dialog(None, tmp_path / "orig", alts, why="why!")
    finally:
        t.stop()
    assert captured["items"] == [str(a) for a in alts]
    assert captured["title"] == "Did you mean…"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def test_has_images_in_respects_min_count(tmp_path):
    _mkimg(tmp_path / "a.tif")
    _mkimg(tmp_path / "b.tif")
    assert has_images_in(tmp_path, min_count=2) is True
    assert has_images_in(tmp_path, min_count=3) is False


def test_has_images_in_ignores_subfolders(tmp_path):
    _mkimg(tmp_path / "sub" / "a.tif")
    assert has_images_in(tmp_path) is False


def test_has_images_in_honours_custom_extension_list(tmp_path):
    _mkimg(tmp_path / "a.npy")
    assert has_images_in(tmp_path) is False
    assert has_images_in(tmp_path, exts=(".npy",)) is True


def test_has_images_in_matches_every_default_extension(tmp_path):
    from spacr.qt.dnd import IMAGE_EXTS
    for i, ext in enumerate(IMAGE_EXTS):
        d = tmp_path / f"d{i}"
        _mkimg(d / f"img{ext.upper()}")     # upper-case → suffix is lowered
        assert has_images_in(d) is True, ext


def test_has_images_in_false_for_missing_path(tmp_path):
    assert has_images_in(tmp_path / "nope") is False


def test_find_image_folders_nearby_returns_siblings_and_children(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    _mkimg(tmp_path / "sib" / "a.tif")
    _mkimg(target / "child" / "b.tif")
    hits = find_image_folders_nearby(target)
    assert set(hits) == {tmp_path / "sib", target / "child"}


def test_find_image_folders_nearby_excludes_the_path_itself(tmp_path):
    target = tmp_path / "target"
    _mkimg(target / "a.tif")
    assert target not in find_image_folders_nearby(target)


def test_find_image_folders_nearby_on_a_file_still_checks_siblings(tmp_path):
    _mkimg(tmp_path / "sib" / "a.tif")
    f = _mkimg(tmp_path / "loose.tif")
    assert find_image_folders_nearby(f) == [tmp_path / "sib"]


def test_find_image_folders_nearby_skips_a_missing_parent(tmp_path):
    """A path whose parent doesn't exist can't be scanned for siblings."""
    orphan = tmp_path / "gone" / "child"
    assert find_image_folders_nearby(orphan) == []


def test_find_image_folders_nearby_honours_min_count(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    _mkimg(tmp_path / "one_img" / "a.tif")
    two = tmp_path / "two_imgs"
    _mkimg(two / "a.tif")
    _mkimg(two / "b.tif")
    assert find_image_folders_nearby(target, min_count=2) == [two]


def test_sample_image_names_is_sorted_and_capped(tmp_path):
    for i in range(12):
        _mkimg(tmp_path / f"img_{i:02d}.tif")
    _mkimg(tmp_path / "notes.txt")
    hits = sample_image_names(tmp_path, n=4)
    assert [p.name for p in hits] == ["img_00.tif", "img_01.tif",
                                      "img_02.tif", "img_03.tif"]


def test_sample_image_names_returns_all_when_fewer_than_n(tmp_path):
    _mkimg(tmp_path / "a.tif")
    _mkimg(tmp_path / "sub" / "deep.tif")
    assert [p.name for p in sample_image_names(tmp_path, n=8)] == ["a.tif"]


def test_sample_image_names_empty_for_a_file(tmp_path):
    f = _mkimg(tmp_path / "a.tif")
    assert sample_image_names(f) == []
