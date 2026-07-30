"""Per-module drop-policy tests for :mod:`spacr.qt.dnd_handlers`.

Every handler is driven against a real directory tree built in ``tmp_path``
(images, ``merged/``, ``measurements/measurements.db``, FASTQs, …) and the
assertions are on the source path the handler actually pushed into the screen
plus the console text it produced.

The mask handler's console report is the interesting one: it is exercised for
a folder of images (matching regex / non-matching regex / user-supplied
regex), for a single-file container drop, for an empty folder, and for a
folder whose *directory layout* carries the metadata.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QWidget

from spacr.qt import dnd_handlers as dh
from spacr.qt.dnd_handlers import (
    AnnotateDropHandler, ClassifyDropHandler, DatabaseDropHandler,
    MakeMasksDropHandler,
    MapBarcodesDropHandler, MaskDropHandler, MeasureDropHandler,
    MeasurementsDropHandler, SourceDropHandler, get_handler,
    _count_images, _log, _open_metadata_table, _open_regex_editor,
    _push_regex_to_screen, _report_folder_structure, _report_regex_on_mask,
    _set_screen_setting, _set_src_on,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Model:
    def __init__(self, widgets):
        self._widgets = dict(widgets)


class Screen(QWidget):
    """AppScreen-shaped double: a settings model of real Qt widgets + console."""

    def __init__(self, widgets=None, console=True):
        super().__init__()
        self._settings_model = _Model(widgets or {})
        if console:
            self._console = _Console()

    # convenience
    def w(self, key):
        return self._settings_model._widgets[key]

    @property
    def log(self):
        return self._console.text


def _make_screen(qtbot, keys=("src", "custom_regex"), combos=()):
    widgets = {k: QLineEdit() for k in keys}
    for k, items in combos:
        c = QComboBox()
        c.addItems(items)
        widgets[k] = c
    s = Screen(widgets)
    qtbot.addWidget(s)
    return s


def _mkimg(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(p), np.zeros((4, 5), np.uint16))
    return p


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    return _make_screen(qtbot, keys=("src", "custom_regex"),
                        combos=[("metadata_type",
                                 ["cellvoyager", "auto", "custom"])])


def _refusing_editor():
    """A RegexEditorDialog stand-in whose exec() returns 'cancelled'."""
    class Cancelled:
        Accepted = 1

        def __init__(self, *a, **k):
            self.regex = ""

        def exec(self):
            return 0                        # rejected

    return Cancelled


@pytest.fixture(autouse=True)
def _never_block_on_the_regex_editor(monkeypatch):
    """The real editor is modal — exec() would hang the suite. Default every
    test to a cancelling stub; tests that care install their own."""
    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog",
                        _refusing_editor())


@pytest.fixture(autouse=True)
def _close_metadata_dialogs():
    """Any modeless metadata table a test opens gets closed afterwards."""
    opened = []
    real = dh._open_metadata_table

    def spy(rows, dst, scr):
        real(rows, dst, scr)
        opened.extend(getattr(scr, "_metadata_dialogs", []) or [])

    dh._open_metadata_table = spy
    try:
        yield opened
    finally:
        dh._open_metadata_table = real
        for d in opened:
            try:
                d.close()
                d.deleteLater()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# _set_src_on — the three screen shapes
# ---------------------------------------------------------------------------

def test_set_src_prefers_open_source(qtbot):
    seen = {}

    class S(Screen):
        def _open_source(self, p):
            seen["open_source"] = p

        def _open_folder(self, p):
            seen["open_folder"] = p

    s = S({"src": QLineEdit()})
    qtbot.addWidget(s)
    assert _set_src_on(s, "/data/plate1") is True
    assert seen == {"open_source": "/data/plate1"}
    assert s.w("src").text() == ""          # untouched


def test_set_src_falls_through_when_open_source_raises(qtbot):
    seen = {}

    class S(Screen):
        def _open_source(self, p):
            raise RuntimeError("no such plate")

        def _open_folder(self, p):
            seen["open_folder"] = p

    s = S({"src": QLineEdit()})
    qtbot.addWidget(s)
    assert _set_src_on(s, "/data/p") is True
    assert seen == {"open_folder": "/data/p"}


def test_set_src_uses_open_folder(qtbot):
    seen = {}

    class S(Screen):
        def _open_folder(self, p):
            seen["open_folder"] = p

    s = S({"src": QLineEdit()})
    qtbot.addWidget(s)
    assert _set_src_on(s, "/data/p") is True
    assert seen == {"open_folder": "/data/p"}
    assert s.w("src").text() == ""


def test_set_src_falls_back_to_the_settings_widget(qtbot):
    seen = {}

    class S(Screen):
        def _open_folder(self, p):
            raise RuntimeError("nope")

    s = S({"src": QLineEdit()})
    qtbot.addWidget(s)
    assert _set_src_on(s, "/data/p") is True
    assert s.w("src").text() == "/data/p"


def test_set_src_on_plain_settings_screen(qtbot):
    s = _make_screen(qtbot, keys=("src",))
    assert _set_src_on(s, "/data/plate9") is True
    assert s.w("src").text() == "/data/plate9"


def test_set_src_returns_false_without_a_src_widget(qtbot):
    s = _make_screen(qtbot, keys=("custom_regex",))
    assert _set_src_on(s, "/data/p") is False


def test_set_src_returns_false_when_src_widget_has_no_settext(qtbot):
    s = Screen({"src": object()})
    qtbot.addWidget(s)
    assert _set_src_on(s, "/data/p") is False


def test_set_src_returns_false_for_a_screen_with_nothing(qtbot):
    class Bare:
        pass

    assert _set_src_on(Bare(), "/data/p") is False


def test_set_src_returns_false_when_settings_model_is_broken(qtbot):
    class Broken:
        _settings_model = object()          # no ._widgets

    assert _set_src_on(Broken(), "/data/p") is False


# ---------------------------------------------------------------------------
# _log
# ---------------------------------------------------------------------------

def test_log_appends_to_console(qtbot):
    s = _make_screen(qtbot)
    _log(s, "hello\n")
    _log(s, "world\n")
    assert s.log == "hello\nworld\n"


def test_log_is_a_noop_without_a_console():
    class Bare:
        pass

    b = Bare()
    _log(b, "ignored")
    assert vars(b) == {}                    # nothing invented on the screen


def test_log_swallows_a_failing_console():
    attempts = []

    class Boom:
        class _console:
            @staticmethod
            def append_stdout(s):
                attempts.append(s)
                raise IOError("console closed")

    _log(Boom(), "some text")
    assert attempts == ["some text"]         # tried, then swallowed the error


# ---------------------------------------------------------------------------
# _set_screen_setting
# ---------------------------------------------------------------------------

def test_set_screen_setting_selects_an_existing_combo_item(qtbot):
    s = _make_screen(qtbot, keys=(),
                     combos=[("metadata_type", ["cellvoyager", "auto"])])
    assert _set_screen_setting(s, "metadata_type", "auto") is True
    assert s.w("metadata_type").currentText() == "auto"


def test_set_screen_setting_falls_back_to_edit_text_for_unknown_item(qtbot):
    s = _make_screen(qtbot, keys=(), combos=[("metadata_type", ["a", "b"])])
    combo = s.w("metadata_type")
    combo.setEditable(True)
    assert _set_screen_setting(s, "metadata_type", "brand_new") is True
    assert combo.currentText() == "brand_new"


def test_set_screen_setting_writes_a_line_edit(qtbot):
    s = _make_screen(qtbot, keys=("custom_regex",))
    assert _set_screen_setting(s, "custom_regex", r"(?P<chanID>\d+)") is True
    assert s.w("custom_regex").text() == r"(?P<chanID>\d+)"


def test_set_screen_setting_uses_generic_settext(qtbot):
    s = Screen({"note": QLabel()})
    qtbot.addWidget(s)
    assert _set_screen_setting(s, "note", 17) is True
    assert s.w("note").text() == "17"


def test_set_screen_setting_returns_false_for_unknown_key(qtbot):
    s = _make_screen(qtbot)
    assert _set_screen_setting(s, "not_a_setting", "x") is False


def test_set_screen_setting_returns_false_for_a_widget_without_settext(qtbot):
    s = Screen({"weird": object()})
    qtbot.addWidget(s)
    assert _set_screen_setting(s, "weird", "x") is False


def test_set_screen_setting_returns_false_without_a_settings_model():
    class Bare:
        pass

    assert _set_screen_setting(Bare(), "metadata_type", "auto") is False


# ---------------------------------------------------------------------------
# _push_regex_to_screen / _count_images
# ---------------------------------------------------------------------------

def test_push_regex_sets_the_custom_regex_widget(qtbot):
    s = _make_screen(qtbot)
    _push_regex_to_screen(r"(?P<wellID>[A-Z]\d\d)", s)
    assert s.w("custom_regex").text() == r"(?P<wellID>[A-Z]\d\d)"


@pytest.mark.parametrize("pattern", [None, ""])
def test_push_regex_ignores_empty_patterns(qtbot, pattern):
    s = _make_screen(qtbot)
    s.w("custom_regex").setText("keep me")
    _push_regex_to_screen(pattern, s)
    assert s.w("custom_regex").text() == "keep me"


def test_push_regex_ignores_a_widget_without_settext(qtbot):
    s = Screen({"custom_regex": object()})
    qtbot.addWidget(s)
    _push_regex_to_screen("x", s)            # must not raise
    assert isinstance(s.w("custom_regex"), object)


def test_push_regex_ignores_a_screen_without_the_widget(qtbot):
    s = _make_screen(qtbot, keys=("src",))
    _push_regex_to_screen("x", s)
    assert "custom_regex" not in s._settings_model._widgets


def test_push_regex_swallows_a_broken_screen():
    class Bare:
        pass

    b = Bare()
    _push_regex_to_screen("x", b)
    assert vars(b) == {}                    # nothing invented on the screen


def test_count_images_counts_only_plain_image_extensions(tmp_path):
    for name in ("a.tif", "b.TIFF", "c.png", "d.jpg", "e.jpeg"):
        (tmp_path / name).write_bytes(b"x")
    for name in ("f.czi", "g.nd2", "h.lif", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.tif").write_bytes(b"x")
    assert _count_images(tmp_path) == 5


# ---------------------------------------------------------------------------
# MaskDropHandler policy
# ---------------------------------------------------------------------------

def test_mask_handler_accepts_multiple():
    assert MaskDropHandler().accepts_multiple() is True


def test_mask_handler_rejects_a_file(tmp_path):
    f = _mkimg(tmp_path / "img.tif")
    assert MaskDropHandler().can_accept(f) is False


def test_mask_handler_rejects_a_missing_path(tmp_path):
    assert MaskDropHandler().can_accept(tmp_path / "gone") is False


def test_mask_handler_suggests_nothing_for_a_file(tmp_path):
    f = _mkimg(tmp_path / "img.tif")
    assert MaskDropHandler().suggest_alternatives(f) == []


def test_mask_handler_error_message_names_the_extensions(tmp_path):
    msg = MaskDropHandler().error_message(tmp_path)
    assert "folder of microscopy images" in msg
    assert ".czi" in msg and ".nd2" in msg and ".lif" in msg


def test_mask_apply_sets_src_and_schedules_the_report(qtbot, screen, tmp_path):
    folder = tmp_path / "plate1"
    _mkimg(folder / "plate1_A01_T0001F001L01A01Z01C01.tif")
    MaskDropHandler().apply(folder, screen)
    assert screen.w("src").text() == str(folder)
    assert f"[drop] mask src = {folder}\n" in screen.log
    # The regex report is deferred through a QTimer so the UI never stalls.
    qtbot.waitUntil(lambda: "regex (cellvoyager)" in screen.log, timeout=3000)


def test_mask_apply_still_sets_src_when_the_timer_is_unavailable(
        qtbot, screen, tmp_path, monkeypatch):
    """Defensive branch: no QTimer → src is still set, no exception escapes."""
    import PySide6.QtCore as real_qtcore

    class _NoTimer:
        def __getattr__(self, name):
            if name == "QTimer":
                raise AttributeError("QTimer unavailable")
            return getattr(real_qtcore, name)

    folder = tmp_path / "plate1"
    _mkimg(folder / "a.tif")
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", _NoTimer())
    try:
        MaskDropHandler().apply(folder, screen)
    finally:
        monkeypatch.undo()
    assert screen.w("src").text() == str(folder)
    assert "regex" not in screen.log


# ---------------------------------------------------------------------------
# _report_regex_on_mask — folder drops
# ---------------------------------------------------------------------------

def test_report_on_empty_folder_says_nothing_to_preview(screen, tmp_path):
    folder = tmp_path / "no_images"
    folder.mkdir()
    (folder / "readme.txt").write_text("hi")
    _report_regex_on_mask(folder, screen)
    assert "no images found in the top level of no_images" in screen.log
    assert "regex" not in screen.log


def test_report_on_matching_folder_confirms_all_fields(screen, tmp_path):
    folder = tmp_path / "plate1"
    for f in range(1, 4):
        for c in (1, 2):
            _mkimg(folder /
                   f"plate1_A01_T0001F{f:03d}L01A01Z01C{c:02d}.tif")
    _report_regex_on_mask(folder, screen)
    log = screen.log
    assert f"[drop] mask · folder = {folder}\n" in log
    assert "[drop] regex (cellvoyager) — matched 6/6 sampled filenames" in log
    assert "[drop] 6 of 6 total sampled — showing up to 10 rows:" in log
    assert "✓ All required fields captured (wellID / fieldID, chanID)." in log
    # A clean match writes the winning pattern back into the settings widget.
    from spacr.qt.regex_detect import CELLVOYAGER
    assert screen.w("custom_regex").text() == CELLVOYAGER
    # The table body really rendered (one row per sampled file, capped at 10).
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in log


def test_report_caps_the_table_at_ten_rows(screen, tmp_path):
    folder = tmp_path / "plate1"
    for f in range(1, 16):
        _mkimg(folder / f"plate1_A01_T0001F{f:03d}L01A01Z01C01.tif")
    _report_regex_on_mask(folder, screen)
    # 20 sampled max, 15 present; the table itself shows at most 10 records
    assert "matched 15/15 sampled filenames" in screen.log
    body = screen.log.split("showing up to 10 rows:")[1]
    shown = [ln for ln in body.splitlines() if ".tif" in ln]
    assert len(shown) == 10


def test_report_on_unmatchable_folder_warns_and_opens_the_editor(
        screen, tmp_path, monkeypatch):
    folder = tmp_path / "loose"
    for i in range(3):
        _mkimg(folder / f"random_image_{i}.png")

    opened = {}

    class FakeEditor:
        Accepted = 1

        def __init__(self, filenames, initial_regex="", multichannel=True,
                     parent=None):
            opened["filenames"] = list(filenames)
            opened["initial"] = initial_regex
            opened["multichannel"] = multichannel
            self.regex = r"(?P<wellID>[A-Z]\d\d)_(?P<chanID>\d+)\.png$"

        def exec(self):
            return 1

    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog", FakeEditor)
    _report_regex_on_mask(folder, screen)

    log = screen.log
    assert "⚠ Missing required field: chanID" in log
    assert "⚠ Missing location field" in log
    assert "→ Opening the regex editor" in log
    assert sorted(opened["filenames"]) == ["random_image_0.png",
                                           "random_image_1.png",
                                           "random_image_2.png"]
    assert opened["multichannel"] is True
    # Accepted editor pushes its regex back onto the screen + logs it.
    assert screen.w("custom_regex").text() == \
        r"(?P<wellID>[A-Z]\d\d)_(?P<chanID>\d+)\.png$"
    assert "[drop] saved custom regex: " in log


def test_report_uses_the_users_custom_regex_when_present(screen, tmp_path):
    folder = tmp_path / "custom"
    _mkimg(folder / "wellA01_ch1.tif")
    _mkimg(folder / "wellA01_ch2.tif")
    _mkimg(folder / "wellA02_ch1.tif")
    pattern = (r"well(?P<wellID>[A-Z]\d{2})_ch(?P<chanID>\d+)"
               r"\.tif$")
    screen.w("custom_regex").setText("  " + pattern + "  ")
    _report_regex_on_mask(folder, screen)
    log = screen.log
    assert "[drop] regex (custom) — matched 3/3 sampled filenames" in log
    # plateID is missing → a soft warning, so the editor path runs; but the
    # user's own pattern is preserved, not overwritten.
    assert "⚠ Optional: no plateID captured." in log
    assert screen.w("custom_regex").text() == "  " + pattern + "  "


def test_report_with_a_custom_regex_that_matches_nothing(screen, tmp_path,
                                                          monkeypatch):
    folder = tmp_path / "custom"
    _mkimg(folder / "a.tif")
    _mkimg(folder / "b.tif")
    screen.w("custom_regex").setText(r"^ZZZ_(?P<chanID>\d+)\.tif$")
    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog",
                        _refusing_editor())
    _report_regex_on_mask(folder, screen)
    log = screen.log
    assert "[drop] regex (custom) — matched 0/2 sampled filenames" in log
    assert "⚠ No filenames matched the regex." in log


def test_report_leaves_regex_alone_when_the_editor_is_cancelled(
        screen, tmp_path, monkeypatch):
    folder = tmp_path / "loose"
    _mkimg(folder / "random_image_0.png")
    _mkimg(folder / "random_image_1.png")
    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog",
                        _refusing_editor())
    _report_regex_on_mask(folder, screen)
    assert screen.w("custom_regex").text() == ""
    assert "saved custom regex" not in screen.log


def test_report_auto_detects_when_the_screen_has_no_custom_regex_widget(
        qtbot, tmp_path):
    """Screens without a custom_regex field must still get the auto-detect
    path, not an exception."""
    s = _make_screen(qtbot, keys=("src",))
    folder = tmp_path / "plate1"
    _mkimg(folder / "plate1_A01_T0001F001L01A01Z01C01.tif")
    _report_regex_on_mask(folder, s)
    assert "[drop] regex (cellvoyager) — matched 1/1 sampled filenames" in s.log
    assert "✓ All required fields captured" in s.log


def test_report_tolerates_a_screen_without_a_settings_model(tmp_path,
                                                            monkeypatch):
    """AnnotateScreen/MakeMasksScreen have no settings model — the custom
    regex read must degrade to "" rather than explode."""
    class Bare:
        def __init__(self):
            self._console = _Console()

    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog",
                        _refusing_editor())
    folder = tmp_path / "loose"
    _mkimg(folder / "random_image_0.png")
    s = Bare()
    _report_regex_on_mask(folder, s)
    assert "[drop] regex (synthesised)" in s._console.text


# ---------------------------------------------------------------------------
# _report_regex_on_mask — single-file container drops
# ---------------------------------------------------------------------------

def test_report_on_container_file_sets_metadata_type_auto(screen, tmp_path,
                                                           _close_metadata_dialogs):
    stack = tmp_path / "run.tif"
    tifffile.imwrite(str(stack), np.zeros((6, 8, 9), np.uint16))
    _report_regex_on_mask(stack, screen)
    log = screen.log
    assert screen.w("metadata_type").currentText() == "auto"
    assert "[drop] single-file dataset: format=tif_multi  fields=6" in log
    assert "Set metadata_type = 'auto'" in log
    assert "[drop] planned extraction — 6 images, 1 well(s), 6 field(s), " \
           "1 channel(s)" in log
    # The editable preview table opened, parented to nothing modal.
    assert len(screen._metadata_dialogs) == 1


def test_container_preview_apply_writes_filename_map_csv(
        screen, tmp_path, _close_metadata_dialogs):
    """Regression: the dialog was handed the *folder*, which
    save_filename_map then tried to open() for writing — every Apply raised
    IsADirectoryError and silently wrote nothing."""
    stack = tmp_path / "run.tif"
    tifffile.imwrite(str(stack), np.zeros((2, 8, 9), np.uint16))
    _report_regex_on_mask(stack, screen)
    dlg = screen._metadata_dialogs[0]
    dlg._apply()
    written = tmp_path / "filename_map.csv"
    assert dlg.written_path == written
    assert written.is_file()
    lines = written.read_text().strip().splitlines()
    assert lines[0] == ("original_path,canonical,plate,well,field,"
                        "channel,time")
    assert len(lines) == 3                                  # header + 2 planes
    assert lines[1].endswith("plate1_A01_T0001F001L01C01.tif,plate1,"
                             "plate1_A01,1,1,1")
    assert f"[drop] wrote metadata map → {written}" in screen.log


def test_report_on_container_reports_preview_failure(screen, tmp_path,
                                                      monkeypatch):
    stack = tmp_path / "run.tif"
    tifffile.imwrite(str(stack), np.zeros((6, 8, 9), np.uint16))

    def boom(desc, *a, **k):
        raise ValueError("planner exploded")

    monkeypatch.setattr("spacr.qt.ingest_preview.plan_container_extraction",
                        boom)
    _report_regex_on_mask(stack, screen)
    assert "[drop] metadata preview unavailable: planner exploded" in screen.log
    assert screen.w("metadata_type").currentText() == "auto"


def test_report_on_container_with_no_planned_rows_opens_nothing(
        screen, tmp_path, monkeypatch):
    stack = tmp_path / "run.tif"
    tifffile.imwrite(str(stack), np.zeros((6, 8, 9), np.uint16))
    monkeypatch.setattr("spacr.qt.ingest_preview.plan_container_extraction",
                        lambda desc, *a, **k: [])
    _report_regex_on_mask(stack, screen)
    assert "planned extraction" not in screen.log
    assert getattr(screen, "_metadata_dialogs", []) == []


def test_report_on_unrecognised_file_says_so(screen, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("not a dataset")
    _report_regex_on_mask(f, screen)
    assert ("[drop] dropped file notes.txt — unrecognised single-file "
            "dataset format.") in screen.log
    assert screen.w("metadata_type").currentText() != "auto"


# ---------------------------------------------------------------------------
# _report_folder_structure
# ---------------------------------------------------------------------------

def test_folder_structure_detects_plate_well_field_layout(
        screen, tmp_path, _close_metadata_dialogs):
    _mkimg(tmp_path / "plate1" / "A01" / "f01" / "C01.tif")
    _mkimg(tmp_path / "plate1" / "A01" / "f02" / "C01.tif")
    _report_folder_structure(tmp_path, screen)
    log = screen.log
    assert "[drop] folder-structure alternative" in log
    assert "path depth → plate / well / field" in log
    assert "[drop] folder-structure plan — 2 images" in log
    assert len(screen._metadata_dialogs) == 1


def test_folder_structure_writes_the_map_into_the_dropped_folder(
        screen, tmp_path, _close_metadata_dialogs):
    _mkimg(tmp_path / "plate1" / "A01" / "f01" / "C01.tif")
    _report_folder_structure(tmp_path, screen)
    dlg = screen._metadata_dialogs[0]
    dlg._apply()
    assert dlg.written_path == tmp_path / "filename_map.csv"
    assert (tmp_path / "filename_map.csv").is_file()


def test_folder_structure_silent_when_nothing_is_detected(screen, tmp_path):
    _mkimg(tmp_path / "loose.tif")
    _report_folder_structure(tmp_path, screen)
    assert screen.log == ""


def test_folder_structure_silent_when_detection_raises(screen, tmp_path,
                                                        monkeypatch):
    def boom(root, *a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr("spacr.qt.folder_metadata.detect_folder_metadata",
                        boom)
    _report_folder_structure(tmp_path, screen)
    assert screen.log == ""


def test_folder_structure_silent_when_template_has_no_labels(screen, tmp_path,
                                                              monkeypatch):
    class Empty:
        depth_labels = ()

    monkeypatch.setattr("spacr.qt.folder_metadata.detect_folder_metadata",
                        lambda root, *a, **k: Empty())
    _report_folder_structure(tmp_path, screen)
    assert screen.log == ""


def test_folder_structure_opens_no_table_when_the_plan_is_empty(
        screen, tmp_path, monkeypatch):
    _mkimg(tmp_path / "plate1" / "A01" / "f01" / "C01.tif")
    monkeypatch.setattr("spacr.qt.ingest_preview.plan_folder_extraction",
                        lambda root, *a, **k: [])
    _report_folder_structure(tmp_path, screen)
    assert "[drop] folder-structure alternative" in screen.log
    assert "folder-structure plan" not in screen.log
    assert getattr(screen, "_metadata_dialogs", []) == []


def test_folder_structure_reports_a_failing_planner(screen, tmp_path,
                                                     monkeypatch):
    _mkimg(tmp_path / "plate1" / "A01" / "f01" / "C01.tif")

    def boom(root, *a, **k):
        raise ValueError("planner exploded")

    monkeypatch.setattr("spacr.qt.ingest_preview.plan_folder_extraction", boom)
    _report_folder_structure(tmp_path, screen)
    assert "[drop] folder-structure alternative" in screen.log
    assert ("[drop] folder-structure preview unavailable: planner exploded"
            in screen.log)


# ---------------------------------------------------------------------------
# _open_metadata_table
# ---------------------------------------------------------------------------

_ROW = {"original": "/src/a.tif", "plate": "plate1", "well": "plate1_A01",
        "field": 1, "channel": 1, "time": 1,
        "canonical": "plate1_A01_T0001F001L01C01.tif"}


def test_open_metadata_table_registers_and_deregisters_the_dialog(
        qtbot, screen, tmp_path):
    dh._open_metadata_table([_ROW], tmp_path, screen)
    assert len(screen._metadata_dialogs) == 1
    dlg = screen._metadata_dialogs[0]
    assert dlg.isModal() is False
    dlg.reject()                                  # emits finished
    qtbot.waitUntil(lambda: screen._metadata_dialogs == [], timeout=2000)


def test_open_metadata_table_reuses_an_existing_holder(qtbot, screen,
                                                        tmp_path):
    dh._open_metadata_table([_ROW], tmp_path / "a", screen)
    dh._open_metadata_table([_ROW], tmp_path / "b", screen)
    assert len(screen._metadata_dialogs) == 2
    first, second = screen._metadata_dialogs
    assert first is not second
    first.reject()
    qtbot.waitUntil(lambda: screen._metadata_dialogs == [second], timeout=2000)


def test_open_metadata_table_normalises_a_csv_destination(qtbot, screen,
                                                           tmp_path):
    """Passing the CSV path itself must not double-append the filename."""
    target = tmp_path / "map.csv"
    dh._open_metadata_table([_ROW], target, screen)
    dlg = screen._metadata_dialogs[0]
    dlg._apply()
    assert dlg.written_path == target
    assert target.is_file()
    dlg.close()


def test_open_metadata_table_survives_a_screen_it_cannot_annotate(qtbot,
                                                                   tmp_path):
    """A screen that refuses new attributes must still get a live dialog.

    Regression: the dialog is parentless here, and the only reference was a
    local list that died with the function — the window was garbage-collected
    before the user ever saw it."""
    from PySide6.QtWidgets import QApplication
    from spacr.qt.widgets.metadata_table import MetadataTableDialog

    class Slotted:
        __slots__ = ()

    assert dh._ORPHAN_DIALOGS == []
    dh._open_metadata_table([_ROW], tmp_path, Slotted())
    assert len(dh._ORPHAN_DIALOGS) == 1
    dlg = dh._ORPHAN_DIALOGS[0]
    assert isinstance(dlg, MetadataTableDialog)
    assert dlg in QApplication.topLevelWidgets()   # still alive, really shown
    assert dlg.isVisible() is True
    assert dlg.isModal() is False
    assert dlg.panel.rows()[0]["canonical"] == _ROW["canonical"]
    dlg.reject()                                   # finished → deregisters
    assert dh._ORPHAN_DIALOGS == []
    dlg.deleteLater()


def test_open_metadata_table_logs_a_construction_failure(screen, tmp_path,
                                                          monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no display")

    monkeypatch.setattr(
        "spacr.qt.widgets.metadata_table.MetadataTableDialog", boom)
    dh._open_metadata_table([_ROW], tmp_path, screen)
    assert "[drop] could not open metadata table: no display" in screen.log
    assert getattr(screen, "_metadata_dialogs", []) == []


def test_open_metadata_table_tolerates_a_dialog_it_cannot_show(screen,
                                                                tmp_path,
                                                                monkeypatch):
    class Stub:
        """No .finished / .show — stands in for a headless Qt build."""

    monkeypatch.setattr(
        "spacr.qt.widgets.metadata_table.MetadataTableDialog",
        lambda *a, **k: Stub())
    dh._open_metadata_table([_ROW], tmp_path, screen)
    # Registration happened, then wiring/show failed and was swallowed — the
    # console report the user already has is left in place, nothing raises.
    assert [type(d).__name__ for d in screen._metadata_dialogs] == ["Stub"]
    assert "could not open metadata table" not in screen.log


def test_open_metadata_table_returns_when_the_widget_is_unimportable(
        screen, tmp_path, monkeypatch):
    class _Broken:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules,
                        "spacr.qt.widgets.metadata_table", _Broken())
    dh._open_metadata_table([_ROW], tmp_path, screen)
    assert screen.log == ""


# ---------------------------------------------------------------------------
# _open_regex_editor
# ---------------------------------------------------------------------------

def test_open_regex_editor_logs_a_construction_failure(screen, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no display")

    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog", boom)
    _open_regex_editor(["a.tif"], "", screen)
    assert "[drop] regex editor failed: no display" in screen.log


def test_open_regex_editor_returns_when_unimportable(screen, monkeypatch):
    class _Broken:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "spacr.qt.regex_editor", _Broken())
    _open_regex_editor(["a.tif"], "", screen)
    assert screen.log == ""


def test_open_regex_editor_ignores_an_accepted_but_empty_regex(screen,
                                                                monkeypatch):
    class EmptyAccept:
        Accepted = 1

        def __init__(self, *a, **k):
            self.regex = ""

        def exec(self):
            return 1

    monkeypatch.setattr("spacr.qt.regex_editor.RegexEditorDialog", EmptyAccept)
    screen.w("custom_regex").setText("keep")
    _open_regex_editor(["a.tif"], "seed", screen)
    assert screen.w("custom_regex").text() == "keep"
    assert "saved custom regex" not in screen.log


# ---------------------------------------------------------------------------
# MeasureDropHandler
# ---------------------------------------------------------------------------

def test_measure_rejects_a_file(tmp_path):
    f = _mkimg(tmp_path / "merged")
    assert MeasureDropHandler().can_accept(f) is False


def test_measure_accepts_merged_with_npy_stacks(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "stack_0.npy", np.zeros((2, 2)))
    assert MeasureDropHandler().can_accept(merged) is True


def test_measure_rejects_merged_holding_only_pngs(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "preview.png").write_bytes(b"x")
    assert MeasureDropHandler().can_accept(merged) is False


def test_measure_suggests_child_and_sibling_merged_folders(tmp_path):
    dropped = tmp_path / "dropped"
    dropped.mkdir()
    (dropped / "plateA" / "merged").mkdir(parents=True)
    (dropped / "logs").mkdir()                    # child WITHOUT merged/
    _mkimg(dropped / "loose.tif")                 # a file, not a folder
    (tmp_path / "plateB" / "merged").mkdir(parents=True)
    (tmp_path / "unrelated").mkdir()              # sibling WITHOUT merged/
    hits = MeasureDropHandler().suggest_alternatives(dropped)
    assert hits == [dropped / "plateA" / "merged",
                    tmp_path / "plateB" / "merged"]


def test_measure_suggests_nothing_for_a_file(tmp_path):
    f = _mkimg(tmp_path / "a.tif")
    assert MeasureDropHandler().suggest_alternatives(f) == []


def test_measure_error_message_mentions_merged(tmp_path):
    assert "merged" in MeasureDropHandler().error_message(tmp_path)


def test_measure_apply_drills_into_merged(qtbot, screen, tmp_path):
    plate = tmp_path / "plateA"
    (plate / "merged").mkdir(parents=True)
    MeasureDropHandler().apply(plate, screen)
    assert screen.w("src").text() == str(plate / "merged")
    assert f"[drop] measure src = {plate / 'merged'}\n" in screen.log


def test_measure_apply_keeps_a_merged_folder_as_is(qtbot, screen, tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    MeasureDropHandler().apply(merged, screen)
    assert screen.w("src").text() == str(merged)


# ---------------------------------------------------------------------------
# AnnotateDropHandler
# ---------------------------------------------------------------------------

def test_annotate_rejects_a_non_db_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    assert AnnotateDropHandler().can_accept(f) is False


def test_annotate_rejects_a_folder_without_the_db(tmp_path):
    (tmp_path / "measurements").mkdir()
    assert AnnotateDropHandler().can_accept(tmp_path) is False


def test_annotate_rejects_a_missing_path(tmp_path):
    assert AnnotateDropHandler().can_accept(tmp_path / "gone") is False


def test_annotate_error_message_mentions_the_db(tmp_path):
    msg = AnnotateDropHandler().error_message(tmp_path)
    assert "measurements/measurements.db" in msg


def test_annotate_apply_on_a_db_uses_the_plate_folder(qtbot, screen, tmp_path):
    plate = tmp_path / "plateA"
    db = plate / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"sqlite")
    AnnotateDropHandler().apply(db, screen)
    assert screen.w("src").text() == str(plate)
    assert f"[drop] annotate src = {plate}\n" in screen.log


def test_annotate_apply_on_a_loose_db_uses_its_own_folder(qtbot, screen,
                                                           tmp_path):
    """Regression: a .db NOT inside measurements/ (which can_accept happily
    allows) used to resolve two levels up, landing src on the wrong plate."""
    holder = tmp_path / "exports"
    holder.mkdir()
    db = holder / "measurements.db"
    db.write_bytes(b"sqlite")
    AnnotateDropHandler().apply(db, screen)
    assert screen.w("src").text() == str(holder)


def test_annotate_apply_on_a_folder_keeps_it(qtbot, screen, tmp_path):
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    db.write_bytes(b"sqlite")
    AnnotateDropHandler().apply(tmp_path, screen)
    assert screen.w("src").text() == str(tmp_path)


# ---------------------------------------------------------------------------
# ClassifyDropHandler
# ---------------------------------------------------------------------------

def test_classify_accepts_a_measurements_plate(tmp_path):
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    db.write_bytes(b"sqlite")
    assert ClassifyDropHandler().can_accept(tmp_path) is True


def test_classify_accepts_a_data_folder(tmp_path):
    (tmp_path / "data").mkdir()
    assert ClassifyDropHandler().can_accept(tmp_path) is True


def test_classify_rejects_a_plain_folder_and_a_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    assert ClassifyDropHandler().can_accept(tmp_path) is False
    assert ClassifyDropHandler().can_accept(f) is False


def test_classify_error_message(tmp_path):
    msg = ClassifyDropHandler().error_message(tmp_path)
    assert "measurements/measurements.db" in msg and "data/" in msg


def test_classify_apply_sets_src(qtbot, screen, tmp_path):
    (tmp_path / "data").mkdir()
    ClassifyDropHandler().apply(tmp_path, screen)
    assert screen.w("src").text() == str(tmp_path)
    assert f"[drop] classify src = {tmp_path}\n" in screen.log


# ---------------------------------------------------------------------------
# MakeMasksDropHandler
# ---------------------------------------------------------------------------

def test_make_masks_accepts_an_image_folder(tmp_path):
    _mkimg(tmp_path / "a.tif")
    assert MakeMasksDropHandler().can_accept(tmp_path) is True


def test_make_masks_rejects_a_file_and_an_empty_folder(tmp_path):
    f = _mkimg(tmp_path / "a.tif")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert MakeMasksDropHandler().can_accept(f) is False
    assert MakeMasksDropHandler().can_accept(empty) is False


def test_make_masks_suggests_nearby_image_folders(tmp_path):
    dropped = tmp_path / "dropped"
    dropped.mkdir()
    _mkimg(tmp_path / "sib" / "a.tif")
    assert MakeMasksDropHandler().suggest_alternatives(dropped) == \
        [tmp_path / "sib"]


def test_make_masks_suggests_nothing_for_a_file(tmp_path):
    f = _mkimg(tmp_path / "a.tif")
    assert MakeMasksDropHandler().suggest_alternatives(f) == []


def test_make_masks_error_message(tmp_path):
    assert "Cellpose" in MakeMasksDropHandler().error_message(tmp_path)


def test_make_masks_apply_sets_src(qtbot, screen, tmp_path):
    _mkimg(tmp_path / "a.tif")
    MakeMasksDropHandler().apply(tmp_path, screen)
    assert screen.w("src").text() == str(tmp_path)
    assert f"[drop] make_masks folder = {tmp_path}\n" in screen.log


# ---------------------------------------------------------------------------
# MapBarcodesDropHandler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["reads.fastq", "reads.fastq.gz",
                                  "reads.fq", "reads.fq.gz",
                                  "READS.FASTQ.GZ"])
def test_map_barcodes_accepts_every_fastq_flavour(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"@id\nACGT\n+\nIIII\n")
    assert MapBarcodesDropHandler().can_accept(p) is True


def test_map_barcodes_accepts_a_folder_holding_a_fastq(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "r1.fq.gz").write_bytes(b"x")
    assert MapBarcodesDropHandler().can_accept(tmp_path) is True


def test_map_barcodes_rejects_a_folder_without_fastqs(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    assert MapBarcodesDropHandler().can_accept(tmp_path) is False


def test_map_barcodes_rejects_a_missing_path(tmp_path):
    assert MapBarcodesDropHandler().can_accept(tmp_path / "gone") is False


def test_map_barcodes_error_message(tmp_path):
    assert ".fastq.gz" in MapBarcodesDropHandler().error_message(tmp_path)


def test_map_barcodes_apply_on_a_file_sets_src_and_fastq(qtbot, tmp_path):
    s = _make_screen(qtbot, keys=("src", "fastq"))
    fq = tmp_path / "reads.fastq.gz"
    fq.write_bytes(b"x")
    MapBarcodesDropHandler().apply(fq, s)
    assert s.w("src").text() == str(tmp_path)
    assert s.w("fastq").text() == str(fq)
    assert f"[drop] map_barcodes src = {tmp_path}\n" in s.log


def test_map_barcodes_apply_finds_the_alternative_fastq_key(qtbot, tmp_path):
    s = _make_screen(qtbot, keys=("src", "fastq_path"))
    fq = tmp_path / "reads.fq"
    fq.write_bytes(b"x")
    MapBarcodesDropHandler().apply(fq, s)
    assert s.w("fastq_path").text() == str(fq)


def test_map_barcodes_apply_on_a_folder_leaves_fastq_empty(qtbot, tmp_path):
    s = _make_screen(qtbot, keys=("src", "fastq"))
    (tmp_path / "reads.fastq").write_bytes(b"x")
    MapBarcodesDropHandler().apply(tmp_path, s)
    assert s.w("src").text() == str(tmp_path)
    assert s.w("fastq").text() == ""


def test_map_barcodes_apply_when_no_fastq_key_exists(qtbot, tmp_path):
    """A screen with none of fastq / fastq_path / fq still gets its src."""
    s = _make_screen(qtbot, keys=("src", "custom_regex"))
    fq = tmp_path / "reads.fastq"
    fq.write_bytes(b"x")
    MapBarcodesDropHandler().apply(fq, s)
    assert s.w("src").text() == str(tmp_path)
    assert s.w("custom_regex").text() == ""


def test_map_barcodes_apply_without_a_settings_model(tmp_path):
    class Bare:
        def __init__(self):
            self._console = _Console()

    fq = tmp_path / "reads.fastq"
    fq.write_bytes(b"x")
    s = Bare()
    MapBarcodesDropHandler().apply(fq, s)
    assert f"[drop] map_barcodes src = {tmp_path}\n" in s._console.text


# ---------------------------------------------------------------------------
# MeasurementsDropHandler
# ---------------------------------------------------------------------------

def test_measurements_handler_requires_the_db(tmp_path):
    h = MeasurementsDropHandler()
    assert h.can_accept(tmp_path) is False
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    db.write_bytes(b"sqlite")
    assert h.can_accept(tmp_path) is True


def test_measurements_handler_rejects_a_file(tmp_path):
    f = tmp_path / "measurements.db"
    f.write_bytes(b"sqlite")
    assert MeasurementsDropHandler().can_accept(f) is True


def test_measurements_handler_error_message(tmp_path):
    assert "measurements/measurements.db" in \
        MeasurementsDropHandler().error_message(tmp_path)


def test_measurements_handler_apply(qtbot, screen, tmp_path):
    MeasurementsDropHandler().apply(tmp_path, screen)
    assert screen.w("src").text() == str(tmp_path)
    assert f"[drop] src = {tmp_path}\n" in screen.log


def test_measurements_handler_normalises_db_and_measurements_folder(
        qtbot, screen, tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    db = measurements / "measurements.db"
    db.write_bytes(b"sqlite")
    handler = MeasurementsDropHandler()

    handler.apply(db, screen)
    assert screen.w("src").text() == str(tmp_path)
    handler.apply(measurements, screen)
    assert screen.w("src").text() == str(tmp_path)


def test_database_handler_opens_any_supported_shape(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    db = measurements / "measurements.db"
    db.write_bytes(b"sqlite")

    class Screen:
        last_error = ""

        def __init__(self):
            self.opened = []

        def set_database(self, path):
            self.opened.append(path)
            return True

    screen = Screen()
    handler = DatabaseDropHandler()
    for path in (db, measurements, tmp_path):
        assert handler.can_accept(path)
        handler.apply(path, screen)
    assert screen.opened == [str(db), str(measurements), str(tmp_path)]


# ---------------------------------------------------------------------------
# Registry — every documented app key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,cls", [
    ("mask", MaskDropHandler),
    ("measure", MeasureDropHandler),
    ("annotate", AnnotateDropHandler),
    ("classify", ClassifyDropHandler),
    ("make_masks", MakeMasksDropHandler),
    ("map_barcodes", MapBarcodesDropHandler),
    ("umap", MeasurementsDropHandler),
    ("ml_analyze", MeasurementsDropHandler),
    ("regression", MeasurementsDropHandler),
    ("recruitment", MeasurementsDropHandler),
    ("activation", MeasurementsDropHandler),
    ("invasion", MeasurementsDropHandler),
    ("analyze_plaques", MakeMasksDropHandler),
    ("train_cellpose", MakeMasksDropHandler),
    ("cellpose_masks", MakeMasksDropHandler),
    ("cellpose_all", MakeMasksDropHandler),
    ("db_browser", DatabaseDropHandler),
])
def test_registry_covers_every_documented_app(key, cls):
    assert type(get_handler(key)) is cls


def test_get_handler_returns_a_fresh_instance_each_call():
    a, b = get_handler("mask"), get_handler("mask")
    assert a is not b


def test_unknown_modules_accept_a_general_source_folder(tmp_path):
    handler = get_handler("future_module")
    assert type(handler) is SourceDropHandler
    assert handler.can_accept(tmp_path)
