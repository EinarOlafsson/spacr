"""N709: every ``setStyleSheet`` reaches Qt only at a moment Qt can take it.

GitHub issues #161, #163 and #164 are native segmentation faults with no
Python cause; the one native backtrace (a hosted Qt run on 2026-10-24) faults
in ``libQt6Widgets`` right after ``QWidget::style()``, inside a
``setStyleSheet`` called from Python, reading a pointer of value 1. These
tests pin the defensive rules of the stylesheet guard in
:mod:`spacr.qt.gui_scale`, one invariant each:

* no sheet is applied from inside a polish, a style change, a paint, a
  reparent or a deletion -- it waits for the event loop;
* a sheet only reaches a live widget (``shiboken6.isValid``);
* waiting sheets are applied one top-level window at a time, newest only;
* a widget being destroyed, or inside a main window that has accepted its
  last close, is never restyled.
"""
from __future__ import annotations

import inspect
import threading

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QWidget

from spacr.qt import gui_scale, theme

pytestmark = pytest.mark.qt

_DELICATE = None


def _delicate_kinds():
    """The event types Qt must not be restyled inside, read independently."""
    global _DELICATE
    if _DELICATE is None:
        _DELICATE = {QEvent.Type.Polish,
                     QEvent.Type.StyleChange, QEvent.Type.Paint,
                     QEvent.Type.DeferredDelete,
                     QEvent.Type.ParentAboutToChange,
                     QEvent.Type.ParentChange}
    return _DELICATE


def _before_its_own_polish(target, kind, values) -> bool:
    """A window's own Polish, heard by a filter before Qt has polished it."""
    if kind != QEvent.Type.Polish:
        return False
    if not any(value is target for value in values):
        return False
    return target.isWindow() and not target.testAttribute(Qt.WA_WState_Polished)


def _a_delicate_handler_is_running(target) -> bool:
    """Is a Python handler for a delicate event anywhere on the stack?"""
    for info in inspect.stack(0):
        if info.function == "paintEvent":
            return True
        values = list(info.frame.f_locals.values())
        for value in values:
            if isinstance(value, QEvent):
                try:
                    kind = value.type()
                except RuntimeError:
                    return True
                if kind not in _delicate_kinds():
                    continue
                if _before_its_own_polish(target, kind, values):
                    continue
                return True
    return False


@pytest.fixture
def applied(qapp, monkeypatch):
    """Record every sheet Qt is actually handed, and how it was reached."""
    gui_scale._install_sheet_guard()
    gui_scale._flush_pending_sheets()
    records = []
    original = gui_scale._apply_now

    def record(obj, text, raw_set):
        records.append({
            "target": obj,
            "text": text,
            "nested": gui_scale._SHEET_GUARD["depth"] > 0,
            "in_handler": _a_delicate_handler_is_running(obj),
            "main_thread": threading.current_thread() is threading.main_thread(),
        })
        original(obj, text, raw_set)

    monkeypatch.setattr(gui_scale, "_apply_now", record)
    yield records
    gui_scale._flush_pending_sheets()
    with gui_scale._PENDING_LOCK:
        gui_scale._PENDING_SHEETS.clear()
    gui_scale._SHEET_GUARD["posted"] = False


@pytest.fixture
def per_window_sheet(qapp):
    """A per-window sheet in force for one test, then the session's back."""
    attribute = theme._WINDOW_SHEET_ATTRIBUTE
    had = hasattr(qapp, attribute)
    previous = getattr(qapp, attribute, None)
    app_sheet = qapp.styleSheet()
    sheet = "QLabel { color: rgb(11, 22, 33); }"
    theme.apply_stylesheet_per_window(qapp, sheet)
    yield sheet
    QCoreApplication.processEvents()
    if had:
        setattr(qapp, attribute, previous)
    elif hasattr(qapp, attribute):
        delattr(qapp, attribute)
    qapp.setStyleSheet(app_sheet)
    QCoreApplication.processEvents()


def _texts_for(records, target):
    """The sheets ``target`` was handed, in order."""
    return [r["text"] for r in records if r["target"] is target]


def _raw_sheet(widget) -> str:
    """The sheet Qt holds now, not a waiting one."""
    raw = gui_scale._ORIGINAL.get((QWidget, "raw_setStyleSheet"))
    assert raw is not None
    assert gui_scale._pending_sheet_for(widget) is None
    return QWidget.styleSheet(widget)


class _RestylesOnStyleChange(QLabel):
    """Restyles itself from the StyleChange a sheet sends, as several do."""

    def __init__(self, sheets):
        super().__init__("n709")
        self._sheets = list(sheets)
        self.seen_inside = []

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.StyleChange and self._sheets:
            sheets, self._sheets = self._sheets, []
            for sheet in sheets:
                self.setStyleSheet(sheet)
            self.seen_inside.append(self.styleSheet())


def test_a_restyle_from_inside_a_style_change_waits_for_the_event_loop(applied):
    label = _RestylesOnStyleChange(["QLabel { color: #aa0000; }"])
    label.setStyleSheet("QLabel { color: #0000aa; }")

    assert label.seen_inside == ["QLabel { color: #aa0000; }"]
    assert _texts_for(applied, label) == ["QLabel { color: #0000aa; }"]
    assert label.styleSheet() == "QLabel { color: #aa0000; }"

    QCoreApplication.processEvents()

    assert _texts_for(applied, label) == ["QLabel { color: #0000aa; }",
                                          "QLabel { color: #aa0000; }"]
    assert not any(r["nested"] or r["in_handler"] for r in applied)
    assert _raw_sheet(label) == "QLabel { color: #aa0000; }"
    label.deleteLater()


def test_nested_restyle_requests_collapse_to_the_newest(applied):
    label = _RestylesOnStyleChange(["QLabel { color: #110000; }",
                                    "QLabel { color: #220000; }",
                                    "QLabel { color: #330000; }"])
    label.setStyleSheet("QLabel { color: #000011; }")
    QCoreApplication.processEvents()

    assert _texts_for(applied, label) == ["QLabel { color: #000011; }",
                                          "QLabel { color: #330000; }"]
    assert not any(r["nested"] for r in applied)
    label.deleteLater()


class _RestylesWhilePainting(QWidget):
    """Puts a sheet on from inside its own paint."""

    def __init__(self):
        super().__init__()
        self.painted = 0

    def paintEvent(self, event):
        self.painted += 1
        self.setStyleSheet("QWidget { background: #102030; }")
        super().paintEvent(event)


def test_a_restyle_from_inside_a_paint_waits_for_the_event_loop(applied):
    widget = _RestylesWhilePainting()
    widget.resize(40, 30)
    widget.grab()

    assert widget.painted >= 1
    assert _texts_for(applied, widget) == []

    QCoreApplication.processEvents()

    assert _texts_for(applied, widget) == ["QWidget { background: #102030; }"]
    assert not any(r["in_handler"] for r in applied)
    widget.deleteLater()


def test_a_waiting_sheet_is_dropped_when_its_widget_dies(applied):
    label = _RestylesOnStyleChange(["QLabel { color: #00aa00; }"])
    label.setStyleSheet("QLabel { color: #0000aa; }")
    assert gui_scale._pending_sheet_for(label) == "QLabel { color: #00aa00; }"

    shiboken6.delete(label)
    QCoreApplication.processEvents()

    assert _texts_for(applied, label) == ["QLabel { color: #0000aa; }"]
    assert not gui_scale._PENDING_SHEETS


def test_a_deleted_widget_is_never_handed_a_sheet(applied):
    widget = QWidget()
    shiboken6.delete(widget)

    widget.setStyleSheet("QWidget { color: red; }")
    QCoreApplication.processEvents()

    assert _texts_for(applied, widget) == []
    assert theme._sheet_one_window(widget) is False


class _RestylesWhileBeingDeleted(QDialog):
    """A dialog that restyles itself and a child as its deletion arrives."""

    def __init__(self, sheet_on_close):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.child = QLabel("child", self)
        QVBoxLayout(self).addWidget(self.child)
        self._sheet_on_close = sheet_on_close
        self.deletion_arrived = False

    def closeEvent(self, event):
        app = QApplication.instance()
        theme.apply_stylesheet_per_window(app, self._sheet_on_close)
        super().closeEvent(event)

    def event(self, event):
        if event.type() == QEvent.DeferredDelete:
            self.deletion_arrived = True
            self.setStyleSheet("QDialog { background: #300000; }")
            self.child.setStyleSheet("QLabel { color: #300000; }")
        return super().event(event)


def test_a_theme_change_during_a_dialog_close_never_restyles_the_dying_dialog(
        applied, per_window_sheet):
    app = QApplication.instance()
    dialog = _RestylesWhileBeingDeleted(
        per_window_sheet + "\nQDialog { margin: 0px; }")
    child = dialog.child
    dialog.show()
    QCoreApplication.processEvents()
    try:
        applied.clear()
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()

        assert not shiboken6.isValid(dialog)
        late = [r for r in applied
                if r["target"] is dialog or r["target"] is child]
        assert all("#300000" not in r["text"] for r in late)
        assert not any(r["nested"] or r["in_handler"] for r in applied)
        assert gui_scale._prune_torn_down() == 0
    finally:
        theme.apply_stylesheet_per_window(app, per_window_sheet)
        QCoreApplication.processEvents()


class _RestylesWhileReparented(QLabel):
    """Restyles itself as Qt, still inside ``setParent``, says it moved."""

    def event(self, event):
        if event.type() == QEvent.ParentChange:
            self.setStyleSheet("QLabel { color: #004400; }")
        return super().event(event)


def test_a_restyle_during_a_reparent_lands_after_it_on_the_live_widget(applied):
    first, second = QWidget(), QWidget()
    label = _RestylesWhileReparented("moving", first)
    applied.clear()

    label.setParent(second)

    assert _texts_for(applied, label) == []
    QCoreApplication.processEvents()
    assert _texts_for(applied, label) == ["QLabel { color: #004400; }"]
    assert label.parentWidget() is second
    assert not any(r["in_handler"] for r in applied)
    first.deleteLater()
    second.deleteLater()


def test_a_restyle_asked_for_on_another_thread_is_applied_on_the_gui_thread(applied):
    widget = QWidget()
    worker = threading.Thread(
        target=lambda: widget.setStyleSheet("QWidget { color: #000044; }"))
    worker.start()
    worker.join(5)

    assert _texts_for(applied, widget) == []
    for _ in range(20):
        QCoreApplication.processEvents()
        if _texts_for(applied, widget):
            break
    assert _texts_for(applied, widget) == ["QWidget { color: #000044; }"]
    assert all(r["main_thread"] for r in applied)
    widget.deleteLater()


def test_waiting_sheets_are_applied_one_top_level_window_at_a_time(applied):
    windows = [QWidget(), QWidget()]
    children = [QLabel("a", windows[0]), QLabel("b", windows[1]),
                QLabel("c", windows[0])]
    for index, target in enumerate(children + windows):
        gui_scale._defer_sheet(target, f"QWidget {{ margin: {index}px; }}",
                               gui_scale._ORIGINAL[(QWidget, "raw_setStyleSheet")])

    QCoreApplication.processEvents()

    order = [r["target"] for r in applied]
    tops = [target.window() for target in order]
    assert len(order) == 5
    runs = [top for index, top in enumerate(tops)
            if index == 0 or tops[index - 1] is not top]
    assert len(runs) == 2
    assert runs[0] is windows[0] and runs[1] is windows[1]
    for window in windows:
        window.deleteLater()


def test_a_window_is_sheeted_by_its_show_never_inside_a_polish(
        applied, per_window_sheet):
    window = QWidget()
    QLabel("inside", window)
    window.ensurePolished()
    window.show()

    sheeted = _texts_for(applied, window)
    assert sheeted, "the window was shown without the theme"
    assert per_window_sheet in _raw_sheet(window)
    assert not any(r["in_handler"] for r in applied)
    QCoreApplication.processEvents()
    window.close()
    window.deleteLater()


def test_a_main_window_that_accepted_its_close_is_never_restyled(applied):
    window = QWidget()
    child = QLabel("panel", window)
    gui_scale._close_without_restyling(window)
    try:
        window.setStyleSheet("QWidget { color: #440000; }")
        child.setStyleSheet("QLabel { color: #440000; }")
        QCoreApplication.processEvents()
        assert _texts_for(applied, window) == []
        assert _texts_for(applied, child) == []
    finally:
        gui_scale._close_without_restyling(window, closing=False)

    child.setStyleSheet("QLabel { color: #004400; }")
    assert _texts_for(applied, child) == ["QLabel { color: #004400; }"]
    window.deleteLater()


class _RestylesWhilePolished(QLabel):
    """Restyles itself from inside its own Polish, after Qt has begun it."""

    def event(self, event):
        if event.type() == QEvent.Polish:
            self.setStyleSheet("QLabel { color: #440044; }")
        return super().event(event)


def test_a_restyle_from_inside_a_polish_waits_for_the_event_loop(applied):
    label = _RestylesWhilePolished("polished")
    label.ensurePolished()

    assert _texts_for(applied, label) == []
    assert label.styleSheet() == "QLabel { color: #440044; }"
    QCoreApplication.processEvents()
    assert _texts_for(applied, label) == ["QLabel { color: #440044; }"]
    assert not any(r["in_handler"] or r["nested"] for r in applied)
    label.deleteLater()


class _DropsAWidgetOnRestyle(QLabel):
    """Lets go of a parentless widget from inside an application restyle.

    The pattern behind the hosted Qt shard 0 core of 2026-10-24
    (``test_ai_error_storm_prompt`` crashed in ``set_gui_scale_live``):
    ``QApplication::setStyle`` walks a copy of ``allWidgets()`` and sends
    each one a ``StyleChange``, and a Python handler that
    frees a widget the copy still lists makes Qt read freed memory.
    """

    def __init__(self, victim):
        super().__init__("dropper")
        self.victim = victim
        self.victim_alive_after_drop = []

    def _drop(self):
        if self.victim is None:
            return
        import weakref

        ref = weakref.ref(self.victim)
        self.victim = None
        self.victim_alive_after_drop.append(ref() is not None)

    def changeEvent(self, event):
        if event.type() == QEvent.StyleChange:
            self._drop()
        super().changeEvent(event)


@pytest.mark.parametrize("restyle", ["sheet", "style"])
def test_no_widget_an_application_restyle_walks_can_be_freed_during_it(
        applied, restyle):
    app = QApplication.instance()
    victim = QWidget()
    victim.ensurePolished()
    dropper = _DropsAWidgetOnRestyle(victim)
    dropper.ensurePolished()
    del victim
    saved_sheet, saved_style = app.styleSheet(), app.style().name()
    try:
        if restyle == "sheet":
            app.setStyleSheet(saved_sheet + "\nQLabel#n709 { margin: 0px; }")
        else:
            QApplication.setStyle("Fusion")
        assert dropper.victim_alive_after_drop == [True]
    finally:
        app.setStyleSheet(saved_sheet)
        QApplication.setStyle(saved_style)
        dropper.deleteLater()
        QCoreApplication.processEvents()


_WALK_STRESS = r'''
import gc, sys
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget

app = QApplication(sys.argv[:1])
if sys.argv[1] == "guarded":
    from spacr.qt import gui_scale
    gui_scale._install_sheet_guard()


class Collector(QLabel):
    def changeEvent(self, event):
        if event.type() == QEvent.StyleChange:
            gc.collect()
        super().changeEvent(event)


keep = [Collector("c") for _ in range(20)]
for widget in keep:
    widget.ensurePolished()
for round_ in range(int(sys.argv[2])):
    for _ in range(40):
        widget = QWidget()
        widget.ensurePolished()
        widget.cycle = widget
        del widget
    app.setStyleSheet("QWidget { margin: %dpx; }" % (round_ % 3))
    gc.collect()
print("survived")
'''


def test_an_application_sheet_survives_widgets_freed_inside_its_walk(tmp_path):
    """The crash itself, in a child process: unguarded it dies, guarded not.

    Every ``StyleChange`` the walk sends runs a collection that frees
    parentless widgets the walk's copy of ``allWidgets()`` still lists.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "walk_stress.py"
    script.write_text(_WALK_STRESS)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    guarded = subprocess.run([sys.executable, str(script), "guarded", "25"],
                             env=env, capture_output=True, text=True,
                             timeout=240)
    assert guarded.returncode == 0, guarded.stderr[-2000:]
    assert "survived" in guarded.stdout
    raw = subprocess.run([sys.executable, str(script), "raw", "25"],
                         env=env, capture_output=True, text=True, timeout=240)
    assert raw.returncode != 0, "the unguarded walk no longer crashes"


class _SizesFromAChildSheetOnFontChange(QWidget):
    """Puts a font-size sheet on a child in its FontChange, then measures."""

    def __init__(self):
        super().__init__()
        self.child = QLabel("Xg", self)
        self.measured = []

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            pixels = max(6, self.fontInfo().pixelSize() * 2)
            self.child.setStyleSheet(f"QLabel {{ font-size: {pixels}px; }}")
            self.measured.append((pixels, self.child.fontInfo().pixelSize()))


def test_a_font_change_handler_sees_the_child_sheet_it_just_set(applied):
    widget = _SizesFromAChildSheetOnFontChange()
    widget.ensurePolished()
    widget.child.ensurePolished()
    larger = widget.font()
    larger.setPointSize(larger.pointSize() + 3)
    widget.setFont(larger)

    assert widget.measured, "the FontChange never arrived"
    wanted, painted = widget.measured[-1]
    assert painted == wanted
    assert _texts_for(applied, widget.child)
    assert not any(r["nested"] for r in applied)
    widget.deleteLater()


def test_hidden_combo_popups_are_never_restyled_until_they_show(
        applied, per_window_sheet, qtbot):
    """Windows nightly 4f723c7d8: restyling hidden combo popup frames crashed.

    A combo box's list lives in a parentless popup frame from the first
    ``view()`` on; every theme pass and change of scale restyled each hidden
    one. They are now left alone until they show.
    """
    from PySide6.QtWidgets import QComboBox

    app = QApplication.instance()
    host = QWidget()
    qtbot.addWidget(host)
    combos = []
    for index in range(6):
        combo = QComboBox(host)
        combo.addItems([f"item {index}-{n}" for n in range(9)])
        combo.view()
        combos.append(combo)
    popups = [combo.view().window() for combo in combos]
    assert all(popup is not host and popup.isWindow() for popup in popups)
    assert {popup.metaObject().className() for popup in popups} == {
        "QComboBoxPrivateContainer"}
    host.show()
    QCoreApplication.processEvents()
    original = gui_scale.current_scale()
    try:
        for round_ in range(4):
            theme.apply_stylesheet_per_window(
                app, per_window_sheet + f"\nQComboBox {{ margin: {round_}px; }}")
            gui_scale.set_gui_scale_live(0.9 if round_ % 2 else 1.1)
            for popup in popups:
                popup.setStyleSheet(f"QFrame {{ margin: {round_}px; }}")
            QCoreApplication.processEvents()
        inside = set()
        for popup in popups:
            inside.add(id(popup))
            inside.update(id(child) for child in popup.findChildren(QWidget))
        assert not [r for r in applied if id(r["target"]) in inside]

        combos[0].showPopup()
        qtbot.waitUntil(popups[0].isVisible, timeout=2000)
        QCoreApplication.processEvents()
        assert [r for r in applied if r["target"] is popups[0]]
        shown = _raw_sheet(popups[0])
        assert shown.startswith(per_window_sheet)
        assert shown.endswith("QFrame { margin: 3px; }")
        combos[0].hidePopup()
    finally:
        gui_scale.set_gui_scale_live(original)


class _CountsStyleChanges(QLabel):
    """Counts the StyleChange events Qt sends it."""

    def __init__(self):
        super().__init__("count")
        self.style_changes = 0

    def changeEvent(self, event):
        if event.type() == QEvent.StyleChange:
            self.style_changes += 1
        super().changeEvent(event)


def test_an_unchanged_sheet_is_not_handed_to_qt_again(applied):
    """Lessons 70 and 85: restyles to an unchanged sheet stalled module opens."""
    label = _CountsStyleChanges()
    label.ensurePolished()
    label.setStyleSheet("QLabel { color: #123456; }")
    first = label.style_changes
    assert first >= 1
    for _ in range(5):
        label.setStyleSheet("QLabel { color: #123456; }")
    assert label.style_changes == first

    label.setStyleSheet("QLabel { color: #654321; }")
    assert label.style_changes > first
    assert _raw_sheet(label) == "QLabel { color: #654321; }"
    label.deleteLater()
