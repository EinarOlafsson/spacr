"""The console panel: where a section starts and stops, how a heading copies
it, what clearing keeps, where a write lands, the AI stream, and the bubbles.

Each test walks a case table for one part of
:mod:`spacr.qt.widgets.console_panel` and asserts the exact text, clipboard or
entry count that part produces.  ``as_text`` is the yardstick throughout: it is
what a person selecting the console by hand would get.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPoint, QRect                      # noqa: E402
from PySide6.QtGui import QTextCursor                                  # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel, QSizePolicy,      # noqa: E402
                               QSpacerItem, QWidget)

from spacr.qt.widgets import console_panel as cp                       # noqa: E402
from spacr.qt.widgets.console_panel import ConsolePanel                # noqa: E402

pytestmark = pytest.mark.qt

UNTOUCHED = "what the user was carrying"


@pytest.fixture
def panel(qtbot):
    """A bare console, sized like a docked pane."""
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    return widget


def _bars(panel):
    """Every topic bar in the console's entry list, top to bottom."""
    out = []
    for index in range(panel._entries.count()):
        widget = panel._entries.itemAt(index).widget()
        if isinstance(widget, cp._TopicBar):
            out.append(widget)
    return out


def _clip():
    """The text on the application clipboard."""
    return QApplication.clipboard().text()


@pytest.mark.usefixtures("console_info_enabled")
def test_a_section_spans_to_the_next_heading_with_something_under_it(
        panel, qtbot, subtests):
    """``_section_span``, ``section_text`` and ``section_body`` agree on
    where every section ends, including a banner nested under a banner."""
    panel.begin_topic("Mask")
    panel.append_stdout("segmenting field 1\n")
    panel.append_stdout("segmenting field 2\n")
    panel.begin_topic("Measure")
    panel.append_stdout("measuring\n")
    bars = _bars(panel)
    titles = [bar.text() for bar in bars]

    with subtests.test(msg="headings in order"):
        assert titles == ["Mask", "spaCR output", "Measure", "spaCR output"]

    with subtests.test(msg="module banner spans past the output banner"):
        mask = bars[0]
        assert panel._section_span(mask) == (0, 3)
        assert panel.section_text(mask) == (
            "=== Mask ===\n\n=== spaCR output ===\n"
            "segmenting field 1\nsegmenting field 2\n")
        body = panel.section_body(mask)
        assert [type(w) for w in body][0] is cp._TopicBar
        assert len(body) == 2

    with subtests.test(msg="the last section runs to the end"):
        last = bars[-1]
        start, stop = panel._section_span(last)
        assert (start, stop) == (4, None)
        assert panel.section_text(last) == (
            "=== spaCR output ===\nmeasuring\n")
        assert panel.section_text(bars[2]) == (
            "=== Measure ===\n\n=== spaCR output ===\nmeasuring\n")

    with subtests.test(msg="a bar never inserted spans nothing"):
        stray = cp._TopicBar("not inserted")
        qtbot.addWidget(stray)
        assert panel._section_span(stray) == (None, None)
        assert panel.section_text(stray) == ""
        assert panel.section_body(stray) == []

    with subtests.test(msg="a bar appended past the stretch spans nothing"):
        tail = cp._TopicBar("appended past the end")
        panel._entries.addWidget(tail)
        assert panel.section_text(tail) == ""


@pytest.mark.usefixtures("console_info_enabled")
def test_a_heading_copies_exactly_its_section_or_leaves_the_clipboard(
        panel, qtbot, monkeypatch, subtests):
    """``_TopicBar._copy_section``: a bounded six-parent walk to the panel,
    blank sections and refusing clipboards leave the clipboard and the
    "copied" flash alone."""
    panel.begin_topic("Mask")
    panel.append_stdout("one\ntwo\n")
    panel.begin_topic("Measure")
    panel.append_stdout("three\n")
    first = _bars(panel)[0]

    with subtests.test(msg="a bar on the panel copies its own section"):
        QApplication.clipboard().setText(UNTOUCHED)
        first._copy_section()
        assert _clip() == panel.section_text(first) == (
            "=== Mask ===\n\n=== spaCR output ===\none\ntwo\n")

    with subtests.test(msg="an orphan bar copies nothing"):
        orphan = cp._TopicBar("spaCR output")
        qtbot.addWidget(orphan)
        assert orphan.parent() is None
        QApplication.clipboard().setText(UNTOUCHED)
        orphan._copy_section()
        assert _clip() == UNTOUCHED

    with subtests.test(msg="a bar with no panel within reach copies nothing"):
        deep = QWidget()
        qtbot.addWidget(deep)
        node = deep
        for _ in range(8):
            node = QWidget(node)
        far = cp._TopicBar("spaCR output", node)
        QApplication.clipboard().setText(UNTOUCHED)
        far._copy_section()
        assert _clip() == UNTOUCHED

    with subtests.test(msg="a bar the panel does not lay out copies nothing"):
        tail = cp._TopicBar("appended past the end")
        panel._entries.addWidget(tail)
        QApplication.clipboard().setText(UNTOUCHED)
        tail._copy_section()
        assert _clip() == UNTOUCHED

    for depth, text, expected_requests, expected_clip, expected_flash in [
            (5, "this section", 1, "this section", [True]),
            (6, "this section", 0, UNTOUCHED, []),
            (0, "copied section\n", 1, "copied section\n", [True]),
            (0, " \n\t", 1, UNTOUCHED, []),
    ]:
        with subtests.test(msg=f"{depth} parents between, text {text!r}"):
            host = ConsolePanel()
            qtbot.addWidget(host)
            requested, flashed = [], []
            monkeypatch.setattr(
                host, "section_text",
                lambda bar, _t=text: requested.append(bar) or _t)
            parent = host
            for _ in range(depth):
                parent = QWidget(parent)
            bar = cp._TopicBar("section", parent=parent)
            monkeypatch.setattr(bar._copy_btn, "flash_copied",
                                lambda: flashed.append(True))
            QApplication.clipboard().setText(UNTOUCHED)
            bar._copy_section()
            assert requested == [bar] * expected_requests
            assert _clip() == expected_clip
            assert flashed == expected_flash

    with subtests.test(msg="a clipboard that refuses is not a copy"):
        flashed = []
        monkeypatch.setattr(type(first._copy_btn), "flash_copied",
                            lambda self: flashed.append(True))
        monkeypatch.setattr(QApplication, "clipboard", staticmethod(
            lambda: (_ for _ in ()).throw(RuntimeError("no clipboard"))))
        first._copy_section()
        assert flashed == []


@pytest.mark.usefixtures("console_info_enabled")
def test_clearing_keeps_only_the_trailing_stretch(panel, qapp, subtests):
    """``clear`` and ``as_text`` walk layout items: a non-widget item is
    skipped, the trailing stretch is kept and the conversation goes too."""
    with subtests.test(msg="the only non-widget entry is the stretch"):
        panel.begin_topic("spaCR output — mask")
        panel.append_stdout("first\n")
        panel.begin_topic("spaCR output — measure")
        panel.append_stdout("second\n")
        count = panel._entries.count()
        assert count == 7
        assert all(panel._entries.itemAt(i).widget() is not None
                   for i in range(count - 1))
        last = panel._entries.itemAt(count - 1)
        assert last.widget() is None and last.spacerItem() is not None
        body = panel.section_body(_bars(panel)[0])
        assert [type(w) for w in body][0] is cp._TopicBar
        assert [w.toPlainText() for w in body[1:]] == ["first\n"]

    with subtests.test(msg="labels and stretches in the entry list"):
        panel._entries.addWidget(QLabel("a plain label"))
        panel._entries.addWidget(QLabel("   "))
        panel._entries.addStretch(1)
        assert panel.as_text() == (
            "=== spaCR output — mask ===\n\n=== spaCR output ===\nfirst\n\n"
            "=== spaCR output — measure ===\n\n=== spaCR output ===\n"
            "second\na plain label\n")

    with subtests.test(msg="a spacer among the entries is skipped"):
        panel._entries.insertWidget(0, QLabel("one"))
        panel._entries.insertItem(1, QSpacerItem(
            1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        panel._entries.insertWidget(2, QLabel("two"))
        panel._ai_messages.append({"role": "user", "content": "hello"})
        panel._console_sent_lengths["stdout:0"] = 12
        bar = _bars(panel)[-1]
        panel.clear()
        assert panel._entries.count() == 1
        assert panel._entries.itemAt(0).spacerItem() is not None
        assert panel.as_text() == "\n"
        assert panel._ai_messages == []
        assert panel._console_sent_lengths == {}
        assert panel._current_stdout is None
        assert panel._last_entry_kind == ""

    with subtests.test(msg="a section torn down before its scroll"):
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        with pytest.raises(RuntimeError):
            bar.rect()
        panel._scroll_widget_to_top(bar)
        assert panel._is_raised(bar) is False


def test_a_write_lands_on_the_gui_thread_once(panel, qtbot, monkeypatch,
                                              subtests):
    """Log records by level, the re-entrancy refusal, the worker-thread
    relay and a console built without the shared log handler."""
    from spacr.qt.verbose_logger import console_write

    with subtests.test(msg="a warning record is shown"):
        panel._on_log_record("something went wrong", logging.WARNING)
        assert "something went wrong" in panel.as_text()

    with subtests.test(msg="a write from inside a write is refused"):
        panel.append_stdout("first\n")
        before = panel.as_text()
        with console_write():
            panel.append_stdout("re-entrant\n")
            panel.append_error("re-entrant traceback\n")
        assert panel.as_text() == before

    with subtests.test(msg="a worker's write is relayed, not done here"):
        relayed = {"stdout": [], "error": [], "notice": []}
        panel._relay_stdout.disconnect()
        panel._relay_error.disconnect()
        panel._relay_notice.disconnect()
        panel._relay_stdout.connect(relayed["stdout"].append)
        panel._relay_error.connect(relayed["error"].append)
        panel._relay_notice.connect(
            lambda source, values: relayed["notice"].append((source, values)))
        monkeypatch.setattr(type(panel), "_on_gui_thread", lambda self: False)
        before = panel._entries.count()
        panel.append_stdout("from a worker\n")
        panel.append_error("Traceback…\n")
        panel.append_notice("a notice", a=1)
        assert relayed == {"stdout": ["from a worker\n"],
                           "error": ["Traceback…\n"],
                           "notice": [("a notice", {"a": 1})]}
        assert panel._entries.count() == before
        monkeypatch.undo()

    with subtests.test(msg="no shared log handler in this process"):
        monkeypatch.setattr("spacr.qt.logging_util.get_signal_handler",
                            lambda: (_ for _ in ()).throw(
                                RuntimeError("no handler in this process")))
        from spacr.qt import preferences
        files = preferences.get_log_file_levels()
        console = preferences.get_log_console_levels()
        preferences.set_log_levels(files | {logging.INFO},
                                   console | {logging.INFO})
        try:
            widget = ConsolePanel()
            qtbot.addWidget(widget)
            widget.append_stdout("still works\n")
            assert widget.as_text() == (
                "=== spaCR output ===\nstill works\n")
        finally:
            preferences.set_log_levels(files, console)


@pytest.mark.usefixtures("console_info_enabled")
def test_the_ai_stream_writes_one_block_and_closes_it(panel, qtbot, subtests):
    """Streamed chunks open one reply block; finishing records the reply,
    an empty one says so and a failure reports the provider's error."""
    with subtests.test(msg="two chunks, one block"):
        panel._current_provider_name = "Claude"
        panel._on_stage("thinking")
        panel._on_chunk("the first half ")
        panel._on_chunk("and the second\n")
        assert panel._last_entry_kind == "ai"
        qtbot.waitUntil(
            lambda: "the first half and the second" in panel.as_text())
        assert [bar.text() for bar in _bars(panel)] == ["spaCR AI"]

    with subtests.test(msg="an empty reply says so"):
        panel._ai_buf = []
        panel._on_stream_finished(True, "")
        assert "(empty response — try again or switch provider)" in \
            panel.as_text()
        assert panel._ai_messages[-1] == {"role": "assistant", "content": ""}

    with subtests.test(msg="a failed stream reports the provider error"):
        panel._on_stream_finished(False, "rate limited")
        assert "[AI error] rate limited" in panel.as_text()


@pytest.mark.usefixtures("console_info_enabled")
def test_long_output_scrolls_the_console_to_both_document_ends(qtbot):
    """The read-only output block grows to its text, so the outer console
    scrolls and both ends of the document can be brought into view."""
    panel = ConsolePanel(follow_log=False, chat=False)
    qtbot.addWidget(panel)
    panel.resize(754, 200)
    panel.show()
    qtbot.waitUntil(panel.isVisible)
    for _ in range(5):
        QApplication.processEvents()
    panel.append_stdout("A measured line from spaCR output\n" * 35)
    qtbot.waitUntil(lambda: panel._current_stdout is not None)
    for _ in range(5):
        QApplication.processEvents()

    block = panel._current_stdout
    outer = panel._scroll
    assert block.isReadOnly()
    assert block.sizeHint().height() > outer.viewport().height()
    assert block.height() >= block.sizeHint().height()
    bar = outer.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    assert not panel._jump.isVisible()
    viewport_height = outer.viewport().height()
    bar.setValue(0)
    QApplication.processEvents()
    assert panel._jump.isVisible()
    assert outer.viewport().height() == viewport_height

    for edge in (QTextCursor.Start, QTextCursor.End):
        cursor = block.textCursor()
        cursor.movePosition(edge)
        block.setTextCursor(cursor)
        block.ensureCursorVisible()
        caret = block.cursorRect(cursor)
        point = block.viewport().mapTo(outer.widget(), caret.center())
        outer.ensureVisible(point.x(), point.y(),
                            (caret.width() + 1) // 2 + 2,
                            (caret.height() + 1) // 2 + 2)
        QApplication.processEvents()
        caret = block.cursorRect(cursor)
        caret_in_panel = QRect(
            block.viewport().mapTo(panel, caret.topLeft()), caret.size())
        clip = QRect(block.viewport().mapTo(panel, QPoint()),
                     block.viewport().size()).intersected(QRect(
                         outer.viewport().mapTo(panel, QPoint()),
                         outer.viewport().size()))
        assert outer.viewport().height() == viewport_height
        assert clip.contains(caret_in_panel), (edge, clip, caret_in_panel)


def test_a_bubble_fits_its_label_to_its_width(qtbot, monkeypatch, subtests):
    """``_Bubble._recalc``: no width means no measurement, a guarded call
    does nothing, and a label that cannot measure falls back to its hint."""
    with subtests.test(msg="no width yet, then a 300 px width"):
        bubble = cp._Bubble("user", "a question")
        qtbot.addWidget(bubble)
        widths = []
        monkeypatch.setattr(type(bubble._label), "heightForWidth",
                            lambda self, w: widths.append(w) or 40)
        bubble.resize(0, 0)
        bubble._recalc()
        assert widths == []
        bubble.resize(300, 40)
        widths.clear()
        bubble._recalc()
        assert widths == [276]
        assert bubble._label.height() == 40
        assert bubble.height() == 40 + bubble._V_PAD
        monkeypatch.undo()

    with subtests.test(msg="a measurement in progress is not restarted"):
        bubble = cp._Bubble("user", "a question")
        qtbot.addWidget(bubble)
        bubble.resize(300, 40)
        bubble._recalc_guard = True
        bubble._label.setFixedHeight(11)
        bubble._recalc()
        assert bubble._label.height() == 11
        bubble._recalc_guard = False

    with subtests.test(msg="a narrow bubble still wraps at 120 px"):
        bubble = cp._Bubble("ai", "an answer")
        qtbot.addWidget(bubble)
        widths = []
        monkeypatch.setattr(type(bubble._label), "heightForWidth",
                            lambda self, w: widths.append(w) or 30)
        bubble.resize(50, 40)
        widths.clear()
        bubble._recalc()
        assert widths == [120]
        monkeypatch.undo()

    with subtests.test(msg="a label that cannot measure uses its hint"):
        bubble = cp._Bubble("ai", "an answer")
        qtbot.addWidget(bubble)
        bubble.resize(300, 40)
        monkeypatch.setattr(type(bubble._label), "heightForWidth",
                            lambda self, width: 0)
        bubble._recalc()
        assert bubble._label.height() == bubble._label.sizeHint().height()
        assert bubble.height() == bubble._label.height() + bubble._V_PAD
