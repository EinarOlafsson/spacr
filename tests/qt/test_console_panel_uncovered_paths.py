"""The console's caps, its clock, and what it hands the AI.

Three things here are load-bearing and none of them is visible from the
outside of a short run. The scrollback has a character cap, and reaching it
must cost what is dropped rather than what is kept -- a run that logs a
million lines has to stay responsive. The working indicator may be started by
a worker thread, and a ``QTimer`` started off its owner's thread never fires.
And the console context attached to an AI question is a budget: a complete
traceback is never cut, ordinary output yields first, and the reader is told
in the same sentence how much went.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QThread

from spacr.qt.widgets import console_panel as cp
from spacr.qt.widgets.console_panel import ConsolePanel

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    return widget


# -- the scrollback cap -------------------------------------------------------

def test_a_block_past_its_cap_drops_whole_paragraphs_off_the_head(
        qtbot, monkeypatch):
    """The newest output survives; the oldest is what goes.

    Trimming from the front costs what is removed. Re-setting the document
    to its own tail -- the approach this replaced -- costs what is kept, on
    every line once the cap is reached, which is how a long run turned the
    console quadratic in its own length.
    """
    monkeypatch.setattr(cp._StdoutBlock, "MAX_CHARS", 200)
    block = cp._StdoutBlock()
    qtbot.addWidget(block)

    for index in range(60):
        block.append(f"line {index:02d}\n")

    text = block.toPlainText()

    assert "line 59" in text
    assert "line 00" not in text
    assert len(text) <= 2 * cp._StdoutBlock.MAX_CHARS


def test_a_block_under_its_cap_keeps_everything(qtbot):
    block = cp._StdoutBlock()
    qtbot.addWidget(block)

    block.append("first\n")
    block.append("second\n")

    assert block.text() == "first\nsecond\n"


def test_a_pinned_block_reports_the_height_it_was_pinned_to(qtbot):
    """A user-dragged section keeps its height until it is freed."""
    block = cp._StdoutBlock("some output\n")
    qtbot.addWidget(block)

    block.set_user_height(300)
    assert block.sizeHint().height() == 300

    block.reset_user_height()
    assert block.sizeHint().height() != 300
    assert block.maximumHeight() == 16_777_215


def test_the_height_answer_is_reused_until_something_changes(qtbot):
    """Qt asks several times per layout pass; walking the document each time
    is what the cache exists to avoid."""
    block = cp._StdoutBlock("one\ntwo\nthree\n")
    qtbot.addWidget(block)

    first = block.sizeHint().height()
    again = block.sizeHint().height()
    assert again == first

    block.append("four\nfive\nsix\nseven\n")

    assert block.sizeHint().height() > first


# -- the working indicator, started from a worker -----------------------------

def test_the_working_dots_cycle_at_a_fixed_width(qtbot):
    """The row must not jitter as the count changes, so the glyphs are padded."""
    dots = cp._WorkingDots()
    qtbot.addWidget(dots)

    seen = [dots.text()]
    for _ in range(3):
        dots._tick()
        seen.append(dots.text())

    assert [text.count("●") for text in seen] == [1, 2, 3, 1]
    assert len({len(text) for text in seen}) == 1


def test_a_worker_asking_for_the_dots_starts_them_on_the_gui_thread(qtbot):
    """A ``QTimer`` started off its owner's thread never fires.

    The stream worker runs on its own thread and asks the indicator to
    start from there, so the call has to be queued onto the thread the
    widget lives on before the timer is touched.
    """
    dots = cp._WorkingDots()
    qtbot.addWidget(dots)

    class _Asker(QThread):
        def run(self):
            dots.start()

    asker = _Asker()
    asker.start()
    assert asker.wait(5000)

    qtbot.waitUntil(lambda: dots._timer.isActive(), timeout=3000)
    assert dots.isVisible() or dots.isVisibleTo(dots.parentWidget() or dots)

    class _Stopper(QThread):
        def run(self):
            dots.stop()

    stopper = _Stopper()
    stopper.start()
    assert stopper.wait(5000)

    qtbot.waitUntil(lambda: not dots._timer.isActive(), timeout=3000)


# -- what goes to the AI with a question --------------------------------------

def test_no_context_is_attached_when_the_reader_turned_it_off(panel,
                                                              monkeypatch):
    """The preference is the reader saying "do not send my console"."""
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ai_settings, "get_console_aware", lambda: False)
    panel.append_stdout("something that must not be sent\n")

    context, status = panel._console_context_for_question("why did it fail?")

    assert context == ""
    assert "must not be sent" not in status


def test_a_console_with_nothing_new_attaches_nothing(panel):
    """A second question about the same output does not re-send it.

    Nothing new is its own answer, said in those words -- not an empty
    payload described as output that was sent.
    """
    from spacr.qt.i18n import tr

    context, status = panel._console_context_for_question("what happened?")

    assert context == ""
    assert status == tr("Console context: no new output")


def test_a_traceback_is_kept_whole_while_old_output_is_dropped(panel,
                                                               monkeypatch):
    """The error is the thing being asked about, so it never gets cut.

    Ordinary stdout yields to the budget first, and the reader is told how
    much of it did not go.
    """
    monkeypatch.setattr(cp, "AI_CONSOLE_CONTEXT_CHARS", 500)
    panel.append_stdout("x" * 4000 + "\n")
    panel.append_error(
        "Traceback (most recent call last):\n  ValueError: boom\n")

    context, status = panel._console_context_for_question("why did it fail?")

    assert "complete traceback" in context
    assert "ValueError: boom" in context
    assert "dropped" in context
    assert "dropped" in status

    # The traceback is charged to the budget first and the surviving stdout
    # tail is only what the traceback left over -- not the whole budget again
    # on top of it.
    budget = cp.AI_CONSOLE_CONTEXT_CHARS
    traceback_at = context.index("--- complete traceback ---")
    kept_stdout = context[context.index("\n") + 1:traceback_at].strip("\n")
    traceback_section = context[traceback_at:]
    spent = len(kept_stdout) + len(traceback_section)
    assert set(kept_stdout) == {"x"}
    assert spent <= budget
    assert spent >= budget - 2       # and the tail fills what is left of it


def test_output_that_fits_the_budget_is_sent_whole(panel):
    panel.append_stdout("a short run said this\n")

    context, status = panel._console_context_for_question("what happened?")

    assert "a short run said this" in context
    assert "dropped" not in context
    assert status


def test_a_repeat_question_only_carries_what_arrived_since(panel):
    """Marking text sent is what stops a long run being re-uploaded per ask."""
    panel.append_stdout("the first thing\n")
    first, _ = panel._console_context_for_question("what happened?")
    panel.append_stdout("the second thing\n")

    second, _ = panel._console_context_for_question("and then?")

    assert "the first thing" in first
    assert "the second thing" in second
    assert second.count("the second thing") == 1


# -- retired stream threads ---------------------------------------------------

def test_a_retired_thread_that_has_already_stopped_is_forgotten(panel):
    """The list is held only until the OS thread has wound down."""
    thread = QThread()
    assert thread.isRunning() is False
    panel._retired.append((thread, None))

    panel._prune_retired()

    assert panel._retired == []


def test_a_retired_thread_whose_object_qt_deleted_is_forgotten(panel):
    """Qt's deferred-delete queue can get there first; that is not an error."""
    thread = QThread()
    panel._retired.append((thread, None))
    shiboken6.delete(thread)

    panel._prune_retired()

    assert panel._retired == []


# -- the split the user drags -------------------------------------------------

def test_moving_the_handle_programmatically_lands_where_a_drag_would(qtbot):
    """A caller restoring a layout and a user dragging leave the same state."""
    widget = ConsolePanel(persist_key="")
    qtbot.addWidget(widget)
    widget.resize(600, 600)

    widget.set_split_sizes(400, 160)

    console_px, chat_px = widget.split_sizes()
    assert chat_px == 160
    total = console_px + chat_px

    widget.set_split_sizes(400, 260)

    moved_console, moved_chat = widget.split_sizes()
    assert moved_chat == 260
    assert moved_console == console_px - 100
    assert moved_console + moved_chat == total


def test_a_zoom_change_is_applied_to_every_entry(panel, monkeypatch):
    """The console follows the global Zoom, so a save restyles what is there."""
    panel.append_stdout("a line of output\n")
    blocks = panel._holder.findChildren(cp._StdoutBlock)
    assert blocks

    monkeypatch.setattr(ConsolePanel, "_zoomed_font_pt", staticmethod(
        lambda: 21))
    panel.apply_zoom()

    assert panel._font_pt == 21
    assert all(block._font_pt == 21 for block in blocks)


# -- headings that belong to no console ---------------------------------------

def test_a_heading_with_no_console_above_it_activates_nothing(qtbot):
    """A bar built outside a panel is inert rather than fatal.

    Topic bars are constructed before they are inserted, and a key press
    that arrived in that window would otherwise walk off the top of the
    widget tree.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    bar = cp._TopicBar("spaCR output")
    qtbot.addWidget(bar)

    bar.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                                Qt.NoModifier))

    assert bar.is_expanded() is True


def test_a_heading_that_is_not_in_this_console_has_no_body(panel, qtbot):
    """Asking about a stranger's heading answers empty, not by guessing."""
    stranger = cp._TopicBar("somebody else's section")
    qtbot.addWidget(stranger)

    assert panel.section_body(stranger) == []
    assert panel.section_text(stranger) == ""


def test_the_chevron_turns_when_a_section_folds(qtbot):
    """A toggle with no indicator is a control found by accident."""
    bar = cp._TopicBar("spaCR output")
    qtbot.addWidget(bar)

    assert bar._chevron.text() == "\u25be"

    bar.set_expanded(False)

    assert bar.is_expanded() is False
    assert bar._chevron.text() == "\u25b8"


# -- the plain-text export ----------------------------------------------------

def test_an_empty_entry_contributes_no_line_to_the_export(panel, qtbot):
    """A blank block, a blank label and a bare widget are not blank lines.

    The export is what a person would have selected by hand, and a console
    that ends in three empty lines is one nobody can paste into a report.
    """
    from PySide6.QtWidgets import QLabel, QWidget

    panel.append_stdout("the only real line\n")
    panel._insert_entry(cp._StdoutBlock())
    panel._insert_entry(QLabel(""))
    panel._insert_entry(QWidget())
    # Real output after the empty entries: a blank line they contributed
    # would sit between the two real lines rather than being stripped off
    # the end.
    panel._insert_entry(cp._StdoutBlock("a later real line\n"))

    text = panel.as_text()

    assert "the only real line" in text
    assert "a later real line" in text
    lines = text.strip().splitlines()
    assert [line for line in lines if not line.strip()] == []
    assert lines[-2:] == ["the only real line", "a later real line"]


# -- the jump control ---------------------------------------------------------

def test_a_scroll_before_the_jump_button_exists_is_harmless(panel):
    """The scrollbar is connected while the panel is still being built.

    ``_refresh_jump_button`` is wired to ``valueChanged`` several statements
    before the button it shows is constructed, so a scroll delivered in that
    window has to find nothing and say nothing.
    """
    del panel._jump

    panel._scroll.verticalScrollBar().valueChanged.emit(0)

    assert not hasattr(panel, "_jump")


# -- a bubble with nothing in it ----------------------------------------------

def test_a_bubble_built_empty_holds_no_text(qtbot):
    """The prefix is written when there is something to prefix, not before."""
    bubble = cp._Bubble("ai")
    qtbot.addWidget(bubble)

    assert bubble._raw_text == ""
    assert bubble._label.text() == ""

    bubble.set_text("here is the answer")

    assert "here is the answer" in bubble._label.text()


# -- submitting a question ----------------------------------------------------

def _fake_provider(monkeypatch, panel):
    """Point the panel at a provider without touching a real service."""
    provider = object()
    monkeypatch.setattr(cp.ai_module, "get_provider", lambda name: provider)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)
    return provider


def test_a_question_carries_the_console_context_into_the_prompt(panel,
                                                                monkeypatch):
    """The console the user is looking at is what the AI is asked about."""
    monkeypatch.setattr(cp.ConsolePanel, "_start_stream",
                        lambda self, system: None)
    _fake_provider(monkeypatch, panel)
    panel.append_stdout("ValueError raised in the counting step\n")
    panel._input.setPlainText("why did that happen?")

    panel._on_submit()

    sent = panel._ai_messages[-1]["content"]
    assert "<spacr_console_context>" in sent
    assert "ValueError raised in the counting step" in sent
    assert sent.startswith("why did that happen?")


def test_a_second_question_while_one_is_streaming_is_ignored(panel,
                                                             monkeypatch):
    """Cancel lives in the actions row; the console does not queue asks."""
    _fake_provider(monkeypatch, panel)
    panel._ai_thread = QThread()
    before = panel._entries.count()
    panel._input.setPlainText("and another thing")

    panel._on_submit()

    assert panel._entries.count() == before
    assert panel._ai_messages == []
    panel._ai_thread = None


def test_a_stream_is_reported_while_it_is_running(panel):
    assert panel.is_ai_streaming() is False

    panel._ai_thread = QThread()
    assert panel.is_ai_streaming() is True

    panel._ai_thread = None
    assert panel.is_ai_streaming() is False


# -- cancelling ---------------------------------------------------------------

class _RecordingWorker:
    """The worker surface the panel touches when a user cancels."""

    def __init__(self, explode: bool = False):
        self.cancelled = 0
        self._explode = explode

    def cancel(self):
        self.cancelled += 1
        if self._explode:
            raise RuntimeError("the subprocess is already gone")


def test_cancelling_reaches_the_worker_that_is_streaming(panel):
    worker = _RecordingWorker()
    panel._ai_worker = worker

    panel.cancel_ai()

    assert worker.cancelled == 1
    panel._ai_worker = None


def test_cancelling_with_nothing_running_does_nothing(panel):
    panel._ai_worker = None

    panel.cancel_ai()

    assert panel.is_ai_streaming() is False


def test_a_worker_that_raises_on_cancel_does_not_stop_the_shutdown(panel):
    """Shutdown must leave no thread behind, whatever the worker does."""
    worker = _RecordingWorker(explode=True)
    panel._ai_worker = worker
    panel._ai_thread = None

    panel.shutdown()

    assert worker.cancelled == 1
    assert panel._ai_worker is None
    assert panel._retired == []


# -- the end of a stream ------------------------------------------------------

def test_a_stream_that_ends_with_no_block_open_appends_nothing(panel):
    """``open_error_flow`` and a cleared console both leave no block to close."""
    panel._current_stdout = None
    panel._ai_buf = ["the ans", "wer"]
    before = panel.as_text()
    finished = []
    panel.ai_stream_finished.connect(lambda: finished.append(True))

    panel._on_stream_finished(True, "the answer")

    assert finished == [True]
    assert panel._ai_messages[-1] == {"role": "assistant",
                                      "content": "the answer"}
    assert panel.as_text() == before
    assert panel._ai_buf == []


# -- opening a plain output block on demand -----------------------------------

def test_a_plain_output_block_is_opened_when_the_last_entry_was_not_one(panel):
    """The helper opens one block and reuses it while stdout keeps arriving."""
    panel._ensure_stdout_block()
    first = panel._current_stdout

    assert first is not None
    assert panel._last_entry_kind == "stdout"

    panel._ensure_stdout_block()

    assert panel._current_stdout is first


def test_a_block_with_nothing_new_is_skipped_while_a_fresh_one_goes(panel):
    """Only what arrived since the last question is attached to this one."""
    panel.append_stdout("the run said this\n")
    first, _ = panel._console_context_for_question("what happened?")
    assert "the run said this" in first

    panel.append_error("Traceback (most recent call last):\n  boom\n")

    second, _ = panel._console_context_for_question("and why?")

    assert "boom" in second
    assert "the run said this" not in second


@pytest.fixture
def linked_log(qtbot, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    block = cp._StdoutBlock('See https://example.org/details. Plain output here.')
    qtbot.addWidget(block)
    block.resize(850, 100)
    block.show()
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    qtbot.waitExposed(block)
    return block, opened


def _console_text_point(block, token, offset=1):
    from PySide6.QtCore import QPoint

    cursor = block.textCursor()
    cursor.setPosition(block.toPlainText().index(token) + offset)
    rect = block.cursorRect(cursor)
    return QPoint(rect.x() + 2, rect.center().y())


def test_log_hover_and_left_click_open_only_the_web_link(linked_log, qtbot):
    from PySide6.QtCore import Qt

    block, opened = linked_log
    link = _console_text_point(block, 'https://')
    plain = _console_text_point(block, 'Plain')
    qtbot.mouseMove(block.viewport(), link)
    assert block.viewport().cursor().shape() == Qt.PointingHandCursor
    qtbot.mouseClick(block.viewport(), Qt.LeftButton, pos=link)
    assert opened == ['https://example.org/details']
    qtbot.mouseMove(block.viewport(), plain)
    assert block.viewport().cursor().shape() == Qt.IBeamCursor
    qtbot.mouseClick(block.viewport(), Qt.LeftButton, pos=plain)
    assert opened == ['https://example.org/details']


@pytest.mark.parametrize('button_name', ['RightButton', 'MiddleButton'])
def test_log_nonleft_release_over_web_link_does_not_launch(linked_log, qtbot, button_name):
    from PySide6.QtCore import Qt

    block, opened = linked_log
    qtbot.mouseRelease(block.viewport(), getattr(Qt, button_name),
                       pos=_console_text_point(block, 'https://'))
    assert opened == []


@pytest.mark.parametrize('address', ['file:///tmp/private.txt', 'ftp://example.org/file',
                                    'javascript:alert(1)'])
def test_log_nonweb_addresses_remain_selectable_text(linked_log, qtbot, address):
    from PySide6.QtCore import Qt

    block, opened = linked_log
    block.setPlainText('Output ' + address + ' finished')
    point = _console_text_point(block, address)
    assert block.link_at(point) == ''
    qtbot.mouseClick(block.viewport(), Qt.LeftButton, pos=point)
    assert opened == []
    assert block.toPlainText() == 'Output ' + address + ' finished'


def test_selecting_log_url_text_does_not_open_the_browser(linked_log, qtbot):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    block, opened = linked_log
    start = _console_text_point(block, 'https://', offset=0)
    end = _console_text_point(block, 'https://', offset=15)
    qtbot.mousePress(block.viewport(), Qt.LeftButton, pos=start)
    global_end = block.viewport().mapToGlobal(end)
    event = QMouseEvent(QEvent.MouseMove, QPointF(end), QPointF(global_end),
                        Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(block.viewport(), event)
    qtbot.mouseRelease(block.viewport(), Qt.LeftButton, pos=end)
    assert block.textCursor().hasSelection()
    assert 'https://' in block.textCursor().selectedText()
    assert opened == []


# -- spaCR AI answers ---------------------------------------------------------

def test_a_hash_heading_becomes_a_bold_line_of_its_own():
    """``# Heading`` lines are headings with nothing after them on the line."""
    assert cp.markdown_lines("# What happened\nThe run stopped.") == [
        ("What happened", True), ("The run stopped.", False)]


def test_blank_lines_collapse_and_a_stray_empty_bold_marker_is_dropped():
    """Runs of blank lines become one; an empty ``** **`` leaves no tail."""
    assert cp.markdown_lines("First.\n\n\n\nSecond.\n** **") == [
        ("First.", False), ("", False), ("Second.", False)]


def test_an_answer_for_an_entry_outside_the_console_opens_a_new_one(panel):
    """A block that was never placed, or lost its place, is not written to."""
    stray = cp._StdoutBlock()
    try:
        filled = panel.fill_ai_entry(stray, "Set src.")
        assert filled is not stray
        assert filled.toPlainText() == "Set src."
        assert filled.property("consoleContextKind") == "spacr_ai"
    finally:
        stray.deleteLater()


def test_a_saved_answer_survives_a_heading_that_is_already_gone(panel):
    """A deleted heading label is skipped; the answer still shows."""
    block = panel.begin_ai_entry(thinking=False)
    shiboken6.delete(block._ai_bar._label)
    filled = panel.fill_ai_entry(block, "Saved advice.", saved=True)
    assert filled is block and block.toPlainText() == "Saved advice."
    other = panel.begin_ai_entry(thinking=False)
    other._ai_bar = None
    assert panel.fill_ai_entry(other, "Again.", saved=True) is other
    assert other.toPlainText() == "Again."


def test_routine_records_arriving_together_are_written_in_one_flush(panel,
                                                                   qtbot):
    """A second INFO record joins the pending flush instead of restarting it."""
    import logging

    panel._on_log_record("first record", logging.INFO)
    assert panel._log_flush.isActive()
    panel._on_log_record("second record", logging.INFO)
    assert panel._log_pending[-2:] == ["first record", "second record"]
    qtbot.waitUntil(lambda: not panel._log_pending)


# -- the scrollback cap and links ---------------------------------------------

def test_one_paragraph_longer_than_the_cap_is_kept_whole(qtbot, monkeypatch):
    """Only whole paragraphs are dropped, so a lone long line stays."""
    monkeypatch.setattr(cp._StdoutBlock, "MAX_CHARS", 200)
    block = cp._StdoutBlock()
    qtbot.addWidget(block)
    block.append("x" * 500)
    assert block.toPlainText() == "x" * 500


def test_a_pointer_over_text_that_cannot_be_read_shows_the_text_cursor(
        linked_log, monkeypatch):
    """A failing link lookup falls back to the ordinary text cursor."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    block, _opened = linked_log

    def broken(_pos):
        """Fail the way a block whose document is being replaced does."""
        raise RuntimeError("document is gone")

    monkeypatch.setattr(block, "link_at", broken)
    block.viewport().setCursor(Qt.PointingHandCursor)
    point = QPointF(5, 5)
    block.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, point, point,
                                     Qt.NoButton, Qt.NoButton,
                                     Qt.NoModifier))
    assert block.viewport().cursor().shape() == Qt.IBeamCursor
