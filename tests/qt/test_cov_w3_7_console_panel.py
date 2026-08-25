"""The console's small parts: the badge, the handles, and every early return.

The console is covered from the outside by a dozen files -- what it prints,
where it scrolls to, how the sections fold, whether a worker thread can write
to it. What is driven here is the inside of the widgets that make it up, and
in particular the paths that END in a return: an empty write, a re-entrant
one, a section with nothing under it, a preference that cannot be read.

Each of those is load-bearing. The re-entrancy refusal is a documented
segfault guard; the empty-write returns are what keeps the profile hook from
building a widget per log record; and the fallbacks exist because this panel
is built before the theme in some hosts.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from spacr.qt.widgets import console_panel as cp
from spacr.qt.widgets.console_panel import ConsolePanel

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    return widget


def _mouse(kind, pos, button=Qt.LeftButton, buttons=Qt.NoButton, glob=None):
    point = QPointF(*pos)
    return QMouseEvent(kind, point, QPointF(*(glob or pos)), button, buttons,
                       Qt.NoModifier)


# ---------------------------------------------------------------------------
# Which colour the AI answers in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider,expected", [
    ("Claude 3.5", cp.AI_COLOR_CLAUDE),
    ("anthropic/claude", cp.AI_COLOR_CLAUDE),
    ("gpt-4o", cp.AI_COLOR_OPENAI),
    ("OpenAI", cp.AI_COLOR_OPENAI),
    ("ChatGPT", cp.AI_COLOR_OPENAI),
    ("gemini-pro", cp.AI_COLOR_GEMINI),
    ("Google", cp.AI_COLOR_GEMINI),
    ("a local llama", cp.AI_COLOR_DEFAULT),
    (None, cp.AI_COLOR_DEFAULT),
])
def test_each_provider_answers_in_its_own_colour(provider, expected):
    assert cp.ai_color_for_provider(provider) == expected


# ---------------------------------------------------------------------------
# The copy badge
# ---------------------------------------------------------------------------

def test_the_badge_draws_even_before_the_theme_exists(qtbot, monkeypatch):
    """The panel is built before the palette in some hosts; it still paints."""
    badge = cp._CopyGlyphButton()
    qtbot.addWidget(badge)
    monkeypatch.setattr(cp, "active_palette",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("no application palette yet")))
    assert not badge.grab().isNull()


# ---------------------------------------------------------------------------
# The topic bar
# ---------------------------------------------------------------------------

def test_a_heading_answers_the_keyboard_and_ignores_other_keys(qtbot):
    """A control only a mouse can reach is one some users cannot reach."""
    bar = cp._TopicBar("spaCR output")
    qtbot.addWidget(bar)
    toggled = []
    bar._activate = lambda: toggled.append(True)

    for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
        bar.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    assert len(toggled) == 3

    bar.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert len(toggled) == 3


def test_a_heading_with_no_panel_above_it_copies_nothing(qtbot):
    """The walk to the panel is bounded; an orphan bar copies nothing.

    Nothing is the outcome that matters: a bar that cannot find its panel
    and clears the clipboard anyway has thrown away whatever the user was
    carrying, and raises nothing while doing it.
    """
    QApplication.clipboard().setText("untouched")

    bar = cp._TopicBar("spaCR output")
    qtbot.addWidget(bar)
    bar._copy_section()                     # no panel anywhere above it
    assert QApplication.clipboard().text() == "untouched"

    deep = QWidget()
    qtbot.addWidget(deep)
    node = deep
    for _ in range(8):
        node = QWidget(node)
    orphan = cp._TopicBar("spaCR output", node)
    orphan._copy_section()                  # panel is further than the walk
    assert QApplication.clipboard().text() == "untouched"


def test_a_heading_with_no_section_under_it_copies_nothing(panel, qtbot):
    """``section_text`` answers "" for a bar the console does not lay out as
    a section, and an empty copy would silently clear the clipboard."""
    bar = cp._TopicBar("appended past the end")
    panel._entries.addWidget(bar)
    assert panel.section_text(bar) == ""

    QApplication.clipboard().setText("untouched")
    bar._copy_section()
    assert QApplication.clipboard().text() == "untouched"


def test_a_clipboard_that_refuses_the_text_is_not_a_crash(panel, monkeypatch):
    """A copy that did not happen may not say it did.

    The badge only flashes "copied" after the clipboard took the text, so
    the flash is what separates a refused copy from a successful one --
    both of which raise nothing.
    """
    panel.begin_topic("a topic")
    panel.append_stdout("a line of output\n")
    bar = next(w for w in panel.findChildren(cp._TopicBar))
    assert panel.section_text(bar).strip()        # there was text to copy

    flashed = []
    monkeypatch.setattr(type(bar._copy_btn), "flash_copied",
                        lambda self: flashed.append(True))
    monkeypatch.setattr(QApplication, "clipboard",
                        staticmethod(lambda: (_ for _ in ()).throw(
                            RuntimeError("no clipboard on this platform"))))
    bar._copy_section()
    assert flashed == []


# ---------------------------------------------------------------------------
# One block of output
# ---------------------------------------------------------------------------

def test_an_empty_append_costs_nothing(qtbot):
    """The profile hook emits a record per function call; an empty one is
    not a paragraph."""
    block = cp._StdoutBlock("first\n")
    qtbot.addWidget(block)
    before = block.toPlainText()
    block.append("")
    assert block.toPlainText() == before


def test_a_pinned_height_wins_over_the_document(qtbot):
    block = cp._StdoutBlock("one\ntwo\nthree\n")
    qtbot.addWidget(block)
    automatic = block.sizeHint().height()
    block.set_user_height(240)
    assert block.sizeHint().height() == 240
    assert block.height() == 240

    block.reset_user_height()
    assert block.sizeHint().height() == automatic


def test_the_pinned_height_stays_inside_the_range_it_offers(qtbot):
    block = cp._StdoutBlock("one\n")
    qtbot.addWidget(block)
    block.set_user_height(1)
    assert block.sizeHint().height() == 48
    block.set_user_height(999_999)
    assert block.sizeHint().height() == 4000


def test_dragging_the_handle_resizes_the_section_and_a_double_click_frees_it(
        qtbot):
    block = cp._StdoutBlock("one\ntwo\nthree\n")
    qtbot.addWidget(block)
    block.resize(400, 120)
    handle = block._height_handle

    handle.mousePressEvent(_mouse(QEvent.MouseButtonPress, (10, 3),
                                  glob=(500, 500)))
    handle.mouseMoveEvent(_mouse(QEvent.MouseMove, (10, 3),
                                 button=Qt.NoButton, buttons=Qt.LeftButton,
                                 glob=(500, 560)))
    assert block.height() == 180
    handle.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, (10, 3)))
    assert handle._press_y is None

    # A move after the release is not a drag.
    handle.mouseMoveEvent(_mouse(QEvent.MouseMove, (10, 3),
                                 button=Qt.NoButton, buttons=Qt.LeftButton,
                                 glob=(500, 900)))
    assert block.height() == 180

    handle.mouseDoubleClickEvent(_mouse(QEvent.MouseButtonDblClick, (10, 3)))
    assert block._user_height is None


def test_the_right_button_does_not_start_a_drag(qtbot):
    block = cp._StdoutBlock("one\n")
    qtbot.addWidget(block)
    handle = block._height_handle
    handle.mousePressEvent(_mouse(QEvent.MouseButtonPress, (10, 3),
                                  button=Qt.RightButton))
    assert handle._press_y is None
    handle.mouseDoubleClickEvent(_mouse(QEvent.MouseButtonDblClick, (10, 3),
                                        button=Qt.RightButton))


# ---------------------------------------------------------------------------
# A chat bubble
# ---------------------------------------------------------------------------

def test_a_bubble_with_no_width_yet_does_not_measure_itself(qtbot,
                                                            monkeypatch):
    """Wrapping to a width of nothing gives a nonsense height, so the
    measurement is not taken at all -- and the count of measurements is the
    only thing that tells "skipped" apart from "measured and discarded"."""
    bubble = cp._Bubble("user", "a question")
    qtbot.addWidget(bubble)

    widths = []
    monkeypatch.setattr(type(bubble._label), "heightForWidth",
                        lambda self, w: widths.append(w) or 40)

    bubble.resize(0, 0)
    bubble._recalc()
    assert widths == []

    bubble.resize(300, 40)
    bubble._recalc()
    assert widths and all(w == max(120, 300 - bubble._H_PAD) for w in widths)


def test_a_bubble_is_as_tall_as_its_text_needs_at_that_width(qtbot):
    """The bubble sizes itself; a fixed height clips the answer."""
    short = cp._Bubble("user", "hi")
    long = cp._Bubble("ai", "an answer long enough to wrap over several "
                            "lines once the bubble is only a little wider "
                            "than a word, which is what a narrow console "
                            "does to every reply it is given")
    qtbot.addWidget(short)
    qtbot.addWidget(long)
    for bubble in (short, long):
        bubble.resize(200, 40)
        bubble.show()
        qtbot.waitExposed(bubble)
    assert long.height() > short.height()
    for bubble in (short, long):
        bubble.hide()


def test_a_bubble_being_measured_does_not_measure_itself_again(qtbot):
    bubble = cp._Bubble("user", "a question")
    qtbot.addWidget(bubble)
    bubble.resize(300, 40)
    bubble._recalc_guard = True
    tall = bubble.height()
    bubble.resize(80, 40)
    assert bubble.height() == tall
    bubble._recalc_guard = False


# ---------------------------------------------------------------------------
# The chat input
# ---------------------------------------------------------------------------

def test_enter_sends_and_shift_enter_makes_a_newline(qtbot):
    box = cp._ChatInput()
    qtbot.addWidget(box)
    sent = []
    box.submitted.connect(lambda: sent.append(box.toPlainText()))

    box.setPlainText("what does this module do?")
    from PySide6.QtGui import QTextCursor

    box.moveCursor(QTextCursor.End)
    box.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                                Qt.NoModifier))
    assert sent == ["what does this module do?"]
    assert box.toPlainText() == "what does this module do?", \
        "plain Enter typed a newline as well as sending"

    box.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                                Qt.ShiftModifier))
    assert box.toPlainText().endswith("\n")
    assert len(sent) == 1

    box.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier,
                                "a"))
    assert box.toPlainText().endswith("a")


def test_a_dropped_file_is_never_read_into_the_chat_box(qtbot, tmp_path):
    """A large file read into the buffer freezes the whole application."""
    box = cp._ChatInput()
    qtbot.addWidget(box)
    box.setPlainText("keep me")

    dropped = QMimeData()
    dropped.setUrls([QUrl.fromLocalFile(str(tmp_path / "plate"))])
    assert box.canInsertFromMimeData(dropped) is False
    box.insertFromMimeData(dropped)
    assert box.toPlainText() == "keep me"

    typed = QMimeData()
    typed.setText(" and this")
    assert box.canInsertFromMimeData(typed) is True
    box.insertFromMimeData(typed)
    assert "and this" in box.toPlainText()


# ---------------------------------------------------------------------------
# Building the panel when the surroundings are not there
# ---------------------------------------------------------------------------

def test_a_panel_builds_without_the_shared_log_handler(qtbot, monkeypatch):
    """The console is usable on its own, outside the application shell."""
    monkeypatch.setattr("spacr.qt.logging_util.get_signal_handler",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("no handler in this process")))
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    widget.append_stdout("still works\n")
    assert "still works" in widget.as_text()


def test_a_panel_builds_without_the_theme(qtbot, monkeypatch):
    monkeypatch.setattr(cp, "active_palette",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("no palette yet")))
    monkeypatch.setattr("spacr.qt.theme.make_transparent",
                        lambda *w: (_ for _ in ()).throw(
                            RuntimeError("no stylesheet yet")))
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    assert widget._split is not None


def test_an_unreadable_zoom_preference_keeps_the_default_size(panel,
                                                              monkeypatch):
    monkeypatch.setattr("spacr.qt.preferences.get_font_scale",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("no settings store")))
    panel.apply_zoom()
    assert panel._font_pt >= 6


def test_a_saved_arrangement_that_cannot_be_restored_falls_back(qtbot,
                                                                monkeypatch):
    widget = ConsolePanel(persist_key="cov_w3_7_console")
    qtbot.addWidget(widget)
    cp.set_split_state("cov_w3_7_console", b"not a splitter state")
    monkeypatch.setattr(type(widget._split), "restoreState",
                        lambda self, blob: (_ for _ in ()).throw(
                            RuntimeError("unreadable blob")))
    applied = []
    monkeypatch.setattr(type(widget), "_apply_default_split",
                        lambda self: applied.append(True))
    widget._restore_split()
    assert applied == [True]


# ---------------------------------------------------------------------------
# Writing to it
# ---------------------------------------------------------------------------

def test_an_empty_write_builds_no_widget(panel):
    before = panel._entries.count()
    panel.append_stdout("")
    panel.append_error("")
    panel.append_notice("")
    panel.append_notice("   \n")
    assert panel._entries.count() == before


def test_a_write_from_a_worker_is_relayed_rather_than_done_here(panel,
                                                                monkeypatch):
    """A worker thread building a QTextDocument is the documented segfault."""
    relayed = {"stdout": [], "error": [], "notice": []}
    # The panel's own slots are taken off first: they are what would deliver
    # the text on the GUI thread, and this test is about the hop, not the
    # landing. Left connected they run inline here -- one thread, so a
    # queued connection is a direct call -- and the answer below would be
    # "it wrote it" either way.
    panel._relay_stdout.disconnect()
    panel._relay_error.disconnect()
    panel._relay_notice.disconnect()
    panel._relay_stdout.connect(lambda text: relayed["stdout"].append(text))
    panel._relay_error.connect(lambda text: relayed["error"].append(text))
    panel._relay_notice.connect(
        lambda source, values: relayed["notice"].append((source, values)))
    monkeypatch.setattr(type(panel), "_on_gui_thread", lambda self: False)

    before = panel._entries.count()
    panel.append_stdout("from a worker\n")
    panel.append_error("Traceback…\n")
    panel.append_notice("a notice", a=1)

    assert relayed["stdout"] == ["from a worker\n"]
    assert relayed["error"] == ["Traceback…\n"]
    assert relayed["notice"] == [("a notice", {"a": 1})]
    assert panel._entries.count() == before, \
        "the worker's text was written on the worker's thread"


def test_a_write_from_inside_a_write_is_refused(panel):
    """The re-entrancy guard: a log record emitted while the console is
    writing must not build a second document inside the first."""
    from spacr.qt.verbose_logger import console_write

    panel.append_stdout("first\n")
    before = panel.as_text()
    with console_write():
        panel.append_stdout("re-entrant\n")
        panel.append_error("re-entrant traceback\n")
    assert panel.as_text() == before


def test_a_warning_record_is_shown_as_an_error(panel):
    panel._on_log_record("something went wrong", logging.WARNING)
    assert "something went wrong" in panel.as_text()


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def test_a_stretch_in_the_layout_is_not_a_line_of_text(panel):
    panel.append_stdout("a line\n")
    panel._entries.addStretch(1)            # an item with no widget
    assert "a line" in panel.as_text()


def test_a_label_in_the_console_is_exported_by_its_text(panel):
    panel.begin_topic("a topic")
    label = QLabel("a plain label")
    panel._entries.addWidget(label)
    panel._entries.addWidget(QLabel("   "))
    text = panel.as_text()
    assert "a plain label" in text
    assert text.count("a plain label") == 1


def test_a_heading_that_is_not_in_this_console_has_no_section(panel, qtbot):
    stranger = cp._TopicBar("not mine")
    qtbot.addWidget(stranger)
    assert panel.section_text(stranger) == ""


def test_the_last_section_runs_to_the_end_of_the_console(panel):
    panel.begin_topic("first")
    panel.append_stdout("under the first\n")
    panel.begin_topic("second")
    panel.append_stdout("under the second\n")

    bars = panel.findChildren(cp._TopicBar)
    last = panel.section_text(bars[-1])
    assert "under the second" in last
    assert "under the first" not in last


# ---------------------------------------------------------------------------
# What the banner says the output came from
# ---------------------------------------------------------------------------

def test_the_banner_names_the_module_and_the_function(panel):
    """"spaCR output" alone does not say which run is printing."""
    panel.set_active_app("Mask")
    assert panel._output_banner("spaCR output") == "spaCR output  —  Mask"

    panel.set_run_context("mask", "preprocess_generate_masks")
    assert panel._output_banner("spaCR output") == (
        "spaCR output  —  mask  —  preprocess_generate_masks")

    panel.set_run_context()
    assert panel._output_banner("spaCR output") == "spaCR output  —  Mask"


# ---------------------------------------------------------------------------
# A bubble that is already being measured
# ---------------------------------------------------------------------------

def test_a_bubble_measuring_itself_does_not_start_again(qtbot):
    """``setFixedHeight`` sends a resize event, which asks for another fit."""
    bubble = cp._Bubble("user", "a question")
    qtbot.addWidget(bubble)
    bubble.resize(300, 40)
    bubble._recalc_guard = True
    bubble._label.setFixedHeight(11)
    bubble._recalc()
    assert bubble._label.height() == 11, "the guarded call re-fitted anyway"
    bubble._recalc_guard = False


def test_a_label_that_cannot_measure_a_wrap_falls_back_to_its_hint(qtbot,
                                                                   monkeypatch):
    bubble = cp._Bubble("ai", "an answer")
    qtbot.addWidget(bubble)
    bubble.resize(300, 40)
    monkeypatch.setattr(type(bubble._label), "heightForWidth",
                        lambda self, width: 0)
    bubble._recalc()
    assert bubble._label.height() == bubble._label.sizeHint().height()
    assert bubble.height() == bubble._label.height() + bubble._V_PAD


# ---------------------------------------------------------------------------
# The chat half, without a provider
# ---------------------------------------------------------------------------

def test_an_empty_message_is_not_sent(panel):
    before = panel._entries.count()
    panel._input.setPlainText("   \n  ")
    panel._on_submit()
    assert panel._entries.count() == before
    assert panel._input.toPlainText() == "   \n  ", \
        "the box was cleared as though something had been sent"


def test_a_streamed_chunk_opens_a_block_in_the_providers_colour(panel):
    panel._current_provider_name = "Claude"
    panel._on_stage("thinking")             # quiet by design
    panel._on_chunk("the first half ")
    panel._on_chunk("and the second\n")
    assert panel._last_entry_kind == "ai"
    assert "the first half and the second" in panel.as_text()


def test_a_stream_that_returned_nothing_says_so(panel):
    panel._ai_buf = []
    panel._on_stream_finished(True, "")
    assert "empty response" in panel.as_text()
    assert panel._ai_messages[-1] == {"role": "assistant", "content": ""}


def test_a_failed_stream_reports_the_provider_error(panel):
    panel._on_stream_finished(False, "rate limited")
    assert "[AI error] rate limited" in panel.as_text()


def test_the_error_flow_needs_a_provider_before_it_can_explain(panel):
    panel.open_error_flow("Traceback (most recent call last): …")
    assert "Enable AI in the actions row" in panel.as_text()
    assert panel._ai_messages == []


# ---------------------------------------------------------------------------
# A section that goes away between the click and the layout
# ---------------------------------------------------------------------------

def test_a_section_torn_down_mid_click_is_not_a_crash(panel, qapp):
    """``raise_section`` scrolls on a zero-timer, so the bar can be gone by
    the time the scroll runs -- which is what ``clear()`` does to it."""
    panel.begin_topic("a topic")
    panel.append_stdout("under it\n")
    bar = panel.findChildren(cp._TopicBar)[-1]

    panel.clear()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)

    with pytest.raises(RuntimeError):
        bar.rect()                          # the C++ half really is gone
    panel._scroll_widget_to_top(bar)
    assert panel._is_raised(bar) is False


def test_copying_the_whole_console_returns_what_it_copied(panel):
    panel.begin_topic("a topic")
    panel.append_stdout("a line\n")
    copied = panel.copy_all()
    assert "a line" in copied
    assert QApplication.clipboard().text() == copied


def test_a_clipboard_that_is_not_there_still_returns_the_text(panel,
                                                              monkeypatch):
    panel.append_stdout("a line\n")
    monkeypatch.setattr(QApplication, "clipboard",
                        staticmethod(lambda: (_ for _ in ()).throw(
                            RuntimeError("no clipboard on this platform"))))
    assert "a line" in panel.copy_all()


def test_clearing_takes_the_entries_and_the_conversation_with_it(panel):
    panel.begin_topic("a topic")
    panel.append_stdout("a line\n")
    panel._ai_messages.append({"role": "user", "content": "hello"})
    panel._console_sent_lengths["stdout:0"] = 12

    panel.clear()
    assert panel._entries.count() == 1      # the trailing stretch, and nothing
    assert panel.as_text().strip() == ""
    assert panel._ai_messages == []
    assert panel._console_sent_lengths == {}
    assert panel._current_stdout is None


def test_the_ai_toggle_and_provider_are_set_from_outside(panel):
    panel.set_ai_active(True)
    assert panel._ai_active is True
    panel.set_ai_provider("Claude")
    assert panel._current_provider_name == "Claude"
    panel.set_ai_provider(None)
    assert panel._current_provider() is None
