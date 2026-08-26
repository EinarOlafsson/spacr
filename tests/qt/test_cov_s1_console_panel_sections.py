"""Where a console section starts and stops, and what closing one has to do.

The console is a transcript with headings, and three separate gestures depend
on one definition of a section's span: copying it, folding it and raising it.
They are driven here from the outside so that they cannot disagree -- a heading
that copies more than it folds is the defect this span exists to prevent.

The teardown paths are here for the opposite reason: they run when the window
is already closing, so a raise in any of them destroys a QThread that is still
running and takes the process with it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import console_panel as cp
from spacr.qt.widgets.console_panel import ConsolePanel

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    return widget


def _bars(panel):
    out = []
    for index in range(panel._entries.count()):
        item = panel._entries.itemAt(index)
        widget = item.widget() if item is not None else None
        if isinstance(widget, cp._TopicBar):
            out.append(widget)
    return out


# ---------------------------------------------------------------------------
# Where a section ends
# ---------------------------------------------------------------------------

def test_a_heading_that_is_not_in_this_console_spans_nothing(panel, qtbot):
    """``_section_span`` is asked about a bar the caller holds. One that was
    never inserted has no start, and every gesture built on the span has to
    answer emptily rather than take the whole transcript."""
    stray = cp._TopicBar("not inserted")
    qtbot.addWidget(stray)

    assert panel._section_span(stray) == (None, None)
    assert panel.section_text(stray) == ""
    assert panel.section_body(stray) == []


def test_a_module_banner_spans_past_the_output_banner_under_it(panel):
    """``append_stdout`` inserts its own "spaCR output" bar, so a module
    banner is followed immediately by another banner. Stopping at the first
    boundary would copy a title and nothing else, and fold a section that
    hides nothing."""
    panel.begin_topic("Mask")
    panel.append_stdout("segmenting field 1\n")
    panel.append_stdout("segmenting field 2\n")
    panel.begin_topic("Measure")
    panel.append_stdout("measuring\n")

    module_bar = _bars(panel)[0]
    text = panel.section_text(module_bar)

    assert "Mask" in text
    assert "segmenting field 1" in text
    assert "segmenting field 2" in text
    assert "Measure" not in text
    body = panel.section_body(module_bar)
    assert any(isinstance(widget, cp._TopicBar) for widget in body), \
        "the output banner under the module banner is part of its section"


def test_the_last_section_runs_to_the_end_of_the_transcript(panel):
    panel.begin_topic("Mask")
    panel.append_stdout("only output\n")

    last = _bars(panel)[-1]

    start, stop = panel._section_span(last)
    assert start is not None
    assert stop is None
    assert "only output" in panel.section_text(last)


def test_copying_a_section_puts_exactly_that_section_on_the_clipboard(panel):
    panel.begin_topic("Mask")
    panel.append_stdout("one\ntwo\n")
    panel.begin_topic("Measure")
    panel.append_stdout("three\n")
    QApplication.clipboard().setText("")

    _bars(panel)[0]._copy_section()

    copied = QApplication.clipboard().text()
    assert "one" in copied and "two" in copied
    assert "three" not in copied


# ---------------------------------------------------------------------------
# Folding and raising
# ---------------------------------------------------------------------------

def test_raising_a_section_leaves_a_nested_heading_the_user_folded_alone(
        panel, qtbot):
    """Expanding the section above a folded heading is not a request to
    unfold what is inside it -- the user folded that one deliberately."""
    panel.begin_topic("Mask")
    panel.append_stdout("first line\nsecond line\n")
    module_bar, output_bar = _bars(panel)[0], _bars(panel)[1]
    panel.collapse_section(output_bar)
    hidden = list(panel.section_body(output_bar))
    assert hidden and all(w.isHidden() for w in hidden)

    panel.collapse_section(module_bar)
    panel.raise_section(module_bar)

    assert module_bar.is_expanded() is True
    assert output_bar.isHidden() is False
    assert output_bar.is_expanded() is False
    assert all(w.isHidden() for w in hidden), \
        "the nested section the user folded must stay folded"
    assert panel._follow_output is False


def test_dragging_back_to_the_bottom_makes_the_view_follow_again(panel):
    """Where the viewport is decides whether new output scrolls it, and it
    answers the same way however the reader got there -- a heading click and
    a scrollbar drag are the same question."""
    panel.begin_topic("Mask")
    for index in range(80):
        panel.append_stdout(f"line {index}\n")
    scrollbar = panel._scroll.verticalScrollBar()
    scrollbar.setRange(0, 900)

    panel._on_console_scrolled(0)
    assert panel._follow_output is False

    panel._on_console_scrolled(880)
    assert panel._follow_output is False

    panel._on_console_scrolled(900)
    assert panel._follow_output is True


# ---------------------------------------------------------------------------
# The working dots
# ---------------------------------------------------------------------------

def test_the_working_dots_take_the_colour_of_whoever_is_answering(qtbot):
    """Every provider answers in its own colour, and the "working" indicator
    is part of that answer -- a grey spinner over a coloured reply says the
    wrong provider is thinking."""
    dots = cp._WorkingDots()
    qtbot.addWidget(dots)

    dots.set_color("#ff8800")

    assert dots._color == "#ff8800"
    assert "#ff8800" in dots.styleSheet()


def test_the_copy_glyph_brightens_under_the_pointer(qtbot, qt_theme_applied):
    """The glyph is the only thing on a heading that can be clicked on its
    own, and it carries no label. Brightening under the cursor is the whole
    affordance, so it has to actually change what is painted."""
    button = cp._CopyGlyphButton()
    qtbot.addWidget(button)
    button.resize(20, 20)

    resting = button.grab().toImage()
    button.underMouse = lambda: True
    hovered = button.grab().toImage()

    assert hovered != resting


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_a_stream_is_not_started_when_no_provider_is_selected(panel):
    """The AI panel is usable with no provider configured; asking one that
    does not exist for a stream would raise inside a keystroke handler."""
    panel._current_provider_name = ""

    panel._start_stream("you are a helpful assistant")

    assert panel._ai_thread is None
    assert panel._ai_worker is None


def test_a_retired_thread_that_is_still_running_is_kept(panel):
    """The list exists to hold a reference until the QThread has exited.
    Dropping one that is still running is the abort this guards against."""
    class _Running:
        def isRunning(self):
            return True

    class _Gone:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    running, gone = _Running(), _Gone()
    panel._retired = [(running, object()), (gone, object())]

    panel._prune_retired()

    assert [pair[0] for pair in panel._retired] == [running]
    panel._retired = []


def test_a_provider_that_raises_on_cancel_does_not_stop_the_shutdown(
        panel, monkeypatch):
    """Shutdown runs while the window is closing. A raise here leaves a live
    QThread behind, which is exactly what Qt aborts the process over."""
    class _Angry:
        def cancel_stream(self):
            raise RuntimeError("the CLI is already gone")

    class _AngryWorker:
        def cancel(self):
            raise RuntimeError("the worker is already gone")

    monkeypatch.setattr(cp.ai_module, "list_providers", lambda: [_Angry()])
    panel._ai_worker = _AngryWorker()
    panel._ai_thread = None

    panel.shutdown()

    assert panel._ai_worker is None
    assert panel._ai_thread is None


# ---------------------------------------------------------------------------
# What the AI is shown of the console
# ---------------------------------------------------------------------------

def test_output_is_offered_to_the_ai_once_and_only_once(panel):
    """The console context is what has NOT been sent yet.

    Re-sending the same lines with every question spends the model's budget
    on text it has already read, and makes a long run's context grow without
    bound. A block is marked sent only after it is actually included.
    """
    from spacr.qt.ai import settings as ai_settings

    if not ai_settings.get_console_aware():
        pytest.skip("console-aware context is switched off in this store")

    panel.append_stdout("segmenting field 1\nsegmenting field 2\n")
    panel.append_error("Traceback (most recent call last):\nBoom\n")

    context, label = panel._console_context_for_question("what happened?")

    assert "segmenting field 2" in context
    assert "--- complete traceback ---" in context
    assert "Boom" in context
    assert label

    again, label = panel._console_context_for_question("and now?")

    assert again == ""
    assert "no new output" in label.lower()
