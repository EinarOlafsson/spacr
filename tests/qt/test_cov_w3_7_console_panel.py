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


import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QKeyEvent, QMouseEvent, QTextCursor
from PySide6.QtWidgets import QApplication

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

@pytest.mark.parametrize("flashing", [False, True])
def test_the_badge_draws_even_before_the_theme_exists(qtbot, monkeypatch, flashing):
    """An unavailable painting palette keeps the idle and copied glyph legible."""
    badge = cp._CopyGlyphButton()
    qtbot.addWidget(badge)
    badge.underMouse = lambda: False
    if flashing:
        badge.flash_copied()
    monkeypatch.setattr(cp, "_painting_palette",
                        lambda: {"fg_dim": "#888888", "button_accent": "#888888"})
    expected = badge.grab().toImage()
    attempts = []

    def unavailable():
        attempts.append(1)
        raise RuntimeError("no application palette yet")

    monkeypatch.setattr(cp, "_painting_palette", unavailable)
    actual = badge.grab().toImage()
    assert attempts, "the glyph must try the cached painting palette"
    assert not actual.isNull()
    assert actual == expected, "an unavailable palette must draw the neutral fallback ink"


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


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def test_a_heading_that_is_not_in_this_console_has_no_section(panel, qtbot):
    stranger = cp._TopicBar("not mine")
    qtbot.addWidget(stranger)
    assert panel.section_text(stranger) == ""


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


@pytest.mark.usefixtures("console_info_enabled")
def test_the_error_flow_needs_a_provider_before_it_can_explain(panel):
    panel.open_error_flow("Traceback (most recent call last): …")
    assert "Enable AI in the actions row" in panel.as_text()
    assert panel._ai_messages == []


# ---------------------------------------------------------------------------
# A section that goes away between the click and the layout
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("console_info_enabled")
def test_copying_the_whole_console_returns_what_it_copied(panel):
    panel.begin_topic("a topic")
    panel.append_stdout("a line\n")
    copied = panel.copy_all()
    assert "a line" in copied
    assert QApplication.clipboard().text() == copied


@pytest.mark.usefixtures("console_info_enabled")
def test_a_clipboard_that_is_not_there_still_returns_the_text(panel,
                                                              monkeypatch):
    panel.append_stdout("a line\n")
    monkeypatch.setattr(QApplication, "clipboard",
                        staticmethod(lambda: (_ for _ in ()).throw(
                            RuntimeError("no clipboard on this platform"))))
    assert "a line" in panel.copy_all()


def test_the_ai_toggle_and_provider_are_set_from_outside(panel):
    panel.set_ai_active(True)
    assert panel._ai_active is True
    panel.set_ai_provider("Claude")
    assert panel._current_provider_name == "Claude"
    panel.set_ai_provider(None)
    assert panel._current_provider() is None
