"""The setup slides keep working when the parts around them do not.

INVARIANTS 10: the fades, the palette, the ambient row and the bundled fonts
are decoration or optional, and every one of them can be
missing on a fresh machine -- which is the machine this screen is built for.
These drive the paths that only appear when something is absent, broken, or
already deleted.
"""
from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QResizeEvent, QShowEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

from spacr.qt.widgets import setup_slides as mod  # noqa: E402
from spacr.qt.widgets.setup_slides import (  # noqa: E402
    GPU_NOTE_BAND,
    SLIDES,
    SetupSlides,
    _say,
    graphics_card,
)


def _boom(*_a, **_k):
    raise RuntimeError("this part of the machine is not here")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A config dir of this test's own, so nothing writes the real profile."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences
    importlib.reload(preferences)
    yield
    importlib.reload(preferences)


@pytest.fixture
def slides(app, qtbot):
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    return dialog


@pytest.fixture
def debug_log(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.qt.setup_slides")
    return caplog


# ------------------------------------------------------------- the machine


def test_the_gpu_verdict_believes_torch_and_admits_defeat(monkeypatch,
                                                          debug_log):
    """The first slide's verdict is what tells a user spaCR can segment here.

    Torch is the authority, so its name is the one shown; when neither torch
    nor NVML can answer, the note has to say nothing was detected rather than
    name a card segmentation would then fail on.
    """
    usable = types.ModuleType("torch")
    usable.cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_name=lambda i: "NVIDIA Test 9000")
    monkeypatch.setitem(sys.modules, "torch", usable)
    monkeypatch.setattr("spacr.qt.widgets.home._nvml", _boom)
    assert graphics_card() == (True, "NVIDIA Test 9000")
    mute = types.ModuleType("torch")
    mute.cuda = types.SimpleNamespace(is_available=_boom)
    monkeypatch.setitem(sys.modules, "torch", mute)
    assert graphics_card() == (False, "")
    assert "torch could not be asked about the GPU" in debug_log.text
    assert "NVML could not name the GPU" in debug_log.text


def test_a_caption_survives_a_broken_translator(monkeypatch):
    """Every caption here goes through ``_say``, and this dialog is built
    before a language has necessarily been read -- so a lookup that raises has
    to leave the English standing, formatted, not take the screen with it."""
    monkeypatch.setattr("spacr.qt.i18n.tr", _boom)
    assert _say("{n} of {total}", n=2, total=7) == "2 of 7"
    assert _say("Language") == "Language"


# ---------------------------------------------------------------- the card


def test_the_card_is_dressed_without_fonts_or_theme(slides, monkeypatch,
                                                    debug_log):
    """A fresh profile meets this dialog before anything else has loaded, so
    the bundled faces and the transparency helper are reached through imports
    that can fail there -- and neither is worth a setup screen."""
    monkeypatch.setattr("spacr.qt.app._load_bundled_fonts", _boom)
    slides._use_the_light_face()
    assert slides.card.font().family() == mod.SLIDE_FONT
    assert "bundled fonts are not loadable here" in debug_log.text
    monkeypatch.delattr("spacr.qt.theme.make_transparent")
    slides._clear_the_containers()
    assert "no theme helper for transparency" in debug_log.text
    monkeypatch.setattr("spacr.qt.theme.make_transparent", _boom,
                        raising=False)
    slides._clear_the_containers()
    assert "a container would not go transparent" in debug_log.text
    assert slides._pages.count() == len(SLIDES)


def test_a_question_that_removed_itself_leaves_no_empty_row(app, qtbot,
                                                            monkeypatch):
    """A question can withdraw -- one whose choices are empty is
    dropped -- and the slide it was on closes over the gap: an empty labelled row
    reads as a broken control rather than a question that does not apply."""
    from spacr.qt import setup_screen
    real = setup_screen.questions
    monkeypatch.setattr(
        setup_screen, "questions",
        lambda: [q for q in real() if q[0] != "colour_blind"])
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    answers = dialog.answers()
    assert "theme" in answers
    assert "colour_blind" not in answers


def test_a_window_that_will_not_go_frameless_is_still_the_window(slides,
                                                                monkeypatch,
                                                                debug_log):
    """The rounded card is the whole look of this screen, but on a platform
    that refuses translucency the dialog keeps its frame and opens anyway --
    this runs inside ``__init__``, so a raise costs the setup screen."""
    assert slides._go_frameless() is True
    monkeypatch.setattr(
        "spacr.qt.widgets.glass._paint_nothing_behind_the_card", _boom)
    assert slides._go_frameless() is False
    assert "the setup screen would not go frameless" in debug_log.text


# --------------------------------------------------------------- the terms


def test_the_screen_builds_without_a_palette(slides, monkeypatch, debug_log):
    """The accent blue is a look; the closing word and the terms are not. A
    theme that cannot answer costs the colour and nothing else -- the word
    keeps its size, the terms note is still there to be filled in, and the
    greyed ink falls back to one that works on both themes."""
    page = slides._closing_page("Done", "Welcome to spaCR")
    assert "color:" in slides._done_word.styleSheet()
    monkeypatch.setattr("spacr.qt.theme.active_palette", _boom)
    page = slides._closing_page("Done", "Welcome to spaCR")
    assert "color:" not in slides._done_word.styleSheet()
    assert f"{mod.DONE_POINTS}pt" in slides._done_word.styleSheet()
    terms = slides._terms_page()
    assert slides._agree_note.styleSheet() == ""
    assert SetupSlides._dim_ink() == "#6b6f76"
    assert "no palette for the greyed terms" in debug_log.text
    assert page.isVisible() is False and terms.isVisible() is False


def test_the_terms_gate_is_drawn_and_the_slide_refuses_to_be_left(slides):
    """The gate is drawn from ``showEvent`` and from every resize, both of
    which can arrive before the terms page exists; the complaint under the
    switch goes when the switch is ticked and at no other time; and Next
    without the tick stays put and says why rather than walking past a
    licence."""
    from spacr.qt import terms as terms_module
    slides._draw_the_terms_gate(True)
    assert slides._agree.isEnabled() is True
    assert slides._scroll_hint.isHidden() is True
    slides._draw_the_terms_gate(False)
    box = slides._agree
    assert box.isEnabled() is False
    assert slides._scroll_hint.isHidden() is False
    slides._agree_note.setVisible(True)
    slides._on_agreement_toggled(False)
    assert slides._agree_note.isHidden() is False
    slides._on_agreement_toggled(True)
    assert slides._agree_note.isHidden() is True
    terms_index = [t for t, _b, _k in SLIDES].index("Terms of use")
    slides._show_slide(terms_index)
    assert slides.next() == terms_index
    assert terms_module.WHY_NOT_YET in slides._agree_note.text()
    for name in ("_agree", "_agree_note", "_terms_body", "_scroll_hint"):
        delattr(slides, name)
    slides._draw_the_terms_gate(True)
    assert box.isEnabled() is False, "nothing was there to enable"
    assert slides._refuse_to_leave_the_terms() == terms_index


def test_a_pending_terms_signal_forgives_a_deleted_switch(slides):
    """A queued scroll signal can arrive while Qt deletes the terms page.

    The Python wrapper remains truthy after its C++ Toggle is gone, so the
    redraw must validate each child before touching it.
    """
    from shiboken6 import delete, isValid

    delete(slides._agree)
    assert isValid(slides._agree) is False
    slides._draw_the_terms_gate(True)
    assert isValid(slides._agree) is False


# ------------------------------------------------------------ the backdrop


def test_the_animation_row_opens_on_something(slides, monkeypatch,
                                              debug_log):
    """The backdrop question opens on what is already true: an unreadable
    preference, or one naming an animation an older spaCR offered, falls back
    to the application's own default -- and if that is not on offer either, to
    the first entry, because a negative index draws an empty combo on the row
    that is supposed to say what the backdrop is."""
    from spacr.qt.widgets import ambient

    _label, box = slides._animation_row()
    assert box.itemData(0, Qt.ToolTipRole)
    monkeypatch.setattr("spacr.qt.preferences.get_ambient_animation", _boom)
    monkeypatch.setattr("spacr.qt.widgets.ambient.animation_note", _boom)
    _label, box = slides._animation_row()
    assert box.currentData() == ambient.DEFAULT_THEME
    assert box.itemData(0, Qt.ToolTipRole) is None
    assert "the stored animation could not be read" in debug_log.text
    monkeypatch.setattr("spacr.qt.preferences.get_ambient_animation",
                        lambda: "supernova")
    _label, box = slides._animation_row()
    assert box.currentData() == ambient.DEFAULT_THEME
    monkeypatch.setattr("spacr.qt.widgets.ambient.ANIMATION_CHOICES",
                        ("aurora", "drift"))
    _label, box = slides._animation_row()
    assert (box.currentIndex(), box.currentData()) == (0, "aurora")
    assert f"no {ambient.DEFAULT_THEME} among the animations offered" in debug_log.text


def test_the_backdrop_is_stored_only_when_one_is_named(slides, monkeypatch):
    """``set_ambient_animation`` both records the choice and applies it, so
    calling it with nothing named would store an empty backdrop over the
    user's real one. An absent row writes nothing at all."""
    written = []
    monkeypatch.setattr("spacr.qt.preferences.set_ambient_animation",
                        written.append)
    box, slides._animation = slides._animation, None
    slides._apply_animation()
    assert written == [], "no row, nothing to store"
    slides._animation = box
    slides._apply_animation()
    assert written == [box.currentData()]


# ----------------------------------------------------------------- motion


def test_the_greeting_arrives_and_leaves_without_an_animator(slides,
                                                             monkeypatch,
                                                             debug_log):
    """The greeting is the only proof the language choice took. Without a
    palette it loses its colour and without an animation engine it simply
    appears and simply goes -- but it must still appear, the next slide must
    still arrive rather than staying at half opacity, and the leftover fade
    must be let go of even when the page stack has gone first."""
    slides._say_hello()
    assert slides._greeting.text() == mod.greeting_for(
        slides._editors["language"].currentData())
    monkeypatch.setattr("spacr.qt.theme.active_palette", _boom)
    monkeypatch.setattr(mod, "QPropertyAnimation", _boom)
    slides._show_the_greeting()
    assert slides._greeting.isHidden() is False
    assert slides._hello is None
    assert "no palette for the greeting" in debug_log.text
    slides._fade_the_greeting_away()
    assert slides._greeting.isHidden() is True
    assert slides._goodbye is None
    slides._show_slide(1, fade=True)
    assert slides.slide() == 1
    assert slides._fade is None
    assert "no cross-fade on this platform" in debug_log.text
    slides._editors.pop("language")
    slides._greeting.setText("kept")
    slides._say_hello()
    assert slides._greeting.text() == "kept", "no list, nothing to greet in"
    slides._fade = types.SimpleNamespace(stop=_boom)
    pages, slides._pages = slides._pages, object()
    slides._drop_the_fade()
    assert slides._fade is None, "a fade that will not stop is still dropped"
    slides._pages = pages


def test_the_gpu_note_is_placed_whatever_the_greeting_is_doing(slides):
    """The note used to be placed behind a check for the greeting, which does
    not exist until the language is confirmed -- so it sat where a QLabel is
    born, the top left corner, and read as stray text above the language list.
    A slide whose note has been taken away still moves."""
    card, greeting = slides.card, slides._greeting
    # TALL ENOUGH THAT THE BAND IS WHAT DECIDES. The note is placed at
    # GPU_NOTE_BAND unless it would hang off the bottom, in which case it
    # is deliberately lifted so a small window cannot hide the answer to
    # "can this machine run spaCR". On a card a few pixels high the clamp
    # wins and this assertion measures the clamp instead of the placement
    # it is about -- which is what happened when the note grew a per-task
    # capability list and became several lines taller.
    # TALL ENOUGH FOR THE NOTE THIS APP ACTUALLY HAS, measured rather than
    # guessed. 1000 was hard-coded here and it is not enough once the
    # application stylesheet is on: the themed note wraps taller, the band
    # plus its height overflows the floor, the clamp lifts it, and this
    # assertion measured the clamp -- the very thing the paragraph above
    # says it must not do.
    #
    # THE SHEET IS NOT A LEAK. `qt_theme_applied` is session-scoped and the
    # conftest deliberately never restores an EMPTY sheet, because doing so
    # would ratchet the whole run to an unstyled app. So a test that runs
    # after anything themed sees the theme, and that is the real condition;
    # running alone, with a 0-byte sheet, is the unusual one. Measured:
    # 0 bytes alone against 71,472 in company.
    #
    # The band needs (1 - GPU_NOTE_BAND) of the card to hold the note plus
    # the nav row, so size the card from the note instead of from a number.
    note = slides._gpu_note
    card.layout().removeWidget(note)
    slides._gpu_note_in_layout = False
    assert card.layout().indexOf(note) == -1
    note.show()
    assert not note.isHidden()
    tall = note.heightForWidth(max(note.width(), 1)) or note.sizeHint().height()
    needed = int((max(tall, 1) + 200) / (1.0 - GPU_NOTE_BAND)) + 1
    card.resize(max(card.width(), 600), max(1000, needed))
    card.layout().activate()
    slides._greeting = None
    slides._place_the_greeting()
    assert slides._gpu_note.y() == int(card.height() * GPU_NOTE_BAND)
    placed = slides._gpu_note.geometry()
    slides.card = None
    slides._place_the_greeting()
    assert slides._gpu_note.geometry() == placed, "no card, nothing to place"
    slides.card, slides._greeting = card, greeting
    del slides._gpu_note
    slides._show_slide(2)
    assert slides.slide() == 2
    assert mod._say(SLIDES[2][0]) in slides._title.text()


def test_the_language_and_the_look_are_applied_as_they_are_chosen(slides,
                                                                  monkeypatch,
                                                                  debug_log):
    """Everything after this reads the STORED language and the applied theme,
    because the only way to know a choice took is to see it. A store that
    refuses, a catalog walker that raises and a setter that fails all cost the
    preview alone -- but a language nobody named must not be written at all,
    or a stray empty list silently resets the profile."""
    from spacr.qt import setup_screen
    asked, applied = [], []

    def _refuse(code):
        asked.append(code)
        raise RuntimeError("the preference store is read-only")

    def _questions():
        out = []
        for q in setup_screen.__dict__["_real_questions"]():
            if q[0] == "theme":
                out.append(q[:3] + (applied.append,) + q[4:])
            elif q[0] == "colour_blind":
                out.append(q[:3] + (_boom,) + q[4:])
            else:
                out.append(q)
        return out

    monkeypatch.setattr("spacr.qt.preferences.set_language", _refuse)
    monkeypatch.setattr("spacr.qt.i18n.retranslate_widget_tree", _boom)
    slides._apply_language()
    assert asked == [slides._editors["language"].currentData()]
    assert "could not store the language" in debug_log.text
    assert "could not retranslate the setup screen" in debug_log.text
    slides._editors["language"] = QComboBox()
    slides._apply_language()
    assert len(asked) == 1, "an empty list names no language"
    slides._editors.pop("language")
    slides._apply_language()
    assert len(asked) == 1, "no list at all names no language"
    monkeypatch.setitem(setup_screen.__dict__, "_real_questions",
                        setup_screen.questions)
    monkeypatch.setattr(setup_screen, "questions", _questions)
    slides._apply_look("theme")
    assert applied == [slides._editors["theme"].currentData()]
    slides._apply_look("colour_blind")
    assert "could not apply colour_blind live" in debug_log.text
    slides._apply_look("no_such_question")
    assert applied == [slides.answers()["theme"]], "one setter ever ran"


def test_the_slides_advance_with_no_timer_to_wait_on(slides, monkeypatch,
                                                     debug_log):
    """The first Next holds the greeting so the language choice can be read.
    Without a timer the hold is what is lost, not the advance -- a setup screen
    that stops on slide one because a QTimer could not be made is a screen
    nobody can finish."""
    monkeypatch.setattr(mod, "QTimer", _boom)
    slides.showEvent(QShowEvent())
    assert "the terms gate could not be re-measured" in debug_log.text
    assert slides._advance_after_the_greeting() == 1, "no wait, it moves"
    assert slides.slide() == 1
    assert slides._next.isEnabled() is True
    assert slides._pending is None
    assert "no timer for the greeting pause" in debug_log.text


def test_the_last_next_finishes_the_setup(slides, monkeypatch, debug_log):
    """"Start spaCR" is the answer to the setup questions: it writes them and
    records that this version was answered, so the screen does not come back on
    every launch -- and it says in the log which answers the model refused
    rather than dropping them silently."""
    from spacr.qt import setup_screen
    recorded = {}

    def _apply(answers):
        recorded["answers"] = answers
        return ["theme was refused"]

    monkeypatch.setattr(setup_screen, "apply", _apply)
    monkeypatch.setattr(setup_screen, "mark_answered",
                        lambda version: recorded.setdefault("version",
                                                            version))
    last = len(SLIDES) - 1
    slides._show_slide(last)
    assert slides.next() == last
    assert recorded["answers"]["theme"] == \
        slides._editors["theme"].currentData()
    assert recorded["version"] == setup_screen.current_version()
    assert "some setup answers were refused" in debug_log.text


def test_the_decoration_forgives_a_card_that_is_already_gone(slides,
                                                             monkeypatch,
                                                             debug_log):
    """The rim aiming at the pointer and the rounded corners are decoration:
    a card taken down mid-gesture must swallow the aim rather than raise out
    of a mouse handler, which Qt turns into a crash, and a platform that
    refuses the mask must still lay the card over the whole window or the
    dialog draws as two stacked surfaces."""
    aimed = []
    card, slides.card = slides.card, types.SimpleNamespace(
        mapFrom=lambda _widget, point: point, flow_towards=aimed.append)
    event = QMouseEvent(QEvent.MouseMove, QPointF(40, 30), Qt.NoButton,
                        Qt.NoButton, Qt.NoModifier)
    slides.mouseMoveEvent(event)
    assert aimed == [QPoint(40, 30)]
    slides.card = object()
    slides.mouseMoveEvent(event)
    slides.card = card
    monkeypatch.setattr("spacr.qt.widgets.glass.round_the_corners", _boom)
    slides.resize(640, 480)
    slides.resizeEvent(QResizeEvent(QSize(640, 480), QSize(720, 560)))
    assert slides.card.size() == slides.size()
    assert "could not round the setup window" in debug_log.text


def test_the_animation_caption_is_catalogued_at_import(monkeypatch,
                                                       debug_log):
    """This module owns exactly one caption and the catalog check reads it at
    import. A catalog that cannot be written must not stop the module
    importing: every screen behind it would go too."""
    added = []
    monkeypatch.setattr("spacr.qt.terms.register_translations", _boom)
    monkeypatch.setattr("spacr.qt.i18n.add_translation",
                        lambda text, translations: added.append(text))
    mod._catalogue_this_screen()
    assert added == [mod.ANIMATION_LABEL]
    assert "the terms captions could not be catalogued" in debug_log.text
    monkeypatch.setattr("spacr.qt.i18n.add_translation", _boom)
    mod._catalogue_this_screen()
    assert "the animation caption could not be catalogued" in debug_log.text
