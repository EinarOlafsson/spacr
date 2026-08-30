"""The setup slides keep working when the parts around them do not.

INVARIANTS 10: the fades, the palette, the ambient row, the bundled fonts and
the provider registry are decoration or optional, and every one of them can be
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
from PySide6.QtGui import QDesktopServices, QMouseEvent, QResizeEvent, QShowEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QMessageBox  # noqa: E402

from spacr.qt.widgets import setup_slides as mod  # noqa: E402
from spacr.qt.widgets.setup_slides import (  # noqa: E402
    GPU_NOTE_BAND,
    SLIDES,
    SetupSlides,
    _held_at_the_top,
    _let_go_of,
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
def slides(app):
    return SetupSlides()


@pytest.fixture
def debug_log(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.qt.setup_slides")
    return caplog


@pytest.fixture
def empty_path(tmp_path, monkeypatch):
    """A PATH with no CLI on it at all."""
    bare = tmp_path / "nothing"
    bare.mkdir()
    monkeypatch.setenv("PATH", str(bare))
    return bare


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


def test_a_dying_dialog_lets_go_of_gh_without_raising(debug_log):
    """``gh auth login`` outlives this screen on purpose, so the detach runs
    while the dialog is being destroyed: a signal that was never connected and
    a process already reparented are both ordinary there, and raising on
    either is the "C++ object already deleted" crash on the way out."""
    seen = []
    _let_go_of(types.SimpleNamespace(
        finished=types.SimpleNamespace(
            disconnect=lambda: seen.append("disconnected")),
        setParent=lambda parent: seen.append(parent) or _boom()))
    assert seen == ["disconnected", None]
    assert "gh process could not be detached" in debug_log.text


def test_the_provider_caption_still_gets_its_cell(app, monkeypatch):
    """The AI caption is pinned to the top of a cell as tall as the marks it
    names; with no mark to measure it keeps its natural height rather than the
    row losing its label, which would read as a control that failed to
    build."""
    assert _held_at_the_top(QLabel("A")).findChild(QLabel).minimumHeight() > 0
    monkeypatch.setattr("spacr.qt.widgets.provider_marks.ProviderMark", _boom)
    bare = _held_at_the_top(QLabel("A"))
    assert bare.findChild(QLabel).minimumHeight() == 0


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


def test_a_question_that_removed_itself_leaves_no_empty_row(app, monkeypatch):
    """A question can withdraw -- the provider one does when no CLI is found
    -- and the slide it was on closes over the gap: an empty labelled row
    reads as a broken control rather than a question that does not apply."""
    from spacr.qt import setup_screen
    real = setup_screen.questions
    monkeypatch.setattr(
        setup_screen, "questions",
        lambda: [q for q in real() if q[0] != "colour_blind"])
    answers = SetupSlides().answers()
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


# ------------------------------------------------------------------ github


def test_the_status_survives_the_dialog_it_belongs_to(slides, monkeypatch):
    """``gh auth login`` finishes long after this screen may be gone and its
    ``finished`` lands on ``_refresh_github``, so the refresh has to notice
    its labels are deleted C++ objects: touching one is a hard crash."""
    from shiboken6 import delete
    assert slides._still_on_screen() is True
    slides._gh_action = "sentinel"
    delete(slides._gh_status)
    assert slides._still_on_screen() is False
    slides._refresh_github()
    assert slides._gh_action == "sentinel", "a gone dialog is not touched"
    monkeypatch.setitem(sys.modules, "shiboken6", None)
    assert slides._still_on_screen() is True, "no shiboken, assume alive"


def test_the_github_row_names_all_three_states(slides, monkeypatch, tmp_path,
                                               debug_log):
    """Signed in, no CLI, and CLI-but-signed-out need three different things
    from the user and this row is the only place that says which. The mark can
    be absent on the way in, and the sentence still has to be written."""
    del slides._gh_mark
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", lambda: "gh")
    slides._refresh_github()
    assert "GitHub CLI" in slides._gh_status.text()
    assert slides._gh_action == "login"
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv("PATH", str(bare))
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", _boom)
    slides._refresh_github()
    assert "not installed" in slides._gh_status.text()
    assert slides._gh_action == "install"
    assert "GitHub auth is not readable here" in debug_log.text
    gh = bare / "gh"
    gh.write_text("#!/bin/sh\nsleep 5\n")
    gh.chmod(0o755)
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", lambda: None)
    slides._refresh_github()
    assert slides._gh_status.text().startswith("not signed in")
    assert slides._gh_action == "login"


def test_the_logo_opens_the_install_page_when_there_is_no_cli(slides,
                                                              monkeypatch,
                                                              debug_log):
    """With no ``gh`` there is nothing to log in to, so the click installs one
    instead: a control that says "not installed" and then does nothing is the
    dead button this row was reported for."""
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url.toString())
                                     or True))
    slides._gh_action = "install"
    assert slides._on_github_mark() is True
    assert opened == [SetupSlides.GITHUB_CLI_PAGE]
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(_boom))
    assert slides._open_in_the_browser(SetupSlides.GITHUB_CLI_PAGE) is False
    assert "could not open" in debug_log.text


def test_a_sign_in_says_whether_it_began(slides, tmp_path, monkeypatch,
                                         debug_log):
    """A login with no word on screen is the "nothing happened" this button
    was reported for. A ``gh`` that cannot be launched -- missing, or refused
    outright by a locked-down machine -- sends the user to a terminal, and one
    that starts is held on the dialog, since a QProcess nobody holds is
    collected mid-login."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    slides._gh_action = "login"
    assert slides._sign_in_to_github() is False
    assert "would not start" in slides._gh_status.text()
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\nsleep 5\n")
    gh.chmod(0o755)
    assert slides._sign_in_to_github() is True
    assert "starting" in slides._gh_status.text().lower()
    slides._gh_process.kill()
    slides._gh_process.waitForFinished(2000)

    class _Refusing:
        MergedChannels = 0
        readyReadStandardOutput = finished = types.SimpleNamespace(
            connect=lambda *_a: None)
        start = _boom

        def __init__(self, *_a):
            pass

        def setProcessChannelMode(self, *_a):
            pass

    monkeypatch.setattr("PySide6.QtCore.QProcess", _Refusing)
    assert slides._sign_in_to_github() is False
    assert "gh auth login would not start" in debug_log.text.replace("`", "")


def test_the_one_time_code_is_shown_even_when_the_prompt_is_closed(slides,
                                                                   debug_log):
    """The code and the page to type it into are the whole of this flow.
    ``gh`` waits on an Enter no GUI can send, and if that write fails the user
    must still be looking at the code -- reading it off the screen is the only
    way the sign-in can finish."""
    chatty = types.SimpleNamespace(
        readAllStandardOutput=lambda: b"copy your code: ABCD-1234\n",
        write=_boom)
    slides._open_in_the_browser = lambda url: False
    assert slides._read_github_output(chatty) == "ABCD-1234"
    assert "ABCD-1234" in slides._gh_status.text()
    assert SetupSlides.GITHUB_DEVICE_PAGE in slides._gh_status.text()
    assert "could not answer the gh prompt" in debug_log.text
    assert slides._read_github_output(
        types.SimpleNamespace(readAllStandardOutput=_boom)) == ""


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
    _label, box = slides._animation_row()
    assert box.itemData(0, Qt.ToolTipRole)
    monkeypatch.setattr("spacr.qt.preferences.get_ambient_animation", _boom)
    monkeypatch.setattr("spacr.qt.widgets.ambient.animation_note", _boom)
    _label, box = slides._animation_row()
    assert box.currentData() == "blobs"
    assert box.itemData(0, Qt.ToolTipRole) is None
    assert "the stored animation could not be read" in debug_log.text
    monkeypatch.setattr("spacr.qt.preferences.get_ambient_animation",
                        lambda: "supernova")
    _label, box = slides._animation_row()
    assert box.currentData() == "blobs"
    monkeypatch.setattr("spacr.qt.widgets.ambient.ANIMATION_CHOICES",
                        ("aurora", "ripple"))
    _label, box = slides._animation_row()
    assert (box.currentIndex(), box.currentData()) == (0, "aurora")
    assert "no blobs among the animations offered" in debug_log.text


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


# ---------------------------------------------------------------- providers


def test_a_provider_is_found_under_either_name_or_falls_back_to_path(
        monkeypatch, empty_path, debug_log):
    """This screen calls OpenAI's provider ``gpt`` and the registry files it
    under ``codex``; a lookup that knows one spelling left that mark drawn as
    neither installed nor signed in, which is how it was reported. A registry
    that raises or cannot be imported falls back to PATH, so the mark still
    says something rather than carrying no state at all."""
    assert SetupSlides._provider_object("claude", "claude").name == "claude"
    assert SetupSlides._provider_object("gpt", "codex").name == "codex"
    assert SetupSlides._provider_object("", "") is None
    monkeypatch.setattr("spacr.qt.ai.providers.get_provider", _boom)
    assert SetupSlides._provider_object("claude", "claude") is None
    assert SetupSlides._provider_is_signed_in("claude", "claude") is False
    monkeypatch.setattr(
        "spacr.qt.ai.providers.get_provider",
        lambda name: types.SimpleNamespace(is_installed=_boom,
                                           is_configured=lambda: True))
    assert SetupSlides.provider_status("claude", "claude") == "not installed"
    assert "could not ask 'claude' what state it is in" in debug_log.text
    assert SetupSlides._provider_is_signed_in("claude", "claude") is True
    monkeypatch.setattr("spacr.qt.ai.providers.get_provider",
                        lambda name: None)
    assert SetupSlides._provider_is_signed_in("claude", "claude") is False
    monkeypatch.setitem(sys.modules, "spacr.qt.ai.providers", None)
    assert SetupSlides._provider_object("claude", "claude") is None


def test_a_login_needs_a_terminal_and_finds_one_per_platform(slides,
                                                             monkeypatch):
    """These CLI logins are conversations -- a code to copy, a key to paste --
    and started without a terminal they hang on a prompt nobody can see, which
    is how the GitHub button used to fail. Every platform gets its own way of
    opening one, and a machine with none says so instead of pretending."""
    launched = []

    monkeypatch.setattr("PySide6.QtCore.QProcess", types.SimpleNamespace(
        startDetached=lambda program, args: launched.append(
            (program, tuple(args))) or True))
    assert slides._run_in_a_terminal("   ") is False
    monkeypatch.setattr(sys, "platform", "darwin")
    assert slides._run_in_a_terminal("claude login") is True
    assert launched[-1][0] == "osascript"
    monkeypatch.setattr(sys, "platform", "win32")
    assert slides._run_in_a_terminal("claude login") is True
    assert launched[-1] == ("cmd", ("/c", "start", "claude", "login"))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert slides._run_in_a_terminal("claude login") is False, "none found"
    monkeypatch.setattr("shutil.which",
                        lambda name: "/usr/bin/xterm" if name == "xterm"
                        else None)
    assert slides._run_in_a_terminal("claude login") is True
    assert launched[-1] == ("xterm", ("-e", "claude", "login"))


def test_choosing_a_provider_it_cannot_set_up_says_what_to_do(slides,
                                                              monkeypatch):
    """Choosing a provider you have not set up is the moment to say what it
    needs -- "there is no popup no prompt for installing or any guidance" --
    and every button in that prompt has to do something. A provider the
    registry does not know has nothing to prompt about."""
    copied = []
    monkeypatch.setattr(QApplication, "clipboard",
                        staticmethod(lambda: types.SimpleNamespace(
                            setText=copied.append)))
    picked = {}

    def _press(text):
        def _exec(box):
            picked["button"] = next(
                (b for b in box.buttons() if b.text() == text), None)
            return 0
        monkeypatch.setattr(QMessageBox, "exec", _exec)
        monkeypatch.setattr(QMessageBox, "clickedButton",
                            lambda box: picked.get("button"))

    provider = types.SimpleNamespace(
        label="Fake", cli_name="fake", install_hint="pip install fake",
        login_command="fake login", is_installed=lambda: False,
        is_logged_in=lambda: False)
    _press("Copy the command")
    assert slides._prompt_to_set_up(provider, "gpt") == \
        "Copied: pip install fake"
    assert copied == ["pip install fake"]
    _press("Later")
    assert slides._prompt_to_set_up(provider, "gpt") == \
        "Fake is not set up yet."
    monkeypatch.setattr(QApplication, "clipboard", staticmethod(_boom))
    _press("Copy the command")
    assert slides._prompt_to_set_up(provider, "gpt") == \
        "Copied: pip install fake", "the command is still on screen to read"
    assert copied == ["pip install fake"], "a refusing clipboard copies none"
    monkeypatch.setattr(SetupSlides, "_provider_object",
                        staticmethod(lambda code, command="": None))
    assert slides._start_provider_login("claude") == ""


def test_the_marks_are_recoloured_from_the_state_they_are_in(slides,
                                                             monkeypatch):
    """The colour of a mark IS its sign-in state, so choosing one has to
    re-ask: an installed-but-signed-out provider looked exactly like one that
    was ready to answer, which is why the three states exist at all."""
    mark = types.SimpleNamespace(status="", available=False, chosen=None,
                                 tip="", repaints=0)
    mark.set_chosen = lambda on: setattr(mark, "chosen", on)
    mark.setToolTip = lambda text: setattr(mark, "tip", text)
    mark.update = lambda: setattr(mark, "repaints", mark.repaints + 1)
    holder = types.SimpleNamespace(_chosen="", _buttons={"claude": mark},
                                   _note=None)
    monkeypatch.setattr(SetupSlides, "_start_provider_login",
                        lambda self, code: "a note nobody can show")
    slides._choose_provider(holder, "claude")
    assert holder._chosen == "claude"
    assert mark.chosen is True
    assert mark.repaints == 1
    assert mark.status in ("ready", "signed out", "not installed")
    assert "Claude" in mark.tip


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
    """"Start spaCR" is the answer to the seven questions: it writes them and
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
