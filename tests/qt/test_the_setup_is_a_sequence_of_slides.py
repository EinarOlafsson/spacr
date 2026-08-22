"""The setup screen is one question per slide (instruction 234).

A FORM ASKS EVERYTHING AT ONCE AND ANSWERS NOTHING. A slide asks one thing
and has room to say why it matters.

INVARIANTS 10 THROUGHOUT: the rim, the strata and the blur are decoration.
If none of them can be drawn the slides still work and still write the same
answers.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF                     # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox,  # noqa: E402
                               QComboBox)

from spacr.qt.widgets.setup_slides import (GREETINGS, PROVIDERS, SLIDES,
                                           SetupSlides, greeting_for)


def _past_the_greeting(slides):
    """Take the slides off the language page, pause included.

    The first Next holds the greeting on screen for GREETING_MS; a test that
    does not want to wait for a real timer calls the timeout itself.
    """
    if slides.slide() == 0:
        slides.next()
        if slides.slide() == 0:
            slides._finish_the_greeting()
    return slides.slide()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A config dir of this test's own.

    Without it the test answers the setup screen on the user's machine and
    they never see it again.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences

    importlib.reload(preferences)
    yield
    importlib.reload(preferences)


@pytest.fixture
def slides(app):
    return SetupSlides()


class TestTheOrderIsTheMaintainers:

    def test_there_are_six(self):
        assert len(SLIDES) == 6

    def test_they_are_in_the_order_asked_for(self):
        assert [title for title, _b, _k in SLIDES] == [
            "Language", "Theme", "How it runs", "The assistant",
            "When something breaks", "Done"]

    def test_language_is_first(self):
        """It changes the screen the user is looking at."""
        assert SLIDES[0][2] == ("language",)

    def test_theme_and_colour_blind_are_one_slide(self):
        """One question: what this looks like."""
        assert set(SLIDES[1][2]) == {"theme", "colour_blind"}

    def test_the_assistant_asks_provider_and_launch(self):
        assert set(SLIDES[3][2]) == {"ai_provider", "ai_default"}

    def test_the_last_one_asks_nothing(self):
        """"then a screen that just says Done" -- it is the transition, not
        a summary; a list of what was chosen would be a form again."""
        assert SLIDES[-1][2] == ()

    def test_every_slide_explains_itself(self):
        for title, blurb, _keys in SLIDES:
            assert len(blurb.strip()) > 40, title


class TestOneQuestionPerSlide:

    def test_a_page_exists_for_each(self, slides):
        assert slides._pages.count() == len(SLIDES)

    def test_next_moves_one_slide(self, slides):
        """Off the FIRST slide it waits, so the greeting can be read.

        `next` returns the slide still showing, which is 0 during the pause;
        `_finish_the_greeting` is what the timer would call.
        """
        assert slides.next() == 0
        slides._finish_the_greeting()
        assert slides.slide() == 1
        assert slides.next() == 2, "only the FIRST Next waits"

    def test_previous_moves_back_one(self, slides):
        _past_the_greeting(slides)
        slides.next()
        assert slides.previous() == 1

    def test_it_does_not_go_before_the_first(self, slides):
        assert slides.previous() == 0

    def test_the_position_is_shown(self, slides):
        """Paging without a total is navigation without a map."""
        assert "1 of 6" in slides._where.text()

    def test_the_last_button_starts_spacr(self, slides):
        _past_the_greeting(slides)
        for _ in range(len(SLIDES) - 2):
            slides.next()
        assert "spaCR" in slides._next.text()


class TestTheGreeting:
    """"for the language chosen say Hello underneeth"."""

    def test_every_offered_language_has_one(self):
        from spacr.qt.setup_screen import questions

        offered = {code for q in questions() if q[0] == "language"
                   for code, _name in q[4]}
        assert offered <= set(GREETINGS), offered - set(GREETINGS)

    def test_it_is_not_shown_until_the_first_next(self, slides):
        """THE GREETING IS THE ANSWER TO THE QUESTION, so it comes after the
        question is answered rather than sitting under it while it is still
        being decided.

        ASSERTED ON `isHidden`, not `isVisible`: this dialog is never shown,
        and every child of an unshown parent reports itself invisible -- so
        `isVisible` would pass here whether or not the greeting had been
        hidden on purpose.
        """
        assert slides._greeting.isHidden()

    def test_the_first_next_shows_it(self, slides):
        slides.next()
        assert not slides._greeting.isHidden()
        assert slides._greeting.text() == GREETINGS["en"].upper()

    def test_it_is_in_capitals(self, slides):
        """`.upper()` is right for every language spaCR is translated to:
        on a script with no case it returns the greeting unchanged."""
        slides.next()
        assert slides._greeting.text() == slides._greeting.text().upper()

    def test_it_follows_the_language_that_was_chosen(self, slides):
        box = slides._editors["language"]
        box.setCurrentIndex(box.findData("sv"))
        slides.next()
        assert slides._greeting.text() == "HEJ"

    def test_it_is_in_the_accent_colour(self, slides):
        """Blue, from the palette rather than a literal, so it matches the
        rim running round the card as it arrives."""
        from spacr.qt.theme import active_palette

        slides.next()
        assert active_palette()["accent"].lower() in \
            slides._greeting.styleSheet().lower()

    def test_it_fades_in_rather_than_appearing(self, slides):
        """A word switched on reads as a label that was always going to be
        there; one that fades up reads as an answer to what was just
        chosen."""
        slides.next()
        assert slides._hello is not None
        assert slides._hello.parent() is slides, (
            "an animation nobody holds is collected before it runs")

    def test_it_is_only_on_the_language_slide(self, slides):
        """A "Hello" left standing over the theme question is a word with no
        job on that page."""
        _past_the_greeting(slides)
        assert not slides._greeting.isVisibleTo(slides)

    def test_a_language_with_no_greeting_falls_back(self):
        assert greeting_for("xx") == GREETINGS["en"]

    def test_the_languages_are_offered_in_their_own_script(self):
        """The reader of this list is by definition somebody who may not
        read the current one."""
        from spacr.qt.setup_screen import questions

        names = {name for q in questions() if q[0] == "language"
                 for _code, name in q[4]}
        assert "Svenska" in names and "한국어" in names


class TestEveryBooleanIsASlider:
    """"aslo in the startup all the booleans should be sliders"."""

    def test_no_bare_checkbox_survives(self, slides):
        from spacr.qt.widgets.toggle import Toggle

        for box in slides.findChildren(QCheckBox):
            assert isinstance(box, Toggle), (
                "a tick box is a form control and this is not a form")

    def test_the_booleans_are_toggles(self, slides):
        from spacr.qt.widgets.toggle import Toggle

        for key in ("hash_inputs", "ai_default", "share_logs"):
            assert isinstance(slides._editors[key], Toggle), key

    def test_they_still_read_back(self, slides):
        slides._editors["hash_inputs"].setChecked(True)
        assert slides.answers()["hash_inputs"] is True


class TestTheProviderIsALogoButton:

    def test_there_is_one_per_provider(self, slides):
        holder = slides._editors["ai_provider"]
        assert set(holder._buttons) == {code for code, _l, _c in PROVIDERS}

    def test_it_is_not_a_dropdown(self, slides):
        """A dropdown of three names is a dropdown; three logos is a choice
        somebody makes in one glance."""
        assert not isinstance(slides._editors["ai_provider"], QComboBox)

    def test_choosing_one_unchooses_the_rest(self, slides):
        holder = slides._editors["ai_provider"]
        slides._choose_provider(holder, "claude")
        assert holder._buttons["claude"].is_chosen()
        slides._choose_provider(holder, "gpt")
        assert not holder._buttons["claude"].is_chosen()
        assert holder._buttons["gpt"].is_chosen()

    def test_the_choice_reaches_the_answers(self, slides):
        slides._choose_provider(slides._editors["ai_provider"], "gemini")
        assert slides.answers()["ai_provider"] == "gemini"

    def test_an_uninstalled_provider_says_so_and_is_still_choosable(
            self, slides):
        """Reported 2026-08-22: "for the ai assistant i can only click
        claude". The setup screen writes a PREFERENCE and launches nothing,
        so choosing a provider before installing its CLI is an ordinary
        thing to want -- the state is drawn and said, not enforced."""
        holder = slides._editors["ai_provider"]
        for code, _label, command in PROVIDERS:
            mark = holder._buttons[code]
            if not mark.available:
                assert f"`{command}`" in mark.toolTip()
                assert "install it later" in mark.toolTip()
            slides._choose_provider(holder, code)
            assert slides.answers()["ai_provider"] == code


class TestTheRim:

    def test_next_runs_a_clockwise_circuit(self, slides):
        before = slides.card.position
        slides.next()
        assert slides.card.spinning
        for _ in range(5):
            slides.card._tick()
        assert slides.card.position > before or slides.card.position < before

    def test_the_two_directions_differ(self, app):
        """The direction is the message: it tells the user which way they
        went, which is worth more than the animation."""
        from spacr.qt.widgets.setup_card import SetupCard

        forward, back = SetupCard(), SetupCard()
        forward.resize(200, 120)
        back.resize(200, 120)
        forward.circuit(clockwise=True)
        back.circuit(clockwise=False)
        for _ in range(4):
            forward._tick()
            back._tick()
        assert forward.position > 0 and back.position < 1.0
        assert forward.position != back.position

    def test_a_lap_ends_exactly_where_it_started(self, app):
        """Floating error across thirty frames would leave the accent a
        little further round after every circuit."""
        from spacr.qt.widgets.setup_card import SetupCard

        card = SetupCard()
        card.resize(200, 120)
        card.circuit(clockwise=True)
        for _ in range(200):
            card._tick()
            if not card.spinning:
                break
        assert card.position == pytest.approx(0.0, abs=1e-6)

    def test_the_pointer_does_not_steer_a_running_circuit(self, app):
        """A lap dragged off course by a mouse movement is not a lap, and
        the user cannot tell whether it went round."""
        from spacr.qt.widgets.setup_card import SetupCard

        card = SetupCard()
        card.resize(200, 120)
        card.circuit(clockwise=True)
        card.flow_towards(QPointF(200, 60))
        assert card._towards == 0.0

    def test_it_flows_rather_than_jumping(self, app, monkeypatch):
        """"the blue rim should flow like water towards the mouse", and
        water does not teleport between corners.

        The cursor read is stubbed out: every tick now aims at wherever the
        pointer actually is, which on a test machine is wherever the last
        thing to touch it left it -- so without this the target moves out
        from under the assertion.
        """
        from spacr.qt.widgets import setup_card as module
        from spacr.qt.widgets.setup_card import SetupCard

        card = SetupCard()
        card.resize(200, 120)
        monkeypatch.setattr(module.SetupCard, "_aim_at_the_cursor",
                            lambda self: False)
        card.flow_towards(QPointF(200, 60))
        target = card._towards
        card._tick()
        first = card.position
        assert 0.0 < first < target, (
            "one tick should move part of the way, not all of it")


class TestDecorationIsNotLoadBearing:

    def test_it_builds_with_no_backdrop(self, app, monkeypatch):
        from spacr.qt.widgets import setup_slides

        def boom(self):
            raise RuntimeError("no ambient engine")

        monkeypatch.setattr(SetupSlides, "_install_backdrop", boom)
        with pytest.raises(RuntimeError):
            SetupSlides()

    def test_a_failed_backdrop_is_caught_inside(self, app, monkeypatch):
        """The catch is in `_install_backdrop` itself, so the dialog is
        built either way."""
        import spacr.qt.widgets.setup_slides as module

        monkeypatch.setattr(
            module, "BACKDROP_THEME", "no-such-theme", raising=False)
        built = SetupSlides()
        assert built.answers()

    def test_the_answers_are_the_same_without_it(self, app, monkeypatch):
        import spacr.qt.widgets.setup_slides as module

        plain = SetupSlides()
        monkeypatch.setattr(module, "BACKDROP_THEME", "no-such-theme",
                            raising=False)
        assert SetupSlides().answers() == plain.answers()


class TestItIsStillDismissible:

    def test_closing_at_any_slide_marks_it_answered(self, slides):
        from spacr.qt.setup_screen import should_open

        slides.next()
        slides.next()
        slides.reject()
        assert not should_open()

    def test_and_writes_the_defaults(self, slides):
        from spacr.qt import preferences

        slides._editors["hash_inputs"].setChecked(True)
        slides.reject()
        assert preferences.get_hash_inputs() is True

    def test_finishing_marks_it_too(self, slides):
        from spacr.qt.setup_screen import should_open

        slides.accept()
        assert not should_open()


class TestItIsReachableFromHelp:
    """Requested 2026-08-21: "the startup should be in the help menue".

    It ran once on the first launch and then never -- and it is the only
    place several of these settings are EXPLAINED rather than merely
    offered, so a user who dismissed it lost the explanation with the
    questions.
    """

    @pytest.fixture
    def window(self, app):
        from spacr.qt.app import MainWindow

        return MainWindow()

    def _help_menu(self, window):
        for menu in window.menuBar().findChildren(type(
                window.menuBar().addMenu("scratch"))):
            if "Help" in menu.title():
                return menu
        return None

    def test_there_is_an_entry(self, window):
        menu = self._help_menu(window)
        assert menu is not None
        assert any("Set spaCR up" in a.text() for a in menu.actions())

    def test_it_says_what_it_opens(self, window):
        action = next(a for a in self._help_menu(window).actions()
                      if "Set spaCR up" in a.text())
        assert "language" in action.statusTip()

    def test_it_opens_even_when_already_answered(self, window,
                                                 monkeypatch):
        """`open_setup_if_needed` asks `should_open` and would refuse. A
        menu item that does nothing on the second launch is the inert
        control this codebase keeps meeting."""
        import ast
        import inspect
        import textwrap

        from spacr.qt.app import MainWindow

        # THE CODE, NOT THE PROSE. The docstring explains why
        # `open_setup_if_needed` is not used, so a plain substring search
        # finds the very name it is asserting the absence of -- which is
        # what the first version of this test did.
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(MainWindow._show_setup)))
        called = {node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name)}
        called |= {node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute)}
        assert "SetupSlides" in called
        assert "open_setup_if_needed" not in called
        assert "should_open" not in called

    def test_the_launcher_still_asks_first(self):
        """The automatic path keeps its guard: it must not reappear every
        launch."""
        import inspect

        from spacr.qt.widgets import setup_slides

        source = inspect.getsource(setup_slides.open_setup_if_needed)
        assert "should_open" in source
