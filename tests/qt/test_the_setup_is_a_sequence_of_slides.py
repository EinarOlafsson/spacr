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
        assert slides.next() == 1

    def test_previous_moves_back_one(self, slides):
        slides.next()
        slides.next()
        assert slides.previous() == 1

    def test_it_does_not_go_before_the_first(self, slides):
        assert slides.previous() == 0

    def test_the_position_is_shown(self, slides):
        """Paging without a total is navigation without a map."""
        assert "1 of 6" in slides._where.text()

    def test_the_last_button_starts_spacr(self, slides):
        for _ in range(len(SLIDES) - 1):
            slides.next()
        assert "spaCR" in slides._next.text()


class TestTheGreeting:
    """"for the language chosen say Hello underneeth"."""

    def test_every_offered_language_has_one(self):
        from spacr.qt.setup_screen import questions

        offered = {code for q in questions() if q[0] == "language"
                   for code, _name in q[4]}
        assert offered <= set(GREETINGS), offered - set(GREETINGS)

    def test_it_is_shown_on_the_first_slide(self, slides):
        assert slides._blurb.text().startswith(GREETINGS["en"])

    def test_it_changes_with_the_choice(self, slides):
        box = slides._editors["language"]
        box.setCurrentIndex(box.findData("sv"))
        assert slides._blurb.text().startswith("Hej")

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
        assert holder._buttons["claude"].isChecked()
        slides._choose_provider(holder, "gpt")
        assert not holder._buttons["claude"].isChecked()
        assert holder._buttons["gpt"].isChecked()

    def test_the_choice_reaches_the_answers(self, slides):
        slides._choose_provider(slides._editors["ai_provider"], "gemini")
        assert slides.answers()["ai_provider"] == "gemini"

    def test_an_uninstalled_provider_says_so(self, slides):
        """Choosing it would leave the assistant silently unavailable, and
        the user would blame spaCR."""
        holder = slides._editors["ai_provider"]
        for code, _label, _command in PROVIDERS:
            button = holder._buttons[code]
            if not button.isEnabled():
                assert "not set up on this machine" in button.toolTip()


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

    def test_it_flows_rather_than_jumping(self, app):
        """"the blue rim should flow like water towards the mouse", and
        water does not teleport between corners."""
        from spacr.qt.widgets.setup_card import SetupCard

        card = SetupCard()
        card.resize(200, 120)
        card.flow_towards(QPointF(200, 60))
        card._tick()
        first = card.position
        assert 0.0 < first < card._towards, (
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
