"""The setup screen asks which backdrop to run, under the theme it belongs to.

"in the startup, under theme should be annimation, degault to blobs."

The backdrop is part of what spaCR looks like, so it is decided where the
look is decided -- immediately under the theme, on the same slide, rather
than found later in Preferences by somebody the motion is bothering.

Three things are load-bearing:

* EVERY CHOICE, ``None`` included, because "turn it off" is one of the
  answers and a question that cannot be answered "no" is not a question;
* THE DEFAULT IS THE APPLICATION'S OWN. Blobs is what
  ``preferences.get_ambient_animation`` falls back to, so the slide opens
  on what is already true instead of offering a second opinion about it;
* IT IS WRITTEN THROUGH ONE SEAM. ``set_ambient_animation`` both stores the
  choice and turns the backdrop on or off, so a profile that picks None
  gets silence rather than a stored None with the animation still running.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QLabel                    # noqa: E402

from spacr.qt.widgets.setup_slides import (ANIMATION_LABEL,        # noqa: E402
                                           SLIDES, SetupSlides)

pytestmark = pytest.mark.qt

THEME_INDEX = [title for title, _b, _k in SLIDES].index("Theme")


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A settings store of this test's own: it writes a real preference."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences

    importlib.reload(preferences)
    yield
    importlib.reload(preferences)


@pytest.fixture
def prefs():
    from spacr.qt import preferences

    return preferences


@pytest.fixture
def ambient():
    from spacr.qt.widgets import ambient as module

    return module


@pytest.fixture
def slides(qtbot):
    made = SetupSlides()
    qtbot.addWidget(made)
    return made


def _row_captions(made):
    """The label at the head of every row on the theme slide, in order."""
    page = made._pages.widget(THEME_INDEX)
    form = page.layout()
    captions = []
    for index in range(form.count()):
        row = form.itemAt(index).layout()
        if row is None:
            continue
        first = row.itemAt(0).widget() if row.count() else None
        if isinstance(first, QLabel):
            captions.append(first.text())
    return captions


class TestItIsAskedWhereItWasAskedFor:

    def test_the_theme_slide_carries_it(self, slides):
        assert ANIMATION_LABEL in _row_captions(slides)

    def test_it_sits_immediately_under_the_theme(self, slides):
        """"under theme should be annimation" -- under THAT question, not
        somewhere else on the slide."""
        captions = _row_captions(slides)

        assert captions.index(ANIMATION_LABEL) == captions.index("Theme") + 1

    def test_it_is_a_chooser(self, slides):
        assert isinstance(slides._animation, QComboBox)

    def test_the_caption_is_in_the_catalog(self):
        """Every word on this screen is translated; this one is asked by the
        slide itself, so the slide catalogues it."""
        from spacr.qt import i18n

        assert i18n.has_translation(ANIMATION_LABEL)
        assert i18n.tr(ANIMATION_LABEL, "ko") != ANIMATION_LABEL


class TestItOffersEveryAnswer:

    def test_every_animation_the_application_has(self, slides, ambient):
        offered = [slides._animation.itemData(i)
                   for i in range(slides._animation.count())]

        assert offered == list(ambient.ANIMATION_CHOICES)

    def test_none_is_one_of_them(self, slides, ambient):
        """"a question that cannot be answered no" -- the reader who finds
        the motion distracting says so here, not in Preferences later."""
        offered = [slides._animation.itemData(i)
                   for i in range(slides._animation.count())]

        assert ambient.NO_ANIMATION in offered

    def test_they_are_named_rather_than_keyed(self, slides, ambient):
        """`blobs` is a key; "Blobs" is what the reader is choosing."""
        for index in range(slides._animation.count()):
            key = slides._animation.itemData(index)
            assert slides._animation.itemText(index) == \
                ambient.animation_label(key)


class TestItDefaultsToBlobs:

    def test_a_fresh_profile_opens_on_blobs(self, slides, ambient):
        assert slides.animation_choice() == "blobs"
        assert slides.animation_choice() == ambient.DEFAULT_THEME

    def test_that_is_what_the_application_already_answers(self, slides,
                                                          prefs):
        """The slide shows the default spaCR has, not a second opinion."""
        assert slides.animation_choice() == prefs.get_ambient_animation()

    def test_a_profile_that_chose_something_opens_on_that(self, prefs,
                                                          qtbot):
        prefs.set_ambient_animation("ripple")
        made = SetupSlides()
        qtbot.addWidget(made)

        assert made.animation_choice() == "ripple"

    def test_a_profile_that_turned_it_off_opens_on_none(self, prefs, ambient,
                                                        qtbot):
        prefs.set_ambient_animation(ambient.NO_ANIMATION)
        made = SetupSlides()
        qtbot.addWidget(made)

        assert made.animation_choice() == ambient.NO_ANIMATION


class TestItIsWrittenThroughTheOneSeam:

    def test_choosing_one_stores_it(self, slides, prefs, qapp):
        box = slides._animation
        box.setCurrentIndex(box.findData("bokeh"))
        qapp.processEvents()

        assert prefs.get_ambient_animation() == "bokeh"

    def test_choosing_one_turns_the_backdrop_on(self, slides, prefs, ambient,
                                                qapp):
        """`set_ambient_animation` is the seam that does both. Writing the
        theme key alone would leave a profile that had switched the backdrop
        off with a stored animation and no animation."""
        box = slides._animation
        box.setCurrentIndex(box.findData(ambient.NO_ANIMATION))
        qapp.processEvents()
        assert prefs.get_ambient_enabled() is False

        box.setCurrentIndex(box.findData("aurora"))
        qapp.processEvents()

        assert prefs.get_ambient_enabled() is True

    def test_choosing_none_turns_it_off(self, slides, prefs, ambient, qapp):
        box = slides._animation
        box.setCurrentIndex(box.findData(ambient.NO_ANIMATION))
        qapp.processEvents()

        assert prefs.get_ambient_animation() == ambient.NO_ANIMATION
        assert prefs.get_ambient_enabled() is False

    def test_a_store_that_refuses_the_write_does_not_stop_setup(
            self, slides, monkeypatch, qapp):
        """A backdrop is decoration; setup is not abandoned over it."""
        import spacr.qt.preferences as preferences

        def boom(_name):
            raise OSError("the settings file is read-only")

        monkeypatch.setattr(preferences, "set_ambient_animation", boom)
        box = slides._animation
        box.setCurrentIndex(box.findData("drift"))
        qapp.processEvents()

        assert slides.animation_choice() == "drift"


class TestTheSlideSurvivesWithoutAnAmbientModule:

    def test_no_ambient_module_leaves_the_other_questions_asked(
            self, monkeypatch, qtbot):
        """INVARIANTS 10: a decorative question that cannot be built is a
        question that is not asked, not a setup screen that will not open."""
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.endswith("ambient") or name == "ambient":
                raise ImportError("no ambient module here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        made = SetupSlides()
        monkeypatch.undo()
        qtbot.addWidget(made)

        assert ANIMATION_LABEL not in _row_captions(made)
        assert "Theme" in _row_captions(made)
        assert made.animation_choice() == ""
