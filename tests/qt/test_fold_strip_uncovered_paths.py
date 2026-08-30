"""What a fold button says when the two lookups behind it are unavailable.

A :class:`~spacr.qt.widgets.fold_strip.FoldButton` is built while a host
screen is being constructed, and both of the things it asks for at that
moment can refuse to answer:

* :mod:`spacr.qt.app` imports screens, so a screen importing the fold strip
  back closes the circle and the registry lookup raises rather than returns;
* :func:`spacr.qt.iconset.app_icon` renders artwork, and a broken or missing
  SVG raises out of the renderer instead of returning a null icon.

Neither may reach the host: a masthead that throws while it is being laid
out takes the whole screen with it. The same rule covers the rest of the
file — a host whose fold list is not a list, a fold nobody has wired a
callback to yet, and a settings section that will not wear the fold's icon.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets import fold_strip                          # noqa: E402
from spacr.qt.widgets.fold_strip import FoldButton, _describe    # noqa: E402

#: A module that is still a tile: the registry knows its display name, its
#: sentence and that it is alpha, so all three answers change when the
#: registry cannot be reached.
REGISTERED_KEY = "classify_merged"


def _refuse_the_app_module(monkeypatch):
    """Make ``from .. import app`` raise inside the fold strip only."""
    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        caller = (globals or {}).get("__name__")
        if (level == 2 and "app" in (fromlist or ())
                and caller == fold_strip.__name__):
            raise ImportError("spacr.qt.app is partially initialised")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)


def test_the_registry_answers_a_folded_button_with_name_sentence_and_stage():
    """The baseline the broken-import case is measured against."""
    name, description, stage = _describe(REGISTERED_KEY)

    assert name == "Classify"
    assert description
    assert stage == "alpha"


def test_a_fold_button_still_describes_itself_when_the_app_module_will_not_import(
        qtbot, monkeypatch):
    """A circular import degrades the button, it does not raise through it.

    Without the registry there is no display name, no sentence and no
    maturity, so the button falls back to the key title-cased and to the
    colour of finished code. That is worse copy than the registry gives,
    and it is still a button on a screen that finished building.
    """
    _refuse_the_app_module(monkeypatch)

    name, description, stage = _describe(REGISTERED_KEY)

    assert name == "Classify Merged"
    assert description == ""
    assert stage == "stable"

    button = FoldButton(REGISTERED_KEY)
    qtbot.addWidget(button)

    assert button.property("stage") == "stable"
    assert button.accessibleName() == "Classify Merged"
    assert button.toolTip() == "Classify Merged"


def test_a_fold_button_falls_back_to_its_initial_when_the_icon_renderer_raises(
        qtbot, monkeypatch):
    """Artwork that throws leaves a named button, not an unbuilt screen.

    The button carries no label of its own, so an exception swallowed into
    "no icon" has to leave the initial behind it; otherwise the fold is an
    empty square that cannot be told from its neighbour.
    """
    def exploding(key):
        raise RuntimeError(f"no renderer for {key}")

    monkeypatch.setattr(fold_strip.iconset, "app_icon", exploding)

    button = FoldButton("timelapse")
    qtbot.addWidget(button)

    assert button.icon().isNull()
    assert button.text() == "T"
    assert button.toolTip()


def test_a_strip_of_folds_survives_an_icon_renderer_that_throws(qtbot,
                                                               monkeypatch):
    """Every button in the strip is built, and each still calls back."""
    def unreadable(key):
        raise OSError(f"cannot read the artwork for {key}")

    monkeypatch.setattr(fold_strip.iconset, "app_icon", unreadable)
    pressed = []

    strip = fold_strip.FoldStrip(
        [("timelapse", lambda: pressed.append("timelapse")),
         ("motility", lambda: pressed.append("motility"))])
    qtbot.addWidget(strip)

    assert strip.keys() == ["timelapse", "motility"]
    assert [b.text() for b in strip.buttons] == ["T", "M"]
    strip.button_for("motility").click()
    assert pressed == ["motility"]


def test_a_switch_shaped_fold_hands_its_callback_the_new_state(qtbot):
    """One callback answers both directions, so it is told which one."""
    states = []

    strip = fold_strip.FoldStrip([("timelapse", states.append, True)])
    qtbot.addWidget(strip)
    button = strip.button_for("timelapse")

    assert button.isCheckable()
    button.click()
    button.click()

    assert states == [True, False]
    assert button.isChecked() is False


def test_a_fold_with_nothing_wired_to_it_is_still_drawn_and_still_inert(qtbot):
    """A host may list a fold before it has somewhere to send it.

    The button belongs to the masthead's layout either way; dropping it
    would renumber the strip and move every other icon.
    """
    strip = fold_strip.FoldStrip([("timelapse", None),
                                  ("motility", lambda: None)])
    qtbot.addWidget(strip)

    assert strip.keys() == ["timelapse", "motility"]
    unwired = strip.button_for("timelapse")
    assert unwired is not None
    unwired.click()
    assert unwired.isChecked() is False


def test_a_host_whose_fold_list_is_not_a_list_is_skipped_not_fatal(monkeypatch):
    """One host's typo costs its own folds, not the whole inventory.

    ``folded_modules`` is walked while a screen is being built, so a
    ``FOLDED_APPS`` that cannot be iterated has to cost that host its rows
    rather than take down the screen that asked what is folded into it.
    """
    import importlib

    intact = fold_strip.folded_modules()
    assert intact, "no folded modules to speak of; the fixture is stale"
    broken = next(host for _n, _d, _s, host in intact.values())
    module = importlib.import_module(broken)
    attribute = "FOLDED_APPS" if hasattr(module, "FOLDED_APPS") else "FOLD_ORDER"
    monkeypatch.setattr(module, attribute, 7)

    surviving = fold_strip.folded_modules()

    assert surviving
    assert all(host != broken for _n, _d, _s, host in surviving.values())
    assert set(surviving) == {key for key, entry in intact.items()
                              if entry[3] != broken}


class _Section:
    """A settings section that reports whether it took the fold's mark."""

    def __init__(self, title, accepts):
        self._title = title
        self._accepts = accepts
        self.asked = []

    def set_source_app(self, key, name):
        self.asked.append((key, name))
        return self._accepts

    def property(self, _name):
        return self._title

    def title(self):
        return self._title.upper()


def test_a_section_that_refuses_the_mark_is_not_reported_as_marked():
    """The return value is the record of what actually carries the icon.

    A caller uses it to say which categories arrived from somewhere else, so
    a section that took no mark must not appear in it — otherwise the run
    log claims artwork nobody can see.
    """
    took = _Section("Timelapse", accepts=True)
    refused = _Section("Tracking", accepts=False)

    marked = fold_strip.mark_folded_sections("timelapse", (took, refused))

    assert marked == ("Timelapse",)
    assert took.asked and refused.asked
    assert took.asked[0][0] == "timelapse"


def test_a_section_whose_marking_raises_is_skipped_and_the_rest_still_marked():
    """One bad section does not cost the others their icon."""
    class _Exploding(_Section):
        def set_source_app(self, key, name):
            raise RuntimeError("no artwork loaded")

    marked = fold_strip.mark_folded_sections(
        "timelapse", (_Exploding("Broken", accepts=True),
                      _Section("Timelapse", accepts=True)))

    assert marked == ("Timelapse",)


def test_a_widget_with_no_mark_to_set_is_passed_over_entirely():
    """Not every section can wear one; those are skipped, not errors."""
    plain = object()
    took = _Section("Timelapse", accepts=True)

    assert fold_strip.mark_folded_sections("timelapse", (plain, took)) == (
        "Timelapse",)
