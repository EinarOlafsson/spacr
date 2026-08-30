"""Registering the bundled Open Sans faces, including every way it declines.

``use_open_sans_for_figures`` runs before every figure is styled, and its whole
design is about NOT doing expensive work: it returns early once it has run, it
asks matplotlib what fonts it already has before adding any, and it survives a
matplotlib that will not answer. Almost none of that had been exercised,
because the first call in a test session takes the fast path and every later
one returns at the ``_registered`` guard.

The comment at line 85 records why this matters: ``addfont`` cost 23 SECONDS
of a 13-second module open on the Mask screen. Every branch below is either
that cost being avoided or a failure that must not stop a figure appearing.
"""
from __future__ import annotations

import builtins
import os

import pytest


@pytest.fixture
def cold(monkeypatch):
    """A module that has not registered anything yet, restored afterwards."""
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "_registered", False, raising=False)
    monkeypatch.setattr(figure_font, "_resolved", False, raising=False)
    return figure_font


# ---------------------------------------------------------------------------
# bundled_faces — line 53, the directory is not there
# ---------------------------------------------------------------------------

def test_a_missing_font_directory_yields_no_faces(monkeypatch):
    """The ``if not os.path.isdir(directory):`` branch.

    Documented behaviour, and the reason it is documented: a figure drawn in
    the wrong font is a blemish and never a reason for a plot not to appear.
    An installation that dropped the resources folder still plots.
    """
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "font_dir",
                        lambda: "/nonexistent/font/dir")
    assert figure_font.bundled_faces() == []


def test_the_shipped_directory_holds_ttf_faces():
    """The other side: the package really does carry the faces."""
    from spacr import figure_font

    faces = figure_font.bundled_faces()
    assert faces
    assert all(f.lower().endswith((".ttf", ".otf")) for f in faces)


# ---------------------------------------------------------------------------
# use_open_sans_for_figures — the early return, and every refusal
# ---------------------------------------------------------------------------

def test_a_second_call_returns_the_first_answer_without_asking_again(monkeypatch):
    """The ``if _registered:`` guard, which is the whole performance argument.

    The check it skips is a set comprehension over every font matplotlib
    knows, and this runs before every figure is styled.
    """
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "_registered", True, raising=False)
    monkeypatch.setattr(figure_font, "_resolved", True, raising=False)

    def explode(*_a, **_k):                       # must not be reached
        raise AssertionError("the registered guard did not short-circuit")

    monkeypatch.setattr(figure_font, "bundled_faces", explode)
    assert figure_font.use_open_sans_for_figures() is True


def test_a_matplotlib_that_cannot_be_imported_is_declined(cold, monkeypatch):
    """Lines 77-78: no font_manager, so no font, and no exception either."""
    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" and "font_manager" in (fromlist or ()):
            raise ImportError("matplotlib is not importable here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)
    assert cold.use_open_sans_for_figures() is False


def _font_manager_where(monkeypatch, cold, *, ttflist, addfont=None):
    """Install a stand-in ``font_manager`` for the duration of one call."""
    import types

    class _FontManager:
        def __init__(self):
            self.added = []

        @property
        def ttflist(self):
            value = ttflist() if callable(ttflist) else ttflist
            if isinstance(value, Exception):
                raise value
            return value

        def addfont(self, path):
            self.added.append(path)
            if addfont is not None:
                addfont(path)

    module = types.ModuleType("matplotlib.font_manager")
    module.fontManager = _FontManager()
    real_import = builtins.__import__

    def patched(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" and "font_manager" in (fromlist or ()):
            shim = types.ModuleType("matplotlib")
            shim.font_manager = module
            return shim
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched)
    return module.fontManager


class _Face:
    def __init__(self, name):
        self.name = name


def test_a_font_manager_that_will_not_list_its_fonts_still_answers(cold,
                                                                   monkeypatch):
    """Lines 82-83 and 99-100: an unreadable ttflist becomes an empty set.

    Both reads of ``ttflist`` are guarded, and both must be: the second one
    runs after faces have been added, so a failure there would otherwise throw
    away the work rather than merely failing to confirm it.
    """
    boom = RuntimeError("this font manager will not enumerate")
    manager = _font_manager_where(monkeypatch, cold, ttflist=lambda: boom)

    assert cold.use_open_sans_for_figures() is False
    # It still TRIED to add the bundled faces -- the empty set means "I do not
    # know what is installed", not "nothing is installed".
    assert manager.added


def test_an_already_installed_open_sans_costs_no_addfont(cold, monkeypatch):
    """The ``if FAMILY not in available:`` branch NOT taken.

    This is the 23-second saving the comment records. A machine with Open Sans
    installed must add nothing at all.
    """
    manager = _font_manager_where(
        monkeypatch, cold, ttflist=[_Face(cold.FAMILY)])

    assert cold.use_open_sans_for_figures() is True
    assert manager.added == []


def test_one_unreadable_face_does_not_cost_the_others(cold, monkeypatch):
    """Lines 94-96: ``continue`` past a face that will not load.

    Eight faces ship. One corrupt file must cost that one face, not the whole
    family -- which is the difference between a figure in the wrong weight and
    a figure in the wrong font.
    """
    state = {"seen": 0}
    faces_before = [_Face("DejaVu Sans")]

    def addfont(path):
        state["seen"] += 1
        if state["seen"] == 1:
            raise OSError("this face is corrupt")

    manager = _font_manager_where(
        monkeypatch, cold,
        ttflist=lambda: faces_before if state["seen"] == 0
        else [_Face("DejaVu Sans"), _Face(cold.FAMILY)],
        addfont=addfont)

    assert cold.use_open_sans_for_figures() is True
    assert len(manager.added) == len(cold.bundled_faces())


def test_faces_that_add_without_resolving_are_reported_as_unresolved(cold,
                                                                     monkeypatch):
    """Lines 103-104: added everything and the family still is not there.

    The honest answer is False. Returning True would tell the style helpers to
    ask for a family matplotlib cannot resolve, and matplotlib answers that
    with its default font and a warning per text object.
    """
    manager = _font_manager_where(
        monkeypatch, cold, ttflist=[_Face("DejaVu Sans")])

    assert cold.use_open_sans_for_figures() is False
    assert manager.added                          # it did try
