"""The helpers behind 291's run-wide Open Sans, held to their promises here.

``tests/test_open_sans_is_the_default_in_the_app_and_runs_only.py`` proves
the property that matters -- every entry point draws a plain ``Figure()`` in
the Open Sans spaCR ships -- and it has to do that in child interpreters,
because an entry point holds the default for the life of its process. The
helpers therefore run in those children, and this process never sees what
they do when something declines.

These tests hold the helpers' own promises in this process:

* ``_default_font_params`` puts Open Sans first without naming it twice, and
  answers nothing -- so the stock default stays -- when matplotlib is missing
  or the bundled faces cannot be registered;
* ``_open_sans_is_the_default`` holds the default and the run marker for the
  length of the run and hands both back afterwards, even when the run raises,
  and never raises on its own account when the face cannot be applied;
* ``_open_sans_if_a_run_started_this`` applies it only when a run's marker is
  in the environment.
"""
from __future__ import annotations

import builtins
import os

import matplotlib
import pytest

from spacr import figure_font

#: The environment variable a run sets so its worker processes can follow it.
MARKER = "SPACR_FIGURES_IN_OPEN_SANS"


@pytest.fixture
def no_marker(monkeypatch):
    """A process no run started."""
    monkeypatch.delenv(MARKER, raising=False)


@pytest.fixture
def registers(monkeypatch):
    """Registration answered as a success, without paying for ``addfont``.

    Whether the bundled faces really resolve is the child-interpreter file's
    question. Here it is only the input these helpers branch on.
    """
    monkeypatch.setattr(figure_font, "use_open_sans_for_figures", lambda: True)


def _family():
    return list(matplotlib.rcParams["font.family"])


# ---------------------------------------------------------------------------
# _default_font_params
# ---------------------------------------------------------------------------

def test_open_sans_leads_both_families_and_is_named_once(registers):
    """A caller that asks for ``sans-serif`` gets Open Sans too, and a list
    that already carried it does not carry it twice."""
    with matplotlib.rc_context(
            {"font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"]}):
        params = figure_font._default_font_params()

    assert params == {
        "font.family": ["Open Sans", "DejaVu Sans"],
        "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
    }


def test_without_matplotlib_there_is_nothing_to_set(monkeypatch):
    """A run that cannot draw is not made to fail by its font."""
    asked = []
    monkeypatch.setattr(figure_font, "use_open_sans_for_figures",
                        lambda: asked.append(True) or True)
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    assert figure_font._default_font_params() == {}
    assert asked == [], "registration was attempted with no matplotlib"


def test_faces_that_cannot_be_registered_leave_the_stock_default(monkeypatch):
    """Naming a family nothing can resolve would print a findfont line per
    drawn string; answering nothing keeps matplotlib's own default."""
    monkeypatch.setattr(figure_font, "use_open_sans_for_figures",
                        lambda: False)
    assert figure_font._default_font_params() == {}


def test_an_unreadable_sans_serif_list_still_puts_open_sans_first(
        registers, monkeypatch):
    """The other families are a courtesy; Open Sans is the point."""
    monkeypatch.setattr(matplotlib, "rcParams", {})
    assert figure_font._default_font_params() == {
        "font.family": ["Open Sans", "DejaVu Sans"],
        "font.sans-serif": ["Open Sans"],
    }


# ---------------------------------------------------------------------------
# _open_sans_is_the_default
# ---------------------------------------------------------------------------

def test_a_run_holds_the_default_and_the_marker_then_hands_both_back(
        registers, no_marker):
    before = _family()

    with figure_font._open_sans_is_the_default() as applied:
        assert applied is True
        assert _family()[0] == "Open Sans"
        assert os.environ.get(MARKER) == "1"

    assert _family() == before, "the run kept the caller's matplotlib"
    assert MARKER not in os.environ, "the run left its marker behind"


def test_a_marker_that_was_already_set_is_put_back_not_removed(
        registers, monkeypatch):
    """A run started inside another run hands the outer one its marker back."""
    monkeypatch.setenv(MARKER, "outer")

    with figure_font._open_sans_is_the_default():
        assert os.environ[MARKER] == "1"

    assert os.environ[MARKER] == "outer"


def test_a_run_that_raises_still_hands_matplotlib_back(registers, no_marker):
    """``spacr.cli.main`` is also called in-process, and a failed run must not
    restyle every later figure of whoever called it."""
    before = _family()

    with pytest.raises(RuntimeError, match="the run failed"):
        with figure_font._open_sans_is_the_default():
            raise RuntimeError("the run failed")

    assert _family() == before
    assert MARKER not in os.environ


def test_a_face_that_cannot_be_registered_runs_in_the_stock_default(
        monkeypatch, no_marker):
    monkeypatch.setattr(figure_font, "use_open_sans_for_figures",
                        lambda: False)
    before = _family()
    ran = []

    with figure_font._open_sans_is_the_default() as applied:
        ran.append(applied)
        assert _family() == before
        assert MARKER not in os.environ, (
            "a worker would follow a default this run never applied")

    assert ran == [False]


def test_parameters_matplotlib_refuses_leave_the_run_going(
        monkeypatch, no_marker):
    """``rc_context`` rejects a parameter this matplotlib does not know. The
    run still happens, in the stock default, with matplotlib as it was."""
    monkeypatch.setattr(
        figure_font, "_default_font_params",
        lambda: {"font.family": ["Open Sans"], "font.not_a_parameter": 1})
    before = _family()
    ran = []

    with figure_font._open_sans_is_the_default() as applied:
        ran.append(applied)
        assert _family() == before
        assert MARKER not in os.environ

    assert ran == [False]
    assert _family() == before


# ---------------------------------------------------------------------------
# _open_sans_if_a_run_started_this
# ---------------------------------------------------------------------------

def test_a_worker_no_run_started_leaves_matplotlib_alone(registers, no_marker):
    """A ``spawn`` pool started from a notebook is the notebook's."""
    before = _family()

    with figure_font._open_sans_if_a_run_started_this() as applied:
        assert applied is False
        assert _family() == before


def test_a_worker_a_run_started_follows_the_run(registers, monkeypatch):
    monkeypatch.setenv(MARKER, "1")
    before = _family()

    with figure_font._open_sans_if_a_run_started_this() as applied:
        assert applied is True
        assert _family()[0] == "Open Sans"

    assert _family() == before
    assert os.environ[MARKER] == "1", "the worker took its parent's marker"
