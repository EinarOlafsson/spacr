"""The lesson lookup is read on the hover path, so it may never raise.

``spacr.qt.tutorials`` is imported from a tooltip handler and from Home.  Its
own docstring sets the contract: "A module with no lesson must cost a missing
link, never an exception in a hover handler."  That covers a module nobody
wrote a lesson for, and it covers the bundled index itself being unreadable --
a wheel built without ``spacr/resources/tutorial_index.json``, or one written
by a half-finished docs build.

``tests/qt/test_a_module_explains_itself_at_the_bottom.py`` covers the happy
lookup and the footer that draws it.  These are the answers that are empty,
plus the title a tooltip asks for, which nothing else reads.
"""
from __future__ import annotations

import pytest

from spacr.qt import tutorials


@pytest.fixture(autouse=True)
def _forget_the_cached_index():
    """The table is cached in a module global; tests must not inherit it."""
    saved = tutorials._INDEX
    tutorials._INDEX = None
    try:
        yield
    finally:
        tutorials._INDEX = saved


def _index(monkeypatch, table):
    monkeypatch.setattr(tutorials, "_INDEX", dict(table))


def test_an_unreadable_index_leaves_every_answer_empty(monkeypatch):
    """A wheel without the bundled table still draws its tooltips.

    The link is dropped, the title is dropped, and the hover handler that
    asked returns -- rather than raising out of a paint.
    """
    def explode(*_args, **_kwargs):
        raise OSError("no such file: tutorial_index.json")

    monkeypatch.setattr("importlib.resources.files", explode)

    assert tutorials._index() == {}
    assert tutorials.lesson_for("measure") == ""
    assert tutorials.lesson_title("measure") == ""
    assert tutorials.tutorial_url("measure") == ""
    assert tutorials.has_tutorial("measure") is False


def test_the_failure_is_cached_too_so_a_hover_does_not_retry_per_pixel(
    monkeypatch,
):
    """Reading the table is attempted ONCE, however many hovers follow."""
    attempts = []

    def explode(*_args, **_kwargs):
        attempts.append(1)
        raise OSError("no such file")

    monkeypatch.setattr("importlib.resources.files", explode)

    for _ in range(5):
        tutorials.lesson_for("measure")

    assert len(attempts) == 1


def test_the_lesson_title_is_the_lessons_own_title(monkeypatch):
    """The tooltip says what the lesson is called, not what the module is."""
    _index(monkeypatch, {
        "measure": {"lesson": "measure-a-screen", "title": "Measure a screen"},
    })

    assert tutorials.lesson_title("measure") == "Measure a screen"
    assert tutorials.lesson_for("measure") == "measure-a-screen"
    assert tutorials.tutorial_url("measure") == (
        f"{tutorials.TUTORIALS_URL}#lesson=measure-a-screen"
    )
    assert tutorials.has_tutorial("measure") is True


def test_a_lesson_recorded_without_a_title_still_links(monkeypatch):
    """The link is what matters; a missing title costs the tooltip's words."""
    _index(monkeypatch, {"measure": {"lesson": "measure-a-screen"}})

    assert tutorials.lesson_title("measure") == ""
    assert tutorials.has_tutorial("measure") is True


@pytest.mark.parametrize("key", ["", None, "no_such_module"])
def test_a_module_with_no_lesson_is_not_sent_to_the_library_front_page(
    monkeypatch, key,
):
    """Empty rather than the index of seventy-three lessons.

    The footer draws the link only when this answers, so an empty string is
    "no link" -- while the front page would be a link that looks like an
    answer and is not one.
    """
    _index(monkeypatch, {"measure": {"lesson": "measure-a-screen"}})

    assert tutorials.tutorial_url(key) == ""
    assert tutorials.lesson_title(key) == ""
    assert tutorials.has_tutorial(key) is False


def test_an_entry_that_is_not_an_object_is_dropped_rather_than_trusted(
    monkeypatch,
):
    """A malformed row in the generated table cannot reach a URL."""
    import json

    class _Raw:
        def read_text(self, encoding=None):
            return json.dumps({"lessons": {
                "good": {"lesson": "one"},
                "bad": "two",
            }})

    class _Anchor:
        def __truediv__(self, _name):
            return _Raw()

    monkeypatch.setattr("importlib.resources.files", lambda _pkg: _Anchor())

    assert set(tutorials._index()) == {"good"}
    assert tutorials.has_tutorial("bad") is False
