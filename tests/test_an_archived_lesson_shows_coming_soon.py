"""A module whose lesson is archived says "Coming soon" instead of linking.

The bundled ``tutorial_index.json`` keeps a ``coming_soon`` table of the
modules whose lesson is archived (``tools/tutorials/archive_lessons.py``).
Their Tutorial word is drawn without a link and with a Coming soon badge, so
the reader learns the lesson exists but cannot open a lesson the library no
longer shows. A module with a visible lesson keeps its link; a module with
neither gets nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from spacr.qt import tutorials

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "tutorials"))
from archived import load_index  # noqa: E402

ARCHIVED_MODULES = sorted(
    entry["app_key"] for entry in load_index()["lessons"].values() if entry.get("app_key"))


@pytest.fixture(autouse=True)
def _forget_the_cached_tables():
    saved = tutorials._INDEX, tutorials._COMING_SOON
    tutorials._INDEX = tutorials._COMING_SOON = None
    try:
        yield
    finally:
        tutorials._INDEX, tutorials._COMING_SOON = saved


def test_the_bundled_table_names_every_archived_module_and_no_visible_one():
    bundled = json.loads((REPO / "spacr/resources/tutorial_index.json")
                         .read_text(encoding="utf-8"))
    assert sorted(bundled.get("coming_soon", {})) == ARCHIVED_MODULES
    assert not set(bundled["coming_soon"]) & set(bundled["lessons"])
    archived = set(load_index()["lessons"])
    assert not archived & {row["lesson"] for row in bundled["catalog"]}
    assert [row["number"] for row in bundled["catalog"]] == list(
        range(1, len(bundled["catalog"]) + 1))


@pytest.mark.parametrize("app_key", ARCHIVED_MODULES)
def test_an_archived_module_gets_a_badge_and_no_link(app_key):
    html = tutorials._tutorial_word_html(app_key)

    assert tutorials.tutorial_url(app_key) == ""
    assert "Coming soon" in html and "Tutorial" in html
    assert "<a " not in html and "#lesson=" not in html


def test_a_module_with_a_lesson_keeps_its_link():
    html = tutorials._tutorial_word_html("mask")

    assert html.startswith('<a href="') and "#lesson=07_mask" in html
    assert "Coming soon" not in html


def test_a_module_with_neither_gets_nothing():
    assert tutorials._tutorial_word_html("no_such_module") == ""
    assert tutorials._tutorial_word_html("") == ""


def test_an_unreadable_table_costs_the_badge_not_an_exception(monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("no such file: tutorial_index.json")

    monkeypatch.setattr("importlib.resources.files", explode)

    assert tutorials._coming_soon() == {}
    assert tutorials._tutorial_word_html(ARCHIVED_MODULES[0] if ARCHIVED_MODULES
                                         else "plaque") == ""
