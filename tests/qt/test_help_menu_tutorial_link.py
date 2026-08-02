"""The Help menu's tutorial entry must open a page that exists.

``docs/source/conf.py`` sets ``html_extra_path = ['_extra']``, so everything
under ``docs/source/_extra/`` is copied verbatim to the site root. The lesson
library therefore publishes at ``<root>/tutorials/`` — plural. Both GUIs
opened the singular ``<root>/tutorial/``, which has never been served.

These tests drive the real ``MainWindow`` menu bar, not the URL constant, so
a future refactor that rewires the action cannot quietly regress the link.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from spacr.qt.app import APPS, DOCS_URL, TUTORIALS_URL, MainWindow


DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _help_actions(window):
    for top in window.menuBar().actions():
        if top.text().replace("&", "") == "Help":
            return list(top.menu().actions())
    raise AssertionError("no Help menu on the menu bar")


def _open_urls(window, monkeypatch):
    import webbrowser

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    for act in _help_actions(window):
        if act.text().endswith("(web)"):
            act.trigger()
    return opened


def test_help_menu_opens_the_published_tutorial_library(win, monkeypatch):
    opened = _open_urls(win, monkeypatch)
    assert opened == [
        "https://einarolafsson.github.io/spacr/tutorials/",
        "https://einarolafsson.github.io/spacr/index.html",
    ]


def test_the_help_menu_never_opens_the_404_singular_path(win, monkeypatch):
    """The exact string that used to ship. GitHub Pages answers it with 404."""
    dead = "https://einarolafsson.github.io/spacr/tutorial/"
    assert dead not in _open_urls(win, monkeypatch)


def test_the_tutorial_link_is_backed_by_a_file_that_html_extra_path_publishes():
    """Prove the URL resolves from the repo, without a network call.

    ``html_extra_path`` copies ``_extra/<x>`` to ``<site root>/<x>``, so an
    ``index.html`` under ``_extra/tutorials/`` is exactly what makes
    ``/tutorials/`` a real page.
    """
    conf = (DOCS_SOURCE / "conf.py").read_text(encoding="utf-8")
    assert re.search(r"^html_extra_path\s*=\s*\['_extra'\]", conf, re.M)

    suffix = TUTORIALS_URL.split("github.io/spacr/", 1)[1].strip("/")
    assert suffix == "tutorials"
    assert (DOCS_SOURCE / "_extra" / suffix / "index.html").is_file()


def test_the_docs_url_points_at_the_generated_landing_page():
    assert DOCS_URL.endswith("/index.html")
    assert (DOCS_SOURCE / "index.rst").is_file()


def test_the_tutorial_action_keeps_its_translated_label(win):
    """`spacr/qt/i18n.py` keys its catalog on the English action text."""
    from spacr.qt.i18n import has_translation

    labels = [a.text() for a in _help_actions(win)]
    assert "Tutorial (web)" in labels
    assert has_translation("Tutorial (web)", "sv")


def test_the_tutorial_action_no_longer_calls_itself_unfinished(win):
    """The library ships 40 lessons; the Tk tooltip said "under construction"."""
    for act in _help_actions(win):
        assert "under construction" not in (act.statusTip() or "").lower()


def test_the_landing_page_app_count_matches_the_shipped_app_list():
    """``index.rst`` claimed "five pipeline apps"; ``len(APPS)`` is 34."""
    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
    match = re.search(r"The GUI ships (\d+) apps", index)
    assert match, "index.rst no longer states how many apps ship"
    assert int(match.group(1)) == len(APPS)


def test_the_landing_page_category_count_matches_the_shipped_sections():
    from spacr.qt.app import SECTIONS

    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"grouped into five categories", index)
    assert len(SECTIONS) == 5
