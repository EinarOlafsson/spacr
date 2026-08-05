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

    ``html_extra_path`` copies ``<staged>/<x>`` to ``<site root>/<x>``, so an
    ``index.html`` under ``_extra/tutorials/`` is exactly what makes
    ``/tutorials/`` a real page.

    Updated 2026-08-04. This used to assert the literal
    ``html_extra_path = ['_extra']``. dba297c6 stopped publishing ``_extra``
    directly -- the tree is 712 MiB, 93% of it one narration track per lesson
    x language x voice, which put the built site at 88% of the GitHub Pages
    1 GB limit -- and now stages a hardlinked subset through
    ``tools/docs_media_budget.py``. The literal is gone, so the regex matched
    nothing and the test reported a working publish path as broken. What has
    to stay true is that ``html_extra_path`` is set to the staging directory
    ``docs_media_budget`` writes, and that the lesson index is in the source
    tree it stages FROM; both are asserted instead of the spelling.
    """
    conf = (DOCS_SOURCE / "conf.py").read_text(encoding="utf-8")
    assert re.search(r"^html_extra_path\s*=\s*\[_staged_extra\]", conf, re.M), (
        "conf.py no longer publishes the staged _extra subset; if the staging "
        "step was removed the 1 GB Pages limit is back in play")
    assert re.search(r"^_budget\.stage\(", conf, re.M), (
        "html_extra_path names a staging directory nothing stages into")

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
    """``index.rst`` claimed "five pipeline apps"; the GUI ships far more.

    Counted AFTER ``register_self_registering_modules()``, which is what
    ``spacr.qt.run`` does before ``MainWindow`` reads the registry — so this
    is the number of apps a launched GUI actually offers, not the number that
    happen to be in the module-level table. Nine apps register that way.

    Counting the module-level table instead made this test order-dependent:
    ``len(APPS)`` answered 53 in a fresh process and 62 in any run where
    something had already triggered the registration pass, so the same
    sentence in ``index.rst`` was right or wrong depending on collection
    order. Registering here is idempotent, and the registry is put back
    afterwards so no later test inherits it.
    """
    import sys

    import spacr.qt
    from spacr.qt import app as app_mod

    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    side = []
    for module_name, attribute, _field in app_mod._META_TARGETS:
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None) if module else None
        if isinstance(table, dict):
            side.append((table, dict(table)))
    try:
        spacr.qt.register_self_registering_modules()
        shipped = len(app_mod.APPS)
    finally:
        app_mod.APPS[:] = apps
        app_mod.APP_FACTORIES.clear()
        app_mod.APP_FACTORIES.update(factories)
        app_mod.APP_STAGE.clear()
        app_mod.APP_STAGE.update(stages)
        app_mod.APP_META.clear()
        app_mod.APP_META.update(meta)
        for table, saved in side:
            table.clear()
            table.update(saved)
        app_mod._refresh_sections()

    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
    match = re.search(r"The GUI ships (\d+) apps", index)
    assert match, "index.rst no longer states how many apps ship"
    assert int(match.group(1)) == shipped


#: How ``index.rst`` is allowed to spell the number of categories. The
#: sentence is prose, so the count is a word, and this is the map between
#: the two — one edit here and one in the sentence when a section is
#: added, rather than a regex nobody can read.
_CATEGORY_WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def test_the_landing_page_category_count_matches_the_shipped_sections():
    """Explore made this six. The word and the live list have to agree.

    Asserted against ``SECTIONS`` rather than a literal because sections
    are derived — one appears the day its first app registers — so a
    number typed here would be a claim nothing could check.
    """
    from spacr.qt.app import SECTIONS

    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
    word = _CATEGORY_WORDS[len(SECTIONS)]
    assert re.search(rf"grouped into {word} categories", index), (
        f"index.rst does not say the GUI has {word} ({len(SECTIONS)}) "
        f"categories, which is what spacr.qt.app.SECTIONS holds: "
        f"{list(SECTIONS)}")
    for section in SECTIONS:
        assert f"*{section}*" in index, (
            f"index.rst names the categories one by one and never names "
            f"{section!r}")
