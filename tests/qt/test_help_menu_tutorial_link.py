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

from spacr.qt.app import DOCS_URL, TUTORIALS_URL, MainWindow


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


def _index_rst() -> str:
    return (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")


def _paragraph(index: str, marker: str) -> str:
    """The one paragraph of ``index.rst`` containing ``marker``.

    Both landing-page checks are about a single sentence, and reading the
    whole file instead would make them fire on any unrelated italics or
    bold anywhere on the page.
    """
    for para in index.split("\n\n"):
        if marker in para:
            return para
    return ""


def test_the_landing_page_states_no_app_count_that_a_fold_falsifies():
    """``index.rst`` used to print a headcount. It could not stay true.

    The sentence went "five pipeline apps", then 67 apps, then 65, and
    every one of those edits was this test failing first. What makes the
    figure unfixable rather than merely tedious is that what it counted
    -- ``len(APPS)`` after ``register_self_registering_modules()``, the
    tiles -- stopped meaning what the sentence said. A FOLDED module
    gives up its tile and keeps everything else: it is shipped,
    reachable, runnable, translated and documented, it just opens from
    its host's masthead. Every fold therefore drops the printed figure
    without anything leaving the build, so the page understates the GUI
    by however many modules have folded.

    It is unstable as well as wrong. Counting the module-level table
    answered 53 in a fresh process and 62 once anything had triggered the
    registration pass, so the same sentence was right or wrong depending
    on test collection order.

    The page states the shape of the inventory instead -- the categories,
    named one by one, and the fact that a module can live on a host
    masthead -- and no total. This test guards that decision from both
    sides: it fails if a count comes back, and it fails if the claim that
    replaced it stops being true.
    """
    index = _index_rst()

    stale = re.search(r"\b(?:ships|offers|has)\s+(?:\d+|"
                      r"five|six|seven|eight|nine|ten|dozens of)\s+apps",
                      index, re.I)
    assert not stale, (
        f"index.rst is printing an app count again ({stale.group(0)!r}). "
        f"A tile count is falsified by the next fold and already "
        f"undercounts what ships: folded modules have no tile and are "
        f"shipped. State the categories, not a total.")

    # ...and the claim that replaced it has to be a live one, not a
    # sentence that would survive folding being ripped out tomorrow.
    from spacr.qt.widgets.fold_strip import folded_modules

    folded = folded_modules()
    assert folded, (
        "no module is folded any more, so index.rst is explaining a "
        "masthead-button route that no longer exists")
    assert "masthead" in index, (
        "index.rst no longer tells the reader that a module with no tile "
        "opens from its host's masthead, which is the only thing standing "
        "in for the count that was removed")

    # Names wrap across lines in reST source, so the newline inside a
    # ``**...**`` span is whitespace, not part of the name.
    named = {" ".join(match.split())
             for match in re.findall(r"\*\*([^*]+)\*\*",
                                     _paragraph(index, "masthead"))}
    assert named, "the fold paragraph names no module at all"
    live = {entry[0] for entry in folded.values()}
    assert named <= live, (
        f"index.rst says {sorted(named - live)} open from a host "
        f"masthead, and no host folds them; a module that was unfolded "
        f"or renamed is being advertised at the wrong address")


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

    Unlike the app count, this one is worth printing: a section appears
    only when a whole new kind of work does, which is rare, and the word
    is checked against the live list here, so it cannot go quietly wrong.
    What is NOT asserted is the wording around it. The sentence has been
    rephrased once already ("grouped into seven categories" became "groups
    its applications into seven categories") and the old regex reported a
    correct page as a page that had dropped the claim.
    """
    from spacr.qt.app import SECTIONS

    index = _index_rst()
    word = _CATEGORY_WORDS[len(SECTIONS)]
    assert re.search(rf"\b{word} categories\b", index), (
        f"index.rst does not say the GUI has {word} ({len(SECTIONS)}) "
        f"categories, which is what spacr.qt.app.SECTIONS holds: "
        f"{list(SECTIONS)}")
    for other in set(_CATEGORY_WORDS.values()) - {word}:
        assert not re.search(rf"\b{other} categories\b", index), (
            f"index.rst also claims {other!r} categories somewhere, so "
            f"one of the two sentences is wrong")
    listed = _paragraph(index, "categories")
    for section in SECTIONS:
        assert f"*{section}*" in listed, (
            f"index.rst names the categories one by one and never names "
            f"{section!r}")
    # ...and the list has to be exactly the live one. A section that is
    # retired has to come off the page too, or the landing page goes on
    # advertising a tab the GUI no longer draws -- which the check above
    # cannot see, because every remaining name is still right.
    named = {" ".join(match.split())
             for match in re.findall(r"(?<![*\w])\*([^*\n]+?)\*(?!\*)",
                                     listed)}
    assert named == set(SECTIONS), (
        f"the category list on index.rst names {sorted(named)}; "
        f"spacr.qt.app.SECTIONS holds {list(SECTIONS)}")
