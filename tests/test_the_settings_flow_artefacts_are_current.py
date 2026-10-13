"""The two settings-flow artefacts must say what the source says today.

``docs/source/_generated/settings_flow.rst`` and
``spacr/qt/screens/settings_flow_index.py`` are both written by one run of
``tools/settings_flow.py --rst``. Neither is edited by hand, and neither has
any way of announcing that the code moved underneath it.

WHY THIS FILE EXISTS. Between the last regeneration and 2026-09-14 the
package gained a settings reader -- ``settings.barcode_set_from_settings``,
added 2026-09-12 with ``barcode_search.py`` -- and a write into a mapping
named ``params`` in ``style_base.font_rc`` (2026-09-14). The page went on
claiming "1092 settings" while the source read 1099 of them, and the
generated index, which the settings panel consults before it offers a
reader the flow page at all, was missing seven keys. Nothing was red. The
guard that existed,
``test_the_flow_index_agrees_with_the_page_it_was_written_from`` in
``tests/test_a_settings_api_link_lands_on_the_setting.py``, compares the two
ARTEFACTS with each other -- so it stays green for exactly as long as they
are stale together, which is the only way they ever go stale, because one
run writes both.

WHAT IS PINNED AS A PROPERTY AND WHAT IS PINNED AS BYTES. The page is
pinned by what it CLAIMS -- which settings it draws a section for, and what
each of those sections says -- so a failure can name the setting that
moved. The index is pinned byte for byte on purpose: it is a pure function
of the page, so its bytes ARE its property, and comparing key sets alone
would miss an edited docstring or a duplicated entry.

WHAT IS DELIBERATELY NOT PINNED, so that nobody reads more assurance into
this file than it gives: the fixed prose the generator puts above the first
section, and the second number in its header -- "of N read anywhere" --
which counts settings the page draws NO section for and therefore moves
when an unrelated module changes a read. The prose is a literal in the
tool, covered by the tool's own tests in
``tests/test_the_settings_flow_analyser.py``; the total is checked here
only for the one thing it cannot be, which is smaller than the number of
sections drawn.

If any of this fails, the fix is never to edit an artefact::

    python tools/settings_flow.py --rst
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "source" / "_generated" / "settings_flow.rst"
INDEX = ROOT / "spacr" / "qt" / "screens" / "settings_flow_index.py"

#: ``.. _setting-flow-<key>:`` -- the anchor a tooltip link lands on.
ANCHOR = re.compile(r"^\.\. _setting-flow-(.+):$", re.M)


@pytest.fixture(scope="module")
def flow():
    """``tools/settings_flow.py``, loaded by path -- ``tools`` is no package."""
    spec = importlib.util.spec_from_file_location(
        "settings_flow", ROOT / "tools" / "settings_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh(flow):
    """The page the generator would write from the source as it stands."""
    return flow.rst_for(flow.analyse())


@pytest.fixture(scope="module")
def page():
    """The committed page.

    NOT SKIPPED WHEN ABSENT, unlike the older checks around this artefact.
    A missing page is the failure this work has already had twice -- once
    because nothing included the fragment, once because ``.gitignore``
    swallowed the wrapper that did -- and a skip is how both stayed
    invisible.
    """
    assert PAGE.exists(), f"{PAGE} is missing; run tools/settings_flow.py --rst"
    return PAGE.read_text(encoding="utf-8")


def _sections(rst: str) -> "dict[str, str]":
    """Each setting's section, from its anchor to the next one."""
    out: "dict[str, str]" = {}
    key = None
    body: "list[str]" = []
    for line in rst.splitlines():
        found = ANCHOR.match(line)
        if found:
            if key is not None:
                out[key] = "\n".join(body)
            key, body = found.group(1), []
        elif key is not None:
            body.append(line)
    if key is not None:
        out[key] = "\n".join(body)
    return out


def test_the_page_draws_a_section_for_every_setting_read_today(page, fresh):
    """A setting the source reads and the page omits has no landing place.

    The settings panel links a tooltip to ``#setting-flow-<key>``. A key
    the page has no section for is a fragment the browser ignores in
    silence, and the reader believes they are looking at the right place.
    The reverse is as bad in a quieter way: a section for a key nothing
    reads any more tells a reader the value still goes somewhere.
    """
    drawn, current = set(_sections(page)), set(_sections(fresh))
    assert drawn == current, (
        f"the page is stale: {len(current - drawn)} settings missing "
        f"{sorted(current - drawn)[:10]}, {len(drawn - current)} stranded "
        f"{sorted(drawn - current)[:10]}; re-run tools/settings_flow.py --rst")


def test_every_section_says_today_what_the_source_says_today(page, fresh):
    """A section can be present and still be wrong about where a value goes.

    Adding a reader to a setting that already had one, or a call between
    two functions already on its path, moves no anchor at all -- so the
    section-set check above stays green while the tree beneath the heading
    names functions the code no longer has.
    """
    was, now = _sections(page), _sections(fresh)
    shared = sorted(set(was) & set(now))
    drifted = [key for key in shared if was[key] != now[key]]
    detail = ""
    if drifted:
        old_lines = was[drifted[0]].splitlines()
        new_lines = now[drifted[0]].splitlines()
        for old, new in zip(old_lines, new_lines):
            if old != new:
                detail = f"\n  {drifted[0]}: {old[:90]!r} -> {new[:90]!r}"
                break
        else:
            detail = (f"\n  {drifted[0]}: {len(old_lines)} lines committed, "
                      f"{len(new_lines)} generated")
    assert not drifted, (
        f"{len(drifted)} of {len(shared)} sections no longer match the "
        f"source: {drifted[:10]}{detail}\n"
        "Do not edit the page: run tools/settings_flow.py --rst")


def test_the_index_module_is_exactly_what_the_page_makes(flow, page):
    """The index is a pure function of the page, so its bytes are its contract.

    It is what the settings panel imports; the page itself is a docs
    artefact that is not in the wheel. An entry here that the page has no
    section for sends a reader to an anchor that does not exist.
    """
    assert INDEX.read_text(encoding="utf-8") == flow._index_module(page), (
        "the generated index is not what the committed page makes; "
        "re-run tools/settings_flow.py --rst, which writes both")


def test_the_page_counts_the_sections_it_actually_drew(page):
    """The header is a claim about the file it sits in, so check it there.

    This one needs no analysis of the package at all, which means it still
    holds in a checkout whose source has moved on -- and it is the check
    that would have caught a hand-edit adding or removing a section.
    """
    header = re.search(r"^(\d+) settings, of (\d+) read anywhere\.$",
                       page, re.M)
    assert header, "the page no longer states how many settings it drew"
    drawn = len(_sections(page))
    assert int(header.group(1)) == drawn, (
        f"the header claims {header.group(1)} sections and the page has "
        f"{drawn}")
    assert int(header.group(2)) >= drawn, (
        "the page cannot draw more sections than there are settings read: "
        f"{header.group(1)} of {header.group(2)}")
