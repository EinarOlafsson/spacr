"""Every tile on the README grid lands on an API page that explains itself.

Instruction 366 part 3, and it is what makes parts 1 and 2 safe: text is only
being REMOVED from the README on the promise that the API says it better, so
the API has to say it first.

MEASURED 2026-09-02, per tile, which is what the instruction asked for before
anything was written. Nine of the twenty-one tiles landed on a module
docstring of six to eight words -- including four of the six CORE pipeline
modules:

    measure           spacr.measure          6 words
    regression        spacr.ml               6
    mask              spacr.core             7
    umap              spacr.core             7
    map_barcodes      spacr.sequencing       8
    analyze_plaques   spacr.submodules       8
    recruitment       spacr.submodules       8
    invasion          spacr.submodules       8
    replication       spacr.submodules       8

against `spacr.foreign` at 1,180 and `spacr.align` at 715, which are the
house standard this file measures the others against.

A WORD COUNT IS NOT THOROUGHNESS, and this file does not claim it is. 366's
bar is a reader's: after landing, can somebody who clicked the tile because
they did not know what the module does say what it is for, what it needs,
what it produces and what to do next? Nothing here can check that. What it
CAN check is the floor -- six words cannot answer four questions -- and that
the floor rises rather than falls.

SO IT IS A RATCHET, like `test_nested_functions_are_documented`: every module
still under the bar is listed with the number it is at, a module may not get
thinner, and one that climbs past the bar has its entry deleted. A tile added
tomorrow that lands on nothing fails immediately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Below this, a landing page cannot answer 366's four questions whatever it
#: says. Set from the measurement: the twelve pages already worth arriving at
#: start at 133 words, the nine that are not sit at 6 to 8, and there is
#: nothing in between to argue about.
BAR = 60

#: Module -> the word count it is stuck at, for the ones still owed.
#:
#: FOUR OF THESE ARE THE OTHER SESSION'S FILES under the two-session protocol
#: in instruction 325 -- `spacr/core.py`, `spacr/ml.py`, `spacr/sequencing.py`
#: and `spacr/submodules.py` -- so they are recorded here rather than
#: rewritten. `spacr/measure.py` was this session's to fix and is no longer
#: in this table.
STILL_OWED = {
    "spacr.core": 7,
    "spacr.submodules": 8,
}


def _tiles():
    """``{tile key: module}`` for every tile the README grid draws."""
    sys.path.insert(0, str(ROOT / "packaging"))
    from generate_readme_visuals import _api_urls, _module_grid

    import spacr.qt

    spacr.qt.register_self_registering_modules()
    urls = _api_urls()
    found = {}
    for entry in _module_grid():
        key = entry[0] if isinstance(entry, (tuple, list)) else entry
        url = urls.get(key, "")
        if not url or "/api/" not in url:
            continue
        path = url.split("/api/")[-1].replace("/index.html", "")
        found[key] = path.replace("/", ".")
    return found


def _words(module: str) -> int:
    """How many words the module's own docstring has, or -1 if it cannot
    be imported -- which is a different failure and is reported as one."""
    try:
        doc = importlib.import_module(module).__doc__ or ""
    except Exception:                                        # noqa: BLE001
        return -1
    return len(doc.split())


@pytest.fixture(scope="module")
def tiles():
    return _tiles()


def test_every_tile_resolves_to_a_module_that_imports(tiles):
    """A tile pointing at a page nothing can build is a dead link."""
    assert tiles, "no tiles were found; the grid or the URL table moved"
    broken = [f"{key} -> {mod}" for key, mod in tiles.items()
              if _words(mod) < 0]
    assert not broken, f"these tiles land on a module that will not import: {broken}"


def test_no_module_lands_on_less_prose_than_it_did(tiles):
    """The ratchet. A module in the table may improve and leave it; it may
    not get thinner, and one that is not in the table may not fall below the
    bar at all."""
    thin = []
    for key, module in sorted(tiles.items()):
        words = _words(module)
        floor = STILL_OWED.get(module)
        if floor is None:
            if words < BAR:
                thin.append(f"{key} ({module}) has {words} words and is not "
                            f"recorded as owed -- it has regressed")
        elif words < floor:
            thin.append(f"{key} ({module}) fell from {floor} to {words}")
    assert not thin, "\n  ".join(["landing pages went backwards:"] + thin)


def test_a_module_that_climbed_past_the_bar_leaves_the_table(tiles):
    """Dead entries are how a ratchet stops ratcheting: a module that now
    explains itself must not go on being excused."""
    stale = [f"{module} is recorded as owed at {floor} words but now has "
             f"{_words(module)}"
             for module, floor in sorted(STILL_OWED.items())
             if _words(module) >= BAR]
    assert not stale, "\n  ".join(["remove these from STILL_OWED:"] + stale)


@pytest.mark.parametrize(
    "module", ["spacr.measure", "spacr.ml", "spacr.sequencing"]
)
def test_each_repaired_module_answers_the_four_questions(tiles, module):
    """Modules fixed against 366's own bar are asserted against it rather
    than only against a word count.

    The four questions the instruction asks a landing page to answer are
    checked as headings, not as prose quality -- a reader still has to judge
    that, and no test can."""
    doc = (importlib.import_module(module).__doc__ or "").upper()
    for phrase in ("WHAT IT IS FOR", "WHAT IT NEEDS", "WHAT IT PRODUCES",
                   "WHAT TO DO NEXT"):
        assert phrase in doc, f"{module}'s landing page never says {phrase}"


def test_two_tiles_do_not_share_one_landing_page(tiles):
    """Recorded as a KNOWN gap rather than asserted clean, because it is not
    fixable from the docstring side.

    Six tiles land on three shared pages: mask and umap both open
    `spacr.core`, and the four toxoplasma assays all open
    `spacr.submodules`. Clicking "Analyze Plaques" and clicking "Recruitment"
    arrive at the same text, which cannot explain either -- so part 3 is not
    finished by prose alone; those tiles need pages of their own.
    """
    shared = {}
    for key, module in tiles.items():
        shared.setdefault(module, []).append(key)
    doubled = {mod: sorted(keys) for mod, keys in shared.items()
               if len(keys) > 1}
    assert doubled == {
        "spacr.core": ["mask", "umap"],
        "spacr.submodules": ["analyze_plaques", "invasion", "recruitment",
                             "replication"],
    }, (f"the tiles that share a landing page changed: {doubled}. Update this "
        f"test and instruction 366 -- a new sharing is a new place where one "
        f"page has to explain two modules.")
