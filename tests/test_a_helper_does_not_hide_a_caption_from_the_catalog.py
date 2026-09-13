"""A caption passed through a helper never reaches the extractor.

`spacr.qt` screens write their chrome through small local helpers -- a
`_set_status(text)` that forwards to `set_translatable_text`, a
`_combo(label, tip, ...)` that builds a labelled combo box. The literal lives
at the HELPER's call site. The extractor walks the calls it knows, sees the
helper forwarding a parameter, and records nothing.

WHAT THAT COST, measured 2026-09-13. `_set_status` alone hid 117 user-facing
status captions across twenty-three screens -- "A comparison is already
running…", "Choose a destination folder first." -- every one of them English in
all nine languages for as long as the wrapper had existed. Naming `_set_status`
in the extractor recovered them.

  IT IS NOT ONE HELPER. Sweeping every function in `spacr/qt` for the same
  shape found **141 more** across 27 files, and `widgets/volcano_explorer.py`
  is 48 of them on its own because it builds its whole settings panel out of
  five local helpers. Item 65 carries the list.

WHY THIS TEST PINS RATHER THAN FAILS. Fixing them needs a rule per helper: the
template's argument position differs -- `_set_status(text)` is first,
`_combo(label, tooltip, ...)` is two, `_line(label, value, tip)` is three --
and naming them blind captures the wrong argument and writes runtime values
into the catalog, which is worse than missing a caption. That is a reading, not
a sweep.

So this holds the line at the measured number. A new helper, or a new literal
handed to an existing one, raises the count and fails -- which is the signal
that was missing when 117 captions sat unseen.

  ITEM 394's LOCAL CHECK CANNOT SEE THESE, and its own name says why:
  `test_prose_that_reaches_a_catalog_can_be_translated`. A caption a helper
  hides never reaches a catalog, so the question of whether it could be
  translated never arises. This test covers the gap before that one.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
QT = ROOT / "spacr" / "qt"

#: Measured 2026-09-13 with the sweep described above. Raise it only with the
#: list of what arrived, the way item 65 records it.
HIDDEN_CAPTIONS = 141


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _sources():
    """Every Qt module except the catalogs and the compact catalog itself.

    `spacr/qt/i18n.py` is excluded BY NAME and the reason is worth keeping: it
    holds the compact `_ROWS` catalog, so its `_row()` helper matches this
    shape with 1,941 literals -- which are the Swedish, German and Spanish
    translations themselves. The file holding the translations is not a file
    hiding them.
    """
    for path in sorted(QT.rglob("*.py")):
        if "i18n_catalogs" in path.parts or path.name == "i18n.py":
            continue
        yield path


@pytest.fixture(scope="module")
def _measured():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from importlib import import_module
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(str(ROOT / "tools"))
    from spacr.qt.i18n_catalogs import en

    known = set(builder._TEXT_METHODS) | set(builder._TEXT_CONSTRUCTORS)
    catalog = (set(en.UI_SOURCES) | set(en.SETTING_LABELS)
               | set(en.SETTING_TOOLTIPS) | set(en.CATEGORY_SOURCES)
               | set(en.MODULE_SUMMARIES))

    trees = {p: ast.parse(p.read_text(encoding="utf-8")) for p in _sources()}

    # A helper is a function that forwards one of its OWN parameters into a
    # call the extractor understands.
    helpers = set()
    for tree in trees.values():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and _call_name(call) in known:
                    forwarded = list(call.args) + [k.value for k in call.keywords]
                    if any(isinstance(a, ast.Name) and a.id in params
                           for a in forwarded):
                        helpers.add(fn.name)
                        break
    helpers -= known

    # It only HIDES something if its callers pass literals the catalog lacks.
    hidden = set()
    for path, tree in trees.items():
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _call_name(call) not in helpers:
                continue
            for arg in call.args:
                if not (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)):
                    continue
                text = arg.value.strip()
                if (len(text.split()) >= 3 and text not in catalog
                        and builder._looks_translatable(text)):
                    hidden.add((str(path.relative_to(ROOT)), text))
    return helpers, hidden


def test_the_sweep_finds_helpers_at_all(_measured):
    """Guards the guard: zero helpers satisfies the count assertion too."""
    helpers, _hidden = _measured
    assert len(helpers) >= 50, sorted(helpers)


def test_no_new_caption_is_hidden_behind_a_helper(_measured):
    """The count may fall freely; it may not rise without a reading."""
    _helpers, hidden = _measured
    assert len(hidden) <= HIDDEN_CAPTIONS, (
        f"{len(hidden)} captions are hidden behind a helper, up from "
        f"{HIDDEN_CAPTIONS}. Each needs its own extractor rule -- the "
        "template's argument position differs per helper -- so this is a "
        "reading, not a sweep. New ones:\n  "
        + "\n  ".join(f"{p}: {t[:70]}" for p, t in sorted(hidden)[:10])
    )
