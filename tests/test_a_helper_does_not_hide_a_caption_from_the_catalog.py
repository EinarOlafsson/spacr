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

WHY THE RULES ARE KEYED. The template's argument position differs per helper
-- `_set_status(text)` is first, `_combo(options, caption)` is second,
`_spin(low, high, step, decimals, caption)` is fifth -- and one name means
different things in different modules: `_say` is defined in twelve of them,
one taking a style key second and one showing data verbatim. So
`_HELPER_CAPTION_RULES` in tools/build_i18n_catalogs.py is keyed by
(call-site module, helper) and names the caption positions, added 2026-09-14.
A literal counts as hidden here only when no rule reaches it.

The count still pins. A new helper, or a new literal handed to a helper
without a rule, raises it and fails -- which is the signal that was missing
when 117 captions sat unseen.

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

#: Measured 2026-09-13 at 141 with the sweep described above. Lowered
#: 2026-09-14 to 1 when `_HELPER_CAPTION_RULES` gave 35 (module, helper) pairs
#: their caption positions. The one left is `app.py`'s About-box `_line`, whose
#: only literals are "spaCR" and "© Olafsson Lab" -- names, not prose, and its
#: prose callers already pass `tr(...)`. Raise it only with the list of what
#: arrived, the way item 65 records it.
HIDDEN_CAPTIONS = 1


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
def builder():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from importlib import import_module
        return import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(str(ROOT / "tools"))


@pytest.fixture(scope="module")
def _measured(builder):
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

    # It only HIDES something if its callers pass literals the catalog lacks
    # and no keyed extractor rule reaches that argument.
    hidden = set()
    for path, tree in trees.items():
        module = path.relative_to(QT).as_posix()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _call_name(call) not in helpers:
                continue
            reached = list(builder._helper_caption_arguments(
                call, module, _call_name(call)))
            for arg in call.args:
                if not (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)):
                    continue
                if any(arg is node for node in reached):
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
        f"{HIDDEN_CAPTIONS}. Each needs its own entry in "
        "_HELPER_CAPTION_RULES -- the template's argument position differs "
        "per helper -- so this is a reading, not a sweep. New ones:\n  "
        + "\n  ".join(f"{p}: {t[:70]}" for p, t in sorted(hidden)[:10])
    )


def test_each_rule_still_names_the_parameter_at_its_position(builder):
    """A signature change must fail here, not capture a runtime value.

    Every rule records the parameter it expects at each caption position. If a
    helper gains an argument in front of its caption, the position now holds
    something else -- a widget, a count, a style key -- and the extractor
    would catalogue whatever literal callers pass there.
    """
    problems = []
    for (caller, helper), (defined_in, pairs) in sorted(
            builder._HELPER_CAPTION_RULES.items()):
        tree = ast.parse((QT / defined_in).read_text(encoding="utf-8"))
        defs = [fn for fn in ast.walk(tree)
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fn.name == helper]
        expected_definitions = 1
        if (defined_in, helper) == ("screens/make_masks.py", "row"):
            parents = {child: parent for parent in ast.walk(tree)
                       for child in ast.iter_child_nodes(parent)}
            assert {parents[fn].name for fn in defs} == {
                "_build_methods_card", "_build_propagate_card"}
            expected_definitions = 2
        if len(defs) != expected_definitions:
            problems.append(f"{caller}:{helper}: {len(defs)} definitions of "
                            f"{helper} in {defined_in}, expected {expected_definitions}")
            continue
        for definition in defs:
            params = [a.arg for a in definition.args.posonlyargs + definition.args.args]
            if params and params[0] in {"self", "cls"}:
                params = params[1:]
            for position, parameter in pairs:
                found = params[position] if position < len(params) else None
                if found != parameter:
                    problems.append(
                        f"{caller}:{helper}: position {position} is {found!r} in "
                        f"{defined_in}:{definition.lineno}, the rule expects {parameter!r}")
    assert not problems, "\n".join(problems)


def test_each_rule_is_called_where_it_is_keyed(builder):
    """A rule for a call that no longer exists is a rule nobody re-reads."""
    stale = []
    for caller, helper in sorted(builder._HELPER_CAPTION_RULES):
        tree = ast.parse((QT / caller).read_text(encoding="utf-8"))
        if not any(isinstance(node, ast.Call) and _call_name(node) == helper
                   for node in ast.walk(tree)):
            stale.append(f"{caller}: no call to {helper}")
    assert not stale, "\n".join(stale)


def test_make_masks_parameter_rows_expose_help_but_not_setting_keys(builder):
    """Both local row helpers carry prose past a programmatic field key."""
    tree = ast.parse((QT / "screens/make_masks.py").read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and _call_name(node) == "row"]
    assert len(calls) == 27
    reached = set()
    keys = set()
    for call in calls:
        keys.add(ast.literal_eval(call.args[0]))
        arguments = list(builder._helper_caption_arguments(call, "screens/make_masks.py", "row"))
        assert arguments == [call.args[1], call.args[3]]
        reached.update(ast.literal_eval(arg) for arg in arguments)
    assert not keys & reached
    assert "Offset" in reached and "Blur first" in reached
    assert any(text.startswith("Subtracted from the Gaussian-weighted local mean") for text in reached)
