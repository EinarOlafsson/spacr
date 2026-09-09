"""Where each setting is read, and the call paths that carry it there.

Instruction 368: a user should be able to click a setting and see the tree of
functions below its entry point that actually use it, every node clickable.

WHY THIS IS A SEPARATE TOOL AND NOT A SPHINX PLUGIN. No documentation
compiler -- Sphinx, mkdocs, pdoc, Doxygen -- does settings-flow analysis,
because none of them knows what a "setting" is. The analysis has to be
purpose-written whatever renders the result, so this emits DATA and the
renderer is a detail. Sphinx keeps its job.

WHAT MAKES IT TRACTABLE, measured 2026-09-02: the package has 156,371 call
sites, of which a naive name match resolves 18%. A tree over that would be
neither buildable nor readable. But the tree does not need the whole call
graph -- it needs the subgraph settings TRAVEL along, and that is 698 call
sites over 562 functions. Two orders of magnitude smaller.

WHAT IT REFUSES TO DO. Static analysis cannot follow a call through
``getattr``, a dispatch dict, a Qt signal, or a callback passed as a value.
Those are recorded as UNRESOLVED rather than guessed at, and the renderer is
required to show them: a tree that silently drops what it could not resolve
is worse than one that admits the gap, because it looks complete.

Run it::

    python tools/settings_flow.py            # writes docs/settings_flow.json
    python tools/settings_flow.py --key cell_channel   # one setting, as a tree
"""
from __future__ import annotations

import argparse
import re
import ast
import json
from collections import defaultdict
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "spacr"
OUTPUT = ROOT / "docs" / "settings_flow.json"
RST_OUTPUT = ROOT / "docs" / "source" / "_generated" / "settings_flow.rst"
#: The keys the page has a section for, as a module the GUI can import.
#:
#: The RST is a docs artefact and is not in the wheel, but the settings
#: panel needs to know which settings the page can answer for before it
#: offers a link to it. A generated frozenset is the smallest thing that
#: carries that across.
INDEX_OUTPUT = (ROOT / "spacr" / "qt" / "screens"
                / "settings_flow_index.py")

#: Names a settings mapping is plausibly bound to. The same rule the
#: consumer-map generator uses, and for the same reason: without it, every
#: string subscript on every object counts as a settings read, and a local
#: dict of filenames made one function the published target for thirty
#: settings.
SETTINGS_NAMES = frozenset({
    "settings", "setting", "cfg", "config", "conf", "opts", "options",
    "params", "parameters", "defaults", "kwargs",
})

#: Object roles whose per-object settings are built with an f-string. Naming
#: them is what turns `f'{object_type}_channel'` from an unresolvable dynamic
#: read into four concrete keys.
OBJECT_ROLES = ("cell", "nucleus", "pathogen", "cytoplasm", "organelle")


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = rel.parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_settings(node, aliases: Set[str]) -> bool:
    if isinstance(node, ast.BoolOp):
        # `settings or {}`, the defensive idiom for a None default. The
        # analyser missed every setting behind one: `configured =
        # dict(settings or {})` in `hit_investigation` made 17 `hit_*`
        # settings look read by nothing, and the settings panel offered
        # their API link no page to land on. Seen through, because the
        # expression IS the settings mapping whenever either side is.
        return any(_is_settings(v, aliases) for v in node.values)
    if isinstance(node, ast.Name):
        return node.id in SETTINGS_NAMES or node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr in SETTINGS_NAMES
    return False


def _is_settings_value(node, aliases: Set[str]) -> bool:
    """Whether an EXPRESSION evaluates to the settings mapping.

    :func:`_is_settings` answers for a name; this answers for the four
    wrappers a caller writes around one -- ``settings.copy()``,
    ``dict(settings)``, ``deepcopy(settings)`` and ``settings or {}`` --
    so the same rule serves every reader instead of being spelled out at
    each of them.

    :param node: the expression.
    :param aliases: local names already known to be the settings mapping.
    :returns: True when the expression is the settings mapping.
    """
    if _is_settings(node, aliases):
        return True
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("copy", "deepcopy"):
            return _is_settings(f.value, aliases)
        if isinstance(f, ast.Name) and f.id in ("dict", "deepcopy"):
            return any(_is_settings(a, aliases) for a in node.args)
    return False


def _filled_from_settings(fn: ast.AST) -> Set[str]:
    """Local names a settings mapping was poured into.

    THE SHAPE EVERY `spacr.settings` FACTORY HAS::

        resolved = {'layout': 'auto', 'z_handling': Z_KEEP, ...}
        resolved.update(dict(settings or {}))
        return resolved

    `resolved` is bound to a dict LITERAL, so no binding rule sees it, and
    the reads one line below -- and in the three screens that call
    `convert.default_settings` -- came back empty for `layout`,
    `plate_naming` and `z_handling`. The update is what makes it the
    settings mapping, and the update is a statement of its own.

    :param fn: the function to read.
    :returns: the local names filled from the settings mapping.
    """
    found: Set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "update"):
            continue
        if not isinstance(f.value, ast.Name):
            continue
        if any(_is_settings_value(a, set()) for a in node.args):
            found.add(f.value.id)
    return found


def _alias_targets(fn: ast.AST,
                   helpers: "Set[str] | frozenset" = frozenset()) -> Set[str]:
    """Local names bound to the settings mapping inside ``fn``.

    Five binding forms, and the fifth is not a spelling of the first four.
    ``x = settings``, ``x = settings.copy()``, ``x = dict(settings)`` and
    ``x = deepcopy(settings)`` all say the mapping is right there in the
    expression. ``x = power_default_settings(settings)`` does not: the
    mapping is what the CALLEE returns, and nothing in this statement says
    so. That is why 20 settings -- all 15 `power_*`, `layout`,
    `plate_naming`, `z_handling`, `guide_fractions_file` and
    `hit_phenotype` -- read as consumed by nothing while being read in
    plain sight one line below the call.

    ``helpers`` is what closes it: the local names, in this module, of
    functions that RETURN the settings mapping. It is computed by
    :func:`_settings_helpers` from the package rather than listed here,
    because a hand-written list of three helpers would go stale the day a
    fourth was written and would fail in exactly this silent way.

    :param fn: the function to read.
    :param helpers: local names that return a settings mapping.
    :returns: the local names bound to the settings mapping.
    """
    found: Set[str] = set()
    for node in ast.walk(fn):
        value = getattr(node, "value", None)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or value is None:
            continue
        ok = _is_settings_value(value, set())
        if isinstance(value, ast.Call) and not ok:
            f = value.func
            if _names_a_helper(f, helpers):
                # HANDED THE MAPPING, AND HANDING ONE BACK. Both halves are
                # required: a helper called on something else returns
                # something else, and a call that takes the settings but
                # returns a bool -- `check_settings` -- is not a binding.
                ok = _takes_settings(value)
        if not ok:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        found |= {t.id for t in targets if isinstance(t, ast.Name)}
    return found | _filled_from_settings(fn)


def _names_a_helper(func: ast.AST, helpers) -> bool:
    """Whether ``func`` names one of ``helpers``.

    Both call spellings, because both are used: `default_settings(...)`
    where the module imported the name, and `convert.default_settings(...)`
    where it imported the module. The attribute form is matched on the
    bare name, which is the same set -- a helper's name is what makes it
    findable either way.

    :param func: the ``func`` of a call node.
    :param helpers: the local helper names.
    :returns: True when the call is to a settings-returning helper.
    """
    if isinstance(func, ast.Name):
        return func.id in helpers
    if isinstance(func, ast.Attribute):
        return func.attr in helpers
    return False


def _takes_settings(call: ast.Call) -> bool:
    """Whether ``call`` is handed the settings mapping, by any argument."""
    if any(_is_settings(a, set()) for a in call.args):
        return True
    return any(k.value is not None and _is_settings(k.value, set())
               for k in call.keywords)


def _returns_settings(fn: ast.AST) -> bool:
    """Whether ``fn`` hands its caller back a settings mapping.

    Asked of the RETURN STATEMENTS, not of the name: `default_settings`,
    `power_default_settings` and `hit_investigation_default_settings` all
    happen to end in the same word, and a rule built on that would accept
    `check_settings`, which returns a bool, and miss the next helper that
    is spelled differently.

    :param fn: the function to read.
    :returns: True when some return hands back the settings mapping.
    """
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    aliases = _alias_targets(fn) | (params & SETTINGS_NAMES)
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            if _is_settings(node.value, aliases):
                return True
    return False


def _settings_helpers(modules: Dict[str, ast.Module]) -> Set[str]:
    """``module.function`` for every function that returns the settings.

    Read from the package once, before the analysis proper, because a
    module can be handed a helper defined in another module -- `convert`
    defines `default_settings` and three screens call it.

    :param modules: the parsed package.
    :returns: qualified names of the settings-returning helpers.
    """
    found: Set[str] = set()
    for module, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _returns_settings(node):
                found.add(f"{module}.{node.name}")
    return found


def _module_tables(tree: ast.Module) -> Dict[str, Set[str]]:
    """``name -> the string keys`` of each module-level table of names.

    THE KEY IS NEVER A LITERAL AT THE CALL SITE, and this is where it is
    one instead. `spacr.seg_qc` writes::

        QC_DEFAULTS = {"min_objects": 5, "tiny_fraction": 0.30, ...}
        SETTING_KEYS = {f"seg_qc_{name}": name for name in QC_DEFAULTS}
        MODE_SETTING = "seg_qc"

    and then reads `(settings or {}).get(key)` through a loop over
    `SETTING_KEYS.items()`. The eleven `seg_qc_*` names do not appear as
    string literals ANYWHERE in the package -- they are synthesised here --
    so a pass that looks for a name at a subscript finds none of them, and
    all twelve settings read as consumed by nobody.

    Three shapes are folded, all at module scope and all of plain strings:
    a dict literal, a tuple/list/set literal, and a dict comprehension over
    an already-folded table whose key is an f-string of constants and the
    loop variable. Anything else is left alone rather than guessed at --
    this is constant folding, not evaluation.

    :param tree: the parsed module.
    :returns: ``{table name: its string keys}``.
    """
    tables: Dict[str, Set[str]] = {}

    def strings(node) -> Set[str]:
        if isinstance(node, ast.Dict):
            return {k.value for k in node.keys
                    if isinstance(k, ast.Constant)
                    and isinstance(k.value, str)}
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return {e.value for e in node.elts
                    if isinstance(e, ast.Constant)
                    and isinstance(e.value, str)}
        return set()

    def folded(node) -> Set[str]:
        """A dict comprehension `{f"pre_{v}": ... for v in TABLE}`."""
        if not isinstance(node, ast.DictComp) or len(node.generators) != 1:
            return set()
        gen = node.generators[0]
        if gen.ifs or not isinstance(gen.target, ast.Name):
            return set()
        source = gen.iter
        if isinstance(source, ast.Attribute) and source.attr in ("keys",):
            source = source.value
        if isinstance(source, ast.Call):
            source = source.func
            if isinstance(source, ast.Attribute):
                source = source.value
        if not isinstance(source, ast.Name) or source.id not in tables:
            return set()
        if not isinstance(node.key, ast.JoinedStr):
            return set()
        parts: List[str] = []
        for piece in node.key.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif (isinstance(piece, ast.FormattedValue)
                  and isinstance(piece.value, ast.Name)
                  and piece.value.id == gen.target.id):
                parts.append("\0")
            else:
                return set()
        template = "".join(parts)
        if template.count("\0") != 1:
            return set()
        head, _, tail = template.partition("\0")
        return {f"{head}{name}{tail}" for name in tables[source.id]}

    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is None:
            continue
        keys = strings(value) or folded(value)
        if not keys:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                tables[target.id] = keys
    return tables


def _module_constants(tree: ast.Module) -> Dict[str, str]:
    """``name -> value`` for each module-level plain string constant."""
    out: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value,
                                                                 str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = value.value
    return out


def _keyed_names(fn: ast.AST, tables: Dict[str, Set[str]],
                 constants: Dict[str, str]) -> Dict[str, Set[str]]:
    """``local name -> the setting names it can hold`` inside ``fn``.

    Two ways a variable comes to hold a setting name without being one:
    it is the loop target over a folded table, or it is a module-level
    string constant -- `MODE_SETTING = "seg_qc"`, read as
    `(settings or {}).get(MODE_SETTING, "report")`, which is a literal
    everywhere except at the call site.

    :param fn: the function to read.
    :param tables: the module's folded tables.
    :param constants: the module's plain string constants.
    :returns: ``{name: the keys it stands for}``.
    """
    keyed: Dict[str, Set[str]] = {name: {value}
                                  for name, value in constants.items()}
    for node in ast.walk(fn):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        source = node.iter
        if isinstance(source, ast.Call) and isinstance(source.func,
                                                       ast.Attribute):
            if source.func.attr not in ("items", "keys"):
                continue
            source = source.func.value
        if not isinstance(source, ast.Name) or source.id not in tables:
            continue
        keys = tables[source.id]
        target = node.target
        # `for key in TABLE` and `for key, value in TABLE.items()`: the
        # KEY is the first name either way, and the value is not one.
        if isinstance(target, ast.Name):
            keyed[target.id] = keys
        elif isinstance(target, ast.Tuple) and target.elts:
            first = target.elts[0]
            if isinstance(first, ast.Name):
                keyed[first.id] = keys
    return keyed


def _keys_from(node, aliases: Set[str],
               keyed: "Dict[str, Set[str]] | None" = None
               ) -> List[Tuple[str, str]]:
    """``(key, form)`` pairs a subscript or ``.get`` call reads."""
    out: List[Tuple[str, str]] = []
    keyed = keyed or {}

    def named(node, form: str) -> bool:
        """A key held in a variable: a folded table's key, or a constant.

        FILTERED AGAINST THE SETTINGS THE GUI DECLARES, and only here. A
        literal at a subscript is self-evidently the key that was written;
        a name is not, and `SETTINGS_NAMES` is deliberately generous about
        what counts as a settings mapping -- `kwargs`, `config`, `opts`.
        Without the filter, `kwargs[TITLE]` in the app catalogue recorded
        `title`, `intro` and `entry` as settings read by a function, which
        is eleven wrong sections on a page whose whole purpose is to say
        where a setting is read. The tooltip table is the list of settings
        a user can see, which is exactly the claim being made.
        """
        if not isinstance(node, ast.Name) or node.id not in keyed:
            return False
        known = _tooltips()
        found = [key for key in sorted(keyed[node.id]) if key in known]
        # DYNAMIC, and marked so. The name is known; the READ still went
        # through a variable, which is what the suffix has always meant
        # here and what the API-link check reads it as.
        for key in found:
            out.append((key, form + "-dynamic"))
        return bool(found)

    def expand(joined: ast.JoinedStr, form: str) -> None:
        parts = [v.value for v in joined.values
                 if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        tail = "".join(parts)
        if not tail or any(c in tail for c in "./\\ %:"):
            return
        # An f-string over the object roles is FOUR concrete keys, not one
        # unresolvable read. This is where most of the dynamic 5% lives.
        for role in OBJECT_ROLES:
            out.append((f"{role}{tail}", form + "-dynamic"))

    if isinstance(node, ast.Subscript) and _is_settings(node.value, aliases):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            out.append((node.slice.value, "subscript"))
        elif isinstance(node.slice, ast.JoinedStr):
            expand(node.slice, "subscript")
        else:
            named(node.slice, "subscript")
    if isinstance(node, ast.Call):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in ("get", "setdefault")
                and node.args and _is_settings(f.value, aliases)):
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.append((a.value, "get"))
            elif isinstance(a, ast.JoinedStr):
                expand(a, "get")
            else:
                named(a, "get")
    return out


def analyse() -> dict:
    """Walk the package and return the flow data."""
    modules: Dict[str, ast.Module] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        try:
            modules[_module_name(path)] = ast.parse(
                path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

    defined: Set[str] = set()
    for name, tree in modules.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                defined.add(f"{name}.{node.name}")

    # BEFORE THE ANALYSIS PROPER: which functions hand a settings mapping
    # back to their caller. `_alias_targets` needs the answer for modules
    # it has not walked yet.
    helper_functions = _settings_helpers(modules)

    reads: Dict[str, List[dict]] = defaultdict(list)
    edges: List[dict] = []
    receivers: Set[str] = set()

    for module, tree in modules.items():
        imports: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    base = module.rsplit(".", node.level)[0] \
                        if node.level <= module.count(".") else "spacr"
                    target = f"{base}.{node.module}"
                else:
                    target = node.module
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{target}.{alias.name}"
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                imports.setdefault(node.name, f"{module}.{node.name}")

        # The module-level tables of setting names, and the constants a
        # read may go through. See `_module_tables`.
        tables = _module_tables(tree)
        constants = _module_constants(tree)

        # The helpers THIS module can reach, by the name it reaches them
        # under. Built from the same import table the call graph uses, so
        # a helper renamed on import is still recognised.
        local_helpers = {name for name, qual in imports.items()
                         if qual in helper_functions}

        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qual = f"{module}.{fn.name}"
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            takes = bool(params & SETTINGS_NAMES)
            if takes:
                receivers.add(qual)
            aliases = (_alias_targets(fn, local_helpers)
                       | (params & SETTINGS_NAMES))
            keyed = _keyed_names(fn, tables, constants)
            for node in ast.walk(fn):
                for key, form in _keys_from(node, aliases, keyed):
                    reads[key].append({"function": qual, "form": form,
                                       "line": node.lineno})
                if not isinstance(node, ast.Call):
                    continue
                passes = any(isinstance(a, ast.Name) and a.id in aliases
                             for a in node.args) or \
                    any(k.value is not None and isinstance(k.value, ast.Name)
                        and k.value.id in aliases for k in node.keywords)
                if not passes:
                    continue
                callee = node.func
                name = callee.id if isinstance(callee, ast.Name) else None
                target = imports.get(name) if name else None
                edges.append({
                    "caller": qual,
                    "callee": target if target in defined else None,
                    "raw": name or (callee.attr
                                    if isinstance(callee, ast.Attribute) else "?"),
                    "confidence": "RESOLVED" if target in defined
                    else "UNRESOLVED",
                    "line": node.lineno,
                })

    return {
        "version": 1,
        "reads": {k: v for k, v in sorted(reads.items())},
        "edges": edges,
        "receivers": sorted(receivers),
    }


@lru_cache(maxsize=1)
def _tooltips() -> Dict[str, str]:
    """``spacr.settings.tooltips``, read with `ast` rather than imported.

    IMPORTING IT WOULD HAVE BEEN ONE LINE and it broke the tool's central
    promise: everything here is pure `ast`, so a broken import cannot
    quietly empty the trees, and the docs build needs no torch, no Qt and
    no GPU. `spacr.settings` pulls all three. The suite has a test on
    that invariant and it is the test that caught this.

    A literal dict at module scope is exactly what `ast.literal_eval`
    exists for, so the cost of keeping the promise is a dozen lines.
    Entries whose value is not a plain string -- an f-string, a
    concatenation, a name -- are skipped rather than guessed at.
    """
    out: Dict[str, str] = {}
    # CATALOG FIRST, SETTINGS.PY SECOND, so settings.py wins ties.
    # `spacr.settings.tooltips` is post-processed at import: the organelle
    # entries are cloned across the four slots and some gain a trailing
    # "Read by ..." sentence. The literal in the file has neither, so a
    # read of it alone dropped 150 lines of help text off the page --
    # measured by diffing the page against the version that imported.
    # The EN i18n catalog holds the RESOLVED strings, clones and all,
    # which is what the GUI hovers -- but it drops the "(float) - " type
    # prefix that the literal carries and a reader wants. So the catalog
    # supplies the entries settings.py has no literal for, and the
    # literal wins wherever both have one.
    for path, name in ((ROOT / "spacr" / "qt" / "i18n_catalogs" / "en.py",
                        "SETTING_TOOLTIPS"),
                       (ROOT / "spacr" / "settings.py", "tooltips")):
        out.update(_literal_dict(path, name))
    return out


def _literal_dict(path: Path, name: str) -> Dict[str, str]:
    """One module-scope ``name = {...}`` of plain strings, read with `ast`.

    Entries whose key or value is not a string constant -- an f-string, a
    concatenation, a name -- are skipped rather than guessed at.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):                           # noqa: BLE001
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == name for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {k.value: v.value
                for k, v in zip(node.value.keys, node.value.values)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
                and isinstance(v, ast.Constant) and isinstance(v.value, str)}
    return {}


def _tooltip_for(key: str) -> str:
    """The setting's own help text, as one RST paragraph.

    Read from the same table the GUI hovers, so the page and the tooltip
    cannot drift: there is one sentence about a setting and both surfaces
    show it.

    Returns "" when the setting has no tooltip, which is not an error --
    the tree is still worth drawing, and an empty paragraph would put a
    blank line under the heading and say nothing.
    """
    text = str(_tooltips().get(key) or "").strip()
    if not text:
        return ""
    # ESCAPED, BECAUSE A TOOLTIP IS PROSE AND RST IS NOT. The tooltips
    # carry ``literals`` deliberately and those must survive, but a lone
    # `*` opens emphasis that never closes and a trailing `_` reads as a
    # reference to a target that does not exist -- both are warnings, and
    # `-W` makes them fatal. So the literals are lifted out, the rest is
    # escaped, and they are put back.
    text = " ".join(text.split())
    parts = re.split(r"(``[^`]*``)", text)
    for i, part in enumerate(parts):
        if part.startswith("``"):
            continue
        part = part.replace("*", r"\*")
        # ANY trailing underscore, not just one before whitespace. RST reads
        # `set_default_` as a reference to a target named "set_default", and
        # the tooltips are full of `set_default_*` and `organelle_`. The
        # first draft escaped `_` only before whitespace or end of string,
        # so `set_default_\*` -- with the star already escaped -- still
        # opened a reference and the build reported six unknown targets.
        part = re.sub(r"(?<=\w)_(?!\w)", r"\\_", part)
        parts[i] = part
    return "".join(parts)


def tree_for(data: dict, key: str, *, depth: int = 6) -> str:
    """The call tree below each entry point that leads to a read of ``key``.

    PRUNED TO BRANCHES THAT REACH A READER. Printing the whole propagation
    graph would be a call-graph dump, and the point is to answer "where does
    this setting go", not "what calls what".
    """
    readers = {hit["function"] for hit in data["reads"].get(key, [])
               if not _supplies_the_default(hit["function"])}
    if not readers:
        return f"{key}: read nowhere that static analysis can see"

    # ONE EDGE PER (caller, callee). A function that calls another three
    # times passes settings three times, and the raw edge list says so --
    # correctly, since it is a record of call SITES. A tree drawn from it
    # repeats the whole subtree three times, which is noise: the question is
    # where the setting goes, not how many times it is handed over.
    out_edges: Dict[str, List[dict]] = defaultdict(list)
    callees: Set[str] = set()
    seen_pairs: Set[Tuple[str, Optional[str]]] = set()
    for edge in data["edges"]:
        pair = (edge["caller"], edge["callee"] or edge["raw"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        out_edges[edge["caller"]].append(edge)
        if edge["callee"]:
            callees.add(edge["callee"])

    def reaches(node: str, seen: Set[str], left: int) -> bool:
        if node in readers:
            return True
        if left <= 0 or node in seen:
            return False
        seen = seen | {node}
        return any(e["callee"] and reaches(e["callee"], seen, left - 1)
                   for e in out_edges.get(node, []))

    roots = sorted(f for f in set(out_edges) | readers
                   if f not in callees and reaches(f, set(), depth))
    lines = [f"{key}"]

    def walk(node: str, prefix: str, seen: Set[str], left: int) -> None:
        mark = READS_MARK if node in readers else ""
        lines.append(f"{prefix}{node}{mark}")
        if left <= 0 or node in seen:
            return
        seen = seen | {node}
        kids = [e for e in out_edges.get(node, [])
                if e["callee"] and e["callee"] not in seen
                and reaches(e["callee"], seen, left - 1)]
        unresolved = [e for e in out_edges.get(node, [])
                      if not e["callee"]]
        for edge in sorted(kids, key=lambda e: e["callee"]):
            walk(edge["callee"], prefix + "    ", seen, left - 1)
        for edge in unresolved[:2]:
            lines.append(f"{prefix}    {edge['raw']}(...)  [UNRESOLVED]")

    for root in roots:
        walk(root, "  ", set(), depth)
    return "\n".join(lines)


#: What `tree_for` appends to a node that reads the setting.
READS_MARK = "  <-- reads it"

#: The RST label that opens each setting's section, and the anchor the
#: GUI links to. One constant, because the page and the index are
#: written in the same run and must agree.
SECTION_MARK = ".. _setting-flow-"

#: Function names that supply a setting's default rather than act on it.
_DEFAULT_SUPPLIER = re.compile(
    r"^spacr\.settings\.(?:set_default_|set_.*_defaults$|get_.*_default_settings$)"
    r"|^spacr\.settings\..*_defaults$")


def _supplies_the_default(qualname: str) -> bool:
    """Whether this function hands the setting its default value.

    NOT A FLOW WORTH DRAWING, and it was 30% of the page. Every setting
    has a defaults-setter by construction -- that is what makes it a
    setting -- so `set_default_settings_preprocess_generate_masks`
    appeared as a reader 803 times across 1057 sections, 636 of them as
    a leaf. "This setting is read by the function that defines its
    default" tells a reader nothing they did not know from the setting
    existing, while costing a third of the cross-references Sphinx has
    to resolve.

    Dropping it is what the page is FOR rather than a concession to the
    build clock: the question a reader arrives with is which function
    acts on the value, and the defaults table was burying that answer
    under itself.
    """
    return bool(_DEFAULT_SUPPLIER.match(qualname))


def _link(qualname: str) -> str:
    """A clickable reference to a function's API page.

    ``:py:func:`~name``` renders as the short name and links to the full one,
    which is what keeps a deep tree readable: the reader wants
    ``_normalize_img_batch``, not ``spacr.io._normalize_img_batch``, on every
    line. AutoAPI publishes the target, and a private function has no page --
    so those are left as plain text rather than emitted as a link Sphinx will
    warn about and the reader will find broken.
    """
    leaf = qualname.rsplit(".", 1)[-1]
    if leaf.startswith("_"):
        return f"``{leaf}``"
    return f":py:func:`~{qualname}`"


def rst_for(data: dict, keys: Optional[List[str]] = None) -> str:
    """The whole flow, as one Sphinx page with every node linked.

    ONE PAGE, NOT 1058. A page per setting would be a file per setting and a
    toctree nobody can scan; a reader arriving from a setting's tooltip wants
    that setting's section, and an anchor per section gives them one while
    keeping the rest browsable.
    """
    reads = data["reads"]
    keys = sorted(keys if keys is not None else reads)
    # NO TITLE. This is an include FRAGMENT -- docs/source/settings_flow.rst
    # owns the heading and the prose around it -- and a second title inside
    # an included file gives Sphinx two documents' worth of structure in one
    # page and a duplicate-label warning that `-W` makes fatal.
    lines = [
        "Generated by ``tools/settings_flow.py`` -- do not edit.",
        "",
        "Each setting lists the call paths that carry it from an entry point",
        "to the functions that read it. Branches that reach no reader are not",
        "drawn. A step marked ``[UNRESOLVED]`` is a call static analysis",
        "cannot follow -- ``getattr``, a dispatch table, a Qt signal, a",
        "callback passed as a value -- and is shown rather than dropped,",
        "because a tree that hides what it could not follow looks complete.",
        "",
    ]
    drawn = 0
    for key in keys:
        if not reads.get(key):
            continue
        tree = tree_for(data, key)
        body = tree.split("\n")[1:]
        if not body:
            continue
        drawn += 1
        lines += [f"{SECTION_MARK}{key}:", "", key, "-" * len(key), ""]
        # WHAT THE SETTING IS, before where it goes. Asked for on
        # 2026-09-08: "when they click the setting itself they should get
        # the tool tip text". The tree says which functions carry the
        # value; without the sentence a reader has to already know what
        # they were looking for to recognise it.
        hint = _tooltip_for(key)
        if hint:
            lines += [hint, ""]
        # A LINE BLOCK, NOT A LITERAL BLOCK. `::` would be simpler and would
        # render the tree as preformatted text -- in which nothing is
        # clickable, and "all branches should be clickable" is the whole
        # request. A line block keeps one source line per output line AND
        # interprets roles, so each node becomes a link to its API page.
        # Indentation is non-breaking spaces because RST collapses ordinary
        # leading whitespace inside a line block and the shape carries the
        # meaning.
        for row in body:
            depth = (len(row) - len(row.lstrip())) // 4
            text = row.strip()
            reader = text.endswith(READS_MARK)
            if reader:
                text = text[: -len(READS_MARK)].strip()
            pad = "\u00a0" * (depth * 4)
            if text.endswith("[UNRESOLVED]"):
                lines.append(f"| {pad}``{text}``")
            elif reader:
                lines.append(f"| {pad}{_link(text)} **-- reads it**")
            else:
                lines.append(f"| {pad}{_link(text)}")
        lines.append("")
        readers = sorted({h["function"] for h in reads[key]
                          if not _supplies_the_default(h["function"])})
        lines.append("Read by " + ", ".join(_link(r) for r in readers) + ".")
        lines.append("")
    lines.insert(2, f"{drawn} settings, of {len(reads)} read anywhere.")
    lines.insert(3, "")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", help="print the tree for one setting")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--rst", action="store_true",
                        help="also write the Sphinx page")
    args = parser.parse_args()

    data = analyse()
    if args.key:
        print(tree_for(data, args.key))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    resolved = sum(1 for e in data["edges"] if e["confidence"] == "RESOLVED")
    print(f"settings read           {len(data['reads'])}")
    print(f"functions taking them   {len(data['receivers'])}")
    print(f"propagation edges       {len(data['edges'])} "
          f"({resolved} resolved, {len(data['edges']) - resolved} unresolved)")
    print(f"written: {args.out.relative_to(ROOT)}")
    if args.rst:
        RST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        rst = rst_for(data)
        RST_OUTPUT.write_text(rst, encoding="utf-8")
        print(f"written: {RST_OUTPUT.relative_to(ROOT)} "
              f"({rst.count(SECTION_MARK)} sections, "
              f"{rst.count(':py:func:')} cross-references)")
        INDEX_OUTPUT.write_text(_index_module(rst), encoding="utf-8")
        print(f"written: {INDEX_OUTPUT.relative_to(ROOT)}")
    return 0


def _index_module(rst: str) -> str:
    """The generated module listing every key the page has a section for.

    COUNTED FROM THE PAGE, not from the analysis, so the two cannot
    disagree. A key here that the page has no section for would send a
    reader to an anchor that does not exist -- the exact defect
    instruction 383 item 3 closed for the API links, and it would be
    careless to reintroduce it one file over.
    """
    keys = sorted(re.findall(rf"^{re.escape(SECTION_MARK)}(.+):$", rst,
                             re.M))
    body = "\n".join(f"    {k!r}," for k in keys)
    return (
        '''"""Settings the flow page has a section for. Generated -- do not edit.

Written by ``tools/settings_flow.py --rst``, counted from the page it
writes in the same run.

The settings panel offers a link here for a setting whose API target is a
module page with no anchor to aim at -- 245 of 796 links, where the
reader lands on a 4,000-line module that may not mention the setting at
all. The flow page names it, shows its help text and lists every function
that reads it, including the private ones an API page cannot address.
"""

SETTINGS_WITH_A_FLOW_SECTION = frozenset({
''' + body + "\n})\n")


if __name__ == "__main__":
    raise SystemExit(main())
