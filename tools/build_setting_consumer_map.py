"""Find, for every user-facing setting, the function that actually READS it.

Instruction 336. A setting's API link is built from the SCREEN's app_key, so
every row on the Mask panel points at Mask's entry point whether the value is
read there or twelve calls down. Fixing the link text is not enough on its
own: a consumer that is a closure has no importable address, so the map has to
say which consumers are addressable before anything can link to them.

This walks the AST rather than grepping, because the enclosing function is the
answer and only a parse knows it. It records every read of the form
``settings['key']``, ``settings.get('key')`` and a ``key=`` keyword argument,
each with the qualified name of the function containing it and whether that
function is nested.

Output is ``docs/setting_consumers.json``, committed so the next audit is a
diff rather than a rerun.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "spacr"


#: ``descriptions`` holds the thirteen MODULE blurbs, not the settings. The
#: per-setting text -- and the 22 ``API:`` overrides this instruction counts --
#: is ``tooltips``; ``expected_types`` carries the same keys and is unioned in
#: so a setting documented in one and not the other is still audited.
KEY_TABLES = ("tooltips", "expected_types")


def setting_keys() -> set[str]:
    """The user-facing setting keys, from ``tooltips`` and ``expected_types``."""
    tree = ast.parse((PKG / "settings.py").read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(getattr(t, "id", "") in KEY_TABLES for t in node.targets):
            continue
        keys |= {k.value for k in node.value.keys
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return keys


#: Names a settings mapping is plausibly bound to at a read site.
#:
#: WHY THE OBJECT HAS TO BE CHECKED AT ALL. Without this the visitor recorded
#: EVERY string subscript on EVERY object as a settings read, so
#: ``paths[f"min_{threshold}"] = path`` -- a local dict of CSV filenames in
#: ``guide_permutation.save_guide_permutation_results`` -- made that function
#: the published API target for thirty settings, including
#: ``nucleus_min_area``. A reader clicking a size setting to learn where it is
#: used arrived at a guide-permutation CSV writer, which reads no settings at
#: all.
SETTINGS_NAMES = frozenset({
    "settings", "setting", "cfg", "config", "conf", "opts", "options",
    "params", "parameters", "defaults", "kwargs",
})


def _settings_alias(value) -> bool:
    """Whether an assigned VALUE is the settings mapping under a new name.

    ``out = dict(settings)``, ``local = settings.copy()``, ``s = settings``.
    Without this the object test below is too strict and loses real reads:
    ``spacr.organelle_types.apply_preset`` copies the mapping to ``out`` and
    then reads ``out.get("organelle_type")``, which is the ONLY public
    consumer of that setting -- drop it and the setting's link falls back to
    a module page.

    An empty literal is deliberately not an alias, which is what keeps
    ``paths: dict[str, Path] = {}`` out.
    """
    if isinstance(value, ast.Name) and value.id in SETTINGS_NAMES:
        return True
    if isinstance(value, ast.Call):
        f = value.func
        # dict(settings) / settings.copy() / deepcopy(settings)
        if isinstance(f, ast.Attribute) and f.attr in ("copy", "deepcopy"):
            return _is_settings_mapping(f.value)
        if isinstance(f, ast.Name) and f.id in ("dict", "deepcopy", "copy"):
            return any(_is_settings_mapping(a) for a in value.args)
    return False


def _is_settings_mapping(node, aliases=frozenset()) -> bool:
    """Whether ``node`` plausibly evaluates to the settings mapping.

    A bare name (``settings[...]``), an attribute whose final component is one
    (``self.settings[...]``), or a local ``aliases`` name assigned from one.

    Deliberately a NAME test rather than type inference: the alternative is
    following assignments across the whole package, and the cost of being
    wrong is a published link pointing at the wrong function. A conservative
    rule that occasionally misses a read is the right trade -- a missed read
    falls back to the module page, a wrong one sends the reader somewhere
    unrelated and says nothing about it.
    """
    if isinstance(node, ast.Name):
        return node.id in SETTINGS_NAMES or node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr in SETTINGS_NAMES
    return False


class Reads(ast.NodeVisitor):
    """Collect setting reads with the qualified function that encloses them."""

    def __init__(self, keys: set[str], module: str) -> None:
        self.keys = keys
        self.module = module
        #: ``(name, kind)`` per enclosing scope. The KIND matters: a method is
        #: ``Class.method`` and perfectly addressable, while a closure is a
        #: function inside a function and has no importable name at all. Both
        #: sit two scopes deep, so counting depth alone cannot tell them apart
        #: -- and which of the two this is decides 336's route.
        self.stack: list[tuple[str, str]] = []
        self.hits: list[dict] = []
        #: Local names currently bound to the settings mapping.
        self.aliases: frozenset = frozenset()

    def _scoped(self, node, name, kind):
        self.stack.append((name, kind))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):        # noqa: N802 - ast naming
        # Names this function binds to the settings mapping, so a read
        # through the copy counts as a read. Collected before descending,
        # because the copy is usually made on the first line and read after.
        before = self.aliases
        found = set(before)
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and _settings_alias(child.value):
                found |= {t.id for t in child.targets if isinstance(t, ast.Name)}
            elif (isinstance(child, ast.AnnAssign) and child.value is not None
                    and _settings_alias(child.value)
                    and isinstance(child.target, ast.Name)):
                found.add(child.target.id)
        self.aliases = frozenset(found)
        try:
            self._scoped(node, node.name, "function")
        finally:
            self.aliases = before

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):           # noqa: N802 - ast naming
        self._scoped(node, node.name, "class")

    def _record(self, key, node, form):
        if key not in self.keys:
            return
        # A closure is what Sphinx cannot address: more than one FUNCTION
        # scope on the stack. Class scopes do not count -- `Class.method` is
        # importable and documents fine.
        functions = sum(1 for _, kind in self.stack if kind == "function")
        self.hits.append({
            "key": key,
            "module": self.module,
            "qualname": ".".join(n for n, _ in self.stack) or "<module>",
            "nested": functions > 1,
            "function_depth": functions,
            "form": form,
            "line": node.lineno,
        })

    def _record_dynamic(self, joined, node, form):
        """Match ``settings.get(f'{object_type}_area_multiplier')``.

        Whole families of keys are built from a prefix at runtime -- every
        organelle repeats the same suffix -- so a literal-only matcher reports
        them as read by nothing at all. That was 110 of 752 settings before
        this, and spot-checking three of them found the f-string rather than a
        dead setting. The static parts of the f-string are matched as
        suffix/infix against the known keys; a hit is recorded as ``dynamic``
        so the map never claims a literal read it did not see.
        """
        parts = [v.value for v in joined.values
                 if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        for part in parts:
            if len(part) < 4:            # too short to identify a key
                continue
            # A FRAGMENT THAT IS NOT PART OF A KEY NAME. `_min_` matched every
            # setting containing it, so an f-string building
            # `{prefix}_min_{n}_wells.csv` claimed thirty of them. A key never
            # contains a dot, a slash or a space, so a fragment that does is
            # building a filename or a message, not a settings key.
            if any(c in part for c in "./\\ %:"):
                continue
            for key in self.keys:
                # SUFFIX OR PREFIX, NOT A BARE INFIX. `{object_type}_min_area`
                # legitimately identifies `cell_min_area` by its tail; `_min_`
                # floating in the middle of a filename identifies nothing.
                if key.endswith(part) or key.startswith(part):
                    self._record(key, node, form + "-dynamic")

    def visit_Subscript(self, node):          # noqa: N802 - ast naming
        if _is_settings_mapping(node.value, self.aliases):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self._record(node.slice.value, node, "subscript")
            elif isinstance(node.slice, ast.JoinedStr):
                self._record_dynamic(node.slice, node, "subscript")
        self.generic_visit(node)

    def visit_Call(self, node):               # noqa: N802 - ast naming
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in ("get", "setdefault")
                and node.args and _is_settings_mapping(f.value, self.aliases)):
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                self._record(a.value, node, "get")
            elif isinstance(a, ast.JoinedStr):
                self._record_dynamic(a, node, "get")
        for kw in node.keywords:
            if kw.arg:
                self._record(kw.arg, node, "keyword")
        self.generic_visit(node)


def main() -> int:
    keys = setting_keys()
    if not keys:
        print("no setting keys found", file=sys.stderr)
        return 1
    hits: list[dict] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        module = rel[:-3].replace("/", ".")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        v = Reads(keys, module)
        v.visit(tree)
        hits.extend(v.hits)

    by_key: dict[str, list[dict]] = {}
    for h in hits:
        by_key.setdefault(h["key"], []).append(h)

    addressable = {k: v for k, v in by_key.items()
                   if any(not h["nested"] for h in v)}
    nested_only = {k: v for k, v in by_key.items()
                   if v and all(h["nested"] for h in v)}
    unread = sorted(keys - set(by_key))

    out = {
        "setting_count": len(keys),
        "with_any_consumer": len(by_key),
        "with_addressable_consumer": len(addressable),
        "nested_consumers_only": len(nested_only),
        "no_consumer_found": len(unread),
        "consumers": {k: sorted(v, key=lambda h: (h["module"], h["line"]))
                      for k, v in sorted(by_key.items())},
        "unread_keys": unread,
    }
    out["targets"] = resolve_targets(out["consumers"])
    out["target_count"] = len(out["targets"])
    out["exact_targets"] = sum(1 for t in out["targets"].values() if t["exact"])
    dest = ROOT / "docs" / "setting_consumers.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(f"settings                       {len(keys)}")
    print(f"  with any consumer            {len(by_key)}")
    print(f"  with an ADDRESSABLE consumer {len(addressable)}")
    print(f"  nested consumers only        {len(nested_only)}")
    print(f"  no consumer found            {len(unread)}")
    print(f"  resolved API targets         {out['target_count']}"
          f" ({out['exact_targets']} exact, "
          f"{out['target_count'] - out['exact_targets']} to an ancestor)")
    # A generated runtime table, so the GUI does not read docs/ at import.
    # Mirrors how the localized catalogs are generated into the package.
    lines = [
        '"""Where each setting is actually READ. Generated -- do not edit.',
        '',
        'Written by ``tools/build_setting_consumer_map.py`` for instruction 336.',
        'The API link used to be built from the SCREEN\'s app_key, so every row',
        'on a panel pointed at that module\'s entry point whether the value was',
        'read there or twelve calls down. These targets come from an AST walk of',
        'the package instead.',
        '',
        '``exact`` is False where the only consumer is a closure: Sphinx cannot',
        'address one, so the link aims at the enclosing function that it can.',
        '"""',
        '',
        '#: ``key -> (module, symbol, exact)``',
        'SETTING_API_TARGETS = {',
    ]
    for key, t in sorted(out["targets"].items()):
        lines.append(f'    {key!r}: ({t["module"]!r}, {t["symbol"]!r}, {t["exact"]!r}),')
    lines.append("}")
    gen = ROOT / "spacr" / "qt" / "screens" / "setting_api_targets.py"
    gen.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {gen.relative_to(ROOT)}")

    print(f"written: {dest.relative_to(ROOT)}")
    return 0


#: Modules that DISPLAY a setting rather than consume it. A link into the Qt
#: layer answers "which widget shows this", which is not the question the
#: reader asked -- they are looking at the widget already. `settings` itself
#: is the declaration site for the same reason.
DISPLAY_ONLY_PREFIXES = ("spacr.qt.", "spacr.settings")


def _rank(hit: dict) -> tuple:
    """Order candidate consumers best-first.

    A literal read is stronger evidence than a keyword argument that merely
    shares the setting's name, and a function is a better answer than a
    closure that cannot be linked to at all. Ties break on the shallowest
    function and then alphabetically, so the committed map is stable across
    runs and a diff means the code moved rather than the sort did.
    """
    form_rank = {"subscript": 0, "get": 0,
                 "subscript-dynamic": 1, "get-dynamic": 1}.get(hit["form"], 2)
    # AutoAPI runs without `private-members`, so an underscore function has no
    # published anchor. Preferring a public consumer keeps the link landing on
    # a heading that exists; where only a private one reads the setting the
    # resolver drops to the module page rather than emitting a dead fragment.
    private = hit["qualname"].rsplit(".", 1)[-1].startswith("_")
    return (hit["nested"], private, form_rank, hit["function_depth"],
            hit["module"], hit["line"])


def resolve_targets(consumers: dict) -> dict:
    """Pick one API target per setting, or none where there is nothing to aim at."""
    targets = {}
    for key, hits in consumers.items():
        usable = [h for h in hits
                  if not h["module"].startswith(DISPLAY_ONLY_PREFIXES)]
        if not usable:
            continue
        best = sorted(usable, key=_rank)[0]
        if best["nested"]:
            # A closure has no importable name. Aim at the ancestor that does
            # and let the map say the read is deeper, rather than inventing an
            # address Sphinx will never publish (instruction 336, route 2).
            outer = best["qualname"].split(".")[0]
            targets[key] = {"module": best["module"], "symbol": outer,
                            "exact": False}
        elif best["qualname"].rsplit(".", 1)[-1].startswith("_"):
            # Read by a private function. The module page is honest; an anchor
            # to an undocumented symbol would land the reader at the top of the
            # page having promised them a heading.
            targets[key] = {"module": best["module"], "symbol": "",
                            "exact": False}
        else:
            targets[key] = {"module": best["module"],
                            "symbol": best["qualname"], "exact": True}
    return targets


if __name__ == "__main__":
    raise SystemExit(main())
