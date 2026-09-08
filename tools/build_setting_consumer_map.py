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
    """The user-facing setting keys.

    ``settings.py``'s ``tooltips`` and ``expected_types``, PLUS the EN i18n
    catalog's ``SETTING_TOOLTIPS``.

    WHY THE CATALOG IS NEEDED. A module may declare its own settings
    locally: `hit_investigation_default_settings` builds a `tips` dict
    inside the function body, so 17 `hit_*` settings appear in the
    Investigate Hit panel and in no module-scope table. This function
    scanned only module scope in one file, so it did not know they were
    settings, so nothing that read them counted as a read, so their API
    links had no target. They were not rare or obscure -- they are most
    of one module's panel.

    The catalog is the right second source because it is the list of
    settings the GUI actually shows: it is generated from the resolved
    runtime tables, so a setting a user can see is in it by
    construction, wherever its tooltip was declared.
    """
    keys: set[str] = set()
    tree = ast.parse((PKG / "settings.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(getattr(t, "id", "") in KEY_TABLES for t in node.targets):
            continue
        keys |= {k.value for k in node.value.keys
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    catalog = PKG / "qt" / "i18n_catalogs" / "en.py"
    try:
        tree = ast.parse(catalog.read_text(encoding="utf-8"))
    except OSError:                                          # noqa: BLE001
        return keys
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(getattr(t, "id", "") == "SETTING_TOOLTIPS"
                   for t in node.targets):
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


def _settings_key_of(node, aliases=frozenset()):
    """The settings key ``node`` evaluates to, or None.

    Recognises the two spellings a value can have when a caller passes a
    setting on: ``settings['key']`` and ``settings.get('key')``. Anything
    else -- a literal, a computed expression, another variable -- is not a
    setting being handed along and is not evidence of one.
    """
    if isinstance(node, ast.Subscript) and _is_settings_mapping(
            node.value, aliases):
        if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str):
            return node.slice.value
        return None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "setdefault")
            and node.args and _is_settings_mapping(node.func.value, aliases)):
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


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
    if isinstance(node, ast.BoolOp):
        # `settings or {}`, the defensive idiom for a None default, is the
        # settings mapping whenever either side is. Missing it hid every
        # read behind `dict(settings or {})` -- 17 `hit_*` settings in
        # `hit_investigation` alone, each of which then had no anchor for
        # its API link to aim at. Found by asking why the settings-flow
        # page had no section for them and grepping for the key.
        return any(_is_settings_mapping(v, aliases) for v in node.values)
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
            # A KEYWORD IS ONLY EVIDENCE IF ITS VALUE CAME FROM SETTINGS.
            #
            # This used to record every `f(x=...)` as a read of the setting
            # `x`, whatever the value was. It was the largest form in the
            # map -- 2,757 of 6,956 hits, 40% -- and most of it said
            # nothing: `test_cellpose_model` calls something with
            # `diameter=30`, a hardcoded literal, and that was recorded as
            # a read of the `diameter` setting, so the API link for
            # `diameter` in the Plaque assay pointed at it. Worse,
            # `flow_threshold=settings['FT']` was recorded as a read of
            # `flow_threshold` when the value read is `FT`.
            #
            # What the pass-along relationship actually looks like is
            # `f(x=settings['x'])` or `f(x=settings.get('x'))`: the value
            # is the setting. That is what is recorded now, and it is
            # recorded for the key the VALUE names, not the parameter.
            if not kw.arg:
                continue
            passed = _settings_key_of(kw.value, self.aliases)
            if passed is not None:
                self._record(passed, node, "keyword")
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
        'Written by ``tools/build_setting_consumer_map.py``.',
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
    wanted = _app_api_modules()
    by_module = resolve_targets_by_module(out["consumers"], wanted)
    lines += [
        "",
        "#: ``key -> {module: (symbol, exact)}``, for the modules an app's",
        "#: help actually links to.",
        "#:",
        "#: Consulted FIRST, by app. The table above answers \"where is this",
        "#: setting read\" with one module for the whole package; this one",
        "#: answers it for the panel the reader is looking at, which is a",
        "#: different question and the one they are asking. A setting the",
        "#: app's own module does not read falls through to the single",
        "#: answer above, which is what it had before.",
        "SETTING_API_TARGETS_BY_MODULE = {",
    ]
    for key, rows in sorted(by_module.items()):
        inner = ", ".join(f"{m!r}: ({s!r}, {e!r})"
                          for m, (s, e) in sorted(rows.items()))
        lines.append(f"    {key!r}: {{{inner}}},")
    lines.append("}")
    print(f"  per-module rows              "
          f"{sum(len(v) for v in by_module.values())} over "
          f"{len(by_module)} setting(s), {len(wanted)} app module(s)")
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


def _is_unrendered_module(module: str) -> bool:
    """Whether AutoAPI publishes no page for ``module``.

    A module whose own name begins with an underscore is private and is
    not rendered, so a link into it is a 404 rather than a page with no
    anchor -- which is a worse failure than the private-FUNCTION case
    handled in `_rank`, where the module page still exists and the
    resolver can honestly drop to it.

    `spacr._v1_v2_bridge` is the one that made this necessary. It really
    does read `channels`, so it ranked first on the evidence and four
    modules' tooltips pointed their API word at a page that is not
    published.
    """
    return any(part.startswith("_") for part in module.split(".")[1:])


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


def _app_api_modules() -> set:
    """Modules that some app's help links to, from the live GUI table.

    The per-module table below is restricted to these. Every module that
    reads a setting would be 2,699 rows answering a question nobody asks:
    the only module worth preferring for a given panel is the one that
    panel's help already points at.
    """
    try:
        # THE TABLE IS BUILT LAZILY, so importing settings_model alone
        # reads it half-filled: 41 of 65 apps, and the 24 missing ones
        # silently lost their per-module rows. `investigate_hit` was one,
        # so every setting in that panel fell back to the key-only
        # answer -- `verbose` in Investigate Hit pointed at
        # `core.preprocess_generate_masks`. Importing the app list is
        # what populates it.
        from spacr.qt import app as _app                     # noqa: F401
        from spacr.qt.screens.settings_model import _APP_API_MODULE
    except Exception:                                        # noqa: BLE001
        return set()
    # NORMALISED. That table stores a DOC PATH -- "core", "qt/screens/pca" --
    # because it is used to build a URL, and the consumer hits carry an
    # import path, "spacr.core". Comparing them raw matches nothing, which
    # is a table of zero rows that still generates and still imports.
    return {"spacr." + str(m).replace("/", ".")
            for m in _APP_API_MODULE.values() if m}


def resolve_targets_by_module(consumers: dict, wanted: set) -> dict:
    """Best target per setting WITHIN each app module that reads it.

    THE MAP ABOVE IS KEYED ON THE SETTING ALONE, which is the defect this
    exists for. `src` is shown in 41 panels and had one destination for
    all of them -- `annotation_dataset`, correct for at most one. The
    function that picks it is not told which app is asking, and no better
    ranking can fix that.

    So the same ranking runs again, once per module, and the caller
    prefers the row for the app it is drawing. A setting that module does
    not read has no row and falls through to the single answer.
    """
    by_module = {}
    for key, hits in consumers.items():
        usable = [h for h in hits
                  if h["module"] in wanted
                  and not h["module"].startswith(DISPLAY_ONLY_PREFIXES)
                  and not _is_unrendered_module(h["module"])]
        if not usable:
            continue
        grouped = {}
        for hit in usable:
            grouped.setdefault(hit["module"], []).append(hit)
        rows = {}
        for module, module_hits in grouped.items():
            best = sorted(module_hits, key=_rank)[0]
            leaf = best["qualname"].rsplit(".", 1)[-1]
            if best["nested"]:
                outer = best["qualname"].split(".")[0]
                symbol = "" if outer.startswith("_") else outer
                rows[module] = (symbol, False)
            elif leaf.startswith("_"):
                rows[module] = ("", False)
            else:
                rows[module] = (best["qualname"], True)
        if rows:
            by_module[key] = rows
    return by_module


def resolve_targets(consumers: dict) -> dict:
    """Pick one API target per setting, or none where there is nothing to aim at."""
    targets = {}
    for key, hits in consumers.items():
        usable = [h for h in hits
                  if not h["module"].startswith(DISPLAY_ONLY_PREFIXES)
                  and not _is_unrendered_module(h["module"])]
        if not usable:
            continue
        best = sorted(usable, key=_rank)[0]
        if best["nested"]:
            # A closure has no importable name. Aim at the ancestor that does
            # and let the map say the read is deeper, rather than inventing an
            # address Sphinx will never publish (instruction 336, route 2).
            #
            # UNLESS THE ANCESTOR IS PRIVATE TOO. The branch below already
            # drops the anchor for a private consumer, and this one did not
            # ask -- so a closure inside `_morphological_measurements` was
            # given an anchor to `_morphological_measurements`, which AutoAPI
            # does not publish. That is the failure the branch below exists to
            # avoid, arrived at by the other route: the browser ignores the
            # fragment silently and the reader lands at the top of the module
            # believing they were taken to a heading. It cost four links --
            # `object_distance_maxima` and `object_distance_intensity`, each
            # in Measure and in External Masks.
            outer = best["qualname"].split(".")[0]
            targets[key] = {"module": best["module"],
                            "symbol": "" if outer.startswith("_") else outer,
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
