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
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "spacr"
OUTPUT = ROOT / "docs" / "settings_flow.json"

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
    if isinstance(node, ast.Name):
        return node.id in SETTINGS_NAMES or node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr in SETTINGS_NAMES
    return False


def _alias_targets(fn: ast.AST) -> Set[str]:
    """Local names bound to the settings mapping inside ``fn``."""
    found: Set[str] = set()
    for node in ast.walk(fn):
        value = getattr(node, "value", None)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or value is None:
            continue
        ok = (isinstance(value, ast.Name) and value.id in SETTINGS_NAMES)
        if isinstance(value, ast.Call):
            f = value.func
            if isinstance(f, ast.Attribute) and f.attr in ("copy", "deepcopy"):
                ok = _is_settings(f.value, set())
            elif isinstance(f, ast.Name) and f.id in ("dict", "deepcopy"):
                ok = any(_is_settings(a, set()) for a in value.args)
        if not ok:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        found |= {t.id for t in targets if isinstance(t, ast.Name)}
    return found


def _keys_from(node, aliases: Set[str]) -> List[Tuple[str, str]]:
    """``(key, form)`` pairs a subscript or ``.get`` call reads."""
    out: List[Tuple[str, str]] = []

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
    if isinstance(node, ast.Call):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in ("get", "setdefault")
                and node.args and _is_settings(f.value, aliases)):
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.append((a.value, "get"))
            elif isinstance(a, ast.JoinedStr):
                expand(a, "get")
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

        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qual = f"{module}.{fn.name}"
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            takes = bool(params & SETTINGS_NAMES)
            if takes:
                receivers.add(qual)
            aliases = _alias_targets(fn) | (params & SETTINGS_NAMES)
            for node in ast.walk(fn):
                for key, form in _keys_from(node, aliases):
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


def tree_for(data: dict, key: str, *, depth: int = 6) -> str:
    """The call tree below each entry point that leads to a read of ``key``.

    PRUNED TO BRANCHES THAT REACH A READER. Printing the whole propagation
    graph would be a call-graph dump, and the point is to answer "where does
    this setting go", not "what calls what".
    """
    readers = {hit["function"] for hit in data["reads"].get(key, [])}
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
        mark = "  <-- reads it" if node in readers else ""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", help="print the tree for one setting")
    parser.add_argument("--out", type=Path, default=OUTPUT)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
