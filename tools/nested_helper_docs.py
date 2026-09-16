"""Source-only inventory shared by nested-helper rendering and translation.

Feature 411 includes functions inside another function, even through control
flow or a local class. It does not publish private top-level functions. Nothing
is imported from the application, and source failures are errors, not omissions.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable


# This is the one rollout switch for both rendering and extraction. Keep it
# empty until the feature-411 Phase B catalog handoff and a reviewed first slice.
ENABLED_MODULES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class HelperDefinition:
    """One lexical definition, including its source location and signature."""

    module: str
    parent_id: str
    name: str
    qualified_key: str
    signature: str
    docstring: str
    path: str
    lineno: int
    end_lineno: int
    is_async: bool
    ignored_by: tuple[str, ...]


@dataclass(frozen=True)
class HelperEntry:
    """One anchor and translation document for all definitions of a local name.

Conditional/repeated definitions retain every signature and source location.
Different docstrings are concatenated in source order; identical docstrings
are included once. No definition wins merely because it was visited first.
"""

    qualified_key: str
    module: str
    parent_id: str
    definitions: tuple[HelperDefinition, ...]

    @property
    def docstring(self) -> str:
        """Return all distinct source documents in their lexical order."""
        return "\n\n".join(dict.fromkeys(
            definition.docstring for definition in self.definitions
            if definition.docstring
        ))

    @property
    def signatures(self) -> tuple[str, ...]:
        """Keep every different callable signature, without evaluating defaults."""
        return tuple(dict.fromkeys(
            definition.signature for definition in self.definitions
        ))

    @property
    def relative_signatures(self) -> tuple[str, ...]:
        """Qualify signatures within the module, including all lexical parents."""
        parent = self.parent_id.removeprefix(self.module + ".")
        return tuple(f"{parent}.{signature}" for signature in self.signatures)

    @property
    def is_async(self) -> bool:
        """Whether every alternative is an async definition."""
        return all(definition.is_async for definition in self.definitions)


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def inventory(
    root: Path, *, ignore_patterns: Iterable[str],
) -> tuple[HelperDefinition, ...]:
    """Enumerate every nested def/async def, retaining explicit ignore reasons.

Lambdas are expressions, not named helper definitions. A class is not a scope
barrier: its methods count when that class is inside an enclosing function.
Private helpers and helpers inside private parents are included. Neither an
absent package nor an unreadable/unparseable source yields a false empty pass.
"""
    root = Path(root).resolve()
    package = root / "spacr"
    if not package.is_dir():
        raise ValueError(f"No spacr source package under {root}")
    patterns = tuple(ignore_patterns)
    definitions: list[HelperDefinition] = []
    for path in sorted(package.rglob("*.py")):
        if not path.resolve().is_relative_to(package.resolve()):
            raise ValueError(f"Source escapes the package: {path}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(path, root)
        parents = {
            child: node for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        ignored_by = tuple(
            pattern for pattern in patterns
            if fnmatch.fnmatch(path.as_posix(), pattern)
        )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            enclosing = parents.get(node)
            scopes: list[str] = []
            has_enclosing_function = False
            while enclosing is not None:
                if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    has_enclosing_function = True
                if isinstance(enclosing, (
                    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
                )):
                    scopes.append(enclosing.name)
                enclosing = parents.get(enclosing)
            if not has_enclosing_function:
                continue
            parent_id = ".".join([module, *reversed(scopes)])
            signature = f"{node.name}({ast.unparse(node.args)})"
            if node.returns is not None:
                signature += f" -> {ast.unparse(node.returns)}"
            definitions.append(HelperDefinition(
                module=module,
                parent_id=parent_id,
                name=node.name,
                qualified_key=f"{parent_id}.{node.name}",
                signature=signature,
                docstring=inspect.cleandoc(ast.get_docstring(node, clean=False) or "").strip(),
                path=path.relative_to(root).as_posix(),
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                ignored_by=ignored_by,
            ))
    return tuple(sorted(definitions, key=lambda item: (item.path, item.lineno)))


def entries(
    definitions: Iterable[HelperDefinition], *, modules: Iterable[str] | None = None,
) -> tuple[HelperEntry, ...]:
    """Group documented, non-ignored definitions into canonical anchor entries.

``modules=None`` means all eligible modules for measurement. Rendering and
translation must pass ``ENABLED_MODULES`` explicitly, including when empty.
An enabled module containing an undocumented nested definition fails closed.
"""
    selected = None if modules is None else frozenset(modules)
    grouped: dict[str, list[HelperDefinition]] = defaultdict(list)
    known_modules: set[str] = set()
    for definition in definitions:
        if definition.ignored_by:
            continue
        known_modules.add(definition.module)
        if selected is not None and definition.module not in selected:
            continue
        if not definition.docstring:
            if selected is not None:
                raise ValueError(f"Enabled helper has no docstring: {definition.qualified_key}")
            continue
        grouped[definition.qualified_key].append(definition)
    if selected is not None and selected - known_modules:
        raise ValueError(f"Enabled modules have no eligible helpers: {sorted(selected - known_modules)}")
    result: list[HelperEntry] = []
    for key, variants in sorted(grouped.items()):
        variants.sort(key=lambda item: (item.path, item.lineno))
        first = variants[0]
        result.append(HelperEntry(
            qualified_key=key, module=first.module, parent_id=first.parent_id,
            definitions=tuple(variants),
        ))
    return tuple(result)


def active_entries(root: Path, *, ignore_patterns: Iterable[str]) -> tuple[HelperEntry, ...]:
    """Resolve the single rollout list; an empty list performs no source scan."""
    if not ENABLED_MODULES:
        return ()
    return entries(inventory(root, ignore_patterns=ignore_patterns), modules=ENABLED_MODULES)


def prepare_jinja(env, *, root: Path, ignore_patterns: Iterable[str]) -> None:
    """Give AutoAPI the same enabled entries used by the translation extractor."""
    by_module: dict[str, list[HelperEntry]] = defaultdict(list)
    for entry in active_entries(root, ignore_patterns=ignore_patterns):
        by_module[entry.module].append(entry)
    env.globals["spacr_nested_helpers"] = dict(by_module)
    env.globals["spacr_helper_only_modules"] = set()
    env.filters["spacr_helper_docstring"] = rendered_docstring


def rendered_docstring(entry: HelperEntry, app) -> str:
    """Apply the same Sphinx docstring processors as ordinary AutoAPI functions.

Napoleon's Google/NumPy conversion runs through this event, not through RST
parsing alone. Only a fresh list of rendering lines is changed; the canonical
source document used by translation hashes remains the original docstring.
"""
    lines = entry.docstring.splitlines()
    if lines:
        lines.append("")
        if "autodoc-process-docstring" in app.events.events:
            app.emit("autodoc-process-docstring", "function", entry.qualified_key,
                     None, None, lines)
    return "\n".join(lines)


def _signature_for_sphinx(signature: str) -> tuple[str, tuple[str, ...]]:
    """Protect arbitrary source default expressions from Sphinx's limited parser.

    Sphinx cannot unparse every valid Python expression (e.g. comparisons).
    Its parser receives inert strings; the directive restores the original
    expressions in the resulting display nodes. This never evaluates defaults.
    """
    name, separator, arguments = signature.partition("(")
    if not separator:
        raise ValueError(f"Missing helper argument list: {signature}")
    function = ast.parse(f"def _helper({arguments}: pass").body[0]
    defaults: list[str] = []
    for group in (function.args.defaults, function.args.kw_defaults):
        for index, expression in enumerate(group):
            if expression is not None:
                defaults.append(ast.unparse(expression))
                group[index] = ast.Constant(value=f"spacr-helper-default-{len(defaults)}")
    safe = f"{name}({ast.unparse(function.args)})"
    if function.returns is not None:
        safe += f" -> {ast.unparse(function.returns)}"
    return safe, tuple(defaults)


def register_sphinx_directive(app) -> None:
    """Register a helper-only Python function renderer; leave AutoAPI unchanged."""
    from docutils import nodes
    from sphinx.domains.python import PyFunction

    class HelperFunction(PyFunction):
        def run(self):
            # Keep the Python domain's canonical function type and anchor rules.
            self.name = "py:function"
            return super().run()

        def handle_signature(self, signature, signode):
            safe, originals = _signature_for_sphinx(signature)
            result = super().handle_signature(safe, signode)
            displayed = [node for node in signode.findall(nodes.inline)
                         if "default_value" in node.get("classes", ())]
            if len(displayed) != len(originals):
                raise RuntimeError(f"Sphinx lost helper defaults: {signature}")
            for node, original in zip(displayed, originals):
                node[:] = [nodes.Text(original)]
            return result

    app.add_directive("spacr-helper-function", HelperFunction)


def helper_page_policy(what: str, name: str, obj, skip: bool, options) -> bool | None:
    """Expose selected helper-only module pages without their hidden top-level API.

The module's own template suppresses its original body when this marks it as
helper-only. Descendants of such an originally hidden module are also skipped;
the helper entries are emitted independently from the canonical inventory.
"""
    if not ENABLED_MODULES:
        return None
    hidden_modules = obj.jinja_env.globals.setdefault("spacr_helper_only_modules", set())
    if what in {"module", "package"} and name in ENABLED_MODULES:
        if skip:
            obj.obj["spacr_helpers_only"] = True
            hidden_modules.add(name)
        return False
    if any(name.startswith(module + ".") for module in hidden_modules):
        return True
    return None


def report(
    definitions: Iterable[HelperDefinition],
    split_blocks: Callable,
) -> dict:
    """Report raw definitions, exclusions, canonical entries and actual blocks.

Block splitting is supplied by the existing documentation extractor. It is
never approximated by counting paragraphs, and repeated definitions are
reported separately from the number of unique anchors they will produce.
"""
    definitions = tuple(definitions)
    eligible = tuple(item for item in definitions if not item.ignored_by)
    canonical = entries(definitions)
    raw_blocks = [
        block for definition in eligible
        for block in split_blocks(definition.docstring)[0]
    ]
    rendered_blocks = [
        block for entry in canonical for block in split_blocks(entry.docstring)[0]
    ]
    serialized = [asdict(item) for item in definitions]
    fingerprint = hashlib.sha256(json.dumps(
        serialized, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {
        "schema": 1,
        "inventory_sha256": fingerprint,
        "counts": {
            "all_definitions": len(definitions),
            "all_modules": len({item.module for item in definitions}),
            "eligible_definitions": len(eligible),
            "eligible_modules": len({item.module for item in eligible}),
            "documented_eligible_definitions": sum(bool(item.docstring) for item in eligible),
            "private_eligible_definitions": sum(item.name.startswith("_") for item in eligible),
            "ignored_definitions": len(definitions) - len(eligible),
            "eligible_definition_blocks": len(raw_blocks),
            "unique_definition_blocks": len(set(raw_blocks)),
            "canonical_entries": len(canonical),
            "canonical_blocks": len(rendered_blocks),
            "unique_canonical_blocks": len(set(rendered_blocks)),
        },
        "by_module": dict(sorted(Counter(item.module for item in eligible).items())),
        "duplicates": {
            item.qualified_key: [definition.lineno for definition in item.definitions]
            for item in canonical if len(item.definitions) > 1
        },
        "definitions": serialized,
    }


def main() -> int:
    """Print the measured inventory or write its explicitly requested JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import build_documentation_i18n as builder

    measured = report(inventory(args.root, ignore_patterns=builder.AUTOAPI_IGNORE),
                      builder.translatable_blocks)
    revision = subprocess.run(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    measured["source_revision"] = revision.stdout.strip()
    text = json.dumps(measured, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps(measured["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
