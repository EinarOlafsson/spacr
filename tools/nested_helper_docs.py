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
