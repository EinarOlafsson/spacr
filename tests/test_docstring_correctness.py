"""Mechanical correctness checks on docstrings.

Instruction 112. A docstring can be present, counted, extracted and
translated into eight languages and still be wrong about the code beside it,
and none of the existing guards would notice -- they check that prose EXISTS.
These check that it AGREES.

Instruction 306 adds the reverse guarantee over an explicit public-callable
boundary.  That boundary is derived from source and signatures, never from
which definitions already have docstrings or ``:param:`` fields.  Otherwise
deleting the prose being checked would make the checker report less debt.

The legacy Tk modules are excluded, the same four excluded from instruction
60's coverage scope, because they are not maintained.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import importlib.util
import inspect
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache

import pytest

#: Sphinx ``:param name:`` up to the next field or the end.
PARAM_FIELD = re.compile(r":param\s+([*\w]+)\s*:(.*?)(?=\n\s*:|\Z)", re.S)

#: Sphinx instance-variable fields accepted only for generated constructors.
IVAR_FIELD = re.compile(
    r"(?m)^[ \t]*:ivar[ \t]+(\*{0,2}[A-Za-z_]\w*)[ \t]*:")

GENERATED_CONSTRUCTOR_CATEGORIES = frozenset({
    "dataclass_constructor",
    "namedtuple_constructor",
})

#: The source spellings Napoleon accepts as parameter sections in these docs.
_NUMPY_PARAMETER_SECTIONS = {"parameters", "other parameters"}
_GOOGLE_PARAMETER_SECTIONS = {
    "args",
    "arguments",
    "keyword args",
    "keyword arguments",
    # Napoleon accepts this Google-style colon form as an alias too. Ten
    # current required parameters use it, so omitting it would merely move the
    # same false-negative from ``Args:`` to ``Parameters:``.
    "parameters",
    "other parameters",
}
_NUMPY_SECTION_UNDERLINE = re.compile(
    r"^[=\-`:'\"~^_*+#<>]{2,}\s*$")
_SOURCE_PARAMETER_NAME = re.compile(r"^\*{0,2}[A-Za-z_]\w*$")

#: "Defaults to X" / "defaults to ``X``".
CLAIMED_DEFAULT = re.compile(r"[Dd]efaults?\s+to\s+``?([^`.,;)\s]+)``?")

#: The retired Tk front end. Not maintained; see instruction 60.
LEGACY_MODULES = {"gui.py", "gui_core.py", "gui_elements.py", "gui_utils.py"}

# These source trees are not Python API inputs.  This mirrors
# ``docs/source/conf.py:autoapi_ignore`` rather than allowing generated
# translation payloads and documentation asset generators to inflate the
# callable inventory.
# Kept in the same order and spelling as ``docs/source/conf.py``. AutoAPI
# applies these with ``fnmatch`` to the ordered path, not by asking whether
# the path happens to contain the same component names in any order.
AUTOAPI_IGNORE = (
    "*/tests/*",
    "*/qt/tutorial/*",
    "*/resources/*/_generators/*",
    "*/qt/i18n_catalogs/*",
)

# Explicit overrides for modules whose leading-underscore spelling or launch
# role defeats the default rule. ``spacr.__main__`` is present in rendered
# AutoAPI; the Qt launch module and tutorial target are CLI-only, while the
# v1/v2 bridge remains a compatibility surface rather than rendered prose.
MODULE_EXPOSURES = {
    "spacr.__main__": "autoapi",
    "spacr.qt.__main__": "cli_only",
    "spacr.qt.tutorial.__main__": "cli_only",
    "spacr._v1_v2_bridge": "compatibility",
}
CLI_ONLY_SYMBOLS = {
    "spacr.qt.run_without_setup",
    "spacr.qt.tutorial.__main__.main",
}

# ``Exception`` exposes a variadic positional constructor, so a useful
# conceptual name cannot be recovered from its signature. This one class has
# a deliberately reviewed, one-argument public contract; other exception
# prose remains subject to the ordinary ghost check.
EXCEPTION_PARAMETER_ALIASES = {
    "spacr.regression_qc.PanelUnavailable": frozenset({"reason"}),
}


@dataclass(frozen=True)
class _PublicCallable:
    """One source-owned callable contract admitted by the API boundary."""

    symbol: str
    category: str
    parameters: frozenset[str]
    required_parameters: frozenset[str]
    docstring: str
    accepted_documented_parameters: frozenset[str]
    variant_count: int
    docless_variant_count: int
    constructor_prose_variant_count: int
    exposure: str = "autoapi"
    accepts_arbitrary_keywords: bool = False


def _documented_functions():
    """``(path, node, docstring)`` for every function carrying ``:param:``."""
    root = pathlib.Path(__file__).resolve().parent.parent / "spacr"
    for path in sorted(root.rglob("*.py")):
        if path.name in LEGACY_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if doc and ":param" in doc:
                yield path, node, doc


def _module_name(root: pathlib.Path, path: pathlib.Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("spacr", *parts))


def _clean_doc(node: ast.AST) -> str:
    """Match the cleaned source body used by the documentation extractor."""
    value = ast.get_docstring(node, clean=False) or ""
    return inspect.cleandoc(value).strip()


def _autoapi_ignore_match(path: pathlib.Path) -> str | None:
    """First configured AutoAPI ignore pattern matching the ordered path."""
    text = str(path)
    return next(
        (pattern for pattern in AUTOAPI_IGNORE
         if fnmatch.fnmatch(text, pattern)),
        None,
    )


def _docs_autoapi_literal(name: str):
    """Read one literal AutoAPI setting without executing Sphinx config."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "source" / "conf.py"
    )
    tree = ast.parse(path.read_text())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError, SyntaxError) as exc:
                raise AssertionError(
                    f"docs/source/conf.py:{name} is no longer static") from exc
    raise AssertionError(f"docs/source/conf.py has no {name}")


def _module_exposure(
    root: pathlib.Path, path: pathlib.Path, module: str,
) -> str | None:
    """Rendered, CLI-only, compatibility-only, or outside the boundary."""
    if path.name in LEGACY_MODULES:
        return None
    if module in MODULE_EXPOSURES:
        return MODULE_EXPOSURES[module]
    # Preserve component order while making the match independent of the
    # checkout's absolute prefix, as AutoAPI's own ordered fnmatch is.
    relative = path.relative_to(root.parent)
    if _autoapi_ignore_match(relative) is not None:
        return None
    module_parts = module.split(".")[1:]
    if any(part.startswith("_") for part in module_parts):
        return None
    return "autoapi"


def _module_scope_nodes(statements):
    """Definitions/assignments that execute in module scope, branch by branch.

    ``if``/``try``/``with``/``match`` do not introduce a Python scope. A
    plain ``tree.body`` scan therefore loses definitions selected by optional
    dependencies -- exactly the FlowView and fractal parser hole this closes.
    Functions and classes *do* introduce scopes and are yielded without
    descending into their implementation bodies.
    """
    for node in statements:
        yield node
        if isinstance(node, ast.If):
            yield from _module_scope_nodes(node.body)
            yield from _module_scope_nodes(node.orelse)
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            yield from _module_scope_nodes(node.body)
            for handler in node.handlers:
                yield from _module_scope_nodes(handler.body)
            yield from _module_scope_nodes(node.orelse)
            yield from _module_scope_nodes(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _module_scope_nodes(node.body)
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            for case in node.cases:
                yield from _module_scope_nodes(case.body)


def _literal_export_names(node: ast.AST, current, path: pathlib.Path):
    """Evaluate one static sequence expression used to update ``__all__``."""
    if isinstance(node, ast.Name) and node.id == "__all__":
        if current is None:
            raise AssertionError(f"{path}: __all__ is read before assignment")
        return current
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_export_names(node.left, current, path)
        right = _literal_export_names(node.right, current, path)
        return left + right
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise AssertionError(
            f"{path}: unresolved dynamic __all__ expression at line "
            f"{getattr(node, 'lineno', '?')}"
        ) from exc
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise AssertionError(f"{path}: __all__ update is not a name sequence")
    if not all(isinstance(name, str) for name in value):
        raise AssertionError(f"{path}: __all__ contains a non-string name")
    if isinstance(value, (set, frozenset)):
        value = sorted(value)
    return tuple(value)


def _all_references(node: ast.AST):
    """Yield module-scope ``__all__`` references without entering scopes."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                         ast.Lambda)):
        return
    if isinstance(node, ast.Name) and node.id == "__all__":
        yield node
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "__all__"
    ):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _all_references(child)


def _has_all_write(node: ast.AST) -> bool:
    """Whether a compound statement contains an attempted ``__all__`` write."""
    return any(
        isinstance(reference, ast.Attribute)
        or (
            isinstance(reference, ast.Name)
            and isinstance(reference.ctx, (ast.Store, ast.Del))
        )
        for reference in _all_references(node)
    )


def _apply_export_statements(statements, states, path: pathlib.Path):
    """Abstractly execute supported top-level ``__all__`` operations."""
    for node in statements:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in targets):
                if len(targets) != 1:
                    raise AssertionError(
                        f"{path}: aliased __all__ assignment is unresolved")
                states = {
                    _literal_export_names(node.value, state, path)
                    for state in states
                }
                continue
            if any(
                isinstance(target, (ast.Attribute, ast.Subscript))
                and "__all__" in ast.unparse(target)
                for target in targets
            ):
                raise AssertionError(
                    f"{path}: unsupported indirect __all__ assignment")
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            if not isinstance(node.op, ast.Add):
                raise AssertionError(f"{path}: unsupported __all__ augmented op")
            states = {
                _literal_export_names(node.target, state, path)
                + _literal_export_names(node.value, state, path)
                for state in states
            }
            continue
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "__all__"
        ):
            call = node.value
            method = call.func.attr
            if method not in {"append", "extend"} or len(call.args) != 1 \
                    or call.keywords:
                raise AssertionError(
                    f"{path}: unsupported dynamic __all__.{method} call")
            updated = set()
            for state in states:
                if state is None:
                    raise AssertionError(
                        f"{path}: __all__.{method} before assignment")
                if method == "append":
                    try:
                        name = ast.literal_eval(call.args[0])
                    except (TypeError, ValueError, SyntaxError) as exc:
                        raise AssertionError(
                            f"{path}: dynamic __all__.append value") from exc
                    if not isinstance(name, str):
                        raise AssertionError(
                            f"{path}: __all__.append requires a string")
                    updated.add(state + (name,))
                else:
                    updated.add(
                        state + _literal_export_names(call.args[0], state, path))
            states = updated
            continue

        if isinstance(node, ast.If):
            if any(_all_references(node.test)):
                raise AssertionError(
                    f"{path}: unresolved __all__ reference in a condition")
            try:
                decision = ast.literal_eval(node.test)
            except (TypeError, ValueError, SyntaxError):
                decision = None
            if isinstance(decision, bool):
                branch = node.body if decision else node.orelse
                states = _apply_export_statements(branch, states, path)
            else:
                yes = _apply_export_statements(node.body, set(states), path)
                no = _apply_export_statements(node.orelse, set(states), path)
                states = yes | no
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            normal = _apply_export_statements(node.body, set(states), path)
            normal = _apply_export_statements(node.orelse, normal, path)
            possible = set(normal)
            for handler in node.handlers:
                possible |= _apply_export_statements(
                    handler.body, set(states), path)
            states = _apply_export_statements(node.finalbody, possible, path)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            states = _apply_export_statements(node.body, states, path)
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            possible = set()
            for case in node.cases:
                possible |= _apply_export_statements(
                    case.body, set(states), path)
            states = possible or states
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)) \
                and _has_all_write(node):
            raise AssertionError(
                f"{path}: loop-dependent __all__ mutation is unresolved")
        elif any(_all_references(node)):
            raise AssertionError(
                f"{path}: unsupported __all__ operation at line "
                f"{getattr(node, 'lineno', '?')}")
    return states


def _static_exports(tree: ast.Module, path: pathlib.Path):
    """Union of every statically possible final ``__all__`` state."""
    states = _apply_export_statements(tree.body, {None}, path)
    if states == {None}:
        return None
    if None in states:
        raise AssertionError(
            f"{path}: __all__ exists on only some top-level paths")
    return frozenset(name for state in states for name in state)


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_property_definition(node: ast.AST) -> bool:
    """Properties are attributes in AutoAPI, not callable method entries."""
    return any(
        _decorator_name(decorator)
        in {"property", "setter", "deleter", "cached_property"}
        for decorator in node.decorator_list
    )


def _literal_keyword(call: ast.AST, name: str):
    if not isinstance(call, ast.Call):
        return None
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        try:
            return ast.literal_eval(keyword.value)
        except (TypeError, ValueError, SyntaxError):
            return None
    return None


def _imported_names(
    tree: ast.Module, module: str, package_module: bool,
) -> dict[str, str]:
    """Map local import spellings to their absolute source symbols."""
    imported_names: dict[str, str] = {}
    package = module if package_module else module.rsplit(".", 1)[0]
    for node in _module_scope_nodes(tree.body):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".", 1)[0]
                imported_names[local] = imported.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                keep = len(parts) - (node.level - 1)
                if keep < 1:
                    continue
                base = ".".join(parts[:keep])
                if node.module:
                    base = f"{base}.{node.module}"
            else:
                base = node.module or ""
            for imported in node.names:
                if imported.name == "*":
                    continue
                local = imported.asname or imported.name
                imported_names[local] = ".".join(
                    part for part in (base, imported.name) if part)
    return imported_names


def _factory_names(imports: dict[str, str], nodes=()):
    """Local aliases for dataclass fields and both NamedTuple factories."""
    dataclasses = {"dataclass"}
    fields = {"field"}
    named_tuples = {"NamedTuple", "namedtuple"}
    for local, target in imports.items():
        if target == "dataclasses.dataclass":
            dataclasses.add(local)
        elif target == "dataclasses.field":
            fields.add(local)
        elif target in {"typing.NamedTuple", "collections.namedtuple"}:
            named_tuples.add(local)
    # Preserve straightforward source aliases (including chains) without
    # executing the module. This covers ``TupleFactory = NamedTuple`` as well
    # as aliases declared directly in an import statement.
    changed = True
    while changed:
        changed = False
        for node in nodes:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            source = _decorator_name(node.value)
            target = targets[0].id
            for names in (dataclasses, fields, named_tuples):
                if source in names and target not in names:
                    names.add(target)
                    changed = True
    return dataclasses, fields, named_tuples


def _is_dataclass(
    node: ast.ClassDef, decorator_names: set[str],
) -> bool:
    return any(
        _decorator_name(decorator) in decorator_names
        or _decorator_name(decorator) == "dataclass"
        for decorator in node.decorator_list
    )


def _dataclass_generates_init(
    node: ast.ClassDef, decorator_names: set[str],
) -> bool:
    for decorator in node.decorator_list:
        if (
            _decorator_name(decorator) in decorator_names
            or _decorator_name(decorator) == "dataclass"
        ):
            return _literal_keyword(decorator, "init") is not False
    return False


def _expression_symbol(
    node: ast.AST, module: str, imports: dict[str, str],
) -> str:
    """Resolve a simple local/imported dotted expression without importing."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return imports.get(node.id, f"{module}.{node.id}")
    if isinstance(node, ast.Attribute):
        parts = ast.unparse(node).split(".")
        head = imports.get(parts[0], f"{module}.{parts[0]}")
        return ".".join((head, *parts[1:]))
    return ""


def _is_named_tuple(
    node: ast.ClassDef, named_tuple_names: set[str],
) -> bool:
    return any(
        _decorator_name(base) in named_tuple_names
        or _decorator_name(base) == "NamedTuple"
        for base in node.bases
    )


def _node_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[frozenset[str], frozenset[str]]:
    """All and required signature names, excluding bound receivers."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    names = {arg.arg for arg in positional}
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)

    required_positional = positional
    if args.defaults:
        required_positional = positional[:-len(args.defaults)]
    required = {arg.arg for arg in required_positional}
    required.update(
        arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    )
    return (
        frozenset(names - {"self", "cls"}),
        frozenset(required - {"self", "cls"}),
    )


def _local_generated_fields(
    node: ast.ClassDef, field_names: set[str],
) -> dict[str, bool]:
    """``{field: required}`` for locally generated constructor fields."""
    fields: dict[str, bool] = {}
    for child in node.body:
        if not isinstance(child, ast.AnnAssign):
            continue
        if not isinstance(child.target, ast.Name):
            continue
        name = child.target.id
        annotation = ast.unparse(child.annotation)
        if (
            name.startswith("_")
            or "ClassVar" in annotation
            or _decorator_name(child.annotation) == "KW_ONLY"
        ):
            continue

        field_call = (
            isinstance(child.value, ast.Call)
            and _decorator_name(child.value) in field_names
        )
        if field_call and _literal_keyword(child.value, "init") is False:
            continue

        required = False
        if child.value is None:
            required = True
        elif field_call:
            keywords = {keyword.arg for keyword in child.value.keywords}
            if "default" not in keywords and "default_factory" not in keywords:
                required = True
        fields[name] = required
    return fields


def _dataclass_constructor_parameters(
    symbol: str,
    node: ast.ClassDef,
    module_info: dict,
    class_index: dict[str, list[tuple[dict, ast.ClassDef]]],
    seen: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve inherited dataclass fields once, then apply local overrides."""
    if symbol in seen:
        raise AssertionError(f"dataclass inheritance cycle at {symbol}")
    inherited: dict[str, bool] = {}
    next_seen = seen | {symbol}
    for base in node.bases:
        base_symbol = _expression_symbol(
            base, module_info["module"], module_info["imports"])
        possible_base_fields: dict[str, bool] = {}
        for base_info, base_node in class_index.get(base_symbol, []):
            if not _is_dataclass(base_node, base_info["dataclass_names"]):
                continue
            base_fields, base_required = _dataclass_constructor_parameters(
                base_symbol, base_node, base_info, class_index, next_seen)
            for name in base_fields:
                possible_base_fields[name] = (
                    possible_base_fields.get(name, False)
                    or name in base_required
                )
        inherited.update(possible_base_fields)
    inherited.update(_local_generated_fields(node, module_info["field_names"]))
    return (
        frozenset(inherited),
        frozenset(name for name, required in inherited.items() if required),
    )


def _functional_namedtuple_parameters(
    node: ast.Assign | ast.AnnAssign,
    named_tuple_names: set[str],
) -> tuple[str, frozenset[str], frozenset[str]] | None:
    """Parse aliased functional ``NamedTuple``/``namedtuple`` declarations."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    call = node.value
    if not isinstance(call, ast.Call):
        return None
    if _decorator_name(call.func) not in named_tuple_names:
        return None

    names: list[str] = []
    if len(call.args) >= 2:
        fields = call.args[1]
        if isinstance(fields, ast.Constant) and isinstance(fields.value, str):
            names = fields.value.replace(",", " ").split()
        elif isinstance(fields, (ast.List, ast.Tuple)):
            for entry in fields.elts:
                if isinstance(entry, ast.Constant) \
                        and isinstance(entry.value, str):
                    names.append(entry.value)
                    continue
                if isinstance(entry, (ast.List, ast.Tuple)) and entry.elts:
                    first = entry.elts[0]
                    if isinstance(first, ast.Constant) \
                            and isinstance(first.value, str):
                        # The annotation is intentionally not literal-evaluated:
                        # ``str``, ``list[int]`` and imported types are all
                        # ordinary static NamedTuple declarations.
                        names.append(first.value)
                        continue
                raise AssertionError("dynamic functional NamedTuple field")
        else:
            raise AssertionError("dynamic functional NamedTuple fields")
    else:
        names = [
            keyword.arg for keyword in call.keywords
            if keyword.arg not in {"defaults", "module", "rename"}
        ]

    defaults = _literal_keyword(call, "defaults")
    optional = len(defaults) if isinstance(defaults, (list, tuple)) else 0
    required = names[:-optional] if optional else names
    return targets[0].id, frozenset(names), frozenset(required)


def _assignment_docstrings(statements) -> dict[int, str]:
    """PEP-258 prose following assignments, including conditional suites."""
    docs: dict[int, str] = {}
    for index, node in enumerate(statements):
        following = statements[index + 1] if index + 1 < len(statements) else None
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            docs[id(node)] = following.value.value.strip()
        if isinstance(node, ast.If):
            docs.update(_assignment_docstrings(node.body))
            docs.update(_assignment_docstrings(node.orelse))
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            docs.update(_assignment_docstrings(node.body))
            for handler in node.handlers:
                docs.update(_assignment_docstrings(handler.body))
            docs.update(_assignment_docstrings(node.orelse))
            docs.update(_assignment_docstrings(node.finalbody))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            docs.update(_assignment_docstrings(node.body))
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            for case in node.cases:
                docs.update(_assignment_docstrings(case.body))
    return docs


def _literal_slots(node: ast.ClassDef) -> frozenset[str]:
    """Return a class's statically declared slot names."""
    names: set[str] = set()
    for child in node.body:
        targets = child.targets if isinstance(child, ast.Assign) else (
            [child.target] if isinstance(child, ast.AnnAssign) else []
        )
        if not any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(child.value)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise AssertionError(
                f"dynamic __slots__ on public class {node.name}") from exc
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, (list, tuple, set, frozenset)) \
                or not all(isinstance(name, str) for name in value):
            raise AssertionError(
                f"non-string __slots__ on public class {node.name}")
        names.update(value)
    return frozenset(names)


def _constructor_virtual_keywords(
    class_node: ast.ClassDef,
    constructor: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[frozenset[str], bool]:
    """Finite conceptual keywords consumed through an explicit ``**mapping``.

    A variadic keyword parameter is usually genuinely open and therefore
    cannot make a documented keyword a ghost. ``RoundResult`` is different:
    it consumes only keys named by a static ``__slots__`` declaration and
    silently ignores typos. Recognising that finite dispatch preserves its
    useful class-level ``:param:`` prose without granting an arbitrary-key
    loophole to every constructor that happens to spell ``**fields``.
    """
    if constructor.args.kwarg is None:
        return frozenset(), False
    mapping_name = constructor.args.kwarg.arg
    slots = _literal_slots(class_node)
    literal_keys: set[str] = set()
    dispatches_slots = False
    forwarded = False

    for child in ast.walk(constructor):
        if isinstance(child, ast.Call):
            for keyword in child.keywords:
                if (
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == mapping_name
                ):
                    forwarded = True
            if (
                isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == mapping_name
                and child.func.attr in {"get", "pop", "setdefault"}
                and child.args
            ):
                key = child.args[0]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    literal_keys.add(key.value)
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name)
            and child.value.id == mapping_name
        ):
            key = child.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                literal_keys.add(key.value)

    if slots:
        for loop in ast.walk(constructor):
            if not isinstance(loop, (ast.For, ast.AsyncFor)) \
                    or not isinstance(loop.target, ast.Name):
                continue
            iterator = ast.unparse(loop.iter)
            if iterator not in {"self.__slots__", "type(self).__slots__"}:
                continue
            loop_name = loop.target.id
            for child in ast.walk(ast.Module(body=loop.body, type_ignores=[])):
                if not isinstance(child, (ast.Call, ast.Subscript)):
                    continue
                if isinstance(child, ast.Call):
                    is_mapping_access = (
                        isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == mapping_name
                        and child.func.attr in {"get", "pop", "setdefault"}
                        and child.args
                    )
                    key = child.args[0] if is_mapping_access else None
                else:
                    is_mapping_access = (
                        isinstance(child.value, ast.Name)
                        and child.value.id == mapping_name
                    )
                    key = child.slice if is_mapping_access else None
                if isinstance(key, ast.Name) and key.id == loop_name:
                    dispatches_slots = True

    finite = dispatches_slots and not forwarded
    accepted = literal_keys | (set(slots) if finite else set())
    return frozenset(accepted), not finite


_BUILTIN_EXCEPTION_NAMES = frozenset({
    "BaseException", "Exception", "ArithmeticError", "AssertionError",
    "AttributeError", "BufferError", "EOFError", "FloatingPointError",
    "GeneratorExit", "ImportError", "IndexError", "KeyError",
    "KeyboardInterrupt", "LookupError", "MemoryError", "NameError",
    "NotImplementedError", "OSError", "OverflowError", "ReferenceError",
    "RuntimeError", "StopAsyncIteration", "StopIteration", "SyntaxError",
    "SystemError", "SystemExit", "TypeError", "ValueError", "Warning",
    "ZeroDivisionError",
})


def _inherits_exception(
    symbol: str,
    node: ast.ClassDef,
    module_info: dict,
    class_index: dict[str, list[tuple[dict, ast.ClassDef]]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Resolve a source-owned exception hierarchy without importing modules."""
    if symbol in seen:
        return False
    next_seen = seen | {symbol}
    for base in node.bases:
        tail = _decorator_name(base)
        if tail in _BUILTIN_EXCEPTION_NAMES:
            return True
        base_symbol = _expression_symbol(
            base, module_info["module"], module_info["imports"])
        for base_info, base_node in class_index.get(base_symbol, []):
            if _inherits_exception(
                base_symbol, base_node, base_info, class_index, next_seen,
            ):
                return True
        # External exception bases such as ``sqlite3.OperationalError`` are
        # not imported merely to ask their MRO. Their conventional terminal
        # spelling is the only static fact available and is narrower than
        # treating every unresolved imported class as an exception.
        if tail.endswith(("Error", "Exception")):
            return True
    return False


def _exception_constructor_parameters(
    symbol: str,
    node: ast.ClassDef,
    module_info: dict,
    class_index: dict[str, list[tuple[dict, ast.ClassDef]]],
    seen: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str], frozenset[str], bool]:
    """Resolve the nearest inherited exception constructor statically."""
    if symbol in seen:
        return frozenset(), frozenset(), frozenset(), False
    next_seen = seen | {symbol}
    parameters: set[str] = set()
    required: set[str] = set()
    accepted: set[str] = set()
    arbitrary = False
    found_source_contract = False

    for base in node.bases:
        base_symbol = _expression_symbol(
            base, module_info["module"], module_info["imports"])
        for base_info, base_node in class_index.get(base_symbol, []):
            if not _inherits_exception(
                base_symbol, base_node, base_info, class_index, next_seen,
            ):
                continue
            constructors = [
                child for child in _module_scope_nodes(base_node.body)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ]
            if not constructors:
                constructors = [
                    child for child in _module_scope_nodes(base_node.body)
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__new__"
                ]
            if constructors:
                found_source_contract = True
                for constructor in constructors:
                    current, current_required = _node_parameters(constructor)
                    virtual, current_arbitrary = _constructor_virtual_keywords(
                        base_node, constructor)
                    parameters.update(current)
                    required.update(current_required)
                    accepted.update(current | virtual)
                    arbitrary |= current_arbitrary
            else:
                inherited = _exception_constructor_parameters(
                    base_symbol, base_node, base_info, class_index, next_seen)
                if inherited[0]:
                    found_source_contract = True
                    parameters.update(inherited[0])
                    required.update(inherited[1])
                    accepted.update(inherited[2])
                    arbitrary |= inherited[3]

    if not found_source_contract:
        # Built-in exception constructors accept an arbitrary positional
        # message/payload tuple, but not arbitrary keyword names.
        parameters.add("args")
        accepted.add("args")
    return (
        frozenset(parameters), frozenset(required), frozenset(accepted),
        arbitrary,
    )


_CATEGORY_PRIORITY = {
    "inherited_or_default_constructor": 0,
    "exception_constructor": 1,
    "namedtuple_constructor": 2,
    "dataclass_constructor": 3,
    "constructor": 4,
    "function": 5,
    "method": 5,
}


def _merge_callable(
    previous: _PublicCallable | None, current: _PublicCallable,
) -> _PublicCallable:
    """Merge alternate top-level branches into one runtime-union contract."""
    if previous is None:
        return current
    if previous.exposure != current.exposure:
        raise AssertionError(
            f"inconsistent exposure for {current.symbol}: "
            f"{previous.exposure} vs {current.exposure}")
    category = max(
        (previous.category, current.category),
        key=lambda value: _CATEGORY_PRIORITY[value],
    )
    docs = []
    for value in (previous.docstring, current.docstring):
        if value and value not in docs:
            docs.append(value)
    return _PublicCallable(
        symbol=current.symbol,
        category=category,
        parameters=previous.parameters | current.parameters,
        required_parameters=(
            previous.required_parameters | current.required_parameters),
        docstring="\n".join(docs),
        accepted_documented_parameters=(
            previous.accepted_documented_parameters
            | current.accepted_documented_parameters
        ),
        variant_count=previous.variant_count + current.variant_count,
        docless_variant_count=(
            previous.docless_variant_count + current.docless_variant_count),
        constructor_prose_variant_count=(
            previous.constructor_prose_variant_count
            + current.constructor_prose_variant_count),
        exposure=current.exposure,
        accepts_arbitrary_keywords=(
            previous.accepts_arbitrary_keywords
            or current.accepts_arbitrary_keywords
        ),
    )


def _source_module_infos(root: pathlib.Path) -> list[dict]:
    """Parse the admitted modules and their source-only visibility state."""
    infos: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        module = _module_name(root, path)
        exposure = _module_exposure(root, path, module)
        if exposure is None:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        imports = _imported_names(
            tree, module, package_module=path.name == "__init__.py")
        nodes = tuple(_module_scope_nodes(tree.body))
        dataclass_names, field_names, named_tuple_names = _factory_names(
            imports, nodes)
        infos.append({
            "path": path,
            "module": module,
            "exposure": exposure,
            "tree": tree,
            "nodes": nodes,
            "exports": _static_exports(tree, path),
            "imports": imports,
            "dataclass_names": dataclass_names,
            "field_names": field_names,
            "named_tuple_names": named_tuple_names,
            "assignment_docs": _assignment_docstrings(tree.body),
        })
    return infos


@lru_cache(maxsize=1)
def _public_callable_inventory() -> tuple[_PublicCallable, ...]:
    """Build the complete source-owned public callable boundary.

    A literal module ``__all__`` is authoritative; modules without one use
    Python's leading-underscore convention.  Public functions and direct
    methods of public top-level classes are admitted whether or not they have
    prose.  Nested functions and private members are implementation details.
    Properties are attribute contracts, and non-constructor dunders are not
    enabled in this project's AutoAPI configuration.

    Every admitted class contributes one constructor record.  An explicit
    ``__init__`` (or ``__new__`` fallback) owns its signature; dataclass and
    NamedTuple fields own generated signatures; other classes have no locally
    declared constructor parameters.  AutoAPI's ``class_content = 'both'``
    makes class and explicit-constructor prose one rendered contract, so both
    are searched for constructor fields.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / "spacr"
    module_infos = _source_module_infos(root)
    class_index: dict[str, list[tuple[dict, ast.ClassDef]]] = {}
    for info in module_infos:
        for node in info["nodes"]:
            if isinstance(node, ast.ClassDef):
                class_index.setdefault(
                    f"{info['module']}.{node.name}", [],
                ).append((info, node))

    records: dict[str, _PublicCallable] = {}

    def admit(record: _PublicCallable) -> None:
        records[record.symbol] = _merge_callable(
            records.get(record.symbol), record)

    for info in module_infos:
        module = info["module"]
        exports = info["exports"]

        def visible(name: str, module=module, exports=exports) -> bool:
            symbol = f"{module}.{name}"
            if symbol in CLI_ONLY_SYMBOLS:
                return True
            if exports is not None:
                return name in exports
            return not name.startswith("_")

        for node in info["nodes"]:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = f"{module}.{node.name}"
                if not visible(node.name):
                    continue
                parameters, required = _node_parameters(node)
                doc = _clean_doc(node)
                exposure = (
                    "cli_only" if symbol in CLI_ONLY_SYMBOLS
                    else info["exposure"]
                )
                admit(_PublicCallable(
                    symbol=symbol,
                    category="function",
                    parameters=parameters,
                    required_parameters=required,
                    docstring=doc,
                    accepted_documented_parameters=parameters,
                    variant_count=1,
                    docless_variant_count=int(not doc),
                    constructor_prose_variant_count=0,
                    exposure=exposure,
                ))
                continue

            functional = None
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                functional = _functional_namedtuple_parameters(
                    node, info["named_tuple_names"])
            if functional is not None:
                name, parameters, required = functional
                if visible(name):
                    symbol = f"{module}.{name}"
                    doc = info["assignment_docs"].get(id(node), "")
                    admit(_PublicCallable(
                        symbol=symbol,
                        category="namedtuple_constructor",
                        parameters=parameters,
                        required_parameters=required,
                        docstring=doc,
                        accepted_documented_parameters=parameters,
                        variant_count=1,
                        docless_variant_count=int(not doc),
                        constructor_prose_variant_count=0,
                        exposure=info["exposure"],
                    ))
                continue

            if not isinstance(node, ast.ClassDef) or not visible(node.name):
                continue

            class_symbol = f"{module}.{node.name}"
            class_doc = _clean_doc(node)
            class_children = tuple(_module_scope_nodes(node.body))
            constructors = [
                child for child in class_children
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ]
            if not constructors:
                constructors = [
                    child for child in class_children
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__new__"
                ]

            accepts_arbitrary_keywords = False
            constructor_prose_variant_count = 0
            accepted: frozenset[str]
            if constructors:
                parameter_names: set[str] = set()
                required_names: set[str] = set()
                accepted_names: set[str] = set()
                constructor_docs: list[str] = []
                for constructor in constructors:
                    parameters, required = _node_parameters(constructor)
                    virtual, arbitrary = _constructor_virtual_keywords(
                        node, constructor)
                    parameter_names.update(parameters)
                    required_names.update(required)
                    accepted_names.update(parameters | virtual)
                    accepts_arbitrary_keywords |= arbitrary
                    constructor_doc = _clean_doc(constructor)
                    if constructor_doc and constructor_doc not in constructor_docs:
                        constructor_docs.append(constructor_doc)
                if not constructor_docs:
                    for constructor in class_children:
                        if not isinstance(
                            constructor,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        ) or constructor.name != "__new__":
                            continue
                        constructor_doc = _clean_doc(constructor)
                        if constructor_doc \
                                and constructor_doc not in constructor_docs:
                            constructor_docs.append(constructor_doc)
                parameters = frozenset(parameter_names)
                required = frozenset(required_names)
                accepted = frozenset(accepted_names)
                category = "constructor"
                constructor_prose_variant_count = len(constructor_docs)
                doc = class_doc
                for constructor_doc in constructor_docs:
                    # AutoAPI PythonClass.docstring uses one literal newline
                    # for class_content='both', including the empty-class case.
                    doc = f"{doc}\n{constructor_doc}"
            elif _dataclass_generates_init(node, info["dataclass_names"]):
                parameters, required = _dataclass_constructor_parameters(
                    class_symbol, node, info, class_index)
                accepted = parameters
                category = "dataclass_constructor"
                doc = class_doc
            elif _is_named_tuple(node, info["named_tuple_names"]):
                local_fields = _local_generated_fields(
                    node, info["field_names"])
                parameters = frozenset(local_fields)
                required = frozenset(
                    name for name, is_required in local_fields.items()
                    if is_required)
                accepted = parameters
                category = "namedtuple_constructor"
                doc = class_doc
            elif _inherits_exception(
                class_symbol, node, info, class_index,
            ):
                parameters, required, accepted, inherited_arbitrary = (
                    _exception_constructor_parameters(
                        class_symbol, node, info, class_index)
                )
                accepted |= EXCEPTION_PARAMETER_ALIASES.get(
                    class_symbol, frozenset())
                accepts_arbitrary_keywords |= inherited_arbitrary
                category = "exception_constructor"
                doc = class_doc
            else:
                parameters = required = accepted = frozenset()
                category = "inherited_or_default_constructor"
                doc = class_doc
            admit(_PublicCallable(
                symbol=class_symbol,
                category=category,
                parameters=parameters,
                required_parameters=required,
                docstring=doc,
                accepted_documented_parameters=accepted,
                variant_count=1,
                docless_variant_count=int(not doc),
                constructor_prose_variant_count=(
                    constructor_prose_variant_count),
                exposure=info["exposure"],
                accepts_arbitrary_keywords=accepts_arbitrary_keywords,
            ))

            for child in class_children:
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if child.name.startswith("_") or _is_property_definition(child):
                    continue
                parameters, required = _node_parameters(child)
                method_symbol = f"{class_symbol}.{child.name}"
                doc = _clean_doc(child)
                admit(_PublicCallable(
                    symbol=method_symbol,
                    category="method",
                    parameters=parameters,
                    required_parameters=required,
                    docstring=doc,
                    accepted_documented_parameters=parameters,
                    variant_count=1,
                    docless_variant_count=int(not doc),
                    constructor_prose_variant_count=0,
                    exposure=info["exposure"],
                ))

    return tuple(sorted(records.values(), key=lambda item: item.symbol))


def _public_callables():
    """Yield the cached immutable inventory to each independent ratchet."""
    yield from _public_callable_inventory()


def _real_parameters(node):
    args = node.args
    names = {a.arg for a in
             list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names - {"self", "cls"}


def _required_parameters(node):
    """Parameters without a positional or keyword-only default."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if args.defaults:
        positional = positional[:-len(args.defaults)]
    names = {arg.arg for arg in positional}
    names.update(
        arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    )
    return names - {"self", "cls"}


def _declared_defaults(node):
    """``{name: literal}`` for parameters with a literal default."""
    args = node.args
    positional = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    out = {}
    if args.defaults:
        for name, default in zip(positional[-len(args.defaults):],
                                 args.defaults):
            out[name] = default
    for name, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            out[name.arg] = default
    return out


def test_no_docstring_names_a_parameter_that_does_not_exist():
    """A ``:param`` for an argument that was renamed or removed.

    The most common way a docstring goes stale, and invisible to any check
    that only asks whether documentation is present.
    """
    ghosts = []
    checked = 0
    for path, node, doc in _documented_functions():
        checked += 1
        real = _real_parameters(node)
        documented = _documented_parameter_names(doc)
        missing = documented - real
        if missing:
            ghosts.append(
                f"{path.name}:{node.name} documents {sorted(missing)} "
                f"but takes {sorted(real)}")

    assert checked > 1000, (
        f"only {checked} documented functions found -- the sweep is not "
        "covering the package, so a green result proves nothing")
    assert not ghosts, "\n  ".join(ghosts)


def _sha256_lines(lines) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _line_indent(line: str) -> int:
    """Return the number of leading whitespace characters in ``line``."""
    return len(line) - len(line.lstrip())


def _section_field_names(line: str, *, google: bool) -> frozenset[str]:
    """Extract one valid source-style parameter field header.

    NumPy permits ``left, right : type`` and a name without a type. Google
    requires its field colon and permits ``name (type): description``. The
    identifier check is deliberately strict: prose, bullets and attribute
    fields must not become parameter documentation merely because they carry
    a colon.
    """
    before, colon, _after = line.strip().partition(":")
    if google:
        if not colon:
            return frozenset()
        typed = re.fullmatch(r"(.+?)\(\s*(.*\S)\s*\)\s*", before)
        if typed:
            before = typed.group(1).strip()

    raw_names = [name.strip() for name in before.split(",") if name.strip()]
    if not raw_names or any(
        _SOURCE_PARAMETER_NAME.fullmatch(name) is None
        for name in raw_names
    ):
        return frozenset()
    return frozenset(name.lstrip("*") for name in raw_names)


def _documented_parameter_names(docstring: str) -> frozenset[str]:
    """Names documented in source formats enabled by ``docs/source/conf.py``.

    This is intentionally source-only: the ordinary test environment does
    not install Sphinx. It mirrors the relevant Napoleon boundary without
    treating arbitrary prose as structured documentation:

    * native reST ``:param name:`` fields;
    * underlined NumPy ``Parameters`` / ``Other Parameters`` sections; and
    * indented Google ``Args:`` / argument / keyword aliases.

    NumPy ``Attributes`` and reST ``:ivar:`` describe object state, not call
    arguments. Markdown-looking ``Parameters:\n- name: ...`` also remains
    outside the boundary because Napoleon renders it as prose, not a parameter
    field.
    """
    documented = {
        name.lstrip("*") for name, _body in PARAM_FIELD.findall(docstring)
    }
    lines = docstring.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = line.strip().lower()
        heading_indent = _line_indent(line)

        if (
            heading in _NUMPY_PARAMETER_SECTIONS
            and index + 1 < len(lines)
            and _NUMPY_SECTION_UNDERLINE.fullmatch(
                lines[index + 1].strip()) is not None
        ):
            field_index = index + 2
            while field_index < len(lines):
                field_line = lines[field_index]
                field_heading = field_line.strip().lower()
                if (
                    field_index + 1 < len(lines)
                    and _NUMPY_SECTION_UNDERLINE.fullmatch(
                        lines[field_index + 1].strip()) is not None
                ):
                    break
                if (
                    _line_indent(field_line) == heading_indent
                    and field_heading.endswith(":")
                    and field_heading[:-1] in _GOOGLE_PARAMETER_SECTIONS
                ):
                    break
                if field_line and _line_indent(field_line) < heading_indent:
                    break
                if (
                    not field_line
                    and field_index + 1 < len(lines)
                    and not lines[field_index + 1]
                ):
                    break
                if field_line and _line_indent(field_line) == heading_indent:
                    documented.update(
                        _section_field_names(field_line, google=False))
                field_index += 1
            index = field_index
            continue

        google_heading = (
            heading[:-1] if heading.endswith(":") else "")
        if google_heading in _GOOGLE_PARAMETER_SECTIONS:
            field_index = index + 1
            while field_index < len(lines) and not lines[field_index]:
                field_index += 1
            if (
                field_index < len(lines)
                and _line_indent(lines[field_index]) > heading_indent
            ):
                field_indent = _line_indent(lines[field_index])
                while field_index < len(lines):
                    field_line = lines[field_index]
                    if (
                        field_line
                        and _line_indent(field_line) <= heading_indent
                    ):
                        break
                    if (
                        field_line
                        and _line_indent(field_line) == field_indent
                    ):
                        documented.update(
                            _section_field_names(field_line, google=True))
                    field_index += 1
                index = field_index
                continue
        index += 1

    return frozenset(documented)


def _generated_constructor_ivar_names(
    item: _PublicCallable,
) -> frozenset[str]:
    """Required generated fields visibly described by exact ``:ivar:``."""
    if item.category not in GENERATED_CONSTRUCTOR_CATEGORIES:
        return frozenset()
    fields = {
        name.lstrip("*") for name in IVAR_FIELD.findall(item.docstring)
    }
    return frozenset(fields & item.required_parameters)


def _missing_required_parameters(item: _PublicCallable) -> frozenset[str]:
    """Required names absent from all callable-appropriate source prose."""
    documented = (
        _documented_parameter_names(item.docstring)
        | _generated_constructor_ivar_names(item)
    )
    return item.required_parameters - documented


def test_baseline_constructor_documents_every_field():
    """Baseline's optional failure reason is part of its public contract."""
    item = next(
        candidate for candidate in _public_callables()
        if candidate.symbol == "spacr.baseline.Baseline"
    )
    documented = _documented_parameter_names(item.docstring)
    assert item.parameters <= documented


@pytest.mark.parametrize("symbol", (
    "spacr.accelerator.Accelerator",
    "spacr.agreement.AgreementReport",
    "spacr.agreement.PairAgreement",
    "spacr.align.AlignResult",
    "spacr.align.PairResult",
    "spacr.align.Tile",
    "spacr.classify_classes.ClassRule",
    "spacr.classifier_quality.Confusion",
    "spacr.confusion.Confusion",
    "spacr.control_names.ControlSpec",
    "spacr.convert.ConversionResult",
    "spacr.convert.Mapping",
    "spacr.convert.SourceImage",
    "spacr.curation.LabelEdit",
    "spacr.custom_features.CustomFeature",
    "spacr.database_schema.Migration",
    "spacr.database_schema.MigrationReport",
    "spacr.feature_dict.ConditionalUnit",
    "spacr.feature_dict.FeatureEntry",
    "spacr.feature_dict.PropertyInfo",
    "spacr.figures.sheet.Sheet",
    "spacr.flowview.layout.GraphLayout",
    "spacr.external_masks.MaskMatch",
    "spacr.feature_dict.FeatureScope",
    "spacr.foreign.Conflict",
    "spacr.foreign.ColumnMap",
    "spacr.benchmark.Recommendation",
    "spacr.portable_paths.RerootReport",
    "spacr.figures.fast_render.RenderedPanel",
    "spacr.figures.scene.SceneReport",
    "spacr.selection.CategoryFilter",
    "spacr.selection.DataFilter",
    "spacr.selection.RangeFilter",
    "spacr.selection.Selection",
    "spacr.sra.RunFile",
    "spacr.api.MaskConfig",
    "spacr.api.MeasureConfig",
    "spacr.gene_measurement_sweep.SweepResult",
    "spacr.macro.Recording",
    "spacr.metadata_resolution.MetadataDecision",
    "spacr.metadata_resolution.MetadataRequest",
    "spacr.metadata_resolution.ResolutionResult",
    "spacr.measure_hooks.RegisteredHook",
    "spacr.measurement_scan.MeasurementEffect",
    "spacr.measurement_scan.ScanResult",
    "spacr.mixed_gpu.TorchMixedResults",
    "spacr.model_check.ModelReport",
    "spacr.multiple_testing.MethodSpec",
    "spacr.hyperparam.Trial",
    "spacr.schema.ColumnCollision",
    "spacr.schema.FieldID",
    "spacr.schema.ObjectID",
    "spacr.illumination.IlluminationField",
    "spacr.illumination.IlluminationModel",
    "spacr.illumination.PreparedIllumination",
    "spacr.flowview.events.NodeAdded",
    "spacr.flowview.events.EdgeAdded",
    "spacr.flowview.events.StageStarted",
    "spacr.flowview.events.StageProgress",
    "spacr.flowview.events.StageMetric",
    "spacr.flowview.events.StageThumbnail",
    "spacr.flowview.events.StageCompleted",
    "spacr.flowview.events.StageFailed",
    "spacr.plate_qc.GradientStats",
    "spacr.plate_qc.RingStats",
    "spacr.plate_measurements.PlateDatabase",
    "spacr.plate_measurements.PlateMerge",
    "spacr.plate_measurements.TableMerge",
    "spacr.plugins.ModelProviderContribution",
    "spacr.plugins.ReportSectionContribution",
    "spacr.report.Table",
    "spacr.report.Figure",
    "spacr.report.Report",
    "spacr.regex_infer.Proposal",
    "spacr.run_recommendations.Recommendation",
    "spacr.runctx.RunContext",
    "spacr.settings_advisor.Advice",
    "spacr.settings_advisor.Undecided",
    "spacr.updater.DryRun",
    "spacr.updater.InstallOffer",
    "spacr.updater.PackageChange",
    "spacr.updater.UpdateInfo",
    "spacr.umap_search.SearchRow",
    "spacr.umap_search.ClusterWalkRow",
    "spacr.curation.CurationEdit",
    "spacr.nonparametric_fits.Curve",
    "spacr.align.CanvasSpec",
    "spacr.attribution.Agreement",
    "spacr.attribution.Attribution",
    "spacr.batch_correction.BatchCorrectionReport",
    "spacr.lineage.LineageNode",
    "spacr.multi_database.MergeDecision",
    "spacr.multi_database.MergePlan",
    "spacr.multi_database.SourceSummary",
    "spacr.train_compare.TrainingRun",
    "spacr.train_compare.Comparison",
    "spacr.figures.stats.Assumption",
    "spacr.annotation_validation.Verdict",
    "spacr.confusion.ConfusionCell",
    "spacr.external_masks.ExternalMaskResult",
    "spacr.flowview.layout.NodeLayout",
    "spacr.crops.MigrationResult",
    "spacr.regex_infer.FieldEvidence",
    "spacr.attribution.MethodSpec",
    "spacr.external_masks.InputGroup",
    "spacr.foreign.MaskMapping",
    "spacr.foreign.ResolvedColumn",
    "spacr.nonparametric_fits.Agreement",
    "spacr.sudoku.SudokuResult",
    "spacr.train_compare.Series",
    "spacr.align.Placement",
    "spacr.report.Section",
    "spacr.figures.panels.Panel",
    "spacr.hit_attribution.HitRunContext",
    "spacr.power_model.ModelData",
    "spacr.predictions.MergeReport",
    "spacr.external_masks.ExternalMaskPlan",
    "spacr.annotation_validation.Screen",
    "spacr.hit_attribution.HitInvestigationResult",
))
def test_repaired_record_documents_every_constructor_parameter(symbol):
    """Each repaired generated record remains callable from its API prose."""
    item = next(
        candidate for candidate in _public_callables()
        if candidate.symbol == symbol
    )
    assert item.parameters <= _documented_parameter_names(item.docstring)


def _required_parameter_omission_inventory(
    items,
    canonical_aliases=frozenset(),
) -> tuple[list[str], Counter[str], Counter[str]]:
    """Exact omissions after narrow generated-field and alias handling."""
    omissions: list[str] = []
    omitted_callables: Counter[str] = Counter()
    omitted_parameters: Counter[str] = Counter()
    for item in items:
        if item.symbol in canonical_aliases:
            continue
        missing = _missing_required_parameters(item)
        if missing:
            omitted_callables[item.category] += 1
            omitted_parameters[item.category] += len(missing)
        omissions.extend(
            f"{item.symbol}:{name}" for name in missing)
    return omissions, omitted_callables, omitted_parameters


def _ghost_parameters(item: _PublicCallable) -> frozenset[str]:
    """Documented names that the callable's reviewed contract cannot take."""
    documented = _documented_parameter_names(item.docstring)
    if item.accepts_arbitrary_keywords:
        return frozenset()
    return documented - item.accepted_documented_parameters


@lru_cache(maxsize=1)
def _documentation_source_contract(
) -> tuple[dict[str, str], dict[str, str]]:
    """Load rendered prose and reviewed aliases without importing ``spacr``."""
    root = pathlib.Path(__file__).resolve().parent.parent
    tools = root / "tools"
    module_path = tools / "build_documentation_i18n.py"
    module_name = "_instruction306_documentation_builder"
    inserted_path = str(tools) not in sys.path
    if inserted_path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = builder
    try:
        spec.loader.exec_module(builder)
        return builder.public_docstrings(), dict(builder.API_DOC_ALIASES)
    finally:
        sys.modules.pop(module_name, None)
        if inserted_path:
            sys.path.remove(str(tools))


def _documentation_public_docstrings() -> dict[str, str]:
    """Exact source prose visible at each rendered AutoAPI boundary."""
    return _documentation_source_contract()[0]


def _documentation_api_doc_aliases() -> dict[str, str]:
    """Reviewed exact aliases used by the production documentation builder."""
    return _documentation_source_contract()[1]


def _callable_signature_contract(item: _PublicCallable):
    """All source-derived signature properties relevant to documentation."""
    return (
        item.category,
        item.exposure,
        item.parameters,
        item.required_parameters,
        item.accepted_documented_parameters,
        item.accepts_arbitrary_keywords,
    )


def _validated_callable_api_doc_aliases(
    items,
    rendered_docs: dict[str, str],
    aliases: dict[str, str],
) -> dict[str, str]:
    """Callable aliases with identical signatures and rendered prose."""
    by_symbol = {item.symbol: item for item in items}
    callable_aliases: dict[str, str] = {}
    for alias, canonical in aliases.items():
        assert canonical not in aliases, (
            f"API doc alias chain is not canonical: {alias} -> {canonical}"
        )
        assert alias in rendered_docs, f"API doc alias is not rendered: {alias}"
        assert canonical in rendered_docs, (
            f"API doc alias target is not rendered: {canonical}"
        )
        assert rendered_docs[alias], f"API doc alias has empty prose: {alias}"
        assert rendered_docs[alias] == rendered_docs[canonical], (
            f"API doc alias prose differs: {alias} -> {canonical}"
        )

        alias_item = by_symbol.get(alias)
        if alias_item is None:
            continue
        canonical_item = by_symbol.get(canonical)
        assert canonical_item is not None, (
            f"callable API doc alias target is absent: {alias} -> {canonical}"
        )
        assert _callable_signature_contract(alias_item) == (
            _callable_signature_contract(canonical_item)
        ), f"callable API doc alias signature differs: {alias} -> {canonical}"
        callable_aliases[alias] = canonical
    return callable_aliases


def _docstring_contract_differences(
    expected: dict[str, str], actual: dict[str, str],
) -> list[str]:
    """Hash-addressed missing or content-stale source documents."""
    differences: list[str] = []
    for symbol, expected_text in expected.items():
        expected_hash = hashlib.sha256(expected_text.encode()).hexdigest()
        if symbol not in actual:
            differences.append(f"{symbol}\0{expected_hash}\0MISSING")
            continue
        actual_hash = hashlib.sha256(actual[symbol].encode()).hexdigest()
        if actual[symbol] != expected_text:
            differences.append(
                f"{symbol}\0{expected_hash}\0{actual_hash}")
    return sorted(differences)


def test_public_boundary_helpers_reject_static_parser_evasions():
    """Exercise every state transition and fail-closed branch directly."""
    assert tuple(_docs_autoapi_literal("autoapi_ignore")) == AUTOAPI_IGNORE
    assert _docs_autoapi_literal("autoapi_options") == [
        "members", "show-inheritance", "show-module-summary",
    ]
    assert _docs_autoapi_literal("autoapi_python_class_content") == "both"

    path = pathlib.Path("synthetic_exports.py")
    tree = ast.parse("""
__all__ = ["discarded"]
__all__ = ("kept",)
__all__ += ["augmented"]
if OPTIONAL_DEPENDENCY:
    __all__.append("conditional_append")
else:
    __all__.extend(("conditional_extend",))
""")
    assert _static_exports(tree, path) == {
        "kept", "augmented", "conditional_append", "conditional_extend",
    }

    unresolved = (
        "__all__ = make_exports()",
        "__all__ = ['x']\nfor name in names:\n    __all__.append(name)",
        "__all__ = ['x']\nalias = __all__",
        "alias = __all__ = ['x']\nalias.append('y')",
        "__all__ = ['x']\n__all__.append(dynamic_name)",
    )
    for source in unresolved:
        try:
            _static_exports(ast.parse(source), path)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"dynamic __all__ state was silently accepted: {source}")

    conditional = ast.parse("""
if OPTIONAL:
    def selected():
        pass
else:
    class Fallback:
        pass
try:
    def attempted():
        pass
except ImportError:
    def unavailable():
        pass
""")
    names = {
        node.name for node in _module_scope_nodes(conditional.body)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert names == {"selected", "Fallback", "attempted", "unavailable"}

    # The AutoAPI patterns describe ordered paths. Merely containing the same
    # component names in a different order is not an exclusion.
    assert _autoapi_ignore_match(
        pathlib.Path("spacr/qt/tutorial/render.py"),
    ) == "*/qt/tutorial/*"
    assert _autoapi_ignore_match(
        pathlib.Path("spacr/tutorial/qt/render.py"),
    ) is None
    assert _autoapi_ignore_match(
        pathlib.Path("spacr/resources/icons/_generators/make.py"),
    ) == "*/resources/*/_generators/*"
    assert _autoapi_ignore_match(
        pathlib.Path("spacr/_generators/icons/resources/make.py"),
    ) is None


def test_generated_constructor_models_resist_signature_evasions():
    """Cover inherited dataclasses, exceptions and both tuple factories."""
    tree = ast.parse("""
from dataclasses import dataclass as record, field as value_field
from typing import NamedTuple as TypedTuple
from collections import namedtuple as tuple_factory

@record
class Base:
    name: str
    hidden: str = value_field(init=False)

@record
class Child(Base):
    count: int = 0

class Failure(Exception):
    def __init__(self, reason: str):
        self.reason = reason

class SpecificFailure(Failure):
    pass

TupleAlias = TypedTuple
class Declared(TupleAlias):
    value: int

Typed = TypedTuple("Typed", [("key", str), ("payload", list[int])])
Plain = tuple_factory("Plain", "name count", defaults=(0,))
ViaAlias = TupleAlias("ViaAlias", [("value", int)])
""")
    imports = _imported_names(tree, "spacr.synthetic", package_module=False)
    nodes = tuple(_module_scope_nodes(tree.body))
    dataclass_names, field_names, named_tuple_names = _factory_names(
        imports, nodes)
    info = {
        "module": "spacr.synthetic",
        "imports": imports,
        "dataclass_names": dataclass_names,
        "field_names": field_names,
    }
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    class_index = {
        f"spacr.synthetic.{name}": [(info, node)]
        for name, node in classes.items()
    }
    parameters, required = _dataclass_constructor_parameters(
        "spacr.synthetic.Child", classes["Child"], info, class_index)
    assert parameters == {"name", "count"}
    assert required == {"name"}
    assert _inherits_exception(
        "spacr.synthetic.SpecificFailure", classes["SpecificFailure"],
        info, class_index,
    )
    exception_contract = _exception_constructor_parameters(
        "spacr.synthetic.SpecificFailure", classes["SpecificFailure"],
        info, class_index,
    )
    assert exception_contract == (
        frozenset({"reason"}), frozenset({"reason"}),
        frozenset({"reason"}), False,
    )
    assert _is_named_tuple(classes["Declared"], named_tuple_names)

    assignments = {
        node.targets[0].id: node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    typed = _functional_namedtuple_parameters(
        assignments["Typed"], named_tuple_names)
    plain = _functional_namedtuple_parameters(
        assignments["Plain"], named_tuple_names)
    via_alias = _functional_namedtuple_parameters(
        assignments["ViaAlias"], named_tuple_names)
    assert typed == (
        "Typed", frozenset({"key", "payload"}),
        frozenset({"key", "payload"}),
    )
    assert plain == (
        "Plain", frozenset({"name", "count"}), frozenset({"name"}),
    )
    assert via_alias == (
        "ViaAlias", frozenset({"value"}), frozenset({"value"}),
    )


def test_public_callable_inventory_is_source_derived_not_docstring_derived():
    """Freeze the whole boundary and every source-owned parameter name.

    The inventory includes examples with no docstring and no ``:param``
    field.  Removing documentation therefore cannot remove the callable from
    this test's denominator.  The signature digest also catches a required
    parameter becoming optional (or the reverse) without changing counts.
    """
    before_modules = set(sys.modules)
    callables = list(_public_callables())
    imported_package_modules = {
        name for name in set(sys.modules) - before_modules
        if name == "spacr" or name.startswith("spacr.")
    }
    assert not imported_package_modules
    by_symbol = {item.symbol: item for item in callables}

    assert len(callables) == len(by_symbol) == 8_367
    assert Counter(item.category for item in callables) == {
        "function": 3_619,
        "method": 3_719,
        "constructor": 390,
        "dataclass_constructor": 441,
        "namedtuple_constructor": 6,
        "exception_constructor": 137,
        "inherited_or_default_constructor": 55,
    }
    assert Counter(item.exposure for item in callables) == {
        "autoapi": 8_362,
        "cli_only": 2,
        "compatibility": 3,
    }
    assert sum(item.variant_count for item in callables) == 8_374
    assert Counter(item.variant_count for item in callables) == {
        1: 8_360,
        2: 7,
    }
    assert sum(
        item.constructor_prose_variant_count for item in callables
    ) == 83
    assert sum(
        item.constructor_prose_variant_count > 0 for item in callables
    ) == 83
    # RE-RECORDED 2026-09-04. Every figure here moved UP together as the
    # package gained callables; not one of them fell, which is the direction
    # check that was run before these numbers were written.
    assert sum(len(item.parameters) for item in callables) == 16_538
    assert sum(len(item.required_parameters) for item in callables) == 8_359
    assert _sha256_lines(
        f"{item.symbol}\0{item.category}\0{item.exposure}\0"
        f"{','.join(sorted(item.parameters))}\0"
        f"{','.join(sorted(item.required_parameters))}\0"
        f"{','.join(sorted(item.accepted_documented_parameters))}\0"
        f"{int(item.accepts_arbitrary_keywords)}\0"
        f"{item.variant_count}\0{item.docless_variant_count}\0"
        f"{item.constructor_prose_variant_count}"
        for item in callables
    ) == "fa57a0a85996cab840b65a06ad1348e4c0e4306ca4c77506bd0e580abd6296d6"

    # Fieldless, docless and generated-constructor contracts all remain in
    # scope.  These are named assertions so a future refactor cannot preserve
    # only the headline count while losing the defect classes that motivated
    # the boundary.
    assert by_symbol["spacr.align.format_plan"].required_parameters == {"plan"}
    assert not PARAM_FIELD.findall(
        by_symbol["spacr.align.format_plan"].docstring)
    # WAS ``LayerStack.add_image``, which has since been documented. The
    # example has to be a callable that is STILL docless, or this assertion
    # stops standing for the class of defect it was written for.
    assert not by_symbol["spacr.qt.bridge.RunRegistry.register"].docstring
    assert by_symbol[
        "spacr.qt.bridge.RunRegistry.register"
    ].required_parameters == {"handle"}
    assert by_symbol["spacr.layers.LayerStack.add_image"].docstring, (
        "add_image lost its docstring again; it is no longer the documented "
        "half of this pair")
    event_filter = by_symbol["spacr.qt.app.MainWindow.eventFilter"]
    assert event_filter.required_parameters == {"event", "watched"}
    assert event_filter.required_parameters <= _documented_parameter_names(
        event_filter.docstring)
    assert "docstring above this line" not in event_filter.docstring
    assert by_symbol["spacr.api.MaskConfig"].category == "dataclass_constructor"
    assert by_symbol["spacr.api.MaskConfig"].required_parameters == {"src"}

    # Generated signatures include inherited fields on the concrete class
    # exactly once. ``name`` was the reviewer-found ThresholdGate omission;
    # FigureStyle exercises the cross-module form of the same resolution.
    threshold = by_symbol["spacr.qt.widgets.gate_spec.ThresholdGate"]
    assert threshold.required_parameters == {"name"}
    assert threshold.parameters == {"name", "parent", "column", "low", "high"}
    assert "font_size" in by_symbol[
        "spacr.gene_measurement_compare.ComparisonStyle"
    ].parameters

    panel_unavailable = by_symbol["spacr.regression_qc.PanelUnavailable"]
    assert panel_unavailable.category == "exception_constructor"
    assert panel_unavailable.parameters == {"args"}
    assert panel_unavailable.accepted_documented_parameters == {"args", "reason"}

    round_result = by_symbol["spacr.active_learning.RoundResult"]
    assert not round_result.accepts_arbitrary_keywords
    assert {"round_index", "n_labels", "model_type"} \
        <= round_result.accepted_documented_parameters
    adversarial_round_doc = _PublicCallable(
        symbol=round_result.symbol,
        category=round_result.category,
        parameters=round_result.parameters,
        required_parameters=round_result.required_parameters,
        docstring=round_result.docstring + "\n:param misspelled_field: typo",
        accepted_documented_parameters=(
            round_result.accepted_documented_parameters),
        variant_count=round_result.variant_count,
        docless_variant_count=round_result.docless_variant_count,
        constructor_prose_variant_count=(
            round_result.constructor_prose_variant_count),
    )
    assert _ghost_parameters(adversarial_round_doc) == {"misspelled_field"}

    # Optional-dependency branches are one public runtime-union contract.
    # These definitions used to disappear because the scanner stopped at
    # ``tree.body``; the conditional __all__.append names disappeared too.
    assert by_symbol["spacr.flowview.items.EdgeItem"].parameters == {
        "edge", "source", "target", "source_running",
    }
    for symbol in (
        "spacr.flowview.items.NodeItem",
        "spacr.flowview.items.edge_width",
        "spacr.flowview.panel.FlowGraphicsView",
        "spacr.flowview.panel.FlowViewPanel",
        "spacr.flowview.panel.inspector_text",
        "spacr.qt.widgets.fractal_cascade.render_into",
        "spacr.qt.widgets.fractal_space.render_space_frame",
        "spacr.qt.widgets.fractal_space.sample_space",
    ):
        assert symbol in by_symbol
    for symbol in (
        "spacr.qt.widgets.fractal_cascade.render_into",
        "spacr.qt.widgets.fractal_space.render_space_frame",
        "spacr.qt.widgets.fractal_space.sample_space",
    ):
        assert by_symbol[symbol].variant_count == 2
        assert by_symbol[symbol].docless_variant_count == 1

    # A literal __all__ closes a module; nested functions, properties,
    # private Qt callbacks and non-constructor special methods are not
    # separately callable entries in this project's AutoAPI contract.
    assert "spacr.annotation.annotate_with" not in by_symbol
    assert "spacr.crashreport.collect.environment" not in by_symbol
    assert "spacr.layers.Layer.name" not in by_symbol
    assert "spacr.layers.Colormap.__eq__" not in by_symbol
    assert "spacr.qt.plate_queue.PlateQueue._serialise" not in by_symbol

    # CLI and compatibility contracts remain in the callable denominator but
    # cannot masquerade as rendered/translated AutoAPI. Both installed
    # console targets survive an excluding __all__ / AutoAPI ignore rule.
    setup_source = (
        pathlib.Path(__file__).resolve().parent.parent / "setup.py"
    ).read_text()
    assert "spacr-server=spacr.qt:run_without_setup" in setup_source
    assert "spacr-tutorial=spacr.qt.tutorial.__main__:main" in setup_source
    assert by_symbol["spacr.__main__.main"].exposure == "autoapi"
    assert by_symbol["spacr.qt.run_without_setup"].exposure == "cli_only"
    assert by_symbol[
        "spacr.qt.tutorial.__main__.main"
    ].exposure == "cli_only"
    assert by_symbol[
        "spacr._v1_v2_bridge.v2_channels_from_settings"
    ].exposure == "compatibility"
    assert by_symbol["spacr.qt.run"].exposure == "autoapi"
    assert "spacr.qt.tutorial.engine.render_tutorial" not in by_symbol


def test_no_new_public_callable_lacks_a_docstring():
    """Ratchet every docless executable variant, not merely each symbol."""
    docless = sorted(
        f"{item.symbol}\0{item.docless_variant_count}\0{item.variant_count}"
        for item in _public_callables()
        if item.docless_variant_count
    )
    assert len(docless) == 627
    assert sum(
        item.docless_variant_count for item in _public_callables()
    ) == 627
    assert _sha256_lines(docless) == (
        "c720ab8625390d2816e470cd1bcf87076e8a90eab74836f49410bea7bfe0df4b"
    )


def test_no_new_public_callable_ghost_parameters():
    """Ratchet class prose and constructor prose against accepted keywords.

    This is deliberately exact debt rather than a zero-only assertion: a
    deleted ``:param:`` field must change the count and digest just as an
    added typo does. Variadic-keyword constructors are open only when static
    analysis cannot prove a finite conceptual key set.
    """
    ghosts: list[str] = []
    ghost_callables: Counter[str] = Counter()
    ghost_parameters: Counter[str] = Counter()
    for item in _public_callables():
        item_ghosts = _ghost_parameters(item)
        if item_ghosts:
            ghost_callables[item.category] += 1
            ghost_parameters[item.category] += len(item_ghosts)
        ghosts.extend(
            f"{item.symbol}:{name}" for name in item_ghosts)

    # `FormulaPanel:frame` is gone -- it documented a parameter the
    # constructor does not take, and was replaced by the real one.
    assert len(ghosts) == 3
    assert sum(ghost_callables.values()) == 2
    assert ghost_callables == {
        "dataclass_constructor": 2,
    }
    assert ghost_parameters == {
        "dataclass_constructor": 3,
    }
    assert sorted(ghosts) == [
        "spacr.qt.widgets.feature_rank.FeatureScore:is_shape_not_shift",
        "spacr.qt.widgets.gate_spec.GateStats:of_parent",
        "spacr.qt.widgets.gate_spec.GateStats:of_total",
    ]
    assert _sha256_lines(ghosts) == (
        "decf552b966848fdd79950125e74ef9bc944430361f5b9c8b19197dd2658197a"
    )


def test_documented_parameter_parser_matches_rendered_source_styles():
    """Keep accepted Napoleon syntax narrow, source-only and adversarial."""
    accepted = """
    :param native: Native reST field.

    Parameters
    ----------
    left, right : numpy.ndarray
        NumPy permits several names on one field.
    *values
        A NumPy field does not require an explicit type.

    Other Parameters
    ----------------
    optional : bool
        A supported secondary NumPy section.

    Args:
        google (str): Typed Google field.

    Arguments:
        plain: Untyped Google field.

    Parameters:
        napoleon_alias (int): Napoleon's Google-style alias.

    Keyword Args:
        keyword_one: First keyword spelling.

    Keyword Arguments:
        **keyword_rest: Remaining keywords.
    """
    assert _documented_parameter_names(inspect.cleandoc(accepted)) == {
        "native",
        "left",
        "right",
        "values",
        "optional",
        "google",
        "plain",
        "napoleon_alias",
        "keyword_one",
        "keyword_rest",
    }

    rejected = """
    Attributes
    ----------
    numpy_attribute : int
        Object state is not a call parameter.

    :ivar rst_attribute: Also object state, not a call parameter.

    Parameters:
    - markdown_name: This bullet is not an indented Google field.
    """
    assert not _documented_parameter_names(inspect.cleandoc(rejected))


def test_generated_constructor_ivars_cannot_leak_to_ordinary_callables():
    """Accept exact generated-field prose without weakening other contracts."""
    generated = _PublicCallable(
        symbol="spacr.synthetic.Generated",
        category="dataclass_constructor",
        parameters=frozenset({"field", "missing", "optional"}),
        required_parameters=frozenset({"field", "missing"}),
        docstring=inspect.cleandoc("""
            :ivar field: Visible generated-field prose.
            :ivar optional: Optional state is not required debt.
        """),
        accepted_documented_parameters=frozenset({
            "field", "missing", "optional",
        }),
        variant_count=1,
        docless_variant_count=0,
        constructor_prose_variant_count=0,
    )
    assert _generated_constructor_ivar_names(generated) == {"field"}
    assert _missing_required_parameters(generated) == {"missing"}
    assert _missing_required_parameters(
        replace(generated, category="namedtuple_constructor")
    ) == {"missing"}

    for category in (
        "function",
        "method",
        "constructor",
        "exception_constructor",
        "inherited_or_default_constructor",
    ):
        ordinary = replace(generated, category=category)
        assert not _generated_constructor_ivar_names(ordinary)
        assert _missing_required_parameters(ordinary) == {"field", "missing"}

    malformed = replace(generated, docstring=":ivar field\nNo field colon.")
    assert not _generated_constructor_ivar_names(malformed)
    assert _missing_required_parameters(malformed) == {"field", "missing"}


def test_callable_boundary_is_cross_checked_with_i18n_extractor():
    """Require the extractor's keys and content to equal rendered source."""
    before_modules = set(sys.modules)
    docs = _documentation_public_docstrings()
    imported_package_modules = {
        name for name in set(sys.modules) - before_modules
        if name == "spacr" or name.startswith("spacr.")
    }
    assert not imported_package_modules

    rendered_documented_callables = {
        item.symbol: item.docstring for item in _public_callables()
        if item.exposure == "autoapi" and item.docstring
    }
    # The gap between the two is the entries AutoAPI never renders: the
    # configured ignore paths plus the CLI/compatibility entries.
    assert len(docs) == 9_406
    assert len(rendered_documented_callables) == 7_738
    assert not _docstring_contract_differences(
        rendered_documented_callables, docs)

    # A key-only comparison would accept this exact evasion: the symbol is
    # still present, but its rendered class/constructor body is source-stale.
    synthetic_expected = {"spacr.Example": "class prose\nconstructor prose"}
    synthetic_stale = {"spacr.Example": "class prose"}
    differences = _docstring_contract_differences(
        synthetic_expected, synthetic_stale)
    assert len(differences) == 1
    assert differences[0].startswith("spacr.Example\0")


def test_generated_constructor_ivar_reduction_is_exact_and_rendered():
    """Freeze the 145 visible fields and four ordinary counterexamples."""
    items = list(_public_callables())
    rendered_docs = _documentation_public_docstrings()
    required_ivars = {
        item.symbol: frozenset(
            name.lstrip("*") for name in IVAR_FIELD.findall(item.docstring)
        ) & item.required_parameters
        for item in items
    }
    required_ivars = {
        symbol: names for symbol, names in required_ivars.items() if names
    }
    by_symbol = {item.symbol: item for item in items}
    generated = {
        symbol: names for symbol, names in required_ivars.items()
        if by_symbol[symbol].category in GENERATED_CONSTRUCTOR_CATEGORIES
    }
    ordinary = {
        symbol: names for symbol, names in required_ivars.items()
        if by_symbol[symbol].category not in GENERATED_CONSTRUCTOR_CATEGORIES
    }

    assert len(required_ivars) == 34
    assert sum(map(len, required_ivars.values())) == 156
    assert len(generated) == 30
    assert sum(map(len, generated.values())) == 145
    assert Counter(by_symbol[symbol].category for symbol in generated) == {
        "dataclass_constructor": 29,
        "namedtuple_constructor": 1,
    }
    assert Counter(
        by_symbol[symbol].category
        for symbol in generated
        for _name in generated[symbol]
    ) == {
        "dataclass_constructor": 140,
        "namedtuple_constructor": 5,
    }
    assert len(ordinary) == 4
    assert sum(map(len, ordinary.values())) == 11
    assert {
        by_symbol[symbol].category for symbol in ordinary
    } == {"constructor"}

    remaining = {
        symbol: _missing_required_parameters(by_symbol[symbol])
        for symbol in generated
    }
    assert sum(not names for names in remaining.values()) == 30
    assert sum(bool(names) for names in remaining.values()) == 0
    assert sum(map(len, remaining.values())) == 0
    assert all(
        rendered_docs[symbol] == by_symbol[symbol].docstring
        for symbol in generated
    )
    # An ordinary constructor's ``:ivar:`` never supplies parameter credit.
    # It may coexist with a real ``:param:`` field, as FilenameMapper now
    # deliberately documents both construction and retained state.
    assert all(
        names - _documented_parameter_names(by_symbol[symbol].docstring)
        <= _missing_required_parameters(by_symbol[symbol])
        for symbol, names in ordinary.items()
    )


def test_callable_api_doc_alias_validation_rejects_contract_mutations():
    """A reviewed alias is unusable after any signature or prose drift."""
    canonical = _PublicCallable(
        symbol="spacr.synthetic.Base.apply",
        category="method",
        parameters=frozenset({"payload", "optional"}),
        required_parameters=frozenset({"payload"}),
        docstring=":param payload: Canonical source prose.",
        accepted_documented_parameters=frozenset({"payload", "optional"}),
        variant_count=1,
        docless_variant_count=0,
        constructor_prose_variant_count=0,
    )
    alias = replace(
        canonical,
        symbol="spacr.synthetic.Child.apply",
        docstring="",
    )
    aliases = {alias.symbol: canonical.symbol}
    rendered_docs = {
        canonical.symbol: canonical.docstring,
        alias.symbol: canonical.docstring,
    }
    assert _validated_callable_api_doc_aliases(
        [canonical, alias], rendered_docs, aliases,
    ) == aliases

    signature_mutations = (
        replace(alias, parameters=frozenset({"renamed", "optional"})),
        replace(alias, required_parameters=frozenset()),
        replace(alias, accepted_documented_parameters=frozenset({"payload"})),
        replace(alias, accepts_arbitrary_keywords=True),
    )
    for mutated_alias in signature_mutations:
        with pytest.raises(AssertionError, match="signature differs"):
            _validated_callable_api_doc_aliases(
                [canonical, mutated_alias], rendered_docs, aliases,
            )

    mutated_docs = dict(rendered_docs)
    mutated_docs[alias.symbol] = "Different rendered prose."
    with pytest.raises(AssertionError, match="prose differs"):
        _validated_callable_api_doc_aliases(
            [canonical, alias], mutated_docs, aliases,
        )


def test_callable_api_doc_alias_reduction_is_exact():
    """Deduplicate only current rendered aliases; keep canonical debt live."""
    items = list(_public_callables())
    by_symbol = {item.symbol: item for item in items}
    declared_aliases = _documentation_api_doc_aliases()
    callable_aliases = _validated_callable_api_doc_aliases(
        items,
        _documentation_public_docstrings(),
        declared_aliases,
    )

    assert len(declared_aliases) == 113
    assert len(callable_aliases) == 107
    assert set(declared_aliases) - set(callable_aliases) == {
        "spacr.layers.ImageLayer.ndim",
        "spacr.layers.ImageLayer.shape",
        "spacr.layers.LabelsLayer.ndim",
        "spacr.layers.LabelsLayer.shape",
        "spacr.layers.PointsLayer.ndim",
        "spacr.layers.ShapesLayer.ndim",
    }
    assert {
        by_symbol[alias].category for alias in callable_aliases
    } == {"method"}
    assert {
        by_symbol[alias].exposure for alias in callable_aliases
    } == {"autoapi"}

    alias_debt = {
        alias: _missing_required_parameters(by_symbol[alias])
        for alias in callable_aliases
        if _missing_required_parameters(by_symbol[alias])
    }
    canonical_debt = {
        canonical: _missing_required_parameters(by_symbol[canonical])
        for canonical in set(callable_aliases.values())
        if _missing_required_parameters(by_symbol[canonical])
    }
    assert len(alias_debt) == 90
    assert sum(map(len, alias_debt.values())) == 140
    assert len(canonical_debt) == 5
    assert sum(map(len, canonical_debt.values())) == 6

    raw = _required_parameter_omission_inventory(items)
    deduplicated = _required_parameter_omission_inventory(
        items, callable_aliases,
    )
    assert len(raw[0]) - len(deduplicated[0]) == 140
    assert raw[1] - deduplicated[1] == {"method": 90}
    assert raw[2] - deduplicated[2] == {"method": 140}


def test_no_new_undocumented_required_public_parameters():
    """Ratchet the full reverse-direction debt while it is repaired.

    The old denominator selected only callables whose prose already contained
    ``:param:`` and reached a misleading zero when those selected fields were
    completed. The source-derived denominator, exact generated-field rule and
    validated rendered aliases expose the real current baseline: 2,570
    omissions across 1,878 public callables. Count, category counts and digest
    are all exact so deleting prose, weakening a boundary, or swapping one
    omission for another cannot turn this test green.

    THE CONSTRUCTOR CATEGORY IS ALL BUT CLOSED: 43 omitted constructors
    became 2, and 64 omitted constructor parameters became 3. The headline
    total still rose, from 2,516 to 2,570, because `function` and `method`
    grew faster than the constructors were documented -- which is what a
    single total hides and these per-category counts do not.
    """
    items = list(_public_callables())
    callable_aliases = _validated_callable_api_doc_aliases(
        items,
        _documentation_public_docstrings(),
        _documentation_api_doc_aliases(),
    )
    omissions, omitted_callables, omitted_parameters = (
        _required_parameter_omission_inventory(items, callable_aliases)
    )

    assert len(omissions) == 2_570
    assert sum(omitted_callables.values()) == 1_878
    assert omitted_callables == {
        "function": 756,
        "method": 1_076,
        "constructor": 2,
        "dataclass_constructor": 42,
        "namedtuple_constructor": 2,
    }
    assert omitted_parameters == {
        "function": 1_127,
        "method": 1_298,
        "constructor": 3,
        "dataclass_constructor": 130,
        "namedtuple_constructor": 12,
    }
    assert _sha256_lines(omissions) == (
        "d84a93c68fee2291b6277fc345415e687aebe47dc57b92511fe2a19d1f3367e3"
    )


def test_no_docstring_claims_a_default_the_signature_contradicts():
    """"Defaults to 0" beside ``param=1`` is worse than no docstring: it is
    believed.

    ``None`` defaults are EXEMPT, and that is not a loophole. ``None`` as a
    sentinel meaning "work it out" is used throughout this package, and the
    useful thing to document there is what it resolves TO --
    ``setup_logging(level=None)`` documenting ``SPACR_LOG_LEVEL``, or
    ``open_crop_source(src=None)`` documenting ``settings['src']``. Flagging
    those reports 14 false positives and no real ones, which is how a check
    gets switched off.
    """
    wrong = []
    for path, node, doc in _documented_functions():
        if "efaults to" not in doc:
            continue
        defaults = _declared_defaults(node)
        for name, body in PARAM_FIELD.findall(doc):
            claim = CLAIMED_DEFAULT.search(body)
            if not claim or name not in defaults:
                continue
            try:
                actual = ast.literal_eval(defaults[name])
            except (ValueError, SyntaxError):
                continue        # a computed default; nothing to compare
            if actual is None:
                continue        # sentinel, documented by what it resolves to
            claimed = claim.group(1).strip("'\"")
            if str(actual) != claimed and repr(actual).strip("'\"") != claimed:
                wrong.append(
                    f"{path.name}:{node.name}({name}) says {claimed!r}, "
                    f"signature has {actual!r}")
    assert not wrong, "\n  ".join(wrong)
