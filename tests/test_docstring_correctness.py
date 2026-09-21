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

    # 8,453 -> 8,462 on 2026-09-07, +9/-0, and the tree was FROZEN at that
    # number by agreement while the localization catalogs regenerated. The
    # nine are the other session's spacr/infection.py (5 public functions)
    # and spacr/suggest.py (4): +8 functions and +1 dataclass constructor.
    # 8,462 -> 8,463 on 2026-09-08, +1/-0: RegexEditorDialog.resizeEvent,
    # which is why `method` moves and no other category does.
    # 8,463 -> 8,465 the same day, +2/-0, and both are one export:
    # `spacr.image_import.ImportResult` joined `__all__` beside the sibling
    # types that were already there. `apply_import` RETURNS it, and being
    # unexported it was undocumented -- so Sphinx could not resolve the bare
    # `ImportResult` in that signature against its own module and matched
    # `spacr.foreign` and `spacr.omero` instead. The two callables are the
    # dataclass constructor and its `summary`, which is why
    # `dataclass_constructor` and `method` each move by one.
    # 8,465 -> 8,459 on 2026-09-09: the six space accessors retired under
    # instruction 364, for a theme nothing could select.
    # 8,459 -> 8,508 on 2026-09-10, +49. The same OPS surface the
    # documented-symbol count decomposes below, less the entries that
    # are modules and attributes rather than callables.
    # 8,508 -> 8,498, -10. Ten of the twelve withdrawn entries are
    # CALLABLES -- the two module-level helpers, TourPilot and its four
    # members, and three of layout_policy's four functions; the modules
    # themselves are not callables and do not count here.
    # 8,498 -> 8,535 on 2026-09-11, +37 against a documented surface that
    # grew by 45. The difference is the eight entries that are not
    # callables: the four new modules' own docstrings (bystanders,
    # embeddings, ops_compose, point_patterns) and four dataclass
    # attributes. Same review as the +45 decomposed below.
    # 8,535 -> 8,582 on 2026-09-11, +47/-0, against a documented surface
    # that grew by 55. The difference is the eight entries that are not
    # callables: the three new modules' own docstrings (ops_objects,
    # ops_sample, ops_store), the Embeddings screen module, and four
    # dataclass attributes. Measured by diffing the symbol lists at
    # 7eb18402f and here rather than by subtracting counts. By module:
    #
    #    13  qt.widgets.dose_response  387     2  spacr.embeddings      386
    #     9  spacr.ops_objects         372     1  spacr.scorecard       370
    #     8  qt.screens.embeddings     386     1  spacr.qt.theme        380
    #     6  spacr.ops_store           372     1  spacr.qt.app          380
    #     6  spacr.ops_sample          372
    #    +1 spacr.qt.theme           380   set_a_sheeted_widgets_own_rule,
    #                                      merged after the count above was
    #                                      first taken. Re-measured rather
    #                                      than pre-written: a ratchet is a
    #                                      measurement of the tree, so
    #                                      whoever merges second re-runs it.
    # 8,584 -> 8,585 on 2026-09-12, +1/-0, and the one is
    # `spacr.infection.border_rules_agree`. Measured from the diff of
    # `spacr/` since the pin rather than by subtracting counts: three
    # commits touched the package and between them added exactly two
    # top-level defs, of which `_setting` is private. The same symbol is
    # what staled the API catalogs and the other two ratchets -- it landed
    # with 377 AFTER the catalogs were built, which is the mid-batch
    # arrival 288's rule exists to prevent.
    # 8,585 -> 8,587 on 2026-09-12, +2/-0: 364's migration adds two public
    # resolvers, `spacr.object_roles.split_role_setting` and
    # `spacr.settings.surviving_setting_name`. Both are module-level
    # functions, which is why every bucket below moves by the same two.
    # 8,587 -> 8,588 with 391's `withdrawn_setting_reason`. The parameter
    # total below FALLS in the same change, which is the shape worth
    # noticing: `spacr.utils.merge_split_objects` lost five relative
    # settings and gained one absolute threshold, so the surface grew by a
    # callable while shrinking by three parameters.
    # 8,588 -> 8,651 on 2026-09-13, +63 and nothing removed. Every one of
    # the 63 is Map Barcodes (c753b7de7), which is worth stating because the module
    # breakdown looks at first like four separate changes:
    #
    #     25  spacr.barcode_search      the new module: 5 dataclass
    #                                   constructors, 8 methods, 12 functions
    #     20  spacr.qt.screens          map_barcodes.BarcodeSearchPanel and
    #                                   its plan/card/install helpers
    #     11  spacr.qt.widgets          read_view.ReadView, ReadRow,
    #                                   BarcodeSpan and barcode_colours
    #      7  spacr.settings            BarcodeEntry, BarcodeSet and
    #                                   barcode_set_from_settings
    #
    # Measured by diffing the symbol set against 5a2825f67, not by
    # subtracting totals: a +63 that was really +64/-1 would read the same
    # from the total alone, and this test exists to catch exactly that.
    # 2026-09-14: 8,651 -> 8,696. Set-differenced against 49c1189f7 rather
    # than subtracted: +45 / -0, every arrival attributable -- 29
    # spacr.curation_queue and 7 spacr.cli_make_masks (396, both new
    # modules), 5 spacr.graph_types and 2 spacr.style_base (293 and 291),
    # 2 under spacr.qt. A +45 that was really +46/-1 reads identically from
    # the total, which is what this measurement style exists to catch.
    # 2026-09-15: 8,696 -> 8,697, +1 / -0: `spacr.graph_types.mark_to_start_on`
    # (293), a module-level function, so `function` and `autoapi` move by the
    # same one below and no other bucket does. Its sibling in `spacr.ml` was
    # public for four commits of the same batch and went private
    # (`_qc_graph_type_and_note`) before this was measured, so it never
    # reached a pin. PROVED BY SUBTRACTION: the inventory with that one symbol
    # dropped returns 8,696 and every category, exposure and variant bucket
    # below at its previous value.
    # 2026-09-15, 372 PART 14-M, measured against nightly 3f27b926a by
    # differencing the inventory LINES, not by subtracting counts: +6 / -0
    # arrivals (spacr.ops_cycles.AlignedField and align_field,
    # spacr.ops_engine.run_ops, spacr.ops_phenotype.phenotype_centres,
    # spacr.ops_sbs.attribute_reads and assign_reads_to_objects) and four
    # changed signatures (WellLayout, round_well_layout,
    # phenotype_site_map, call_reads). The same move took c3f562c4f's
    # 8,696 to 8,702 before the rebase.
    # 8,703 -> 8,684 on 2026-09-15, +0 / -19 by differencing the inventory
    # LINES against the switch commit: the old OPS engine was deleted, taking the 16
    # callables of the old engine's module and the three stitcher defaults in
    # `spacr.settings` that nothing called. No line was added or changed.
    # 2026-09-15: 8,684 -> 8,685, +1 / -0 by set difference against
    # origin/nightly df1216b3f: `spacr.barcode_search.SearchThresholds`, the frozen
    # dataclass a barcode search judges by -- public because
    # `search_barcodes(thresholds=...)` takes one. Six other callables from
    # the same batch were made private BEFORE this was measured, because
    # nothing outside the package calls them: graph_spec's `_kind_and_note`
    # and `GraphSpec._kind_note`, settings' three barcode-reference helpers,
    # and the live search's `_watch_the_form`. `stream_dataset.coordinate_column`
    # and `.settings_for_method` stay public here as wrappers over the private
    # `_stream_selection`, so neither left this inventory. Per bucket, only
    # `dataclass_constructor`, `autoapi` and the one-variant count move, each
    # by one.
    # 8,685 -> 8,698 on 2026-09-15, +13 / -0 by differencing the inventory
    # LINES against nightly dca970671: 412's seven module-level functions in
    # spacr.qt.make_masks_demo, and 416's four functions in
    # spacr.install_cleanup (find_old_installs, remove_install,
    # run_update_sequence, start_update_helper) plus the InstallRecord and
    # RemovalReport dataclass constructors. PROVED BY SUBTRACTION: the
    # inventory without those 13 symbols returns 8,685, every category,
    # exposure, variant and parameter bucket below at its previous value,
    # and the digest 2494eb63... byte for byte. No existing line changed.
    # 2026-09-20, AND THE SAME DRIFT THE API EXTRACTOR'S PINS HAD. Every
    # number in this file was last set on 2026-09-15 and this file has been
    # failing ever since, so none of them could be checked while five days
    # of work landed. The surface growth is accounted for by module in
    # test_api_i18n_extractor.py's note of this date -- +412 / -2 against
    # origin/nightly df1216b3f, bucketed by module because 412 names are a
    # list nobody reads -- and today's own additions are on top of it:
    # spacr.object_classifier (item 449) and the infection report writer
    # (item 377).
    #
    # SAID PLAINLY BECAUSE IT MATTERS: these four numbers were RE-MEASURED,
    # not set-differenced one metric at a time. What that does and does not
    # establish: every delta is an INCREASE, so nothing was removed from
    # the public surface unnoticed, and the increase has the same cause as
    # the one the sibling file accounts for symbol by symbol. It does not
    # prove which individual symbol moved which of these four counts. A
    # reader who needs that should set-difference the metric they care
    # about rather than trust this paragraph.
    # 9,047 -> 9,223 on 2026-09-21, +176 / -0 by set difference against
    # 6ae5e1b36, where these numbers were last set; the inventory built
    # there reproduces every pin below byte for byte, digest included, so
    # nothing left the surface and no existing line changed. By module:
    # 57 qt.widgets.plaque_preview, 31 plaque_papers, 15 timeflows_model,
    # 11 qt.make_masks_datasets, 10 qt.ai.pty_sign_in, 8 qt.ops_stitch_demo,
    # 6 import_examples, 6 qt.screens.foreign, 5 qt.import_demo,
    # 4 qt.assay_examples, 4 qt.widgets.measurements_example,
    # 3 qt.widgets.plate_layout, 3 qt.widgets.preview_refresh, 2 each in
    # qt.screens.app_screen, qt.screens.experiment_design and submodules,
    # and one each in example_archives, plaque, qt.mask_engine,
    # timeflows_qc, qt.widgets.card, qt.screens.make_masks and
    # qt.screens.settings_model.
    assert len(callables) == len(by_symbol) == 9_223
    # +30 function, +14 method, +1 constructor, +4 dataclass_constructor
    # on 2026-09-10 -- the OPS modules are mostly module-level functions,
    # which is why `function` carries most of the move, and the four
    # dataclasses are Registration, StitchedWell, Alignment and WellLayout.
    # -6 function, -3 method, -1 constructor on the withdrawal of the
    # same four symbols: the two module-level helpers and three of
    # `layout_policy`'s functions are the six, `TourPilot.dragged`,
    # `.restarted` and `.steer` are the three methods, and `TourPilot`
    # itself is the constructor. `.flying` is a property and lands in
    # neither bucket, which is why the numbers do not sum to twelve.
    # +24 function, +6 method, +5 dataclass_constructor, +2
    # exception_constructor on 2026-09-11 -- the same 37 the total above
    # decomposes. The new modules are mostly module-level functions, which
    # is why `function` carries two thirds of the move; the five dataclasses
    # are EmbeddingResult, EmbeddingSpec, Window, InteractionSurface and
    # SelectivityIndex, and the two exceptions are EmbeddingError and
    # ComposeError.
    # +25 function, +11 method, +7 dataclass_constructor, +3
    # exception_constructor, +1 constructor on 2026-09-11 -- the same 47.
    # The OPS and dose-response modules are mostly module-level functions,
    # which is why `function` carries over half the move; the seven
    # dataclasses are WindowObject, PlateObject, Readiness, PlateSpec,
    # PlateReport, PooledFit and Checkerboard, and the three exceptions are
    # ObjectsError, SampleError and StoreError.
    # `function` 3,723 -> 3,724 on 2026-09-12, the same single arrival as
    # the total above: `border_rules_agree` is a module-level function, so
    # it lands in this bucket and in no other. A count that moved here
    # WITHOUT the total moving, or the other way round, would mean a
    # symbol changed category rather than arrived.
    # All five moving categories are Map Barcodes (c753b7de7), +63 in total
    # and the per-category split is the evidence that they ARRIVED rather
    # than changed category: function +18, method +33, constructor +2,
    # dataclass_constructor +8, namedtuple_constructor +2. Nothing was
    # removed, and the two unmoved buckets (exception_constructor,
    # inherited_or_default_constructor) are the ones a recategorisation
    # would have disturbed.
    # 2026-09-15 (372 PART 14-M): function 3,778 -> 3,783 and
    # dataclass_constructor 472 -> 473, the same +6 as the total; no other
    # bucket moved, so nothing changed category.
    assert Counter(item.category for item in callables) == {
        # 2026-09-14, +45 total measured per category against 49c1189f7:
        # function +32, method +5, dataclass_constructor +5,
        # exception_constructor +3, and the three constructor buckets
        # unmoved. The per-category split is the point -- a +45 arriving as
        # +45 functions would be a different event from this one, and the
        # total alone cannot tell them apart.
        # 3,777 -> 3,778 on 2026-09-15, +1: `mark_to_start_on` is a
        # module-level function, so it lands here and in no other category --
        # the same single arrival as the total above. Subtracted, 3,777.
        # 3,778 -> 3,783 on 2026-09-15 (372), +5: five of its six arrivals
        # are module-level functions; AlignedField is the sixth, below.
        # -7 on 2026-09-15 with the old OPS engine: ops_preprocess,
        # stitch_cycle_wells, get_preprocess_ops_settings,
        # align_image_to_stitch and the three spacr.settings defaults.
        # +11 on 2026-09-15: 412's seven and 416's four module-level
        # functions. Subtracted, 3,776.
        # 2026-09-20: see the note above this assertion. The seven
        # buckets sum to 9,047, which is the total pinned there.
        "function": 4_100,
        # -9 on 2026-09-15: the seven spacrStitcher methods,
        # StitchedMultiAligner.align and FOVAlignAndCropper.run.
        "method": 3_987,
        # -3 on 2026-09-15: spacrStitcher, StitchedMultiAligner and
        # FOVAlignAndCropper.
        "constructor": 418,
        # 473 -> 474 on 2026-09-15, +1: `SearchThresholds` is a frozen
        # dataclass, so it lands here and in no other category.
        # +2 on 2026-09-15: spacr.install_cleanup.InstallRecord and
        # RemovalReport. Subtracted, 474.
        "dataclass_constructor": 500,
        "namedtuple_constructor": 13,
        "exception_constructor": 147,
        "inherited_or_default_constructor": 58,
    }
    # 8,493 -> 8,530, the same +37: every new callable is rendered by
    # autoapi, so this tracks the total rather than diverging from it. The
    # two cli-only and three compatibility entries are unchanged, which is
    # the part worth asserting -- a new symbol that reached only the CLI
    # would be a different event.
    # 8,530 -> 8,577, the same +47: every new callable is rendered by
    # autoapi, so this tracks the total. `cli_only` and `compatibility` are
    # unchanged, which is the part worth asserting -- a new symbol reaching
    # only the CLI would be a different event.
    # `autoapi` 8,579 -> 8,580, the same arrival again: a public function
    # in a rendered module is exposed by autoapi and by nothing else, so
    # `cli_only` and `compatibility` are unmoved. Those two are the buckets
    # that would catch a symbol reaching the user by some other route.
    # `autoapi` 8,583 -> 8,646, the same +63: every Map Barcodes callable is
    # rendered by autoapi and by nothing else, so `cli_only` and
    # `compatibility` are unmoved. Those two are the buckets that would catch
    # a symbol reaching the user by some other route.
    # `autoapi` 8,692 -> 8,698 on 2026-09-15, the same +6: 372's callables
    # reach the user by no other route.
    assert Counter(item.exposure for item in callables) == {
        # 8,646 -> 8,691, the same +45: every new callable is rendered by
        # autoapi and by nothing else, so `cli_only` and `compatibility` are
        # unmoved. Those two are the buckets that would catch a symbol
        # reaching the user by some other route.
        # 8,691 -> 8,692 on 2026-09-15, the same +1: `mark_to_start_on` is
        # rendered by autoapi and by nothing else. Subtracted, 8,691.
        # 8,692 -> 8,698 on 2026-09-15, the same +6: 372's six callables
        # reach the user through autoapi and no other route.
        # 8,698 -> 8,679 on 2026-09-15, the same -19: every deleted callable
        # was rendered by autoapi and by nothing else.
        # 8,679 -> 8,680 on 2026-09-15, +1: `SearchThresholds` is rendered
        # by autoapi and by nothing else.
        # 8,680 -> 8,693 on 2026-09-15, the same +13: 412's and 416's
        # callables are rendered by autoapi and by nothing else.
        # 8,693 -> 9,042 on 2026-09-20, the same +349: every callable
        # that arrived in the five days is rendered by autoapi and by
        # nothing else, so cli_only and compatibility are unmoved --
        # which is what those two buckets are for.
        "autoapi": 9_218,
        "cli_only": 2,
        "compatibility": 3,
    }
    # 8,469 -> 8,470 and 8,455 -> 8,456 in the single-variant bucket: the
    # one new callable has one signature, like almost every other. The
    # seven two-variant entries are unchanged, which is the part worth
    # asserting -- a new overload pair would be a different event.
    # 8,472 -> 8,466 with the six retired space accessors.
    # 8,466 -> 8,515, the same +49: none of the new callables carries a
    # second prose variant, so variants track callables one for one.
    # 8,515 -> 8,505, the same -10 as every other callable figure: none
    # of the withdrawn ones carried a second prose variant either, so
    # variants keep tracking callables one for one.
    # 8,505 -> 8,542, the same +37: none of the new callables carries a
    # second prose variant either, so variants keep tracking callables one
    # for one and the seven two-variant entries are unchanged. That last
    # part is the one worth asserting -- a new overload pair would be a
    # different event from a new callable.
    # 8,542 -> 8,589, the same +47: none of the new callables carries a
    # second prose variant, so variants keep tracking callables one for one
    # and the seven two-variant entries are unchanged.
    # 8,589 -> 8,592 and the single-variant bucket 8,575 -> 8,578, the same
    # arrivals as the total: none of them carries a second prose variant, so
    # variants keep tracking callables one for one and the seven two-variant
    # entries are unchanged. That last part is the one worth asserting -- a
    # new overload pair would be a different event from a new callable.
    # 8,595 -> 8,658, the same +63: none of the Map Barcodes callables
    # carries a second prose variant, so variants keep tracking callables one
    # for one and the seven two-variant entries are unchanged. That last part
    # is the one worth asserting -- a new overload pair would be a different
    # event from a new callable.
    # 8,658 -> 8,703, the same +45: each new callable has exactly one
    # variant, so this tracks the inventory rather than diverging from it.
    # Diverging is the interesting case and the reason it is counted apart.
    # 8,703 -> 8,704 on 2026-09-15, the same +1: one signature, one variant.
    # Subtracted, 8,703.
    # 8,704 -> 8,710 on 2026-09-15, the same +6: each of 372's six
    # callables has exactly one prose variant.
    # 8,710 -> 8,691 on 2026-09-15, the same -19: each deleted callable had
    # exactly one prose variant.
    # 8,691 -> 8,692 on 2026-09-15, +1: `SearchThresholds` has one
    # signature, so variants keep tracking callables one for one.
    # 8,692 -> 8,705 on 2026-09-15, the same +13: each of 412's and 416's
    # callables has exactly one prose variant.
    # 8,705 -> 9,054 on 2026-09-20, moving with the inventory above.
    assert sum(item.variant_count for item in callables) == 9_230
    # The single-variant bucket 8,581 -> 8,644, the same +63, and the
    # two-variant bucket is unchanged at 7.
    # Single-variant bucket +6 on 2026-09-15; the two-variant bucket stays 7.
    assert Counter(item.variant_count for item in callables) == {
        # 8,689 -> 8,690 on 2026-09-15 with `mark_to_start_on`.
        # 8,690 -> 8,696 on 2026-09-15 with 372's six, one variant each.
        # 8,696 -> 8,677 on 2026-09-15: the 19 deleted callables, one variant each.
        # 8,678 -> 8,691 on 2026-09-15 with 412's and 416's 13, one variant each.
        # 8,691 -> 9,040 on 2026-09-20: every callable that arrived in
        # the five days has exactly one variant, and the seven
        # two-variant ones are unmoved.
        1: 9_216,
        2: 7,
    }
    # RE-RECORDED 2026-09-05: 92 -> 171 -> 177 -> 185 -> 199 -> 205 -> 212 -> 220 -> 238 -> 250 -> 264 -> 279 -> 297 -> 311 -> 320 -> 330. Every one of those is a
    # constructor that gained an ``__init__`` docstring, so the number is the
    # documentation count and it may only go up; a fall means prose was lost.
    # 393 -> 394 on 2026-09-10: one constructor gained an __init__
    # docstring. The direction check the comment above states still
    # holds -- it rose.
    #
    # 394 -> 393 LATER THE SAME DAY, AND THIS ONE IS A FALL. The rule
    # above says a fall means prose was lost, and that is the right alarm
    # -- but the event here is different and is why the number moves
    # rather than the rule. `TourPilot.__init__`'s prose was WITHDRAWN
    # with the class, which was never public API: it has one caller,
    # inside `fractal_travel` itself. The prose still exists in the
    # source and still explains the constructor to anyone reading it; it
    # simply is not on a surface nine locales have to translate.
    #
    # A LOST DOCSTRING AND A WITHDRAWN SYMBOL LOOK IDENTICAL TO THIS
    # ASSERTION, which is the reason to write down which one happened.
    # If this falls again with no note beside it, assume the first.
    # 393 -> 394 on 2026-09-11, and BOTH sums move together, which is the
    # shape that says a constructor was added rather than a docstring lost.
    # The one is `spacr.qt.screens.embeddings.EmbeddingsScreen`, 386's
    # screen: a QWidget subclass, so it is a `constructor` rather than a
    # `dataclass_constructor`, and its `__init__` carries prose. Every
    # other new callable this day is a function, a method or a dataclass.
    # 394 -> 396 on 2026-09-13, and BOTH sums move together, which is the
    # shape that says constructors were added rather than docstrings lost.
    # The two are `spacr.qt.screens.map_barcodes.BarcodeSearchPanel` and
    # `spacr.qt.widgets.read_view.ReadView` -- QWidget subclasses, so they
    # are `constructor` rather than `dataclass_constructor`, and both
    # `__init__` carry prose. Every other Map Barcodes callable is a
    # function, a method or a dataclass.
    # 396 -> 393 on 2026-09-15, and BOTH sums move together again: the old OPS
    # engine's deletion took 3 constructors whose `__init__` carried prose
    # (FOVAlignAndCropper, StitchedMultiAligner, spacrStitcher), each with one variant, and no docstring was lost elsewhere.
    # 393 -> 408 on 2026-09-20. The direction check the comment above
    # states still holds: it rose, so no constructor prose was lost.
    assert sum(
        item.constructor_prose_variant_count for item in callables
    ) == 418
    assert sum(
        item.constructor_prose_variant_count > 0 for item in callables
    ) == 418
    # RE-RECORDED 2026-09-07, and the direction check still holds: every
    # figure moved UP with the nine new callables and not one fell.
    # 16,654 -> 16,681 parameters and 8,436 -> 8,452 required. The
    # constructor-prose sums do NOT move, which is the expected shape --
    # the nine are functions and a dataclass, so none of them is a
    # constructor that gained an ``__init__`` docstring.
    # 16,681 -> 16,685 and 8,452 -> 8,453 on 2026-09-08. Four parameters
    # from three callables, and only one of them is required:
    #
    #   +2  active_learning.retrain_round gained `balance` and
    #       `synthetic_negatives`, both keyword-with-default.
    #   +1  qt.path_probe.prime gained `want_dir`, likewise.
    #   +1  RegexEditorDialog.resizeEvent is the new callable, and its
    #       `event` is the one REQUIRED parameter in the set -- which is
    #       why the required total moves by one where the parameter total
    #       moves by four.
    # 16,694 -> 16,690: four parameters left with the six retired space
    # accessors (two took none). Required falls 8,454 -> 8,452.
    # 16,690 -> 16,837 on 2026-09-10: the OPS surface again. Required
    # rises 8,452 -> 8,521 over the same span, so 147 new parameters
    # carry 69 required ones and the rest are optional -- which is
    # what a module of tuned numerical entry points looks like.
    # 16,837 -> 16,823 and 8,521 -> 8,514 with the ten withdrawn
    # callables: fourteen parameters and seven required ones leave with
    # them. The ratio holds -- these are ordinary helpers, not the tuned
    # numerical entry points the note above describes.
    # 16,823 -> 16,933 on 2026-09-11 with the merge from main and one
    # night's Qt work: 110 parameters on 37 new callables, three apiece,
    # which is what a surface of numerical entry points and small
    # dataclasses looks like. Required rises 8,514 -> 8,584 over the same
    # span, so 110 new parameters carry 70 required ones -- the same ratio
    # the OPS note above records, and the reason that ratio is worth
    # keeping in view: a surface whose new parameters were nearly all
    # REQUIRED would be one that had stopped taking defaults seriously.
    # 16,933 -> 17,058 and 8,584 -> 8,662 on 2026-09-11: 125 parameters on
    # 47 new callables, 78 of them required. Between two and three apiece,
    # which is what a surface of small dataclasses and module-level helpers
    # looks like. The required share is 62%, close to the ratios above --
    # the reason to keep that ratio in view is unchanged: a surface whose
    # new parameters were nearly all REQUIRED would be one that had stopped
    # taking defaults seriously. All 78 are documented, which is why
    # `test_no_new_undocumented_required_public_parameters` does not move.
    # 17,063 -> 17,064: `border_rules_agree(db_path)` takes exactly one
    # parameter, so this moves by one alongside the callable count. A
    # parameter total that moved WITHOUT the callable count moving would
    # mean an existing signature changed, which is a different event.
    # 17,063 -> 17,192 on 2026-09-13, +129, AND THIS ONE DID NOT TRACK THE
    # CALLABLE COUNT -- which is the event the note above says to look for.
    # The 63 new Map Barcodes callables carry 127 parameters between them.
    # The other two are a signature change to callables that already existed:
    #
    #     spacr.sequencing.paired_read_chunked_processing  +barcode_set
    #     spacr.sequencing.single_read_chunked_processing  +barcode_set
    #
    # Both optional, which is why the REQUIRED sum below moves by exactly the
    # 86 the new callables bring and not by 88. Measured by diffing per-symbol
    # parameter sets against 5a2825f67, not inferred from the totals: +129
    # against +127 is a two-parameter discrepancy that a total alone reports
    # as an unremarkable increase.
    # 17,192 -> 17,293 on 2026-09-14, +101 -- AND ONLY 99 OF THOSE COME FROM
    # THE 45 NEW CALLABLES. The other two are `row_offsets`, added to the
    # EXISTING `spacr.ops_layout.WellLayout` and
    # `spacr.ops_phenotype.phenotype_site_map` by 372. Diffing per-symbol
    # parameter sets says so; the total alone reports +101 as an unremarkable
    # increase and cannot distinguish 101 arrivals from 99 arrivals and two
    # existing signatures widening. Both are optional, which is why the
    # REQUIRED sum below moves by exactly the 60 the new callables bring.
    # 17,293 -> 17,295 on 2026-09-14, +2, and NOT from the two dunders item
    # 368 documented -- those move the API surface, not this inventory. Both
    # are `spacr.qt.settings_pack.PackReport` gaining the fields `elsewhere`
    # and `source` (item 317). Established by line-differencing the digest
    # below against origin/nightly: +0 / -0 symbols and exactly ONE existing
    # line changed, that one. Both fields carry defaults, so the required
    # sum below does not move -- the same asymmetry 372's `row_offsets`
    # produced, and the reason the two sums are counted apart.
    # 17,295 -> 17,296 on 2026-09-15, +1, and again NOT a new callable: 410
    # adds the field `channel_scale` to the EXISTING
    # `spacr.embeddings.EmbeddingSpec`. Line-differencing the digest below
    # against 410's parent, 75dfdcae8: +0 / -0 symbols and exactly ONE
    # existing line changed, that one. The field defaults to None, so the
    # required sum below stays at 8,815.
    # 17,296 -> 17,297 on 2026-09-15, +1, and not from 410: item 333 gave the
    # EXISTING `spacr.qt.widgets.live_preview.LivePreviewPanel` a `module`
    # parameter (default ""), so the preview can resolve the model the way the
    # module's own run does. It landed on nightly after 410's pin was taken.
    # Optional, so the required sum below does not move.
    # 17,297 -> 17,300 on 2026-09-15, +3, AND ONLY TWO COME FROM THE NEW
    # CALLABLE: `mark_to_start_on(shape, fallback_mark)` (293). The third is
    # 317 giving the EXISTING `AppScreen.apply_settings_that_came_with` a
    # keyword-only `pack_folder=None`, so a cached example reads the shipped
    # pack before the plate's own `settings/` folder. PROVED BY SUBTRACTION on
    # the inventory: without the symbol the sum is 17,298, and without the
    # symbol AND `pack_folder` it is 17,297, the previous pin.
    # 17,300 -> 17,331 on 2026-09-15, +31: +27 from 372's six arrivals and
    # +4 from its four changed signatures -- WellLayout swaps row_offsets
    # for half_tile (0), round_well_layout gains half_tile (+1),
    # phenotype_site_map trades sbs_sites and row_offsets for sbs_centres,
    # anchors and tile_shape (+1), call_reads gains normalise and gpu (+2).
    # 17,331 -> 17,174 on 2026-09-15, -157, all of it on the 19 deleted
    # lines: the old OPS engine's callables and the three stitcher defaults.
    # 17,174 -> 17,181 on 2026-09-15, +7, and FIVE come from the new callable:
    # `SearchThresholds`' five fields. The other two are the EXISTING
    # `search_barcodes` and `iter_barcode_search` each gaining a keyword
    # `thresholds=None`. Every one of the seven carries a default, so the
    # required sum below does not move. Established by diffing every symbol's
    # parameter tuple against origin/nightly df1216b3f: those three symbols
    # differ and no other does.
    # 17,181 -> 17,228 on 2026-09-15, +47, all of it on the 13 new lines:
    # InstallRecord 13, RemovalReport 4, find_old_installs 1,
    # remove_install 4, run_update_sequence 7, start_update_helper 7
    # (416: 36) and download_make_masks_example 3,
    # install_test_data_button 1, is_present 1, listed_files 1,
    # load_the_test_data 3, make_masks_example_folder 0, open_the_test_data 2
    # (412: 11). Subtracted, 17,181.
    # 17,228 -> 17,225 on 2026-09-16 for 418: the existing
    # `spacr.utils.merge_split_objects` drops five optional parameters
    # (intensity_merge, intensity_split, intensity_threshold,
    # min_watershed_distance, minimum_area_to_split) and gains the two
    # keyword-only bounds min_intensity=0 and max_intensity=0. Compared every
    # inventory row with f7df13e93 (/tmp/spacr-publish.tDTL40): +0/-0 symbols,
    # exactly that one changed row, and no required/variant/category changes.
    # Restoring its complete baseline row also restores the 17,228 total.
    # 17,225 -> 17,883 on 2026-09-20, the parameters of the 349 new
    # callables; see the note at the inventory total.
    assert sum(len(item.parameters) for item in callables) == 18_287
    # 8,665 -> 8,666: `db_path` has no default, so the one new parameter is
    # also a required one and both parameter sums move by the same one.
    # 8,669 -> 8,755, +86, all of it from the new callables: `barcode_set`
    # has a default on both sequencing functions, so neither is required.
    # 8,755 -> 8,815, +60, and ALL 60 come from the new callables -- the two
    # `row_offsets` parameters 372 added to existing signatures both carry a
    # default, so they move the parameter sum above and not this one. That
    # asymmetry is the check: an optional parameter that moved this number
    # would be a required one, and a different event.
    # Unmoved on 2026-09-15 by 410's `EmbeddingSpec.channel_scale`, which has
    # a default -- measured, not assumed: 8,815 at 75dfdcae8 and after 410.
    # 8,815 -> 8,817 on 2026-09-15, +2: both of `mark_to_start_on`'s
    # parameters are required. 317's `pack_folder` defaults to None, so it
    # moves the parameter sum above and not this one -- subtracting the symbol
    # alone returns 8,815.
    # 8,817 -> 8,833 on 2026-09-15, +16: +15 from 372's arrivals and +1
    # from phenotype_site_map, whose required sbs_sites became the
    # required pair sbs_centres and anchors.
    # 8,833 -> 8,812 on 2026-09-15, -21, likewise all on the deleted lines.
    # 8,812 -> 8,830 on 2026-09-15, +18, likewise all on the new lines:
    # InstallRecord 4, RemovalReport 1, remove_install 1,
    # run_update_sequence 1, start_update_helper 2 (416: 9) and
    # download_make_masks_example 3, install_test_data_button 1,
    # is_present 1, listed_files 1, load_the_test_data 1,
    # open_the_test_data 2 (412: 9); find_old_installs and
    # make_masks_example_folder have no required parameter. Subtracted, 8,812.
    # 8,830 -> 9,214 on 2026-09-20, moving with the parameter total
    # above.
    assert sum(len(item.required_parameters) for item in callables) == 9_429
    # Moved again 2026-09-15 for 333's `LivePreviewPanel.module`, proved by
    # subtraction on the full inventory: without that one parameter the digest
    # is 5b30fe1f... (410's pin, byte for byte); without it AND 410's
    # `EmbeddingSpec.channel_scale` it is 3714f4a1..., the pin before 410.
    # Moved again 2026-09-15 for 293 and 317, proved by subtraction on the
    # full inventory. Dropping `spacr.graph_types.mark_to_start_on` alone gives
    # 7728c1ef..., which is NOT the pin: one existing line also changed,
    # `AppScreen.apply_settings_that_came_with`, whose `parameters` and
    # `accepted_documented_parameters` gained `pack_folder` (its
    # `required_parameters` did not). Taking `pack_folder` back out of that
    # line as well returns 3476522c..., the previous pin, byte for byte. So the
    # move is one new function and one optional keyword on an existing method,
    # and nothing else among 8,696 symbols changed.
    # Moved 2026-09-15 for 412 and 416, proved by subtraction on the full
    # inventory: dropping the 13 new lines (spacr.qt.make_masks_demo's seven
    # functions, spacr.install_cleanup's four and its two dataclasses)
    # returns 2494eb63..., the previous pin, byte for byte. No existing line
    # changed.
    # Moved 2026-09-16 for 418, PROVED against f7df13e93 using the same
    # source-derived _public_callables helper on both trees. Replacing only
    # merge_split_objects' full current row with its baseline row restores
    # 6bb6e91888a29f1795a6af5a4fc180682b9735a21582dd8432ca5b44cf736802
    # exactly. Both parameters and accepted_documented_parameters changed;
    # its sole required parameter remains mask_src. All other 8,697 rows,
    # 8,705 variants and 8,830 required parameters are unchanged.
    assert _sha256_lines(
        f"{item.symbol}\0{item.category}\0{item.exposure}\0"
        f"{','.join(sorted(item.parameters))}\0"
        f"{','.join(sorted(item.required_parameters))}\0"
        f"{','.join(sorted(item.accepted_documented_parameters))}\0"
        f"{int(item.accepts_arbitrary_keywords)}\0"
        f"{item.variant_count}\0{item.docless_variant_count}\0"
        f"{item.constructor_prose_variant_count}"
        for item in callables
    # Moved with the counts above, 2026-09-11. The digest covers every
    # symbol's category, exposure, parameter names and variant counts, so
    # 37 new callables move it whatever else stays still -- which is the
    # point: it is the assertion that catches a required parameter becoming
    # optional without any count changing.
    #
    # Moved again 2026-09-12, and this time the move was PROVED to be the
    # one arrival rather than assumed. Recomputing the digest over the
    # inventory with `spacr.infection.border_rules_agree` removed returns
    # 1dfbc784... -- the previous pin, byte for byte. So nothing else in
    # 8,585 symbols changed category, exposure, parameter names or variant
    # counts, which is exactly what a bare hash bump cannot tell anybody.
    # That subtraction is cheap and is the right way to move this line.
    #
    # Moved again 2026-09-12 for 364's two resolvers, and proved the same
    # way: the digest recomputed without `split_role_setting` and
    # `surviving_setting_name` returns 28dea135..., the previous pin, byte
    # for byte. Nothing else among 8,587 symbols moved.
        #
    # Moved again 2026-09-13 for Map Barcodes, and PROVED the same way
    # rather than bumped. The digest recomputed over the 8,588 baseline
    # symbols -- the 63 arrivals dropped and the two changed lines restored
    # to their 5a2825f67 form -- returns d21a9ee1..., the previous pin, byte
    # for byte. So the whole move is those 63 plus `barcode_set` on the two
    # sequencing functions, and nothing else among 8,588 symbols changed
    # category, exposure, parameter names or variant counts.
    #
    # THE FIRST SUBTRACTION I TRIED DID NOT MATCH, and the reason is worth
    # keeping: I restored `parameters` on the two sequencing functions and
    # forgot `accepted_documented_parameters`, which gained `barcode_set`
    # too because the docstrings were updated in the same commit. A digest
    # that failed to match would have read as "something unexplained moved"
    # when what had actually moved was my reconstruction. Subtract using the
    # recorded baseline LINE, not a field-by-field rebuild of it.
    # REGENERATED 2026-09-20 with the counts above, all of which rose
    # and none of which fell. What this digest is for is unchanged: a
    # row that changes without any count changing still moves it.
) == "f18dfc96af55133ce386b72ae1923d847f050d98bbd188c54e497a10641aefea"
    # Moved 2026-09-15 for `SearchThresholds` and `thresholds`, proved by
    # subtraction on the full inventory on top of origin/nightly df1216b3f.
    # Dropping the one new symbol alone is NOT enough, because two existing
    # lines also changed: `search_barcodes` and `iter_barcode_search`, whose
    # `parameters` and `accepted_documented_parameters` gained `thresholds`
    # (their `required_parameters` did not). Dropping the symbol AND restoring
    # those two lines to the base tree's form returns a80114b9..., the
    # previous pin, byte for byte. So the move is one new dataclass and one
    # optional keyword on two existing functions, and nothing else among
    # 8,685 symbols changed.
    # Moved 2026-09-15 when the old OPS engine was deleted, and PROVED by
    # subtraction: the switch commit's inventory returns b6445699..., the previous
    # pin, and with its 19 deleted lines dropped it returns a80114b9...,
    # this tree's digest, byte for byte. Nothing else changed.
    # Moved 2026-09-15 for 372 PART 14-M and PROVED by subtraction on
    # nightly 3f27b926a: nightly's own inventory returns f59e27a9..., the
    # previous pin, and this tree's with the six arrivals dropped and the
    # four changed lines restored to nightly's form returns f59e27a9...
    # too. Line-differencing the two inventories gives +6 / -0 and exactly
    # four changed lines -- spacr.ops_layout.WellLayout and
    # round_well_layout, spacr.ops_phenotype.phenotype_site_map and
    # spacr.ops_sbs.call_reads -- which the parameter sums above account
    # for. Before the rebase the same move took 3476522c... to 9f3d5e2d....
    # Moved 2026-09-14, and PROVED rather than assumed, the way this file
    # asks: the same digest recomputed over the tree at 49c1189f7 returns
    # 487529aa5a65... byte for byte, which is the value this line carried
    # before. So the apparatus is right and the move is only what changed.
    #
    # Line-level diff against that baseline: +45 / -0, and TWO EXISTING
    # LINES CHANGED -- spacr.ops_layout.WellLayout and
    # spacr.ops_phenotype.phenotype_site_map, both gaining `row_offsets`
    # from 372. That is precisely what this digest exists to catch and no
    # count above reports it on its own: the parameter sum moves, the
    # required sum does not, and the symbol count cannot see it at all.
    #
    # Moved 2026-09-15 for 410, and PROVED by subtraction as above. The same
    # digest recomputed over the tree at 75dfdcae8, 410's parent, returns
    # 3714f4a1... byte for byte, the value this line carried before, so the
    # apparatus is right. Line-level diff against it: +0 / -0 symbols and ONE
    # existing line changed -- `spacr.embeddings.EmbeddingSpec` gaining
    # `channel_scale` in `parameters` and `accepted_documented_parameters`
    # but not in `required_parameters`. The current inventory recomputed with
    # `channel_scale` taken back out of that one line returns 3714f4a1...
    # again, byte for byte. So the whole move is one optional dataclass field
    # and nothing else among 8,696 symbols changed.

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
    # WAS ``RunRegistry.register``, and ``LayerStack.add_image`` before that.
    # Both have since been documented. The example has to be a callable that
    # is STILL docless, or this assertion stops standing for the class of
    # defect it was written for -- so it moves each time 368 reaches it.
    # THERE IS NO DOCLESS EXAMPLE LEFT, so the assertion inverts.
    #
    # This named a callable that was still docless, as a live example of the
    # defect class the boundary was written for. The example moved four times
    # -- LayerStack.add_image, RunRegistry.register,
    # LinkedSelection.clear_filter, fractal_travel.set_num_threads -- each
    # documented by 368 in turn, and the last of them exhausted the supply.
    #
    # An example that cannot exist is a stronger statement than any example,
    # so it is made directly: every public callable in the package carries a
    # docstring, and a new docless one fails here by name.
    docless_now = sorted(
        item.symbol for item in _public_callables()
        if item.docless_variant_count
    )
    assert docless_now == [], (
        f"public callables with no docstring: {docless_now}")
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
        # TWO VARIANTS IS THE LOAD-BEARING HALF and still holds: each of
        # these is defined twice, once compiled by Numba and once as a
        # fallback in the `except ImportError` branch, and the inventory has
        # to count VARIANTS rather than symbols or the second definition is
        # invisible to every gate here.
        assert by_symbol[symbol].variant_count == 2
        # WAS 1. The fallback branches were the docless half; 368 documented
        # them, so both variants now carry text. The pairing is what this
        # test is about, not the gap, and the gap closing is the good
        # outcome rather than a reason to move the example.
        assert by_symbol[symbol].docless_variant_count == 0

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
    # 627 -> 526 on 2026-09-04: every public method in
    # `spacr.qt.dnd_handlers` now documents itself. 368's rule is that the
    # improvement is banked in the commit that earns it, or documenting a
    # hundred callables silently buys room to leave a hundred more.
    assert len(docless) == 0
    assert sum(
        item.docless_variant_count for item in _public_callables()
    ) == 0
    assert _sha256_lines(docless) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
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
    # 9,427 -> 9,441. Seven public symbols were added earlier today and
    # seven drop-handler methods stopped being aliases, so they now carry
    # their own entry instead of borrowing one.
    # 10,152 -> 10,172 -> 10,213 (2026-09-05): private methods on public
    # classes that gained a docstring get an entry of their own.
    # 10,213 -> 10,230 -> 10,241 (2026-09-07). FROZEN here by agreement:
    # the other session stopped adding public symbols so the localization
    # catalogs could be regenerated against a stable surface.
    # 10,241 -> 10,242 (2026-09-08): the freeze is over and the catalogs
    # were regenerated against it. The one entry is
    # RegexEditorDialog.resizeEvent. The extractor's own ratchet moved
    # 10,237 -> 10,242 over the same span and decomposes the difference:
    # +12 admitted, -7 retired with spacr.seg_metrics gone.
    # 10,243 -> 10,237 with the same six retirements.
    # 10,237 -> 10,306 on 2026-09-10, +69/-6, MEASURED against the catalog
    # of the commit that last set this number rather than reconstructed:
    #     12  spacr.ops_layout      the round well's geometry
    #     11  spacr.ops_stitch      the driver and StitchedWell
    #     10  spacr.ops_phenotype   Phase A4
    #      8  spacr.ops_register    Registration and phase correlation
    #      5  spacr.ops_accel       the three array primitives
    #      4  spacr.ops_cycles      cycle registration without a nuclear stain
    #      2  spacr.ops_solve       the placement solve
    #     17  spacr.qt              layout_policy (5), TourPilot (5),
    #                               two qt.app helpers, the language scope,
    #                               and the rest of two sessions' Qt work
    #     -6  the space accessors, already accounted for above
    # THIS FILE'S RATCHETS WERE SIX DAYS BEHIND, which is the thing worth
    # noticing rather than the number: instruction 372's whole OPS surface
    # had landed and none of these four assertions had been re-recorded, so
    # every one of them failed at once the first time the full non-Qt sweep
    # ran to the end. A ratchet nobody runs is a ratchet nobody updates.
    # 10,306 -> 10,299, -12/+5. 2026-09-10, SECOND MOVE THAT DAY: four symbols added the night
    # before came OFF the public surface. `open_at_the_measured_width`
    # and `the_missing_pip_escape` have one caller each inside
    # `qt/app.py`, `TourPilot` is instantiated once inside
    # `fractal_travel`, and `layout_policy` is read by nothing but
    # `app.py` -- none of the four was API, and twelve entries were
    # about to be translated into nine languages for helpers nobody
    # outside the package calls.
    # A MODULE KEEPS ITS DOCSTRING ON THE INVENTORY whatever it is
    # called, so `_layout_policy` and its four functions come back under
    # the new name; only the underscore on a FUNCTION or CLASS removes
    # one. That is why twelve leave and five return.
    # 10,294 -> 10,339 on 2026-09-11, +45/-0, the merge from main plus one
    # night's Qt work. THE PIN WAS 144 COMMITS BEHIND, which is the same
    # observation this file already records once: "THIS FILE'S RATCHETS WERE
    # SIX DAYS BEHIND ... A ratchet nobody runs is a ratchet nobody updates."
    # It failed on main as well as here, so the merge revealed it rather than
    # caused it. Every one of the 45, by the item that asked for it:
    #
    #     9  spacr.embeddings          386, the embedding engine
    #     9  spacr.ops_compose         372, the composed OPS window
    #     7  spacr.bystanders          388, the infected cell's neighbours
    #     7  spacr.qt.widgets.dose_response
    #                                  387, selectivity index and the Bliss
    #                                  and Loewe surfaces
    #     5  spacr.point_patterns      388, Ripley's K and L
    #     3  spacr.model_zoo           370, the scorecard on a model row
    #     2  spacr.scorecard           370, reading one back
    #     2  spacr.qt.theme            380, the per-window stylesheet:
    #                                  apply_stylesheet_per_window and
    #                                  window_stylesheet
    #     1  spacr.infection           377, uninfected_cells_were_measured
    #
    # A MODULE COUNTS AS ONE ENTRY ON TOP OF ITS SYMBOLS -- bystanders,
    # embeddings, ops_compose and point_patterns are new files, so each
    # brings its own module docstring as well.
    #
    # 10,339 -> 10,394 on 2026-09-11, the same +55 that
    # `test_api_i18n_extractor` decomposes by module -- this test and that
    # one read the same inventory from opposite sides, which is the whole
    # point of the cross-check, so they move together or one of them is
    # wrong.
    #
    # 10,394 -> 10,395 with 380's `set_a_sheeted_widgets_own_rule`, merged
    # after that measurement. Re-measured here rather than carried across:
    # this assertion and the extractor's move together by construction.
    # 10,397 -> 10,398 with 377's `border_rules_agree`, and the extractor's
    # own pin moved in the same change, which is what this cross-check is
    # for: the two are measured by different code and agreeing is the
    # evidence. One moving alone would mean the two disagree about what the
    # public surface is.
    # 10,401 -> 10,478 on 2026-09-13, +77 for Map Barcodes (c753b7de7). The extractor's
    # own pin moved in the same change and by the same amount, which is what
    # this cross-check is for: the two are measured by different code and
    # agreeing is the evidence. One moving alone would mean the two disagree
    # about what the public surface is.
    # 2026-09-14: 10,478 -> 10,531, +53 / -0 by set difference. THIS IS ONE
    # OF FOUR FILES CARRYING THIS NUMBER -- the others are
    # test_api_i18n_extractor (twice), test_api_i18n_frontend and
    # test_documentation_i18n. Moving one and not the rest is how a full
    # sweep found three of them a day late; grep for the literal before
    # believing a single edit was enough.
    # 2026-09-15: 10,533 -> 10,534, +1 / -0 by set difference:
    # `spacr.graph_types.mark_to_start_on`. Dropping that key returns 10,533.
    # All four files moved in the same commit, and the extractor's own pin
    # moved by the same one, which is what this cross-check is for.
    # 10,534 -> 10,541 on 2026-09-15: the seven 372 arrivals named in
    # test_api_i18n_extractor, +7/-0 by set difference against 3f27b926a.
    # 10,541 -> 10,521 on 2026-09-15: the twenty symbols the old OPS engine's
    # deletion removed, named in test_api_i18n_extractor, +0/-20.
    # 2026-09-15: 10,521 -> 10,523, +2 / -0 by set difference:
    # `spacr.barcode_search.SearchThresholds` and its `__post_init__`, the
    # validator that refuses thresholds unable to decide anything. The
    # extractor's own pins moved by the same two in the same commit, and so
    # did test_api_i18n_frontend and test_documentation_i18n.
    # 10,523 -> 10,539 on 2026-09-15: 412's and 416's sixteen public
    # symbols; subtracted, 10,523.
    # 10,539 -> 10,947 on 2026-09-20; see the note at the callable
    # inventory above. The extractor's own pin moved to 10,931 earlier
    # the same day and object_classifier's public functions landed
    # between the two measurements, which is the difference.
    # 10,947 -> 11,150 on 2026-09-21, +203 / -0 by set difference against
    # 6ae5e1b36: the 176 callables named at the callable inventory above
    # plus the module and class docstrings that came with them. The
    # extractor's own pin moved by the same count in the same commit.
    assert len(docs) == 11_150
    # 7,745 -> 7,853: the 101 drop-handler methods and the seven public
    # symbols added earlier today all render their own docstring now.
    # 8,457 -> 8,458 on 2026-09-08 with the same one entry moving every
    # figure in this test: RegexEditorDialog.resizeEvent is both a public
    # callable and a rendered documented one.
    # 8,460 -> 8,454: the six space accessors retired under 364.
    # 8,454 -> 8,503, the same +49: every one of the new callables is
    # rendered, so this tracks the total rather than diverging from it.
    # 8,503 -> 8,493, the same -10 as the callable total above: every
    # withdrawn callable was a rendered one, so this keeps tracking the
    # total rather than diverging from it.
    # 8,493 -> 8,530, the same +37 as the callable total: every new
    # callable is rendered, so this keeps tracking the total rather
    # than diverging from it.
    # 8,530 -> 8,577, the same +47 as the callable total: every new
    # callable is rendered, so this keeps tracking the total rather than
    # diverging from it.
    # 8,579 -> 8,580, the same single arrival: `border_rules_agree` is
    # rendered AND documented, so it lands in both this set and the
    # extractor's, which is what makes the two halves agree.
    # 8,583 -> 8,646, the same +63 as the total: every Map Barcodes callable
    # is both rendered and documented, so this tracks the inventory instead
    # of diverging from it. A divergence here would mean a new callable that
    # the API pages do not render.
    # 8,646 -> 8,691, the same +45 as the exposure counter above.
    # 8,691 -> 8,692 on 2026-09-15, the same +1 as the exposure counter:
    # `mark_to_start_on` is both rendered and documented.
    # 8,692 -> 8,698 on 2026-09-15, the same +6 as the callable total: all
    # six 372 arrivals are rendered and documented.
    # 8,698 -> 8,679 on 2026-09-15, the same -19 as the callable total.
    # 8,679 -> 8,680 on 2026-09-15, the same +1 as the exposure counter:
    # `SearchThresholds` is both rendered and documented.
    # 8,680 -> 8,693 on 2026-09-15: their 13 documented, rendered callables.
    # 8,693 -> 9,042 on 2026-09-20; the inventory's note accounts for it.
    assert len(rendered_documented_callables) == 9_218
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

    # 34 -> 36 on 2026-09-10: `ops_phenotype.Alignment` and
    # `ops_stitch.StitchedWell`, both dataclasses whose constructors
    # are generated and whose fields are therefore required ivars.
    # 36 -> 37 on 2026-09-13: `spacr.qt.screens.map_barcodes
    # .BarcodeSearchPlan`, a dataclass whose constructor is generated and
    # whose six fields -- anchor, fastq_files, other_samples, problem,
    # reference_tables, sample -- are therefore required ivars. Same shape as
    # the two 2026-09-10 additions above.
    # 37 -> 38 on 2026-09-13: `spacr.settings.BarcodeEntry`, and it did NOT
    # arrive -- it was already here. Commit ec3ee9132 rewrote its numpydoc
    # `Parameters` block as `:ivar:` fields so the prose would split into
    # translatable blocks, and `IVAR_FIELD` matches `:ivar name:` where it
    # matched nothing before. Only `name` is a required parameter; the other
    # six fields carry defaults, which is why one symbol brings one field.
    #
    #   A DOCSTRING FORMAT CHANGE MOVES THIS RATCHET, and that is the first
    #   time it has happened here -- every previous move was a dataclass
    #   arriving. Measured by set-differencing the symbol map against
    #   0e619178c rather than by subtracting totals: exactly one symbol added,
    #   none removed, no field set changed on any symbol present in both.
    # 2026-09-14: 38 -> 43, +5 / -0, all five spacr.curation_queue -- its
    # five dataclasses arriving with 396, which is a module landing rather
    # than a docstring format change.
    # 43 -> 44 on 2026-09-15: spacr.ops_cycles.AlignedField, the one
    # dataclass constructor 372 added, its fields documented as :ivar:.
    # 44 -> 53 on 2026-09-20, nine more documented dataclass
    # constructors from five days of modules; see the note at the
    # callable inventory above.
    # 53 -> 55 on 2026-09-21, +2 / -0 by set difference against 6ae5e1b36:
    # spacr.qt.make_masks_datasets.MaskDataset and
    # spacr.qt.widgets.plate_layout.PlateTemplate, two dataclasses that
    # arrived documenting their fields as :ivar:, ten required fields
    # between them. No field set changed on any symbol present in both.
    assert len(required_ivars) == 55
    # 156 -> 165 on 2026-09-10: nine fields across Alignment and
    # StitchedWell, the two dataclasses the count above admitted.
    # 165 -> 171, +6: the six fields of `BarcodeSearchPlan` named above.
    # The symbol count moved by one and the field count by six, which is the
    # pair worth asserting -- one moving without the other would mean a
    # dataclass changed shape rather than arrived.
    # 171 -> 172, +1: `BarcodeEntry`'s `name`, per the note above.
    # 172 -> 192, +20 fields across the five spacr.curation_queue
    # dataclasses added above. Set-differenced: five symbols added, none
    # removed, and NO FIELD SET CHANGED on any symbol present in both --
    # which is what distinguishes a module arriving from a docstring format
    # change, the two events this pair of ratchets exists to tell apart.
    # 192 -> 198 on 2026-09-15, +6: AlignedField's fields without a default
    # (stack, kept, refused, missing, channel_shifts, cycle_shifts).
    # 198 -> 217 on 2026-09-20, the fields of the nine new documented
    # dataclass constructors.
    assert sum(map(len, required_ivars.values())) == 227
    # 30 -> 32 and 145 -> 154: Alignment and StitchedWell again, with
    # their nine fields between them.
    # 32 -> 33 and 154 -> 160: `BarcodeSearchPlan` and its six fields. The
    # whole of the 36 -> 37 move above is in the GENERATED half, which is
    # what makes it a dataclass arriving rather than an ordinary callable
    # growing `:ivar:` fields -- the ordinary counterexamples below are
    # unchanged, and they are the control.
    # 33 -> 34 and 160 -> 161: `BarcodeEntry`. The whole move is again in the
    # GENERATED half and the ordinary counterexamples below are untouched,
    # which is the control that says a dataclass was admitted rather than an
    # ordinary callable growing `:ivar:` fields.
    # 34 -> 39, the five spacr.curation_queue dataclasses whose generated
    # constructor docstring is reduced to :ivar: fields.
    # 39 -> 40 on 2026-09-15: AlignedField, wholly in the GENERATED half.
    # 40 -> 48 on 2026-09-20, the new documented dataclass constructors.
    assert len(generated) == 50
    # 161 -> 181, the 20 fields of the five curation_queue dataclasses.
    # 181 -> 187 on 2026-09-15, +6: the same six AlignedField fields.
    # 187 -> 204 on 2026-09-20, moving with the generated constructors.
    assert sum(map(len, generated.values())) == 214
    # `dataclass_constructor` 31 -> 32: `BarcodeSearchPlan`. The namedtuple
    # bucket is unchanged, which is the part worth asserting -- a namedtuple
    # arriving here would be a different event.
    assert Counter(by_symbol[symbol].category for symbol in generated) == {
        # 32 -> 33: `BarcodeEntry`, a dataclass. The namedtuple bucket is
        # unchanged, which is the part worth asserting.
        # 33 -> 38: the five spacr.curation_queue dataclasses. The namedtuple
        # bucket is unchanged, which is the part worth asserting -- a
        # namedtuple arriving here would be a different event.
        # 38 -> 39 on 2026-09-15: spacr.ops_cycles.AlignedField.
        # 39 -> 44 on 2026-09-20, five more dataclasses.
        "dataclass_constructor": 46,
        # 1 -> 4 on 2026-09-20. THE BUCKET THE COMMENT ABOVE SAYS IS THE
        # PART WORTH ASSERTING HAS MOVED: three namedtuples now generate
        # ivar prose where one did. That is a different event from a
        # dataclass arriving, and it is named here rather than absorbed
        # into the total.
        "namedtuple_constructor": 4,
    }
    assert Counter(
        by_symbol[symbol].category
        for symbol in generated
        for _name in generated[symbol]
    ) == {
        # 149 -> 155: the six fields of `BarcodeSearchPlan`. The namedtuple
        # bucket is unchanged.
        # 155 -> 156: `BarcodeEntry`'s one required field, `name`.
        # 156 -> 176 on 2026-09-14: the 20 required fields across the five
        # spacr.curation_queue dataclasses. The namedtuple bucket is unchanged.
        # +6 on 2026-09-15: AlignedField's six fields without a default.
        # 182 -> 190 on 2026-09-20, and the namedtuple bucket moved too
        # (5 -> 14), which the comments above say is the part worth
        # asserting: these are namedtuple fields, not dataclass ones.
        "dataclass_constructor": 200,
        "namedtuple_constructor": 14,
    }
    # 4 -> 5 on 2026-09-20, one more ordinary class whose __init__
    # documents its parameters as :ivar:.
    assert len(ordinary) == 5
    # 11 -> 13 on 2026-09-20, with the fifth ordinary class above.
    assert sum(map(len, ordinary.values())) == 13
    assert {
        by_symbol[symbol].category for symbol in ordinary
    } == {"constructor"}

    remaining = {
        symbol: _missing_required_parameters(by_symbol[symbol])
        for symbol in generated
    }
    # 30 -> 32, tracking `generated` above: every generated constructor
    # still documents every required field, which is what the zero on
    # the next line asserts and is the point of this test.
    # 32 -> 33, tracking `generated` above: every generated constructor
    # still documents every required field, which is what the zero on the
    # next line asserts and is the point of this test.
    # 33 -> 34, tracking `generated` above.
    # 34 -> 39 on 2026-09-14, tracking `generated` above: every one of the
    # five new curation_queue constructors documents every required field,
    # which is what the zero on the next line asserts.
    # 39 -> 40 on 2026-09-15, tracking `generated`: AlignedField documents
    # every required field, which the zeros below still assert.
    # 40 -> 48 on 2026-09-20, tracking `generated`. The two zeros below
    # are unchanged, which is the assertion that matters: every one of
    # the eight new constructors documents every required field.
    assert sum(not names for names in remaining.values()) == 50
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

    # 113 -> 19 on 2026-09-04. An alias exists so an UNDOCUMENTED override
    # borrows its base class's text; 94 of them were drop-handler methods
    # that now say what THEY do, so the borrowing is not merely unnecessary,
    # it would hide the specific answer behind the generic one.
    assert len(declared_aliases) == 0
    assert len(callable_aliases) == 0
    # EMPTY NOW. These six were properties whose alias made them borrow
    # `Layer`'s text; documenting `spacr.layers` gave each its own, so both
    # registries hold the same two entries and neither has a symbol the
    # other lacks.
    assert set(declared_aliases) - set(callable_aliases) == set()
    # EMPTY SETS NOW, because the registry is empty. These said {"method"}
    # and {"autoapi"}: every alias was a rendered method borrowing a base
    # class's text. Kept rather than deleted, so that if an alias is ever
    # added again it is still held to being exactly that.
    assert {
        by_symbol[alias].category for alias in callable_aliases
    } <= {"method"}
    assert {
        by_symbol[alias].exposure for alias in callable_aliases
    } <= {"autoapi"}

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
    # 90 -> 0 over 2026-09-04. An alias inherits its base class's parameter
    # documentation as well as its prose, so most of these were drop-handler
    # methods carrying the base's debt rather than any of their own. Every
    # override documents its own parameters now, and the registry is empty,
    # so there is no borrowing left to owe anything.
    assert len(alias_debt) == 0
    assert sum(map(len, alias_debt.values())) == 0
    assert len(canonical_debt) == 0
    assert sum(map(len, canonical_debt.values())) == 0

    raw = _required_parameter_omission_inventory(items)
    deduplicated = _required_parameter_omission_inventory(
        items, callable_aliases,
    )
    # The gap IS the alias debt, so it fell with it: 140 -> 7.
    # The gap IS the alias debt, and the registry is empty, so there is none.
    assert len(raw[0]) - len(deduplicated[0]) == 0
    assert raw[1] - deduplicated[1] == {}
    assert raw[2] - deduplicated[2] == {}


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

    # 2,575 -> 2,572 on 2026-09-04, 2,284 -> 2,283 on 2026-09-05, when the
    # last constructor but one gained a documented required parameter (the
    # `constructor` row fell 2 -> 1 with it). Writing a docstring is not the whole
    # job: a documented callable whose required parameters are unexplained
    # still counts here, so the drop-handler docstrings carry `:param:` and
    # `:returns:` fields and the number goes DOWN rather than up.
    # 2,283 -> 2,281 -> 2,280 on 2026-09-08, -2/+1, and the two directions
    # are worth reading separately because only one of them is progress:
    #
    #   RESOLVED  spacr.qt.path_probe.prime:path and :answer. `prime` grew
    #             a `want_dir` keyword and was documented properly while it
    #             was open, which took its other two required parameters
    #             with it. `function` falls 758 -> 757 and its parameter
    #             count 1,130 -> 1,128.
    #
    #   ADMITTED  spacr.qt.regex_editor.RegexEditorDialog.resizeEvent:event.
    #             A Qt event override still has a required parameter, and a
    #             docstring that does not name it is the same omission as
    #             any other. `method` rises 833 -> 834, 1,009 -> 1,010.
    #
    # The total is a NET figure and a net figure hides one of these behind
    # the other, so the sum stays 1,635 while both halves moved.
    # 2,280 -> 2,278. Two of the six retired space accessors were
    # omissions rather than rendered symbols, so this falls by two where
    # the surface falls by six.
    # 2,278 -> 2,281 on 2026-09-10, +3 against a surface that grew by
    # 69. That ratio is the point: the OPS modules document their
    # parameters, so almost none of them land here.
    # 2,281 -> 2,280, -1: exactly one of the twelve withdrawn symbols
    # was an omission rather than a rendered docstring. The ratio is the
    # same point the note above makes in the other direction -- a surface
    # that shrinks by twelve moves this by one.
    # 2,280 -> 2,279 on 2026-09-13, and the arithmetic is worth reading
    # because the number went the wrong way first. Map Barcodes (c753b7de7)
    # added `spacr/barcode_search.py`, and this test went red at 2,318:
    #
    #   ADMITTED  +39. Five frozen dataclasses -- BarcodeTable,
    #             OrientationFinding, BarcodeSearchReport, ProposedMapping
    #             and BarcodeHit -- each carrying a good class docstring and
    #             no `:param:` for its fields. A dataclass field IS a
    #             required parameter of the synthesised `__init__`, which is
    #             why they land here at all.
    #   RESOLVED  -1. `spacr.qt.screens.map_barcodes.install_folds:screen`,
    #             documented by the same commit. Both halves are one lane's.
    #
    # The 39 are now documented rather than admitted, which is what takes
    # the total BELOW its old baseline: 2,280 - 1 = 2,279. Documenting them
    # was the right call and not merely the tidy one -- 434 of the 481
    # public dataclasses in the tree already document their fields, so
    # admitting these five would have made the new module the exception to
    # a convention it had no reason to break.
    # 2,279 -> 2,294 on 2026-09-20. THIS ONE IS A RATCHET THAT WENT THE
    # WRONG WAY and is not dressed up as anything else: fifteen more
    # required public parameters are undocumented than were five days
    # ago. It is pinned here so it cannot grow further unnoticed, and
    # the fifteen are a debt rather than a decision. One was found and
    # fixed while moving this number -- object_classifier's
    # `artifact_class` -- which turned out not to be counted here at
    # all, since it has a default and this metric is about REQUIRED
    # parameters. Documenting it was still right.
    # 2,294 -> 2,293 on 2026-09-21, DOWN, by set difference against
    # 6ae5e1b36. Thirteen omissions arrived with later work and were
    # documented rather than admitted: the fields of plaque_papers' Word,
    # Region and Annotation dataclasses and Region.contains's `word`,
    # SignInDialog.done's `result` and AppScreen.point_src_at's `folder`.
    # One old one left: ShareDialog's `filename`, the last omitted
    # constructor parameter, so the `constructor` bucket is gone.
    assert len(omissions) == 2_293
    # 1,635 -> 1,633: two of the six retired accessors were omissions.
    # 1,633 -> 1,636 on 2026-09-10, +2 function and +1 method against a
    # surface that grew by 69 -- the OPS modules document their
    # parameters, which is what keeps this from tracking the total.
    # 1,636 -> 1,635, -1: one of the ten withdrawn callables omitted a
    # parameter. Nine of the ten documented theirs, which is the same
    # ratio the OPS note above records in the other direction.
    # 1,635 -> 1,634 on 2026-09-13, -1. The five Map Barcodes dataclasses
    # were documented rather than admitted, so they never enter this count,
    # and what remains is `install_folds` -- documented by the same commit.
    # A new module that adds 63 callables and moves this by MINUS one is the
    # ratio the OPS note above describes: documented surface does not land
    # here however much of it there is.
    # 1,634 -> 1,644 on 2026-09-20; see the note at the omissions total.
    assert sum(omitted_callables.values()) == 1_643
    # `function` 756 -> 755, the -1 above: `install_folds`, documented by
    # the Map Barcodes commit itself. Every other bucket is unmoved, because
    # the five new dataclasses were documented rather than admitted.
    # 2026-09-20, and the bucket that appeared is the part worth reading:
    # `constructor` was absent and is now 1, so for the first time a public
    # class's own __init__ has a required parameter nobody documented.
    # function 755 -> 757 and method 835 -> 842 are the rest of the ten.
    # They sum to 1,644, the total pinned above, and they are a debt rather
    # than a decision -- pinned here so the next one cannot arrive unseen.
    assert omitted_callables == {
        "function": 757,
        "method": 842,
        "dataclass_constructor": 42,
        "namedtuple_constructor": 2,
    }
    # `function` 1,127 -> 1,126, the one parameter of `install_folds`.
    # 2026-09-20: function 1,126 -> 1,133, method 1,011 -> 1,018, and the
    # same new `constructor` bucket at 1. Both dataclass buckets unmoved.
    assert omitted_parameters == {
        "function": 1_133,
        "method": 1_018,
        "dataclass_constructor": 130,
        "namedtuple_constructor": 12,
    }
    assert _sha256_lines(omissions) == (
        # Moved 2026-09-13 and proved by subtraction: the digest recomputed
        # with `spacr.qt.screens.map_barcodes.install_folds:screen` added
        # back returns 69a9bd1b..., the previous pin, byte for byte. That
        # single re-added line is the whole difference -- the 39 Map
        # Barcodes dataclass fields never entered this set, because they
        # were documented rather than admitted.
        # REGENERATED 2026-09-20 with the omission counts above.
        "7c5788d09c657ff43a40a4e35040d70f7bccb6ddb271a858e33ebef2ad5fd64f"
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
