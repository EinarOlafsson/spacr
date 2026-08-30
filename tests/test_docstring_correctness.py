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
import hashlib
import pathlib
import re
from collections import Counter
from dataclasses import dataclass

#: Sphinx ``:param name:`` up to the next field or the end.
PARAM_FIELD = re.compile(r":param\s+([*\w]+)\s*:(.*?)(?=\n\s*:|\Z)", re.S)

#: "Defaults to X" / "defaults to ``X``".
CLAIMED_DEFAULT = re.compile(r"[Dd]efaults?\s+to\s+``?([^`.,;)\s]+)``?")

#: The retired Tk front end. Not maintained; see instruction 60.
LEGACY_MODULES = {"gui.py", "gui_core.py", "gui_elements.py", "gui_utils.py"}

# These source trees are not Python API inputs.  This mirrors
# ``docs/source/conf.py:autoapi_ignore`` rather than allowing generated
# translation payloads and documentation asset generators to inflate the
# callable inventory.
IGNORED_SOURCE_PARTS = {
    ("qt", "tutorial"),
    ("qt", "i18n_catalogs"),
}

# ``__main__`` is the public ``python -m`` entry point.  ``_v1_v2_bridge`` is
# the one underscore module deliberately named in spacr's lazy public-module
# registry as a compatibility surface.  No other private module gets widened
# merely because an implementation helper inside it has a public-looking name.
PUBLIC_SPECIAL_MODULES = {
    "spacr.__main__",
    "spacr.qt.__main__",
    "spacr._v1_v2_bridge",
}


@dataclass(frozen=True)
class _PublicCallable:
    """One source-owned callable contract admitted by the API boundary."""

    symbol: str
    category: str
    parameters: frozenset[str]
    required_parameters: frozenset[str]
    docstring: str


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


def _source_is_in_public_boundary(
    root: pathlib.Path, path: pathlib.Path, module: str,
) -> bool:
    """Whether *path* can own canonical public AutoAPI callables."""
    relative = path.relative_to(root)
    parts = relative.parts
    if path.name in LEGACY_MODULES or "tests" in parts:
        return False
    if any(all(part in parts for part in ignored)
           for ignored in IGNORED_SOURCE_PARTS):
        return False
    if "resources" in parts and "_generators" in parts:
        return False

    module_parts = module.split(".")[1:]
    if any(part.startswith("_") for part in module_parts):
        return module in PUBLIC_SPECIAL_MODULES
    return True


def _static_exports(tree: ast.Module, path: pathlib.Path):
    """Return a literal ``__all__``, or ``None`` when none is declared.

    AutoAPI honours module ``__all__``.  A dynamic declaration would make a
    source-only CI inventory ambiguous, so it fails with the owning path
    rather than silently falling back to name-based visibility.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise AssertionError(
                f"{path}: __all__ must be a literal source inventory"
            ) from exc
        assert isinstance(value, (list, tuple, set, frozenset)), (
            f"{path}: __all__ is not a sequence of public names")
        assert all(isinstance(name, str) for name in value), (
            f"{path}: __all__ contains a non-string public name")
        return frozenset(value)
    return None


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


def _dataclass_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Local spellings of ``dataclass`` and ``field`` in one module."""
    decorators = {"dataclass"}
    fields = {"field"}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "dataclasses":
            continue
        for imported in node.names:
            local = imported.asname or imported.name
            if imported.name == "dataclass":
                decorators.add(local)
            elif imported.name == "field":
                fields.add(local)
    return decorators, fields


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


def _is_named_tuple(node: ast.ClassDef) -> bool:
    return any(_decorator_name(base) == "NamedTuple" for base in node.bases)


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


def _generated_constructor_parameters(
    node: ast.ClassDef, field_names: set[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Locally declared dataclass/NamedTuple constructor fields.

    Inherited fields remain owned and checked on the class that declares
    them.  This avoids repeating one missing field on every subclass while
    still admitting every new locally generated constructor parameter.
    """
    names: set[str] = set()
    required: set[str] = set()
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

        names.add(name)
        if child.value is None:
            required.add(name)
        elif field_call:
            keywords = {keyword.arg for keyword in child.value.keywords}
            if "default" not in keywords and "default_factory" not in keywords:
                required.add(name)
    return frozenset(names), frozenset(required)


def _public_callables():
    """Yield the complete source-owned public callable boundary.

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
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        module = _module_name(root, path)
        if not _source_is_in_public_boundary(root, path, module):
            continue
        exports = _static_exports(tree, path)
        dataclass_names, field_names = _dataclass_names(tree)

        def visible(name: str, exports=exports) -> bool:
            if exports is not None:
                return name in exports
            return not name.startswith("_")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not visible(node.name):
                    continue
                parameters, required = _node_parameters(node)
                yield _PublicCallable(
                    f"{module}.{node.name}", "function", parameters, required,
                    ast.get_docstring(node) or "",
                )
                continue
            if not isinstance(node, ast.ClassDef) or not visible(node.name):
                continue

            class_symbol = f"{module}.{node.name}"
            class_doc = ast.get_docstring(node) or ""
            constructor = next((
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ), None)
            if constructor is None:
                constructor = next((
                    child for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__new__"
                ), None)

            if constructor is not None:
                parameters, required = _node_parameters(constructor)
                category = "constructor"
                constructor_doc = ast.get_docstring(constructor) or ""
                doc = "\n".join(
                    text for text in (class_doc, constructor_doc) if text)
            elif _dataclass_generates_init(node, dataclass_names):
                parameters, required = _generated_constructor_parameters(
                    node, field_names)
                category = "dataclass_constructor"
                doc = class_doc
            elif _is_named_tuple(node):
                parameters, required = _generated_constructor_parameters(
                    node, field_names)
                category = "namedtuple_constructor"
                doc = class_doc
            else:
                parameters = required = frozenset()
                category = "inherited_or_default_constructor"
                doc = class_doc
            yield _PublicCallable(
                class_symbol, category, parameters, required, doc)

            for child in node.body:
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if child.name.startswith("_") or _is_property_definition(child):
                    continue
                parameters, required = _node_parameters(child)
                yield _PublicCallable(
                    f"{class_symbol}.{child.name}", "method", parameters,
                    required, ast.get_docstring(child) or "",
                )


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
        documented = {name.lstrip("*") for name, _ in PARAM_FIELD.findall(doc)}
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


def test_public_callable_inventory_is_source_derived_not_docstring_derived():
    """Freeze the whole boundary and every source-owned parameter name.

    The inventory includes examples with no docstring and no ``:param``
    field.  Removing documentation therefore cannot remove the callable from
    this test's denominator.  The signature digest also catches a required
    parameter becoming optional (or the reverse) without changing counts.
    """
    callables = list(_public_callables())
    by_symbol = {item.symbol: item for item in callables}

    assert len(callables) == len(by_symbol) == 7_951
    assert Counter(item.category for item in callables) == {
        "function": 3_448,
        "method": 3_521,
        "constructor": 367,
        "dataclass_constructor": 420,
        "namedtuple_constructor": 6,
        "inherited_or_default_constructor": 189,
    }
    assert sum(len(item.parameters) for item in callables) == 15_595
    assert sum(len(item.required_parameters) for item in callables) == 7_920
    assert _sha256_lines(
        f"{item.symbol}\0{item.category}\0"
        f"{','.join(sorted(item.parameters))}\0"
        f"{','.join(sorted(item.required_parameters))}"
        for item in callables
    ) == "a12e73876a726023bad22b9ca172872621c616d879a74e8704af4f1128ecdf25"

    # Fieldless, docless and generated-constructor contracts all remain in
    # scope.  These are named assertions so a future refactor cannot preserve
    # only the headline count while losing the defect classes that motivated
    # the boundary.
    assert by_symbol["spacr.api.run_mask"].required_parameters == {"config"}
    assert not PARAM_FIELD.findall(by_symbol["spacr.api.run_mask"].docstring)
    assert not by_symbol["spacr.layers.LayerStack.add_image"].docstring
    assert by_symbol[
        "spacr.layers.LayerStack.add_image"
    ].required_parameters == {"data"}
    assert by_symbol["spacr.api.MaskConfig"].category == "dataclass_constructor"
    assert by_symbol["spacr.api.MaskConfig"].required_parameters == {"src"}

    # A literal __all__ closes a module; nested functions, properties,
    # private Qt callbacks and non-constructor special methods are not
    # separately callable entries in this project's AutoAPI contract.
    assert "spacr.annotation.annotate_with" not in by_symbol
    assert "spacr.crashreport.collect.environment" not in by_symbol
    assert "spacr.layers.Layer.name" not in by_symbol
    assert "spacr.layers.Colormap.__eq__" not in by_symbol
    assert "spacr.qt.plate_queue.PlateQueue._serialise" not in by_symbol

    # These two otherwise-private module spellings are deliberate public
    # entry/compatibility contracts, not an accidental widening.
    assert "spacr.__main__.main" in by_symbol
    assert "spacr._v1_v2_bridge.v2_channels_from_settings" in by_symbol


def test_no_new_public_callable_lacks_a_docstring():
    """Ratchet the exact docless debt independently of parameter fields."""
    docless = sorted(
        item.symbol for item in _public_callables() if not item.docstring)
    assert len(docless) == 666
    assert _sha256_lines(docless) == (
        "940a9c6ca8ebefd1861cced45aa442cfca1aa78b09461ec68960240e482e01d4"
    )


def test_no_new_undocumented_required_public_parameters():
    """Ratchet the full reverse-direction debt while it is repaired.

    The old denominator selected only callables whose prose already contained
    ``:param:`` and reached a misleading zero when those selected fields were
    completed.  The source-derived denominator above exposes the real current
    baseline: 4,398 omissions across 2,855 public callables.  Count, category
    counts and digest are all exact so deleting a field/docstring, or swapping
    one omission for another, cannot turn this test green.
    """
    omissions: list[str] = []
    omitted_callables: Counter[str] = Counter()
    omitted_parameters: Counter[str] = Counter()
    for item in _public_callables():
        documented = {
            name.lstrip("*")
            for name, _body in PARAM_FIELD.findall(item.docstring)
        }
        missing = item.required_parameters - documented
        if missing:
            omitted_callables[item.category] += 1
            omitted_parameters[item.category] += len(missing)
        omissions.extend(
            f"{item.symbol}:{name}" for name in missing)

    assert len(omissions) == 4_398
    assert sum(omitted_callables.values()) == 2_855
    assert omitted_callables == {
        "function": 1_221,
        "method": 1_378,
        "constructor": 55,
        "dataclass_constructor": 196,
        "namedtuple_constructor": 5,
    }
    assert omitted_parameters == {
        "function": 1_780,
        "method": 1_700,
        "constructor": 86,
        "dataclass_constructor": 804,
        "namedtuple_constructor": 28,
    }
    assert _sha256_lines(omissions) == (
        "9545872216e09fc725db2c35b128be82ff6a6eb1eb71860c7f648671f6675047"
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
