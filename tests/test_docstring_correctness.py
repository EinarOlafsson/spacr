"""Mechanical correctness checks on docstrings.

Instruction 112. A docstring can be present, counted, extracted and
translated into eight languages and still be wrong about the code beside it,
and none of the existing guards would notice -- they check that prose EXISTS.
These check that it AGREES.

Both pass today over 1,934 documented functions. They are ratchets: what they
buy is that a signature change cannot silently leave its docstring behind.

The legacy Tk modules are excluded, the same four excluded from instruction
60's coverage scope, because they are not maintained.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import re

#: Sphinx ``:param name:`` up to the next field or the end.
PARAM_FIELD = re.compile(r":param\s+([*\w]+)\s*:(.*?)(?=\n\s*:|\Z)", re.S)

#: "Defaults to X" / "defaults to ``X``".
CLAIMED_DEFAULT = re.compile(r"[Dd]efaults?\s+to\s+``?([^`.,;)\s]+)``?")

#: The retired Tk front end. Not maintained; see instruction 60.
LEGACY_MODULES = {"gui.py", "gui_core.py", "gui_elements.py", "gui_utils.py"}


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


def _public_field_documented_functions():
    """AutoAPI-visible functions/methods that use ``:param:`` fields."""
    root = pathlib.Path(__file__).resolve().parent.parent / "spacr"

    def visible(name):
        return (
            not name.startswith("_")
            or (
                name != "__init__"
                and name.startswith("__")
                and name.endswith("__")
            )
        )

    for path in sorted(root.rglob("*.py")):
        if path.name in LEGACY_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(("spacr", *parts))
        for node in tree.body:
            candidates = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if visible(node.name):
                    candidates.append((node.name, node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                candidates.extend(
                    (f"{node.name}.{child.name}", child)
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and visible(child.name)
                )
            for qualified_name, candidate in candidates:
                doc = ast.get_docstring(candidate)
                if doc and ":param" in doc:
                    yield f"{module}.{qualified_name}", candidate, doc


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


def test_no_new_undocumented_required_public_parameters():
    """Ratchet the reverse direction while the recorded legacy debt is fixed.

    The digest prevents one omission from replacing another without changing
    the total.  This is deliberately not presented as zero debt: the exact
    baseline remains 197 required parameters across 119 public callables.
    """
    omissions = []
    checked = 0
    for symbol, node, doc in _public_field_documented_functions():
        checked += 1
        documented = {
            name.lstrip("*") for name, _body in PARAM_FIELD.findall(doc)
        }
        omissions.extend(
            f"{symbol}:{name}"
            for name in _required_parameters(node) - documented
        )
    omissions.sort()
    digest = hashlib.sha256("\n".join(omissions).encode()).hexdigest()

    assert checked == 2_067
    assert len(omissions) == 197
    assert digest == (
        "30076e69fd8b032a7e01eeca78c2433b01c7f42a822fd2692cb2026023436bc0"
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
