"""Feature 411's source inventory: real lexical scopes, not guessed API names."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
helpers = importlib.import_module("nested_helper_docs")
builder = importlib.import_module("build_documentation_i18n")


def _source(tmp_path, text, relative="example.py"):
    path = tmp_path / "spacr" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_every_lexical_scope_is_walked_without_importing_source(tmp_path):
    source = '''
raise RuntimeError("Never import application code to build its inventory")
def _private_top_level():
    """This parent is not a nested helper."""
    if True:
        def _private_child(value, /, *, flag=False):
            """Return the value when enabled."""
            return value
    for _ in ():
        def below_loop():
            """Explain the loop helper."""
    try:
        with context():
            async def below_context(item: "Unknown" = factory()) -> str:
                """Explain the async helper without evaluating its defaults."""
    except Exception:
        def recovery():
            """Explain recovery."""
    match 1:
        case 1:
            def below_match():
                """Explain the branch helper."""
    class Local:
        def method(self):
            """A method inside a local class is still inside the outer function."""
            def deeper():
                """Explain the deeper helper."""
    callback = lambda value: value
class Public:
    def _method(self):
        """A top-level class method is not a nested helper."""
        def leaf():
            """Explain the method's helper."""
'''
    if not hasattr(ast, "Match"):
        source = source.replace("    match 1:\n        case 1:", "    if True:\n        if True:")
    _source(tmp_path, source)
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    by_key = {item.qualified_key: item for item in definitions}
    assert set(by_key) == {
        "spacr.example._private_top_level._private_child",
        "spacr.example._private_top_level.below_loop",
        "spacr.example._private_top_level.below_context",
        "spacr.example._private_top_level.recovery",
        "spacr.example._private_top_level.below_match",
        "spacr.example._private_top_level.Local.method",
        "spacr.example._private_top_level.Local.method.deeper",
        "spacr.example.Public._method.leaf",
    }
    child = by_key["spacr.example._private_top_level._private_child"]
    assert child.parent_id == "spacr.example._private_top_level"
    assert child.signature == "_private_child(value, /, *, flag=False)"
    asynchronous = by_key["spacr.example._private_top_level.below_context"]
    assert asynchronous.is_async
    assert "factory()" in asynchronous.signature
    assert asynchronous.signature.endswith(" -> str")
    assert all(item.lineno <= item.end_lineno for item in definitions)


@pytest.mark.parametrize("signature, originals", [
    ("outer.child(value, /, *, required)", ()),
    ("outer.child(value=None, /, *, required, flag=False)", ("None", "False")),
    ("Outer.method.child(value: str = 'a,b') -> str", ("'a,b'",)),
    ("outer.child(value=must_never_be_called())", ("must_never_be_called()",)),
    ("outer.child(value=first is not None, *, other=left if flag else right)",
     ("first is not None", "left if flag else right")),
    ("outer.child(value=lambda left, right: left + right, *, table={x: x for x in source})",
     ("lambda left, right: left + right", "{x: x for x in source}")),
])
def test_signature_rendering_protects_defaults_without_evaluating_them(signature, originals):
    safe, measured = helpers._signature_for_sphinx(signature)
    assert measured == originals
    assert safe.partition("(")[0] == signature.partition("(")[0]
    function = ast.parse("def child(" + safe.partition("(")[2] + ": pass").body[0]
    defaults = [*function.args.defaults,
                *(item for item in function.args.kw_defaults if item is not None)]
    assert [ast.literal_eval(value) for value in defaults] == [
        f"spacr-helper-default-{index}" for index in range(1, len(originals) + 1)
    ]
    if " -> str" in signature:
        assert ast.unparse(function.returns) == "str"


def test_signature_without_an_argument_list_fails_closed():
    with pytest.raises(ValueError, match="Missing helper argument list"):
        helpers._signature_for_sphinx("outer.child")


def test_ignores_are_reported_not_silently_removed(tmp_path):
    source = 'def outer():\n    def inner():\n        """Inner documentation."""\n'
    _source(tmp_path, source)
    _source(tmp_path, source, "qt/tutorial/fixture.py")
    _source(tmp_path, source, "qt/tutorials.py")
    _source(tmp_path, source, "_private_module.py")
    definitions = helpers.inventory(tmp_path, ignore_patterns=builder.AUTOAPI_IGNORE)
    assert len(definitions) == 4
    excluded = [item for item in definitions if item.ignored_by]
    assert [item.qualified_key for item in excluded] == [
        "spacr.qt.tutorial.fixture.outer.inner",
    ]
    assert excluded[0].ignored_by == ("*/qt/tutorial/*",)
    assert {item.qualified_key for item in helpers.entries(definitions)} == {
        "spacr.example.outer.inner", "spacr.qt.tutorials.outer.inner",
        "spacr._private_module.outer.inner",
    }


def test_package_init_and_deeply_nested_names_are_canonical(tmp_path):
    _source(tmp_path, '''
def factory():
    def first():
        """First."""
        def second():
            """Second."""
''', "sub/__init__.py")
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    assert [item.qualified_key for item in definitions] == [
        "spacr.sub.factory.first", "spacr.sub.factory.first.second",
    ]


def test_duplicate_definitions_keep_every_document_and_signature(tmp_path):
    _source(tmp_path, '''
def outer():
    if flag:
        def choose(value):
            """Return the first policy's value."""
    else:
        def choose(value, fallback=None):
            """Return the other policy's value or fallback."""
''')
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    entry, = helpers.entries(definitions)
    assert len(definitions) == 2
    assert entry.qualified_key == "spacr.example.outer.choose"
    assert entry.signatures == ("choose(value)", "choose(value, fallback=None)")
    assert entry.docstring == (
        "Return the first policy's value.\n\n"
        "Return the other policy's value or fallback."
    )
    assert entry.definitions[0].lineno < entry.definitions[1].lineno


def test_identical_duplicate_prose_counts_as_one_anchor_document(tmp_path):
    _source(tmp_path, '''
def outer():
    def value(first):
        """Return the selected value."""
    def value(second):
        """Return the selected value."""
''')
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    report = helpers.report(definitions, builder.translatable_blocks)
    assert report["counts"]["eligible_definitions"] == 2
    assert report["counts"]["eligible_definition_blocks"] == 2
    assert report["counts"]["canonical_entries"] == 1
    assert report["counts"]["canonical_blocks"] == 1
    assert len(report["duplicates"]["spacr.example.outer.value"]) == 2


def test_rendering_processors_leave_the_translation_source_unchanged(tmp_path):
    _source(tmp_path, 'def outer():\n    def inner():\n        """Canonical source."""\n')
    entry, = helpers.entries(helpers.inventory(tmp_path, ignore_patterns=()))
    original = entry.docstring
    calls = []

    def process(*arguments):
        calls.append(arguments[:-1])
        arguments[-1][:] = ["Processed rendering only.", ""]

    app = types.SimpleNamespace(
        events=types.SimpleNamespace(events={"autodoc-process-docstring": object()}),
        emit=process,
    )
    assert helpers.rendered_docstring(entry, app) == "Processed rendering only.\n"
    assert calls == [("autodoc-process-docstring", "function",
                      "spacr.example.outer.inner", None, None)]
    assert entry.docstring == original == "Canonical source."
    app.events.events = {}
    assert helpers.rendered_docstring(entry, app) == original + "\n"
    assert len(calls) == 1


def test_slice_switch_has_positive_and_empty_counterparts(tmp_path):
    source = 'def outer():\n    def inner():\n        """Inner documentation."""\n'
    _source(tmp_path, source)
    _source(tmp_path, source, "other.py")
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    assert len(helpers.entries(definitions)) == 2
    assert helpers.entries(definitions, modules=()) == ()
    selected = helpers.entries(definitions, modules={"spacr.example"})
    assert [item.qualified_key for item in selected] == ["spacr.example.outer.inner"]
    with pytest.raises(ValueError, match="no eligible helpers"):
        helpers.entries(definitions, modules={"spacr.typo"})


def test_undocumented_helpers_are_counted_and_cannot_be_enabled(tmp_path):
    _source(tmp_path, 'def outer():\n    def inner():\n        pass\n')
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    assert len(definitions) == 1
    assert definitions[0].docstring == ""
    assert helpers.entries(definitions) == ()
    with pytest.raises(ValueError, match="no docstring: spacr.example.outer.inner"):
        helpers.entries(definitions, modules={"spacr.example"})


def test_absent_package_and_bad_source_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="No spacr source package"):
        helpers.inventory(tmp_path, ignore_patterns=())
    source = _source(tmp_path, "def broken(:")
    with pytest.raises(SyntaxError):
        helpers.inventory(tmp_path, ignore_patterns=())
    source.write_text("def valid():\n    pass\n", encoding="utf-8")
    assert helpers.inventory(tmp_path, ignore_patterns=()) == ()


def test_inventory_does_not_follow_a_source_link_outside_package(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def valid():\n    pass\n", encoding="utf-8")
    directory = tmp_path / "spacr"
    directory.mkdir()
    try:
        (directory / "link.py").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"This platform does not permit source symlinks: {error}")
    with pytest.raises(ValueError, match="escapes the package"):
        helpers.inventory(tmp_path, ignore_patterns=())


def test_field_blocks_use_the_actual_extractor_not_paragraph_count(tmp_path):
    _source(tmp_path, '''
def outer():
    def inner(value):
        """Return a value.

        :param value: The value to return.
        :returns: The input unchanged.
        """
''')
    definitions = helpers.inventory(tmp_path, ignore_patterns=())
    report = helpers.report(definitions, builder.translatable_blocks)
    assert report["counts"]["canonical_blocks"] == 3
    assert report["counts"]["canonical_entries"] == 1
    assert report == helpers.report(definitions, builder.translatable_blocks)
    assert json.loads(json.dumps(report))["counts"] == report["counts"]


def test_real_conditional_rank_helpers_share_one_anchor_and_keep_both_policies():
    definitions = helpers.inventory(ROOT, ignore_patterns=builder.AUTOAPI_IGNORE)
    eligible = helpers.entries(definitions)
    entry = next(item for item in eligible if item.qualified_key == "spacr.hits._rank.key")
    assert len(entry.definitions) == 2
    assert "selection" in entry.docstring and "q low" in entry.docstring
    assert entry.signatures == ("key(hit: Hit)",)


def test_source_visitor_cross_checks_the_parent_walk_on_real_source():
    class CountByVisitor(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.count = 0

        def visit_FunctionDef(self, node):
            if self.depth:
                self.count += 1
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

    visitor = CountByVisitor()
    paths = sorted((ROOT / "spacr").rglob("*.py"))
    assert paths
    for path in paths:
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
    measured = helpers.inventory(ROOT, ignore_patterns=builder.AUTOAPI_IGNORE)
    assert visitor.count > 0
    assert len(measured) == visitor.count


@pytest.mark.parametrize("old,new,contract", [
    (
        "if not has_enclosing_function:", "if False:",
        test_every_lexical_scope_is_walked_without_importing_source,
    ),
    (
        "if definition.ignored_by:", "if False:",
        test_ignores_are_reported_not_silently_removed,
    ),
    (
        "if selected is not None and definition.module not in selected:",
        "if False:", test_slice_switch_has_positive_and_empty_counterparts,
    ),
])
def test_contracts_go_red_when_the_actual_source_guard_is_removed(
    tmp_path, monkeypatch, old, new, contract,
):
    """Prove exclusion/empty-selection assertions detect a deliberately broken source."""
    contract(tmp_path)
    source = Path(helpers.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1
    mutant = types.ModuleType("_feature_411_deliberate_mutant")
    mutant.__file__ = helpers.__file__
    monkeypatch.setitem(sys.modules, mutant.__name__, mutant)
    exec(compile(source.replace(old, new), helpers.__file__, "exec"), vars(mutant))
    monkeypatch.setattr(sys.modules[__name__], "helpers", mutant)
    with pytest.raises(AssertionError):
        contract(tmp_path)
