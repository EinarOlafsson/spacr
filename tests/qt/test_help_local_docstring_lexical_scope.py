"""Offline Help reads control-flow declarations without crossing scopes."""

import ast
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def source_reader(monkeypatch, tmp_path):
    from spacr.qt import help_search

    path = tmp_path / "not_an_importable_module.py"
    monkeypatch.setattr(
        help_search, "_module_file",
        lambda name: path if name == "spacr.offline_fixture" else None,
    )

    def read(source, qualname):
        path.write_text(
            'raise RuntimeError("Help must never execute this source")\n'
            + textwrap.dedent(source),
            encoding="utf-8",
        )
        return help_search.local_docstring("spacr.offline_fixture." + qualname)

    return read


def test_real_installer_onboarding_helper_has_offline_prose():
    from spacr.qt.help_search import local_docstring

    source = Path(__file__).resolve().parents[2] / "spacr/qt/app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_installer_onboarding"
    ]
    assert len(matches) == 1
    expected = ast.get_docstring(matches[0])
    assert expected and "tour" in expected
    imported_before = "spacr.qt.app" in sys.modules
    actual = local_docstring(
        "spacr.qt.app.MainWindow.__init__._finish_installer_onboarding"
    )
    assert actual == expected
    assert ("spacr.qt.app" in sys.modules) == imported_before


@pytest.mark.parametrize("block", [
    "if enabled:\n{helper}",
    "if enabled:\n    pass\nelse:\n{helper}",
    "try:\n{helper}\nexcept Exception:\n    pass",
    "try:\n    pass\nexcept Exception:\n{helper}",
    "try:\n    pass\nfinally:\n{helper}",
    "with context():\n{helper}",
    "for value in values:\n{helper}",
    "while enabled:\n{helper}",
    "match value:\n    case 1:\n{match_helper}",
])
def test_control_flow_helpers_are_available_without_execution(source_reader, block):
    helper = 'def helper():\n    """Useful offline help."""\n'
    body = block.format(
        helper=textwrap.indent(helper, "    "),
        match_helper=textwrap.indent(helper, "        "),
    )
    source = "def outer():\n" + textwrap.indent(body, "    ")
    assert source_reader(source, "outer.helper") == "Useful offline help."


def test_qualified_async_and_class_scopes_survive_control_flow(source_reader):
    source = '''
        if enabled:
            class Panel:
                try:
                    async def prepare(self):
                        with context():
                            def finish():
                                """Finish this panel's own preparation."""
                except Exception:
                    pass
    '''
    assert source_reader(source, "Panel.prepare.finish") == (
        "Finish this panel's own preparation."
    )


@pytest.mark.parametrize("qualname", ["target", "outer.target"])
def test_unmatched_function_and_class_scopes_are_not_searched(source_reader, qualname):
    source = '''
        def unrelated():
            if enabled:
                def target():
                    """Wrong function scope."""
        class Unrelated:
            def target(self):
                """Wrong class scope."""
        def outer():
            if enabled:
                def decoy():
                    try:
                        def target():
                            """Wrong nested function scope."""
                    finally:
                        pass
                class Decoy:
                    def target(self):
                        """Wrong nested class scope."""
    '''
    assert source_reader(source, qualname) == ""


def test_an_explicit_nested_scope_still_resolves_its_own_helper(source_reader):
    source = '''
        def outer():
            if enabled:
                def decoy():
                    try:
                        def target():
                            """The explicitly requested helper."""
                    finally:
                        pass
    '''
    assert source_reader(source, "outer.decoy.target") == (
        "The explicitly requested helper."
    )
