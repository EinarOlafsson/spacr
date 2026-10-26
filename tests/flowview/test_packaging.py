"""FlowView's feature spelling does not add a second graph stack."""

from __future__ import annotations

import ast
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_flowview_extra_is_valid_and_reuses_the_core_qt_binding():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    extras = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "setup":
            continue
        extras_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "extras_require"),
            None,
        )
        if extras_keyword is not None:
            extras = ast.literal_eval(extras_keyword.value)
            break

    assert extras is not None
    requirements = [Requirement(value) for value in extras["flowview"]]
    assert [requirement.name for requirement in requirements] == ["PySide6", "PySide6"]
    core = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "dependencies"
                    for target in node.targets))
    core_qt = [Requirement(value) for value in core if Requirement(value).name == "PySide6"]
    assert [(str(r.specifier), str(r.marker)) for r in requirements] == [
        (str(r.specifier), str(r.marker)) for r in core_qt]
    for platform, allowed, refused in (
            ("linux", ("6.6.3.1", "6.11.2", "6.12.1"), ("6.5.3", "6.12.0", "7.0")),
            ("darwin", ("6.6.3.1", "6.10.3"), ("6.5.3", "6.11.0", "6.12.0", "7.0"))):
        applying = [r for r in requirements if r.marker.evaluate({"sys_platform": platform})]
        assert len(applying) == 1, (platform, applying)
        for version in allowed:
            assert version in applying[0].specifier, (platform, version)
        for version in refused:
            assert version not in applying[0].specifier, (platform, version)
