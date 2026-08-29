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
    assert [requirement.name for requirement in requirements] == ["PySide6"]
    assert str(requirements[0].specifier) == "<7,>=6.6"
