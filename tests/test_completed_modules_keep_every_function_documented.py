"""Keep completed Instruction 368 modules fully documented.

The broader documentation checks cover public callables and nested helpers,
but not private module-level functions.  This zero-debt registry closes that
gap one finished module at a time without pretending the whole package is
already complete.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FULLY_DOCUMENTED = (
    "spacr/attribution_columns.py",
    "spacr/cli_plugins.py",
    "spacr/cli_repro.py",
    "spacr/column_groups.py",
    "spacr/figures/summary.py",
    "spacr/foreign.py",
    "spacr/flowview/_classify_stages.py",
    "spacr/flowview/classify_blueprint.py",
    "spacr/intensity_rescale.py",
    "spacr/localisation.py",
    "spacr/mask_io.py",
    "spacr/normalization.py",
    "spacr/notebook_export.py",
    "spacr/regression_failure.py",
    "spacr/regression_layout.py",
    "spacr/well_scope.py",
    "spacr/well_spec.py",
)


def _undocumented_functions(path: Path) -> list[str]:
    """Return every function or method in ``path`` without a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{node.name} (line {node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not ast.get_docstring(node)
    ]


@pytest.mark.parametrize("relative", FULLY_DOCUMENTED)
def test_every_function_in_a_finished_module_stays_documented(relative):
    """A finished phase-4 module may not regain an undocumented helper."""
    missing = _undocumented_functions(ROOT / relative)
    assert not missing, (
        f"{relative} was fully documented and gained undocumented functions: "
        f"{missing}"
    )


def test_the_scanner_detects_a_removed_docstring(tmp_path):
    """The guard observes both the documented and regressed source shapes."""
    source = tmp_path / "documented.py"
    source.write_text(
        'def helper(value):\n    """Return the supplied value."""\n'
        "    return value\n",
        encoding="utf-8",
    )
    assert _undocumented_functions(source) == []

    source.write_text(
        "def helper(value):\n    return value\n",
        encoding="utf-8",
    )
    assert _undocumented_functions(source) == ["helper (line 1)"]
