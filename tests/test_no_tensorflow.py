"""
Guard tests: spacr must not (re-)introduce a TensorFlow dependency.

Stardist was the only remaining TF-backed component; if any future commit
re-adds a `stardist` code path or an `import tensorflow` / `import keras`,
one of these tests should fail loudly.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "spacr"


def _all_py_files():
    return sorted(PKG_ROOT.glob("*.py"))


@pytest.mark.parametrize("path", _all_py_files(), ids=lambda p: p.name)
def test_no_stardist_references(path):
    """No source file may reference stardist — imports, calls, strings, or comments."""
    src = path.read_text()
    if "stardist" in src.lower():
        # Show the first offending line for a clear failure message.
        for i, line in enumerate(src.splitlines(), 1):
            if "stardist" in line.lower():
                pytest.fail(f"{path.name}:{i}: {line.strip()!r} still references stardist")
        pytest.fail(f"{path.name}: contains 'stardist'")


@pytest.mark.parametrize("path", _all_py_files(), ids=lambda p: p.name)
def test_no_tensorflow_or_keras_imports(path):
    """Fail if any module imports tensorflow or standalone keras."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.skip(f"{path.name} did not parse")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                assert mod not in ("tensorflow", "keras"), (
                    f"{path.name}:{node.lineno} imports {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in ("tensorflow", "keras"), (
                f"{path.name}:{node.lineno} imports from {node.module}"
            )


def test_object_module_has_no_segment_stardist():
    import spacr.object as m
    assert not hasattr(m, "_segment_stardist")
    assert not hasattr(m, "_load_stardist_model")


def test_settings_module_has_no_stardist_keys():
    import spacr.settings as s
    # expected_types dict should not carry stardist knobs anymore.
    et = getattr(s, "expected_types", {})
    stardist_keys = [k for k in et if "stardist" in k.lower()]
    assert not stardist_keys, f"stardist keys still in expected_types: {stardist_keys}"
    # Same check on descriptions dict.
    d = getattr(s, "descriptions", {})
    star_desc = [k for k in d if "stardist" in k.lower()]
    assert not star_desc, f"stardist keys still in descriptions: {star_desc}"
