"""The v2 mask pipeline answered from its own inline fallbacks.

`preprocess_generate_masks` dispatches to `run_v2` and RETURNS, while
`set_default_settings_preprocess_generate_masks` is only called further down
inside the per-source loop that the v2 branch never reaches. So every
``settings.get(key, fallback)`` in that branch answered with the fallback
written beside it rather than with the module's declared default.

One of them changes segmentation. ``cell_FT`` is declared 1.0 and the inline
fallback was 0.4, and it is forwarded verbatim to
``model.eval(flow_threshold=...)``. Cellpose's ``remove_bad_flow_masks``
discards a mask whose flow error exceeds the threshold, so on a field with
per-object flow errors {0.00, 0.12, 0.30, 0.75} the v1 pipeline keeps four
cells and the v2 branch kept three -- same plate, same settings dict, same
weights, different answer depending only on ``pipeline_style``.
"""

import ast
import inspect
import pathlib

import pytest

import spacr.core as core
import spacr.settings as S


def _v2_branch_source():
    """The body of the `pipeline_style == 'v2'` branch, as text."""
    source = inspect.getsource(core.preprocess_generate_masks)
    tree = ast.parse(source.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "pipeline_style" in ast.unparse(node.test):
            return "\n".join(ast.unparse(stmt) for stmt in node.body)
    raise AssertionError("the v2 branch is gone; update this test")


def test_the_branch_applies_the_defaults_before_reading_the_dict():
    body = _v2_branch_source()
    applied = body.index("set_default_settings_preprocess_generate_masks")
    first_get = body.index("settings.get")
    assert applied < first_get, (
        "the v2 branch reads settings before applying the declared defaults, "
        "so every inline fallback is live again")


def test_the_organelle_defaults_are_applied_too():
    assert "_set_organelle_defaults" in _v2_branch_source()


def test_cell_ft_is_the_declared_value_not_the_inline_fallback():
    """The one fallback that changes segmentation output."""
    declared = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/does-not-matter"})["cell_FT"]
    assert declared == 1.0, (
        "the declared cell_FT moved; this test pins the value the v2 branch "
        "must now agree with")

    body = _v2_branch_source()
    assert "settings.get('cell_FT', 0.4)" in body or \
           'settings.get("cell_FT", 0.4)' in body, (
        "the inline fallback is gone entirely -- fine, but then this test "
        "should assert the new spelling instead")


def test_the_defaults_helper_is_idempotent():
    """It is now called twice on the v1 path, so it must be setdefault-only."""
    once = S.set_default_settings_preprocess_generate_masks({"src": "/tmp/x"})
    twice = S.set_default_settings_preprocess_generate_masks(dict(once))
    assert once == twice


def test_a_value_the_caller_set_still_wins():
    """setdefault, not overwrite: an explicit cell_FT must survive."""
    given = S.set_default_settings_preprocess_generate_masks(
        {"src": "/tmp/x", "cell_FT": 0.25})
    assert given["cell_FT"] == 0.25
