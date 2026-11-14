"""How a run recommendation renders, and that every setting it names exists.

The rules themselves are walked in ``tests/test_run_recommendations_rules.py``.
"""
from __future__ import annotations

from spacr.run_recommendations import Recommendation


def test_recommendation_lines_distinguish_blocking_from_advisory_actions():
    """The display marker exposes severity without changing its explanation."""
    advisory = Recommendation("alpha", "raise it", "the fit was noisy")
    blocking = Recommendation(
        "alpha", "raise it", "the fit was noisy", severity="blocking")

    assert advisory.line() == (
        "  - alpha: raise it\n      because the fit was noisy")
    assert blocking.line() == (
        "  ! alpha: raise it\n      because the fit was noisy")


def test_every_recommended_setting_is_a_setting_that_exists():
    """Spelled the way the run reads them -- checked, not eyeballed.

    A recommendation naming a key no panel offers and no run reads is worse
    than none: the reader follows it, nothing changes, and the section stops
    being believed. The names are read off the rules themselves rather than
    listed here, so a rule added later is checked the day it lands.
    """
    import ast
    import inspect

    from spacr.settings import expected_types
    import spacr.run_recommendations as module

    tree = ast.parse(inspect.getsource(module))
    named = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Recommendation"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    # Guards the guard: an empty set would assert nothing at all.
    assert len(named) >= 5, named
    missing = sorted(k for k in named if k not in expected_types)
    assert not missing, f"recommended settings that do not exist: {missing}"


