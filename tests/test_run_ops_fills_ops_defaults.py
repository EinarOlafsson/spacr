"""A direct ``run_ops`` call resolves the same defaults as the command line.

``run_ops`` used to leave unset keys to inline fallbacks in each module, 22 of
which disagreed with ``OPS_DEFAULTS`` (the decode threshold mode fell back to
"manual", the default is "auto"). There is now one table: ``run_ops`` fills it
once, and the readers take a missing value from it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import spacr
from spacr import ops_engine, ops_settings


class _Stop(Exception):
    """Raised by the capture hook once the settings are seen."""


def _seen_by_run_ops(monkeypatch, settings):
    """The settings dict ``run_ops`` hands to phase planning."""
    seen = {}

    def capture(got, phases):
        seen.update(got)
        raise _Stop

    monkeypatch.setattr(ops_engine, "_run_phases", capture)
    with pytest.raises(_Stop):
        ops_engine.run_ops(settings)
    return seen


def test_a_direct_run_ops_call_resolves_exactly_ops_defaults(monkeypatch):
    seen = _seen_by_run_ops(monkeypatch, {})
    assert seen == ops_settings.ops_defaults({})
    assert seen["ops_decode_threshold_mode"] == ops_settings.OPS_DEFAULTS["ops_decode_threshold_mode"]
    assert seen["randomize"] is ops_settings.OPS_DEFAULTS["randomize"]


def test_a_set_value_wins_over_the_default(monkeypatch):
    seen = _seen_by_run_ops(monkeypatch, {"ops_decode_threshold_mode": "manual",
                                          "randomize": False})
    assert seen["ops_decode_threshold_mode"] == "manual"
    assert seen["randomize"] is False
    assert seen["n_workers"] == ops_settings.OPS_DEFAULTS["n_workers"]


def test_the_cli_fills_through_the_same_function():
    from spacr.cli import MODULES

    module = MODULES["ops"]
    assert module.defaults_entry == "spacr.ops_settings:ops_defaults"


def test_ops_setting_takes_the_declared_default_for_unset_and_empty():
    declared = ops_settings.OPS_DEFAULTS["n_workers"]
    for unset in ({}, {"n_workers": None}, {"n_workers": ""}):
        assert ops_settings.ops_setting(unset, "n_workers") == declared
    assert ops_settings.ops_setting({"n_workers": 3}, "n_workers") == 3
    assert ops_settings.ops_setting({"n_workers": 0}, "n_workers") == 0


def test_no_module_carries_a_second_default_for_a_declared_key():
    """``settings.get(key, literal)`` in ops_*.py may only repeat OPS_DEFAULTS."""
    defaults = ops_settings.OPS_DEFAULTS
    wrong = []
    for path in sorted(pathlib.Path(spacr.__file__).parent.glob("ops_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in defaults
                    and isinstance(node.args[1], ast.Constant)):
                continue
            if node.args[1].value != defaults[node.args[0].value]:
                wrong.append(f"{path.name}:{node.lineno} {node.args[0].value}")
    assert not wrong, wrong
