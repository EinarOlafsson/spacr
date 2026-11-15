"""Power-model backends and sweep bookkeeping: what is installed, what is a value, how a point is keyed.

The refusals of the design, fit, estimate, scoring and sweep paths live in
``tests/test_power_model_refusals.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import power_model as pm


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def test_a_shadowed_or_half_removed_package_counts_as_not_installed(
        monkeypatch):
    """``find_spec`` itself can raise on a namespace-package shadow; that is
    'not usable', not a crash on start-up."""
    import importlib.util

    def _confused(name):
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(importlib.util, "find_spec", _confused)
    assert pm._module_installed("numpyro") is False


def test_the_backends_report_says_what_this_interpreter_can_run():
    available = pm.available_backends()
    assert available["torch"] is True
    assert set(available) >= {"torch", "numpyro", "pymc"}


# --------------------------------------------------------------------------
# sweep bookkeeping
# --------------------------------------------------------------------------

def test_a_string_is_one_value_and_not_a_sweep_over_its_characters():
    """The classic version of this bug: 'cellpose' as a four-way sweep."""
    assert pm._is_scalar("cellpose") is True
    assert pm._is_scalar(b"raw") is True
    assert pm._is_scalar(np.float64(3.0)) is True
    assert pm._is_scalar(np.array(3.0)) is True
    assert pm._is_scalar(np.array([1, 2])) is False
    assert pm._is_scalar([1, 2]) is False
    assert pm._is_scalar(range(3)) is False


def test_a_numpy_scalar_hashes_the_same_as_the_python_number_it_equals():
    """A grid built with ``np.linspace`` must resume a grid written by hand."""
    assert pm._jsonable(np.float64(2.5)) == 2.5
    assert pm._jsonable(np.int64(3)) == 3
    assert pm._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert pm._jsonable([np.int64(1), 2]) == [1, 2]
    assert pm._jsonable(None) is None
    assert pm._jsonable({"a": 1}) == repr({"a": 1})


def test_the_same_point_keys_the_same_however_the_numbers_were_written():
    left = pm._run_key({"n_genes": np.int64(4)}, 0, 7, "torch")
    right = pm._run_key({"n_genes": 4}, 0, 7, "torch")
    assert left == right
    assert left != pm._run_key({"n_genes": 5}, 0, 7, "torch")


def test_a_missing_simulator_says_what_to_pass_instead(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and "simulate_screen" in fromlist:
            raise ImportError("no simulator installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(pm.PowerFitError, match="Pass simulate_fn"):
        pm._default_simulator()


def test_the_seed_reaches_a_simulator_by_whichever_name_it_accepts():
    def by_seed(seed=0, **kw):
        return pd.DataFrame({"seed": [seed]})

    def by_rng(n_genes=4, rng=None):
        return pd.DataFrame({"kind": [type(rng).__name__]})

    def by_random_state(n_genes=4, random_state=None):
        return pd.DataFrame({"random_state": [random_state]})

    def by_nothing():
        return pd.DataFrame({"x": [1]})

    frame, channel = pm._call_simulator(by_seed, {}, 11)
    assert channel == "seed" and frame.iloc[0]["seed"] == 11

    frame, channel = pm._call_simulator(by_rng, {}, 11)
    assert channel == "rng" and frame.iloc[0]["kind"] == "Generator"

    frame, channel = pm._call_simulator(by_random_state, {}, 11)
    assert channel == "random_state" and frame.iloc[0]["random_state"] == 11

    _frame, channel = pm._call_simulator(by_nothing, {}, 11)
    assert channel == "numpy-global"


def test_a_simulator_with_no_readable_signature_is_given_the_seed_anyway():
    """Builtins and C callables have no signature; assuming they take
    nothing would silently unseed the sweep."""
    calls = {}

    class Opaque:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return pd.DataFrame({"x": [1]})

        @property
        def __signature__(self):
            raise ValueError("no signature for this object")

    _frame, channel = pm._call_simulator(Opaque(), {"n_genes": 3}, 5)
    assert channel == "seed"
    assert calls == {"n_genes": 3, "seed": 5}


def test_the_default_simulator_is_the_one_the_port_ships_with():
    from spacr.power_simulate import simulate_screen

    assert pm._default_simulator() is simulate_screen
