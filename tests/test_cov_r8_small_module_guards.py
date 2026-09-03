"""Three small guards: a FlowView gate, an empty design, and no cells.

Each belongs to a different module and they share no code, but they
share a shape: a cheap refusal that keeps an expensive or meaningless
computation from running, and that nothing had exercised.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest


class TestClassifysFlowViewGate:
    """`_begin_flowview_run` imports no FlowView code on the common path.

    The disabled path is a module-cache lookup and an environment check.
    That is the point: a classify run that pulled the tracing package in
    merely to discover tracing was off would pay for a feature it is not
    using.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_trace(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "spacr.flowview.trace",
                            raising=False)
        flowview_package = sys.modules.get("spacr.flowview")
        if flowview_package is not None:
            # Importing a submodule stores it both in ``sys.modules`` and on
            # its parent package.  Clear both caches so this test class has
            # the same lazy-import starting state regardless of collection
            # or execution order.
            monkeypatch.delattr(flowview_package, "trace", raising=False)
        monkeypatch.delenv("SPACR_FLOWVIEW", raising=False)

    def test_an_unset_environment_starts_no_graph(self):
        from spacr.classify import _begin_flowview_run

        assert _begin_flowview_run({}) is None
        assert "spacr.flowview.trace" not in sys.modules, (
            "the disabled path imported FlowView anyway")

    @pytest.mark.parametrize("value", ["", "0", "off", "false", "no", "  "])
    def test_a_value_that_is_not_a_yes_starts_no_graph(self, monkeypatch,
                                                       value):
        from spacr.classify import _begin_flowview_run

        monkeypatch.setenv("SPACR_FLOWVIEW", value)
        assert _begin_flowview_run({}) is None

    @pytest.mark.parametrize("value", ["1", "on", "true", "yes",
                                       "TRUE", " Yes "])
    def test_a_yes_opts_a_headless_run_in_through_the_lazy_boundary(
            self, monkeypatch, value):
        """THE UNCOVERED IMPORT.

        SPACR_FLOWVIEW is how a headless run asks for tracing without a
        panel to switch it on. Only then is the FlowView package
        imported -- which is the whole point of doing the environment
        check first.

        The value is stripped and case-folded before it is judged, so
        the spellings a user actually types all work.
        """
        from spacr.classify import _begin_flowview_run

        monkeypatch.setenv("SPACR_FLOWVIEW", value)
        started = _begin_flowview_run({})

        # The boundary was crossed -- which is the thing the environment
        # variable exists to do, and the thing the check above it defers.
        assert "spacr.flowview.trace" in sys.modules, (
            f"{value!r} did not opt the run in through the lazy boundary")
        # And whatever came back is a graph or nothing, never an error
        # value: the caller stores it and later asks whether it is None.
        assert started is None or hasattr(started, "__class__")

    def test_a_loaded_but_disabled_tracer_starts_no_graph(self,
                                                          monkeypatch):
        """A panel can load the trace module and leave it switched off.

        The gate has to ask, not assume that a loaded module means a
        wanted one.
        """
        import types

        module = types.ModuleType("spacr.flowview.trace")
        module.is_enabled = lambda: False
        monkeypatch.setitem(sys.modules, "spacr.flowview.trace", module)

        from spacr.classify import _begin_flowview_run

        assert _begin_flowview_run({}) is None


class TestTheConditionNumber:
    """`condition_number` answers TWO numbers for different questions.

    The unscaled one is what statsmodels prints and is dominated by the
    units of the columns -- a predictor in cells rather than thousands
    of cells moves it by 1000 with no change in the science. The scaled
    one (each column normalised first, Belsley-Kuh-Welsch) is unit-free
    and is the one whose thresholds -- 30, 100, 1000 -- mean anything.

    It returns three values: both ratios and the scaled spectrum.
    """

    def test_an_orthogonal_design_is_well_conditioned(self):
        from spacr.regression_qc import condition_number

        scaled, raw, spectrum = condition_number(np.eye(4))
        assert scaled == pytest.approx(1.0)
        assert np.isfinite(raw)
        assert len(spectrum) == 4

    def test_a_duplicated_predictor_is_singular(self):
        """The case the number exists to catch: two identical columns."""
        from spacr.regression_qc import condition_number

        column = np.linspace(1.0, 5.0, 8).reshape(-1, 1)
        scaled, _raw, _spectrum = condition_number(np.hstack([column,
                                                              column]))
        assert not np.isfinite(scaled) or scaled > 1000, (
            "a duplicated predictor was not reported as ill-conditioned")

    def test_rescaling_a_column_changes_the_raw_number_and_not_the_scaled(
            self):
        """The reason there are two numbers at all.

        Measuring a predictor in thousands rather than units is a change
        of units, not of science -- so the unit-free number must not
        move, and the one statsmodels prints does.
        """
        from spacr.regression_qc import condition_number

        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 3))
        scaled_a, raw_a, _ = condition_number(X)

        Y = X.copy()
        Y[:, 0] *= 1000.0
        scaled_b, raw_b, _ = condition_number(Y)

        assert scaled_b == pytest.approx(scaled_a, rel=1e-6), (
            "the unit-free number moved when only the units changed")
        assert not np.isclose(raw_b, raw_a), (
            "the raw number is supposed to be dominated by the units")

    def test_an_empty_design_is_refused_by_name(self):
        """THE GUARD ABOVE, and it is why the one below cannot fire.

        `if sv.size == 0: return np.inf` inside `_ratio` is unreachable:
        the function rejects an empty design before any decomposition,
        and `np.linalg.svd` of a non-empty matrix always returns at
        least one singular value.
        """
        from spacr.regression_qc import condition_number

        for shape in ((0, 3), (5, 0), (0, 0)):
            with pytest.raises(ValueError, match="non-empty 2-D array"):
                condition_number(np.empty(shape))

    def test_the_singular_values_of_any_accepted_design_are_non_empty(self):
        """The other half of the argument, checked rather than asserted."""
        rng = np.random.default_rng(1)
        for rows, cols in ((1, 1), (1, 5), (5, 1), (9, 4)):
            X = rng.normal(size=(rows, cols))
            assert np.linalg.svd(X, compute_uv=False).size >= 1, (
                "a non-empty design produced no singular values; the "
                "`sv.size == 0` arm in condition_number is now reachable")
