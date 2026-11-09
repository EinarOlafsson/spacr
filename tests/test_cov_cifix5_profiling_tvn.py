"""The ``'tvn'`` profiling step refuses controls it cannot whiten on.

``test_profiling_tvn.py`` covers a screen without negative controls; these
cover the two refusals of :func:`spacr.sp_stats._whiten_profiles` itself:
too few complete control wells, and controls the whitening cannot use. Both
must surface as the profiling error the Measure run reports, not as a bare
``ValueError`` from the whitening.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s

FEATURES = ["a", "b", "c"]


def _wells(n=8, seed=0):
    """``n`` wells with three random features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "well", [f"w{k}" for k in range(n)])
    return frame


def test_one_complete_control_well_is_too_few():
    wells = _wells()
    negative = np.zeros(len(wells), bool)
    negative[:2] = True
    finite = np.ones(len(wells), bool)
    finite[1] = False  # the second control has a missing feature
    with pytest.raises(s._ProfilingError, match="at least 2 negative-control wells.*1 have"):
        s._whiten_profiles(wells, FEATURES, finite, negative, 0.0, [])


def test_constant_controls_are_refused_as_a_profiling_error():
    wells = _wells()
    negative = np.zeros(len(wells), bool)
    negative[:3] = True
    wells.loc[negative, FEATURES] = 1.0
    with pytest.raises(s._ProfilingError, match="constant in the reference rows") as caught:
        s._whiten_profiles(wells, FEATURES, np.ones(len(wells), bool), negative, 0.0, [])
    assert isinstance(caught.value.__cause__, ValueError)


def test_only_complete_wells_are_whitened_and_the_step_is_noted():
    wells = _wells(n=12)
    negative = np.zeros(len(wells), bool)
    negative[:6] = True
    finite = np.ones(len(wells), bool)
    finite[-1] = False
    wells.loc[len(wells) - 1, "a"] = np.nan
    notes = []
    out = s._whiten_profiles(wells, FEATURES, finite, negative, 0.0, notes)
    pd.testing.assert_frame_equal(out.iloc[[-1]], wells.iloc[[-1]])
    controls = out.loc[negative, FEATURES].to_numpy(float)
    # Whitened on the controls: their covariance is the identity.
    np.testing.assert_allclose(np.cov(controls, rowvar=False, ddof=1), np.eye(3), atol=1e-6)
    assert len(notes) == 1
