"""The coefficient threshold, and estimating it from the controls.

Asked for on 2026-08-16: "in the most recent graphs i see a significance
threshold but i do not see a coefficient threshold (which should be based on
the control gRNAs, it might be to high or it might have been removed)".

TWO ANSWERS, both pinned here.

It was never removed -- it is OFF BY DEFAULT. `effect_threshold=None` with
`threshold_method='value'` resolves to None and draws no line, which is why a
volcano shows an alpha line and no effect line until someone sets one.

And "it might be too high" is true of one method and not another. Measured on
the tsg101 screen (823 guides, 24 controls, 3x):

    std, all guides      1.8685    2.2x the control-based cut
    mad, all guides      0.8379    1.01x -- 88 guides pass
    mad, controls only   0.8322            89 guides pass

So the inflation argument is decisive against `std` and nearly irrelevant
against `mad`, which is what a median absolute deviation is for. `control`
earns its place by being the defensible one to write in a methods section,
and because a stronger screen than this one pulls `mad` up while leaving the
controls where they are -- which the last test here demonstrates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.volcano_style import VolcanoStyle, _resolve_effect_threshold


def _effects(n_guides=800, n_controls=24, hit_shift=0.0, seed=0):
    """Controls and guides drawn from one null, plus optional planted hits."""
    rng = np.random.default_rng(seed)
    controls = rng.normal(0.0, 1.0, n_controls)
    guides = rng.normal(0.0, 1.0, n_guides)
    if hit_shift:
        n_hits = n_guides // 10
        guides[:n_hits] += hit_shift
    values = np.concatenate([controls, guides])
    mask = np.zeros(values.size, bool)
    mask[:n_controls] = True
    return values, mask


# --------------------------------------------------------------------------- #
#  It was never removed: it is off until asked for
# --------------------------------------------------------------------------- #

def test_no_effect_threshold_is_drawn_by_default():
    """The answer to "it might have been removed". It was not."""
    style = VolcanoStyle()
    assert style.effect_threshold is None
    assert style.threshold_method == "value"
    assert _resolve_effect_threshold(np.array([1.0, 2.0, 3.0]), style) is None


def test_an_explicit_value_is_scaled_by_the_multiplier():
    style = VolcanoStyle(effect_threshold=0.5, threshold_multiplier=3.0)
    assert _resolve_effect_threshold(np.array([1.0]), style) == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
#  The control-based cut
# --------------------------------------------------------------------------- #

def test_the_control_cut_is_estimated_from_the_controls_only():
    """Planted hits must not move it. That is the whole point."""
    clean, mask = _effects(hit_shift=0.0, seed=1)
    spiked, _ = _effects(hit_shift=6.0, seed=1)     # same nulls, big hits
    style = VolcanoStyle(threshold_method="control", threshold_multiplier=3.0)

    a = _resolve_effect_threshold(clean, style, mask)
    b = _resolve_effect_threshold(spiked, style, mask)

    assert a == pytest.approx(b), (
        "the control cut moved when hits were planted -- it is not being "
        "estimated from the controls alone")


def test_a_strong_screen_pulls_mad_up_but_not_the_control_cut():
    """Why `control` earns its place even though it matches `mad` on a
    screen with a modest signal."""
    spiked, mask = _effects(hit_shift=8.0, seed=2)

    mad = _resolve_effect_threshold(
        spiked, VolcanoStyle(threshold_method="mad", threshold_multiplier=3.0))
    control = _resolve_effect_threshold(
        spiked, VolcanoStyle(threshold_method="control",
                             threshold_multiplier=3.0), mask)

    assert mad > control, (
        "with a strong planted signal the all-guides MAD should sit above "
        "the control-only cut")


def test_std_is_the_one_that_inflates():
    """Measured on real data at 2.2x. Reproduced here in principle."""
    spiked, mask = _effects(hit_shift=8.0, seed=3)

    std = _resolve_effect_threshold(
        spiked, VolcanoStyle(threshold_method="std", threshold_multiplier=3.0))
    control = _resolve_effect_threshold(
        spiked, VolcanoStyle(threshold_method="control",
                             threshold_multiplier=3.0), mask)

    assert std > control * 1.3


# --------------------------------------------------------------------------- #
#  It refuses rather than guessing
# --------------------------------------------------------------------------- #

def test_it_refuses_without_a_control_column():
    """Silently falling back to `mad` would hand back a number the user
    would reasonably describe as control-based in a paper."""
    style = VolcanoStyle(threshold_method="control")
    with pytest.raises(ValueError) as caught:
        _resolve_effect_threshold(np.array([1.0, 2.0]), style, None)
    assert "control_column" in str(caught.value)


def test_it_refuses_with_too_few_controls():
    """A MAD over three points is noise pretending to be a threshold -- and
    a threshold that comes out too LOW calls everything a hit."""
    values = np.array([0.1, -0.2, 0.3, 5.0, -4.0])
    mask = np.array([True, True, True, False, False])
    style = VolcanoStyle(threshold_method="control")

    with pytest.raises(ValueError) as caught:
        _resolve_effect_threshold(values, style, mask)
    assert "at least" in str(caught.value)


def test_it_refuses_when_the_controls_have_no_spread():
    """Zero MAD would cut at zero and mark every guide significant."""
    values = np.concatenate([np.full(10, 0.5), np.array([3.0, -3.0])])
    mask = np.zeros(values.size, bool)
    mask[:10] = True
    style = VolcanoStyle(threshold_method="control")

    with pytest.raises(ValueError) as caught:
        _resolve_effect_threshold(values, style, mask)
    assert "zero spread" in str(caught.value)


def test_an_unknown_method_names_the_ones_that_exist():
    style = VolcanoStyle(threshold_method="vibes")
    with pytest.raises(ValueError) as caught:
        _resolve_effect_threshold(np.array([1.0]), style)
    assert "control" in str(caught.value)


# --------------------------------------------------------------------------- #
#  Through the renderer's own preparation path
# --------------------------------------------------------------------------- #

def test_the_control_column_is_read_off_the_results_frame():
    from spacr.volcano_style import _prepare

    values, mask = _effects(n_guides=200, n_controls=20, seed=4)
    frame = pd.DataFrame({
        "standardized_marginal_effect": values,
        "adjusted_p_value": np.full(values.size, 0.01),
        "guide": [f"g{i}" for i in range(values.size)],
        "is_control": mask,
    })
    style = VolcanoStyle(threshold_method="control", control_column="is_control")

    _frame, _x, _y, _raw, _sig, cut = _prepare(frame, style)

    assert cut is not None and cut > 0


def test_a_missing_control_column_says_which_one():
    from spacr.volcano_style import _prepare

    frame = pd.DataFrame({
        "standardized_marginal_effect": [1.0, 2.0],
        "adjusted_p_value": [0.01, 0.2],
        "guide": ["a", "b"],
    })
    style = VolcanoStyle(threshold_method="control",
                         control_column="not_a_column")

    with pytest.raises(ValueError) as caught:
        _prepare(frame, style)
    assert "not_a_column" in str(caught.value)
