"""spacr.active_learning.should_stop -- "flat" and "unmeasurable" are different sentences.

``should_stop`` appends "the held-out set is too small to resolve this" to a
flat verdict.  These tests hold it to *withholding* that sentence when the
plateau is measurable, or when no held-out size was recorded at all, and each
drives both sides of its pair in one function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al

def _curve(accuracies, new_labels=30, n_holdout=200):
    """A learning-curve frame with a chosen held-out accuracy per round."""
    n = len(accuracies)
    return pd.DataFrame({
        "round": list(range(n)),
        "finished_utc": [""] * n,
        "n_labels": list(np.cumsum([new_labels] * n)),
        "n_new_labels": [new_labels] * n,
        "n_holdout": [n_holdout] * n,
        "holdout_accuracy": list(accuracies),
        "holdout_f1_macro": list(accuracies),
        "per_class": [{}] * n,
        "split_rule": [""] * n,
        "model_type": [""] * n,
        "model_path": [""] * n,
        "card_path": [""] * n,
        "measure": [""] * n,
        "diversity": [""] * n,
        "notes": [[]] * n,
        "gain": pd.Series(list(accuracies)).diff(),
    })


# ---------------------------------------------------------------------------
# should_stop — "flat" and "unmeasurable" are different sentences
# ---------------------------------------------------------------------------

def test_a_measurable_plateau_is_not_excused_as_held_out_noise(tmp_path):
    """A real plateau must not be softened with a standard-error caveat.

    The stopping rule says "stop" for both a plateau it can measure and one
    that is merely below the noise floor, but the two call for different
    work: the first means more labels of this kind are not buying accuracy,
    the second means the held-out set is too small to tell. If the caveat
    about a small held-out set were attached to every flat verdict, an
    annotator with 20 000 held-out objects would be told to go and collect
    more held-out data they already have — and would keep annotating.
    """
    accuracies = [0.90, 0.90, 0.9025]

    measurable = al.should_stop(_curve(accuracies, n_holdout=20000),
                                label_window=50, min_gain=0.003)
    assert measurable.stop is True
    assert measurable.trend == "flat"
    assert measurable.gain == pytest.approx(0.0025)
    assert measurable.confident is True
    assert measurable.noise == pytest.approx(
        float(np.sqrt(0.9025 * 0.0975 / 20000)))
    assert "not buying measurable accuracy" in measurable.reason
    assert "standard error" not in measurable.reason
    assert "unmeasurable" not in measurable.reason

    # Identical curve, smaller held-out set: the same 0.25 % is now inside
    # the noise, and the caveat the verdict above withheld appears.
    unmeasurable = al.should_stop(_curve(accuracies, n_holdout=200),
                                  label_window=50, min_gain=0.003)
    assert unmeasurable.stop is True
    assert unmeasurable.trend == "flat"
    assert unmeasurable.gain == pytest.approx(0.0025)
    assert unmeasurable.confident is False
    assert "held-out set is only 200 objects" in unmeasurable.reason
    assert "unmeasurable, not that it is provably zero" in unmeasurable.reason


# ---------------------------------------------------------------------------
# should_stop — a curve that never recorded a held-out size still gets a
# verdict, and gets it without a standard error it cannot compute
# ---------------------------------------------------------------------------

def test_a_curve_with_no_recorded_holdout_size_still_returns_a_flat_verdict():
    """A missing ``n_holdout`` must not cost the annotator the stopping rule.

    ``noise`` is a binomial standard error, and it needs the size of the
    held-out set to exist. Rounds recorded by an older spaCR — or by a
    caller that assembled the curve by hand from ``holdout_accuracy`` alone
    — carry no ``n_holdout``, so ``noise`` is ``None``. The flat branch then
    has to decide whether to attach the "the held-out set is too small"
    caveat while holding no number to justify it, and the only honest answer
    is to withhold it: quoting a standard error of zero would tell the
    annotator the plateau is proven when nothing measured it.

    The ordering of that guard is the part worth pinning. ``noise is not
    None`` has to be tested *before* ``abs(gain) <= noise``, because
    comparing a float to ``None`` raises ``TypeError`` in Python 3. A
    refactor that swaps the two halves — or replaces them with
    ``abs(gain) <= (noise or 0)`` — turns "no held-out size recorded" from a
    usable verdict into a crash, or into a confident claim of convergence.
    """
    accuracies = [0.90, 0.90, 0.9025]

    sizeless = al.should_stop(_curve(accuracies, n_holdout=0),
                              label_window=50, min_gain=0.003)
    assert sizeless.stop is True
    assert sizeless.trend == "flat"
    assert sizeless.gain == pytest.approx(0.0025)
    assert sizeless.noise is None
    # No held-out size means no basis for calling the plateau confident.
    assert sizeless.confident is False
    assert "not buying measurable accuracy" in sizeless.reason
    assert "standard error" not in sizeless.reason
    assert "unmeasurable" not in sizeless.reason

    # The same curve WITH a held-out size, so the withheld caveat above is
    # the missing number doing its job and not a sentence nobody writes.
    sized = al.should_stop(_curve(accuracies, n_holdout=200),
                           label_window=50, min_gain=0.003)
    assert sized.stop is True
    assert sized.trend == "flat"
    assert sized.noise == pytest.approx(float(np.sqrt(0.9025 * 0.0975 / 200)))
    assert "held-out set is only 200 objects" in sized.reason
    assert "standard error" in sized.reason
