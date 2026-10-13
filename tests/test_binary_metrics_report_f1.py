"""Binary training printed `F1(macro): nan` on every run.

Reported as issue #78. `_binary_metrics` returned accuracy, neg_accuracy,
pos_accuracy, prauc and optimal_threshold -- and no `f1_macro`. The print
sites read it with `.get('f1_macro', float('nan'))`, so the MISSING key
rendered as nan rather than raising.

The metric was not failing; it was absent. Output read

    Train F1(macro): nan, Val F1(macro): nan

beside a healthy accuracy of 0.82-0.93, which reads as a broken metric on a
working model rather than a metric nobody computed. `utils.py` even carries
diagnostic logic that flags runs where more than 20% of `f1_macro` values are
nan, implying the column was expected to exist.
"""

import numpy as np
import pytest

from spacr.deep_spacr import _binary_metrics


def _labelled(n=200, seed=0):
    """Probabilities correlated with the truth, so accuracy is healthy."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, n), 0, 1)
    return y, p


def test_binary_metrics_reports_a_finite_f1_macro():
    """The defect, stated as the number that used to be nan."""
    metrics = _binary_metrics(*_labelled())

    assert "f1_macro" in metrics, "the key the print sites read is still absent"
    assert np.isfinite(metrics["f1_macro"]), (
        "f1_macro is nan again, which is what the print line showed on every "
        "run")
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_f1_tracks_accuracy_on_a_balanced_problem():
    """A returned-but-wrong number would pass the test above.

    On a balanced binary problem macro F1 and accuracy stay close, so a
    number that wandered far from accuracy means the metric is computed
    against the wrong thing.
    """
    metrics = _binary_metrics(*_labelled())
    assert abs(metrics["f1_macro"] - metrics["accuracy"]) < 0.1, (
        f"f1_macro {metrics['f1_macro']:.4f} is far from accuracy "
        f"{metrics['accuracy']:.4f} on a balanced problem")


def test_a_perfect_classifier_scores_one():
    y = np.array([0, 0, 1, 1])
    metrics = _binary_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_an_always_negative_classifier_is_not_rewarded():
    """Accuracy alone would read 0.9 on a 90/10 split; macro F1 is what
    exposes that, which is why the print line asks for it."""
    y = np.array([0] * 90 + [1] * 10)
    metrics = _binary_metrics(y, np.full(100, 0.01))

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["f1_macro"] < 0.6, (
        "macro F1 rewarded a classifier that never predicts the minority "
        "class, so it is not being averaged over classes")


def test_the_key_matches_what_the_multiclass_helper_returns():
    """The print sites are shared, so the two helpers must agree on the name."""
    from spacr.deep_spacr import _multiclass_metrics

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 90)
    probs = rng.random((90, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    assert "f1_macro" in _multiclass_metrics(y, probs)
    assert "f1_macro" in _binary_metrics(*_labelled())
