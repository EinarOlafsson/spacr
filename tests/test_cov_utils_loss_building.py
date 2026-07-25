"""CPU coverage for spacr.utils' loss-building block.

Covers the tail of ``suggest_training_changes`` (class-imbalance heuristic),
``estimate_class_counts`` (folder fast path *and* the slow DataLoader
fallback) and every branch of ``build_loss`` -- including the nested
``_focal_bce`` / ``_focal_ce`` / ``_asl`` / ``_auto_choice`` helpers.

All losses are checked against an independent closed-form reference: with
``gamma=0`` focal losses collapse to (weighted) cross-entropy / BCE, and with
``gamma_pos=gamma_neg=0, clip=0`` the asymmetric loss collapses to BCE, so the
assertions pin real numbers rather than "it returned a tensor".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


@pytest.fixture(autouse=True)
def _no_figures():
    """Never let a stray figure survive a test in this module."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# suggest_training_changes -- heuristic 8: accuracy >> macro-F1
# ---------------------------------------------------------------------------

def _write_progress(dst, *, n, val_acc, val_f1, val_loss=None):
    """Write train/validation progress CSVs in the layout _save_progress uses."""
    os.makedirs(dst, exist_ok=True)
    epochs = np.arange(1, n + 1)
    if val_loss is None:
        val_loss = np.linspace(0.70, 0.20, n)
    tr = pd.DataFrame({
        "epoch": epochs,
        "loss": np.linspace(0.70, 0.05, n),
        "accuracy": np.linspace(0.60, 0.98, n),
        "f1_macro": np.linspace(0.60, 0.97, n),
    })
    va = pd.DataFrame({
        "epoch": epochs,
        "loss": val_loss,
        "accuracy": np.full(n, val_acc),
        "f1_macro": np.full(n, val_f1),
    })
    tp = os.path.join(dst, "train_progress.csv")
    vp = os.path.join(dst, "validation_progress.csv")
    tr.to_csv(tp, index=False)
    va.to_csv(vp, index=False)
    return tp, vp


def test_advice_flags_class_imbalance_when_accuracy_beats_macro_f1(tmp_path):
    """val_accuracy - val_f1_macro > 0.10 raises the imbalance flag + advice."""
    from spacr.utils import suggest_training_changes
    tr, va = _write_progress(str(tmp_path), n=30, val_acc=0.95, val_f1=0.60)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)

    assert "class_imbalance_suspected" in out["flags"]
    joined = " ".join(out["suggestions"])
    assert "macro-F1" in joined
    assert "per-class metrics" in joined
    # the summary still carries the scalars that drove the decision
    assert out["summary"]["final_metrics"]["val_accuracy"] == pytest.approx(0.95)
    assert out["summary"]["final_metrics"]["val_f1_macro"] == pytest.approx(0.60)


def test_advice_no_imbalance_flag_when_gap_is_small(tmp_path):
    """A 0.03 gap is below the 0.10 threshold -> flag must be absent."""
    from spacr.utils import suggest_training_changes
    tr, va = _write_progress(str(tmp_path), n=30, val_acc=0.90, val_f1=0.87)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    assert "class_imbalance_suspected" not in out["flags"]


def test_advice_suggestions_are_deduplicated(tmp_path):
    """Several heuristics fire at once; each suggestion appears exactly once."""
    from spacr.utils import suggest_training_changes
    # flat val loss (plateau) that also regressed from its best, plus a huge
    # accuracy/F1 gap -> multiple overlapping suggestion blocks.
    n = 30
    val_loss = np.concatenate([np.full(5, 0.30), np.full(n - 5, 0.50)])
    tr, va = _write_progress(str(tmp_path), n=n, val_acc=0.95, val_f1=0.40,
                             val_loss=val_loss)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    assert len(out["suggestions"]) == len(set(out["suggestions"]))
    assert "class_imbalance_suspected" in out["flags"]
    assert "past_best_regression" in out["flags"]


# ---------------------------------------------------------------------------
# estimate_class_counts
# ---------------------------------------------------------------------------

class _ListLoader:
    """Minimal stand-in for a DataLoader yielding (images, labels, paths)."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_estimate_class_counts_from_folders(tmp_path, capsys):
    """Fast path: count files per class dir, ignore subdirs and missing dirs."""
    from spacr.utils import estimate_class_counts
    src = tmp_path / "train"
    (src / "nc").mkdir(parents=True)
    (src / "pc").mkdir(parents=True)
    for i in range(3):
        (src / "nc" / f"a{i}.png").write_bytes(b"x")
    for i in range(5):
        (src / "pc" / f"b{i}.png").write_bytes(b"x")
    (src / "pc" / "nested").mkdir()          # directories must not be counted
    (src / "pc" / "nested" / "c.png").write_bytes(b"x")

    counts = estimate_class_counts(None, 3, src=str(src),
                                   classes=["nc", "pc", "absent"])
    assert counts.dtype == torch.long
    assert counts.tolist() == [3, 5, 0]
    assert "Class counts (from folders)" in capsys.readouterr().out


def test_estimate_class_counts_dataloader_fallback(capsys):
    """Slow path: iterate the loader, warn, and bincount long targets."""
    from spacr.utils import estimate_class_counts
    batches = [
        (torch.zeros(4, 1, 2, 2), torch.tensor([0, 1, 1, 2]), ["p"] * 4),
        (torch.zeros(3, 1, 2, 2), torch.tensor([2, 2, 0]), ["p"] * 3),
    ]
    counts = estimate_class_counts(_ListLoader(batches), 3)
    assert counts.dtype == torch.long
    assert counts.tolist() == [2, 2, 3]
    assert int(counts.sum()) == 7
    out = capsys.readouterr().out
    assert "iterating DataLoader" in out


def test_estimate_class_counts_fallback_accepts_one_hot_and_float(capsys):
    """One-hot (N,C) and binary float targets both resolve to indices."""
    from spacr.utils import estimate_class_counts
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    binary = torch.tensor([0.1, 0.9, 0.8, 0.2])   # -> 0,1,1,0
    batches = [
        (torch.zeros(3), onehot, ["p"] * 3),
        (torch.zeros(4), binary, ["p"] * 4),
    ]
    counts = estimate_class_counts(_ListLoader(batches), 2)
    assert counts.tolist() == [3, 4]
    capsys.readouterr()


def test_estimate_class_counts_fallback_truncates_out_of_range_labels(capsys):
    """Labels >= num_classes are dropped rather than overflowing the tensor."""
    from spacr.utils import estimate_class_counts
    batches = [(torch.zeros(4), torch.tensor([0, 1, 2, 3]), ["p"] * 4)]
    counts = estimate_class_counts(_ListLoader(batches), 2)
    assert counts.shape == (2,)
    assert counts.tolist() == [1, 1]
    capsys.readouterr()


def test_estimate_class_counts_fallback_used_when_classes_missing(capsys):
    """src without classes (or vice versa) must not take the folder path."""
    from spacr.utils import estimate_class_counts
    batches = [(torch.zeros(2), torch.tensor([1, 1]), ["p"] * 2)]
    counts = estimate_class_counts(_ListLoader(batches), 2, src="/nonexistent",
                                   classes=None)
    assert counts.tolist() == [0, 2]
    assert "iterating DataLoader" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# build_loss -- shared batches
# ---------------------------------------------------------------------------

def _mc_batch(C=3, n=8):
    return torch.randn(n, C), torch.randint(0, C, (n,))


def _bin_batch(n=8):
    return torch.randn(n, 1), torch.tensor([0., 1., 1., 0., 1., 0., 0., 1.])[:n]


# ----- target coercion inside the nested _infer_indices --------------------

def test_ce_accepts_one_hot_targets():
    """A 2-D target is argmax'ed to indices (nested _infer_indices branch)."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    onehot = F.one_hot(idx, num_classes=3).float()
    fn = build_loss("ce", num_classes=3)
    got = fn(logits, onehot)
    assert torch.allclose(got, F.cross_entropy(logits, idx), atol=1e-6)


def test_ce_smooth_accepts_one_hot_targets_and_smooths():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    onehot = F.one_hot(idx, num_classes=3).float()
    fn = build_loss("ce_smooth", num_classes=3, label_smoothing=0.2)
    got = fn(logits, onehot)
    expected = F.cross_entropy(logits, idx, label_smoothing=0.2)
    assert torch.allclose(got, expected, atol=1e-6)
    assert not torch.allclose(got, F.cross_entropy(logits, idx), atol=1e-4)


# ----- binary branch -------------------------------------------------------

def test_focal_bce_without_alpha_reduces_to_bce_at_gamma_zero():
    from spacr.utils import build_loss
    logits, target = _bin_batch()
    fn = build_loss("focal_bce", num_classes=1, focal_gamma=0.0,
                    focal_alpha=None)
    expected = F.binary_cross_entropy_with_logits(logits, target.view(-1, 1))
    assert torch.allclose(fn(logits, target), expected, atol=1e-6)


def test_focal_bce_alpha_reweights_positives_and_negatives():
    """alpha branch: positives get alpha, negatives get (1 - alpha)."""
    from spacr.utils import build_loss
    logits, target = _bin_batch()
    y = target.view(-1, 1)
    alpha = 0.25
    fn = build_loss("focal_bce", num_classes=1, focal_gamma=0.0,
                    focal_alpha=alpha)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    expected = ((alpha * y + (1 - alpha) * (1 - y)) * ce).mean()
    assert torch.allclose(fn(logits, target), expected, atol=1e-6)
    # alpha=1 silences the negatives entirely
    fn1 = build_loss("focal_bce", num_classes=1, focal_gamma=0.0,
                     focal_alpha=1.0)
    assert torch.allclose(fn1(logits, target), (y * ce).mean(), atol=1e-6)


def test_focal_bce_gamma_downweights_easy_examples():
    from spacr.utils import build_loss
    logits = torch.tensor([[6.0], [-6.0], [0.2]])
    target = torch.tensor([1.0, 0.0, 1.0])
    easy = build_loss("focal_bce", num_classes=1, focal_gamma=2.0)(logits, target)
    plain = build_loss("focal_bce", num_classes=1, focal_gamma=0.0)(logits, target)
    assert float(easy) < float(plain)


def test_binary_bce_matches_torch():
    from spacr.utils import build_loss
    logits, target = _bin_batch()
    fn = build_loss("binary_cross_entropy_with_logits", num_classes=1)
    expected = F.binary_cross_entropy_with_logits(logits, target.view(-1, 1))
    assert torch.allclose(fn(logits, target), expected, atol=1e-6)


def test_binary_rejects_multiclass_loss_types():
    """num_classes=1 only accepts BCE flavours."""
    from spacr.utils import build_loss
    for lt in ("ce", "ce_weighted", "asl", "logit_adjust_ce"):
        with pytest.raises(ValueError, match="not valid for binary"):
            build_loss(lt, num_classes=1)


# ----- multiclass focal-CE alpha handling ---------------------------------

def test_focal_ce_scalar_alpha_scales_cross_entropy():
    """Scalar alpha -> a = float(alpha) branch."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    fn = build_loss("focal_ce", num_classes=3, focal_gamma=0.0, focal_alpha=0.25)
    expected = 0.25 * F.cross_entropy(logits, idx)
    assert torch.allclose(fn(logits, idx), expected, atol=1e-6)


def test_focal_ce_per_class_alpha_tensor_indexes_by_label():
    """Per-class alpha tensor -> a = alpha[y_idx] branch."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    alpha = torch.tensor([0.1, 0.5, 0.4])
    fn = build_loss("focal_ce", num_classes=3, focal_gamma=0.0,
                    focal_alpha=alpha)
    per_sample = F.cross_entropy(logits, idx, reduction="none")
    expected = (alpha[idx] * per_sample).mean()
    got = fn(logits, idx)
    assert torch.allclose(got, expected, atol=1e-6)
    # and it really is label-dependent, not a global scale
    assert not torch.allclose(got, alpha.mean() * F.cross_entropy(logits, idx),
                              atol=1e-4)


def test_focal_ce_single_element_alpha_tensor_is_treated_as_scalar():
    """A 1-element tensor falls through to the float(alpha) branch."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    fn = build_loss("focal_ce", num_classes=3, focal_gamma=0.0,
                    focal_alpha=torch.tensor([0.3]))
    assert torch.allclose(fn(logits, idx), 0.3 * F.cross_entropy(logits, idx),
                          atol=1e-6)


def test_focal_ce_alpha_tensor_is_cast_to_float():
    """An integer per-class alpha tensor is cast, so gather/mul do not raise."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    alpha_int = torch.tensor([1, 2, 3])
    fn = build_loss("focal_ce", num_classes=3, focal_gamma=0.0,
                    focal_alpha=alpha_int)
    per_sample = F.cross_entropy(logits, idx, reduction="none")
    expected = (alpha_int.float()[idx] * per_sample).mean()
    out = fn(logits, idx)
    assert out.dtype == torch.float32
    assert torch.allclose(out, expected, atol=1e-6)


def test_focal_ce_without_alpha_reduces_to_ce_at_gamma_zero():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    fn = build_loss("focal_ce", num_classes=3, focal_gamma=0.0)
    assert torch.allclose(fn(logits, idx), F.cross_entropy(logits, idx),
                          atol=1e-6)


def test_focal_ce_gamma_downweights_confident_correct_predictions():
    from spacr.utils import build_loss
    logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    idx = torch.tensor([0, 1])
    focused = build_loss("focal_ce", num_classes=3, focal_gamma=2.0)(logits, idx)
    plain = build_loss("focal_ce", num_classes=3, focal_gamma=0.0)(logits, idx)
    assert float(focused) < float(plain)


# ----- asymmetric loss -----------------------------------------------------

def test_asl_reduces_to_bce_with_zero_gammas_and_no_clip():
    """gamma_pos=gamma_neg=0, clip=0 -> plain multi-label BCE over one-hot."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    fn = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                    asl_gamma_neg=0.0, asl_clip=0.0)
    y = F.one_hot(idx, num_classes=3).float()
    expected = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(fn(logits, idx), expected, atol=1e-5)


def test_asl_accepts_one_hot_and_index_targets_identically():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    y = F.one_hot(idx, num_classes=3).float()
    fn = build_loss("asl", num_classes=3)
    assert torch.allclose(fn(logits, idx), fn(logits, y), atol=1e-6)


def test_asl_clip_shrinks_the_negative_term():
    """The probability-margin clip removes easy-negative gradient/loss."""
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    no_clip = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                         asl_gamma_neg=0.0, asl_clip=0.0)(logits, idx)
    clipped = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                         asl_gamma_neg=0.0, asl_clip=0.05)(logits, idx)
    assert float(clipped) < float(no_clip)


def test_asl_negative_focusing_downweights_easy_negatives():
    from spacr.utils import build_loss
    logits = torch.tensor([[4.0, -6.0, -6.0]])
    idx = torch.tensor([0])
    hard = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                      asl_gamma_neg=0.0, asl_clip=0.0)(logits, idx)
    focused = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                         asl_gamma_neg=4.0, asl_clip=0.0)(logits, idx)
    assert float(focused) < float(hard)
    assert float(focused) >= 0.0


def test_asl_alias_asymmetric_loss():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    a = build_loss("asymmetric_loss", num_classes=3)(logits, idx)
    b = build_loss("asl", num_classes=3)(logits, idx)
    assert torch.allclose(a, b, atol=1e-6)


# ----- auto heuristic ------------------------------------------------------

def test_auto_picks_logit_adjust_ce_for_rare_classes():
    """min class proportion < 0.10 -> logit_adjust_ce (unweighted CE at tau=0)."""
    from spacr.utils import build_loss
    counts = torch.tensor([950, 50])
    logits, idx = _mc_batch(C=2)
    auto = build_loss("auto", num_classes=2, class_counts=counts,
                      logit_adjust_tau=0.0)(logits, idx)
    # logit_adjust_ce with tau=0 leaves the logits untouched -> plain CE
    assert torch.allclose(auto, F.cross_entropy(logits, idx), atol=1e-6)
    # ...which is NOT what "ce" would have produced with these counts
    ce = build_loss("ce", num_classes=2, class_counts=counts)(logits, idx)
    assert not torch.allclose(auto, ce, atol=1e-4)


def test_auto_picks_weighted_ce_for_mild_imbalance():
    """min proportion >= 0.10 -> 'ce', which still applies count weights."""
    from spacr.utils import build_loss
    counts = torch.tensor([700, 300])
    logits, idx = _mc_batch(C=2)
    auto = build_loss("auto", num_classes=2, class_counts=counts)(logits, idx)
    priors = counts.float() / counts.sum()
    inv = 1.0 / priors
    w = inv / inv.mean()
    assert torch.allclose(auto, F.cross_entropy(logits, idx, weight=w),
                          atol=1e-6)
    assert not torch.allclose(auto, F.cross_entropy(logits, idx), atol=1e-4)


def test_auto_without_counts_is_plain_ce():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    got = build_loss("auto", num_classes=3)(logits, idx)
    assert torch.allclose(got, F.cross_entropy(logits, idx), atol=1e-6)


def test_auto_binary_is_bce():
    from spacr.utils import build_loss
    logits, target = _bin_batch()
    got = build_loss("auto", num_classes=1)(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target.view(-1, 1))
    assert torch.allclose(got, expected, atol=1e-6)


def test_auto_with_all_zero_counts_does_not_divide_by_zero():
    """clamp_min(1) on the count sum keeps _auto_choice finite."""
    from spacr.utils import build_loss
    counts = torch.tensor([0, 0])
    logits, idx = _mc_batch(C=2)
    out = build_loss("auto", num_classes=2, class_counts=counts)(logits, idx)
    assert torch.isfinite(out)


# ----- weighting / error paths --------------------------------------------

def test_ce_weighted_requires_counts():
    from spacr.utils import build_loss
    with pytest.raises(ValueError, match="requires class_counts"):
        build_loss("ce_weighted", num_classes=3)


def test_ce_weighted_uses_inverse_frequency_weights():
    from spacr.utils import build_loss
    counts = torch.tensor([10.0, 30.0, 60.0])
    logits, idx = _mc_batch()
    got = build_loss("ce_weighted", num_classes=3, class_counts=counts)(logits, idx)
    priors = counts / counts.sum()
    inv = 1.0 / priors
    w = inv / inv.mean()
    assert torch.allclose(got, F.cross_entropy(logits, idx, weight=w), atol=1e-6)


def test_zero_counts_are_clamped_before_inversion():
    """A class with 0 samples must not produce inf/nan weights."""
    from spacr.utils import build_loss
    counts = torch.tensor([10, 0, 90])
    logits, idx = _mc_batch()
    out = build_loss("ce_weighted", num_classes=3, class_counts=counts)(logits, idx)
    assert torch.isfinite(out)


def test_logit_adjust_requires_counts():
    from spacr.utils import build_loss
    with pytest.raises(ValueError, match="requires class_counts"):
        build_loss("logit_adjust_ce", num_classes=3)


def test_unknown_loss_type_raises():
    from spacr.utils import build_loss
    with pytest.raises(ValueError, match="Unknown loss_type"):
        build_loss("nope", num_classes=3)


def test_loss_type_is_case_insensitive_and_none_defaults_to_ce():
    from spacr.utils import build_loss
    logits, idx = _mc_batch()
    upper = build_loss("CE_Smooth", num_classes=3, label_smoothing=0.1)(logits, idx)
    assert torch.allclose(upper, F.cross_entropy(logits, idx, label_smoothing=0.1),
                          atol=1e-6)
    default = build_loss(None, num_classes=3)(logits, idx)
    assert torch.allclose(default, F.cross_entropy(logits, idx), atol=1e-6)


def test_losses_are_differentiable():
    """Every builder returns a scalar that back-propagates to the logits."""
    from spacr.utils import build_loss
    counts = torch.tensor([10, 40, 50])
    for lt in ("ce", "ce_smooth", "ce_weighted", "focal_ce", "logit_adjust_ce",
               "asl"):
        logits = torch.randn(6, 3, requires_grad=True)
        idx = torch.randint(0, 3, (6,))
        loss = build_loss(lt, num_classes=3, class_counts=counts,
                          focal_alpha=0.5, logit_adjust_tau=1.0)(logits, idx)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_logit_adjust_uses_menon_sign():
    from spacr.utils import build_loss
    counts = torch.tensor([950, 50])
    priors = counts.float() / counts.sum()
    logits = torch.zeros(4, 2)
    idx = torch.tensor([0, 0, 0, 1])
    fn = build_loss("logit_adjust_ce", num_classes=2, class_counts=counts,
                    logit_adjust_tau=1.0)
    # priors sum to 1, so logsumexp(log priors) == 0 and the reference loss is
    # simply the mean negative log-prior of the labels: (3*0.0513 + 2.9957)/4.
    expected = F.cross_entropy(logits + 1.0 * priors.log(), idx)
    assert float(expected) == pytest.approx(0.7874, abs=1e-3)
    assert torch.allclose(fn(logits, idx), expected, atol=1e-5)
