"""CPU coverage (wave 2) for the ``spacr.utils.build_loss`` multiclass tails
that the first pass over this region left untouched: the tensor-``focal_alpha``
normalisation, the ``logit_adjust_ce`` / ``la_ce`` branch and the asymmetric
loss (``asl``) closure.

Everything runs on tiny CPU tensors, offline, with no figures.  Each test pins
an exact numeric identity (plain cross-entropy, the Menon logit adjustment,
or the BCE degenerate case of ASL) rather than merely calling the closure.

The sibling file ``test_cov_utils_augment_stats.py`` covers
augment_classes / annotate_predictions / add_images_to_tar / fishers_odds / MLR
in the same source region; this file only fills the loss-builder gap.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Guard against a stray figure leaking out of an import side effect."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def batch():
    """Deterministic 6-sample, 3-class logits/label batch."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 3, generator=g)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    return logits, y


# ---------------------------------------------------------------------------
# focal_ce: tensor focal_alpha
# ---------------------------------------------------------------------------

def test_focal_ce_per_class_alpha_tensor_is_cast_to_float32(batch):
    """A per-class alpha tensor (numel == num_classes) is cast to float32."""
    from spacr.utils import build_loss
    logits, y = batch
    alpha = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    loss_fn = build_loss("focal_ce", num_classes=3, focal_alpha=alpha, focal_gamma=2.0)

    loss = loss_fn(logits, y)
    # The cast on the alpha tensor is what keeps the loss in float32 even though
    # the caller handed in a float64 tensor.
    assert loss.dtype == torch.float32
    assert loss.ndim == 0
    assert float(loss) > 0.0
    # the caller's tensor is not mutated in place
    assert alpha.dtype == torch.float64


def test_focal_ce_alpha_tensor_of_wrong_length_is_left_alone(batch):
    """numel != num_classes skips the cast, so a float64 alpha stays float64."""
    from spacr.utils import build_loss
    logits, y = batch
    alpha = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    loss_fn = build_loss("focal_ce", num_classes=3, focal_alpha=alpha, focal_gamma=2.0)

    loss = loss_fn(logits, y)
    assert loss.dtype == torch.float64          # no .to(torch.float) happened

    # same alpha values as the 3-element case for the classes actually used,
    # so the value agrees with the cast variant to float32 precision
    cast_fn = build_loss("focal_ce", num_classes=3,
                         focal_alpha=torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64),
                         focal_gamma=2.0)
    assert float(loss) == pytest.approx(float(cast_fn(logits, y)), rel=1e-6)


def test_focal_ce_per_class_alpha_weights_each_sample_by_its_class(batch):
    """alpha=[1,0,0] zeroes every non-zero-class sample and leaves class 0 intact."""
    from spacr.utils import build_loss
    logits, _ = batch
    alpha = torch.tensor([1.0, 0.0, 0.0])
    weighted = build_loss("focal_ce", num_classes=3, focal_alpha=alpha)
    unweighted = build_loss("focal_ce", num_classes=3, focal_alpha=None)

    all_class1 = torch.ones(6, dtype=torch.long)
    assert float(weighted(logits, all_class1)) == 0.0

    all_class0 = torch.zeros(6, dtype=torch.long)
    assert float(weighted(logits, all_class0)) == pytest.approx(
        float(unweighted(logits, all_class0)), rel=1e-6)


def test_focal_ce_per_class_alpha_accepts_one_hot_targets(batch):
    """One-hot targets are argmax-ed, giving the same loss as index targets."""
    from spacr.utils import build_loss
    logits, y = batch
    alpha = torch.tensor([1.0, 0.5, 0.25])
    loss_fn = build_loss("focal_ce", num_classes=3, focal_alpha=alpha)
    onehot = F.one_hot(y, num_classes=3).float()
    assert float(loss_fn(logits, onehot)) == pytest.approx(float(loss_fn(logits, y)))


# ---------------------------------------------------------------------------
# logit_adjust_ce / la_ce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["logit_adjust_ce", "la_ce"])
def test_logit_adjust_requires_class_counts(name):
    """Both aliases refuse to build without class_counts."""
    from spacr.utils import build_loss
    with pytest.raises(ValueError, match="requires class_counts"):
        build_loss(name, num_classes=3, class_counts=None)


def test_logit_adjust_with_tau_zero_is_plain_cross_entropy(batch):
    """tau == 0 leaves logit_adjust unset, so no shift is applied and no weights."""
    from spacr.utils import build_loss
    logits, y = batch
    counts = torch.tensor([90, 9, 1])
    loss_fn = build_loss("logit_adjust_ce", num_classes=3, class_counts=counts,
                         logit_adjust_tau=0.0)

    loss = loss_fn(logits, y)
    assert float(loss) == pytest.approx(float(F.cross_entropy(logits, y)), rel=1e-6)
    # explicitly NOT the class-weighted CE that 'ce' would give for these counts
    weighted = build_loss("ce_weighted", num_classes=3, class_counts=counts)
    assert float(loss) != pytest.approx(float(weighted(logits, y)), rel=1e-3)


@pytest.mark.parametrize("name", ["logit_adjust_ce", "la_ce"])
def test_logit_adjust_shifts_logits_by_menon_prior(name, batch):
    """tau > 0 adds +tau*log(prior) to the logits before cross-entropy.

    Menon et al. 2020: the TRAIN-time adjustment is +tau*log(prior). The
    negated form is the post-hoc inference correction and must not be used
    here — applied during training it compounds the class imbalance.
    """
    from spacr.utils import build_loss
    logits, y = batch
    counts = torch.tensor([90, 9, 1])
    tau = 1.0
    loss_fn = build_loss(name, num_classes=3, class_counts=counts,
                         logit_adjust_tau=tau)

    c = torch.clamp(counts.to(torch.float), min=1.0)
    priors = c / c.sum()
    expected_shift = tau * priors.log()
    expected = F.cross_entropy(logits + expected_shift, y)

    loss = loss_fn(logits, y)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    # the adjustment really moved the loss away from the unadjusted value
    assert float(loss) != pytest.approx(float(F.cross_entropy(logits, y)), rel=1e-3)


def test_logit_adjust_accepts_one_hot_targets_and_backprops(batch):
    """One-hot targets are argmax-ed and the closure stays differentiable."""
    from spacr.utils import build_loss
    logits, y = batch
    counts = torch.tensor([50, 40, 10])
    loss_fn = build_loss("logit_adjust_ce", num_classes=3, class_counts=counts,
                         logit_adjust_tau=0.5)

    onehot = F.one_hot(y, num_classes=3).float()
    assert float(loss_fn(logits, onehot)) == pytest.approx(float(loss_fn(logits, y)))

    x = logits.clone().requires_grad_(True)
    loss_fn(x, y).backward()
    assert x.grad is not None
    assert x.grad.shape == logits.shape
    assert torch.isfinite(x.grad).all()


def test_logit_adjust_via_auto_on_imbalanced_counts(batch):
    """'auto' picks logit_adjust_ce when the rarest class is below 10%."""
    from spacr.utils import build_loss
    logits, y = batch
    counts = torch.tensor([90, 9, 1])
    auto_fn = build_loss("auto", num_classes=3, class_counts=counts,
                         logit_adjust_tau=1.0)
    explicit = build_loss("logit_adjust_ce", num_classes=3, class_counts=counts,
                          logit_adjust_tau=1.0)
    assert float(auto_fn(logits, y)) == pytest.approx(float(explicit(logits, y)))


# ---------------------------------------------------------------------------
# asl / asymmetric_loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["asl", "asymmetric_loss"])
def test_asl_index_targets_are_one_hot_encoded(name, batch):
    """1-D index targets are expanded to one-hot, matching an explicit one-hot."""
    from spacr.utils import build_loss
    logits, y = batch
    loss_fn = build_loss(name, num_classes=3)
    onehot = F.one_hot(y, num_classes=3).float()

    from_idx = loss_fn(logits, y)
    from_onehot = loss_fn(logits, onehot)
    assert from_idx.ndim == 0
    assert float(from_idx) > 0.0
    assert float(from_idx) == pytest.approx(float(from_onehot))


def test_asl_degenerates_to_bce_when_gammas_and_clip_are_zero(batch):
    """gamma_pos=gamma_neg=0 and clip=0 make ASL exactly mean BCE-with-logits."""
    from spacr.utils import build_loss
    logits, y = batch
    loss_fn = build_loss("asl", num_classes=3, asl_gamma_pos=0.0,
                         asl_gamma_neg=0.0, asl_clip=0.0)
    onehot = F.one_hot(y, num_classes=3).float()

    expected = F.binary_cross_entropy_with_logits(logits, onehot)
    assert float(loss_fn(logits, y)) == pytest.approx(float(expected), rel=1e-5)


def test_asl_reshapes_flattened_float_targets(batch):
    """A non-1-D target is view(-1, num_classes)-ed, so a flat (1, N*C) row works."""
    from spacr.utils import build_loss
    logits, y = batch
    loss_fn = build_loss("asl", num_classes=3)
    onehot = F.one_hot(y, num_classes=3).float()
    flat = onehot.reshape(1, -1)          # (1, 18) -> viewed back to (6, 3)
    assert flat.shape == (1, 18)
    assert float(loss_fn(logits, flat)) == pytest.approx(float(loss_fn(logits, onehot)))


def test_asl_rewards_confident_correct_predictions(batch):
    """Logits that match the labels score far lower than logits that invert them."""
    from spacr.utils import build_loss
    _, y = batch
    loss_fn = build_loss("asl", num_classes=3)
    onehot = F.one_hot(y, num_classes=3).float()
    good = (onehot * 12.0) - 6.0          # +6 on the true class, -6 elsewhere
    bad = -good

    assert float(loss_fn(good, y)) < float(loss_fn(bad, y))
    assert float(loss_fn(good, y)) < 1e-2
    assert torch.isfinite(loss_fn(bad, y))


def test_asl_clip_lowers_the_negative_term(batch):
    """A positive asl_clip shifts the negative probabilities and shrinks the loss."""
    from spacr.utils import build_loss
    logits, y = batch
    no_clip = build_loss("asl", num_classes=3, asl_clip=0.0)
    clipped = build_loss("asl", num_classes=3, asl_clip=0.05)
    assert float(clipped(logits, y)) < float(no_clip(logits, y))


def test_asl_is_differentiable(batch):
    """The ASL closure backpropagates into the logits."""
    from spacr.utils import build_loss
    logits, y = batch
    loss_fn = build_loss("asl", num_classes=3)
    x = logits.clone().requires_grad_(True)
    loss_fn(x, y).backward()
    assert x.grad is not None and x.grad.shape == logits.shape
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0
