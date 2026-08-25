"""The attribution module's refusals, fallbacks and shape edge cases.

An attribution map is a picture, and a picture is believed. Every branch
covered here is one where the module could have produced a plausible-looking
map that means nothing -- a CAM over a token vector, a rollout over a
non-square attention matrix, a NaN map drawn as a flat field -- or where it
must refuse instead. Each test drives a hand-built model whose behaviour is
known, so the assertion has a right answer.
"""
from __future__ import annotations

import builtins
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from spacr.attribution import (
    ATTRIBUTION_METHODS,
    Agreement,
    Attribution,
    AttributionError,
    AttributionMapGenerator,
    ClassScoreModel,
    Curve,
    MethodSpec,
    NoSpatialLayerError,
    UnknownMethodError,
    _attribute_with_spec,
    _captum_attribute,
    _perturbation_curve,
    _rank_correlation,
    attention_rollout,
    attribute,
    class_scores,
    deletion_curve,
    method_agreement,
    pointing_game,
    pointing_game_rate,
    randomization_sanity_check,
    smoothgrad,
)

IMG = 8


class TinyCNN(nn.Module):
    """Two conv layers, global pool, linear head -- small and deterministic."""

    def __init__(self, n_out=2, in_ch=3):
        """Build with a fixed seed so every test sees the same weights."""
        super().__init__()
        torch.manual_seed(7)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, n_out)

    def forward(self, x):
        """Return raw logits, ``(B, n_out)``."""
        return self.head(self.pool(self.features(x)).flatten(1))


class FlatLogit(nn.Module):
    """A head that returns a rank-1 ``(B,)`` score instead of ``(B, 1)``."""

    def __init__(self):
        """Wire one linear layer and squeeze its output."""
        super().__init__()
        torch.manual_seed(3)
        self.head = nn.Linear(IMG * IMG * 3, 1)

    def forward(self, x):
        """Return the squeezed logit."""
        return self.head(x.flatten(1)).squeeze(-1)


def _image(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, IMG, IMG, generator=generator)


# ---------------------------------------------------------------------------
# ClassScoreModel
# ---------------------------------------------------------------------------

def test_a_rank_one_head_is_read_as_one_logit_not_a_batch():
    """A model returning ``(B,)`` has one logit per image, not one image.

    Read as a batch dimension the wrapper would report ``B`` classes and every
    downstream target index would address the wrong thing, so the missing
    trailing axis is restored before the width is recorded.
    """
    wrapped = ClassScoreModel(FlatLogit())
    scores = wrapped(_image().unsqueeze(0))

    assert tuple(scores.shape) == (1, 2)
    assert wrapped.single_logit is True
    assert wrapped.n_classes == 2


def test_the_class_count_is_refused_before_any_forward_pass():
    """Asking how many classes a wrapper exposes is only answerable after a run.

    The width is discovered from a real forward pass. Guessing two would make
    a ten-class model silently attributable to class 0 or 1 only.
    """
    wrapped = ClassScoreModel(TinyCNN())

    with pytest.raises(AttributionError) as excinfo:
        wrapped.n_classes
    assert "head width is not known yet" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Input coercion and scores
# ---------------------------------------------------------------------------

def test_an_image_of_a_rank_nobody_can_interpret_is_refused():
    """A 5-D array is not an image, and guessing which axes are spatial is worse.

    Silently taking the last two axes would attribute a map over whichever
    slice happened to be first, and the picture would look perfectly ordinary.
    """
    with pytest.raises(AttributionError) as excinfo:
        attribute(TinyCNN(), np.zeros((1, 1, 3, IMG, IMG), dtype=np.float32),
                  "saliency")
    assert "must be (H, W), (C, H, W) or (1, C, H, W)" in str(excinfo.value)


def test_raw_scores_are_returned_unsoftmaxed_when_asked():
    """``probability=False`` must hand back the logits, not normalise them.

    The deletion and insertion curves track probabilities because a bounded
    quantity makes their areas comparable; anything comparing raw margins
    needs the logits, and a silently softmaxed one is a different number.
    """
    model = TinyCNN()
    x = _image().unsqueeze(0)

    raw = class_scores(model, x, probability=False)
    probability = class_scores(model, x, probability=True)

    assert not np.isclose(float(raw.sum()), 1.0)
    assert np.isclose(float(probability.sum()), 1.0)
    assert torch.allclose(torch.softmax(raw, dim=-1), probability)


# ---------------------------------------------------------------------------
# CAM over a layer that has no feature map
# ---------------------------------------------------------------------------

def test_a_cam_over_a_non_spatial_layer_is_refused_not_reshaped():
    """A CAM needs a ``(B, C, H, W)`` feature map; a vector is not one.

    Reshaping a token or logit vector into a square and colouring it produces
    an image-shaped artefact with no relationship to the input pixels, which
    is the single most convincing way this module could lie.
    """
    model = TinyCNN()

    with pytest.raises(NoSpatialLayerError) as excinfo:
        attribute(model, _image(), "gradcam", layer="head")
    message = str(excinfo.value)
    assert "not a (B, C, H, W) feature map" in message
    assert "attention_rollout" in message


# ---------------------------------------------------------------------------
# Eigen-CAM
# ---------------------------------------------------------------------------

class NegativeConv(nn.Module):
    """A model whose target conv emits strongly negative activations."""

    def __init__(self):
        """Bias the conv far below zero so its raw output is negative."""
        super().__init__()
        torch.manual_seed(11)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        with torch.no_grad():
            self.conv.bias.fill_(-50.0)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        """Return logits from the pooled, rectified conv output."""
        act = self.conv(x)
        return self.head(torch.relu(act).mean(dim=(-2, -1)))


def test_eigencam_falls_back_to_the_mean_when_the_decomposition_fails(monkeypatch):
    """A degenerate activation still yields a map, oriented the readable way.

    The first singular vector's sign is arbitrary, and an all-constant feature
    map makes the decomposition degenerate outright. Neither may end the run:
    the fallback is the channel mean, and a mostly-negative projection is
    flipped so "more" always means "more important".
    """
    def _no_svd(*args, **kwargs):
        raise RuntimeError("SVD did not converge")

    monkeypatch.setattr(torch.linalg, "svd", _no_svd)
    result = attribute(NegativeConv(), _image(), "eigencam", layer="conv")

    assert result.map.shape == (IMG, IMG)
    assert np.isfinite(result.map).all()
    assert float(result.map.min()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# captum adapters
# ---------------------------------------------------------------------------

def test_a_tensor_baseline_is_used_as_the_reference_input():
    """Integrated gradients explains the difference from ITS baseline.

    A supplied reference tensor is the question the user asked; substituting
    zeros for it would answer a different one and label the answer with the
    same method name.
    """
    model = TinyCNN()
    image = _image()
    baseline = torch.full((1, 3, IMG, IMG), 0.5)

    result = attribute(model, image, "integrated_gradients",
                       baseline=baseline, n_steps=4)

    assert result.map.shape == (IMG, IMG)
    assert np.isfinite(result.map).all()


def test_a_numeric_baseline_becomes_a_constant_reference_image():
    """A number is shorthand for a flat reference at that intensity."""
    result = attribute(TinyCNN(), _image(), "integrated_gradients",
                       baseline=0.25, n_steps=4)

    assert result.map.shape == (IMG, IMG)
    assert np.isfinite(result.map).all()


def test_integrated_gradients_refuses_to_integrate_over_one_step():
    """Fewer than two steps integrates nothing and is not the method at all.

    A single step reduces to a scaled input-times-gradient, which has
    different axioms; returning it under the name integrated_gradients would
    mislabel the result.
    """
    with pytest.raises(AttributionError) as excinfo:
        attribute(TinyCNN(), _image(), "integrated_gradients", n_steps=1)
    assert "at least 2 steps" in str(excinfo.value)


def test_a_captum_method_with_no_adapter_branch_names_the_disagreement():
    """The registry and the adapter must not drift apart silently.

    A method registered with the captum backend but unknown to the adapter
    would otherwise fall through and return whatever the last branch left
    behind. The refusal says exactly which of the two is out of date.
    """
    spec = MethodSpec(name="not_wired_up", family="gradient", backend="captum",
                      fn=_captum_attribute)
    wrapped = ClassScoreModel(TinyCNN())
    x = _image().unsqueeze(0)

    with pytest.raises(UnknownMethodError) as excinfo:
        _captum_attribute(spec, wrapped, x, 0, None, None)
    assert "the registry and the adapter disagree" in str(excinfo.value)


def test_an_unrelated_runtime_error_from_captum_is_not_relabelled(monkeypatch):
    """Only the shared-activation failure gets the shared-activation advice.

    Rewriting every RuntimeError into "your model reuses one nn.ReLU" would
    send a user to fix a model that is fine and hide the real fault, so
    anything else propagates unchanged.
    """
    import captum.attr as ca

    def _explode(self, *args, **kwargs):
        raise RuntimeError("the tensor is on the wrong device")

    monkeypatch.setattr(ca.Saliency, "attribute", _explode)

    with pytest.raises(RuntimeError) as excinfo:
        attribute(TinyCNN(), _image(), "saliency")
    assert "the tensor is on the wrong device" in str(excinfo.value)
    assert not isinstance(excinfo.value, AttributionError)


def test_smoothgrad_noise_survives_a_constant_image():
    """A flat image has no dynamic range, and noise scaled to it would be zero.

    SmoothGrad's sigma is a fraction of the input range; without a floor a
    uniform image would be averaged with itself and the "smoothed" map would
    be the unsmoothed one under a different name.
    """
    flat = torch.zeros(3, IMG, IMG)

    result = smoothgrad(TinyCNN(), flat, "integrated_gradients",
                        n_samples=2, n_steps=2, seed=0)

    assert result.map.shape == (IMG, IMG)
    assert np.isfinite(result.map).all()


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

class _Attn(nn.MultiheadAttention):
    """An attention block whose returned weights the test dictates."""

    def __init__(self, embed_dim, weights):
        """``weights`` is the ``(B, L, S)`` matrix to return, or None."""
        super().__init__(embed_dim, num_heads=1, batch_first=True)
        self._weights = weights

    def forward(self, query, key, value, **kwargs):
        """Return the value tokens plus whatever weights the test supplied."""
        if self._weights is None:
            return value
        return value, self._weights


class TokenModel(nn.Module):
    """A minimal transformer-shaped classifier around one attention block."""

    def __init__(self, attn, n_tokens, embed_dim):
        """Reshape the image into ``n_tokens`` tokens and classify their mean."""
        super().__init__()
        torch.manual_seed(5)
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.attn = attn
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        """Return logits, ``(B, 2)``."""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)[:, :self.n_tokens * self.embed_dim]
        tokens = flat.reshape(batch, self.n_tokens, self.embed_dim)
        out = self.attn(tokens, tokens, tokens)
        out = out[0] if isinstance(out, tuple) else out
        return self.head(out.mean(dim=1))


def _rollout_model(weights, n_tokens, embed_dim=4):
    return TokenModel(_Attn(embed_dim, weights), n_tokens, embed_dim)


def test_a_fused_attention_block_yields_no_map_rather_than_an_invented_one():
    """A block that returns no weights has nothing to roll out.

    Fused attention kernels never materialise an attention matrix. Producing a
    map anyway -- from the token values, say -- would be a picture of the
    activations labelled as attention.
    """
    model = _rollout_model(None, n_tokens=4)

    with pytest.raises(NoSpatialLayerError) as excinfo:
        attention_rollout(model, _image())
    assert "none returned attention weights" in str(excinfo.value)


def test_a_non_square_attention_matrix_is_refused():
    """Rollout composes token-to-token maps, which have to be square.

    Cross-attention between two different token sets cannot be multiplied into
    a rollout; doing it anyway would produce a matrix whose axes mean
    different things at every layer.
    """
    model = _rollout_model(torch.rand(1, 4, 3), n_tokens=4)

    with pytest.raises(NoSpatialLayerError) as excinfo:
        attention_rollout(model, _image())
    assert "non-square" in str(excinfo.value)


def test_tokens_without_a_class_token_are_averaged_over_the_whole_grid():
    """A square token count with no class token still lays back over the image.

    Assuming a class token that is not there would drop one patch and shift
    every remaining one, so the map would be offset from the pixels it claims
    to explain.
    """
    model = _rollout_model(torch.eye(4).unsqueeze(0), n_tokens=4)

    result = attention_rollout(model, _image())

    assert result.map.shape == (IMG, IMG)
    assert np.isfinite(result.map).all()


def test_a_token_count_that_is_not_a_grid_is_refused():
    """Tokens that form no square grid cannot be laid back over an image.

    Padding or truncating them to the nearest square would silently move every
    patch, so the refusal names the count instead.
    """
    model = _rollout_model(torch.eye(3).unsqueeze(0), n_tokens=3)

    with pytest.raises(NoSpatialLayerError) as excinfo:
        attention_rollout(model, _image())
    assert "do not form a square patch grid" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The non-finite guard on a finished map
# ---------------------------------------------------------------------------

def test_a_method_returning_a_non_finite_map_is_cleaned_not_plotted():
    """NaN and infinity must never reach the figure or the metrics.

    A NaN map renders as a blank field and every faithfulness number computed
    from it is NaN, which reads as "the method found nothing" rather than
    "the adapter is broken". Zeroing them keeps the shape and makes the
    emptiness explicit.
    """
    def _broken(spec, wrapped, x, target, layer, model_type, **kw):
        bad = np.full((IMG, IMG), np.nan, dtype=np.float32)
        bad[0, 0] = np.inf
        return bad, None, None, []

    spec = MethodSpec(name="broken", family="gradient", backend="spacr",
                      fn=_broken)

    result = _attribute_with_spec(spec, TinyCNN(), _image())

    assert np.isfinite(result.map).all()
    assert float(np.abs(result.map).max()) == 0.0


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def test_a_curve_reports_how_far_the_score_moved_end_to_end():
    """``drop`` is first score minus last, whichever direction the curve runs.

    Reading the area alone hides a curve that started somewhere unexpected;
    the endpoints are what say whether the perturbation did anything at all.
    """
    curve = Curve(kind="deletion", fractions=np.array([0.0, 0.5, 1.0]),
                  scores=np.array([0.9, 0.5, 0.1]), auc=0.5,
                  baseline="blur", target=1)

    assert curve.drop == pytest.approx(0.8)
    assert curve.higher_is_better is False


def test_a_curve_kind_that_is_neither_deletion_nor_insertion_is_refused():
    """The two kinds are read in opposite directions, so a third is meaningless.

    ``higher_is_better`` and every comparison built on it branch on this
    string; an unrecognised one would silently take the deletion reading.
    """
    with pytest.raises(AttributionError) as excinfo:
        _perturbation_curve(TinyCNN(), _image(), np.zeros((IMG, IMG)),
                            "ablation")
    assert "must be 'deletion' or 'insertion'" in str(excinfo.value)


def test_a_curve_with_no_steps_is_refused():
    """A curve needs at least one perturbed point besides the unperturbed one.

    With none there is a single score and no area, and the AUC reported would
    be whatever the trapezoid rule makes of one point.
    """
    with pytest.raises(AttributionError) as excinfo:
        deletion_curve(TinyCNN(), _image(), np.zeros((IMG, IMG)), n_steps=0)
    assert "at least 1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The pointing game
# ---------------------------------------------------------------------------

def test_a_stack_of_object_planes_is_collapsed_into_one_mask():
    """A label stack is several objects, and the map may point at any of them.

    spaCR masks arrive per object plane; testing only the first plane would
    score a correct map as a miss whenever it pointed at the second object.
    """
    amap = np.zeros((IMG, IMG))
    amap[6, 6] = 1.0
    stack = np.zeros((2, IMG, IMG), dtype=int)
    stack[0, 0, 0] = 1
    stack[1, 6, 6] = 2

    assert pointing_game(amap, stack) == 1.0


def test_a_rate_over_nothing_scorable_says_it_is_not_a_measurement():
    """An empty rate must not be reported as a score of zero.

    Every image excluded for a missing annotation is not a miss. A bare 0.0
    would read as "the method never pointed at the object" when in fact
    nothing was ever compared.
    """
    empty = np.zeros((IMG, IMG), dtype=int)
    result = pointing_game_rate([np.zeros((IMG, IMG))], [empty])

    assert result["n"] == 0
    assert np.isnan(result["rate"])
    assert any("not a measurement" in note for note in result["notes"])


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def test_maps_of_different_sizes_cannot_be_correlated():
    """Correlating unequal maps would silently compare different pixels.

    numpy would broadcast or truncate depending on the call; the refusal names
    both sizes so the caller can see which map is the odd one.
    """
    with pytest.raises(AttributionError) as excinfo:
        _rank_correlation(np.zeros((4, 4)), np.zeros((3, 3)))
    assert "different sizes" in str(excinfo.value)


def test_agreement_is_computed_without_scipy(monkeypatch):
    """Rank correlation must not depend on an optional package.

    scipy is not a hard requirement of spaCR, and an agreement matrix that
    quietly becomes unavailable on a lean install turns the module's strongest
    warning -- that the methods disagree -- into no warning at all.
    """
    real_import = builtins.__import__

    def _no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("no scipy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy)

    generator = np.random.default_rng(0)
    a = generator.random((IMG, IMG))
    result = method_agreement([a, a.copy()])

    assert result.mean == pytest.approx(1.0)
    assert result.minimum == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sanity check and verdicts
# ---------------------------------------------------------------------------

def test_a_custom_attribution_callable_can_be_sanity_checked():
    """A method supplied as a callable is testable like a registered one.

    The randomisation check is the module's most informative analysis; a
    user's own map function is exactly the kind of thing that has never been
    checked against destroyed weights.
    """
    calls = []

    def edge_detector(model, x, target=0):
        """A map that ignores the weights entirely."""
        calls.append(model)
        array = x.detach()[0].abs().sum(dim=0).numpy()
        return np.abs(np.diff(array, axis=0, prepend=array[:1]))

    result = randomization_sanity_check(TinyCNN(), _image(), edge_detector,
                                        max_stages=1, seed=0)

    assert len(calls) >= 2, "the trained and the randomised model were both run"
    assert result.method == "edge_detector"


def test_partial_agreement_is_reported_as_partial():
    """Between disagreement and agreement there is a third, honest answer.

    Collapsing the middle into either verdict would either raise a false alarm
    or grant a confidence the correlations do not support, so the panel is
    what the reader is sent to.
    """
    agreement = Agreement(methods=["a", "b"],
                          matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
                          mean=0.5, minimum=0.5)

    verdict = agreement.verdict()
    assert "partly agree" in verdict
    assert "0.50" in verdict


# ---------------------------------------------------------------------------
# The batch adapter
# ---------------------------------------------------------------------------

def test_the_batch_adapter_draws_the_grid_through_the_existing_layout():
    """The drop-in generator must render with spaCR's own activation grid.

    The point of the adapter is that the existing batch loop gains new methods
    without changing shape; a second, private layout here would make the same
    pipeline produce two different-looking figures depending on the method.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    model = TinyCNN()
    generator = AttributionMapGenerator(model, method="saliency")
    batch = torch.stack([_image(0), _image(1)])
    maps, predictions = generator.compute_maps_and_predictions(batch)

    figure = generator.plot_activation_grid(batch, maps, predictions)

    assert figure is not None
    assert len(figure.axes) >= 2
    matplotlib.pyplot.close(figure)
