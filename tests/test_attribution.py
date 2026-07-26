"""Attribution methods and the checks that say whether any of them mean anything.

Everything here runs on a hand-built CNN of a few conv layers, on the CPU, with
no download and no training. That is not a shortcut — it is what makes the
assertions possible. On a synthetic model we *know* which region the score
depends on, so "occlusion localises to the corner" and "the deletion curve
collapses" are checks with a right answer, which they never are on a real
classifier.

The suite pins the properties the module lives or dies by:

* every registered method returns a **finite map at the input's spatial shape**;
* a **single-logit head and a C-logit head both work** for every method, and the
  case where confusing one for the other inverts the answer is constructed and
  asserted against;
* SmoothGrad **reduces run-to-run variance**;
* deletion and insertion AUCs **respond to a known dependence** and go flat when
  the map is uninformative;
* the pointing game scores **1.0 inside and 0.0 outside** a known mask;
* the **model-randomisation sanity check flags a method that ignores the
  weights** — an edge detector is built exactly so it can be caught;
* a transformer gets **attention rollout or a named error**, never a silent
  empty CAM;
* a bad layer name **lists the layers that do exist**.
"""
from __future__ import annotations

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
    NOT_AN_EXPLANATION,
    NoSpatialLayerError,
    SanityCheck,
    UnknownMethodError,
    attention_rollout,
    attribute,
    class_scores,
    compare_methods,
    conv_layer_names,
    deletion_curve,
    faithfulness,
    insertion_curve,
    list_methods,
    method_agreement,
    methods_by_family,
    pointing_game,
    pointing_game_rate,
    randomization_sanity_check,
    recommended_layer,
    resolve_layer,
    smoothgrad,
)


IMG = 16


# ---------------------------------------------------------------------------
# Models — small enough that every test is milliseconds, real enough that the
# gradient, CAM and perturbation families all have something to attach to.
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """Three conv layers, global pool, linear head of ``n_out`` logits."""

    def __init__(self, n_out=1, in_ch=3):
        """Build the network with a fixed seed so every test sees one model."""
        super().__init__()
        torch.manual_seed(1234)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1), nn.ReLU(),
            nn.Conv2d(6, 8, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, n_out)

    def forward(self, x):
        """Return the raw logits, ``(B, n_out)``."""
        return self.head(self.pool(self.features(x)).flatten(1))


class CornerNet(nn.Module):
    """A model whose score depends only on the top-left ``size × size`` corner.

    Hand-wired rather than trained: the single conv passes channel 0 through,
    the forward pass masks everything outside the corner, and the head turns the
    corner's mean into a logit. Every faithfulness assertion in this file rests
    on that known dependence.
    """

    def __init__(self, n_out=1, size=6, gain=8.0):
        """Wire the weights so only the corner can move the logit."""
        super().__init__()
        self.size = int(size)
        self.conv = nn.Conv2d(3, 2, 3, padding=1)
        self.head = nn.Linear(2, n_out)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 1, 1] = 1.0
            self.conv.bias.zero_()
            self.head.weight.zero_()
            self.head.weight[-1, 0] = float(gain)
            self.head.bias.zero_()

    def forward(self, x):
        """Score the top-left corner only."""
        feat = self.conv(x)
        mask = torch.zeros_like(feat)
        mask[:, :, :self.size, :self.size] = 1.0
        return self.head((feat * mask).mean(dim=(2, 3)))


class TinyViT(nn.Module):
    """Patch-embed conv + two ``nn.MultiheadAttention`` blocks + a linear head."""

    def __init__(self, n_out=2, dim=8, patch=4, heads=2):
        """Build a transformer small enough to attribute in milliseconds."""
        super().__init__()
        torch.manual_seed(7)
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn1 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        """Patchify, attend twice with residuals, classify the class token."""
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        a, _ = self.attn1(t, t, t)
        t = t + a
        a, _ = self.attn2(t, t, t)
        t = t + a
        return self.head(t[:, 0])


class MLPNet(nn.Module):
    """No convolution and no attention — nothing for a CAM or rollout to hook."""

    def __init__(self, n_out=2):
        """Flatten and classify."""
        super().__init__()
        torch.manual_seed(3)
        self.fc = nn.Linear(3 * IMG * IMG, n_out)

    def forward(self, x):
        """Return the logits."""
        return self.fc(x.flatten(1))


@pytest.fixture
def image():
    """A fixed random 3-channel image."""
    torch.manual_seed(99)
    return torch.randn(3, IMG, IMG)


@pytest.fixture
def corner_image():
    """A blank image with a bright top-left corner, matching :class:`CornerNet`."""
    x = torch.zeros(3, IMG, IMG)
    x[0, :6, :6] = 2.0
    return x


def _fast_kwargs(name):
    """Keyword arguments that keep the slow methods fast on a 16x16 image."""
    if name == "integrated_gradients":
        return {"n_steps": 4}
    if name == "occlusion":
        return {"window": 6, "stride": 4}
    if name == "feature_ablation":
        return {"block": 8}
    if name == "scorecam":
        return {"batch_size": 4}
    return {}


#: Every method that works on a plain CNN — i.e. all but attention rollout.
CNN_METHODS = [n for n in sorted(ATTRIBUTION_METHODS)
               if n != "attention_rollout"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_every_family_the_item_asked_for_is_present(self):
        families = methods_by_family()
        assert set(families) == {"cam", "gradient", "perturbation", "attention"}
        assert set(families["cam"]) == {
            "gradcam", "gradcam_pp", "scorecam", "xgradcam", "layercam",
            "eigencam"}
        assert set(families["gradient"]) == {
            "saliency", "integrated_gradients", "guided_backprop",
            "input_x_gradient", "deeplift"}
        assert set(families["perturbation"]) == {"occlusion",
                                                 "feature_ablation"}
        assert families["attention"] == ["attention_rollout"]

    def test_the_libraries_do_the_maths_not_this_module(self):
        """Everything that torchcam or captum already implements is theirs.

        Only Eigen-CAM (absent from torchcam 0.4) and attention rollout are
        implemented here, and both say so.
        """
        by_backend = {}
        for name, spec in ATTRIBUTION_METHODS.items():
            by_backend.setdefault(spec.backend, set()).add(name)
        assert by_backend["torchcam"] == {"gradcam", "gradcam_pp", "scorecam",
                                          "xgradcam", "layercam"}
        assert by_backend["captum"] == {
            "saliency", "integrated_gradients", "guided_backprop",
            "input_x_gradient", "deeplift", "occlusion", "feature_ablation"}
        assert by_backend["spacr"] == {"eigencam", "attention_rollout"}

    def test_list_methods_can_be_filtered_by_family(self):
        assert "gradcam" in list_methods("cam")
        assert "gradcam" not in list_methods("gradient")
        assert list_methods() == sorted(ATTRIBUTION_METHODS)

    def test_an_unknown_method_names_the_registered_ones(self, image):
        with pytest.raises(UnknownMethodError) as excinfo:
            attribute(TinyCNN(), image, "grad_cam")
        message = str(excinfo.value)
        assert "grad_cam" in message
        assert "gradcam" in message and "occlusion" in message


# ---------------------------------------------------------------------------
# Every method, both head shapes
# ---------------------------------------------------------------------------

class TestEveryMethodOnBothHeads:
    @pytest.mark.parametrize("name", CNN_METHODS)
    @pytest.mark.parametrize("n_out", [1, 3])
    def test_finite_map_at_the_input_shape(self, name, n_out, image):
        model = TinyCNN(n_out=n_out)
        att = attribute(model, image, name, **_fast_kwargs(name))
        assert att.map.shape == (IMG, IMG)
        assert att.map.dtype == np.float32
        assert np.isfinite(att.map).all()
        assert att.method == name
        assert att.n_classes == (2 if n_out == 1 else n_out)
        assert att.single_logit is (n_out == 1)

    @pytest.mark.parametrize("name", CNN_METHODS)
    def test_every_class_of_a_multi_logit_head_is_addressable(self, name, image):
        model = TinyCNN(n_out=3)
        for target in (0, 1, 2):
            att = attribute(model, image, name, target=target,
                            **_fast_kwargs(name))
            assert att.target == target
            assert np.isfinite(att.map).all()

    @pytest.mark.parametrize("name", CNN_METHODS)
    def test_both_classes_of_a_binary_head_are_addressable(self, name, image):
        """A single logit is two classes, not one.

        Code that indexes a ``(B, 1)`` head by class would raise or silently
        return class 1's map for class 0; both classes must be reachable and the
        second must not be an index error.
        """
        model = TinyCNN(n_out=1)
        for target in (0, 1):
            att = attribute(model, image, name, target=target,
                            **_fast_kwargs(name))
            assert att.target == target
            assert att.n_classes == 2
            assert np.isfinite(att.map).all()

    def test_a_target_outside_the_head_is_refused_with_the_head_width(self,
                                                                     image):
        with pytest.raises(AttributionError) as excinfo:
            attribute(TinyCNN(n_out=1), image, "saliency", target=2)
        assert "binary head" in str(excinfo.value)
        with pytest.raises(AttributionError) as excinfo:
            attribute(TinyCNN(n_out=3), image, "saliency", target=5)
        assert "3 classes" in str(excinfo.value)

    def test_the_model_is_left_in_the_mode_it_arrived_in(self, image):
        model = TinyCNN(n_out=2)
        model.train()
        attribute(model, image, "saliency")
        assert model.training is True
        model.eval()
        attribute(model, image, "gradcam")
        assert model.training is False

    def test_a_batch_of_more_than_one_is_refused_not_silently_sliced(self):
        with pytest.raises(AttributionError) as excinfo:
            attribute(TinyCNN(), torch.randn(4, 3, IMG, IMG), "saliency")
        assert "one image at a time" in str(excinfo.value)

    def test_a_numpy_image_and_a_greyscale_image_both_work(self):
        model = TinyCNN(n_out=2, in_ch=1)
        att = attribute(model, np.zeros((IMG, IMG), dtype=np.float32),
                        "saliency")
        assert att.map.shape == (IMG, IMG)


# ---------------------------------------------------------------------------
# The head-shape bug this module exists to make impossible
# ---------------------------------------------------------------------------

class TestSingleLogitVersusMultiLogit:
    """The case where treating one head as the other inverts the answer.

    A single-logit head predicts class 0 when the logit is negative. Attributing
    the raw logit always explains *class 1*, so for such an image the map you get
    back is the map for the class the model rejected. On a CAM, whose ReLU keeps
    only the evidence pointing one way, that is not a subtle difference — the two
    maps are close to complements. These tests construct exactly that input and
    assert the inversion does not happen.
    """

    @staticmethod
    def _negative_logit_model_and_image():
        """A single-logit model that firmly predicts class 0 for this image."""
        model = CornerNet(n_out=1, gain=8.0)
        with torch.no_grad():                 # flip the sign: corner -> class 0
            model.head.weight[-1, 0] = -8.0
        x = torch.zeros(3, IMG, IMG)
        x[0, :6, :6] = 2.0
        return model, x

    def test_the_prediction_of_a_single_logit_head_is_read_correctly(self):
        model, x = self._negative_logit_model_and_image()
        wrapped = ClassScoreModel(model)
        raw = model(x[None])
        assert float(raw) < 0                  # negative logit -> class 0
        att = attribute(model, x, "gradcam")
        assert att.predicted == 0
        assert att.target == 0, (
            "target defaulted to the wrong class: a negative single logit is "
            "class 0, and reading (B, 1) as if it were (B, C) argmaxes to 0 of "
            "a one-wide tensor, which is class 1's score")
        probs = class_scores(wrapped, x[None])
        assert float(probs[0, 0]) > float(probs[0, 1])

    def test_the_default_target_map_is_class_zeros_not_class_ones(self):
        """The inversion, asserted directly.

        ``target=None`` must reproduce ``target=0`` (the prediction) and must
        NOT reproduce ``target=1``. Attributing the raw logit gives class 1's
        map, so a broken implementation makes the second comparison the equal
        one.
        """
        model, x = self._negative_logit_model_and_image()
        default = attribute(model, x, "gradcam").map
        cls0 = attribute(model, x, "gradcam", target=0).map
        cls1 = attribute(model, x, "gradcam", target=1).map
        assert np.allclose(default, cls0)
        assert not np.allclose(default, cls1)
        # And the two classes really do disagree, so the check above has teeth.
        assert float(np.abs(cls0 - cls1).max()) > 0

    def test_the_two_classes_of_a_binary_head_have_opposite_signed_maps(self):
        """``[-z, +z]`` means class 0's signed attribution negates class 1's."""
        model, x = self._negative_logit_model_and_image()
        a0 = attribute(model, x, "input_x_gradient", target=0)
        a1 = attribute(model, x, "input_x_gradient", target=1)
        assert a0.raw is not None and a1.raw is not None
        assert np.allclose(a0.raw, -a1.raw, atol=1e-5)

    def test_class_zero_is_not_an_all_zero_map(self):
        """The ``[0, z]`` view would give class 0 a zero gradient everywhere.

        That is the quiet failure this wrapper exists to avoid: a plausible,
        completely uninformative map.
        """
        model, x = self._negative_logit_model_and_image()
        att = attribute(model, x, "saliency", target=0)
        assert float(np.abs(att.map).max()) > 0
        assert not att.is_flat()

    def test_a_two_logit_head_is_not_pushed_through_the_binary_view(self, image):
        model = TinyCNN(n_out=2)
        att = attribute(model, image, "saliency")
        assert att.single_logit is False
        assert att.n_classes == 2
        raw = model(image[None])
        assert att.predicted == int(raw.argmax(dim=-1))
        assert att.target == att.predicted

    def test_class_scores_agree_with_the_head_for_both_shapes(self, image):
        for n_out in (1, 2, 4):
            model = TinyCNN(n_out=n_out)
            probs = class_scores(model, image[None])
            assert probs.shape == (1, 2 if n_out == 1 else n_out)
            assert pytest.approx(1.0, abs=1e-5) == float(probs.sum())


# ---------------------------------------------------------------------------
# SmoothGrad
# ---------------------------------------------------------------------------

class TestSmoothGrad:
    @staticmethod
    def _across_run_std(model, image, n_samples, method="saliency", runs=5):
        """Mean per-pixel standard deviation of the map over repeated runs.

        Each map is divided by its own sum rather than min-max stretched.
        Attribution scales are arbitrary and differ between methods, so the maps
        must be put on a common scale before their spread means anything — and
        min-max is the wrong way to do it here, because it *rescales* an
        averaged map back to full range and hides exactly the variance
        reduction being measured.
        """
        maps = []
        for seed in range(runs):
            att = smoothgrad(model, image, method, n_samples=n_samples,
                             sigma=0.4, target=1, seed=seed)
            m = np.asarray(att.map, dtype=np.float64)
            total = float(m.sum())
            maps.append(m / total if total > 0 else m)
        return float(np.mean(np.std(np.stack(maps, axis=0), axis=0)))

    def test_more_samples_reduce_run_to_run_variance(self, image):
        """SmoothGrad's whole claim: averaging suppresses the fluctuation.

        Measured the only way it can be — the same configuration run several
        times with different noise draws, and the spread between those runs.
        """
        model = TinyCNN(n_out=2)
        noisy = self._across_run_std(model, image, n_samples=1)
        smoothed = self._across_run_std(model, image, n_samples=24)
        assert smoothed < noisy, (noisy, smoothed)

    def test_it_works_for_a_cam_too_where_noise_tunnel_does_not_apply(self,
                                                                     image):
        model = TinyCNN(n_out=2)
        noisy = self._across_run_std(model, image, 1, method="gradcam", runs=4)
        smoothed = self._across_run_std(model, image, 20, method="gradcam",
                                        runs=4)
        assert smoothed < noisy, (noisy, smoothed)

    def test_the_same_seed_reproduces(self, image):
        model = TinyCNN(n_out=2)
        a = smoothgrad(model, image, "saliency", n_samples=6, seed=5, target=1)
        b = smoothgrad(model, image, "saliency", n_samples=6, seed=5, target=1)
        assert np.allclose(a.map, b.map)

    def test_it_handles_a_single_logit_head(self, image):
        att = smoothgrad(TinyCNN(n_out=1), image, "saliency", n_samples=4,
                         target=0, seed=0)
        assert att.single_logit is True
        assert att.target == 0
        assert np.isfinite(att.map).all()

    def test_zero_samples_is_refused(self, image):
        with pytest.raises(AttributionError) as excinfo:
            smoothgrad(TinyCNN(), image, "saliency", n_samples=0)
        assert "at least 1" in str(excinfo.value)

    def test_an_unknown_base_method_is_named(self, image):
        with pytest.raises(UnknownMethodError):
            smoothgrad(TinyCNN(), image, "nope", n_samples=2)

    def test_the_result_says_it_was_smoothed(self, image):
        att = smoothgrad(TinyCNN(n_out=2), image, "saliency", n_samples=3,
                         seed=0)
        assert att.params["smoothgrad"] is True
        assert any("SmoothGrad" in n for n in att.notes)


# ---------------------------------------------------------------------------
# Deletion / insertion
# ---------------------------------------------------------------------------

class TestDeletionAndInsertion:
    def test_a_map_that_finds_the_corner_beats_one_that_does_not(self,
                                                                corner_image):
        """The synthetic model's dependence is known, so the AUCs have a right
        answer: deleting the corner must destroy the score, and deleting
        anything else must not."""
        model = CornerNet(n_out=1)
        good = np.zeros((IMG, IMG), dtype=np.float32)
        good[:6, :6] = 1.0
        bad = np.zeros((IMG, IMG), dtype=np.float32)
        bad[10:, 10:] = 1.0

        good_del = deletion_curve(model, corner_image, good, target=1,
                                  n_steps=8, baseline="zero")
        bad_del = deletion_curve(model, corner_image, bad, target=1,
                                 n_steps=8, baseline="zero")
        assert good_del.auc < bad_del.auc
        assert good_del.scores[0] > good_del.scores[-1]

        good_ins = insertion_curve(model, corner_image, good, target=1,
                                   n_steps=8, baseline="zero")
        bad_ins = insertion_curve(model, corner_image, bad, target=1,
                                  n_steps=8, baseline="zero")
        assert good_ins.auc > bad_ins.auc

    def test_a_map_pointing_away_from_the_dependence_has_a_flat_start(
            self, corner_image):
        """The finding the item asks for: a flat deletion curve.

        Ranking only pixels the model ignores means the score does not move
        while they are removed.
        """
        model = CornerNet(n_out=1)
        bad = np.zeros((IMG, IMG), dtype=np.float32)
        bad[10:, 10:] = 1.0
        curve = deletion_curve(model, corner_image, bad, target=1, n_steps=8,
                               baseline="zero")
        # The first steps remove only ignored pixels, so the score is unchanged.
        assert curve.scores[0] == pytest.approx(curve.scores[1], abs=1e-6)

    def test_the_two_curves_are_bounded_probabilities(self, corner_image):
        model = CornerNet(n_out=1)
        amap = np.random.default_rng(0).random((IMG, IMG))
        for curve in (deletion_curve(model, corner_image, amap, target=1,
                                     n_steps=6),
                      insertion_curve(model, corner_image, amap, target=1,
                                      n_steps=6)):
            assert isinstance(curve, Curve)
            assert 0.0 <= curve.auc <= 1.0
            assert ((curve.scores >= 0) & (curve.scores <= 1)).all()
            assert curve.fractions[0] == 0.0
            assert curve.fractions[-1] == pytest.approx(1.0)

    def test_deletion_and_insertion_point_in_opposite_directions(self):
        assert deletion_curve.__doc__ and insertion_curve.__doc__
        model = CornerNet(n_out=1)
        x = torch.zeros(3, IMG, IMG)
        x[0, :6, :6] = 2.0
        amap = np.zeros((IMG, IMG), dtype=np.float32)
        amap[:6, :6] = 1.0
        assert deletion_curve(model, x, amap, target=1, n_steps=6,
                              baseline="zero").higher_is_better is False
        assert insertion_curve(model, x, amap, target=1, n_steps=6,
                               baseline="zero").higher_is_better is True

    def test_an_attribution_object_is_accepted_directly(self, corner_image):
        model = CornerNet(n_out=1)
        att = attribute(model, corner_image, "occlusion", target=1, window=6,
                        stride=4)
        curve = deletion_curve(model, corner_image, att, n_steps=6,
                               baseline="zero")
        assert curve.target == 1

    def test_a_map_of_the_wrong_shape_is_refused(self, image):
        with pytest.raises(AttributionError) as excinfo:
            deletion_curve(TinyCNN(), image, np.zeros((8, 8)), n_steps=4)
        assert "must line up" in str(excinfo.value)

    def test_an_unknown_baseline_is_refused_with_the_options(self, image):
        with pytest.raises(AttributionError) as excinfo:
            deletion_curve(TinyCNN(), image, np.zeros((IMG, IMG)), n_steps=4,
                           baseline="grey")
        assert "'zero'" in str(excinfo.value) and "'blur'" in str(excinfo.value)

    @pytest.mark.parametrize("baseline", ["zero", "mean", "blur", "uniform"])
    def test_every_baseline_runs(self, corner_image, baseline):
        curve = deletion_curve(CornerNet(n_out=1), corner_image,
                               np.random.default_rng(1).random((IMG, IMG)),
                               target=1, n_steps=4, baseline=baseline)
        assert np.isfinite(curve.auc)

    def test_faithfulness_reports_everything_with_its_caveats(self,
                                                             corner_image):
        model = CornerNet(n_out=1)
        amap = np.zeros((IMG, IMG), dtype=np.float32)
        amap[:6, :6] = 1.0
        mask = np.zeros((IMG, IMG), dtype=bool)
        mask[:6, :6] = True
        out = faithfulness(model, corner_image, amap, target=1, n_steps=6,
                           baseline="zero", mask=mask)
        assert out["deletion_auc"] == out["deletion"].auc
        assert out["insertion_auc"] == out["insertion"].auc
        assert out["pointing_game"] == 1.0
        assert NOT_AN_EXPLANATION in out["notes"]

    def test_a_flat_map_is_labelled_as_meaningless(self, corner_image):
        out = faithfulness(CornerNet(n_out=1), corner_image,
                           np.ones((IMG, IMG)), target=1, n_steps=4,
                           baseline="zero")
        assert out["flat"] is True
        assert any("flat" in n for n in out["notes"])


# ---------------------------------------------------------------------------
# Pointing game
# ---------------------------------------------------------------------------

class TestPointingGame:
    def test_one_inside_and_zero_outside(self):
        amap = np.zeros((IMG, IMG), dtype=np.float32)
        amap[3, 4] = 1.0
        inside = np.zeros((IMG, IMG), dtype=bool)
        inside[2:6, 3:7] = True
        assert pointing_game(amap, inside) == 1.0
        assert pointing_game(amap, ~inside) == 0.0

    def test_a_spacr_integer_label_plane_is_accepted_as_is(self):
        amap = np.zeros((IMG, IMG), dtype=np.float32)
        amap[9, 9] = 5.0
        labels = np.zeros((IMG, IMG), dtype=np.int32)
        labels[8:12, 8:12] = 7          # object id 7, as merged/*.npy stores it
        assert pointing_game(amap, labels) == 1.0

    def test_tolerance_grows_the_mask(self):
        amap = np.zeros((IMG, IMG), dtype=np.float32)
        amap[5, 5] = 1.0
        mask = np.zeros((IMG, IMG), dtype=bool)
        mask[6, 6] = True
        assert pointing_game(amap, mask) == 0.0
        assert pointing_game(amap, mask, tolerance=1) == 1.0

    def test_an_empty_mask_is_an_error_not_a_miss(self):
        with pytest.raises(AttributionError) as excinfo:
            pointing_game(np.ones((IMG, IMG)), np.zeros((IMG, IMG), dtype=bool))
        assert "annotation is missing" in str(excinfo.value)

    def test_a_shape_mismatch_is_refused(self):
        with pytest.raises(AttributionError):
            pointing_game(np.ones((IMG, IMG)), np.ones((8, 8), dtype=bool))

    def test_the_rate_excludes_unusable_masks_rather_than_scoring_them(self):
        good = np.zeros((IMG, IMG), dtype=np.float32)
        good[1, 1] = 1.0
        mask = np.zeros((IMG, IMG), dtype=bool)
        mask[0:3, 0:3] = True
        out = pointing_game_rate([good, good, good],
                                 [mask, ~mask, np.zeros_like(mask)])
        assert out["n"] == 2 and out["hits"] == 1
        assert out["rate"] == 0.5
        assert len(out["skipped"]) == 1

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(AttributionError):
            pointing_game_rate([np.ones((IMG, IMG))], [])

    def test_a_real_attribution_of_the_corner_model_points_at_the_corner(
            self, corner_image):
        model = CornerNet(n_out=1)
        att = attribute(model, corner_image, "occlusion", target=1, window=6,
                        stride=2)
        mask = np.zeros((IMG, IMG), dtype=bool)
        mask[:6, :6] = True
        assert pointing_game(att, mask) == 1.0


# ---------------------------------------------------------------------------
# The model-randomisation sanity check — the point of the module
# ---------------------------------------------------------------------------

class TestRandomizationSanityCheck:
    def test_a_method_that_ignores_the_weights_is_flagged(self, image):
        """Adebayo et al.'s test, on a method built to fail it.

        ``_edge_detector`` never touches the model. Randomising every weight
        cannot change its output, the correlation is 1, and the check must say
        so — anything else and the check would pass a pure image filter as an
        explanation of a classifier.
        """
        def _edge_detector(model, img, target=0):
            """A Sobel-ish magnitude of the image; the model is never used."""
            arr = np.asarray(img.detach() if hasattr(img, "detach") else img,
                             dtype=np.float64).reshape(-1, IMG, IMG).sum(axis=0)
            dy, dx = np.gradient(arr)
            return np.abs(dy) + np.abs(dx)

        check = randomization_sanity_check(
            TinyCNN(n_out=2), image, "edge_detector",
            attribute_fn=_edge_detector, seed=0)
        assert isinstance(check, SanityCheck)
        assert check.final_similarity == pytest.approx(1.0, abs=1e-9)
        assert check.passed is False
        assert "FAILS" in check.verdict()
        assert check.gap == pytest.approx(0.0, abs=1e-9)

    def test_a_method_that_reads_the_weights_can_pass(self, image):
        """The check must be able to say yes, or flagging means nothing.

        This method returns a map built from the model's own first-layer
        weights, so randomising them changes it completely.
        """
        def _weight_reader(model, img, target=0):
            """A map made only of the model's weights — maximally sensitive."""
            weights = next(model.parameters()).detach().flatten()
            tile = weights.repeat((IMG * IMG // weights.numel()) + 1)
            return tile[:IMG * IMG].reshape(IMG, IMG).numpy().astype(float)

        check = randomization_sanity_check(
            TinyCNN(n_out=2), image, "weight_reader",
            attribute_fn=_weight_reader, seed=0)
        assert check.passed is True
        assert "PASSES" in check.verdict()
        assert check.gap > 0.5

    def test_every_stage_is_reported_in_the_order_applied(self, image):
        check = randomization_sanity_check(TinyCNN(n_out=2), image, "gradcam",
                                           seed=0)
        assert check.stages
        names = [n for n, _s in check.stages]
        assert "head" in names[0] or names[0].startswith("features")
        assert all(isinstance(s, float) for _n, s in check.stages)
        assert check.mode == "cascading"

    def test_independent_mode_randomises_one_layer_at_a_time(self, image):
        check = randomization_sanity_check(TinyCNN(n_out=2), image, "saliency",
                                           mode="independent", seed=0,
                                           max_stages=2)
        assert check.mode == "independent"
        assert len(check.stages) == 3          # 2 capped stages + <all layers>
        assert check.stages[-1][0] == "<all layers>"

    def test_the_verdict_always_comes_from_a_fully_randomised_model(self,
                                                                   image):
        """max_stages caps what is *reported*, never what is *judged*."""
        capped = randomization_sanity_check(TinyCNN(n_out=2), image, "saliency",
                                            seed=0, max_stages=1)
        full = randomization_sanity_check(TinyCNN(n_out=2), image, "saliency",
                                          seed=0)
        assert len(capped.stages) < len(full.stages)
        assert capped.final_similarity == pytest.approx(full.final_similarity,
                                                        abs=1e-9)

    def test_the_original_model_is_never_modified(self, image):
        model = TinyCNN(n_out=2)
        before = [p.detach().clone() for p in model.parameters()]
        randomization_sanity_check(model, image, "saliency", seed=0)
        after = list(model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_a_flat_randomised_map_is_inconclusive_not_a_pass(self, image):
        """A method that returns a constant map for the randomised model has no
        rank correlation. Calling that a pass would reward degenerating."""
        def _flat_after_randomisation(model, img, target=0):
            """Real map for the trained model, constant for a randomised one."""
            weight = float(next(model.parameters()).detach().flatten()[0])
            if abs(weight - _flat_after_randomisation.first) > 1e-9:
                return np.ones((IMG, IMG))
            rng = np.random.default_rng(0)
            return rng.random((IMG, IMG))

        model = TinyCNN(n_out=2)
        _flat_after_randomisation.first = float(
            next(model.parameters()).detach().flatten()[0])
        check = randomization_sanity_check(
            model, image, "degenerate", attribute_fn=_flat_after_randomisation,
            seed=0)
        assert not np.isfinite(check.final_similarity)
        assert check.passed is False
        assert any("inconclusive" in n for n in check.notes)

    def test_an_unknown_mode_is_refused(self, image):
        with pytest.raises(AttributionError):
            randomization_sanity_check(TinyCNN(), image, "saliency",
                                       mode="sideways")

    def test_a_model_with_no_parameters_is_refused(self, image):
        class NoParams(nn.Module):
            """A model with no learnable parameters at all."""

            def forward(self, x):
                """Return a fixed-width score derived from the input."""
                return torch.stack([x.mean(dim=(1, 2, 3)),
                                    -x.mean(dim=(1, 2, 3))], dim=-1)

        with pytest.raises(AttributionError) as excinfo:
            randomization_sanity_check(NoParams(), image, "saliency")
        assert "nothing to randomise" in str(excinfo.value)

    @pytest.mark.parametrize("name", ["saliency", "gradcam", "guided_backprop",
                                      "input_x_gradient"])
    def test_the_check_runs_for_the_real_methods_and_reports_a_number(
            self, name, image):
        check = randomization_sanity_check(TinyCNN(n_out=2), image, name,
                                           seed=0, max_stages=2)
        assert isinstance(check.passed, bool)
        assert isinstance(check.verdict(), str)
        assert 0.0 <= check.gap <= 1.0


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

class TestAgreement:
    def test_identical_maps_agree_and_reversed_rankings_do_not(self):
        """Agreement is a rank correlation of *magnitudes*.

        Negating a map is therefore not disagreement — the same pixels are still
        ranked highest, which is what a reader of the map acts on. Reversing the
        ranking is, so that is what the disagreement case does.
        """
        rng = np.random.default_rng(4)
        a = rng.random((IMG, IMG))
        agree = method_agreement([a, a.copy()])
        assert agree.mean == pytest.approx(1.0, abs=1e-9)
        assert method_agreement([a, -a]).mean == pytest.approx(1.0, abs=1e-9)
        disagree = method_agreement([a, a.max() - a])
        assert disagree.minimum < 0
        assert "DISAGREE" in disagree.verdict()

    def test_the_matrix_is_symmetric_with_a_unit_diagonal(self):
        rng = np.random.default_rng(5)
        maps = [rng.random((IMG, IMG)) for _ in range(3)]
        agree = method_agreement(maps)
        assert agree.matrix.shape == (3, 3)
        assert np.allclose(agree.matrix, agree.matrix.T)
        assert np.allclose(np.diag(agree.matrix), 1.0)

    def test_a_constant_map_is_excluded_not_scored_as_disagreement(self):
        rng = np.random.default_rng(6)
        agree = method_agreement([rng.random((IMG, IMG)), np.ones((IMG, IMG))])
        assert not np.isfinite(agree.mean)
        assert any("constant" in n for n in agree.notes)

    def test_fewer_than_two_maps_is_refused(self):
        with pytest.raises(AttributionError):
            method_agreement([np.ones((IMG, IMG))])

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(AttributionError):
            method_agreement([np.ones((IMG, IMG)), np.ones((8, 8))])

    def test_attribution_objects_keep_their_method_names(self, image):
        atts = compare_methods(TinyCNN(n_out=2), image,
                               ["saliency", "input_x_gradient"])
        agree = method_agreement(atts)
        assert agree.methods == ["saliency", "input_x_gradient"]
        assert isinstance(agree, Agreement)
        assert NOT_AN_EXPLANATION in agree.notes


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------

class TestCompareMethods:
    def test_every_method_explains_the_same_class(self, image):
        model = TinyCNN(n_out=3)
        atts = compare_methods(model, image, ["gradcam", "saliency",
                                              "occlusion"], target=2,
                               window=6, stride=4)
        assert [a.target for a in atts] == [2, 2, 2]
        assert [a.method for a in atts] == ["gradcam", "saliency", "occlusion"]

    def test_a_method_that_cannot_run_is_recorded_not_raised(self, image):
        atts = compare_methods(MLPNet(), image, ["saliency", "gradcam"])
        assert atts[0].map.any()
        assert any("FAILED" in n for n in atts[1].notes)
        assert not atts[1].map.any()
        assert any("placeholder" in n for n in atts[1].notes)

    def test_skip_failures_off_re_raises(self, image):
        with pytest.raises(NoSpatialLayerError):
            compare_methods(MLPNet(), image, ["gradcam"], skip_failures=False)


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

class TestTransformers:
    def test_rollout_produces_a_map_at_the_input_shape(self, image):
        att = attribute(TinyViT(), image, "attention_rollout")
        assert att.map.shape == (IMG, IMG)
        assert np.isfinite(att.map).all()
        assert att.family == "attention"
        assert any("not class-conditional" in n for n in att.notes)

    def test_rollout_works_for_a_single_logit_transformer_head(self, image):
        att = attention_rollout(TinyViT(n_out=1), image, target=0)
        assert att.n_classes == 2
        assert att.single_logit is True
        assert np.isfinite(att.map).all()

    def test_a_cam_on_a_patch_embedding_is_refused_by_name(self, image):
        """The pure-ViT trap: there *is* a Conv2d, and a CAM over it is junk.

        Grad-CAM on the patch embedding renders happily — a picture of local
        image statistics with no class information in it. Silence here is the
        failure mode the item forbids.
        """
        with pytest.raises(NoSpatialLayerError) as excinfo:
            attribute(TinyViT(), image, "gradcam", model_type="vit_tiny")
        message = str(excinfo.value)
        assert "vit_tiny" in message
        assert "patch embedding" in message
        assert "attention_rollout" in message

    def test_the_patch_embedding_cam_can_be_forced_explicitly(self, image):
        att = attribute(TinyViT(), image, "gradcam", allow_pre_attention=True)
        assert att.map.shape == (IMG, IMG)

    def test_gradient_methods_work_on_a_transformer(self, image):
        for name in ("saliency", "integrated_gradients", "occlusion"):
            att = attribute(TinyViT(), image, name, **_fast_kwargs(name))
            assert np.isfinite(att.map).all()
            assert att.map.shape == (IMG, IMG)

    def test_a_conv_free_model_names_itself_in_the_cam_error(self, image):
        with pytest.raises(NoSpatialLayerError) as excinfo:
            attribute(MLPNet(), image, "gradcam", model_type="linear_probe")
        message = str(excinfo.value)
        assert "linear_probe" in message
        assert "no Conv2d layer" in message
        assert "attention_rollout" in message

    def test_rollout_on_a_model_with_no_attention_names_the_alternative(self,
                                                                       image):
        with pytest.raises(NoSpatialLayerError) as excinfo:
            attribute(TinyCNN(n_out=2), image, "attention_rollout",
                      model_type="tiny_cnn")
        message = str(excinfo.value)
        assert "tiny_cnn" in message
        assert "convolutional layers" in message

    def test_rollout_on_a_model_with_neither_says_so(self, image):
        with pytest.raises(NoSpatialLayerError) as excinfo:
            attribute(MLPNet(), image, "attention_rollout")
        message = str(excinfo.value)
        assert "no Conv2d layer either" in message
        assert "no attention weights" in message

    def test_an_unknown_head_fusion_is_refused(self, image):
        with pytest.raises(AttributionError):
            attention_rollout(TinyViT(), image, head_fusion="median")

    def test_discard_ratio_runs(self, image):
        att = attention_rollout(TinyViT(), image, discard_ratio=0.5)
        assert np.isfinite(att.map).all()


# ---------------------------------------------------------------------------
# Occlusion localisation, layers, and the CAM contract
# ---------------------------------------------------------------------------

class TestOcclusionAndLayers:
    def test_occlusion_localises_to_the_corner_the_model_depends_on(
            self, corner_image):
        model = CornerNet(n_out=1)
        att = attribute(model, corner_image, "occlusion", target=1, window=4,
                        stride=2)
        assert att.peak() == (0, 0)
        corner = att.map[:6, :6].max()
        elsewhere = att.map[8:, 8:].max()
        assert corner > elsewhere * 2, (corner, elsewhere)

    def test_feature_ablation_localises_too(self, corner_image):
        att = attribute(CornerNet(n_out=1), corner_image, "feature_ablation",
                        target=1, block=4)
        row, col = att.peak()
        assert row < 6 and col < 6

    def test_a_bad_layer_name_lists_the_available_layers(self, image):
        with pytest.raises(AttributionError) as excinfo:
            attribute(TinyCNN(n_out=2), image, "gradcam",
                      layer="features.99.conv")
        message = str(excinfo.value)
        assert "features.99.conv" in message
        assert "features.4" in message           # the real last conv
        assert "usual CAM target" in message

    def test_resolve_layer_and_the_helpers_agree(self):
        model = TinyCNN(n_out=2)
        names = conv_layer_names(model)
        assert names == ["features.0", "features.2", "features.4"]
        assert recommended_layer(model) == "features.4"
        assert resolve_layer(model, "features.4") is model.features[4]
        assert recommended_layer(MLPNet()) is None

    def test_a_named_earlier_layer_changes_the_cam(self, image):
        model = TinyCNN(n_out=2)
        late = attribute(model, image, "gradcam", layer="features.4").map
        early = attribute(model, image, "gradcam", layer="features.0").map
        assert not np.allclose(late, early)

    def test_eigencam_is_class_agnostic_and_says_so(self, image):
        model = TinyCNN(n_out=3)
        a = attribute(model, image, "eigencam", target=0)
        b = attribute(model, image, "eigencam", target=2)
        assert np.allclose(a.map, b.map)
        assert any("ignores the class" in n for n in a.notes)

    def test_deeplift_on_a_reused_relu_names_the_fix(self, image):
        """spaCR's most common backbone hits this, and captum's message does not
        tell the user what to do about it.

        torchvision's ResNets build one ``nn.ReLU(inplace=True)`` and call it at
        several points. DeepLIFT needs a distinct module per activation, so it
        dies with a message about missing module attributes; this replaces it
        with one that names the working alternatives.
        """
        class ReusedRelu(nn.Module):
            """One ReLU instance, called three times — as a ResNet does."""

            def __init__(self):
                """Build two convs sharing a single activation module."""
                super().__init__()
                self.relu = nn.ReLU()
                self.a = nn.Conv2d(3, 4, 3, padding=1)
                self.b = nn.Conv2d(4, 4, 3, padding=1)
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                """Call ``self.relu`` more than once on purpose."""
                h = self.relu(self.a(x))
                h = self.relu(self.b(h))
                return self.head(self.relu(h).mean(dim=(2, 3)))

        with pytest.raises(AttributionError) as excinfo:
            attribute(ReusedRelu(), image, "deeplift")
        message = str(excinfo.value)
        assert "reuses one activation module" in message
        assert "integrated_gradients" in message
        # And the alternatives it names really do work on the same model.
        for name in ("integrated_gradients", "input_x_gradient", "occlusion"):
            att = attribute(ReusedRelu(), image, name, **_fast_kwargs(name))
            assert np.isfinite(att.map).all()

    def test_a_layer_that_never_runs_is_reported(self, image):
        class Detached(nn.Module):
            """Owns a conv that the forward pass never calls."""

            def __init__(self):
                """Build a used path and an unused conv."""
                super().__init__()
                self.used = nn.Conv2d(3, 4, 3, padding=1)
                self.unused = nn.Conv2d(4, 4, 3, padding=1)
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                """Classify without ever touching ``unused``."""
                return self.head(self.used(x).mean(dim=(2, 3)))

        with pytest.raises(AttributionError) as excinfo:
            attribute(Detached(), image, "gradcam", layer="unused")
        assert "never ran" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Attribution result object
# ---------------------------------------------------------------------------

class TestAttributionObject:
    def test_normalized_is_bounded_and_survives_a_flat_map(self, image):
        att = attribute(TinyCNN(n_out=2), image, "saliency")
        norm = att.normalized()
        assert norm.min() >= 0.0 and norm.max() <= 1.0
        flat = Attribution(method="x", map=np.full((4, 4), 3.0), target=0,
                           n_classes=2, single_logit=False, predicted=0)
        assert flat.is_flat() is True
        assert np.array_equal(flat.normalized(), np.zeros((4, 4)))

    def test_peak_and_shape(self):
        amap = np.zeros((4, 5), dtype=np.float32)
        amap[2, 3] = 9.0
        att = Attribution(method="x", map=amap, target=0, n_classes=2,
                          single_logit=False, predicted=0)
        assert att.peak() == (2, 3)
        assert att.shape == (4, 5)

    def test_the_single_logit_note_is_always_attached(self, image):
        att = attribute(TinyCNN(n_out=1), image, "saliency")
        assert any("single-logit binary head" in n for n in att.notes)
        att2 = attribute(TinyCNN(n_out=2), image, "saliency")
        assert not any("single-logit binary head" in n for n in att2.notes)

    def test_the_signed_per_channel_form_is_kept_where_it_exists(self, image):
        att = attribute(TinyCNN(n_out=2), image, "input_x_gradient")
        assert att.raw is not None and att.raw.shape == (3, IMG, IMG)
        cam = attribute(TinyCNN(n_out=2), image, "gradcam")
        assert cam.raw is None
        assert cam.layer == "features.4"


# ---------------------------------------------------------------------------
# The batch adapter the activation pipeline uses
# ---------------------------------------------------------------------------

class TestAttributionMapGenerator:
    def test_it_returns_maps_and_predictions_for_a_batch(self):
        torch.manual_seed(2)
        batch = torch.randn(3, 3, IMG, IMG)
        gen = AttributionMapGenerator(TinyCNN(n_out=2), method="gradcam")
        maps, preds = gen.compute_maps_and_predictions(batch)
        assert tuple(maps.shape) == (3, IMG, IMG)
        assert tuple(preds.shape) == (3,)
        assert preds.dtype == torch.long
        assert torch.isfinite(maps).all()

    def test_predictions_are_right_for_a_single_logit_head(self):
        model = CornerNet(n_out=1)
        batch = torch.zeros(2, 3, IMG, IMG)
        batch[0, 0, :6, :6] = 2.0          # positive logit -> class 1
        batch[1, 0, :6, :6] = -2.0         # negative logit -> class 0
        maps, preds = AttributionMapGenerator(
            model, method="saliency").compute_maps_and_predictions(batch)
        assert preds.tolist() == [1, 0]

    def test_smoothgrad_is_applied_when_asked(self):
        torch.manual_seed(3)
        batch = torch.randn(2, 3, IMG, IMG)
        gen = AttributionMapGenerator(TinyCNN(n_out=2), method="saliency",
                                      smoothgrad_samples=4)
        maps, _preds = gen.compute_maps_and_predictions(batch)
        assert torch.isfinite(maps).all()

    def test_an_unknown_method_is_refused_up_front(self):
        with pytest.raises(UnknownMethodError):
            AttributionMapGenerator(TinyCNN(), method="grod_cam")

    def test_the_legacy_aliases_point_at_the_same_call(self):
        gen = AttributionMapGenerator(TinyCNN(n_out=2), method="saliency")
        assert gen.compute_gradcam_and_predictions == \
            gen.compute_maps_and_predictions
        assert gen.compute_saliency_and_predictions == \
            gen.compute_maps_and_predictions
