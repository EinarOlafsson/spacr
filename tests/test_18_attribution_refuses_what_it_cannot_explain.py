"""Item 288: the item-18 attribution methods on the models they must refuse.

``test_18_modern_attribution_methods.py`` pins that HiRes-CAM, Ablation-CAM,
GradientSHAP / DeepSHAP and Chefer relevance find a planted signal on a
well-behaved model. This file pins the other side of each method: the
architectures and wirings where the method has no honest answer, and the
edges where it does but reaches it another way.

* HiRes-CAM on a layer that is off the gradient path (a frozen feature
  extractor) is refused, not drawn as a zero map.
* Ablation-CAM on a block that returns ``(feature_map, extra)`` ablates the
  feature map and still finds the signal.
* SHAP's reference distribution is never a single image, however the
  baselines are spelled.
* Chefer relevance refuses cross-attention, attention a caller asked not to
  see, and a token count that is not a patch grid; it reads a model with no
  class token by averaging, and a block that already averaged its heads as
  one head.
* ``architecture_kind`` and ``method_applicability`` classify every kind of
  backbone they name, from a loaded model or from a name alone.

Every model is tiny, hand-wired and on the CPU, so the right answer is known.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from spacr import attribution as att  # noqa: E402

IMG = 16
CORNER = 6


@pytest.fixture
def image():
    """Signal in channel 0's top-left corner; noise in channels 1-2."""
    rng = np.random.default_rng(0)
    img = np.zeros((3, IMG, IMG), dtype="float32")
    img[0, :CORNER, :CORNER] = 1.0
    img[1:] = rng.random((2, IMG, IMG)).astype("float32")
    return img


def _corner_conv():
    conv = nn.Conv2d(3, 4, 3, padding=1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[0, 0, 1, 1] = 1.0
        conv.weight[1, 1, 1, 1] = 1.0
    return conv


def _corner_head():
    head = nn.Linear(4, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1, 0] = 8.0
        head.weight[0, 0] = -8.0
    return head


def _corner_mask():
    mask = torch.zeros(1, 1, IMG, IMG)
    mask[..., :CORNER, :CORNER] = 1.0
    return mask


class FrozenFeatures(nn.Module):
    """A pretrained extractor run without gradient, and a trainable head."""

    def __init__(self):
        super().__init__()
        self.conv = _corner_conv()
        self.head = _corner_head()

    def forward(self, x):
        with torch.no_grad():
            feat = torch.relu(self.conv(x))
        return self.head(feat.mean(dim=(2, 3)))


class TupleBlock(nn.Module):
    """A feature block that also hands back an auxiliary tensor."""

    def __init__(self):
        super().__init__()
        self.conv = _corner_conv()

    def forward(self, x):
        feat = self.conv(x)
        return feat, feat.mean()


class TupleCNN(nn.Module):
    """CornerCNN with its convolution inside a tuple-returning block."""

    def __init__(self):
        super().__init__()
        self.block = TupleBlock()
        self.head = _corner_head()
        self.register_buffer("mask", _corner_mask())

    def forward(self, x):
        feat, _aux = self.block(x)
        feat = torch.relu(feat) * self.mask
        return self.head(feat.mean(dim=(2, 3)))


class CornerCNN(nn.Module):
    """Score depends only on channel 0 inside the top-left corner."""

    def __init__(self):
        super().__init__()
        self.conv = _corner_conv()
        self.head = _corner_head()
        self.register_buffer("mask", _corner_mask())

    def forward(self, x):
        feat = torch.relu(self.conv(x)) * self.mask
        return self.head(feat.mean(dim=(2, 3)))


class SingleHeadAttention(nn.MultiheadAttention):
    """A one-head block written out by hand, returning ``(B, L, S)`` weights.

    The weights it returns are the tensor its output is computed from, so
    they are on the gradient path -- unlike torch's head-averaged copy.
    """

    def __init__(self, dim):
        super().__init__(dim, 1, batch_first=True)

    def forward(self, query, key, value, need_weights=True,
                average_attn_weights=True, **_kw):
        dim = self.embed_dim
        w, b = self.in_proj_weight, self.in_proj_bias
        q = query @ w[:dim].T + b[:dim]
        k = key @ w[dim:2 * dim].T + b[dim:2 * dim]
        v = value @ w[2 * dim:].T + b[2 * dim:]
        weights = torch.softmax(q @ k.transpose(1, 2) / dim ** 0.5, dim=-1)
        return self.out_proj(weights @ v), weights


class TinyViT(nn.Module):
    """One attention block over 4x4 patches, hand-wired onto the corner patch.

    :param cls_token: prepend a class token and read the head from it; without
        one the head reads the mean over the patch tokens.
    :param wiring: ``"self"`` (ordinary self-attention), ``"cross"`` (only the
        class token queries the patches), ``"no_weights"`` (the caller passes
        ``need_weights=False`` positionally, so no weights come back) or
        ``"averaged"`` (the caller asks for head-averaged weights) or
        ``"single_head"`` (a hand-rolled block returning 3-D weights).
    """

    def __init__(self, cls_token=True, wiring="self", dim=4, patch=4):
        super().__init__()
        self.cls_token = cls_token
        self.wiring = wiring
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = (SingleHeadAttention(dim) if wiring == "single_head"
                     else nn.MultiheadAttention(dim, 2, batch_first=True))
        self.head = nn.Linear(dim, 2)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.bias.zero_()
            self.embed.weight[0, 0] = 1.0 / (patch * patch)
            self.attn.in_proj_weight.zero_()
            self.attn.in_proj_bias.zero_()
            self.attn.in_proj_weight[:dim, :] = 0.5 * torch.eye(dim)
            self.attn.in_proj_weight[dim:2 * dim, :] = 0.5 * torch.eye(dim)
            self.attn.in_proj_weight[2 * dim:] = torch.eye(dim)
            self.attn.out_proj.weight.copy_(torch.eye(dim))
            self.attn.out_proj.bias.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[1, 0] = 8.0
            self.head.weight[0, 0] = -8.0

    def forward(self, x):
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        if self.cls_token:
            tokens = torch.cat(
                [self.cls.expand(x.shape[0], -1, -1), tokens], 1)
        if self.wiring == "cross":
            out, _ = self.attn(tokens[:, :1], tokens, tokens)
        elif self.wiring == "no_weights":
            out, _ = self.attn(tokens, tokens, tokens, None, False)
        elif self.wiring == "averaged":
            out, _ = self.attn(tokens, tokens, tokens,
                               average_attn_weights=True)
        else:
            out, _ = self.attn(tokens, tokens, tokens)
        pooled = out[:, 0] if self.cls_token else out.mean(dim=1)
        return self.head(pooled)


def _in_corner_patch(amap, patch=4):
    row, col = divmod(int(np.argmax(amap)), amap.shape[1])
    return row < patch and col < patch


class TestHiResCam:
    def test_a_layer_off_the_gradient_path_is_refused(self, image):
        """A frozen extractor has no gradient to weight its channels by."""
        with pytest.raises(att.AttributionError) as excinfo:
            att.attribute(FrozenFeatures(), image, "hirescam", target=1,
                          layer="conv")
        assert "received no gradient" in str(excinfo.value)


class TestAblationCam:
    def test_a_block_returning_a_tuple_has_its_feature_map_ablated(
            self, image):
        result = att.attribute(TupleCNN(), image, "ablation_cam", target=1,
                               layer="block", batch_size=1)
        assert result.layer == "block"
        assert any("re-scored the image 4 times" in n for n in result.notes)
        assert np.isfinite(result.map).all()
        assert not result.is_flat()
        assert _in_corner_patch(result.map, CORNER)


class TestShapBaselines:
    def test_a_comma_separated_string_names_each_reference(self):
        x = torch.rand(1, 3, IMG, IMG)
        refs = att._shap_baselines(x, {"shap_baselines": "zero, mean"})
        assert refs.shape == (2, 3, IMG, IMG)
        assert torch.equal(refs[0], torch.zeros(3, IMG, IMG))
        assert torch.allclose(refs[1], x.mean(dim=(-2, -1), keepdim=True)
                              .expand_as(x)[0])

    def test_a_single_reference_is_joined_by_a_blurred_copy(self):
        """One reference would turn GradientSHAP back into integrated
        gradients."""
        x = torch.rand(1, 3, IMG, IMG)
        refs = att._shap_baselines(x, {"shap_baselines": ["zero"]})
        assert refs.shape == (2, 3, IMG, IMG)
        assert torch.allclose(refs[1:], att._blur(x))

    def test_a_string_of_baselines_reaches_gradient_shap(self, image):
        result = att.attribute(CornerCNN(), image, "gradient_shap", target=1,
                               shap_baselines="zero mean", shap_samples=4)
        assert result.params["shap_baselines"] == "zero mean"
        assert _in_corner_patch(result.map, CORNER)

    def test_without_captum_the_error_says_what_to_install(
            self, image, monkeypatch):
        monkeypatch.setitem(sys.modules, "captum.attr", None)
        with pytest.raises(att.AttributionError) as excinfo:
            att.attribute(CornerCNN(), image, "gradient_shap", target=1)
        assert "pip install captum" in str(excinfo.value)


class TestChefer:
    def test_a_training_model_is_put_back_in_training(self, image):
        model = TinyViT().train()
        result = att.chefer_relevance(model, image, target=1)
        assert model.training is True
        assert result.map.shape == (IMG, IMG)
        assert _in_corner_patch(result.map)

    def test_without_a_class_token_relevance_is_averaged_over_tokens(
            self, image):
        result = att.attribute(TinyViT(cls_token=False), image, "chefer",
                               target=1)
        assert any("No class token" in note for note in result.notes)
        assert result.map.shape == (IMG, IMG)
        assert not result.is_flat()
        assert _in_corner_patch(result.map)

    def test_weights_averaged_before_they_are_returned_are_refused(
            self, image):
        """torch averages the heads AFTER using them, so the averaged copy
        is not on the gradient path and has no gradient to weight."""
        with pytest.raises(att.NoSpatialLayerError) as excinfo:
            att.attribute(TinyViT(wiring="averaged"), image, "chefer",
                          target=1)
        assert "not on the gradient path" in str(excinfo.value)

    def test_a_single_head_block_returning_3d_weights_is_read(self, image):
        """A hand-rolled block whose (B, L, S) weights ARE the ones used."""
        result = att.attribute(TinyViT(wiring="single_head"), image,
                               "chefer", target=1)
        assert result.map.shape == (IMG, IMG)
        assert not result.is_flat()
        assert _in_corner_patch(result.map)

    def test_cross_attention_is_refused(self, image):
        with pytest.raises(att.NoSpatialLayerError) as excinfo:
            att.attribute(TinyViT(wiring="cross"), image, "chefer", target=1)
        assert "non-square" in str(excinfo.value)

    def test_weights_the_caller_turned_off_are_refused(self, image):
        """need_weights=False passed positionally is left alone, and then
        there is nothing to weight."""
        with pytest.raises(att.NoSpatialLayerError) as excinfo:
            att.attribute(TinyViT(wiring="no_weights"), image, "chefer",
                          target=1)
        assert "not on the gradient path" in str(excinfo.value)

    def test_tokens_that_are_not_a_grid_are_refused(self):
        """An 8x12 image in 4x4 patches is six tokens: no square grid."""
        wide = np.zeros((3, 8, 12), dtype="float32")
        wide[0, :4, :4] = 1.0
        with pytest.raises(att.NoSpatialLayerError) as excinfo:
            att.attribute(TinyViT(cls_token=False), wide, "chefer", target=1)
        assert "do not form a square patch grid" in str(excinfo.value)


class MaxVitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)


class SwinBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)


class ShiftedWindowAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)


class TestArchitectureKind:
    @pytest.mark.parametrize("model_type, kind", [
        ("swin_t", "swin"), ("maxvit_t", "hybrid"), ("vit_b_16", "vit"),
        ("resnet50", "cnn")])
    def test_by_name(self, model_type, kind):
        assert att.architecture_kind(model_type=model_type) == kind

    @pytest.mark.parametrize("model, kind", [
        (nn.Sequential(MaxVitBlock()), "hybrid"),
        (nn.Sequential(SwinBlock()), "swin"),
        (nn.Sequential(ShiftedWindowAttention()), "swin"),
        (nn.Sequential(nn.Flatten(), nn.Linear(3, 2)), "other"),
    ])
    def test_by_loaded_model(self, model, kind):
        assert att.architecture_kind(model) == kind

    def test_nothing_to_go_on(self):
        assert att.architecture_kind() == "unknown"


class TestMethodApplicability:
    def test_an_unknown_method_is_refused_with_the_registered_ones(self):
        with pytest.raises(att.UnknownMethodError) as excinfo:
            att.method_applicability("grad_cam")
        assert "hirescam" in str(excinfo.value)

    def test_a_cam_on_a_model_without_convolutions(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
        ok, reason = att.method_applicability("hirescam", model=model)
        assert not ok and "no Conv2d feature map" in reason

    def test_attention_methods_on_swin(self):
        ok, reason = att.method_applicability("chefer", model_type="swin_t")
        assert not ok and "shifted local windows" in reason


class TestTheChecksAroundTheMethods:
    def test_a_curve_puts_a_training_model_back_in_training(self, image):
        model = CornerCNN().train()
        amap = att.attribute(model, image, "hirescam", target=1).map
        assert model.training is True
        curve = att.deletion_curve(model, image, amap, target=1, n_steps=4)
        assert model.training is True
        assert curve.scores.shape == (5,)
        assert curve.scores[-1] < curve.scores[0]

    def test_a_curve_leaves_an_evaluating_model_evaluating(self, image):
        model = CornerCNN().eval()
        amap = att.attribute(model, image, "hirescam", target=1).map
        curve = att.insertion_curve(model, image, amap, target=1, n_steps=4)
        assert model.training is False
        assert curve.scores[-1] > curve.scores[0]

    def test_the_pointing_game_rate_with_every_mask_usable(self):
        amap = np.zeros((4, 4))
        amap[3, 3] = 1.0
        inside = np.zeros((4, 4), dtype=int)
        inside[3, 3] = 7
        outside = np.zeros((4, 4), dtype=int)
        outside[0, 0] = 7
        out = att.pointing_game_rate([amap, amap], [inside, outside])
        assert out == {"rate": 0.5, "hits": 1, "n": 2, "skipped": [],
                       "notes": [att.CRITERION_CAVEATS["pointing_game"]]}

    def test_the_pointing_game_rate_excludes_a_missing_mask(self):
        amap = np.zeros((4, 4))
        amap[0, 0] = 1.0
        hit = np.zeros((4, 4), dtype=int)
        hit[0, 0] = 1
        empty = np.zeros((4, 4), dtype=int)
        out = att.pointing_game_rate([amap, amap], [hit, empty])
        assert out["rate"] == 1.0 and out["hits"] == 1 and out["n"] == 1
        assert len(out["skipped"]) == 1 and "image 1" in out["skipped"][0]
        assert any("1 of 2 images were excluded" in n for n in out["notes"])

    def test_a_parameter_with_no_spread_is_randomised_at_a_fixed_scale(self):
        """All-zero or single-valued parameters have no scale to copy."""
        layer = nn.Linear(3, 1)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(2.0)
        att._randomize_module(layer, torch.Generator().manual_seed(0))
        assert torch.count_nonzero(layer.weight) == 3
        assert float(layer.weight.abs().max()) < 0.5
        assert float(layer.bias) != 2.0
        assert abs(float(layer.bias)) < 0.5
