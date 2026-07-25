"""Branch coverage for the torch building blocks in ``spacr.utils``.

Covers the head-stripping / output-unwrapping machinery of ``TorchModel`` and
``TorchModel_v2``, every branch of ``FocalLossWithLogits``, and the ``ResNet``
wrapper.

The TorchVision ``models`` namespace referenced by ``spacr.utils`` is swapped
for a tiny in-process stand-in module for the exotic branches (``_fc`` /
``heads`` / ``head`` heads, dict + tuple + namedtuple backbone outputs, legacy
``pretrained=`` API, ImageNet weight selection).  That keeps the tests CPU-only,
offline (no checkpoint downloads) and sub-second, while still driving the real
product code end to end.  Where a real backbone is cheap enough (``resnet18``
with ``weights=None``) the genuine article is used instead.
"""
from __future__ import annotations

import types
from collections import namedtuple

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")


# ---------------------------------------------------------------------------
# Tiny stand-in backbones (all CPU, ~a few hundred params each)
# ---------------------------------------------------------------------------

_AuxOut = namedtuple("_AuxOut", ["logits", "aux"])


class _FakeTiny(nn.Module):
    """conv -> global-avg-pool -> <head_name> linear head.

    ``head_name`` decides which attribute name the classification head lives
    under, which is exactly what ``_remove_head_for_features`` dispatches on.
    """

    head_name = "fc"
    feat_dim = 8

    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        self.init_pretrained = pretrained
        self.init_weights_arg = weights
        self.conv = nn.Conv2d(3, self.feat_dim, 3, stride=2, padding=1)
        setattr(self, self.head_name, nn.Linear(self.feat_dim, 4))

    def _features(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(self.conv(x), 1), 1)

    def forward(self, x):
        return getattr(self, self.head_name)(self._features(x))


class _FakeFc(_FakeTiny):
    head_name = "fc"


class _FakeUnderscoreFc(_FakeTiny):
    """Old-style EfficientNet: head lives under ``_fc``, there is no ``fc``."""

    head_name = "_fc"


class _FakeHeads(_FakeTiny):
    """TorchVision ViT: head lives under ``heads``."""

    head_name = "heads"


class _FakeSpatialHead(nn.Module):
    """Swin-style ``head`` attribute; emits a 4-D map once the head is gone."""

    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.head = nn.Linear(24, 4)

    def _map(self, x):
        return F.adaptive_avg_pool2d(self.conv(x), 2)  # (N, 6, 2, 2)

    def forward(self, x):
        m = self._map(x)
        if isinstance(self.head, nn.Identity):
            return self.head(m)                     # 4-D -> forces a flatten
        return self.head(torch.flatten(m, 1))


class _FakeSpatialClassifier(nn.Module):
    """Same idea but under ``classifier`` (the branch TorchModel_v2 uses)."""

    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.classifier = nn.Linear(24, 4)

    def forward(self, x):
        m = F.adaptive_avg_pool2d(self.conv(x), 2)  # (N, 6, 2, 2)
        if isinstance(self.classifier, nn.Identity):
            return self.classifier(m)
        return self.classifier(torch.flatten(m, 1))


class _FakeAux(_FakeTiny):
    """Inception/GoogLeNet-style: ``aux_logits`` flag + namedtuple output."""

    def __init__(self, pretrained=False, weights=None):
        super().__init__(pretrained=pretrained, weights=weights)
        self.aux_logits = True

    def forward(self, x):
        return _AuxOut(logits=self.fc(self._features(x)), aux=None)


class _FakeTupleOut(_FakeTiny):
    """Backbone that returns ``(primary, aux)``."""

    def forward(self, x):
        return (self.fc(self._features(x)), None)


class _FakeDictOut(_FakeTiny):
    """Detection/segmentation-style dict output -> must be rejected."""

    def forward(self, x):
        return {"out": self.fc(self._features(x))}


class _FakeBadShape(_FakeTiny):
    """Returns a 1-D tensor, which cannot be interpreted as (N, F) features."""

    def forward(self, x):
        return self.fc(self._features(x)).reshape(-1)


class _FakeDropout(_FakeTiny):
    """Backbone carrying Dropout / Dropout2d so `_apply_dropout_rate` bites."""

    def __init__(self, pretrained=False, weights=None):
        super().__init__(pretrained=pretrained, weights=weights)
        self.drop2d = nn.Dropout2d(0.1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        m = self.drop2d(self.conv(x))
        f = self.drop(torch.flatten(F.adaptive_avg_pool2d(m, 1), 1))
        return self.fc(f)


class _FakeMaxvit(nn.Module):
    """MaxViT-T shape: a multi-layer ``classifier`` Sequential."""

    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(8, 5),
            nn.Linear(5, 4),
        )

    def forward(self, x):
        pooled = torch.flatten(F.adaptive_avg_pool2d(self.conv(x), 1), 1)
        return self.classifier(pooled)


class _FakeResNet1000(nn.Module):
    """Stand-in for torchvision resnets: 1000-way output, records `weights`."""

    def __init__(self, weights=None, pretrained=False):
        super().__init__()
        self.weights_arg = weights
        self.pretrained_arg = pretrained
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.head = nn.Linear(16, 1000)

    def forward(self, x):
        return self.head(torch.flatten(F.adaptive_avg_pool2d(self.conv(x), 1), 1))


def _fake_models_module():
    mod = types.ModuleType("fake_torchvision_models")
    mod.tiny_fc = _FakeFc
    mod.tiny_underscore_fc = _FakeUnderscoreFc
    mod.tiny_heads = _FakeHeads
    mod.tiny_head_spatial = _FakeSpatialHead
    mod.tiny_classifier_spatial = _FakeSpatialClassifier
    mod.tiny_aux = _FakeAux
    mod.tiny_tuple = _FakeTupleOut
    mod.tiny_dict = _FakeDictOut
    mod.tiny_bad = _FakeBadShape
    mod.tiny_dropout = _FakeDropout
    mod.maxvit_t = _FakeMaxvit
    for name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        setattr(mod, name, _FakeResNet1000)
    return mod


@pytest.fixture
def fake_models(monkeypatch):
    """Swap ``spacr.utils.models`` for the tiny offline stand-in namespace."""
    import spacr.utils as U

    mod = _fake_models_module()
    monkeypatch.setattr(U, "models", mod)
    return mod


# ---------------------------------------------------------------------------
# TorchModel: backbone init / weight-choice fallbacks
# ---------------------------------------------------------------------------

def test_torchmodel_falls_back_to_legacy_pretrained_kwarg(fake_models):
    """No ``<name>_weights`` enum exists -> legacy ``pretrained=`` API is used."""
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_fc", pretrained=True, num_classes=3,
                   image_size=32)
    assert m._get_weight_choice() is None
    assert m.base_model.init_pretrained is True     # legacy kwarg forwarded
    assert m.base_model.init_weights_arg is None    # new-style kwarg NOT used
    assert m.num_ftrs == 8
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_torchmodel_maxvit_keeps_all_but_last_classifier_layer(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="maxvit_t", pretrained=False, num_classes=2,
                   image_size=32)
    # last nn.Linear dropped, the rest of the Sequential survives
    assert isinstance(m.base_model.classifier, nn.Sequential)
    assert len(m.base_model.classifier) == 2
    assert isinstance(m.base_model.classifier[-1], nn.Linear)
    assert m.num_ftrs == 5
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_torchmodel_pushes_dropout_rate_into_backbone_and_head(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_dropout", pretrained=False, num_classes=2,
                   dropout_rate=0.25, image_size=32)
    # every Dropout* module in the backbone was rewritten
    assert m.base_model.drop.p == pytest.approx(0.25)
    assert m.base_model.drop2d.p == pytest.approx(0.25)
    # ...and the SPACR head got its own dropout, used by forward()
    assert m.use_dropout is True
    assert isinstance(m.dropout, nn.Dropout)
    assert m.dropout.p == pytest.approx(0.25)
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# TorchModel: head removal for every supported attribute name
# ---------------------------------------------------------------------------

def test_torchmodel_disables_aux_logits_and_unwraps_namedtuple(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_aux", pretrained=False, num_classes=2,
                   image_size=32)
    assert m.base_model.aux_logits is False          # aux head switched off
    assert isinstance(m.base_model.fc, nn.Identity)
    assert m.num_ftrs == 8                           # .logits was unwrapped
    with torch.no_grad():
        raw = m._run_backbone_raw(torch.zeros(1, 3, 32, 32))
    assert isinstance(raw, torch.Tensor) and raw.shape == (1, 8)


def test_torchmodel_removes_underscore_fc_head(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_underscore_fc", pretrained=False,
                   num_classes=2, image_size=32)
    assert isinstance(m.base_model._fc, nn.Identity)
    assert not hasattr(m.base_model, "fc")
    assert m.num_ftrs == 8


def test_torchmodel_removes_vit_heads_head(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_heads", pretrained=False, num_classes=4,
                   image_size=32)
    assert isinstance(m.base_model.heads, nn.Identity)
    assert m.num_ftrs == 8
    with torch.no_grad():
        out = m.eval()(torch.rand(3, 3, 32, 32))
    assert out.shape == (3, 4)


def test_torchmodel_removes_swin_head_and_flattens_spatial_features(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_head_spatial", pretrained=False,
                   num_classes=2, image_size=32)
    assert isinstance(m.base_model.head, nn.Identity)
    assert m.num_ftrs == 24                       # 6 * 2 * 2, flattened
    with torch.no_grad():
        feats = m._run_backbone(torch.rand(2, 3, 32, 32))
    assert feats.shape == (2, 24)                 # _run_backbone flattens too
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_torchmodel_unwraps_tuple_backbone_output(fake_models):
    from spacr.utils import TorchModel

    m = TorchModel(model_name="tiny_tuple", pretrained=False, num_classes=2,
                   image_size=32)
    assert m.num_ftrs == 8
    with torch.no_grad():
        raw = m._run_backbone_raw(torch.zeros(1, 3, 32, 32))
    assert isinstance(raw, torch.Tensor) and raw.shape == (1, 8)


def test_torchmodel_rejects_dict_backbone_output(fake_models):
    from spacr.utils import TorchModel

    with pytest.raises(RuntimeError, match="dict"):
        TorchModel(model_name="tiny_dict", pretrained=False, num_classes=2,
                   image_size=32)


def test_torchmodel_rejects_non_2d_feature_output(fake_models):
    from spacr.utils import TorchModel

    with pytest.raises(RuntimeError, match="unexpected shape"):
        TorchModel(model_name="tiny_bad", pretrained=False, num_classes=2,
                   image_size=32)


# ---------------------------------------------------------------------------
# TorchModel_v2
# ---------------------------------------------------------------------------

def test_torchmodel_v2_unknown_model_raises():
    from spacr.utils import TorchModel_v2

    with pytest.raises(ValueError, match="Unknown torchvision model"):
        TorchModel_v2(model_name="definitely_not_a_torchvision_model",
                      pretrained=False, num_classes=2)


def test_torchmodel_v2_legacy_pretrained_fallback(fake_models):
    from spacr.utils import TorchModel_v2

    m = TorchModel_v2(model_name="tiny_fc", pretrained=True, num_classes=3)
    assert m._get_weight_choice() is None
    assert m.base_model.init_pretrained is True
    assert m.base_model.init_weights_arg is None
    assert isinstance(m.base_model.fc, nn.Identity)
    assert m.num_ftrs == 8
    assert m.use_dropout is False
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_torchmodel_v2_maxvit_keeps_all_but_last_classifier_layer(fake_models):
    from spacr.utils import TorchModel_v2

    m = TorchModel_v2(model_name="maxvit_t", pretrained=False, num_classes=2)
    assert len(m.base_model.classifier) == 2
    assert m.num_ftrs == 5
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_torchmodel_v2_applies_dropout_rate_and_uses_it_in_forward(fake_models):
    from spacr.utils import TorchModel_v2

    m = TorchModel_v2(model_name="tiny_dropout", pretrained=False,
                      num_classes=2, dropout_rate=0.3)
    assert m.base_model.drop.p == pytest.approx(0.3)
    assert m.base_model.drop2d.p == pytest.approx(0.3)
    assert m.use_dropout is True and m.dropout.p == pytest.approx(0.3)
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_torchmodel_v2_blanks_classifier_and_flattens_spatial(fake_models):
    from spacr.utils import TorchModel_v2

    m = TorchModel_v2(model_name="tiny_classifier_spatial", pretrained=False,
                      num_classes=5)
    assert isinstance(m.base_model.classifier, nn.Identity)
    assert m.num_ftrs == 24                      # flattened (6, 2, 2)
    with torch.no_grad():
        feats = m._run_backbone(torch.rand(2, 3, 32, 32))
    assert feats.ndim == 4                       # backbone itself stays 4-D
    with torch.no_grad():
        out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 5)                   # forward() did the flatten


def test_torchmodel_v2_checkpointed_forward(fake_models):
    from spacr.utils import TorchModel_v2

    m = TorchModel_v2(model_name="tiny_fc", pretrained=False, num_classes=2,
                      use_checkpoint=True).eval()
    plain = TorchModel_v2(model_name="tiny_fc", pretrained=False, num_classes=2)
    plain.load_state_dict(m.state_dict())
    plain.eval()

    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    out = m(x)
    assert out.shape == (2, 2)
    with torch.no_grad():
        assert torch.allclose(out.detach(), plain(x), atol=1e-6)


# ---------------------------------------------------------------------------
# FocalLossWithLogits
# ---------------------------------------------------------------------------

def _ref_bce_focal(logits, target, alpha, gamma):
    p = torch.sigmoid(logits)
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    pt = target * p + (1 - target) * (1 - p)
    return alpha * (1 - pt) ** gamma * bce


def test_focal_loss_binary_matches_reference_and_reductions():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([-1.5, 0.4, 2.0, -0.2])
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    expected = _ref_bce_focal(logits, target, alpha=0.75, gamma=2.0)

    none = FocalLossWithLogits(alpha=0.75, gamma=2.0, reduction="none")
    per_sample = none(logits, target)
    assert per_sample.shape == (4,)
    assert torch.allclose(per_sample, expected, atol=1e-6)

    mean = FocalLossWithLogits(alpha=0.75, gamma=2.0, reduction="mean")
    assert torch.allclose(mean(logits, target), expected.mean(), atol=1e-6)

    summed = FocalLossWithLogits(alpha=0.75, gamma=2.0, reduction="sum")
    assert torch.allclose(summed(logits, target), expected.sum(), atol=1e-6)


def test_focal_loss_gamma_zero_is_scaled_bce():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([-0.8, 1.1, 0.3])
    target = torch.tensor([1.0, 1.0, 0.0])
    loss = FocalLossWithLogits(alpha=2.0, gamma=0.0, reduction="mean")(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    assert torch.allclose(loss, 2.0 * bce, atol=1e-6)


def test_focal_loss_single_column_logits_are_treated_as_binary():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[-1.0], [0.7], [1.9]])
    target = torch.tensor([0.0, 1.0, 1.0])
    expected = _ref_bce_focal(logits.view_as(target), target, alpha=1.0, gamma=2.0)
    loss = FocalLossWithLogits(reduction="none")(logits, target)
    assert loss.shape == (3,)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_focal_loss_multilabel_uses_bce_branch():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[-1.0, 0.5, 2.0], [0.1, -0.4, 1.2]])
    target = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    expected = _ref_bce_focal(logits, target, alpha=1.0, gamma=1.5)
    loss = FocalLossWithLogits(gamma=1.5, reduction="none")(logits, target)
    assert loss.shape == (2, 3)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_focal_loss_multiclass_gamma_zero_equals_cross_entropy():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.4, 0.3], [-0.7, 0.0, 2.2]])
    target = torch.tensor([0, 1, 2])
    loss = FocalLossWithLogits(alpha=1.0, gamma=0.0, reduction="mean")(logits, target)
    assert torch.allclose(loss, F.cross_entropy(logits, target), atol=1e-6)


def test_focal_loss_multiclass_downweights_easy_examples():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[6.0, -6.0], [0.05, -0.05]])   # easy, then hard
    target = torch.tensor([0, 0])
    ce = F.cross_entropy(logits, target, reduction="none")
    focal = FocalLossWithLogits(gamma=2.0, reduction="none")(logits, target)
    assert focal.shape == (2,)
    # the confident sample is suppressed far harder than the uncertain one
    assert float(focal[0] / ce[0]) < float(focal[1] / ce[1])
    assert float(focal[0]) < float(focal[1])


def test_focal_loss_multiclass_tensor_alpha_is_per_class():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[1.0, 0.2, -0.5], [0.3, 1.7, 0.1]])
    target = torch.tensor([0, 2])
    alpha = torch.tensor([0.5, 1.0, 3.0])
    ce = F.cross_entropy(logits, target, reduction="none")
    p = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    expected = alpha[target] * (1 - p) ** 2.0 * ce
    loss = FocalLossWithLogits(alpha=alpha, gamma=2.0, reduction="none")(logits, target)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_focal_loss_multiclass_casts_float_targets_to_long():
    from spacr.utils import FocalLossWithLogits

    logits = torch.tensor([[1.0, 0.2, -0.5], [0.3, 1.7, 0.1]])
    fn = FocalLossWithLogits(gamma=2.0, reduction="sum")
    as_long = fn(logits, torch.tensor([0, 1]))
    as_float = fn(logits, torch.tensor([0.0, 1.0]))
    assert torch.allclose(as_long, as_float, atol=1e-6)


# ---------------------------------------------------------------------------
# ResNet wrapper
# ---------------------------------------------------------------------------

def test_resnet_rejects_unknown_type():
    from spacr.utils import ResNet

    with pytest.raises(ValueError, match="Invalid resnet_type"):
        ResNet(resnet_type="resnet999")


def test_resnet_rejects_bad_init_weights():
    from spacr.utils import ResNet

    with pytest.raises(ValueError, match="init_weights should be either"):
        ResNet(resnet_type="resnet18", init_weights="somewhere-else")


def test_resnet_random_init_builds_head_and_forwards():
    """Real torchvision resnet18 with weights=None (no download)."""
    from spacr.utils import ResNet

    m = ResNet(resnet_type="resnet18", init_weights="none")
    assert m.use_dropout is False
    assert not hasattr(m, "dropout")
    assert m.fc1.in_features == 1000 and m.fc1.out_features == 500
    assert m.fc2.in_features == 500 and m.fc2.out_features == 1

    out = m.eval()(torch.rand(2, 3, 64, 64))
    assert out.shape == (2,)                       # flattened single logit
    assert torch.isfinite(out).all()


def test_resnet_dropout_branch_is_applied():
    from spacr.utils import ResNet

    m = ResNet(resnet_type="resnet18", dropout_rate=0.5, init_weights="none")
    assert m.use_dropout is True
    assert isinstance(m.dropout, nn.Dropout)
    assert m.dropout.p == pytest.approx(0.5)

    m.train()
    torch.manual_seed(0)
    out = m(torch.rand(2, 3, 64, 64))
    assert out.shape == (2,)
    assert out.requires_grad is True               # forward set requires_grad


def test_resnet_imagenet_branch_passes_pretrained_weights(fake_models):
    from torchvision.models.resnet import ResNet18_Weights

    from spacr.utils import ResNet

    m = ResNet(resnet_type="resnet18", init_weights="imagenet")
    # the IMAGENET1K_V1 enum for the requested depth reached the constructor
    assert m.resnet.weights_arg is ResNet18_Weights.IMAGENET1K_V1
    out = m.eval()(torch.rand(2, 3, 32, 32))
    assert out.shape == (2,)


def test_resnet_checkpointed_forward_matches_plain(fake_models):
    from spacr.utils import ResNet

    torch.manual_seed(0)
    ckpt = ResNet(resnet_type="resnet34", use_checkpoint=True,
                  init_weights="none").eval()
    plain = ResNet(resnet_type="resnet34", use_checkpoint=False,
                   init_weights="none").eval()
    plain.load_state_dict(ckpt.state_dict())

    x = torch.rand(2, 3, 32, 32)
    out = ckpt(x)
    assert out.shape == (2,)
    assert out.requires_grad is True
    assert torch.allclose(out.detach(), plain(x).detach(), atol=1e-6)
