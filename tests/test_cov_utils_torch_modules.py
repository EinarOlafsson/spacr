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


# ---------------------------------------------------------------------------
# TorchModel_v2
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FocalLossWithLogits
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ResNet wrapper
# ---------------------------------------------------------------------------
