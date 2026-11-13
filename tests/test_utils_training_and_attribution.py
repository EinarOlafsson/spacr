"""Training, loss, model-wrapper and attribution helpers of ``spacr.utils``.

One table-driven test per unit (two or three for the large ones), each row run
as a subtest so a failing case names itself.  Everything is CPU-only, offline
and works on 8-64 px tensors: tiny stand-in backbones replace TorchVision
wherever a real one would download weights or cost seconds.

Units: FocalLossWithLogits, ResNet, TorchModel, TorchModel_v2,
_list_torchvision_model_names, choose_model, calculate_loss,
compute_irm_penalty, estimate_class_counts, pick_best_model and build_loss
(attribution helpers and suggest_training_changes live in
``test_utils_training_and_attribution_2.py``).
"""
from __future__ import annotations

import os
import tarfile
import types
from collections import namedtuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ===========================================================================
# Stand-in TorchVision backbones (all CPU, a few hundred parameters each)
# ===========================================================================

_AuxOut = namedtuple("_AuxOut", ["logits", "aux"])


class _FakeTiny(nn.Module):
    """conv -> global-avg-pool -> ``<head_name>`` linear head."""

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
    """Old-style EfficientNet: head under ``_fc``, no ``fc``."""

    head_name = "_fc"


class _FakeHeads(_FakeTiny):
    """TorchVision ViT: head under ``heads``."""

    head_name = "heads"


class _FakeDictOut(_FakeTiny):
    """Detection-style dict output -> must be rejected."""

    def forward(self, x):
        return {"out": self.fc(self._features(x))}


class _FakeBadShape(_FakeTiny):
    """1-D output, which cannot be read as (N, F) features."""

    def forward(self, x):
        return self.fc(self._features(x)).reshape(-1)


class _FakeDropout(_FakeTiny):
    """Backbone carrying Dropout / Dropout2d so the dropout rewrite bites."""

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
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(8, 5),
                                        nn.Linear(5, 4))

    def forward(self, x):
        pooled = torch.flatten(F.adaptive_avg_pool2d(self.conv(x), 1), 1)
        return self.classifier(pooled)


class _FakeResNet1000(nn.Module):
    """Stand-in for torchvision resnets: 1000-way output, records ``weights``."""

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
    mod.tiny_dict = _FakeDictOut
    mod.tiny_bad = _FakeBadShape
    mod.tiny_dropout = _FakeDropout
    mod.maxvit_t = _FakeMaxvit
    for name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        setattr(mod, name, _FakeResNet1000)
    return mod


def _headless_backbone():
    """A backbone with no ``fc``/``classifier``/``head`` of any spelling."""

    class Headless(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1),
                                          nn.AdaptiveAvgPool2d(1))

        def forward(self, x):
            return self.features(x)

    return Headless()


def _patch_headless_into_torchvision(mp):
    from torchvision import models
    mp.setattr(models, "headless_net", lambda **kwargs: _headless_backbone(),
               raising=False)


# ===========================================================================
# FocalLossWithLogits
# ===========================================================================

def _ref_bce_focal(logits, target, alpha, gamma):
    p = torch.sigmoid(logits)
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    pt = target * p + (1 - target) * (1 - p)
    return alpha * (1 - pt) ** gamma * bce


def _ref_ce_focal(logits, target, alpha, gamma):
    ce = F.cross_entropy(logits, target, reduction="none")
    p = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    a = alpha[target] if torch.is_tensor(alpha) else alpha
    return a * (1 - p) ** gamma * ce


def test_focal_loss_with_logits_matches_its_reference_on_every_branch(subtests):
    """Binary, single-column, multilabel and multiclass inputs each match the
    closed-form focal loss for every reduction; gamma=0 collapses to (scaled)
    BCE/CE, easy examples are down-weighted and float class targets are cast."""
    from spacr.utils import FocalLossWithLogits

    b_logits = torch.tensor([-1.5, 0.4, 2.0, -0.2])
    b_target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    sc_logits = torch.tensor([[-1.0], [0.7], [1.9]])
    sc_target = torch.tensor([0.0, 1.0, 1.0])
    ml_logits = torch.tensor([[-1.0, 0.5, 2.0], [0.1, -0.4, 1.2]])
    ml_target = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    mc_logits = torch.tensor([[1.0, 0.2, -0.5], [0.3, 1.7, 0.1]])
    mc_target = torch.tensor([0, 2])
    alpha_t = torch.tensor([0.5, 1.0, 3.0])
    ce3_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.4, 0.3], [-0.7, 0.0, 2.2]])
    ce3_target = torch.tensor([0, 1, 2])

    b_ref = _ref_bce_focal(b_logits, b_target, 0.75, 2.0)
    cases = [
        # (id, ctor kwargs, logits, target, expected, expected shape or None)
        ("binary-none", dict(alpha=0.75, gamma=2.0, reduction="none"),
         b_logits, b_target, b_ref, (4,)),
        ("binary-mean", dict(alpha=0.75, gamma=2.0, reduction="mean"),
         b_logits, b_target, b_ref.mean(), ()),
        ("binary-sum", dict(alpha=0.75, gamma=2.0, reduction="sum"),
         b_logits, b_target, b_ref.sum(), ()),
        ("binary-gamma0-is-2x-bce", dict(alpha=2.0, gamma=0.0, reduction="mean"),
         torch.tensor([-0.8, 1.1, 0.3]), torch.tensor([1.0, 1.0, 0.0]),
         2.0 * F.binary_cross_entropy_with_logits(
             torch.tensor([-0.8, 1.1, 0.3]), torch.tensor([1.0, 1.0, 0.0])), ()),
        ("single-column-is-binary", dict(reduction="none"), sc_logits, sc_target,
         _ref_bce_focal(sc_logits.view_as(sc_target), sc_target, 1.0, 2.0), (3,)),
        ("multilabel-bce", dict(gamma=1.5, reduction="none"), ml_logits, ml_target,
         _ref_bce_focal(ml_logits, ml_target, 1.0, 1.5), (2, 3)),
        ("multiclass-gamma0-is-ce", dict(alpha=1.0, gamma=0.0, reduction="mean"),
         ce3_logits, ce3_target, F.cross_entropy(ce3_logits, ce3_target), ()),
        ("multiclass-per-class-alpha", dict(alpha=alpha_t, gamma=2.0, reduction="none"),
         mc_logits, mc_target, _ref_ce_focal(mc_logits, mc_target, alpha_t, 2.0), (2,)),
    ]
    for cid, kw, logits, target, expected, shape in cases:
        with subtests.test(case=cid):
            got = FocalLossWithLogits(**kw)(logits, target)
            assert tuple(got.shape) == shape
            assert torch.allclose(got, expected, atol=1e-6)

    with subtests.test(case="multiclass-downweights-easy-examples"):
        logits = torch.tensor([[6.0, -6.0], [0.05, -0.05]])   # easy, then hard
        target = torch.tensor([0, 0])
        ce = F.cross_entropy(logits, target, reduction="none")
        focal = FocalLossWithLogits(gamma=2.0, reduction="none")(logits, target)
        assert focal.shape == (2,)
        assert float(focal[0] / ce[0]) < float(focal[1] / ce[1])
        assert float(focal[0]) < float(focal[1])

    with subtests.test(case="multiclass-float-targets-cast-to-long"):
        fn = FocalLossWithLogits(gamma=2.0, reduction="sum")
        assert torch.allclose(fn(mc_logits, torch.tensor([0, 1])),
                              fn(mc_logits, torch.tensor([0.0, 1.0])), atol=1e-6)


# ===========================================================================
# ResNet wrapper
# ===========================================================================

def test_resnet_wrapper_validates_its_arguments_and_forwards_one_logit(subtests):
    """Unknown depths / init modes are refused by message; every valid build
    yields a 1000->500->1 head, a flattened (N,) logit, optional dropout, the
    IMAGENET1K_V1 enum for 'imagenet', and a checkpointed forward equal to the
    plain one."""
    from spacr.utils import ResNet

    for cid, kw, match in [
        ("unknown-type", dict(resnet_type="resnet999"), "Invalid resnet_type"),
        ("bad-init", dict(resnet_type="resnet18", init_weights="somewhere-else"),
         "init_weights should be either"),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError, match=match):
                ResNet(**kw)

    with subtests.test(case="real-resnet18-random-init"):
        m = ResNet(resnet_type="resnet18", init_weights="none")
        assert m.use_dropout is False and not hasattr(m, "dropout")
        assert (m.fc1.in_features, m.fc1.out_features) == (1000, 500)
        assert (m.fc2.in_features, m.fc2.out_features) == (500, 1)
        out = m.eval()(torch.rand(2, 3, 64, 64))
        assert out.shape == (2,) and torch.isfinite(out).all()

    with subtests.test(case="real-resnet18-dropout"):
        m = ResNet(resnet_type="resnet18", dropout_rate=0.5, init_weights="none")
        assert m.use_dropout is True and isinstance(m.dropout, nn.Dropout)
        assert m.dropout.p == pytest.approx(0.5)
        m.train()
        torch.manual_seed(0)
        out = m(torch.rand(2, 3, 64, 64))
        assert out.shape == (2,) and out.requires_grad is True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(U, "models", _fake_models_module())
        with subtests.test(case="imagenet-weights-enum"):
            from torchvision.models.resnet import ResNet18_Weights
            m = ResNet(resnet_type="resnet18", init_weights="imagenet")
            assert m.resnet.weights_arg is ResNet18_Weights.IMAGENET1K_V1
            assert m.eval()(torch.rand(2, 3, 32, 32)).shape == (2,)

        with subtests.test(case="checkpointed-forward-matches-plain"):
            torch.manual_seed(0)
            ckpt = ResNet(resnet_type="resnet34", use_checkpoint=True,
                          init_weights="none").eval()
            plain = ResNet(resnet_type="resnet34", use_checkpoint=False,
                           init_weights="none").eval()
            plain.load_state_dict(ckpt.state_dict())
            x = torch.rand(2, 3, 32, 32)
            out = ckpt(x)
            assert out.shape == (2,) and out.requires_grad is True
            assert torch.allclose(out.detach(), plain(x).detach(), atol=1e-6)


# ===========================================================================
# TorchModel / TorchModel_v2
# ===========================================================================

def _write_png_seeded(path, seed, size=32):
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint16).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def _activation_project(tmp_path, n_images=4):
    """``<root>/datasets/ds.tar`` of crops named like spaCR crops + measurements dir."""
    root = tmp_path / "proj"
    (root / "measurements").mkdir(parents=True)
    ds_dir = root / "datasets"
    ds_dir.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    names = []
    for i in range(n_images):
        name = f"plate1_A01_1_{i}.png"
        _write_png_seeded(raw / name, seed=i)
        names.append(name)
    tar_path = ds_dir / "ds.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in names:
            tar.add(raw / name, arcname=name)
    return root, str(tar_path), names


def _activation_settings(tar_path, model_path, **over):
    s = {"dataset": tar_path, "model_path": model_path, "model_type": "resnet18",
         "cam_type": "saliency_image", "target_layer": None, "image_size": 32,
         "batch_size": 3, "channels": [1, 2, 3], "normalize": False,
         "normalize_input": True, "save": False, "plot": False,
         "correlation": False, "overlay": True, "shuffle": False, "n_jobs": 0,
         "manders_thresholds": [15, 50, 75]}
    s.update(over)
    return s


def test_torchmodel_strips_every_head_spelling_and_validates_backbone_output(
        subtests, tmp_path):
    """TorchModel blanks the head under fc/_fc/heads/multi-layer classifier,
    infers num_ftrs from a real forward, rewrites every Dropout, keeps a
    headless backbone, refuses dict / non-2-D outputs, and a 2-class model
    still yields one activation map per image."""
    from spacr.utils import TorchModel

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(U, "models", _fake_models_module())
        cases = [
            # (id, model_name, extra kwargs, num_classes, image_size, ftrs, check)
            ("maxvit-keeps-all-but-last-linear", "maxvit_t", {}, 2, 224, 5,
             lambda b: (isinstance(b.classifier, nn.Sequential)
                        and len(b.classifier) == 2
                        and isinstance(b.classifier[-1], nn.Linear))),
            ("underscore-fc", "tiny_underscore_fc", {}, 2, 32, 8,
             lambda b: isinstance(b._fc, nn.Identity) and not hasattr(b, "fc")),
            ("vit-heads", "tiny_heads", {}, 4, 32, 8,
             lambda b: isinstance(b.heads, nn.Identity)),
            ("dropout-pushed-into-backbone", "tiny_dropout",
             {"dropout_rate": 0.25}, 2, 32, 8,
             lambda b: (b.drop.p == pytest.approx(0.25)
                        and b.drop2d.p == pytest.approx(0.25))),
        ]
        for cid, name, extra, ncls, size, ftrs, check in cases:
            with subtests.test(case=cid):
                m = TorchModel(model_name=name, pretrained=False,
                               num_classes=ncls, image_size=size, **extra)
                assert check(m.base_model)
                assert m.num_ftrs == ftrs
                if "dropout_rate" in extra:
                    assert m.use_dropout is True and isinstance(m.dropout, nn.Dropout)
                    assert m.dropout.p == pytest.approx(extra["dropout_rate"])
                with torch.no_grad():
                    out = m.eval()(torch.rand(3, 3, 32, 32))
                assert out.shape == (3, ncls)

        for cid, name, match in [("dict-output", "tiny_dict", "dict"),
                                 ("non-2d-output", "tiny_bad", "unexpected shape")]:
            with subtests.test(case=cid):
                with pytest.raises(RuntimeError, match=match):
                    TorchModel(model_name=name, pretrained=False, num_classes=2,
                               image_size=32)

    with subtests.test(case="headless-backbone-keeps-its-output"):
        with pytest.MonkeyPatch.context() as mp:
            _patch_headless_into_torchvision(mp)
            m = TorchModel(model_name="headless_net", pretrained=False,
                           num_classes=3)
            assert m.num_ftrs == 4
            assert m(torch.randn(1, 3, 32, 32)).shape == (1, 3)

    with subtests.test(case="two-class-model-one-activation-map-per-image"):
        from spacr.deep_spacr import generate_activation_map
        mp_path = tmp_path / "m2.pth"
        torch.save(TorchModel(model_name="resnet18", pretrained=False,
                              num_classes=2, image_size=32), str(mp_path))
        root, tar_path, names = _activation_project(tmp_path, n_images=4)
        generate_activation_map(_activation_settings(tar_path, str(mp_path),
                                                     save=True, batch_size=2))
        pngs = sorted((root / "datasets" / "ds" / "saliency_image").rglob("*.png"))
        assert {p.name for p in pngs} == set(names)
        assert {p.parent.parent.parent.name for p in pngs} <= {"class_0", "class_1"}


def test_torchmodel_v2_builds_named_backbones_and_refuses_unknown_names(subtests):
    """TorchModel_v2 refuses an unknown name, drops only MaxViT's last
    classifier layer, rewrites dropout and uses it in forward, keeps a headless
    backbone, and its checkpointed forward equals the plain forward."""
    from spacr.utils import TorchModel_v2

    with subtests.test(case="unknown-model"):
        with pytest.raises(ValueError, match="Unknown torchvision model"):
            TorchModel_v2(model_name="definitely_not_a_torchvision_model",
                          pretrained=False, num_classes=2)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(U, "models", _fake_models_module())
        with subtests.test(case="maxvit"):
            m = TorchModel_v2(model_name="maxvit_t", pretrained=False, num_classes=2)
            assert len(m.base_model.classifier) == 2 and m.num_ftrs == 5
            with torch.no_grad():
                assert m.eval()(torch.rand(2, 3, 32, 32)).shape == (2, 2)

        with subtests.test(case="dropout"):
            m = TorchModel_v2(model_name="tiny_dropout", pretrained=False,
                              num_classes=2, dropout_rate=0.3)
            assert m.base_model.drop.p == pytest.approx(0.3)
            assert m.base_model.drop2d.p == pytest.approx(0.3)
            assert m.use_dropout is True and m.dropout.p == pytest.approx(0.3)
            with torch.no_grad():
                assert m.eval()(torch.rand(2, 3, 32, 32)).shape == (2, 2)

        with subtests.test(case="checkpointed-forward"):
            m = TorchModel_v2(model_name="tiny_fc", pretrained=False, num_classes=2,
                              use_checkpoint=True).eval()
            plain = TorchModel_v2(model_name="tiny_fc", pretrained=False,
                                  num_classes=2)
            plain.load_state_dict(m.state_dict())
            plain.eval()
            x = torch.rand(2, 3, 32, 32, requires_grad=True)
            out = m(x)
            assert out.shape == (2, 2)
            with torch.no_grad():
                assert torch.allclose(out.detach(), plain(x), atol=1e-6)

    with subtests.test(case="headless-backbone"):
        with pytest.MonkeyPatch.context() as mp:
            _patch_headless_into_torchvision(mp)
            m = TorchModel_v2(model_name="headless_net", pretrained=False,
                              num_classes=2)
            assert m.num_ftrs == 4
            assert m(torch.randn(1, 3, 224, 224)).shape == (1, 2)


# ===========================================================================
# _list_torchvision_model_names / choose_model
# ===========================================================================

def _tv_module(**attrs):
    fake = types.ModuleType("fake_tv_models")
    for k, v in attrs.items():
        setattr(fake, k, v)
    return fake


def _raising_list_models(module=None):
    raise RuntimeError("list_models unavailable in this torchvision")


def test_list_torchvision_model_names_offers_only_factories(subtests):
    """The list_models API is trusted when it answers; otherwise the module is
    scanned for public lower_snake_case callables only.  The real answer is a
    set holding resnet18/resnet50 and no *_Weights enum."""
    cases = [
        ("api-raises-falls-back-to-scan",
         _tv_module(list_models=_raising_list_models, tiny_net=lambda **kw: None,
                    _private_net=lambda **kw: None, NOT_CALLABLE=42),
         {"tiny_net", "list_models"}),
        ("api-answers-is-trusted",
         _tv_module(list_models=lambda module=None: ["from_api"],
                    from_dict=lambda **kw: None),
         {"from_api"}),
        ("no-api-keeps-factories-not-classes-or-enums",
         _tv_module(resnet18=lambda **kw: None, ResNet=type("ResNet", (), {}),
                    ResNet18_Weights=type("ResNet18_Weights", (), {})),
         {"resnet18"}),
    ]
    for cid, fake, expected in cases:
        with subtests.test(case=cid):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(U, "tv_models", fake)
                assert U._list_torchvision_model_names() == expected

    with subtests.test(case="real-torchvision"):
        names = U._list_torchvision_model_names()
        assert isinstance(names, set) and names
        assert {"resnet18", "resnet50"} <= names
        assert not [n for n in names if n.endswith("_Weights")]


def _fake_torch_model(z, record):
    """Return a TorchModel stand-in whose forward yields ``z``."""

    class _FakeTorchModel:
        def __init__(self, **kwargs):
            record.update(kwargs)

        def eval(self):
            record["eval_called"] = True
            return self

        def __call__(self, x):
            record["forward_shape"] = tuple(x.shape)
            return z

        def __repr__(self):
            return "<FakeTorchModel repr marker>"

    return _FakeTorchModel


def test_choose_model_builds_the_named_model_and_refuses_broken_ones(subtests, capsys):
    """Unknown names (and near misses) are raised by name; 'custom' builds the
    custom classifier; constructor kwargs are derived from the call; a backbone
    whose probe forward is a dict, a non-tensor or the wrong width is refused;
    a real resnet18 binary head yields (1, 1) logits."""
    cpu = torch.device("cpu")
    for cid, name, match in [
        ("unknown-name", "definitely_not_a_torchvision_model",
         "definitely_not_a_torchvision_model"),
        ("near-miss-is-named", "resnet_18", "resnet18"),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError, match=match):
                U.choose_model(name, cpu)

    with subtests.test(case="custom"):
        assert isinstance(U.choose_model("custom", cpu, height=32, width=32),
                          U.CustomCellClassifier)

    refusals = [
        ("dict-output", {"logits": torch.zeros(1, 2)}, "dict, not logits"),
        ("wrong-logit-width", torch.zeros(1, 5), r"Expected logits of shape \(1,2\)"),
        ("non-tensor-output", [0.1, 0.9], r"Expected logits of shape \(1,2\)"),
    ]
    for cid, z, match in refusals:
        with subtests.test(case=cid):
            record = {}
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(U, "TorchModel", _fake_torch_model(z, record))
                with pytest.raises(ValueError, match=match):
                    U.choose_model("resnet18", cpu, init_weights=False,
                                   num_classes=2, height=16, width=16)
            assert record["eval_called"] is True
            assert record["forward_shape"] == (1, 3, 16, 16)

    builds = [
        # (id, z, call kwargs, expected record subset, expect printed repr)
        ("verbose-and-kwargs", torch.zeros(1, 3),
         dict(init_weights=False, dropout_rate=0.0, use_checkpoint=True,
              num_classes=3, height=64, width=64, verbose=True),
         dict(model_name="resnet18", pretrained=False, dropout_rate=None,
              use_checkpoint=True, num_classes=3, image_size=64,
              forward_shape=(1, 3, 64, 64)), True),
        ("falsy-height-defaults-to-224", torch.zeros(1, 1),
         dict(init_weights=False, num_classes=0, height=0, width=0),
         dict(image_size=224, forward_shape=(1, 3, 224, 224), num_classes=1), False),
        ("positive-dropout-passes-through", torch.zeros(1, 2),
         dict(init_weights=True, dropout_rate=0.25, num_classes=2, height=16,
              width=16),
         dict(dropout_rate=0.25, pretrained=True), False),
    ]
    for cid, z, kw, expected, printed in builds:
        with subtests.test(case=cid):
            record = {}
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(U, "TorchModel", _fake_torch_model(z, record))
                model = U.choose_model("resnet18", cpu, **kw)
            assert repr(model) == "<FakeTorchModel repr marker>"
            for k, v in expected.items():
                if v is None or isinstance(v, bool):
                    assert record[k] is v, k
                else:
                    assert record[k] == v, k
            assert ("<FakeTorchModel repr marker>" in capsys.readouterr().out) is printed

    with subtests.test(case="real-resnet18-binary-head"):
        model = U.choose_model("resnet18", cpu, init_weights=False,
                               dropout_rate=0.0, num_classes=1, height=32, width=32)
        model.eval()
        with torch.no_grad():
            assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 1)


# ===========================================================================
# calculate_loss
# ===========================================================================

def _ref_focal_bce_none(logits, y, alpha, gamma):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    return alpha * (1 - pt).pow(gamma) * ce


def _ref_focal_ce_none(logits, idx, alpha, gamma):
    log_p = F.log_softmax(logits, dim=1)
    log_pt = log_p.gather(1, idx.view(-1, 1)).squeeze(1)
    return -alpha * (1 - log_pt.exp()).pow(gamma) * log_pt


def test_calculate_loss_routes_each_target_shape_to_the_right_loss(subtests):
    """Binary (N,1)/(N,) logits use BCE, long (N,) targets use CE, any other
    1-D target is one-hot encoded and (N,C) targets use multilabel BCE; plain
    and focal variants match closed-form references, 'sum'/'mean' reduce the
    per-sample tensor and an unknown focal reduction returns it unreduced."""
    g = _gen(0)
    rand8_1 = torch.randn(8, 1, generator=g)
    rand8 = torch.randn(8, generator=g)
    rand8_4 = torch.randn(8, 4, generator=g)
    bin8 = torch.randint(0, 2, (8,), generator=g).float()
    idx8 = torch.randint(0, 4, (8,), generator=g)
    ml8 = torch.randint(0, 2, (8, 4), generator=g).float()
    oh = lambda t, c: F.one_hot(t.long(), num_classes=c).float()  # noqa: E731

    cases = [
        # (id, logits, target, route, reference target, per-sample shape)
        ("binary", torch.tensor([[2.5], [-3.0], [0.4], [-0.2]]),
         torch.tensor([1.0, 0.0, 1.0, 0.0]), "bce", None, (4, 1)),
        ("binary-random", rand8_1, bin8, "bce", None, (8, 1)),
        ("binary-1d-logits-unsqueezed", rand8, bin8, "bce", None, (8, 1)),
        ("binary-small", torch.tensor([[0.7], [-0.7]]), torch.tensor([1.0, 0.0]),
         "bce", None, (2, 1)),
        ("multiclass", torch.tensor([[3.0, 0.0, -1.0], [0.1, 0.2, 0.3],
                                     [-2.0, 4.0, 0.0]]),
         torch.tensor([0, 2, 1]), "ce", None, (3,)),
        ("multiclass-2", torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
         torch.tensor([0, 1]), "ce", None, (2,)),
        ("multiclass-random", rand8_4, idx8, "ce", None, (8,)),
        ("1d-float-is-one-hot",
         torch.tensor([[2.0, -1.0, 0.5, 0.0], [0.0, 1.0, -2.0, 0.3],
                       [-1.0, 0.2, 0.4, 3.0]]),
         torch.tensor([0.0, 1.0, 3.0]), "ml", oh(torch.tensor([0, 1, 3]), 4), (3, 4)),
        ("1d-int32-is-one-hot", torch.zeros(2, 3),
         torch.tensor([1, 2], dtype=torch.int32), "ml",
         oh(torch.tensor([1, 2]), 3), (2, 3)),
        ("1d-float-two-classes", torch.tensor([[1.0, -1.0], [0.5, 0.5]]),
         torch.tensor([0.0, 1.0]), "ml", oh(torch.tensor([0, 1]), 2), (2, 2)),
        ("multilabel-2d", rand8_4, ml8, "ml", ml8, (8, 4)),
    ]
    for cid, logits, target, route, ref_y, shape in cases:
        out2 = logits.unsqueeze(1) if logits.ndim == 1 else logits
        if route == "bce":
            y = target.float().view(-1, 1)
            plain_ref = F.binary_cross_entropy_with_logits(out2, y, reduction="none")
            focal_ref = _ref_focal_bce_none(out2, y, 1.0, 2.0)
        elif route == "ce":
            plain_ref = F.cross_entropy(out2, target, reduction="none")
            focal_ref = _ref_focal_ce_none(out2, target, 1.0, 2.0)
        else:
            plain_ref = F.binary_cross_entropy_with_logits(out2, ref_y, reduction="none")
            focal_ref = _ref_focal_bce_none(out2, ref_y, 1.0, 2.0)
        for focal, ref in ((False, plain_ref), (True, focal_ref)):
            with subtests.test(case=cid, focal=focal):
                none = U.calculate_loss(logits, target, prefer_focal=focal,
                                        reduction="none")
                assert tuple(none.shape) == shape
                assert torch.all(none >= 0)
                assert torch.allclose(none, ref, atol=1e-6)
                total = U.calculate_loss(logits, target, prefer_focal=focal,
                                         reduction="sum")
                mean = U.calculate_loss(logits, target, prefer_focal=focal)
                assert total.ndim == 0 and mean.ndim == 0
                assert float(total) == pytest.approx(float(ref.sum()), rel=1e-5, abs=1e-7)
                assert float(mean) == pytest.approx(float(ref.mean()), rel=1e-5, abs=1e-7)
                if focal:
                    for odd in ("batchmean", "elementwise"):
                        weird = U.calculate_loss(logits, target, prefer_focal=True,
                                                 reduction=odd)
                        assert tuple(weird.shape) == shape
                        assert torch.allclose(weird, none)
                    plain_sum = U.calculate_loss(logits, target, reduction="sum")
                    assert float(total) < float(plain_sum)
                    if route == "ml" and target.ndim == 1:
                        # same answer as handing the one-hot matrix in directly
                        direct = U.calculate_loss(logits, ref_y, prefer_focal=True,
                                                  reduction="sum")
                        assert float(total) == pytest.approx(float(direct), rel=1e-6)

    with subtests.test(case="focal-binary-alpha-scales-linearly"):
        logits = torch.tensor([[1.0], [-1.0]])
        target = torch.tensor([1.0, 0.0])
        one = U.calculate_loss(logits, target, prefer_focal=True, alpha=1.0,
                               reduction="sum")
        two = U.calculate_loss(logits, target, prefer_focal=True, alpha=2.0,
                               reduction="sum")
        assert float(two) == pytest.approx(2 * float(one), rel=1e-6)

    with subtests.test(case="1d-float-target-is-not-cross-entropy"):
        logits, target = cases[7][1], cases[7][2]
        ce = F.cross_entropy(logits, target.long())
        assert float(U.calculate_loss(logits, target)) != pytest.approx(float(ce),
                                                                       rel=1e-3)


# ===========================================================================
# compute_irm_penalty
# ===========================================================================

def test_compute_irm_penalty_sums_squared_pairwise_gradient_products(subtests):
    """d(loss*w)/dw == loss, so the penalty is the sum over environment pairs
    of (l_i*l_j)^2 and stays differentiable; fewer than two environments give
    the plain float 0.0."""
    cpu = torch.device("cpu")
    cases = [
        ("three-envs", [0.5, 2.0, 1.5],
         (0.5 * 2.0) ** 2 + (0.5 * 1.5) ** 2 + (2.0 * 1.5) ** 2),
        ("one-env", [3.0], 0.0),
        ("no-envs", [], 0.0),
    ]
    for cid, losses, expected in cases:
        with subtests.test(case=cid):
            w = torch.tensor([1.0], requires_grad=True)
            penalty = U.compute_irm_penalty([torch.tensor(v) for v in losses], w, cpu)
            if len(losses) < 2:
                assert isinstance(penalty, float) and penalty == 0.0
            else:
                assert penalty.requires_grad
                assert float(penalty.detach()) == pytest.approx(expected)


# ===========================================================================
# estimate_class_counts
# ===========================================================================

class _ListLoader:
    """Minimal stand-in for a DataLoader yielding (images, labels, paths)."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_estimate_class_counts_from_folders_or_by_iterating_the_loader(
        subtests, tmp_path, capsys):
    """With src+classes, files per class folder are counted (subdirs, missing
    dirs and the crop-format sidecar excluded); otherwise the loader is
    iterated with a warning and index / one-hot / binary-float labels are
    bincounted, dropping labels >= num_classes."""
    from spacr.utils import estimate_class_counts

    with subtests.test(case="folders"):
        src = tmp_path / "train"
        (src / "nc").mkdir(parents=True)
        (src / "pc").mkdir(parents=True)
        for i in range(3):
            (src / "nc" / f"a{i}.png").write_bytes(b"x")
        for i in range(5):
            (src / "pc" / f"b{i}.png").write_bytes(b"x")
        (src / "pc" / "nested").mkdir()
        (src / "pc" / "nested" / "c.png").write_bytes(b"x")
        capsys.readouterr()
        counts = estimate_class_counts(None, 3, src=str(src),
                                       classes=["nc", "pc", "absent"])
        assert counts.dtype == torch.long and counts.tolist() == [3, 5, 0]
        assert "Class counts (from folders)" in capsys.readouterr().out

    with subtests.test(case="crop-format-sidecar-is-not-a-sample"):
        from PIL import Image
        from spacr.io import generate_dataset_from_lists
        source = tmp_path / "crops"
        source.mkdir()
        class_data = [[], []]
        for class_index, row in enumerate(("A", "B")):
            for well in range(1, 5):
                for obj in range(3):
                    path = source / f"plate1_{row}{well:02d}_1_{obj}.png"
                    Image.fromarray(np.full((4, 4, 3), class_index * 100,
                                            dtype=np.uint8)).save(path)
                    class_data[class_index].append(str(path))
        generate_dataset_from_lists(str(tmp_path / "training"), class_data,
                                    classes=["c1", "c2"], test_split=0.25)
        train_root = str(tmp_path / "training" / "train")
        assert os.path.isfile(os.path.join(train_root, "c1", ".spacr_crop_format.json"))
        pngs = [len([f for f in os.listdir(os.path.join(train_root, c))
                     if f.endswith(".png")]) for c in ("c1", "c2")]
        counts = estimate_class_counts(None, 2, src=train_root, classes=["c1", "c2"])
        assert counts.tolist() == pngs

    loader_cases = [
        # (id, batches, num_classes, kwargs, expected)
        ("index-targets",
         [(torch.zeros(4, 1, 2, 2), torch.tensor([0, 1, 1, 2]), ["p"] * 4),
          (torch.zeros(3, 1, 2, 2), torch.tensor([2, 2, 0]), ["p"] * 3)],
         3, {}, [2, 2, 3]),
        ("one-hot-and-binary-float",
         [(torch.zeros(3), torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), ["p"] * 3),
          (torch.zeros(4), torch.tensor([0.1, 0.9, 0.8, 0.2]), ["p"] * 4)],
         2, {}, [3, 4]),
        ("out-of-range-labels-dropped",
         [(torch.zeros(4), torch.tensor([0, 1, 2, 3]), ["p"] * 4)], 2, {}, [1, 1]),
        ("src-without-classes-iterates",
         [(torch.zeros(2), torch.tensor([1, 1]), ["p"] * 2)], 2,
         {"src": str(tmp_path / "nonexistent"), "classes": None}, [0, 2]),
    ]
    for cid, batches, ncls, kw, expected in loader_cases:
        with subtests.test(case=cid):
            capsys.readouterr()
            counts = estimate_class_counts(_ListLoader(batches), ncls, **kw)
            assert counts.dtype == torch.long
            assert tuple(counts.shape) == (ncls,)
            assert counts.tolist() == expected
            assert "iterating DataLoader" in capsys.readouterr().out


# ===========================================================================
# pick_best_model
# ===========================================================================

def test_pick_best_model_ranks_checkpoints_by_role_then_accuracy(subtests, tmp_path):
    """Tagged checkpoints rank by accuracy before epoch, non-.pth files and
    untagged names sort last (a .pth is still returned when nothing parses),
    and stored artifacts rank a milestone above an unscored checkpoint."""
    cases = [
        ("highest-accuracy", ["m_epoch_1_acc_0.80.pth", "m_epoch_5_acc_0.95.pth",
                              "m_epoch_3_acc_0.90.pth", "unrelated.txt"],
         {"m_epoch_5_acc_0.95.pth"}),
        ("accuracy-beats-epoch", ["m_epoch_99_acc_0.10.pth", "m_epoch_1_acc_0.99.pth"],
         {"m_epoch_1_acc_0.99.pth"}),
        ("untagged-sort-last", ["no_tag_at_all.pth", "another_untagged.pth",
                                "m_epoch_2_acc_0.60.pth", "notes.txt"],
         {"m_epoch_2_acc_0.60.pth"}),
        ("nothing-parses-still-a-pth", ["alpha.pth", "beta.pth", "ignore.csv"],
         {"alpha.pth", "beta.pth"}),
    ]
    for cid, names, allowed in cases:
        with subtests.test(case=cid):
            d = tmp_path / cid
            d.mkdir()
            for name in names:
                (d / name).write_bytes(b"")
            best = U.pick_best_model(str(d))
            assert os.path.dirname(best) == str(d)
            assert os.path.basename(best) in allowed

    with subtests.test(case="stored-role-and-missing-accuracy"):
        d = tmp_path / "artifacts"
        d.mkdir()
        milestone = d / "model_epoch_3.pth"
        torch.save({"artifact_role": "milestone", "metrics": {"accuracy": 0.91},
                    "training_state": {"epoch": 3}}, milestone)
        torch.save({"artifact_role": "checkpoint", "metrics": {"accuracy": None},
                    "training_state": {"epoch": 7}}, d / "model_epoch_7.pth")
        assert U.pick_best_model(str(d)) == str(milestone)


# ===========================================================================
# build_loss
# ===========================================================================

def _mc(C=3, n=8, seed=0):
    g = _gen(seed)
    return torch.randn(n, C, generator=g), torch.randint(0, C, (n,), generator=g)


def _bin(n=8, seed=0):
    return (torch.randn(n, 1, generator=_gen(seed)),
            torch.tensor([0., 1., 1., 0., 1., 0., 0., 1.])[:n])


def _inv_freq(counts):
    priors = counts.float() / counts.float().sum()
    inv = 1.0 / priors
    return inv / inv.mean()


def test_build_loss_matches_a_closed_form_reference_for_every_loss_type(subtests):
    """Every loss name (and alias, any case, None -> ce, 'auto' by class
    balance) computes exactly its reference: (weighted / smoothed) CE, BCE,
    focal BCE/CE with scalar / per-class / one-element alpha, the Menon
    +tau*log(prior) logit adjustment, and ASL's BCE degenerate case; one-hot,
    index and flattened targets give identical values."""
    from spacr.utils import build_loss

    lg, idx = _mc()
    oh = F.one_hot(idx, num_classes=3).float()
    bl, bt = _bin()
    by = bt.view(-1, 1)
    bce_none = F.binary_cross_entropy_with_logits(bl, by, reduction="none")
    lg2, idx2 = _mc(C=2, seed=1)
    lg6 = torch.randn(6, 3, generator=_gen(2))
    y6 = torch.tensor([0, 1, 2, 0, 1, 2])
    c901 = torch.tensor([90, 9, 1])
    ce_none = F.cross_entropy(lg, idx, reduction="none")
    pri901 = c901.float() / c901.sum()
    a3 = torch.tensor([0.1, 0.5, 0.4])
    ai = torch.tensor([1, 2, 3])
    c = torch.tensor([10.0, 30.0, 60.0])

    cases = [
        # (id, loss_type, kwargs, logits, target, expected)
        ("ce", "ce", {}, lg, idx, F.cross_entropy(lg, idx)),
        ("ce-alias-cross_entropy", "cross_entropy", {}, lg, idx, F.cross_entropy(lg, idx)),
        ("ce-one-hot", "ce", {}, lg, oh, F.cross_entropy(lg, idx)),
        ("none-defaults-to-ce", None, {}, lg, idx, F.cross_entropy(lg, idx)),
        ("ce-with-counts-is-weighted", "ce", {"class_counts": c}, lg, idx,
         F.cross_entropy(lg, idx, weight=_inv_freq(c))),
        ("ce_smooth-one-hot", "ce_smooth", {"label_smoothing": 0.2}, lg, oh,
         F.cross_entropy(lg, idx, label_smoothing=0.2)),
        ("ce_smooth-case-insensitive", "CE_Smooth", {"label_smoothing": 0.1}, lg, idx,
         F.cross_entropy(lg, idx, label_smoothing=0.1)),
        ("label_smoothing-alias", "label_smoothing", {"label_smoothing": 0.1}, lg, idx,
         F.cross_entropy(lg, idx, label_smoothing=0.1)),
        ("ce_weighted", "ce_weighted", {"class_counts": c}, lg, idx,
         F.cross_entropy(lg, idx, weight=_inv_freq(c))),
        ("ce_weighted-zero-count-clamped", "ce_weighted",
         {"class_counts": torch.tensor([10, 0, 90])}, lg, idx,
         F.cross_entropy(lg, idx, weight=_inv_freq(torch.tensor([10.0, 1.0, 90.0])))),
        ("bce", "bce", {}, bl, bt, F.binary_cross_entropy_with_logits(bl, by)),
        ("bce-long-name", "binary_cross_entropy_with_logits", {}, bl, bt,
         F.binary_cross_entropy_with_logits(bl, by)),
        ("auto-binary-is-bce", "auto", {}, bl, bt, F.binary_cross_entropy_with_logits(bl, by)),
        ("focal_bce-gamma0-no-alpha", "focal_bce", {"focal_gamma": 0.0}, bl, bt,
         F.binary_cross_entropy_with_logits(bl, by)),
        ("focal_bce-alpha", "focal_bce", {"focal_gamma": 0.0, "focal_alpha": 0.25},
         bl, bt, ((0.25 * by + 0.75 * (1 - by)) * bce_none).mean()),
        ("focal_bce-alpha1-silences-negatives", "focal_bce",
         {"focal_gamma": 0.0, "focal_alpha": 1.0}, bl, bt, (by * bce_none).mean()),
        ("focal-alias-binary", "focal", {"focal_gamma": 0.0}, bl, bt,
         F.binary_cross_entropy_with_logits(bl, by)),
        ("focal_ce-gamma0", "focal_ce", {"focal_gamma": 0.0}, lg, idx,
         F.cross_entropy(lg, idx)),
        ("focal_loss-alias", "focal_loss", {"focal_gamma": 0.0}, lg, idx,
         F.cross_entropy(lg, idx)),
        ("focal_ce-scalar-alpha", "focal_ce", {"focal_gamma": 0.0, "focal_alpha": 0.25},
         lg, idx, 0.25 * F.cross_entropy(lg, idx)),
        ("focal_ce-one-element-alpha", "focal_ce",
         {"focal_gamma": 0.0, "focal_alpha": torch.tensor([0.3])}, lg, idx,
         0.3 * F.cross_entropy(lg, idx)),
        ("focal_ce-per-class-alpha", "focal_ce", {"focal_gamma": 0.0, "focal_alpha": a3},
         lg, idx, (a3[idx] * ce_none).mean()),
        ("focal_ce-int-alpha-cast", "focal_ce", {"focal_gamma": 0.0, "focal_alpha": ai},
         lg, idx, (ai.float()[idx] * ce_none).mean()),
        ("focal_ce-per-class-alpha-one-hot", "focal_ce",
         {"focal_gamma": 0.0, "focal_alpha": a3}, lg, oh, (a3[idx] * ce_none).mean()),
        ("logit_adjust-tau0-is-ce", "logit_adjust_ce",
         {"class_counts": c901, "logit_adjust_tau": 0.0}, lg6, y6, F.cross_entropy(lg6, y6)),
        ("logit_adjust-menon", "logit_adjust_ce",
         {"class_counts": c901, "logit_adjust_tau": 1.0}, lg6, y6,
         F.cross_entropy(lg6 + pri901.log(), y6)),
        ("la_ce-menon", "la_ce", {"class_counts": c901, "logit_adjust_tau": 1.0},
         lg6, y6, F.cross_entropy(lg6 + pri901.log(), y6)),
        ("logit_adjust-one-hot", "logit_adjust_ce",
         {"class_counts": c901, "logit_adjust_tau": 0.5}, lg6,
         F.one_hot(y6, 3).float(), F.cross_entropy(lg6 + 0.5 * pri901.log(), y6)),
        ("logit_adjust-zero-logits-exact", "logit_adjust_ce",
         {"class_counts": torch.tensor([950, 50]), "logit_adjust_tau": 1.0},
         torch.zeros(4, 2), torch.tensor([0, 0, 0, 1]),
         torch.tensor((3 * -np.log(0.95) - np.log(0.05)) / 4, dtype=torch.float32)),
        ("auto-rare-class-is-logit-adjust", "auto",
         {"class_counts": c901, "logit_adjust_tau": 1.0}, lg6, y6,
         F.cross_entropy(lg6 + pri901.log(), y6)),
        ("auto-rare-class-tau0-is-unweighted-ce", "auto",
         {"class_counts": torch.tensor([950, 50]), "logit_adjust_tau": 0.0}, lg2, idx2,
         F.cross_entropy(lg2, idx2)),
        ("auto-mild-imbalance-is-weighted-ce", "auto",
         {"class_counts": torch.tensor([700, 300])}, lg2, idx2,
         F.cross_entropy(lg2, idx2, weight=_inv_freq(torch.tensor([700, 300])))),
        ("auto-no-counts-is-ce", "auto", {}, lg, idx, F.cross_entropy(lg, idx)),
        ("auto-all-zero-counts-finite", "auto", {"class_counts": torch.tensor([0, 0])},
         lg2, idx2, F.cross_entropy(lg2, idx2, weight=torch.ones(2))),
        ("asl-degenerates-to-bce", "asl",
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 0.0, "asl_clip": 0.0}, lg, idx,
         F.binary_cross_entropy_with_logits(lg, oh)),
        ("asl-degenerate-one-hot", "asl",
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 0.0, "asl_clip": 0.0}, lg6,
         F.one_hot(y6, 3).float(),
         F.binary_cross_entropy_with_logits(lg6, F.one_hot(y6, 3).float())),
    ]
    for cid, lt, kw, logits, target, expected in cases:
        with subtests.test(case=cid):
            got = build_loss(lt, num_classes=logits.shape[1] if logits.shape[1] > 1 else 1,
                             **kw)(logits, target)
            assert got.ndim == 0 and torch.isfinite(got)
            assert got.dtype == torch.float32
            assert torch.allclose(got, expected.to(torch.float32), atol=1e-5)

    with subtests.test(case="logit_adjust-reference-value"):
        expected = F.cross_entropy(torch.zeros(4, 2) + torch.tensor([0.95, 0.05]).log(),
                                   torch.tensor([0, 0, 0, 1]))
        assert float(expected) == pytest.approx(0.7874, abs=1e-3)

    # Every target encoding reaches the same value.
    same = [
        ("asl-index-vs-one-hot", "asl", {}, lg6, y6, F.one_hot(y6, 3).float()),
        ("asymmetric_loss-index-vs-one-hot", "asymmetric_loss", {}, lg6, y6,
         F.one_hot(y6, 3).float()),
        ("asl-flattened-target", "asl", {}, lg6, F.one_hot(y6, 3).float().reshape(1, -1),
         F.one_hot(y6, 3).float()),
        ("focal_ce-alpha-one-hot", "focal_ce", {"focal_alpha": torch.tensor([1.0, 0.5, 0.25])},
         lg6, F.one_hot(y6, 3).float(), y6),
    ]
    for cid, lt, kw, logits, t1, t2 in same:
        with subtests.test(case=cid):
            fn = build_loss(lt, num_classes=3, **kw)
            a, b = fn(logits, t1), fn(logits, t2)
            assert float(a) > 0.0
            assert float(a) == pytest.approx(float(b), rel=1e-6)

    with subtests.test(case="asymmetric_loss-alias-equals-asl"):
        assert torch.allclose(build_loss("asymmetric_loss", num_classes=3)(lg, idx),
                              build_loss("asl", num_classes=3)(lg, idx), atol=1e-6)


def test_build_loss_shapes_and_weights_each_loss_as_documented(subtests):
    """Focusing (gamma), clipping and alpha change the loss in the documented
    direction: easy examples / negatives are down-weighted, clip shrinks the
    negative term, a per-class alpha zeroes classes it gives 0, a float64
    per-class alpha is cast to float32 without mutating the caller's tensor
    (a wrong-length one is left alone), and logit adjustment is not class
    weighting."""
    from spacr.utils import build_loss

    lg6 = torch.randn(6, 3, generator=_gen(2))
    y6 = torch.tensor([0, 1, 2, 0, 1, 2])
    lg, idx = _mc()

    smaller = [
        # (id, (loss_type, num_classes, kw_small), (kw_large), logits, target)
        ("focal_bce-gamma-downweights-easy", "focal_bce", 1, {"focal_gamma": 2.0},
         {"focal_gamma": 0.0}, torch.tensor([[6.0], [-6.0], [0.2]]),
         torch.tensor([1.0, 0.0, 1.0])),
        ("focal_ce-gamma-downweights-confident", "focal_ce", 3, {"focal_gamma": 2.0},
         {"focal_gamma": 0.0}, torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]),
         torch.tensor([0, 1])),
        ("asl-clip-shrinks-negatives", "asl", 3,
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 0.0, "asl_clip": 0.05},
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 0.0, "asl_clip": 0.0}, lg, idx),
        ("asl-clip-shrinks-default-gammas", "asl", 3, {"asl_clip": 0.05},
         {"asl_clip": 0.0}, lg6, y6),
        ("asl-negative-focusing", "asl", 3,
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 4.0, "asl_clip": 0.0},
         {"asl_gamma_pos": 0.0, "asl_gamma_neg": 0.0, "asl_clip": 0.0},
         torch.tensor([[4.0, -6.0, -6.0]]), torch.tensor([0])),
    ]
    for cid, lt, ncls, kw_small, kw_large, logits, target in smaller:
        with subtests.test(case=cid):
            small = build_loss(lt, num_classes=ncls, **kw_small)(logits, target)
            large = build_loss(lt, num_classes=ncls, **kw_large)(logits, target)
            assert 0.0 <= float(small) < float(large)

    with subtests.test(case="asl-rewards-confident-correct"):
        fn = build_loss("asl", num_classes=3)
        good = F.one_hot(y6, 3).float() * 12.0 - 6.0
        assert float(fn(good, y6)) < 1e-2
        assert float(fn(good, y6)) < float(fn(-good, y6))
        assert torch.isfinite(fn(-good, y6))

    with subtests.test(case="focal_ce-alpha-is-label-dependent"):
        a3 = torch.tensor([0.1, 0.5, 0.4])
        got = build_loss("focal_ce", num_classes=3, focal_gamma=0.0, focal_alpha=a3)(lg, idx)
        assert not torch.allclose(got, a3.mean() * F.cross_entropy(lg, idx), atol=1e-4)

    with subtests.test(case="focal_ce-alpha-zero-classes"):
        weighted = build_loss("focal_ce", num_classes=3,
                              focal_alpha=torch.tensor([1.0, 0.0, 0.0]))
        unweighted = build_loss("focal_ce", num_classes=3, focal_alpha=None)
        assert float(weighted(lg6, torch.ones(6, dtype=torch.long))) == 0.0
        zeros = torch.zeros(6, dtype=torch.long)
        assert float(weighted(lg6, zeros)) == pytest.approx(float(unweighted(lg6, zeros)),
                                                            rel=1e-6)

    for cid, alpha, out_dtype in [
        ("float64-per-class-alpha-cast", [1.0, 0.5, 0.25], torch.float32),
        ("float64-wrong-length-alpha-left-alone", [1.0, 0.5, 0.25, 0.125], torch.float64),
    ]:
        with subtests.test(case=cid):
            a = torch.tensor(alpha, dtype=torch.float64)
            loss = build_loss("focal_ce", num_classes=3, focal_alpha=a,
                              focal_gamma=2.0)(lg6, y6)
            assert loss.dtype == out_dtype and loss.ndim == 0 and float(loss) > 0.0
            assert a.dtype == torch.float64
            ref = build_loss("focal_ce", num_classes=3,
                             focal_alpha=torch.tensor([1.0, 0.5, 0.25]),
                             focal_gamma=2.0)(lg6, y6)
            assert float(loss) == pytest.approx(float(ref), rel=1e-6)

    c901 = torch.tensor([90, 9, 1])
    unadjusted = F.cross_entropy(lg6, y6)
    for cid, lt, kw, differs_from in [
        ("logit_adjust-tau0-is-not-weighted-ce", "logit_adjust_ce",
         {"class_counts": c901, "logit_adjust_tau": 0.0},
         build_loss("ce_weighted", num_classes=3, class_counts=c901)(lg6, y6)),
        ("logit_adjust-tau1-moves-the-loss", "logit_adjust_ce",
         {"class_counts": c901, "logit_adjust_tau": 1.0}, unadjusted),
        ("la_ce-tau1-moves-the-loss", "la_ce",
         {"class_counts": c901, "logit_adjust_tau": 1.0}, unadjusted),
        ("auto-rare-tau0-is-not-weighted-ce", "auto",
         {"class_counts": c901, "logit_adjust_tau": 0.0},
         build_loss("ce", num_classes=3, class_counts=c901)(lg6, y6)),
        ("ce_smooth-differs-from-ce", "ce_smooth", {"label_smoothing": 0.2}, unadjusted),
        ("auto-mild-is-not-plain-ce", "auto",
         {"class_counts": torch.tensor([60, 25, 15])}, unadjusted),
    ]:
        with subtests.test(case=cid):
            got = build_loss(lt, num_classes=3, **kw)(lg6, y6)
            assert float(got) != pytest.approx(float(differs_from), rel=1e-3)


def _force_cpu(mp):
    import spacr.accelerator as accelerator
    mp.setattr(accelerator, "_CACHED", None)
    mp.setattr(torch.cuda, "is_available", lambda: False)


def _onehot_loaders(n_batches=2, batch=4, n_feat=4, n_classes=2, seed=0):
    g = _gen(seed)
    out = []
    for b in range(n_batches):
        x = torch.rand(batch, n_feat, generator=g)
        idx = torch.tensor([i % n_classes for i in range(batch)], dtype=torch.long)
        y = F.one_hot(idx, num_classes=n_classes).float()
        out.append((x, y, [f"b{b}_{i}.png" for i in range(batch)]))
    return out


def test_build_loss_refuses_invalid_requests_and_backs_every_offered_option(
        subtests, tmp_path):
    """Unknown names, multiclass names on a binary head and count-based losses
    without counts are refused by message; every builder back-propagates
    finite gradients; every loss_type the GUI offers builds a callable; and
    train_model feeds one-hot targets to the loss as (N,) indices."""
    from spacr.utils import build_loss

    refusals = [("unknown", "nope", 3, "Unknown loss_type"),
                ("unknown-2", "bogus_loss", 3, "Unknown loss_type 'bogus_loss'"),
                ("ce_weighted-needs-counts", "ce_weighted", 3, "requires class_counts"),
                ("logit_adjust-needs-counts", "logit_adjust_ce", 3, "requires class_counts"),
                ("la_ce-needs-counts", "la_ce", 3, "requires class_counts")]
    refusals += [(f"binary-refuses-{lt}", lt, 1, "not valid for binary")
                 for lt in ("ce", "ce_weighted", "asl", "logit_adjust_ce")]
    for cid, lt, ncls, match in refusals:
        with subtests.test(case=cid):
            with pytest.raises(ValueError, match=match):
                build_loss(lt, num_classes=ncls)

    counts = torch.tensor([10, 40, 50])
    for lt in ("ce", "ce_smooth", "ce_weighted", "focal_ce", "logit_adjust_ce",
               "asl", "auto"):
        with subtests.test(case=f"differentiable-{lt}"):
            logits = torch.randn(6, 3, generator=_gen(3)).requires_grad_(True)
            y = torch.randint(0, 3, (6,), generator=_gen(4))
            loss = build_loss(lt, num_classes=3, class_counts=counts, focal_alpha=0.5,
                              logit_adjust_tau=1.0, label_smoothing=0.1)(logits, y)
            assert loss.ndim == 0 and torch.isfinite(loss)
            loss.backward()
            assert logits.grad is not None and logits.grad.shape == logits.shape
            assert torch.isfinite(logits.grad).all()
            assert float(logits.grad.abs().sum()) > 0.0

    with subtests.test(case="gui-options-all-build"):
        from spacr import settings_spec as GU
        specs = GU.convert_settings_dict_for_gui({"optimizer_type": "adamw",
                                                  "loss_type": "auto"})
        _, opt_options, _ = specs["optimizer_type"]
        assert set(opt_options) == {"adamw", "adam", "adamax", "adagrad", "adadelta",
                                    "asgd", "sgd", "rmsprop", "nadam", "radam"}
        _, loss_options, _ = specs["loss_type"]
        assert loss_options
        c2 = torch.tensor([80.0, 20.0])
        for name in loss_options:
            n_classes = 1 if name == "binary_cross_entropy_with_logits" else 2
            fn = build_loss(name, num_classes=n_classes, class_counts=c2)
            assert callable(fn), name

    with subtests.test(case="train-model-argmaxes-one-hot-targets"):
        from spacr.deep_spacr import train_model
        seen = []
        real_build_loss = U.build_loss

        def spy_build_loss(**kwargs):
            fn = real_build_loss(**kwargs)

            def wrapped(logits, target):
                seen.append(tuple(target.shape))
                return fn(logits, target)
            return wrapped

        with pytest.MonkeyPatch.context() as mp:
            _force_cpu(mp)
            mp.setattr("spacr.utils.build_loss", spy_build_loss)
            mp.setattr("spacr.utils.choose_model",
                       lambda *a, **k: nn.Sequential(nn.Linear(4, 2)))
            mp.setattr("spacr.io._save_progress", lambda *a, **k: None)
            dst = tmp_path / "model"
            dst.mkdir()
            _trained, path = train_model(str(tmp_path), str(dst), "resnet18",
                                         _onehot_loaders(2), epochs=1,
                                         num_classes=2, schedule=None)
        assert len(seen) >= 2
        assert all(shape == (4,) for shape in seen), seen
        assert os.path.exists(path)
