"""Item 897: the channel-adaptive CNN and ViT, the well transformer, and their registry.

The first 25 tests port the reference's ``smoke_tests.py`` for
``spacr_arch.py``: forward and backward at 1 to 12 channels, missing
channels, channel dropout and test-time channel subsets, the MAE and DINO
losses, and the arrayed and pooled well transformer (padding and permutation
invariance, target independence, context sensitivity). The rest cover the
registry (``choose_model`` and ``build_channel_model``) and one training step
of each architecture on tiny CPU tensors.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spacr import channel_models as cm
from spacr.channel_models import (ChannelDINO, ChannelMAE, SpacrChannelCNN, SpacrChannelViT,
                                  SpacrWellTransformer, count_params)

STAINS = ["dna", "tubulin", "gh2ax", "actin", "er", "rna", "agp", "mito", "parasite", "gbp1", "ubiquitin",
          "my_new_marker"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _fb(model, x, stains, **kw):
    model.train()
    y = model.classify(x, stains, **kw)
    y.sum().backward()
    assert torch.isfinite(y).all()
    return y


@pytest.mark.parametrize("C", [1, 3, 4, 5, 12])
def test_cnn_forward_backward_at_any_channel_count(C):
    m = SpacrChannelCNN(stains=STAINS[:C], num_classes=7, dims=(8, 16, 32, 64), depths=(1, 1, 1, 1))
    y = _fb(m, torch.randn(2, C, 64, 64), STAINS[:C])
    assert tuple(y.shape) == (2, 7)
    assert m.stem.w0.grad is not None


@pytest.mark.parametrize("C", [1, 3, 4, 5, 12])
def test_vit_forward_backward_at_any_channel_count(C):
    m = SpacrChannelViT(stains=STAINS[:C], img=64, patch=16, size="mini", num_classes=7)
    y = _fb(m, torch.randn(2, C, 64, 64), STAINS[:C])
    assert tuple(y.shape) == (2, 7)
    assert m.patch_embed.weight.grad is not None


@pytest.mark.parametrize("C", [1, 4, 12])
def test_bag_of_channels_vit_and_its_mask(C):
    m = SpacrChannelViT(stains=STAINS[:C], img=64, patch=16, size="mini", num_classes=3, bag_of_channels=True)
    _fb(m, torch.randn(2, C, 64, 64), STAINS[:C])
    if C > 1:
        m.eval()
        x = torch.randn(2, C, 64, 64)
        av = torch.ones(2, C, dtype=torch.bool)
        av[1, -1] = False
        x2 = x.clone()
        x2[1, -1] = 50.0
        with torch.no_grad():
            assert (m(x, avail=av) - m(x2, avail=av)).abs().max() < 1e-4


@pytest.mark.parametrize("img,patch", [(96, 12), (128, 16)])
def test_native_resolution_cnn_and_vit(img, patch):
    y1 = _fb(SpacrChannelCNN(num_classes=3, dims=(8, 16, 32, 64), depths=(1, 1, 1, 1)),
             torch.randn(2, 4, img, img), None)
    y2 = _fb(SpacrChannelViT(img=img, patch=patch, size="mini", num_classes=3), torch.randn(2, 4, img, img), None)
    assert y1.shape == y2.shape == (2, 3)


def test_default_sizes_have_the_reference_parameter_counts():
    assert round(count_params(SpacrChannelCNN()) / 1e6, 2) == 2.43
    assert round(count_params(SpacrChannelViT(size="tiny")) / 1e6, 2) == 5.57
    assert round(count_params(SpacrChannelViT(size="small")) / 1e6, 2) == 21.76


_IMAGE_MODELS = [(SpacrChannelCNN, {"dims": (8, 16, 32, 64), "depths": (1, 1, 1, 1)}),
                 (SpacrChannelViT, {"size": "mini", "img": 96, "patch": 12})]


@pytest.mark.parametrize("M,kw", _IMAGE_MODELS)
def test_a_masked_channel_has_no_influence(M, kw):
    m = M(stains=STAINS[:4], **kw).eval()
    x = torch.randn(3, 4, 96, 96)
    av = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    x2 = x.clone()
    x2[1, 1] = torch.randn(96, 96) * 5
    x2[2, 2:] = 9.0
    with torch.no_grad():
        a, b = m(x, avail=av), m(x2, avail=av)
        sub = m(x[2:3, :2], STAINS[:2])
    d = (a - b).abs().max(1).values
    assert d[1] < 1e-4 and d[2] < 1e-4
    assert (sub - a[2:3]).abs().max() < 1e-4


@pytest.mark.parametrize("M,kw", _IMAGE_MODELS)
def test_channel_dropout_in_training_and_subset_at_test(M, kw):
    m = M(stains=STAINS[:5], **kw)
    m.train()
    kept = set()
    for _ in range(12):
        m(torch.randn(2, 5, 96, 96))
        kept.add(m.last_channel_weights.shape[-1])
    assert len(kept) > 1 and max(kept) <= 5
    m.eval()
    m.eval_channel_subset = ["dna", "actin"]
    with torch.no_grad():
        y = m(torch.randn(2, 5, 96, 96))
    assert m.last_channel_weights.shape[-1] == 2
    assert y.shape[0] == 2


def test_mae_loss_with_whole_channel_masking():
    enc = SpacrChannelViT(img=96, patch=12, size="mini")
    enc.channel_dropout = False
    mae = ChannelMAE(enc, dec_dim=64, dec_depth=1, dec_heads=4, p_whole_channel=1.0)
    loss = mae(torch.randn(2, 4, 96, 96))
    loss.backward()
    assert 0.5 < float(loss) < 2.0
    assert enc.patch_embed.weight.grad is not None
    l12 = ChannelMAE(SpacrChannelViT(stains=STAINS[:12], img=96, patch=12, size="mini"), dec_dim=64, dec_depth=1,
                     dec_heads=4)(torch.randn(2, 12, 96, 96), STAINS[:12])
    assert torch.isfinite(l12)


def test_dino_loss_and_ema():
    d = ChannelDINO(SpacrChannelViT(img=96, patch=12, size="mini"), SpacrChannelViT(img=96, patch=12, size="mini"),
                    out=256)
    d.train()
    before = d.t_enc.patch_embed.weight.clone()
    loss = d(torch.randn(3, 4, 96, 96), torch.randn(3, 4, 96, 96))
    loss.backward()
    torch.optim.SGD([p for p in d.parameters() if p.requires_grad], lr=1.0).step()
    d.ema(0.5)
    assert torch.isfinite(loss) and 0 < float(loss) < 2 * np.log(256)
    assert not torch.equal(before, d.t_enc.patch_embed.weight)
    assert not any(p.requires_grad for p in d.t_enc.parameters())


def test_arrayed_well_transformer_ignores_padding_and_order():
    m = SpacrWellTransformer(64, 5, "arrayed", dim=32).eval()
    x = torch.randn(3, 40, 64)
    mask = torch.zeros(3, 40, dtype=torch.bool)
    mask[0, :40] = True
    mask[1, :7] = True
    mask[2, :1] = True
    with torch.no_grad():
        o = m(x, mask)
        x2 = x.clone()
        x2[1, 7:] = 100.0
        x2[2, 1:] = -50.0
        o2 = m(x2, mask)
        perm = torch.randperm(7)
        x3 = x.clone()
        x3[1, :7] = x[1, perm]
        o3 = m(x3, mask)
        single = m(x[2:3, :1], mask[2:3, :1])
    assert (o["logits"] - o2["logits"]).abs().max() < 1e-4
    assert (o["logits"][1] - o3["logits"][1]).abs().max() < 1e-4
    assert (single["logits"] - o["logits"][2:3]).abs().max() < 1e-4
    assert torch.allclose(o["attn"][1, :, 7:], torch.zeros(1))
    assert torch.allclose(o["attn"].sum(-1), torch.ones(1))
    m.train()
    m(x, mask)["logits"].sum().backward()
    assert m.inp[2].weight.grad is not None


def test_pooled_well_transformer_targets_are_independent_and_context_matters():
    m = SpacrWellTransformer(64, 4, "pooled", dim=32).eval()
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "g1"):
                mod.g1.fill_(1.0)
                mod.g2.fill_(1.0)
        torch.nn.init.normal_(m.film.weight, std=0.1)
    t = torch.randn(2, 6, 64)
    c = torch.randn(2, 30, 64)
    cmask = torch.ones(2, 30, dtype=torch.bool)
    cmask[1, 20:] = False
    with torch.no_grad():
        a = m(t, c, cmask)["logits"]
        t2 = t.clone()
        t2[:, 1:] = torch.randn(2, 5, 64)
        b = m(t2, c, cmask)["logits"]
        c2 = c.clone()
        c2[1, 20:] = 77.0
        d = m(t, c2, cmask)["logits"]
        e = m(t, torch.randn(2, 30, 64), cmask)["logits"]
        f = m(t, None, None)["logits"]
        g = m(t, c, torch.zeros(2, 30, dtype=torch.bool))["logits"]
        attn = m(t, c, cmask, return_attn=True)["ctx_attn"]
    assert (a[:, 0] - b[:, 0]).abs().max() < 1e-5
    assert (a - d).abs().max() < 1e-5
    assert (a - e).abs().max() > 1e-4
    assert (f - g).abs().max() < 1e-5
    assert attn.shape == (2, 4, 6, 31)
    m.train()
    m(t, c, cmask)["logits"].sum().backward()
    assert m.tgt[2].weight.grad is not None


def test_pooled_batch_builder_never_puts_a_target_in_its_context():
    from spacr.well_set_training import make_pooled_batch

    rng = np.random.default_rng(0)
    cbw = {w: np.arange(w * 50, w * 50 + 50) for w in range(4)}
    ok = np.ones(200, bool)
    for mode in ("same", "other", "none"):
        for _ in range(50):
            tg, cx = make_pooled_batch(rng, cbw, [0, 1, 2, 3], 16, 40, ok, ok, mode, [0, 1, 2, 3])
            for w, (t, c) in enumerate(zip(tg, cx)):
                assert len(t) == 16
                assert not set(t) & set(c)
                if mode == "other":
                    assert len(c) == 40 and not set(c) & set(cbw[w])
                if mode == "none":
                    assert len(c) == 0


def test_stain_ids_resolve_aliases_and_unknown_names():
    ids = cm.stain_ids(["DAPI", "phalloidin", "my_marker", "x"]).tolist()
    assert ids[:2] == [cm.STAIN_VOCAB.index("dna"), cm.STAIN_VOCAB.index("actin")]
    assert ids[2:] == [cm.STAIN_VOCAB.index("other_2"), cm.STAIN_VOCAB.index("other_3")]


def test_hierarchical_channel_sample_keeps_the_always_slot():
    g = torch.Generator().manual_seed(1)
    picks = [cm.hierarchical_channel_sample(5, g, always=[0, 9]).tolist() for _ in range(40)]
    assert all(p[0] == 0 and p == sorted(p) for p in picks)
    assert {len(p) for p in picks} == {1, 2, 3, 4, 5}


def test_penetrant_fraction_counts_cells_above_the_control_threshold():
    probs = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]])
    thr = torch.tensor([0.5, 0.5])
    assert float(cm.penetrant_fraction(probs, torch.tensor([0]), thr)) == pytest.approx(2 / 3)
    mask = torch.tensor([[True, True, False]])
    assert float(cm.penetrant_fraction(probs, torch.tensor([0]), thr, mask)) == pytest.approx(0.5)


def test_constructor_and_forward_reject_inconsistent_inputs():
    with pytest.raises(ValueError, match="divisible"):
        SpacrChannelViT(img=100, patch=32)
    with pytest.raises(ValueError, match="mode"):
        SpacrWellTransformer(8, 2, mode="bogus")
    with pytest.raises(ValueError, match="stain names"):
        SpacrChannelCNN(stains=["dna"], dims=(8, 16, 32, 64), depths=(1, 1, 1, 1))(torch.randn(1, 2, 32, 32))
    with pytest.raises(ValueError, match="stain names"):
        SpacrChannelViT(stains=["dna"], img=32, patch=16, size="mini")(torch.randn(1, 2, 32, 32))


@pytest.mark.parametrize("arch", ["spacr_channel_cnn", "spacr_channel_vit"])
def test_choose_model_builds_the_channel_image_models(arch):
    from spacr.utils import choose_model

    m = choose_model(arch, torch.device("cpu"), channels=4, height=64, num_classes=3, dropout_rate=0.2,
                     chan_dict={"stains": ["dna", "tubulin", "gh2ax", "actin"]})
    assert isinstance(m, cm.ChannelClassifier)
    assert m.stains == ["dna", "tubulin", "gh2ax", "actin"]
    m.eval()
    with torch.no_grad():
        assert m(torch.randn(2, 4, 64, 64)).shape == (2, 3)


def test_choose_model_sends_the_well_transformer_to_the_well_set_path():
    from spacr.utils import choose_model

    with pytest.raises(ValueError, match="well_set"):
        choose_model("spacr_well_transformer", torch.device("cpu"))


def test_build_channel_model_registry_and_errors():
    with pytest.raises(ValueError, match="train_arch"):
        cm.build_channel_model("resnet18", num_classes=2)
    with pytest.raises(ValueError, match="in_dim"):
        cm.build_channel_model("spacr_well_transformer", num_classes=2)
    well = cm.build_channel_model("spacr_well_transformer", num_classes=2, in_dim=16, options={"dim": 32})
    assert isinstance(well, SpacrWellTransformer)
    vit = cm.build_channel_model("spacr_channel_vit", num_classes=2, channels=3, image_size=32,
                                 options={"size": "mini", "patch": 16, "dataset": "bbbc021",
                                          "eval_subset": ["dna"]})
    assert vit.stains == ["dna", "tubulin", "actin"] and vit.eval_subset == [0]
    assert cm.default_stains(2) == ["other_0", "other_1"]
    with pytest.raises(ValueError, match="2 stain names"):
        cm.default_stains(3, ["a", "b"])
    with pytest.raises(ValueError, match="not among"):
        vit.set_subsets(["golgi"])


@pytest.mark.parametrize("arch", ["spacr_channel_cnn", "spacr_channel_vit"])
def test_one_training_step_of_each_image_model_moves_its_weights(arch):
    opts = {"dims": (8, 16, 32, 64), "depths": (1, 1, 1, 1)} if arch == "spacr_channel_cnn" else {
        "size": "mini", "patch": 16}
    m = cm.build_channel_model(arch, num_classes=2, channels=3, stains=["dna", "tubulin", "actin"], image_size=32,
                               options=dict(opts, train_subset=["dna", "actin"], eval_subset=["dna"]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in m.head.parameters()]
    m.train()
    loss = torch.nn.functional.cross_entropy(m(torch.randn(4, 3, 32, 32)), torch.tensor([0, 1, 0, 1]))
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.head.parameters()))
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 3, 32, 32))
    assert m.enc.last_channel_weights.shape[-1] == 1


def test_one_training_step_of_the_well_transformer():
    from spacr.well_set_training import train_arrayed

    X = np.random.default_rng(0).normal(size=(60, 8)).astype(np.float32)
    rows = [np.arange(i * 10, i * 10 + 10) for i in range(6)]
    y = np.array([0, 1, 0, 1, 0, 1])
    cfg = {"arm": "settf", "steps": 1, "bag_batch": 2, "bag_cells": [2, 5], "dim": 16, "depth": 1}
    model = train_arrayed(cfg, X, rows, y, 2, np.arange(6), torch.device("cpu"))
    torch.manual_seed(0)
    fresh = SpacrWellTransformer(8, 2, "arrayed", dim=16, depth=1)
    assert not torch.equal(fresh.cell_head.weight, model.cell_head.weight)
    assert model.training
