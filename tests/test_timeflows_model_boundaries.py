"""CPU contracts for Timeflows' encoder, tiles and training lifecycle.

Tiny trainable encoders prove tensor/gradient behavior; these are not
pretrained-model accuracy measurements. CTC fixtures are real TIFF files.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import tifffile

from spacr import timeflows_model as tm

torch = pytest.importorskip("torch")
nn = torch.nn


class _Encoder(nn.Module):
    """Small SAM-shaped encoder, including the channel-last block contract."""

    def __init__(self, positions):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 2, stride=2)
        self.pos_embed = nn.Parameter(torch.ones(1, 4, 5, 4)) if positions else None
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.GELU()])
        self.neck = nn.Conv2d(4, 4, 1)


class _Sam(nn.Module):
    ps = 2

    def __init__(self, positions):
        super().__init__()
        self.encoder = _Encoder(positions)


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("channels", [1, 3])
def test_encoder_bridge_shares_weights_and_preserves_gradients(positions, channels):
    sam = _Sam(positions).double()
    bridge = tm.CellposeSamFeatures(sam)
    frame = torch.linspace(0, 1, channels * 8 * 10).reshape(1, channels, 8, 10)
    projection = sam.encoder.patch_embed.proj
    expected = torch.nn.functional.conv2d(
        frame.double(), projection.weight[:, :channels], projection.bias, stride=2)
    expected = expected.permute(0, 2, 3, 1)
    if positions:
        expected = expected + sam.encoder.pos_embed
    for block in sam.encoder.blocks:
        expected = block(expected)
    expected = sam.encoder.neck(expected.permute(0, 3, 1, 2)).float()

    actual = bridge(frame)
    assert actual.shape == (1, 4, 4, 5) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    assert [id(p) for p in bridge.parameters()] == [id(p) for p in sam.parameters()]
    actual.square().mean().backward()
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad[:, :channels].abs().sum() > 0
    if channels == 1:
        assert not projection.weight.grad[:, 1:].any()


def test_wrapped_encoder_is_registered_and_moves_with_time_head():
    sam = _Sam(False).double()
    net = tm.TimeflowsNet(tm.CellposeSamFeatures(sam), channels=4, ps=2).float()
    assert net.encoder is sam
    assert sam.encoder.patch_embed.proj.weight.dtype == torch.float32
    assert "encoder.encoder.patch_embed.proj.weight" in net.state_dict()
    result = net(torch.zeros(1, 3, 8, 10), torch.ones(1, 3, 8, 10))
    assert result.shape == (1, 3, 8, 10)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_multichannel_inputs_keep_order_and_leave_source_unchanged(channels):
    frame = np.broadcast_to(np.arange(channels), (5, 7, channels)).astype(np.float64)
    frame = frame[:, ::-1]
    before = frame.copy()
    result = tm._to_input(frame)
    expected = ([0] * 3 if channels == 1 else [0, 1, 0] if channels == 2 else [0, 1, 2])
    assert result.shape == (1, 3, 5, 7) and result.dtype == torch.float32
    assert result.is_contiguous()
    np.testing.assert_array_equal(result[0, :, 0, 0].numpy(), expected)
    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize("shape", [(9, 13), (257, 451), (449, 513)])
def test_prediction_tiles_cover_borders_and_average_overlaps(shape):
    class PixelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, first, second):
            assert not self.training and not torch.is_grad_enabled()
            assert first.shape == second.shape == (1, 3, tm.TILE, tm.TILE)
            self.calls.append(float(first[0, 0, 0, 0]))
            return torch.stack((first[:, 0], second[:, 0], first[:, 0] * 0), dim=1)

    first = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
    second = first * 2 + 1
    before = first.copy(), second.copy()
    net = PixelModel()
    result = tm.predict_pair(net, first, second)
    assert result["vector"].shape == (2,) + shape
    np.testing.assert_allclose(result["vector"][0], first, rtol=1e-6)
    np.testing.assert_allclose(result["vector"][1], second, rtol=1e-6)
    np.testing.assert_array_equal(result["successor"], np.full(shape, 0.5))
    assert len(net.calls) == (1 if shape == (9, 13) else 6 if shape == (257, 451) else 9)
    np.testing.assert_array_equal(first, before[0])
    np.testing.assert_array_equal(second, before[1])


@pytest.mark.parametrize("full_steps", [0, 2])
def test_training_freezes_encoder_until_full_stage_and_reports_progress(full_steps):
    class Backbone(nn.Module):
        channels, ps = 4, 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 8, stride=8)

        def forward(self, x):
            return self.conv(x)

    torch.manual_seed(17)
    net = tm.TimeflowsNet(Backbone())
    labels = np.zeros((32, 32), np.int32)
    labels[8:16, 8:16] = 1
    moved = np.roll(labels, 4, axis=1)
    pair = tm._Pair((labels > 0).astype(np.float32), (moved > 0).astype(np.float32), labels, moved)
    initial = net.backbone.conv.weight.detach().clone()
    initial_head = net.up.weight.detach().clone()
    observations = []

    def observe(line):
        observations.append((line, net.backbone.conv.weight.detach().clone(),
                             net.backbone.conv.weight.requires_grad))

    losses = tm.train_timeflows(net, [pair], head_steps=2, full_steps=full_steps,
                               weights=np.array([7.0]), log=observe)
    assert len(losses) == 2 + full_steps and np.isfinite(losses).all()
    assert not torch.equal(initial_head, net.up.weight)
    heads = [entry for entry in observations if entry[0].startswith("head")]
    assert len(heads) == 2
    assert all(torch.equal(initial, weights) and not trainable for _, weights, trainable in heads)
    full = [entry for entry in observations if entry[0].startswith("full")]
    assert len(full) == full_steps
    if full_steps:
        assert all(trainable for _, _, trainable in full)
        assert not torch.equal(initial, full[-1][1])
    else:
        assert torch.equal(initial, net.backbone.conv.weight)


def test_unsupervised_background_has_zero_loss_and_gradient():
    blank = np.zeros((7, 9), np.int32)
    targets = {key: torch.from_numpy(value)[None] for key, value in tm.time_targets(blank, blank).items()}
    output = torch.randn(1, 3, 7, 9, requires_grad=True)
    loss = tm.timeflows_loss(output, targets)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(output.grad).all() and not output.grad.any()


@pytest.mark.parametrize("case", ["empty_source", "empty_target", "no_successor", "too_far"])
def test_linking_abstains_instead_of_inventing_a_successor(case):
    first = np.zeros((30, 50), np.int32)
    first[3:7, 3:7] = 9
    second = np.zeros_like(first)
    second[23:27, 43:47] = 71
    probability = np.ones(first.shape, np.float32)
    if case == "empty_source":
        first[:] = 0
    elif case == "empty_target":
        second[:] = 0
    elif case == "no_successor":
        probability[:] = 0.49
    prediction = {"vector": np.zeros((2,) + first.shape), "successor": probability}
    assert tm.link_by_timeflows(first, second, prediction) == {}


def test_scramble_without_ground_truth_does_not_run_predictor():
    def no_prediction(*args):
        pytest.fail("an unscorable pair must not run inference")

    blank = np.zeros((8, 8), np.int32)
    result = tm.scramble_test(blank, blank, no_prediction, blank, blank)
    assert result["objects"] == 0
    assert np.isnan(result["model"]) and np.isnan(result["iou"])


def test_ctc_pair_limit_reads_only_selected_endpoints_and_ignores_gaps(tmp_path, monkeypatch):
    for folder in ("01", "01_ST/SEG", "01_GT/TRA"):
        (tmp_path / folder).mkdir(parents=True)
    for index in [0, 1, 2, 3, 5, 6, 7]:
        labels = np.zeros((8, 10), np.uint16)
        labels[2:5, 2:5] = 17
        for folder, prefix, array in [("01", "t", labels * 10),
                                      ("01_ST/SEG", "man_seg", labels),
                                      ("01_GT/TRA", "man_track", labels)]:
            tifffile.imwrite(tmp_path / folder / f"{prefix}{index:03d}.tif", array)
    for name in ("t_notes.tif", "notes.txt", "wrong000.tif"):
        (tmp_path / "01" / name).write_text("not an image")
    original = tifffile.imread
    reads = []

    def read(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(tifffile, "imread", read)
    pairs = tm.ctc_pairs(str(tmp_path), max_pairs=2)
    assert len(pairs) == 2 and len(reads) == 12
    assert {tm._frame_number(path.rsplit("/", 1)[-1]) for path in reads} == {0, 1, 6, 7}
    assert all(set(np.unique(pair.labels_t)) == {0, 17} for pair in pairs)
    reads.clear()
    assert tm.ctc_pairs(str(tmp_path), "02") == [] and reads == []
    assert len(tm.ctc_pairs(str(tmp_path))) == 5


def test_cli_with_no_pairs_stops_before_model_creation_or_output(tmp_path, monkeypatch):
    def no_model(**kwargs):
        pytest.fail("missing training pairs must not trigger model creation/download")

    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(
        models=types.SimpleNamespace(CellposeModel=no_model)))
    output = tmp_path / "model.pt"
    with pytest.raises(SystemExit, match="no usable pairs"):
        tm.main(["--movies", str(tmp_path / "missing"), "--out", str(output), "--device", "cpu"])
    assert not output.exists() and not output.with_suffix(".pt.json").exists()
