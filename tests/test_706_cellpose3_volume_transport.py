"""Cellpose 3 volumes cross the real file transport and worker adapter intact."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import _segmentation_backends as B
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


class _Prediction:
    def __init__(self):
        self.calls = []
        self.output = None

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis, z_axis=z_axis, do_3D=do_3D,
                                 stitch_threshold=stitch_threshold)
        self.calls.append(dict(shape=x.shape, channels=channels,
                               channel_axis=channel_axis, z_axis=z_axis,
                               do_3D=do_3D, anisotropy=anisotropy,
                               stitch_threshold=stitch_threshold,
                               normalize=normalize, min_size=min_size))
        shape = tuple(size for index, size in enumerate(x.shape)
                      if channel_axis is None or index != channel_axis % x.ndim)
        labels = self.output
        if labels is None:
            labels = np.zeros(shape, np.uint16)
            labels[tuple(slice(1, -1) for _ in shape)] = 7
        return labels, [None, None, np.zeros(shape, np.float32)], None


def _adapter(model=None):
    adapter = object.__new__(B._Cellpose3Adapter)
    adapter._model = _Prediction() if model is None else model
    adapter._sized = False
    adapter.ignored, adapter.translated = set(), set()
    return adapter


def _remote(tmp_path, adapter):
    calls = []
    adapters = {("test", "cpu", "{}"): adapter}

    class Worker:
        def request(self, operation, **payload):
            assert operation == "segment"
            calls.append(payload)
            return B._worker_segment("cellpose3", payload, adapters)

    remote = object.__new__(B._RemoteBackend)
    remote.name, remote.label, remote.env = "cellpose3", "Cellpose 3", str(tmp_path)
    remote.model, remote.device, remote.options, remote._said = "test", "cpu", {}, set()
    remote._worker_for = lambda *args: Worker()
    return remote, calls


@pytest.mark.parametrize("stitch", [0.0, 0.3])
@pytest.mark.parametrize("shape,z_axis,channel_axis", [
    ((4, 6, 8), 0, None),
    ((4, 6, 8, 1), 0, -1),
    ((2, 6, 4, 8), 2, 0),
])
def test_volume_roundtrip_keeps_axes_channels_spacing_and_stitch_mode(
        tmp_path, shape, z_axis, channel_axis, stitch):
    adapter = _adapter()
    remote, sent = _remote(tmp_path, adapter)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    before = image.copy()
    cancelled = lambda: False
    labels, flows, styles = remote.eval(
        image, channel_axis=channel_axis, z_axis=z_axis, do_3D=stitch == 0,
        anisotropy=2.5, stitch_threshold=stitch, min_size=3,
        should_cancel=cancelled)
    expected = tuple(size for index, size in enumerate(shape)
                     if channel_axis is None or index != channel_axis % len(shape))
    assert labels.shape == expected and np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels)) == {0, 1}
    assert flows[2].shape == expected and styles is None
    assert len(sent) == 1 and len(sent[0]["inputs"]) == 1
    assert sent[0]["should_cancel"] is cancelled
    assert not Path(sent[0]["outputs"]).exists()
    assert {name: sent[0]["params"][name] for name in (
        "channel_axis", "z_axis", "do_3D", "anisotropy", "stitch_threshold")} == {
            "channel_axis": channel_axis, "z_axis": z_axis, "do_3D": stitch == 0,
            "anisotropy": 2.5, "stitch_threshold": stitch}
    call = adapter._model.calls[0]
    assert call["shape"] == shape and call["channel_axis"] == channel_axis
    assert call["z_axis"] == z_axis and call["anisotropy"] == 2.5
    assert call["do_3D"] is (stitch == 0) and call["stitch_threshold"] == stitch
    assert call["channels"] == ([1, 2] if shape[0] == 2 else [0, 0])
    np.testing.assert_array_equal(image, before)


def test_volume_batch_remains_a_batch_and_does_not_split_z_planes(tmp_path):
    adapter = _adapter()
    remote, sent = _remote(tmp_path, adapter)
    images = [np.zeros((4, 6, 8), np.float32), np.ones((5, 7, 9), np.float32)]
    masks, flows, _ = remote.eval(images, channel_axis=None, z_axis=0,
                                 do_3D=True, anisotropy=2)
    assert [mask.shape for mask in masks] == [image.shape for image in images]
    assert len(flows) == 2 and len(sent[0]["inputs"]) == 2
    assert [call["shape"] for call in adapter._model.calls] == [image.shape for image in images]


def test_ordinary_two_channel_field_keeps_the_existing_plane_protocol(tmp_path):
    adapter = _adapter()
    remote, sent = _remote(tmp_path, adapter)
    image = np.zeros((6, 8, 2), np.float32)
    masks, flows, _ = remote.eval(image, channel_axis=-1, normalize=False)
    assert isinstance(masks, list) and masks[0].shape == (6, 8)
    assert len(flows) == 1
    call = adapter._model.calls[0]
    assert call["channels"] == [1, 2] and call["channel_axis"] == -1
    assert call["normalize"] is True
    assert call["do_3D"] is False and call["z_axis"] is None
    assert "do_3D" not in sent[0]["params"] and "stitch_threshold" not in sent[0]["params"]


@pytest.mark.parametrize("shape,updates,message", [
    ((4, 6, 8), {"z_axis": None}, "native 3D"),
    ((4, 6, 8), {"z_axis": 0.5}, "axes"),
    ((4, 6, 8), {"channel_axis": -1}, "axes"),
    ((4, 6, 8, 2), {"channel_axis": 4}, "axes"),
    ((4, 6, 8, 2), {"channel_axis": 0}, "axes"),
    ((1, 6, 8), {}, "axes"),
    ((4, 6, 8), {"anisotropy": float("nan")}, "anisotropy"),
])
def test_bad_volume_geometry_is_rejected_before_prediction(shape, updates, message):
    adapter = _adapter()
    params = dict(channel_axis=None, do_3D=True, z_axis=0, anisotropy=2)
    params.update(updates)
    with pytest.raises(ValueError, match=message):
        adapter.eval([np.zeros(shape, np.float32)], **params)
    assert not adapter._model.calls


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_stitch_threshold_is_not_silently_treated_as_a_plane(tmp_path, value):
    adapter = _adapter()
    remote, sent = _remote(tmp_path, adapter)
    with pytest.raises(ValueError, match="stitch threshold"):
        remote.eval(np.zeros((4, 6, 8)), channel_axis=None, z_axis=0,
                    do_3D=False, stitch_threshold=value, anisotropy=2)
    with pytest.raises(ValueError, match="stitch threshold"):
        adapter.eval([np.zeros((4, 6, 8))], channel_axis=None, z_axis=0,
                     do_3D=False, stitch_threshold=value, anisotropy=2)
    assert not sent and not adapter._model.calls


def test_nonfinite_volume_is_rejected_before_prediction():
    adapter = _adapter()
    image = np.zeros((4, 6, 8), np.float32)
    image[1, 2, 3] = float("nan")
    with pytest.raises(ValueError, match="finite volume"):
        adapter.eval([image], channel_axis=None, z_axis=0, do_3D=True, anisotropy=2)
    assert not adapter._model.calls


@pytest.mark.parametrize("native,stitch", [(True, 0), (False, 0.3)])
def test_installed_model_cannot_silently_drop_the_volume_contract(native, stitch):
    def incompatible(image, *, channel_axis):
        pytest.fail("an incompatible model was evaluated")

    adapter = _adapter(SimpleNamespace(eval=incompatible))
    with pytest.raises(ValueError, match="cannot accept"):
        adapter.eval([np.zeros((4, 6, 8))], channel_axis=None, z_axis=0,
                     do_3D=native, stitch_threshold=stitch, anisotropy=2)


@pytest.mark.parametrize("labels,message", [
    (np.zeros((4, 6), np.uint16), "match the volume"),
    (np.zeros((4, 6, 8), np.float32), "invalid integer"),
    (np.full((4, 6, 8), -1), "invalid integer"),
    (np.full((4, 6, 8), 2**63, np.uint64), "invalid integer"),
])
def test_invalid_model_labels_are_rejected_before_relabeling(labels, message):
    adapter = _adapter()
    adapter._model.output = labels
    with pytest.raises(ValueError, match=message):
        adapter.eval([np.zeros((4, 6, 8))], channel_axis=None, z_axis=0,
                     do_3D=True, anisotropy=2)
    assert len(adapter._model.calls) == 1


def test_cancelled_volume_request_preserves_callback_and_removes_transport_files(tmp_path):
    adapter = _adapter()
    remote, _ = _remote(tmp_path, adapter)
    folders = []
    cancelled = lambda: True

    class Worker:
        def request(self, operation, **payload):
            assert payload["should_cancel"] is cancelled and cancelled()
            folders.append(Path(payload["outputs"]))
            assert len(payload["inputs"]) == 1
            raise InterruptedError("cancelled")

    remote._worker_for = lambda *args: Worker()
    with pytest.raises(InterruptedError, match="cancelled"):
        remote.eval(np.zeros((4, 6, 8)), channel_axis=None, z_axis=0,
                    do_3D=True, anisotropy=2, should_cancel=cancelled)
    assert len(folders) == 1 and not folders[0].exists()
    assert not adapter._model.calls
