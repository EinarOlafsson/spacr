"""Spotiflow as a spot detector in an environment of its own.

Its worker answers ``spotiflow_spots`` with coordinates; spaCR's side sends
the image through the ``.npy`` hand-off and refuses, with the reason, when
the backend is not installed; OPS can name it as its sequencing-spot
detector, and it then decodes one field at a time.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB
from spacr import ops_engine


@pytest.fixture(autouse=True)
def _sandboxed(tmp_path, monkeypatch):
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(SB, "_CANDIDATES", {})
    return tmp_path


def _installed(root):
    env = Path(root) / "spotiflow"
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": "spotiflow"})
    return env


def test_readiness_says_to_install_it_from_the_model_zoo(tmp_path):
    ready, reason = SB._spotiflow_readiness(str(tmp_path / "backends"))
    assert not ready and "Model Zoo" in reason
    with pytest.raises(ImportError, match="Spotiflow is not installed"):
        SB._spotiflow_spots(np.zeros((4, 4)), root=str(tmp_path / "backends"))
    _installed(tmp_path / "backends")
    assert SB._spotiflow_readiness(str(tmp_path / "backends"))[0]


def test_spots_are_asked_of_spotiflows_own_worker(tmp_path):
    root = tmp_path / "backends"
    _installed(root)
    seen = {}

    class _Worker:
        def request(self, op, **payload):
            seen.update(payload, op=op, image=np.load(payload["image"]))
            return {"spots": [[1.5, 2.25], [3.0, 4.0]]}

    def worker_for(name, env):
        seen["name"] = name
        return _Worker()

    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    spots = SB._spotiflow_spots(image, root=str(root), worker_for=worker_for)
    assert (seen["name"], seen["op"]) == ("spotiflow", "spotiflow_spots")
    assert seen["threshold"] is None and seen["model"] == "general"
    np.testing.assert_array_equal(seen["image"], image.astype(np.float32))
    np.testing.assert_array_equal(spots, [[1.5, 2.25], [3.0, 4.0]])


def test_the_worker_loads_each_model_once_and_returns_coordinates(
        tmp_path, monkeypatch):
    loads = []

    class _Network:
        def predict(self, image, **kwargs):
            return (np.array([[2.0, 3.5]]),
                    types.SimpleNamespace(prob=np.array([0.8]),
                                          heatmap=np.zeros(image.shape)))

    monkeypatch.setattr(SB, "_spotiflow_network",
                        lambda model, device: loads.append(model) or _Network())
    path = tmp_path / "image.npy"
    np.save(path, np.ones((6, 7), np.float32))
    adapters = {}
    for _ in range(2):
        reply = SB._handle("spotiflow", {
            "protocol": SB._PROTOCOL, "id": 1, "op": "spotiflow_spots",
            "image": str(path), "threshold": 0.5, "device": "cpu"}, adapters)
    assert reply["ok"], reply
    assert reply["spots"] == [[2.0, 3.5]] and reply["probabilities"] == [0.8]
    assert loads == ["general"]


def test_ops_refuses_spotiflow_that_cannot_run_rather_than_swap_it():
    with pytest.raises(ValueError, match="spotiflow' cannot run"):
        ops_engine._spot_detector({"ops_spot_detector": "Spotiflow"})


def test_ops_takes_spotiflow_once_installed(tmp_path):
    _installed(tmp_path / "backends")
    assert ops_engine._spot_detector(
        {"ops_spot_detector": "spotiflow"}) == "spotiflow"
    assert "spotiflow" in ops_engine._BACKEND_SPOT_DETECTORS


def test_spotiflow_keeps_its_own_threshold_in_ops():
    stack = np.zeros((2, 4, 8, 9), np.float32)
    stack[:, 1, 3, 4] = 50
    seen = {}

    def detect(image, threshold):
        seen["threshold"] = threshold
        return [[3.2, 4.4]]

    peaks = ops_engine._spotnet_peaks(stack, detect=detect, threshold=None)
    assert seen["threshold"] is None
    np.testing.assert_array_equal(peaks, [[3, 4]])
