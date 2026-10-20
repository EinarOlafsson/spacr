"""Item 691: MitoNet v1 mini as a third-party Model Zoo row, CPU and offline.

The post-processing is checked on synthetic heads, the weights download on
an opener that never touches the network, and the network on a tiny
TorchScript stand-in with MitoNet's three outputs.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from spacr import model_zoo as mz


def _heads(shape, discs):
    """Synthetic MitoNet heads for discs ``(row, col, radius)``."""
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    semantic = np.zeros(shape, np.float32)
    centres = np.zeros(shape, np.float32)
    offsets = np.zeros((2,) + shape, np.float32)
    for r, c, radius in discs:
        inside = (rows - r) ** 2 + (cols - c) ** 2 <= radius ** 2
        semantic[inside] = 0.9
        offsets[0][inside] = r - rows[inside]
        offsets[1][inside] = c - cols[inside]
        centres = np.maximum(centres, np.exp(
            -((rows - r) ** 2 + (cols - c) ** 2) / 8.0).astype(np.float32))
    return semantic, centres, offsets


def test_touching_discs_come_apart_by_their_centres():
    semantic, centres, offsets = _heads((64, 96), [(32, 30, 14), (32, 57, 14)])
    labels = mz.mitonet_instances(semantic, centres, offsets, min_size=10)
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[32, 30] != labels[32, 57]
    assert labels[32, 30] > 0 and labels[32, 57] > 0
    assert (labels > 0).sum() == (semantic >= 0.5).sum()


def test_small_instances_and_weak_centres_are_dropped():
    semantic, centres, offsets = _heads((64, 64), [(20, 20, 10), (50, 50, 3)])
    labels = mz.mitonet_instances(semantic, centres, offsets, min_size=64)
    assert labels.max() == 1
    assert labels[20, 20] == 1 and labels[50, 50] == 0
    empty = mz.mitonet_instances(semantic, centres * 0.01, offsets)
    assert not empty.any()


def test_assignment_in_chunks_matches_one_pass():
    semantic, centres, offsets = _heads((48, 80), [(24, 20, 12), (24, 50, 12)])
    whole = mz.mitonet_instances(semantic, centres, offsets, min_size=1)
    chunked = mz.mitonet_instances(semantic, centres, offsets, min_size=1,
                                   chunk=7)
    np.testing.assert_array_equal(whole, chunked)


def test_wrong_head_shapes_are_refused():
    with pytest.raises(ValueError):
        mz.mitonet_instances(np.zeros((4, 4)), np.zeros((4, 4)),
                             np.zeros((4, 4)))


def test_the_row_is_third_party_with_its_licence_and_citation():
    record = mz._mitonet_record()
    assert record["licence"] == "CC-BY-4.0"
    assert record["code_licence"] == "BSD-3-Clause"
    assert "Conrad" in record["citation"] and "Cell Systems" in record["citation"]
    assert record["uri"].startswith("https://zenodo.org/records/6861565/")
    entry = mz._entry_from_mapping(record)
    assert entry.kind == "mitonet" and entry.licence == "CC-BY-4.0"
    assert mz.MITONET_KEY in mz.THIRD_PARTY_MODEL_KEYS
    assert mz.is_third_party(entry)
    assert mz.source_of(entry) == "Third-party"
    assert "Third-party" in mz.ZOO_SOURCES
    assert "Third-party" not in mz.DEFAULT_ZOO_SOURCES
    assert not mz.is_spacr_owned(entry)


def test_the_weights_download_once_through_a_mocked_opener(tmp_path,
                                                           monkeypatch):
    payload = b"\x80not-really-torchscript" * 64
    digest = hashlib.sha256(payload).hexdigest()
    record = dict(mz._mitonet_record(), sha256=digest,
                  size_bytes=len(payload))
    monkeypatch.setattr(mz, "_mitonet_record", lambda: record)
    monkeypatch.setattr(mz, "_MITONET_VERIFIED", {})
    calls = []

    def opener(uri):
        calls.append(uri)
        return iter([payload[:100], payload[100:]]), len(payload)

    with pytest.raises(mz.ModelUnreadable):
        mz.mitonet_weights(dest=tmp_path, download=False)
    path = mz.mitonet_weights(dest=tmp_path, opener=opener)
    assert path == tmp_path / "MitoNet_v1_mini.pth"
    assert path.read_bytes() == payload
    assert calls == [record["uri"]]
    assert mz.mitonet_weights(dest=tmp_path, opener=opener) == path
    assert len(calls) == 1


def test_a_corrupted_download_is_refused(tmp_path, monkeypatch):
    record = dict(mz._mitonet_record(), sha256="0" * 64)
    monkeypatch.setattr(mz, "_mitonet_record", lambda: record)
    with pytest.raises(mz.ChecksumMismatch):
        mz.mitonet_weights(dest=tmp_path,
                           opener=lambda uri: (iter([b"junk"]), 4))
    assert not list(Path(tmp_path).glob("*.pth"))


def _stand_in(tmp_path):
    """A TorchScript module with MitoNet's three outputs."""
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def forward(self, x):
            bright = (x < 0).float()
            return {"sem_logits": (bright - 0.5) * 20.0,
                    "ctr_hmp": torch.nn.functional.avg_pool2d(
                        bright, 9, 1, 4),
                    "offsets": torch.zeros(x.shape[0], 2, x.shape[2],
                                           x.shape[3])}

    path = tmp_path / "standin.pt"
    torch.jit.script(Net()).save(str(path))
    return path


def test_segment_mitonet_runs_tiled_on_the_cpu(tmp_path):
    weights = _stand_in(tmp_path)
    image = np.full((300, 260), 220, np.uint8)
    image[40:80, 40:80] = 20
    image[200:250, 150:210] = 20
    seen = []
    whole = mz.segment_mitonet(image, weights=weights, device="cpu",
                               min_size=1)
    tiled = mz.segment_mitonet(image, weights=weights, device="cpu",
                               min_size=1, tile=128, halo=32,
                               on_tile=lambda d, t: seen.append((d, t)))
    assert whole.shape == image.shape
    np.testing.assert_array_equal(whole > 0, image < 128)
    np.testing.assert_array_equal(tiled > 0, image < 128)
    assert seen[-1] == (len(seen), len(seen)) and len(seen) == 9
    assert mz._mitonet_input(image.astype(np.float32) / 255.0).max() <= 1.0
