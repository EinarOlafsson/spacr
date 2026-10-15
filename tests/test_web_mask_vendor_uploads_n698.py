"""Real scene-aware CZI and cached public ND2 inputs, plus reader refusals."""
import hashlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from tools.web_mask_generator import inputs


CZI = Path(__file__).parent / "data/watch_vendor_f548/zeiss_czi/two_scenes_tzc.czi"
ND2 = Path(os.environ.get("SPACR_TEST_DOWNLOADS", "/tmp/spacr_test_downloads")) / "MeOh_high_fluo_003.nd2"


def test_actual_czi_scene_time_depth_and_channel_selection():
    czifile = pytest.importorskip("czifile")
    before = hashlib.sha256(CZI.read_bytes()).hexdigest()
    description = inputs.describe_upload(CZI)
    assert description["format"] == "CZI"
    assert len(description["series"]) == 2
    assert description["series"][1]["axes"] == "TCZYX"
    plane = inputs.read_upload_plane(CZI, series=1, indices={"T": 1, "Z": 2}, channels=[1, 0])
    with czifile.CziFile(CZI) as reader:
        expected = reader.scenes[1].asarray()[1, [1, 0], 2]
    np.testing.assert_array_equal(plane["image"], expected)
    assert plane["image"].dtype == np.uint16
    assert plane["selection"] == {"format": "CZI", "channels": [1, 0], "series": 1, "scene_id": 1, "source_axes": "TCZYX", "indices": {"T": 1, "Z": 2}}
    assert hashlib.sha256(CZI.read_bytes()).hexdigest() == before


def test_czi_limits_and_selection_precede_scene_decode(monkeypatch):
    czifile = pytest.importorskip("czifile")
    with czifile.CziFile(CZI) as reader:
        scene_type = type(reader.scenes[0])
    def no_decode(*args, **kwargs):
        pytest.fail("CZI decoded before limits or selection were validated")
    monkeypatch.setattr(scene_type, "asarray", no_decode)
    for kwargs in ({}, {"indices": {"T": 0, "Z": 0}, "max_decoded_bytes": 1}, {"indices": {"T": 0, "Z": 0}, "max_plane_pixels": 1}):
        with pytest.raises(ValueError):
            inputs.read_upload_plane(CZI, **kwargs)


def test_actual_cached_nd2_time_point_and_reader_pixels():
    nd2reader = pytest.importorskip("nd2reader")
    if not ND2.is_file():
        pytest.skip("public ND2 fixture not cached; this test never downloads")
    before = hashlib.sha256(ND2.read_bytes()).hexdigest()
    description = inputs.describe_upload(ND2)
    assert description["series"][0]["shape"] == [13, 1, 1, 600, 800]
    plane = inputs.read_upload_plane(ND2, indices={"T": 5})
    with nd2reader.ND2Reader(str(ND2)) as reader:
        expected = np.asarray(reader.get_frame_2D(t=5, z=0, c=0, v=0))
    np.testing.assert_array_equal(plane["image"], expected[None])
    assert plane["image"].dtype == expected.dtype
    assert plane["selection"] == {"format": "ND2", "channels": [0], "series": 0, "position": 0, "source_axes": "TZCYX", "indices": {"T": 5, "Z": 0}}
    assert hashlib.sha256(ND2.read_bytes()).hexdigest() == before


def test_missing_vendor_readers_are_explained(tmp_path, monkeypatch):
    for suffix, module in ((".czi", "czifile"), (".nd2", "nd2reader")):
        path = tmp_path / ("upload" + suffix)
        path.write_bytes(b"placeholder, no decoder must run")
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, module, None)
            with pytest.raises(ValueError, match=module):
                inputs.describe_upload(path)


class Handle:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("attributes,message", [({}, "scene metadata"), ({"scenes": {}}, "no readable scenes")])
def test_legacy_or_empty_czi_is_refused_instead_of_guessing(tmp_path, monkeypatch, attributes, message):
    path = tmp_path / "input.czi"
    path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "czifile", types.SimpleNamespace(CziFile=lambda path: Handle(**attributes)))
    with pytest.raises(ValueError, match=message):
        inputs.describe_upload(path)


@pytest.mark.parametrize("sizes,message", [({"t": 20}, "spatial dimensions"), ({"x": 4, "y": 5, "q": 2}, "unsupported axes")])
def test_incomplete_or_unknown_nd2_metadata_is_refused(tmp_path, monkeypatch, sizes, message):
    path = tmp_path / "input.nd2"
    path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "nd2reader", types.SimpleNamespace(ND2Reader=lambda path: Handle(sizes=sizes)))
    with pytest.raises(ValueError, match=message):
        inputs.describe_upload(path)
