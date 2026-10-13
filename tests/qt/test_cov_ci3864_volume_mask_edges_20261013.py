"""The edges of a native 3D mask session and of reading an MRC volume.

Each refusal is pinned by the sentence the user reads, and each quiet case
by what the mask holds afterwards:

* a section along an axis that is not X, Y or Z, and an object ID of 0,
  are refused with the reason, before anything is written;
* interpolating between two sections that already touch fills nothing;
* assigning objects in a mask whose label type is full says so instead of
  wrapping an ID around onto another object;
* a saved geometry is what a label shows, not a re-derived mesh;
* an MRC file without ``mrcfile`` installed, one beyond the memory budget
  and one holding complex numbers are refused with the reason, and one
  whose header carries no voxel size opens in pixels instead of in a
  made-up physical unit.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from spacr.qt import mask_engine
from spacr.qt.mask_engine import VolumeMaskSession, _read_mrc_volume


def _session(mask, dtype=np.uint16):
    image = np.zeros(mask.shape, dtype=np.float32)
    return VolumeMaskSession(image, np.asarray(mask, dtype=dtype))


def test_a_section_along_an_unknown_axis_is_refused():
    session = _session(np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="X, Y or Z axis"):
        session.plane("W", 0)


def test_object_id_zero_is_refused_before_anything_is_written():
    mask = np.zeros((4, 6, 6))
    mask[0, 1:3, 1:3] = 1
    session = _session(mask)
    before = session.mask.copy()
    with pytest.raises(ValueError, match="positive object ID"):
        session.interpolate(0)
    assert np.array_equal(session.mask, before)


def test_sections_that_already_touch_have_no_gap_to_fill():
    mask = np.zeros((4, 6, 6))
    mask[1, 1:4, 1:4] = 2
    mask[2, 2:5, 2:5] = 2
    session = _session(mask)
    before = session.mask.copy()
    assert session.interpolate(2) == 0
    assert np.array_equal(session.mask, before)


def test_a_full_label_type_refuses_to_assign_another_object():
    mask = np.zeros((5, 24, 24), dtype=np.uint8)
    spots = [(z, y, x) for z in (0, 2, 4) for y in range(0, 24, 2)
             for x in range(0, 24, 2)]
    for label, spot in zip(range(1, 256), spots):
        mask[spot] = label
    mask[spots[255]] = 1
    session = _session(mask, dtype=np.uint8)
    with pytest.raises(ValueError, match="No unused object identifier"):
        session.assign_objects()
    assert np.array_equal(session.mask, mask)


def test_a_saved_geometry_is_what_the_label_shows():
    mask = np.zeros((3, 4, 4))
    mask[1, 1:3, 1:3] = 5
    session = _session(mask)
    saved = object()
    session.geometries[5] = saved
    assert session.geometry_for_label(5) is saved


def _mrc(tmp_path, data, voxel_size=None):
    mrcfile = pytest.importorskip("mrcfile")
    path = tmp_path / "volume.mrc"
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(data)
        if voxel_size is not None:
            handle.voxel_size = voxel_size
    return path


def test_an_mrc_volume_without_mrcfile_says_what_to_install(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mrcfile", None)
    with pytest.raises(ValueError, match=r"spacr\[tem\]"):
        _read_mrc_volume(tmp_path / "volume.mrc")


def test_an_mrc_volume_beyond_the_budget_is_refused(tmp_path, monkeypatch):
    path = _mrc(tmp_path, np.zeros((3, 4, 5), dtype=np.float32))
    monkeypatch.setattr(VolumeMaskSession, "MAX_VOXELS", 10)
    with pytest.raises(ValueError, match="memory budget"):
        _read_mrc_volume(path)


def test_an_mrc_volume_of_complex_numbers_is_refused(tmp_path):
    path = _mrc(tmp_path, np.zeros((3, 4, 5), dtype=np.complex64))
    with pytest.raises(ValueError, match="finite numeric"):
        _read_mrc_volume(path)


def test_an_mrc_volume_without_a_voxel_size_opens_in_pixels(tmp_path):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    path = _mrc(tmp_path, data, voxel_size=0.0)
    image, axes, spacing, unit = _read_mrc_volume(path)
    assert np.array_equal(image, data)
    assert axes == "ZYX"
    assert spacing == (1.0, 1.0, 1.0)
    assert unit == "pixel"


def test_an_mrc_volume_with_a_voxel_size_opens_in_nanometres(tmp_path):
    path = _mrc(tmp_path, np.zeros((3, 4, 5), dtype=np.float32),
                voxel_size=(20.0, 30.0, 40.0))
    _image, _axes, spacing, unit = _read_mrc_volume(path)
    assert unit == "nm"
    assert spacing == pytest.approx((4.0, 3.0, 2.0))
    assert mask_engine.VolumeMaskSession.MAX_VOXELS > 60
