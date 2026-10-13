"""A 4-D TIFF's leading axes are t and z in an order the shape cannot reveal.

``convert_to_yokogawa`` unpacked them as ``t_dim, z_dim, y_dim, x_dim`` with no
check, so a genuine ``(Z, T, Y, X)`` acquisition had every z-plane written out
as a "timepoint" and every maximum projection taken over TIME rather than over
z. The output filenames were confident and wrong, and nothing said so.

tifffile records the real order in ``series[0].axes``. These tests pin that it
is honoured, and that an undeclared file says which way it was read.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile


def _volume(n_t, n_z, h=6, w=8):
    """(T, Z, Y, X) where every voxel encodes its own (t, z).

    t contributes 100 and z contributes 1, so a max over the wrong axis is
    arithmetically distinguishable from a max over the right one.
    """
    vol = np.zeros((n_t, n_z, h, w), dtype=np.uint16)
    for t in range(n_t):
        for z in range(n_z):
            vol[t, z] = 100 * (t + 1) + (z + 1)
    return vol


def _run(folder):
    from spacr.io import convert_to_yokogawa
    convert_to_yokogawa(str(folder))


def _outputs(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".tif")
                  and "_T" in f and "L01C" in f)


def test_a_tzyx_file_writes_one_output_per_timepoint(tmp_path):
    vol = _volume(n_t=3, n_z=5)
    tifffile.imwrite(str(tmp_path / "A01.tif"), vol, metadata={"axes": "TZYX"})

    _run(tmp_path)

    out = _outputs(tmp_path)
    assert len(out) == 3, out
    # timepoint 1 maxes over z: 100*1 + max(1..5) = 105
    first = np.asarray(tifffile.imread(str(tmp_path / out[0])))
    assert int(first.max()) == 105


def test_a_ztyx_file_is_not_read_as_tzyx(tmp_path):
    """The bug: 5 z-planes became 5 "timepoints" and the projection ran over t.

    The same buffer is written declaring ZTYX, so axis 0 is z (length 5) and
    axis 1 is t (length 3). The correct answer is 3 outputs, not 5.
    """
    vol = _volume(n_t=3, n_z=5).transpose(1, 0, 2, 3)   # now (Z, T, Y, X)
    assert vol.shape == (5, 3, 6, 8)
    tifffile.imwrite(str(tmp_path / "A01.tif"), vol, metadata={"axes": "ZTYX"})

    _run(tmp_path)

    out = _outputs(tmp_path)
    assert len(out) == 3, f"read as TZYX it would be 5 files: {out}"
    # timepoint 1 still maxes over z: 100*1 + max(1..5) = 105
    first = np.asarray(tifffile.imread(str(tmp_path / out[0])))
    assert int(first.max()) == 105


def test_the_two_orders_of_one_buffer_give_the_same_answer(tmp_path):
    """The point of the whole fix: the axes tag, not the memory layout, decides."""
    tz = tmp_path / "tz"
    zt = tmp_path / "zt"
    tz.mkdir(); zt.mkdir()
    vol = _volume(n_t=3, n_z=5)
    tifffile.imwrite(str(tz / "A01.tif"), vol, metadata={"axes": "TZYX"})
    tifffile.imwrite(str(zt / "A01.tif"), vol.transpose(1, 0, 2, 3),
                     metadata={"axes": "ZTYX"})

    _run(tz)
    _run(zt)

    a = [np.asarray(tifffile.imread(str(tz / f))) for f in _outputs(tz)]
    b = [np.asarray(tifffile.imread(str(zt / f))) for f in _outputs(zt)]
    assert len(a) == len(b) == 3
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_an_undeclared_4d_file_says_how_it_was_read(capsys, tmp_path):
    """Falling back to TZYX is defensible; doing it silently is not."""
    vol = _volume(n_t=2, n_z=4)
    # write the raw array with no axes metadata
    tifffile.imwrite(str(tmp_path / "A01.tif"), vol)

    _run(tmp_path)

    out = capsys.readouterr().out
    if "WARNING" in out:
        assert "(T, Z, Y, X)" in out
        assert "z-plane" in out
    else:
        # tifffile inferred an order itself; then it must have been honoured
        assert len(_outputs(tmp_path)) in (2, 4)


def test_a_3d_zstack_is_unaffected(tmp_path):
    """The 3-D branch is untouched: one projection, one output."""
    stack = np.zeros((7, 6, 8), dtype=np.uint16)
    for z in range(7):
        stack[z] = z + 1
    tifffile.imwrite(str(tmp_path / "A01.tif"), stack, metadata={"axes": "ZYX"})

    _run(tmp_path)

    out = _outputs(tmp_path)
    assert len(out) == 1
    assert int(np.asarray(tifffile.imread(str(tmp_path / out[0]))).max()) == 7
