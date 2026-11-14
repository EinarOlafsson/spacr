"""Table-driven tests of the merged plane-layout manifest and crop colour planes.

A crop is cut from a plane and a colour slot, and both have to be right.  A
malformed plane-layout sidecar must stop the run (ignoring it would measure
the wrong plane), and ``build_png_channels`` must put each source channel in
the colour slot the mapping names, keep greyscale as one plane, fill unused
slots with one shared blank and refuse indices past the end of the array.
Everything is synthetic, tiny and offline.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr.crops import (MERGED_LAYOUT_SIDECAR, CorruptMergedFile, CropError,
                         build_png_channels, read_merged_plane_layout)

_GOOD = {"version": 1, "mask_plane_order": ["cell", "nucleus"],
         "mask_dims": {"cell": 2, "nucleus": 3},
         "intensity_channels": [0, 1]}


def _layout(**overrides):
    """A copy of the valid two-channel manifest with ``overrides`` applied."""
    payload = dict(_GOOD)
    payload.update(overrides)
    return payload


# (case id, sidecar text or payload or None for no sidecar, refusal fragment or None)
_LAYOUT_CASES = [
    ("no sidecar is a legacy folder", None, None),
    ("valid manifest", _GOOD, None),
    ("not json", "{not json at all", "cannot read merged plane layout"),
    ("future version", _layout(version=2), "expected an object with version 1"),
    ("a list, not an object", [1, 2], "expected an object with version 1"),
    ("version only", {"version": 1}, "expected intensity_channels list, "
     "mask_plane_order list and mask_dims object"),
    ("channels not a list", _layout(intensity_channels="01"),
     "expected intensity_channels list"),
    ("duplicate role", _layout(mask_plane_order=["cell", "cell"],
                               mask_dims={"cell": 2}),
     "invalid mask_plane_order"),
    ("unknown role", _layout(mask_plane_order=["spaceship"],
                             mask_dims={"spaceship": 2}),
     "invalid mask_plane_order"),
    ("boolean dim", _layout(mask_plane_order=["cell"], mask_dims={"cell": True}),
     "invalid mask dim for 'cell'"),
    ("text dim", _layout(mask_plane_order=["cell"], mask_dims={"cell": "two"}),
     "invalid mask dim for 'cell'"),
    ("missing dim", _layout(mask_plane_order=["cell"], mask_dims={}),
     "invalid mask dim for 'cell'"),
    ("dim off the channel count", _layout(mask_plane_order=["cell"],
                                          mask_dims={"cell": 7}),
     "inconsistent mask dims"),
    ("dims swapped", _layout(mask_dims={"cell": 3, "nucleus": 2}),
     "inconsistent mask dims"),
    ("extra dim not in the order", _layout(mask_plane_order=["cell"],
                                           mask_dims={"cell": 2, "nucleus": 3}),
     "inconsistent mask dims"),
]


def test_read_merged_plane_layout_accepts_only_a_consistent_manifest(tmp_path, subtests):
    """Absent sidecar -> None, a valid one is returned with int dims, any malformed one raises."""
    for index, (case_id, payload, fragment) in enumerate(_LAYOUT_CASES):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"merged{index}"
            folder.mkdir()
            if payload is not None:
                text = payload if isinstance(payload, str) else json.dumps(payload)
                (folder / MERGED_LAYOUT_SIDECAR).write_text(text, encoding="utf-8")
            if fragment is not None:
                with pytest.raises(CorruptMergedFile) as raised:
                    read_merged_plane_layout(str(folder))
                assert fragment in str(raised.value)
                assert MERGED_LAYOUT_SIDECAR in str(raised.value)
                continue
            # The folder and a .npy inside it read the same sidecar.
            for target in (str(folder), str(folder / "plate1_A01_1.npy")):
                if payload is None:
                    assert read_merged_plane_layout(target) is None
                else:
                    assert read_merged_plane_layout(target) == _GOOD


def test_read_merged_plane_layout_coerces_numeric_text_dims(tmp_path):
    """A dim stored as the text "2" is returned as the integer 2, the rest untouched."""
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / MERGED_LAYOUT_SIDECAR).write_text(json.dumps(
        _layout(mask_plane_order=["cell"], mask_dims={"cell": "2"})),
        encoding="utf-8")
    assert read_merged_plane_layout(str(folder)) == {
        "version": 1, "mask_plane_order": ["cell"], "mask_dims": {"cell": 2},
        "intensity_channels": [0, 1]}


_SRC = np.stack([np.full((4, 4), 7, np.uint16), np.full((4, 4), 9, np.uint16),
                 np.arange(16, dtype=np.uint16).reshape(4, 4)], axis=2)

# (case id, data, mapping, dtype, expected per-plane values or refusal fragment)
_PNG_CASES = [
    ("one channel three times stays grey", _SRC, {"r": 2, "g": 2, "b": 2}, None,
     [_SRC[:, :, 2]]),
    ("unused slot is blank", _SRC[:, :, :2], {"r": None, "g": 1, "b": 0}, None,
     [np.zeros((4, 4), np.uint16), _SRC[:, :, 1], _SRC[:, :, 0]]),
    ("two unused slots share one blank", _SRC, {"r": None, "g": 2, "b": None}, None,
     [np.zeros((4, 4), np.uint16), _SRC[:, :, 2], np.zeros((4, 4), np.uint16)]),
    ("file order is red green blue", _SRC, {"r": 0, "g": 1, "b": 2}, None,
     [_SRC[:, :, 0], _SRC[:, :, 1], _SRC[:, :, 2]]),
    ("negative index counts from the end", _SRC, {"r": -1, "g": 0, "b": 1}, None,
     [_SRC[:, :, 2], _SRC[:, :, 0], _SRC[:, :, 1]]),
    ("dtype cast", _SRC, {"r": 0, "g": 1, "b": 2}, np.uint8,
     [_SRC[:, :, 0], _SRC[:, :, 1], _SRC[:, :, 2]]),
    ("index past the end", _SRC[:, :, :2], {"r": 5, "g": 1, "b": 0}, None,
     "png_channel_mapping['r'] = 5 is out of range for an array with 2 channels"),
    ("negative index past the start", _SRC[:, :, :2], {"r": 0, "g": -3, "b": 0}, None,
     "png_channel_mapping['g'] = -3 is out of range for an array with 2 channels"),
    ("two-dimensional input", np.zeros((4, 4), np.uint16), {"r": 0}, None,
     "build_png_channels needs a (H, W, C) array; got (4, 4)"),
]


def test_build_png_channels_places_each_source_in_its_colour_slot(subtests):
    """Exact planes per mapping: grey collapses to one plane, None is zeros, bad indices raise."""
    for case_id, data, mapping, dtype, expected in _PNG_CASES:
        with subtests.test(msg=case_id):
            if isinstance(expected, str):
                with pytest.raises(CropError) as raised:
                    build_png_channels(data, mapping, dtype=dtype)
                assert expected in str(raised.value)
                continue
            out = build_png_channels(data, mapping, dtype=dtype)
            assert out.shape == (4, 4, len(expected))
            assert out.dtype == (np.dtype(dtype) if dtype else data.dtype)
            for plane, want in enumerate(expected):
                np.testing.assert_array_equal(out[:, :, plane], want)
