"""What Make Masks' loader repairs, and what it refuses, on the way in.

``load_image_and_mask`` brings one field to the editor's 2-D uint16 grey and
its labels to the narrowest unsigned dtype that keeps every ID.  Colour is
collapsed with BT.601 weights (alpha dropped), signed and float fields are
shifted and stretched in order, and a blank field stays blank.  A volume,
five planes, non-finite pixels, a mask that is not one label per pixel,
fractional or negative labels and a label past uint32 are refused with a
message naming the file -- the screen has no modal dialog, and a silent
truncation would put a mask on disk that no longer keys against the
measurements made from it.  Neither file is ever rewritten by loading.

Pure numpy and tifffile: the backend has no Qt dependency.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as me


def _write(folder, name, array, *, mask=None, **tiff_options):
    tifffile.imwrite(folder / name, array, **tiff_options)
    if mask is not None:
        tifffile.imwrite(folder / "masks" / name, mask)
    return str(folder), name


def _rgb():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255         # pure red
    rgb[1, 1, 1] = 255         # pure green
    rgb[2, 2, 2] = 255         # pure blue
    return rgb


def _rgba():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[0, 0, 0] = 255        # one red pixel
    rgba[..., 3] = 255         # opaque everywhere: must not reach the grey
    return rgba


def _one_plane():
    stack = np.zeros((4, 4, 1), dtype=np.uint16)
    stack[1, 1, 0] = 7
    return stack


_YXC = dict(photometric="minisblack", metadata={"axes": "YXC"})


def test_load_image_and_mask_brings_every_field_to_uint16_grey(tmp_path,
                                                               subtests):
    """Each pixel type loads as the exact 2-D uint16 display, with an empty uint8 mask."""
    (tmp_path / "masks").mkdir()
    # case, pixels, tiff options, expected grey
    cases = [
        ("rgb uses BT.601 weights", _rgb(), {},
         [[33370, 0, 0, 0], [0, 65535, 0, 0], [0, 0, 12727, 0],
          [0, 0, 0, 0]]),
        ("rgba drops its alpha plane", _rgba(), {},
         [[65535, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("a single-plane stack is squeezed, uint16 kept", _one_plane(), _YXC,
         [[0, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("signed float keeps its order", np.array(
            [[-2., -1.], [0., 2.]], dtype=np.float32), {},
         [[0, 16383], [32767, 65535]]),
        ("signed int32 keeps its order", np.array(
            [[-2147483648, -1073741824], [0, 2147483647]], dtype=np.int32),
         {}, [[0, 16383], [32767, 65535]]),
        ("constant negative float is zero", np.full((3, 3), -5, np.float32),
         {}, np.zeros((3, 3)).tolist()),
        ("constant negative int is zero", np.full((3, 3), -5, np.int32),
         {}, np.zeros((3, 3)).tolist()),
        ("an all-zero float field stays blank",
         np.zeros((3, 3), np.float32), {}, np.zeros((3, 3)).tolist()),
        ("a positive uint8 field stretches to its maximum",
         np.array([[0, 51], [102, 255]], dtype=np.uint8), {},
         [[0, 13107], [26214, 65535]]),
    ]
    for index, (case, pixels, options, grey) in enumerate(cases):
        with subtests.test(msg=case):
            name = f"field{index}.tif"
            args = _write(tmp_path, name, pixels, **options)
            before = (tmp_path / name).read_bytes()
            image, mask = me.load_image_and_mask(*args)
            assert image.dtype == np.uint16
            np.testing.assert_array_equal(image, grey)
            assert mask.dtype == np.uint8 and mask.shape == image.shape
            assert not mask.any()
            assert (tmp_path / name).read_bytes() == before

    # case, pixels, tiff options, ValueError fragments
    refusals = [
        ("five planes", np.zeros((4, 4, 5), np.uint8), _YXC,
         ["Unsupported channel count 5 in", "five planes.tif",
          "expected grayscale, RGB, or RGBA"]),
        ("a volume", np.zeros((3, 4, 4, 2), np.uint8), {},
         ["Unsupported image shape (3, 4, 4, 2) in",
          "Make Masks expects one 2-D field"]),
    ]
    for value in (np.nan, np.inf, -np.inf):
        image = np.zeros((3, 3), dtype=np.float32)
        image[1, 1] = value
        refusals.append((f"a {value} pixel", image, {},
                         ["Image contains non-finite values:",
                          f"a {value} pixel.tif"]))
    for case, pixels, options, fragments in refusals:
        with subtests.test(msg=f"refused: {case}"):
            with pytest.raises(ValueError) as raised:
                me.load_image_and_mask(*_write(tmp_path, f"{case}.tif",
                                               pixels, **options))
            for fragment in fragments:
                assert fragment in str(raised.value)


def test_load_image_and_mask_keeps_every_label_id_or_refuses_the_mask(
        tmp_path, subtests):
    """Each saved mask loads with its exact IDs in the narrowest dtype, or is refused."""
    (tmp_path / "masks").mkdir()
    field = np.zeros((4, 4), dtype=np.uint16)
    past_uint16 = np.zeros((4, 4), dtype=np.int32)
    past_uint16[1:3, 1:3] = 70000
    past_uint16[3, 3] = 4464           # what 70000 used to wrap onto
    labels = np.array([[0, 7], [900, 65535]], dtype=np.uint16)
    stored_uint32 = np.full((4, 4), 3, dtype=np.uint32)
    # case, image, mask, expected dtype, expected labels
    cases = [
        ("a trailing axis is squeezed", field,
         np.ones((4, 4, 1), np.uint8), np.uint8, np.ones((4, 4))),
        ("whole float labels load", field, np.full((4, 4), 3.0, np.float32),
         np.uint8, np.full((4, 4), 3)),
        ("uint16 IDs are kept", np.zeros((2, 2), np.float32), labels,
         np.uint16, labels),
        ("a label past uint16 widens without wrapping", field, past_uint16,
         np.uint32, past_uint16),
        ("uint32 storage is kept", field, stored_uint32, np.uint32,
         stored_uint32),
    ]
    for index, (case, image, mask, dtype, expected) in enumerate(cases):
        with subtests.test(msg=case):
            name = f"field{index}.tif"
            args = _write(tmp_path, name, image, mask=mask)
            mask_path = tmp_path / "masks" / name
            before = mask_path.read_bytes()
            grey, loaded = me.load_image_and_mask(*args)
            assert grey.shape == loaded.shape == image.shape
            assert loaded.dtype == dtype
            np.testing.assert_array_equal(loaded, expected)
            assert mask_path.read_bytes() == before

    # case, image, mask, ValueError fragments (the mask path is added)
    refusals = [
        ("three planes", field, np.zeros((4, 4, 3), np.uint8),
         ["Unsupported mask shape (4, 4, 3) in", "expected a 2-D label image"]),
        ("a shape that is not the image's", field, np.zeros((3, 3), np.uint8),
         ["Mask shape (3, 3) does not match image shape (4, 4) for"]),
        ("non-finite labels", field, np.full((4, 4), np.nan, np.float32),
         ["Mask contains non-finite values:"]),
        ("fractional labels", field, np.full((4, 4), 1.5, np.float32),
         ["Mask must contain non-negative integer labels:"]),
        ("negative float labels", field, np.full((4, 4), -1.0, np.float32),
         ["Mask must contain non-negative integer labels:"]),
        ("negative integer labels", field, np.full((4, 4), -1, np.int16),
         ["Mask must contain non-negative integer labels:"]),
        ("a label past uint32", field, np.full((4, 4), 2**32, np.int64),
         ["Mask label 4294967296 exceeds uint32 capacity:"]),
    ]
    for case, image, mask, fragments in refusals:
        with subtests.test(msg=f"refused: {case}"):
            name = f"{case}.tif"
            args = _write(tmp_path, name, image, mask=mask)
            mask_path = tmp_path / "masks" / name
            before = mask_path.read_bytes()
            with pytest.raises(ValueError) as raised:
                me.load_image_and_mask(*args)
            message = str(raised.value)
            for fragment in fragments:
                assert fragment in message
            if "does not match" in fragments[0]:
                assert message.endswith(f"for {name}.")
            else:
                assert str(mask_path) in message
            assert mask_path.read_bytes() == before
