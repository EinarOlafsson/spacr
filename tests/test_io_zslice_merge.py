"""Ingest must not silently discard z-planes.

``_rename_and_organize_image_files`` groups source images by the key
``utils._extract_filename_metadata`` builds, which INCLUDES ``sliceID``. So the
behaviour depended on the metadata regex in a way nobody would guess:

* a regex with no ``sliceID`` group — every plane lands in one key, and the
  ``np.max(np.stack(images))`` inside the loop is a genuine maximum-intensity
  projection over the stack;
* a regex WITH ``sliceID`` (the CellVoyager and CQ1 conventions, and the
  bundled ``CV_REGEX``) — each plane gets its own key, so that projection is a
  no-op over a single image, and the per-channel assignment let each plane
  REPLACE the one before it. A 21-plane stack silently became one arbitrarily
  chosen plane, decided by ``os.listdir`` order, with no warning and nothing
  in the log.

That is silent data loss, and it is worse than the projection, which at least
sees every plane. These tests pin the combine.

Neither case *preserves* z — ingest still collapses it, and 3-D segmentation is
gated on exactly that (see ``spacr.zstack``). This is only about not throwing
planes away.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

CV_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
            r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
            r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")

#: The same convention with no sliceID group, so every plane shares one key.
NO_SLICE_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
                  r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
                  r"Z\d+C(?P<chanID>\d+)\.tif")


def _plane(folder, value, well="A01", field="001", z="01", chan="01"):
    """One CellVoyager-named plane whose pixels are all ``value``."""
    name = f"plate1_{well}_T0001F{field}L01A01Z{z}C{chan}.tif"
    tifffile.imwrite(str(folder / name),
                     np.full((6, 8), value, dtype=np.uint16))
    return name


def _ingest(src, regex=CV_REGEX):
    from spacr.io import _rename_and_organize_image_files
    return _rename_and_organize_image_files(
        str(src), regex, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)


def _stacks(src):
    return sorted(os.listdir(src / "stack"))


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "plate1"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [("01", "02", "03"),
                                   ("03", "02", "01"),
                                   ("02", "03", "01")])
def test_every_z_plane_reaches_the_merged_image(src, order):
    """The answer must not depend on which plane happened to be read last.

    The brightest plane is z02 in every ordering, so a "last one wins" ingest
    gives a different answer for each — which is exactly what it did.
    """
    values = {"01": 100, "02": 900, "03": 300}
    for z in order:
        _plane(src, values[z], z=z)
    # a second channel so the stack has a channel axis to index
    _plane(src, 50, z="01", chan="02")

    _ingest(src)

    stacks = _stacks(src)
    assert stacks == ["plate1_A01_1_1.npy"], stacks
    arr = np.load(src / "stack" / "plate1_A01_1_1.npy")
    assert int(arr[..., 0].max()) == 900


def test_a_regex_without_a_sliceid_group_still_projects(src):
    """The other half of the split: one key, so the in-loop MIP does the work."""
    for z, value in (("01", 100), ("02", 900), ("03", 300)):
        _plane(src, value, z=z)
    _plane(src, 50, z="01", chan="02")

    _ingest(src, NO_SLICE_REGEX)

    arr = np.load(src / "stack" / "plate1_A01_1_1.npy")
    assert int(arr[..., 0].max()) == 900


# ---------------------------------------------------------------------------
# Nothing else changed
# ---------------------------------------------------------------------------

def test_a_single_plane_is_unchanged(src):
    """The common case — no z at all — must behave exactly as before."""
    _plane(src, 250, z="01")
    _plane(src, 60, z="01", chan="02")

    _ingest(src)

    arr = np.load(src / "stack" / "plate1_A01_1_1.npy")
    assert arr.shape == (6, 8, 2)
    assert int(arr[..., 0].max()) == 250
    assert int(arr[..., 1].max()) == 60


def test_combining_is_per_channel(src):
    """Two channels must stay two channels, each maxed over its own planes."""
    _plane(src, 100, z="01", chan="01")
    _plane(src, 200, z="02", chan="01")
    _plane(src, 700, z="01", chan="02")
    _plane(src, 800, z="02", chan="02")

    _ingest(src)

    arr = np.load(src / "stack" / "plate1_A01_1_1.npy")
    assert arr.shape[-1] == 2
    assert int(arr[..., 0].max()) == 200
    assert int(arr[..., 1].max()) == 800


def test_two_fields_do_not_bleed_into_each_other(src):
    """The combine is keyed on the output name, which carries the field."""
    _plane(src, 100, field="001", z="01")
    _plane(src, 200, field="001", z="02")
    _plane(src, 900, field="002", z="01")
    for f in ("001", "002"):
        _plane(src, 10, field=f, z="01", chan="02")

    _ingest(src)

    stacks = _stacks(src)
    assert len(stacks) == 2, stacks
    maxima = sorted(int(np.load(src / "stack" / s)[..., 0].max()) for s in stacks)
    assert maxima == [200, 900]


def test_two_wells_do_not_bleed_into_each_other(src):
    _plane(src, 100, well="A01", z="01")
    _plane(src, 200, well="A01", z="02")
    _plane(src, 900, well="A02", z="01")
    for w in ("A01", "A02"):
        _plane(src, 10, well=w, z="01", chan="02")

    _ingest(src)

    stacks = _stacks(src)
    assert len(stacks) == 2, stacks
    maxima = sorted(int(np.load(src / "stack" / s)[..., 0].max()) for s in stacks)
    assert maxima == [200, 900]
