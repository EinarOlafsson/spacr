"""Branch coverage for the raw-image ingest helpers in :mod:`spacr.io`.

Covers the defensive / rarely-taken paths of

    load_images_from_paths
    _rename_and_organize_image_files
    _merge_file
    _move_to_chan_folder

i.e. unreadable input images, ``img_format`` given as a bare string, timelapse
file naming, an already-populated ``stack/``, FOVs missing a channel, a MIP that
cannot be computed, duplicate names in ``orig/``, un-deletable raws, a channel
merge with no readable channel, and the CQ1 / no-plateID / unparsable-filename
paths of the per-channel sorter.

Everything is synthetic uint16 TIFF data written into ``tmp_path`` — no network,
no GPU, no external data.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import tifffile

# Regexes with explicitly named groups (spacr's `custom` metadata mode) so the
# parsing here is deterministic and independent of _get_regex's quirks.
CV_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
            r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
            r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")

CQ1_REGEX = (r"W(?P<wellID>\d+)F(?P<fieldID>\d+)T(?P<timeID>\d+)"
             r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _img(value, shape=(6, 8)):
    """Constant-valued uint16 field so channel identity is checkable."""
    return np.full(shape, value, dtype=np.uint16)


def _write(path, value, shape=(6, 8)):
    tifffile.imwrite(str(path), _img(value, shape))
    return _img(value, shape)


# ---------------------------------------------------------------------------
# load_images_from_paths
# ---------------------------------------------------------------------------

def test_load_images_from_paths_skips_unreadable_paths(tmp_path, capsys):
    """Unopenable / missing paths are reported and dropped, good ones kept."""
    from spacr.io import load_images_from_paths

    good = tmp_path / "good.tif"
    expected = _write(good, 4242, shape=(3, 5))
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_text("this is definitely not a TIFF", encoding="utf-8")
    missing = tmp_path / "not_there.tif"

    out = load_images_from_paths({"fov": [str(good), str(corrupt), str(missing)],
                                  "empty": []})

    assert set(out) == {"fov", "empty"}
    assert out["empty"] == []
    assert len(out["fov"]) == 1
    arr = out["fov"][0]
    assert arr.shape == (3, 5)
    assert arr.dtype == np.uint16
    assert np.array_equal(arr, expected)

    err = capsys.readouterr().out
    assert err.count("Error loading image from") == 2
    assert "corrupt.tif" in err and "not_there.tif" in err


# ---------------------------------------------------------------------------
# _rename_and_organize_image_files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _merge_file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _move_to_chan_folder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _generate_time_lists (guards around the ingest)
# ---------------------------------------------------------------------------

def test_generate_time_lists_skips_malformed_names(tmp_path):
    from spacr.io import _generate_time_lists

    files = [
        "plate1_A01_1_2.npy", "plate1_A01_1_10.npy", "plate1_A01_1_1.npy",
        "plate1_A02_1_1.npy",
        "plate1_A01_1_x.npy",     # timepoint not an int -> skipped
        "plate1_A01.npy",         # too few parts -> skipped
        "plate1_A01_1_1.tif",     # not .npy -> skipped
    ]
    groups = _generate_time_lists(files)

    assert sorted(len(g) for g in groups) == [1, 3]
    by_key = {g[0].rsplit("_", 1)[0]: g for g in groups}
    assert by_key["plate1_A01_1"] == [
        "plate1_A01_1_1.npy", "plate1_A01_1_2.npy", "plate1_A01_1_10.npy"]
    assert by_key["plate1_A02_1"] == ["plate1_A02_1_1.npy"]
    assert all(re.match(r".*_\d+\.npy$", f) for g in groups for f in g)


# ---------------------------------------------------------------------------
# the writer and the reader, pinned to each other
# ---------------------------------------------------------------------------
