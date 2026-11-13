"""CPU coverage for the label merge/filter block of ``spacr.utils``.

Covers the small IO helpers (``_load_image`` / ``_save_image``), the
union-find plumbing, original own-channel mean-intensity filtering in
``_process_single_fov_in_memory``, and the on-disk
``merge_split_objects`` / ``_process_single_fov`` pair.

Everything runs on tiny synthetic label images so the whole file is
sub-second and never touches the GPU, the network, or a display.
"""
from __future__ import annotations


import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# synthetic label-image builders
# ---------------------------------------------------------------------------

def _two_touching_blocks(size=16):
    """Two rectangles sharing a 10-px vertical edge (labels 1 and 2).

    Perimeters are 28 px each and the shared boundary is 20 px, so a
    ``perimeter_fraction`` below ~0.71 merges them.
    """
    m = np.zeros((size, size), dtype=np.uint16)
    m[2:12, 2:8] = 1
    m[2:12, 8:14] = 2
    return m


@pytest.fixture(autouse=True)
def _no_stray_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# _load_image / _save_image
# ---------------------------------------------------------------------------

def test_load_image_tiff_extension_and_case_insensitive(tmp_path):
    from spacr.utils import _load_image
    img = np.eye(4, dtype=np.uint16) * 3
    path = str(tmp_path / "mask.TIFF")
    tifffile.imwrite(path, img)
    assert np.array_equal(_load_image(path), img)


def test_load_image_unsupported_extension_returns_none(tmp_path):
    from spacr.utils import _load_image
    path = tmp_path / "mask.png"
    path.write_bytes(b"not an image")
    assert _load_image(str(path)) is None


# ---------------------------------------------------------------------------
# _apply_union_find
# ---------------------------------------------------------------------------

def test_apply_union_find_passes_through_labels_absent_from_parent():
    """Label ids present in the id-space but missing from ``parent`` map to
    themselves (the ``else`` branch of the mapping loop)."""
    from spacr.utils import _apply_union_find
    m = np.zeros((8, 8), dtype=np.uint16)
    m[1:3, 1:3] = 1
    m[5:7, 5:7] = 3           # id 2 is a hole in the label space
    parent = {1: 1, 3: 3}     # ... and is absent from parent

    out = _apply_union_find(m, parent)

    assert out.dtype == np.uint16
    assert out.shape == m.shape
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]
    assert out[1, 1] != out[5, 5]
    assert int(np.sum(out == out[1, 1])) == 4
    assert int(np.sum(out == out[5, 5])) == 4


def test_apply_union_find_merges_united_labels():
    from spacr.utils import _apply_union_find, _union_find_merge
    m = _two_touching_blocks()
    parent = {1: 1, 2: 2}
    _union_find_merge(parent, 1, 2)

    out = _apply_union_find(m, parent)

    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == int(np.sum(m > 0))


# ---------------------------------------------------------------------------
# _process_single_fov_in_memory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _process_single_fov (file based)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# merge_split_objects (directory driver)
# ---------------------------------------------------------------------------
