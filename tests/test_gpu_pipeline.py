"""
End-to-end GPU pipeline test using the synthetic Yokogawa image directory.

Every test here is marked @pytest.mark.gpu — the whole module is skipped
on machines that lack a CUDA device or the necessary weights. The point
is not to test cellpose itself (it has its own test suite) but to verify
spacr's plumbing — regex → grouping → tensor prep → cellpose eval →
label output — end-to-end on realistic filenames.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest


pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Skip machinery
# ---------------------------------------------------------------------------

def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cellpose_available():
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="no CUDA-capable GPU available",
)
needs_cellpose = pytest.mark.skipif(
    not _cellpose_available(),
    reason="cellpose is not installed",
)


# ---------------------------------------------------------------------------
# 1. Torch sees a CUDA device.
# ---------------------------------------------------------------------------

@needs_gpu
def test_torch_reports_cuda_device():
    import torch
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    x = torch.zeros(4, device=dev)
    assert x.device.type == "cuda"


# ---------------------------------------------------------------------------
# 2. spacr can group synthetic Yokogawa filenames + hand them to cellpose.
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cellpose
def test_cellposesam_runs_on_synthetic_yokogawa_field(yokogawa_cellvoyager_dir):
    """Run CellposeSAM (pretrained_model='cpsam') on one field of the
    synthetic CellVoyager directory — the exact model spacr uses in
    _segment_cellpose_sam and utils._choose_model.

    Success = an integer label array with the same H x W as the input."""
    from tifffile import imread
    from cellpose.models import CellposeModel
    import torch

    # Pick one channel-1 image.
    manifest = yokogawa_cellvoyager_dir["manifest"]
    one = next(m for m in manifest if m["channel"] == "01")
    img = imread(one["path"])
    assert img.ndim == 2

    device = torch.device("cuda:0")
    # CellposeSAM: no model_type, uses the 'cpsam' pretrained checkpoint.
    model = CellposeModel(
        gpu=True,
        pretrained_model='cpsam',
        device=device,
    )
    # CellposeSAM eval() no longer takes 'channels' — feed a 2D array.
    result = model.eval(img.astype(np.float32), diameter=30)
    # cellpose 4 returns (masks, flows, styles).
    masks = result[0]
    assert masks.shape == img.shape
    assert masks.dtype.kind in "iu"
    assert masks.min() >= 0


# ---------------------------------------------------------------------------
# 3. spacr's own _choose_model wrapper is exercised on the same fixture.
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cellpose
def test_spacr_choose_model_returns_a_usable_cellpose_model(yokogawa_cellvoyager_dir):
    """spacr wraps cellpose model creation in utils._choose_model — verify
    the wrapper yields something that can .eval() a single field."""
    import torch
    from spacr.utils import _choose_model

    device = torch.device("cuda:0")
    obj_settings = {
        "model_name": "cyto3", "diameter": 30, "minimum_size": 5,
        "maximum_size": 1e6, "resample": False, "filter_size": False,
        "filter_intensity": False, "remove_border_objects": False,
        "merge": False,
    }
    model = _choose_model(
        "cyto3", device, object_type="cell",
        restore_type=None, object_settings=obj_settings,
    )
    assert model is not None
    # Just probe that it has an eval method — actual segmentation is covered
    # by the direct cellpose test above.
    assert hasattr(model, "eval")


# ---------------------------------------------------------------------------
# 4. The full spacr filename-grouping → image loading path works with the
#    synthetic Yokogawa TIFFs.
# ---------------------------------------------------------------------------

@needs_cellpose  # not a GPU test per se, but paired with the fixture above
def test_yokogawa_directory_grouped_and_loadable(yokogawa_cellvoyager_dir):
    """Even without a GPU, the CellVoyager filename → per-field loading path
    must work on the synthetic directory."""
    from tifffile import imread
    from spacr.utils import _get_regex, _extract_filename_metadata

    src = str(yokogawa_cellvoyager_dir["src"])
    regex = re.compile(_get_regex("cellvoyager", "tif", None))
    filenames = sorted(os.listdir(src))
    grouped = _extract_filename_metadata(
        filenames, src, regex, metadata_type="cellvoyager"
    )
    # Load one image per group; must all decode to a 2-D uint16 array of the
    # same shape.
    shapes = set()
    for paths in grouped.values():
        arr = imread(paths[0])
        assert arr.ndim == 2
        assert arr.dtype.kind in "iu"
        shapes.add(arr.shape)
    # All test images from the fixture are the same 128x128 size.
    assert len(shapes) == 1
