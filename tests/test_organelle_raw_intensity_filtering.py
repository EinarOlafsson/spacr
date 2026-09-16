"""418: organelle backends filter original own-channel means after segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.object import generate_organelle_masks_sam
from tests.test_cov_object_organelle_sam import (
    _base_settings, _make_npz, BLOB_CENTERS, _blob_field,
)


@pytest.mark.parametrize("role,channel", [("organelle", 1), ("organelleb", 2)])
def test_organelle_bounds_use_original_slot_channel_not_preprocessed_npz(
        tmp_path, role, channel):
    src = tmp_path / "masks"
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    _make_npz(str(src), ["f1.npy"], channel_specs=("blob", "blob", "blob"))
    raw = np.stack([_blob_field(fg=1000)] * 3, axis=-1)
    yy, xx = np.indices(raw.shape[:2])
    for index, (cy, cx) in enumerate(BLOB_CENTERS):
        raw[(yy-cy)**2 + (xx-cx)**2 <= 3**2, channel] = [5, 20, 40, 100][index]
    raw_path = raw_dir / "f1.npy"
    np.save(raw_path, raw)
    unchanged = raw_path.read_bytes()
    settings = _base_settings(**{
        "number_of_organelles": 2,
        "organelle_channel": 1, "organelleb_channel": 2,
        "cellpose_organelle_channel": 1, "cellpose_organelleb_channel": 2,
        f"{role}_min_intensity": 20, f"{role}_max_intensity": 40,
        "organelle_gaussian_sigma": 0,
        "organelleb_gaussian_sigma": 0,
        "seg_qc": "off",
    })
    generate_organelle_masks_sam(str(src), settings, role)
    filtered = np.load(src / f"{role}_mask_stack" / "f1.npy")
    assert [bool(filtered[cy, cx]) for cy, cx in BLOB_CENTERS] == [False, True, True, False]
    assert set(np.unique(filtered)) == {0, 1, 2}
    assert raw_path.read_bytes() == unchanged

    settings.update({f"{role}_min_intensity": 0, f"{role}_max_intensity": 0})
    # Existing valid masks are intentionally skipped even with resume=False.
    # A fresh output root proves that disabling both bounds runs the backend
    # again and restores the removed objects, rather than rereading its cache.
    unfiltered_src = tmp_path / "unfiltered" / "masks"
    _make_npz(str(unfiltered_src), ["f1.npy"], channel_specs=("blob", "blob", "blob"))
    generate_organelle_masks_sam(str(unfiltered_src), settings, role)
    unfiltered = np.load(unfiltered_src / f"{role}_mask_stack" / "f1.npy")
    assert all(unfiltered[cy, cx] for cy, cx in BLOB_CENTERS)
    assert set(np.unique(unfiltered)) == {0, 1, 2, 3, 4}
    assert raw_path.read_bytes() == unchanged
