"""Item 516: measured diameters must come out cell > nucleus > pathogen.

On Mask generation's test plate the estimator reported cell 7-8 px, nucleus
91 px and pathogen 11 px. The curated masks shipped with the same plate say
175, 95 and 32 px. The nucleus stain is solid and was measured right; the
cell and pathogen stains threshold into a few whole objects plus hundreds of
4-10 px specks, and a plain median over components counted every speck as an
object. The estimator now sets specks aside against the area-weighted typical
object before taking the median.

Two checks: a synthetic plate built the same way (whole objects plus many
specks that hold almost no area) that always runs, and the real cached test
plate, measured against its own ground-truth masks, when it is on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacr.diameter import _region_diameters, _without_specks, estimate_diameters

CHANNELS = {"cell": 1, "nucleus": 0, "pathogen": 2}


def _disc(img, yy, xx, cy, cx, diameter, value=3000.0):
    """Add a solid disc of ``diameter`` px at ``(cy, cx)``."""
    r = diameter / 2.0
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] += value


def _speckled_plate(root, n_fields=3, size=960, cell=150, nucleus=90, pathogen=30):
    """Write a ``merged/`` plate of whole objects plus specks in cell and pathogen.

    Every field holds a 4 x 4 grid of cells; each cell has a nucleus at its
    centre and one pathogen off-centre. The cell and pathogen channels also
    carry many 6 px specks, which outnumber the objects but hold little area,
    as the test plate's mitochondrial cell stain and parasite marker do.
    """
    merged = Path(root) / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    step = size // 4
    for f in range(n_fields):
        rng = np.random.default_rng(516 + f)
        planes = [np.full((size, size), 200.0, np.float32) for _ in range(3)]
        for gy in range(4):
            for gx in range(4):
                cy = step // 2 + gy * step
                cx = step // 2 + gx * step
                _disc(planes[1], yy, xx, cy, cx, cell)
                _disc(planes[0], yy, xx, cy, cx, nucleus)
                _disc(planes[2], yy, xx, cy + 55, cx, pathogen)
        for plane, n_specks in ((planes[1], 80), (planes[2], 40)):
            for _ in range(n_specks):
                cy, cx = rng.integers(8, size - 8, 2)
                _disc(plane, yy, xx, cy, cx, 6)
        stack = []
        for plane in planes:
            plane += rng.normal(0, 25, plane.shape)
            stack.append(np.clip(plane, 0, 65535).astype(np.uint16))
        np.save(merged / f"plate1_A{f + 1:02d}_1_1.npy", np.stack(stack, axis=-1))
    return str(root), {"cell": cell, "nucleus": nucleus, "pathogen": pathogen}


def _assert_ordered_and_close(est, reference):
    """Cell > nucleus > pathogen, each within 20 % of its reference."""
    for obj in CHANNELS:
        assert est[obj].usable, est[obj].note
    got = {obj: est[obj].diameter for obj in CHANNELS}
    assert got["cell"] > got["nucleus"] > got["pathogen"], got
    for obj, ref in reference.items():
        assert abs(got[obj] - ref) <= 0.2 * ref, (obj, got[obj], ref, est[obj].note)


def test_specks_do_not_decide_the_diameter(tmp_path):
    """Whole objects set the diameter even when specks outnumber them."""
    src, truth = _speckled_plate(tmp_path)
    est = estimate_diameters(src, CHANNELS, n_fields=3)
    _assert_ordered_and_close(est, truth)
    assert "specks" in est["cell"].note


def test_without_specks_keeps_a_real_second_population():
    """Small objects that carry a real share of the stain are not specks."""
    specks = np.concatenate([np.full(200, 6.0), np.full(20, 150.0)])
    assert set(_without_specks(specks)) == {150.0}
    two_sizes = np.concatenate([np.full(81, 12.0), np.full(9, 60.0)])
    assert _without_specks(two_sizes).size == two_sizes.size
    assert _without_specks(np.array([5.0])).tolist() == [5.0]


def _cached_test_plate():
    """The Mask generation test plate's ``merged/`` arrays, when downloaded."""
    root = Path.home() / ".cache" / "spacr" / "example_data" / "plate1"
    arrays = sorted((root / "merged").glob("*.npy"))
    if not arrays:
        return None, []
    try:
        if np.load(arrays[0], mmap_mode="r").shape[-1] < 7:
            return None, []
    except Exception:
        return None, []
    return root, arrays


def _mask_reference(arrays):
    """Median diameter per object type from the plate's curated masks.

    Planes 4, 5 and 6 of each merged array are the cell, nucleus and pathogen
    label masks. Objects touching the border are dropped, as the estimator
    drops them.
    """
    pools = {"cell": [], "nucleus": [], "pathogen": []}
    for path in arrays:
        arr = np.load(path, mmap_mode="r")
        for obj, plane in (("cell", 4), ("nucleus", 5), ("pathogen", 6)):
            labels = np.asarray(arr[..., plane]).astype(np.int64)
            pools[obj].append(
                _region_diameters(labels, int(labels.max()), 0.0, float(labels.size))
            )
    return {obj: float(np.median(np.concatenate(v))) for obj, v in pools.items()}


def test_the_mask_generation_test_plate_measures_in_biological_order():
    """The real test plate: cell > nucleus > pathogen, near the curated masks."""
    root, arrays = _cached_test_plate()
    if root is None:
        pytest.skip("the Mask generation test plate is not downloaded")
    reference = _mask_reference(arrays)
    assert reference["cell"] > reference["nucleus"] > reference["pathogen"]
    est = estimate_diameters(str(root), CHANNELS, n_fields=5)
    _assert_ordered_and_close(est, reference)
