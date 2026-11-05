"""The bounding-box lookups give the full-image result, with or without boxes.

``get_components``, ``_map_child_to_parent`` and ``_measure_intensity_distance``
look at each object inside its own bounding box. The oracle here is the same
function with the boxes taken away, which falls back to the whole field, so the
two paths must agree exactly on a small multi-object label image.
"""
import numpy as np
import pandas as pd

import spacr.measure as M


def _field():
    rng = np.random.default_rng(0)
    cell = np.zeros((60, 70), np.int32)
    nuc = np.zeros_like(cell)
    path = np.zeros_like(cell)
    k = 0
    for r in range(3, 50, 22):
        for c in range(3, 60, 26):
            k += 1
            cell[r:r + 18, c:c + 20] = k
            nuc[r + 3:r + 9, c + 3:c + 9] = k
            if k % 2:
                path[r + 8:r + 14, c + 9:c + 17] = k + 100
    path[cell == 0] = 0
    channels = rng.random((60, 70, 2))
    return cell, nuc, path, channels


def _run(cell, nuc, path, channels):
    return (
        M.get_components(cell, nuc, path),
        M._map_child_to_parent(nuc, cell),
        M._map_child_to_parent(path, cell, 'pathogen'),
        M._measure_intensity_distance(
            cell, nuc, path, channels,
            {'distance_gaussian_sigma': 2, 'verbose': False}),
    )


def test_bounding_box_path_equals_full_image_path(monkeypatch):
    args = _field()
    with_boxes = _run(*args)
    monkeypatch.setattr(M, '_label_bounding_boxes', lambda _mask: {})
    without_boxes = _run(*args)
    for got, want in zip(with_boxes, without_boxes):
        if isinstance(got, pd.DataFrame):
            pd.testing.assert_frame_equal(got, want, check_exact=True)
        else:
            for g, w in zip(got, want):
                pd.testing.assert_frame_equal(g, w, check_exact=True)
