"""Item 799: a field measured without a pathogen mask writes no +inf column.

The distance transform of an all-zero mask is +inf everywhere, so before the
fix every ``*_pathogen_*`` distance column of a pathogen-free run was +inf and
broke every downstream mean, scaling and distance.
"""

import numpy as np

from spacr import measure as M


def _field():
    cell = np.zeros((32, 48), dtype=np.uint16)
    cell[2:30, 2:24] = 1
    cell[2:30, 24:46] = 2
    nucleus = np.zeros_like(cell)
    nucleus[12:18, 8:16] = 1
    nucleus[12:18, 30:38] = 2
    channels = np.random.default_rng(0).integers(
        0, 500, size=(32, 48, 2)).astype(np.uint16)
    return cell, nucleus, np.zeros_like(cell), channels


def _settings(**over):
    s = {'cell_mask_dim': 0, 'nucleus_mask_dim': 1, 'pathogen_mask_dim': None,
         'organelle_mask_dim': None, 'cytoplasm': False,
         'distance_gaussian_sigma': 1}
    s.update(over)
    return s


def _numeric_inf_columns(frame):
    numeric = frame.select_dtypes(include=[np.number])
    return [c for c in numeric.columns if np.isinf(numeric[c]).any()]


def test_no_pathogen_mask_dim_writes_no_pathogen_distance_column():
    cell, nucleus, pathogen, channels = _field()
    out = M._measure_intensity_distance(
        cell, nucleus, pathogen, channels, _settings())
    assert not [c for c in out.columns if 'pathogen' in c]
    assert out['cell_channel_0_distance_to_nucleus'].notna().all()
    assert _numeric_inf_columns(out) == []


def test_a_segmented_pathogen_absent_from_this_field_is_nan_not_inf():
    cell, nucleus, pathogen, channels = _field()
    out = M._measure_intensity_distance(
        cell, nucleus, pathogen, channels, _settings(pathogen_mask_dim=2))
    column = out['cell_channel_1_distance_to_pathogen']
    assert column.isna().all()
    assert _numeric_inf_columns(out) == []


def test_object_distances_skip_a_role_that_was_never_segmented():
    cell, nucleus, pathogen, channels = _field()
    frames = M._morphological_measurements(
        cell, nucleus, pathogen, None, None,
        _settings(object_distances=True), zernike=False,
        channel_arrays=channels)
    for frame in frames:
        if frame is None or not len(frame):
            continue
        assert not [c for c in frame.columns if 'to_pathogen' in c
                    or 'pathogen_centre' in c or 'pathogen_surface' in c]
        assert _numeric_inf_columns(frame) == []
