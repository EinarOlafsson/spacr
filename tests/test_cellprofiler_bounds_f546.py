"""CellProfiler centres outside a field never borrow its edge object IDs."""

import numpy as np
import pandas as pd
import pytest

from spacr import measure


@pytest.mark.parametrize('object_name', ['Cells', 'Speckles'])
@pytest.mark.parametrize('case', ['outside', 'nonfinite', 'inside'])
def test_centres_match_only_inside_the_image_preserving_rows_and_features(
        tmp_path, object_name, case):
    # Every edge is labelled, making accidental clamping observable. The
    # unknown object name also exercises mask-role inference using hit counts.
    labels = np.arange(1, 21, dtype=np.uint16).reshape(4, 5)
    labels[1, 1] = 0
    merged = tmp_path / 'merged'
    merged.mkdir()
    stem = 'plate1_A01_1'
    np.save(merged / f'{stem}.npy', labels[..., None])
    if case == 'outside':
        centres = [(-0.01, 2), (2, -0.01), (5, 2), (2, 4),
                   (-100, 2), (100, 2), (2, -100), (2, 100)]
        expected = [None] * len(centres)
    elif case == 'nonfinite':
        centres = [(float('nan'), 2), (2, float('nan')),
                   (float('inf'), 2), (2, float('inf')),
                   (-float('inf'), 2), (2, -float('inf'))]
        expected = [None] * len(centres)
    else:
        centres = [(0, 0), (4, 3), (0.1, 0.1), (1.6, 2.4),
                   (4.99, 3.99), (1.5, 2.5), (0.5, 0.5), (1, 1)]
        # Preserve existing nearest-pixel / ties-to-even rounding, including
        # valid subpixels which round past the last integer pixel coordinate.
        expected = [1, 20, 1, 13, 20, 13, 1, None]
    values = np.array([[1, i + 1, x, y, 100 + i]
                       for i, (x, y) in enumerate(centres)], dtype=float)
    output = tmp_path / 'objects.npy'
    np.save(output, values)
    reply = {'images': {'1': [f'{stem}_ch0.tif']}, 'objects': {
        object_name: {'columns': ['ImageNumber', 'ObjectNumber',
                                 'Location_Center_X', 'Location_Center_Y',
                                 'AreaShape_Area'], 'path': str(output)}}}
    settings = {'cell_mask_dim': 0, 'nucleus_mask_dim': None,
                'pathogen_mask_dim': None, 'timelapse': False}

    tables = measure._cellprofiler_tables(reply, str(merged), settings)

    frame = tables[f'cellprofiler_{object_name.lower()}']
    assert len(frame) == len(centres)
    assert [-1 if pd.isna(value) else value for value in frame['object_label']] == [
        -1 if label is None else label for label in expected]
    assert frame['prcfo'].fillna('').tolist() == [
        '' if label is None else f'plate1_r1_c1_f1_o{label}'
        for label in expected]
    assert frame['cp_object_number'].tolist() == list(range(1, len(centres) + 1))
    assert frame['cp_AreaShape_Area'].tolist() == values[:, 4].tolist()
    np.testing.assert_equal(frame['cp_Location_Center_X'].to_numpy(), values[:, 2])
    np.testing.assert_equal(frame['cp_Location_Center_Y'].to_numpy(), values[:, 3])
    np.testing.assert_array_equal(np.load(merged / f'{stem}.npy')[..., 0], labels)
