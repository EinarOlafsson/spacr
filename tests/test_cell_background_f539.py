"""Measured cell background excludes foreground and feeds bleaching correction."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

from spacr import measure as M
from spacr.feature_dict import search_features
from spacr.timelapse import _bleach_correct_table


def settings():
    return dict(radial_dist=False, calculate_correlation=False,
                homogeneity=False, homogeneity_distances=[],
                distance_gaussian_sigma=0, cell_mask_dim=1,
                nucleus_mask_dim=None, pathogen_mask_dim=None,
                measure_gpu=False)


def masks():
    labels = np.zeros((24, 28), dtype=np.uint16)
    labels[7:13, 7:13] = 7
    labels[7:13, 13:19] = 42
    return labels


def measured(labels, image, outside=True):
    empty = np.zeros_like(labels)
    return M._intensity_measurements(
        labels, empty, empty, empty, empty, image[..., None], settings(),
        periphery=False, outside=outside)[0]


@pytest.mark.parametrize('spacing', [None, (4., 1., 1.)])
def test_cell_ring_matches_independent_background_mask_with_neighbour(spacing):
    labels = masks()
    if spacing is not None:
        labels = np.stack([labels] * 3)
    image = np.arange(labels.size, dtype=float).reshape(labels.shape) + 100
    image[labels > 0] = 100000
    original = image.copy()
    actual = M._outside_intensity(labels, image, exclude_foreground=True,
                                  spacing=spacing)
    for row in actual:
        region = labels == row[0]
        ring = (binary_dilation(region, iterations=5) if spacing is None
                else distance_transform_edt(~region, sampling=spacing) <= 5)
        values = image[ring & (labels == 0)]
        np.testing.assert_allclose(row[1:], [values.mean(), *np.percentile(
            values, [5, 10, 25, 50, 75, 85, 95])])
    assert [row[0] for row in actual] == [7, 42]
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('labels', [np.full((8, 9), 7),
                                    np.tile([7, 7, 42, 42], (8, 1))])
def test_fully_foreground_fields_retain_all_labels_with_missing_background(labels):
    actual = M._outside_intensity(labels, np.full(labels.shape, 100.),
                                  exclude_foreground=True)
    assert [row[0] for row in actual] == list(np.unique(labels))
    assert all(np.isnan(row[1:]).all() for row in actual)


def test_historical_child_ring_still_includes_neighbour_signal():
    labels = masks()
    image = np.full(labels.shape, 50.)
    image[labels > 0] = 1000
    child = M._outside_intensity(labels, image)
    cell = M._outside_intensity(labels, image, exclude_foreground=True)
    assert all(row[1] > 50 for row in child)
    assert all(row[1] == 50 for row in cell)


def test_actual_measure_emits_background_without_changing_existing_values():
    labels = masks()
    image = np.full(labels.shape, 33000., dtype=np.float32)
    image[labels == 7] += 600
    image[labels == 42] += 1200
    prior = measured(labels, image, outside=False)
    result = measured(labels, image)
    pd.testing.assert_frame_equal(result[prior.columns], prior)
    assert result['cell_channel_0_outside_percentile_50'].tolist() == [33000.] * 2
    assert result['cell_channel_0_outside_region_label'].tolist() == [7, 42]


def test_actual_measure_and_correction_keep_missing_background_unknown():
    labels = np.full((12, 12), 7, dtype=np.uint16)
    frame = measured(labels, np.full(labels.shape, 33600., dtype=np.float32))
    assert frame['cell_channel_0_outside_percentile_50'].isna().all()
    frame = frame.rename(columns={'label': 'object_label'})
    frame['timeID'] = 't0'
    corrected, _ = _bleach_correct_table(frame, 'cell', 'ratio')
    assert corrected['object_label'].tolist() == [7]
    assert corrected['cell_channel_0_mean_intensity'].isna().all()


def test_measured_cell_background_restores_fluorescence_not_camera_offset():
    labels = masks()
    frames = []
    for time, scale in enumerate([1., .75, .5]):
        image = np.full(labels.shape, 33000., dtype=np.float32)
        image[labels == 7] += 600 * scale
        image[labels == 42] += 1200 * scale
        frame = measured(labels, image).rename(columns={'label': 'object_label'})
        frame['timeID'] = f't{time}'
        frame['cell_area'] = 36.
        frame['plateID'] = 'plate1'
        frame['rowID'] = 'r1'
        frame['columnID'] = 'c1'
        frame['fieldID'] = 'f1'
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    original = source.copy(deep=True)
    corrected, fits = _bleach_correct_table(source, 'cell', 'ratio')
    np.testing.assert_allclose(corrected['cell_channel_0_mean_intensity'],
                               [33600., 34200.] * 3)
    np.testing.assert_allclose(corrected['cell_channel_0_integrated_intensity'],
                               np.array([33600., 34200.] * 3) * 36)
    assert fits['background'].tolist() == ['cell_channel_0_outside_percentile_50']
    pd.testing.assert_frame_equal(source, original)


def test_feature_search_exposes_only_cell_outside_not_cell_periphery():
    outside = search_features('outside', object_type='cell')
    assert {'outside_mean', 'outside_percentile_<p>'} <= {x.doc.key for x in outside}
    assert 'periphery_mean' not in {
        x.doc.key for x in search_features('periphery', object_type='cell')}
