"""Item 1008 P5/P8: per-cell skeleton and legacy radial value contracts."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M


def _legacy_dataframe(radial_distributions, object_type):
    """Reproduce the pre-1008 radial conversion as an independent reference."""
    frame = pd.DataFrame()
    for (cell, label, channel), values in radial_distributions.items():
        for index in range(len(values)):
            frame.loc[label, f'{object_type}_rad_dist_channel_{channel}_bin_{index}'] = values[index]
        frame.loc[label, 'cell_id'] = cell
    return frame.reset_index().rename(columns={'index': 'label'})


@pytest.mark.parametrize('radial', [
    {},
    {(7, 11, 0): []},
    {(7, 11, 2): [0, 1, np.nan], (9, 3, 0): [2, 0], (7, 11, 0): [4, 5]},
    {(7, 11, 0): [1, 2, 3], (9, 3, 0): [4, 5], (2, 11, 0): [9]},
    {(7, 11, 0): [], (9, 3, 1): [4, 5]},
])
def test_radial_dataframe_preserves_values_columns_identity_and_order(radial):
    """Sparse bins and repeated objects retain every historical table value."""
    pd.testing.assert_frame_equal(M._create_dataframe(radial, 'nucleus'),
                                  _legacy_dataframe(radial, 'nucleus'))


def test_radial_dataframe_matches_the_934_fixture():
    """The existing Measure fixture keeps its six-bin radial measurements."""
    from tests.test_measure_ops_features_934 import _merged_field

    merged = _merged_field()
    radial = M._calculate_radial_distribution(merged[..., 3], merged[..., 4],
                                             merged[..., :3])
    pd.testing.assert_frame_equal(M._create_dataframe(radial, 'nucleus'),
                                  _legacy_dataframe(radial, 'nucleus'))


def _skeleton_objects(dtype):
    """Return interior, edge, touching, sparse and one-pixel cell labels."""
    labels = np.zeros((160, 180), np.int64)
    labels[65:88, 75:98] = 3
    labels[:18, :20] = 7
    labels[140:, 155:] = 11
    labels[80:99, 98:114] = 29
    labels[28, 145] = 1000000
    image = np.random.default_rng(1008).integers(1, 1000, (*labels.shape, 3)).astype(dtype)
    image[..., 1] = 0
    return labels, image


@pytest.mark.parametrize('dtype', [np.uint16, np.float32, np.float64])
def test_skeleton_matches_whole_field_per_object_and_channel(dtype):
    """Crop execution exactly reproduces the legacy helper for each cell."""
    labels, image = _skeleton_objects(dtype)
    result = M.measure_cytoskeleton(labels, image, [2, 0, 1]).set_index('label')
    assert list(result.index) == [3, 7, 11, 29, 1000000]
    assert list(result.columns) == [f'channel_{ch}_{stat}' for ch in (2, 0, 1)
                                    for stat in ('skeleton_length', 'skeleton_branch_points')]
    for label in result.index:
        for channel in (2, 0, 1):
            reference = M._analyze_cytoskeleton(image, labels == label, channel)
            for statistic in ('skeleton_length', 'skeleton_branch_points'):
                expected = 0 if reference.empty else reference.iloc[0][statistic]
                assert result.loc[label, f'channel_{channel}_{statistic}'] == expected


def test_skeleton_channel_names_indices_and_all_channels_agree():
    """Channel selections resolve to the same numerical image indices."""
    labels, image = _skeleton_objects(np.uint16)
    indexed = M.measure_cytoskeleton(labels, image, [2, 0])
    named = M.measure_cytoskeleton(labels, image, 'AF750,DAPI',
                                 channel_names=['DAPI', 'GFP', 'AF750'])
    pd.testing.assert_frame_equal(indexed, named)
    pd.testing.assert_frame_equal(M.measure_cytoskeleton(labels, image, None),
                                  M.measure_cytoskeleton(labels, image, [0, 1, 2]))


def test_skeleton_unknown_name_uses_translated_actionable_message(monkeypatch):
    """Invalid names stop before measurement and pass through the translator."""
    from spacr.qt import i18n

    messages = []

    def translated(message, **values):
        """Record the canonical runtime template and interpolate its values."""
        messages.append(message)
        return 'translated: ' + message.format(**values)

    monkeypatch.setattr(i18n, 'tr', translated)
    labels, image = _skeleton_objects(np.uint16)
    with pytest.raises(ValueError, match='translated: .*missing.*channel_names'):
        M.measure_cytoskeleton(labels, image, ['missing'], channel_names=['DAPI', 'GFP', 'AF750'])
    assert len(messages) == 1


@pytest.mark.parametrize('channel', [-1, 3, 0.5, True])
def test_skeleton_invalid_channel_index_stops(channel):
    """Invalid image-channel identities cannot select a different plane."""
    labels, image = _skeleton_objects(np.uint16)
    with pytest.raises(ValueError):
        M.measure_cytoskeleton(labels, image, [channel])


def test_skeleton_empty_and_single_signal_objects_keep_their_rows():
    """Background-only and one-pixel signals produce finite zero statistics."""
    labels = np.zeros((40, 42), np.int64)
    labels[1:8, 2:9] = 4
    labels[25, 30] = 9
    image = np.zeros((*labels.shape, 1))
    image[25, 30, 0] = 1
    result = M.measure_cytoskeleton(labels, image, None)
    assert list(result['label']) == [4, 9]
    assert np.all(result.drop(columns='label').to_numpy() == 0)
    empty = M.measure_cytoskeleton(np.zeros_like(labels), image, None)
    assert empty.empty
    assert list(empty.columns) == list(result.columns)


def test_skeleton_public_arrival_contract_equals_source():
    """The exact public docstring ships with the new callable."""
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / 'tests/data/release_contracts/1008c_cytoskeleton_api_arrivals_2026-11-21.json').read_text())
    tree = ast.parse((root / 'spacr/measure.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'measure_cytoskeleton')
    assert contract == {'spacr.measure.measure_cytoskeleton': ast.get_docstring(function)}


@pytest.mark.parametrize('dtype', [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('placement', ['interior', 'top_left', 'bottom_right',
                                       'disconnected', 'ring', 'full_field'])
@pytest.mark.parametrize('signal', ['random', 'constant', 'gradient', 'zero', 'one_pixel'])
def test_corrected_skeleton_exact_crop_parity_in_ninety_cases(dtype, placement, signal):
    """Local thresholding keeps exact results across shape, signal and dtype cases."""
    mask = np.zeros((160, 180), bool)
    if placement == 'interior':
        mask[62:87, 77:102] = True
    elif placement == 'top_left':
        mask[:25, :25] = True
    elif placement == 'bottom_right':
        mask[-25:, -25:] = True
    elif placement == 'disconnected':
        mask[20:35, 30:42] = True
        mask[95:110, 120:135] = True
    elif placement == 'ring':
        mask[62:87, 77:102] = True
        mask[67:82, 82:97] = False
    else:
        mask[:] = True
    image = np.random.default_rng(1008).integers(0, 1000, (*mask.shape, 1)).astype(dtype)
    if signal == 'constant':
        image[mask, 0] = 100
    elif signal == 'gradient':
        image[..., 0] = np.indices(mask.shape).sum(axis=0)
    elif signal in ('zero', 'one_pixel'):
        image[mask, 0] = 0
        if signal == 'one_pixel':
            rows, columns = np.nonzero(mask)
            image[rows[0], columns[0], 0] = 1
    measured = M.measure_cytoskeleton(mask, image, [0]).iloc[0]
    reference = M._analyze_cytoskeleton(image, mask, 0)
    for statistic in ('skeleton_length', 'skeleton_branch_points'):
        expected = 0 if reference.empty else reference.iloc[0][statistic]
        assert measured[f'channel_0_{statistic}'] == expected


def test_skeleton_never_skeletonizes_pixels_outside_the_object(monkeypatch):
    """A positive local offset cannot turn distant zero pixels into object skeleton."""
    mask = np.zeros((180, 200), bool)
    mask[70:90, 80:100] = True
    image = np.zeros((*mask.shape, 1))
    image[mask, 0] = np.random.default_rng(1008).integers(1, 1000, mask.sum())
    skeletonize = M.morphology.skeletonize
    inputs, skeletons = [], []

    def recorded(values):
        """Record actual skeleton input and output without changing the calculation."""
        inputs.append(np.asarray(values).copy())
        result = skeletonize(values)
        skeletons.append(result.copy())
        return result

    monkeypatch.setattr(M.morphology, 'skeletonize', recorded)
    corrected = M._analyze_cytoskeleton(image, mask, 0)
    legacy = M._analyze_cytoskeleton(image, mask, 0, object_support=False)
    assert not inputs[0][~mask].any()
    assert not skeletons[0][~mask].any()
    assert inputs[1][~mask].any()
    assert skeletons[1][~mask].any()
    assert corrected.iloc[0]['skeleton_length'] == 3
    assert legacy.iloc[0]['skeleton_length'] == 23
