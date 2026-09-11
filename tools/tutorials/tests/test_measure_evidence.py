from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure_evidence import verify_field


def example():
    raw = np.zeros((4, 4, 7), dtype=np.uint16)
    raw[:2, :2, 4] = 1
    raw[2:, :2, 4] = 2
    raw[:, 2:, 4] = 3
    raw[:, :2, 6] = 9  # One child label crosses raw parent labels 1 and 2.
    for ch in range(4):
        raw[..., ch] = (np.arange(16).reshape(4, 4) + 1) * (ch + 1)
    rows = []
    for key, selected in ((1, raw[..., 4] < 3), (3, raw[..., 4] == 3)):
        row = {'object_label': key, 'cell_area': int(selected.sum())}
        for ch in range(4):
            pixels = raw[..., ch][selected]
            row[f'cell_channel_{ch}_integrated_intensity'] = int(pixels.sum())
            row[f'cell_channel_{ch}_mean_intensity'] = float(pixels.mean())
        rows.append(row)
    return raw, rows


def test_checks_the_merged_union_and_unchanged_object_against_pixels():
    raw, rows = example(); before = raw.copy(); saved = deepcopy(rows)
    result = verify_field(raw, rows, {1: 2})
    assert result['values_checked'] == 18
    assert result['source_supported_merges'][0]['source_bridges'] == [
        {'child_plane': 6, 'child_labels': [9]}]
    assert result['mean_intensity_maximum_error'] == 0
    assert np.array_equal(raw, before)
    assert rows == saved


@pytest.mark.parametrize('key,message', [
    ('cell_area', 'cell area'),
    ('cell_channel_2_integrated_intensity', 'integrated intensity'),
    ('cell_channel_3_mean_intensity', 'mean intensity')])
def test_wrong_saved_number_is_rejected(key, message):
    raw, rows = example(); rows[0][key] += 1
    with pytest.raises(ValueError, match=message): verify_field(raw, rows, {1: 2})


def test_nan_mean_is_not_a_passing_comparison():
    raw, rows = example(); rows[0]['cell_channel_0_mean_intensity'] = float('nan')
    with pytest.raises(ValueError, match='mean intensity'): verify_field(raw, rows, {1: 2})


def test_merge_without_source_child_bridge_is_rejected():
    raw, rows = example(); raw[..., 6] = 0
    with pytest.raises(ValueError, match='shared source child'): verify_field(raw, rows, {1: 2})


def test_source_union_cannot_be_silently_omitted():
    raw, rows = example()
    with pytest.raises(ValueError, match='cell area'): verify_field(raw, rows)


def test_duplicate_measured_identity_is_rejected():
    raw, rows = example(); rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match='Duplicate'): verify_field(raw, rows, {1: 2})


@pytest.mark.parametrize('merges,message', [({2: 1}, 'survivor'), ({1: 3}, 'counted twice'),
                                          ({1: 99}, 'source label is absent')])
def test_bad_merge_identity_is_rejected(merges, message):
    raw, rows = example()
    with pytest.raises(ValueError, match=message): verify_field(raw, rows, merges)


@pytest.mark.parametrize('kind', ['empty', 'wrong_dtype', 'wrong_planes', 'background'])
def test_empty_or_wrong_source_is_not_evidence(kind):
    raw, rows = example()
    if kind == 'empty': rows = []
    elif kind == 'wrong_dtype': raw = raw.astype(np.uint8)
    elif kind == 'wrong_planes': raw = raw[..., :6]
    else: rows[0]['object_label'] = 0
    with pytest.raises(ValueError): verify_field(raw, rows, {1: 2})
