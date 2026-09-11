"""Verify preserved preview evidence, including the exact area boundary."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_preview_evidence import check_arrays


def example():
    baseline = np.array([[0, 1, 2], [2, 3, 3], [3, 0, 0]], dtype=np.uint16)
    variant = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint16)
    outputs = {'cell': {'shape': [3, 3], 'objects': 3}}
    changes = {'filter': {'minimum_area': 3, 'before': 3, 'after': 1, 'restored': 3,
                          'model_rerun': False, 'raw_mask_unchanged': True},
               'model_option': {'rerun_completed': True, 'objects_after': 1,
                                'array_sha256': hashlib.sha256(variant.tobytes()).hexdigest()}}
    return baseline, variant, outputs, changes


def test_matching_native_trace_positive_including_exact_threshold():
    result = check_arrays(*example())
    assert result['after'] == 1 and result['before'] == result['restored'] == 3
    assert result['segmentation_quality_ranking_claimed'] is False


@pytest.mark.parametrize('key,value', [('after', 2), ('restored', 1),
    ('model_rerun', True), ('raw_mask_unchanged', False)])
def test_false_filter_trace_rejected_after_positive(key, value):
    check_arrays(*example())
    a, b, output, changes = example(); changes['filter'][key] = value
    with pytest.raises(ValueError): check_arrays(a, b, output, changes)


def test_changed_parameter_array_rejected_after_positive():
    check_arrays(*example())
    a, b, output, changes = example(); b[0, 0] = 1
    with pytest.raises(ValueError): check_arrays(a, b, output, changes)
