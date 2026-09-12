"""Small synthetic UNIT fixtures verify checks, never tutorial recordings."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_compare_example import independently_check
from spacr.model_compare import compare_masks


@pytest.mark.parametrize('defect', [None, 'count', 'matched_count', 'iou', 'duplicate', 'fraction'])
def test_actual_pixel_checks_then_each_incorrect_report(defect):
    a = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    b = a.copy(); b[0, 1] = 0
    row = compare_masks(a, b)
    assert independently_check(a, b, row) == {
        'objects_a': 2, 'objects_b': 2, 'matched_pairs_pixel_checked': 2,
        'foreground_disagreement_pixels': 1}
    if defect is None:
        return
    row = deepcopy(row)
    if defect == 'count':
        row.n_objects_a += 1
    elif defect == 'matched_count':
        row.n_matched += 1
    elif defect == 'iou':
        x, y, score = row.matches[0]; row.matches[0] = (x, y, score / 2)
    elif defect == 'duplicate':
        row.matches[1] = row.matches[0]
    elif defect == 'fraction':
        row.iou_matched_fraction /= 2
    with pytest.raises(ValueError):
        independently_check(a, b, row)
