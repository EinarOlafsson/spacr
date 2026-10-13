from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compose_anndata_capture import check_sources


def sources():
    return (dict(accepted=True, gui_exports_completed=False, original_unchanged=True,
                 private_measurements_unchanged=True, remaining_workers=False, exports=[]),
            dict(accepted=True, gui_defect_fixed=False, source_unchanged=True,
                 exports=[dict(accepted=True, shape=[2, 3], matrix_cells_checked=6,
                               maximum_float32_discrepancy=0)]*6))


def test_verified_api_and_navigation_only_gui_are_distinguished():
    check_sources(*sources())


@pytest.mark.parametrize('which,key,value', [
    (0, 'accepted', False), (0, 'gui_exports_completed', True),
    (0, 'original_unchanged', False), (0, 'private_measurements_unchanged', False),
    (0, 'remaining_workers', True), (0, 'exports', [{}]),
    (1, 'accepted', False), (1, 'gui_defect_fixed', True),
    (1, 'source_unchanged', False), (1, 'exports', [{}]*5),
    (1, 'exports', [{}]*6)])
def test_partial_or_mislabelled_evidence_is_rejected(which, key, value):
    records = deepcopy(sources()); records[which][key] = value
    with pytest.raises(ValueError): check_sources(*records)


@pytest.mark.parametrize('key,value', [('accepted', False), ('shape', [0, 3]),
    ('matrix_cells_checked', 1), ('maximum_float32_discrepancy', 1)])
def test_every_export_needs_full_independent_matrix_evidence(key, value):
    records = deepcopy(sources()); records[1]['exports'][0][key] = value
    with pytest.raises(ValueError, match='complete independent matrix'):
        check_sources(*records)
