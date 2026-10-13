"""Do not accept legacy guessed identity as database-level split evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_classify_split import plan, marker_profile, TRAIN_WELLS, TEST_WELLS


def rows():
    return [dict(prcfo='_'.join(well) + f'_f1_o{label}',
                 plateID=well[0], rowID=well[1], columnID=well[2], infected=label,
                 png_path=f'/legacy/{well[1]}_{well[2]}_{label}.png')
            for well in sorted(TRAIN_WELLS | TEST_WELLS) for label in (1, 2)]


def test_positive_preserves_metadata_classes_and_whole_wells():
    original = rows()
    selected = plan(original, 1)
    assert len(selected) == 8
    for split, wells in [('train', TRAIN_WELLS), ('test', TEST_WELLS)]:
        assert {tuple(r['well']) for r in selected if r['split'] == split} == wells
        assert {r['class_name'] for r in selected if r['split'] == split} == {'infected_1', 'infected_2'}
    assert original == rows()
    assert selected == plan(list(reversed(original)), 1)


@pytest.mark.parametrize('defect', ['wrong_well', 'duplicate', 'legacy_name', 'missing_class', 'missing_label'])
def test_rejects_named_metadata_defects(defect):
    records = deepcopy(rows())
    if defect == 'wrong_well':
        # Swap two SAME-CLASS metadata rows: group counts stay valid, so
        # only the named canonical-identity guard can reject the defect.
        records[0]['rowID'], records[4]['rowID'] = records[4]['rowID'], records[0]['rowID']
    elif defect == 'duplicate':
        records.append(dict(records[0]))
    elif defect == 'legacy_name':
        records[0]['prcfo'] = 'plate1_E01_10_1_4'
    elif defect == 'missing_class':
        records.pop()
    else:
        records[0]['infected'] = None
    with pytest.raises(ValueError):
        plan(records, 1)


def test_marker_timestamps_do_not_change_the_pixel_format():
    marker = dict(spacr_crop_format=3, channel_order='declared_rgb', narrowing='high-byte')
    assert marker_profile(json.dumps(dict(marker, updated_utc='first'))) == marker_profile(json.dumps(dict(marker, updated_utc='later')))


def test_marker_format_change_is_not_just_a_timestamp_change():
    with pytest.raises(ValueError):
        marker_profile(json.dumps(dict(spacr_crop_format=2, channel_order='declared_rgb', narrowing='high-byte')))
