from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curate_evidence import painted_disk,check_mask,check_tracks


def test_disk_uses_radius_and_preserves_every_other_pixel():
    source=np.zeros((7,7),dtype=np.uint16)
    source[0,0]=9
    expected=source.copy()
    for y,x in [(2,3),(3,2),(3,3),(3,4),(4,3)]:expected[y,x]=18
    actual=painted_disk(source,(3,3),1,18)
    assert np.array_equal(actual,expected)
    assert np.count_nonzero(source)==1
    assert check_mask(actual,expected)['pixels']==49
    actual[0,0]=8
    with pytest.raises(ValueError,match='independent pixel'):
        check_mask(actual,expected)


@pytest.mark.parametrize('case',['dimension','dtype','radius','centre','label'])
def test_bad_brush_inputs_after_real_positive_counterpart(case):
    source=np.zeros((3,3),dtype=np.uint16)
    assert np.count_nonzero(painted_disk(source,(1,1),1,2))==5
    centre,radius,label=(1,1),1,2
    if case=='dimension':source=np.zeros(3,dtype=np.uint16)
    elif case=='dtype':source=source.astype(float)
    elif case=='radius':radius=0
    elif case=='centre':centre=(float('nan'),1)
    else:label=-1
    with pytest.raises(ValueError,match='finite 2-D'):
        painted_disk(source,centre,radius,label)


@pytest.mark.parametrize('case',['shape','dtype'])
def test_mask_shape_and_type_matter_after_positive_counterpart(case):
    mask=np.zeros((3,3),dtype=np.uint16)
    assert check_mask(mask,mask)['passed']
    changed=mask[:2] if case=='shape' else mask.astype(np.uint32)
    with pytest.raises(ValueError,match='actual mask differs'):
        check_mask(changed,mask)


def rows():
    return [dict(track_id=t,frame=f,original_label=t,x=10.25+f,y=20.5+t)
            for t in (2,3) for f in (0,1)]


def test_all_track_cells_and_row_order_are_checked():
    expected=rows()
    assert check_tracks(expected[::-1],expected)==dict(passed=True,rows=4,tracks=2,cells=20)
    split=deepcopy(expected)
    split[1]['track_id']=4
    assert check_tracks(split,split)['tracks']==3
    with pytest.raises(ValueError,match='identities'):
        check_tracks(split,expected)
    joined=deepcopy(split);joined[1]['track_id']=2
    assert check_tracks(joined,expected)['rows']==4


@pytest.mark.parametrize('case',['columns','duplicate','missing','coordinate','nonfinite'])
def test_invalid_track_output_is_detected_after_good_data(case):
    expected=rows(); actual=deepcopy(expected)
    assert check_tracks(actual,expected)['cells']==20
    if case=='columns':actual[0]['unexpected']=0
    elif case=='duplicate':actual.append(actual[0].copy())
    elif case=='missing':actual.pop()
    elif case=='coordinate':actual[0]['x']+=.01
    else:actual[0]['y']=float('nan')
    with pytest.raises(ValueError):check_tracks(actual,expected)
