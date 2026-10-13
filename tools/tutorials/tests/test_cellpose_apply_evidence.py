from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cellpose_apply_evidence import normalize_field,minimum_area_labels,require_pixels


def test_normalization_is_per_field_clipped_and_preserves_input():
    image=np.array([[0,10,20],[30,40,50]],dtype=np.uint16)
    expected=np.array([[0,0,.3333333333333333],[.6666666666666666,1,1]])
    assert require_pixels(normalize_field(image,20,80),expected)==6
    assert image.tolist()==[[0,10,20],[30,40,50]]
    with pytest.raises(ValueError,match='independent expectation'):
        require_pixels(normalize_field(image,0,100),expected)


@pytest.mark.parametrize('case',['dimension','nonfinite','percentiles','constant'])
def test_bad_normalization_after_valid_example(case):
    image=np.array([[0.,1.],[2.,3.]])
    assert normalize_field(image,0,100).max()==1
    lo,hi=0,100
    if case=='dimension':image=image.ravel()
    elif case=='nonfinite':image[0,0]=np.nan
    elif case=='percentiles':lo,hi=99,2
    else:image[:]=1
    with pytest.raises(ValueError):normalize_field(image,lo,hi)


def test_area_boundary_renumbers_without_merging_adjacent_ids():
    raw=np.array([[0,7,7],[2,2,9]],dtype=np.int32)
    copy=raw.copy()
    assert minimum_area_labels(raw,0).tolist()==[[0,2,2],[1,1,3]]
    assert minimum_area_labels(raw,2).tolist()==[[0,2,2],[1,1,0]]
    assert not minimum_area_labels(raw,3).any()
    assert minimum_area_labels(raw,2).dtype==np.uint16
    assert np.array_equal(raw,copy)


@pytest.mark.parametrize('case',['dimension','dtype','negative','cutoff'])
def test_bad_mask_after_real_filter(case):
    mask=np.array([[0,2],[2,9]],dtype=np.int32);cutoff=2
    assert np.count_nonzero(minimum_area_labels(mask,cutoff))==2
    if case=='dimension':mask=mask.ravel()
    elif case=='dtype':mask=mask.astype(float)
    elif case=='negative':mask[0,0]=-1
    else:cutoff=-1
    with pytest.raises(ValueError):minimum_area_labels(mask,cutoff)


def test_mismatch_and_shape_checks_have_positive_counterpart():
    expected=np.arange(4).reshape(2,2)
    assert require_pixels(expected.copy(),expected)==4
    with pytest.raises(ValueError,match='independent expectation'):
        require_pixels(expected.ravel(),expected)
    with pytest.raises(ValueError,match='independent expectation'):
        require_pixels(expected+1,expected)
