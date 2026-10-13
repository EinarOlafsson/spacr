"""Independent arithmetic for the bounded Cellpose Apply tutorial example."""
import numpy as np


def normalize_field(image, low=2, high=99):
    """An explicit per-image percentile stretch, without spaCR's loader."""
    if image.ndim!=2 or not np.isfinite(image).all() or not 0<=low<high<=100:
        raise ValueError('Expected finite 2-D pixels and ordered percentiles')
    lower,upper=np.percentile(image,[low,high])
    if upper<=lower:raise ValueError('The demonstration needs nonconstant contrast')
    return np.clip((image.astype(np.float64)-lower)/(upper-lower),0,1)


def minimum_area_labels(mask, minimum):
    """Drop small objects and renumber surviving IDs in their original order."""
    if mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer) or mask.min()<0 or minimum<0:
        raise ValueError('Expected nonnegative 2-D labels and a nonnegative cutoff')
    output=np.zeros(mask.shape,dtype=np.uint16)
    next_id=0
    for label in sorted(set(mask.ravel())-{0}):
        pixels=mask==label
        if np.count_nonzero(pixels)>=minimum:
            next_id+=1
            output[pixels]=next_id
    return output


def require_pixels(actual,expected):
    if actual.shape!=expected.shape or not np.array_equal(actual,expected):
        raise ValueError('The actual pixels differ from the independent expectation')
    return int(actual.size)
