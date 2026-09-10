"""Independent array and table checks for private Curate teaching copies."""
import math
import numpy as np


def painted_disk(before, centre, radius, label):
    """Expected 2-D unit-pixel brush footprint, without the app brush API."""
    if (before.ndim != 2 or not np.issubdtype(before.dtype, np.integer)
            or len(centre) != 2 or not all(math.isfinite(float(v)) for v in (*centre, radius))
            or radius <= 0 or not isinstance(label, int) or label < 0):
        raise ValueError('Expected a finite 2-D integer-mask brush exercise')
    yy, xx = np.indices(before.shape)
    inside = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2
    expected = before.copy()
    expected[inside] = label
    return expected


def check_mask(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError('The actual mask differs from the independent pixel prediction')
    return dict(passed=True, pixels=int(actual.size), labels=np.unique(actual).tolist())


def check_tracks(actual, expected):
    """Compare all keys and values; only CSV float-roundtrip error is allowed."""
    columns = {'frame', 'original_label', 'track_id', 'x', 'y'}
    def indexed(rows):
        result = {}
        for row in rows:
            if set(row) != columns:
                raise ValueError('Track columns differ from the exact recorded schema')
            key = (int(row['track_id']), int(row['frame']))
            if key in result:
                raise ValueError('Duplicate track/frame key')
            result[key] = row
        return result
    found, wanted = indexed(actual), indexed(expected)
    if set(found) != set(wanted):
        raise ValueError('Track identities or frame population differ')
    for key in wanted:
        for column in columns:
            a, b = float(found[key][column]), float(wanted[key][column])
            if (not math.isfinite(a) or not math.isfinite(b)
                    or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('A recorded track value differs')
    return dict(passed=True, rows=len(found), tracks=len({k[0] for k in found}), cells=5*len(found))
