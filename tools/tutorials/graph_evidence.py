"""Independent checks of tutorial chart artists, not the application's renderer."""
from collections import Counter
import math


def check_points(expected, displayed):
    """Check every finite x/y pair, retaining duplicate observations."""
    def pairs(rows):
        result = []
        for row in rows:
            x, y = map(float, row)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('Nonfinite point requires a separate missing-data audit')
            result.append((x, y))
        return Counter(result)
    wanted, actual = pairs(expected), pairs(displayed)
    if wanted != actual:
        raise ValueError('Displayed point coordinates or multiplicities differ')
    return sum(wanted.values())


def check_histogram(values, edges, heights):
    """Count directly into actual bin edges; do not call the renderer or np.histogram."""
    if len(edges) != len(heights) + 1 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError('Invalid histogram edges')
    counts = [0] * len(heights)
    for value in values:
        if not math.isfinite(value):
            raise ValueError('Nonfinite histogram value requires a separate audit')
        for i, (low, high) in enumerate(zip(edges, edges[1:])):
            if low <= value < high or (i == len(counts)-1 and value == high):
                counts[i] += 1
                break
        else:
            raise ValueError('Histogram edges omit an observation')
    if counts != list(heights):
        raise ValueError('Histogram bar counts differ')
    return counts


def check_brush(expected_keys, published_keys, visible_count, handoff_enabled):
    """Verify publication independently; never conflate it with visible selection."""
    if not expected_keys or set(expected_keys) != set(published_keys):
        raise ValueError('Published brush identities differ from the rectangle')
    if visible_count != 0 or handoff_enabled:
        raise ValueError('The recorded broken handoff changed; review the lesson')
    return {'published_object_keys': len(set(published_keys)),
            'visible_selected_rows': visible_count, 'handoff_enabled': False,
            'annotation_handoff_works': False}
