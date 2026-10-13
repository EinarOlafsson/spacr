"""Independent group membership and continuous scatter scales for tutorial 63.

Only finite, nondegenerate continuous axes, complete object identities and a
bounded unsampled table are covered. The actual twelve-level facet cap and
unused wrap slots are explicit, rather than hidden by a row-count assertion.
"""
from collections import Counter
from itertools import product
import math
import re

IDENTITY = ('plateID', 'rowID', 'columnID', 'fieldID', 'object_label')


def natural(value):
    return tuple((1, int(v)) if v.isdigit() else (0, v.casefold())
                 for v in re.split(r'(\d+)', str(value)) if v)


def layout(records, facet_row, facet_col, wrap=0):
    if not records or len(records) > 10000 or not 0 <= wrap <= 12:
        raise ValueError('Unsupported bounded scatter population or wrap')
    levels = []
    for key in (facet_row, facet_col):
        if key is not None and any(r.get(key) is None for r in records):
            raise ValueError('Missing facet labels are outside this proof')
        levels.append(sorted({str(r[key]) for r in records}, key=natural)[:12] if key else [None])
    seats = []
    for i, (rlevel, clevel) in enumerate(product(*levels)):
        row, col = divmod(i, len(levels[1]))
        indices = [j for j, r in enumerate(records)
                   if (not facet_row or str(r[facet_row]) == rlevel)
                   and (not facet_col or str(r[facet_col]) == clevel)]
        seats.append(dict(row=row, col=col, row_level=rlevel, col_level=clevel,
                          index=indices, occupied=True))
    shape = (len(levels[0]), len(levels[1]))
    if wrap and not (facet_row and facet_col):
        width = min(wrap, len(seats)); height = math.ceil(len(seats)/width)
        shape = (height, width)
        for i, seat in enumerate(seats): seat['row'], seat['col'] = divmod(i, width)
        while len(seats) < height*width:
            row, col = divmod(len(seats), width)
            seats.append(dict(row=row, col=col, row_level=None, col_level=None,
                              index=[], occupied=False))
    return shape, seats


def limits(records, indices, key):
    values = [float(records[i][key]) for i in indices]
    if not values: return None
    if not all(math.isfinite(v) for v in values) or min(values) == max(values):
        raise ValueError('Only finite nondegenerate continuous axes are checked')
    lo, hi = min(values), max(values)
    pad = .05*(hi-lo)
    return lo-pad, hi+pad


def _same_group(mode, a, b):
    if mode == 'shared': return True
    if mode == 'free': return (a['row'], a['col']) == (b['row'], b['col'])
    if mode == 'row': return a['row'] == b['row']
    if mode == 'col': return a['col'] == b['col']
    raise ValueError('Unknown scale mode')


def verify_trellis(result, records, *, x, y, facet_row=None, facet_col=None,
                   scale_x='shared', scale_y='shared', wrap=0):
    shape, seats = layout(records, facet_row, facet_col, wrap)
    g = result.spec.graph
    actual_ids = [tuple(row) for row in result.frame[list(IDENTITY)].itertuples(index=False, name=None)]
    expected_ids = [tuple(r[k] for k in IDENTITY) for r in records]
    if (result.data.strategy != 'full' or result.data.n_total != len(records)
            or result.data.n_shown != len(records) or actual_ids != expected_ids
            or len(set(expected_ids)) != len(expected_ids)
            or (g.x, g.y, g.facet_row, g.facet_col) != (x, y, facet_row, facet_col)
            or (result.spec.scale_x, result.spec.scale_y, result.spec.wrap) != (scale_x, scale_y, wrap)
            or result.kinds[x] != 'continuous' or result.kinds[y] != 'continuous'):
        raise ValueError('Trellis population, identity, channels or numeric roles differ')
    if result.shape != shape or len(result.panels) != len(seats):
        raise ValueError('Trellis grid shape or wrap padding differs')
    checked = []; errors = []
    for actual, wanted in zip(result.panels, seats):
        for key in ('row', 'col', 'row_level', 'col_level', 'occupied'):
            if getattr(actual, key) != wanted[key]:
                raise ValueError('Trellis group position, label or occupied state differs')
        if list(actual.index) != wanted['index'] or actual.n != len(wanted['index']):
            raise ValueError('Trellis exact group membership differs')
        p = dict(row=actual.row, col=actual.col, row_level=actual.row_level,
                 col_level=actual.col_level, occupied=bool(actual.occupied), n=actual.n)
        for axis, key, mode in [('x', x, scale_x), ('y', y, scale_y)]:
            indices = [i for other in seats if _same_group(mode, wanted, other) for i in other['index']]
            expected = limits(records, indices, key)
            observed = getattr(actual.scales, axis+'_limits')
            categorical = getattr(actual.scales, axis+'_levels')
            if (categorical is not None or ((expected is None) != (observed is None))
                    or (expected is not None and (len(observed) != 2 or any(
                        not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(expected, observed))))):
                raise ValueError('Trellis axis limits differ from independent scale-group extrema')
            if expected is not None: errors.extend(abs(a-b) for a, b in zip(expected, observed))
            p[axis+'_limits'] = list(expected) if expected is not None else None
        checked.append(p)
    shown = sum(p['n'] for p in checked)
    return dict(rows=len(records), shown_points=shown, outside_displayed_levels=len(records)-shown,
                shape=list(shape), real_panels=sum(p['occupied'] for p in checked),
                empty_real_panels=sum(p['occupied'] and p['n']==0 for p in checked),
                unused_wrap_slots=sum(not p['occupied'] for p in checked),
                x=x, y=y, facet_row=facet_row, facet_col=facet_col,
                scale_x=scale_x, scale_y=scale_y, wrap=wrap, panels=checked,
                max_axis_limit_error=max(errors, default=0))


def verify_rendered_axes(canvas, result, records, proof):
    """Compare the actual Matplotlib marks, visible axes and titles to the proof."""
    if set(canvas._axes) != {(p['row'], p['col']) for p in proof['panels']}:
        raise ValueError('Rendered panel set differs from independently checked layout')
    count = 0
    for p, group in zip(proof['panels'], result.panels):
        ax = canvas._axes[p['row'], p['col']]
        if ax.get_visible() != p['occupied']:
            raise ValueError('Rendered visibility confuses empty groups and unused slots')
        if not p['occupied']: continue
        for axis in ('x', 'y'):
            expected = p[axis+'_limits']
            if expected is not None and any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                for a, b in zip(getattr(ax, 'get_'+axis+'lim')(), expected)):
                raise ValueError('Painted axis does not use independently checked limits')
        wanted = Counter((round(float(records[i][proof['x']]), 8), round(float(records[i][proof['y']]), 8))
                         for i in group.index)
        points = ax.collections[0].get_offsets() if ax.collections else []
        actual = Counter((round(float(x), 8), round(float(y), 8)) for x, y in points)
        if actual != wanted:
            raise ValueError('Rendered scatter points differ from independent group measurements')
        title_parts = [v for v in (p['row_level'], p['col_level']) if v is not None]
        title = ' · '.join(title_parts)
        expected_title = (title+'  ·  ' if title else '') + f"n = {p['n']:,}"
        if ax.get_title() != expected_title:
            raise ValueError('Rendered title does not identify the group and exact count')
        count += len(points)
    return dict(rendered_points_independently_checked=count, point_comparison_decimal_places=8,
                actual_axis_limits_and_group_titles_checked=True)
