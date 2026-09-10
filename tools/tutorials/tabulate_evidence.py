"""Independent, bounded pivot arithmetic for the real Tabulate tutorial.

No spaCR aggregation, pandas aggregation or exported result is used to compute
the expected cells. This deliberately excludes missing axis labels, infinite
values and capped grids rather than silently claiming those cases were tested.
"""
from collections import defaultdict
import csv
from itertools import product
import math
import re
import statistics


def _natural(value):
    return tuple((1, int(p)) if p.isdigit() else (0, p.casefold())
                 for p in re.split(r'(\d+)', str(value)) if p)


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    if not math.isfinite(value):
        raise ValueError('Infinite values are outside this bounded proof')
    return value


def _stats(values, quantile):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return dict(n=0, **{k: None for k in ('mean', 'median', 'sd', 'sem', 'min', 'max', 'quantile')})
    at = (n - 1) * quantile
    low, high = math.floor(at), math.ceil(at)
    sd = statistics.stdev(ordered) if n > 1 else None
    return dict(n=n, mean=statistics.fmean(ordered), median=statistics.median(ordered),
                sd=sd, sem=sd / math.sqrt(n) if sd is not None else None,
                min=ordered[0], max=ordered[-1],
                quantile=ordered[low] + (ordered[high] - ordered[low]) * (at-low))


def expected_grid(records, *, rows, cols=(), values=(), aggs=('n', 'mean', 'sd'), quantile=.75):
    keys = tuple(rows) + tuple(cols)
    if (not records or len(records) > 10000 or len(keys) != len(set(keys))
            or any(r.get(k) is None for r in records for k in keys)
            or not 0 <= quantile <= 1 or not set(aggs) <= set(_stats([], .5)) or 'n' not in aggs):
        raise ValueError('Unsupported population, axes or statistic in bounded proof')
    levels = [tuple(sorted({str(r[k]) for r in records}, key=_natural)) for k in keys]
    rlevels = tuple(product(*levels[:len(rows)]))
    clevels = tuple(product(*levels[len(rows):]))
    if len(rlevels) > 2000 or len(clevels) > 200 or len(rlevels)*len(clevels) > 100000:
        raise ValueError('Capped grids are outside this bounded proof')
    groups = defaultdict(list)
    for record in records:
        groups[tuple(str(record[k]) for k in keys)].append(record)
    layers = tuple((v, a) for v in values for a in aggs) if values else (('', 'n'),)
    cells = []
    for rk in rlevels:
        for ck in clevels:
            group = groups[rk + ck]
            stats = {}
            for value in values:
                numeric = [_number(r[value]) for r in group]
                stats[value] = _stats([v for v in numeric if v is not None], quantile)
            data = {key: (None if not group else len(group) if not values else stats[key[0]][key[1]])
                    for key in layers}
            cells.append(dict(row=rk, col=ck, count=len(group), present=bool(group), data=data))
    return dict(row_levels=rlevels, col_levels=clevels, layers=layers, cells=cells)


def verify_pivot(result, records, *, rows, cols=(), values=(), aggs=('n', 'mean', 'sd'), quantile=.75):
    wanted = expected_grid(records, rows=rows, cols=cols, values=values, aggs=aggs, quantile=quantile)
    s = result.spec
    if (result.n_source_rows != len(records) or result.hidden_rows != 0
            or tuple(s.rows) != tuple(rows) or tuple(s.cols) != tuple(cols)
            or tuple(s.values) != tuple(values) or tuple(s.aggs) != tuple(aggs)
            or s.quantile != quantile or tuple(result.row_keys) != tuple(rows)
            or tuple(result.col_keys) != tuple(cols)):
        raise ValueError('Pivot population or requested specification differs')
    shape = (len(wanted['row_levels']), len(wanted['col_levels']))
    if (result.row_levels != wanted['row_levels'] or result.col_levels != wanted['col_levels']
            or result.shape != shape or result.sizes.shape != shape or result.present.shape != shape
            or set(result.layers) != set(wanted['layers'])
            or any(a.shape != shape for a in result.layers.values())):
        raise ValueError('Pivot axis levels, ordering or layer structure differs')
    errors = []
    for i, cell in enumerate(wanted['cells']):
        r, c = divmod(i, shape[1])
        if int(result.sizes[r, c]) != cell['count'] or bool(result.present[r, c]) != cell['present']:
            raise ValueError('Pivot group membership or empty-cell meaning differs')
        for key, value in cell['data'].items():
            actual = float(result.layers[key][r, c])
            equal = math.isnan(actual) if value is None else math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-10)
            if not equal:
                raise ValueError('Pivot statistic differs from independent arithmetic')
            if value is not None:
                errors.append(abs(actual-value))
    headers = list(rows)
    for ck in wanted['col_levels']:
        for value, agg in wanted['layers']:
            label = f'{agg}({value})' if value else agg
            headers.append(' · '.join((*ck, label)))
    export_rows = []
    for rk in wanted['row_levels']:
        line = list(rk)
        for cell in wanted['cells']:
            if cell['row'] == rk:
                line.extend(cell['data'][key] for key in wanted['layers'])
        export_rows.append(line)
    return dict(source_rows=len(records), row_keys=list(rows), col_keys=list(cols),
                values=list(values), aggs=list(aggs), quantile=quantile,
                shape=list(shape), present=sum(c['present'] for c in wanted['cells']),
                empty=[dict(row=list(c['row']), col=list(c['col'])) for c in wanted['cells'] if not c['present']],
                max_numeric_error=max(errors, default=0), csv_headers=headers, csv_rows=export_rows)


def verify_csv(path, proof):
    with open(path, newline='') as stream:
        data = list(csv.reader(stream))
    if not data or data[0] != proof['csv_headers'] or len(data)-1 != len(proof['csv_rows']):
        raise ValueError('Export headers or full row count differs from independent pivot')
    for line, expected in zip(data[1:], proof['csv_rows']):
        if len(line) != len(expected):
            raise ValueError('Export column count differs from independent pivot')
        for written, value in zip(line, expected):
            if value is None:
                equal = written == ''
            elif isinstance(value, str):
                equal = written == value
            else:
                try:
                    equal = math.isclose(float(written), value, rel_tol=1e-10, abs_tol=1e-10)
                except ValueError:
                    equal = False
            if not equal:
                raise ValueError('Export value or blank differs from independent pivot')
    return dict(rows=len(data)-1, columns=len(data[0]), independently_compared=True)
