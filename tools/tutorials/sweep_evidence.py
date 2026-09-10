"""Independent CSV and GUI checks for the two-trial tutorial sweep."""
import math


def same_value(actual,expected):
    missing={'','None','nan'}
    if str(actual) in missing and str(expected) in missing:return True
    try:
        a,b=float(actual),float(expected)
    except (TypeError,ValueError):return str(actual)==str(expected)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def check_display(headers,displayed,saved):
    if not headers or len(headers)!=len(set(headers)) or 'trial_id' not in headers:
        raise ValueError('Missing or repeated displayed headers')
    wanted={str(row['trial_id']):row for row in saved}
    if len(wanted)!=len(saved):raise ValueError('Repeated saved trial identity')
    seen=set()
    for cells in displayed:
        if len(cells)!=len(headers):raise ValueError('Incomplete displayed row')
        row=dict(zip(headers,cells));key=str(row['trial_id'])
        if key in seen or key not in wanted:raise ValueError('Repeated or unknown displayed trial')
        seen.add(key)
        for column,value in row.items():
            if column not in wanted[key] or not same_value(value,wanted[key][column]):
                raise ValueError('Displayed trial cell differs from its saved CSV')
    if seen!=set(wanted):raise ValueError('A saved trial is missing from the display')
    return dict(rows=len(seen),columns=len(headers),cells=len(seen)*len(headers),passed=True)


def count_result_rows(rows,threshold=.05):
    if not 0<threshold<1:raise ValueError('Expected a fractional significance threshold')
    count=0
    for row in rows:
        value=row['q_value']
        if value is None or str(value) in ('','nan'):continue
        value=float(value)
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid stored adjusted P value')
        count+=value<threshold
    return dict(n_results=len(rows),n_below_alpha=count)
