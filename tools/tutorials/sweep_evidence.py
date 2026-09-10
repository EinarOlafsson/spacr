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


def check_result_family(snapshot,saved,level):
    """Check every displayed cell and plotted identity/coordinate, not inference validity."""
    if level not in ('grna','gene') or snapshot['level']!=level or snapshot['p_axis']!='raw':
        raise ValueError('Unexpected selected family or plotted P axis')
    selected=[r for r in saved if r['level']==level]
    headers=snapshot['headers']
    expected_headers=[c for c in saved[0] if any(r[c] not in ('',None) for r in selected)]
    if headers!=expected_headers:raise ValueError('Displayed coefficient columns differ')
    # Reuse the independently tested order-insensitive full-cell matcher.
    columns=['trial_id' if c=='feature' else c for c in headers]
    # The saved log-P column can contain +inf when the stored P is exactly
    # zero. Allow that exact literal only in that column, not arbitrary
    # non-finite coefficients or plotting coordinates.
    def log_literal(column,value):
        return 'saved positive infinity' if column=='-log10(p_value)' and value=='inf' else value
    expected=[{('trial_id' if c=='feature' else c):log_literal(c,v) for c,v in r.items()} for r in selected]
    displayed=[[log_literal(headers[i] if i<len(headers) else '',v) for i,v in enumerate(row)]
               for row in snapshot['rows']]
    table=check_display(columns,displayed,expected)
    plotted=[r for r in selected if r['feature']!='Intercept']
    keys=[r['feature'] for r in plotted]
    if snapshot['keys']!=keys or len(snapshot['points'])!=len(keys):
        raise ValueError('Plotted identities or point count differ')
    positive=[float(r['p_value']) for r in plotted if float(r['p_value'])>0]
    floor=min(positive)*1e-3 if positive else 1e-303
    seen=set();error=0.
    for x,y,index in snapshot['points']:
        if not isinstance(index,int) or index in seen or not 0<=index<len(keys):
            raise ValueError('Plotted point attribution differs')
        seen.add(index);row=plotted[index]
        wanted_x=float(row['coefficient']);wanted_y=-math.log10(max(floor,min(1.,float(row['p_value']))))
        if not same_value(x,wanted_x) or not same_value(y,wanted_y):
            raise ValueError('Plotted coefficient or log-P coordinate differs')
        error=max(error,abs(x-wanted_x),abs(y-wanted_y))
    return dict(passed=True,level=level,table=table,plotted_points=len(plotted),
                maximum_coordinate_error=error,inference_validated=False)
