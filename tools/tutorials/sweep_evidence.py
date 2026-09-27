"""Independent CSV and GUI checks for the two-trial tutorial sweep."""
import math

SUPPORT_COLUMNS = ('n_rows_fitted_grna', 'n_wells_grna',
                   'n_rows_fitted_gene', 'n_wells_gene')


def check_support_geometry(evidence, saved):
    """Require both actual trials' four support cells and headers fully in frame."""
    def rectangle(value):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError('Missing support rectangle')
        x, y, width, height = [_finite_number(v, 'support rectangle') for v in value]
        if width <= 0 or height <= 0:
            raise ValueError('Empty support rectangle')
        return x, y, width, height

    def contained(inner, outer):
        x, y, width, height = rectangle(inner)
        left, top, wide, high = rectangle(outer)
        if x < left or y < top or x + width > left + wide or y + height > top + high:
            raise ValueError('Clipped measured support evidence')

    if evidence.get('frame') != '12_saved_guide_family_restored':
        raise ValueError('Support is not shown in the narrated frame')
    viewport, header_viewport = evidence['viewport'], evidence['header_viewport']
    contained(viewport, evidence['window'])
    contained(header_viewport, evidence['window'])
    headers = evidence['headers']
    if len(headers) != 4 or {h['name'] for h in headers} != set(SUPPORT_COLUMNS):
        raise ValueError('Missing or duplicate measured support headers')
    for header in headers:
        if header['visible'] is not True:
            raise ValueError('Hidden measured support header')
        contained(header['rect'], header_viewport)
    wanted = {str(row['trial_id']): row for row in saved}
    if len(saved) != 2 or len(wanted) != 2:
        raise ValueError('Expected two distinct measured trials')
    expected = {(trial, column) for trial in wanted for column in SUPPORT_COLUMNS}
    seen = set()
    rectangles = []
    for cell in evidence['cells']:
        key = (str(cell['trial_id']), cell['column'])
        if key not in expected or key in seen:
            raise ValueError('Unknown or duplicate measured support cell')
        seen.add(key)
        if cell['visible'] is not True:
            raise ValueError('Hidden measured support cell')
        contained(cell['rect'], viewport)
        matching = next(header for header in headers if header['name'] == key[1])
        x, y, width, height = rectangle(cell['rect'])
        hx, _, hw, _ = rectangle(matching['rect'])
        if x < hx or x + width > hx + hw:
            raise ValueError('Measured support cell is outside its header')
        if _count(cell['value'], key[1]) != _count(wanted[key[0]].get(key[1]), key[1]):
            raise ValueError('Visible measured support differs from saved trial')
        rectangles.append(cell['rect'])
    if seen != expected:
        raise ValueError('Missing measured support cells')
    rectangles += [header['rect'] for header in headers]
    left, top = min(r[0] for r in rectangles), min(r[1] for r in rectangles)
    right = max(r[0] + r[2] for r in rectangles)
    bottom = max(r[1] + r[3] for r in rectangles)
    return dict(passed=True, trials=2, columns=4, cells=8,
                frame=evidence['frame'], bounds=[left, top, right-left, bottom-top])


def _finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Missing or invalid {label}') from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f'Non-finite or invalid {label}')
    return number


def _count(value, label):
    number = _finite_number(value, label)
    if number < 0 or not number.is_integer():
        raise ValueError(f'Invalid count: {label}')
    return int(number)


def check_trial_set(rows):
    """Require this lesson's two successful, separately identified ridge trials."""
    if len(rows) != 2 or any(row.get('status') != 'ok' for row in rows):
        raise ValueError('Expected exactly two successful trials')
    for key in ('trial_id', 'folder'):
        values = [str(row.get(key, '')).strip() for row in rows]
        if any(not value for value in values) or len(set(values)) != 2:
            raise ValueError(f'Expected distinct trial {key} values')
    if sorted(_finite_number(row.get('alpha'), 'ridge penalty') for row in rows) != [.01, .1]:
        raise ValueError('Expected ridge penalties 0.01 and 0.1')
    return dict(passed=True, trials=2, penalties=[.01, .1])


def check_fitted_support(prepared, row, actual, requested, qc=None, fit_designs=None):
    """Check support for this complete, untransformed long ridge example only.

    Complete formula inputs justify comparing prepared rows with fitted rows
    here. This is an execution check, not a claim of valid ridge inference.
    """
    expected_settings = dict(regression_type='ridge', model_data_layout='long',
        level='both', dependent_variable='pred', inference='parametric',
        analysis_mode='regression', analysis_unit='well', agg_type='mean',
        intercept='fitted', model_plate_position=False,
        random_row_column_effects=False, batch_correction='none')
    for key, value in expected_settings.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f'Unsupported actual tutorial setting: {key}')
        if key in requested and requested[key] != value:
            raise ValueError(f'Unsupported requested tutorial setting: {key}')
        if key in row and str(row[key]).lower() != str(value).lower():
            raise ValueError(f'Unsupported saved tutorial setting: {key}')
    for settings in (actual, requested, row):
        if 'transform' not in settings or settings['transform'] not in (None, '', 'None', 'none'):
            raise ValueError('Expected an explicitly untransformed tutorial trial')
        if _finite_number(settings.get('alpha'), 'ridge penalty') != _finite_number(row.get('alpha'), 'ridge penalty'):
            raise ValueError('Requested and actual ridge penalties differ')
    if not prepared:
        raise ValueError('Missing prepared regression rows')
    identifiers = ('prc', 'grna', 'gene', 'plateID', 'rowID', 'columnID')
    if any('screenID' in item for item in prepared):
        identifiers += ('screenID',)
    responses = {}
    for item in prepared:
        for key in identifiers:
            value = item.get(key)
            if value is None or str(value).strip().lower() in ('', 'none', 'nan', 'null', 'inf', '-inf'):
                raise ValueError(f'Missing prepared formula identity: {key}')
        fraction = _finite_number(item.get('fraction'), 'prepared fraction')
        response = _finite_number(item.get('pred'), 'prepared response')
        if not 0 <= fraction <= 1:
            raise ValueError('Invalid prepared fraction')
        if 'cell_count' in item and _finite_number(item['cell_count'], 'prepared cell count') <= 0:
            raise ValueError('Invalid prepared cell count')
        well = item['prc']
        if well in responses and responses[well] != response:
            raise ValueError('The prepared well response is not constant')
        responses[well] = response
    expected = dict(n_rows_fitted=len(prepared),
        n_wells=len({item['prc'] for item in prepared}),
        n_guides=len({item['grna'] for item in prepared}),
        n_genes=len({item['gene'] for item in prepared}))
    for key, value in expected.items():
        prepared_key = 'n_rows_prepared' if key == 'n_rows_fitted' else key + '_prepared'
        for measured_key in (prepared_key, key, key + '_grna', key + '_gene'):
            if _count(row.get(measured_key), measured_key) != value:
                raise ValueError(f'Fitted support differs from prepared inputs: {measured_key}')
    levels = {}
    for level in ('grna', 'gene'):
        columns = _count(row.get('n_design_columns_' + level), 'design columns ' + level)
        if columns == 0:
            raise ValueError('Empty fitted design')
        levels[level] = dict(expected, n_design_columns=columns)
    if fit_designs is not None:
        if not isinstance(fit_designs, dict) or set(fit_designs) != set(levels):
            raise ValueError('Missing or unexpected saved fit levels')
        for level, expected_counts in levels.items():
            recorded = fit_designs[level]
            if not isinstance(recorded, dict) or recorded.get('layout') != 'long':
                raise ValueError('Unexpected saved fitted layout')
            for key, value in expected_counts.items():
                if _count(recorded.get(key), f'saved fit {level}/{key}') != value:
                    raise ValueError(f'Saved fit record disagrees with support: {level}/{key}')
    qc_checked = []
    for level, evidence in (qc or {}).items():
        if level not in levels or evidence.get('regression_type') != 'ridge':
            raise ValueError('Unexpected fitted QC evidence')
        for key, measured_key in (('n_observations', 'n_rows_fitted'),
                                 ('n_unique_wells', 'n_wells'),
                                 ('n_predictors', 'n_design_columns')):
            if _count(evidence.get(key), 'QC ' + key) != levels[level][measured_key]:
                raise ValueError(f'Fitted QC disagrees with saved support: {level}/{key}')
        qc_checked.append(level)
    return dict(passed=True, prepared=expected, fitted=levels,
                qc_levels=qc_checked, saved_fit_records_checked=fit_designs is not None,
                inference_validated=False)


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
