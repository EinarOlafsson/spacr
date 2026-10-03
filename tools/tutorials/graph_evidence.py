"""Independent checks of tutorial chart artists, not the application's renderer."""
import math
from collections import Counter


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


def check_annotation_recipe(base, annotated, definition, *, extract=False):
    """Check all lesson outputs independently of the application's evaluator."""
    import pandas as pd

    outputs = ['genotype', 'replicate', 'condition']
    extracted = None
    if extract:
        import re
        entry = definition.get('columns', [])[3:]
        if (len(entry) != 1 or entry[0].get('kind') != 'extract'
                or entry[0].get('column') != 'row_number' or entry[0].get('metadata_column') != 'rowID'):
            raise ValueError('Expected one Extract text output, row_number from rowID')
        expected = [(m.group(1) if (m := re.fullmatch(r'r(\d+)', str(v))) else None) for v in base.rowID]
        actual = [None if pd.isna(v) else str(v) for v in annotated['row_number']]
        if expected != actual:
            raise ValueError('Extract text differs from an independent regex over rowID')
        extracted = dict(Counter(v for v in expected if v is not None))
        annotated = annotated.drop(columns=['row_number'])
        definition = dict(definition, columns=definition['columns'][:3])
    if (not base.index.equals(annotated.index)
            or list(annotated.columns) != list(base.columns) + outputs
            or not base.equals(annotated.loc[:, base.columns])):
        raise ValueError('Annotation changed source columns, row order or index')
    columns = definition.get('columns', [])
    if definition.get('version') != 3 or [c['column'] for c in columns] != outputs:
        raise ValueError('Expected the ordered three-column recipe')
    rules = [
        [('WildType', 'columnID', {'c1', 'c2', 'c3'}),
         ('mutant', 'columnID', {'c7', 'c4', 'c5', 'c6'})],
        [('replicate 1', 'rowID', {'r1', 'r4', 'r5', 'r6'}),
         ('replicate 1', 'rowID', {'r7', 'r9', 'r10'})],
    ]
    for column, expected in zip(columns[:2], rules):
        actual = column.get('conditions', [])
        if column.get('kind') != 'rules' or len(actual) != len(expected):
            raise ValueError('Expected separate rule boxes')
        for rule, (label, metadata, values) in zip(actual, expected):
            if (rule.get('name') != label or rule.get('metadata_column') != metadata
                    or rule.get('match_mode') != 'values'
                    or set(rule.get('include_values', [])) != values
                    or any(rule.get(key) for key in ('exclude_values', 'manual_rows', 'include', 'exclude'))):
                raise ValueError('Exact-value rule differs from the lesson')
    if columns[2] != dict(column='condition', kind='template', parts=[
            dict(kind='column', column='genotype'), dict(kind='text', text='_'),
            dict(kind='column', column='replicate')]):
        raise ValueError('Combination order or separator differs')
    expected_rows = []
    for column, row in zip(base.columnID, base.rowID):
        genotype = ('WildType' if column in {'c1', 'c2', 'c3'} else
                    'mutant' if column in {'c4', 'c5', 'c6', 'c7'} else None)
        replicate = 'replicate 1' if row in {'r1', 'r4', 'r5', 'r6', 'r7', 'r9', 'r10'} else None
        expected_rows.append((genotype, replicate,
                              f'{genotype}_{replicate}' if genotype and replicate else None))
    actual_rows = [tuple(None if pd.isna(value) else value for value in row)
                   for row in annotated[outputs].itertuples(index=False, name=None)]
    if expected_rows != actual_rows:
        raise ValueError('Annotation output differs from independent full-row labels')
    return {'rows_checked': len(base), 'outputs': outputs,
            'counts': {name: dict(Counter(row[i] for row in expected_rows if row[i] is not None))
                       for i, name in enumerate(outputs)},
            'combined_missing_rows': sum(row[2] is None for row in expected_rows),
            'illustrative_labels_only': True, 'source_columns_unchanged': True,
            'extract_text_row_number': extracted, 'published': False}
