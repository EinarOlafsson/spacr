"""Independent, bounded SQLite identity/containment oracle for tutorial 56.

No spaCR lineage, schema, selection, Qt or pandas implementation is imported.
Only finite positive integer labels, plain underscore-free field tokens and
the genuine cell/nucleus/pathogen tables are accepted by this example oracle.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import sqlite3

FIELDS = ('plateID', 'rowID', 'columnID', 'fieldID')
TABLES = ('cell', 'nucleus', 'pathogen')


def label(value, *, nullable=False):
    if nullable and value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or int(value) != value:
        raise ValueError('This oracle requires finite positive integer labels')
    return int(value)


def expected_from_rows(tables):
    if set(tables) != set(TABLES):
        raise ValueError('This example requires exactly cell/nucleus/pathogen')
    by_table = {}
    for table in TABLES:
        rows = []
        seen = set()
        for record in tables[table]:
            tokens = tuple(record[k] for k in FIELDS)
            if any(not isinstance(t, str) or not t or '_' in t or '%' in t for t in tokens):
                raise ValueError('This identity proof requires plain field tokens')
            number = label(record['object_label'])
            identity = tokens + (number,)
            if identity in seen:
                raise ValueError('Duplicate source object within table and field')
            seen.add(identity)
            field = '_'.join(tokens)
            rows.append(dict(key=f'{field}_{table}{number}', table=table, label=number,
                             field=field, identity=identity,
                             parent_label=label(record['cell_id'], nullable=True) if table != 'cell' else None))
        by_table[table] = rows
    roots = sorted(by_table['cell'], key=lambda r: (r['field'], r['label']))
    parents = {r['identity']: r['key'] for r in roots}
    children = defaultdict(list); orphans = []
    for table in TABLES[1:]:
        for row in sorted(by_table[table], key=lambda r: r['label']):
            parent = parents.get(row['identity'][:4] + (row['parent_label'],))
            if parent is None:
                orphans.append(dict(key=row['key'], table=table, label=row['label'],
                                    parent_id='' if row['parent_label'] is None else str(row['parent_label'])))
            else:
                children[parent].append(row)
    flat = []; families = {}
    for root in roots:
        family = children[root['key']]
        families[root['key']] = [root['key']] + [r['key'] for r in family]
        for depth, row in [(0, root)] + [(1, r) for r in family]:
            flat.append(dict(key=row['key'], table=row['table'], label=row['label'], field=row['field'],
                             parent_key=root['key'] if depth else '', depth=depth,
                             n_children=0 if depth else len(family)))
    return dict(rows=flat, roots=[r['key'] for r in roots], families=families,
                orphans=sorted(orphans, key=lambda r:r['key']),
                source_counts={t:len(by_table[t]) for t in TABLES},
                attached_counts=dict(Counter(r['table'] for r in flat if r['depth'])),
                childless=sum(len(families[r['key']]) == 1 for r in roots),
                no_pathogens=sum(not any(c['table']=='pathogen' for c in children[r['key']]) for r in roots))


def read_expected(path):
    con = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    con.row_factory = sqlite3.Row
    try:
        present = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not set(TABLES) <= present or any(t.startswith('organelle') for t in present):
            raise ValueError('Unexpected source object tables for bounded oracle')
        tables = {}
        for table in TABLES:
            columns = [*FIELDS, 'object_label'] + (['cell_id'] if table != 'cell' else [])
            rows = list(con.execute('SELECT '+','.join(columns)+' FROM '+table+' LIMIT 200001'))
            if len(rows)>200000:
                raise ValueError('Source exceeds independent per-table bound')
            tables[table] = [dict(r) for r in rows]
        return expected_from_rows(tables)
    finally:
        con.close()


def verify_forest(observed, orphan_rows, expected):
    if observed != expected['rows']:
        raise ValueError('Actual full forest differs in identity, parent, depth, order or child count')
    if sorted(orphan_rows, key=lambda r:r['key']) != expected['orphans']:
        raise ValueError('Actual orphan identities or claimed parents differ')
    return dict(source_counts=expected['source_counts'], full_nodes=len(observed),
                attached_counts=expected['attached_counts'], roots=len(expected['roots']),
                orphans=len(orphan_rows), childless=expected['childless'],
                no_pathogens=expected['no_pathogens'])


def verify_tree(actual, expected, *, limit=2000):
    # Header sorting can reorder whole subtrees; it must not change membership
    # or attach a child to a different field/table. Compare every GUI item.
    roots = set(expected['roots'][:limit])
    wanted = {r['key']:r for r in expected['rows'] if r['key'] in roots or r['parent_key'] in roots}
    if len(actual)!=len(wanted) or len({r['key'] for r in actual})!=len(actual):
        raise ValueError('Actual tree has missing, repeated or excess visible-domain objects')
    for row in actual:
        if row['key'] not in wanted or row != wanted[row['key']]:
            raise ValueError('Actual tree row identity or containment differs')
    return dict(displayed_roots=len(roots), displayed_nodes=len(actual),
                omitted_roots=max(0,len(expected['roots'])-limit),
                all_displayed_item_identities_and_parent_edges_checked=True)


def verify_selected(actual, expected):
    if list(actual)!=list(expected):
        raise ValueError('Published selected/family keys differ in identity, multiplicity or order')
    return list(actual)


def verify_unavailable_crop(status, opener_present):
    if opener_present or status!='Open the Annotate screen first — it is what shows crops.':
        raise ValueError('Actual unavailable-crop precondition or message differs')
    return dict(opener_present=False,status=status,crop_opened=False)
