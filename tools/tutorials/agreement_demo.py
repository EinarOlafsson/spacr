"""Sixteen real downloaded crops with explicitly artificial teaching labels.

The source database and images are read only. No annotation is presented as
biological truth or as a human reviewer's work. The independent calculations
use Python counters and fractions, never spaCR's agreement implementation.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from stage_lesson import read, write

COLUMNS = ('DEMO_A', 'DEMO_B', 'DEMO_C')
LABELS = tuple(zip(
    [1] * 7 + [2] * 7 + [None, None],
    [1, 1, 1, 1, 1, 2, None, 1, 1, 2, 2, 2, 2, None, 1, None],
    [1] * 8 + [2] * 7 + [None],
))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(stage):
    stage = Path(stage).resolve()
    source = stage / 'annotate_fresh/example_data/plate1'
    database = source / 'measurements/measurements.db'
    before = digest(database)
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        schema = db.execute("SELECT sql FROM sqlite_master WHERE name='png_list'").fetchone()[0]
        columns = [row[1] for row in db.execute('PRAGMA table_info(png_list)')]
        original = db.execute('SELECT * FROM png_list ORDER BY png_path LIMIT 16').fetchall()
    if len(original) != 16 or any(name in columns for name in COLUMNS):
        raise ValueError('Expected sixteen original crops without these demonstration labels')
    runs = stage / 'agreement_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='DEMO-labels-real-crops-', dir=runs))
    (root / 'measurements').mkdir()
    (root / 'data/crops').mkdir(parents=True)
    target = root / 'measurements/measurements.db'
    path_index = columns.index('png_path')
    entries, copied = [], []
    source_prefix = Path('/home/olafsson/.cache/spacr/example_data/plate1')
    with sqlite3.connect(target) as db:
        db.execute(schema)
        for name in COLUMNS:
            db.execute(f'ALTER TABLE png_list ADD COLUMN "{name}" INTEGER')
        for values, labels in zip(original, LABELS):
            original_path = Path(values[path_index])
            incoming = (source / original_path.relative_to(source_prefix)).resolve()
            if not incoming.is_relative_to(source) or not incoming.is_file():
                raise ValueError('The original downloadable crop is missing or leaves the example')
            relative = 'data/crops/' + incoming.name
            outgoing = root / relative
            if outgoing.exists():
                raise ValueError('Distinct objects would overwrite the same crop filename')
            shutil.copy2(incoming, outgoing)
            if digest(incoming) != digest(outgoing):
                raise ValueError('Copied crop bytes changed')
            row = list(values)
            row[path_index] = relative
            row.extend(labels)
            db.execute('INSERT INTO png_list VALUES (' + ','.join('?' for _ in row) + ')', row)
            entries.append(dict(png_path=relative, **dict(zip(COLUMNS, labels))))
            copied.append(dict(original=str(incoming), copied=str(outgoing), sha256=digest(incoming)))
    if digest(database) != before:
        raise ValueError('Original downloaded annotation database changed')
    manifest = dict(source_database=str(database), source_database_sha256=before,
        source_rows_sha256=hashlib.sha256(json.dumps(original).encode()).hexdigest(),
        database=str(target), database_sha256=digest(target), rows=entries, crops=copied,
        labels_are_artificial=True, biological_or_human_review_claim=False,
        scope='Sixteen unchanged downloaded crop images; all DEMO labels deliberately assigned for teaching')
    write(root / 'manifest.json', manifest)
    return manifest


def expected(rows, columns):
    values = [[row[column] for column in columns] for row in rows]
    complete = [row for row in values if all(value is not None for value in row)]
    if not complete:
        raise ValueError('Demonstration requires overlapping labels')
    result = dict(n_rows=len(values), n_complete=len(complete),
        n_partial=sum(0 < sum(v is not None for v in row) < len(columns) for row in values),
        n_unlabelled=sum(all(v is None for v in row) for row in values),
        n_disagreements=sum(len({v for v in row if v is not None}) > 1 for row in values),
        percent_agreement=float(Fraction(sum(len(set(row)) == 1 for row in complete), len(complete))))
    pairs = []
    for left, right in itertools.combinations(columns, 2):
        paired = [(row[left], row[right]) for row in rows if row[left] is not None and row[right] is not None]
        counts = Counter(paired)
        n = len(paired)
        agree = sum(a == b for a, b in paired)
        a, b = Counter(x for x, _ in paired), Counter(y for _, y in paired)
        chance = sum(Fraction(a[label] * b[label], n * n) for label in set(a) | set(b))
        observed = Fraction(agree, n)
        pairs.append(dict(column_a=left, column_b=right, n_compared=n, n_agree=agree,
            n_disagree=n-agree, n_abstained=sum((row[left] is None) != (row[right] is None) for row in rows),
            n_neither=sum(row[left] is None and row[right] is None for row in rows),
            percent_agreement=float(observed), expected_agreement=float(chance),
            kappa=float((observed-chance)/(1-chance)),
            confusion=[[counts[(a,b)] for b in (1,2)] for a in (1,2)]))
    if len(columns) == 2:
        result['overall_kappa'] = pairs[0]['kappa']
    else:
        ratings = len(columns)
        observed = sum(Fraction(sum(n*(n-1) for n in Counter(row).values()), ratings*(ratings-1))
                       for row in complete) / len(complete)
        total = Counter(v for row in complete for v in row)
        chance = sum(Fraction(n, ratings*len(complete))**2 for n in total.values())
        result['overall_kappa'] = float((observed-chance)/(1-chance))
    result['pairs'] = pairs
    result['disagreement_paths'] = [row['png_path'] for row in rows
        if len({row[column] for column in columns if row[column] is not None}) > 1]
    return result


def verify(actual, independent):
    import math
    for key, wanted in independent.items():
        if key == 'pairs':
            if len(actual[key]) != len(wanted):
                raise ValueError('Wrong pair count')
            for got_pair, wanted_pair in zip(actual[key], wanted):
                verify(got_pair, wanted_pair)
        elif isinstance(wanted, float):
            if not math.isclose(actual[key], wanted, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{key} differs from the independent label calculation')
        elif actual[key] != wanted:
            raise ValueError(f'{key} differs from the independent label calculation')
    return True
