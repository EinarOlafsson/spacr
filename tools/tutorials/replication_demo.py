"""Check the preserved synthetic replication teaching data independently.

These are invented parasite rows, not images, treatments or biological results.
Never overwrite the original demonstration: each actual GUI run gets a new copy.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile

from stage_lesson import write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(counts):
    """Count known integer observations, keeping off-ladder values explicit."""
    counts = list(counts)
    buckets = Counter(str(n) if n in (1, 2, 4, 8, 16) else
                      'gt16' if n > 16 and n & (n - 1) == 0 else
                      'non_power_of_two' for n in counts)
    ladder = [n for n in counts if n > 0 and n & (n - 1) == 0]
    result = dict(n_vacuoles=len(counts), n_parasites=sum(counts))
    for name in ('1', '2', '4', '8', '16', 'gt16', 'non_power_of_two'):
        result['n_' + name] = buckets[name]
        result['frac_' + name] = buckets[name] / len(counts)
    result.update(non_power_of_two_fraction=buckets['non_power_of_two'] / len(counts),
        qc_flag_non_power_of_two=buckets['non_power_of_two'] / len(counts) > .2,
        median_parasites_per_vacuole=statistics.median(counts),
        n_power_of_two=len(ladder),
        median_doublings=statistics.median(math.log2(n) for n in ladder) if ladder else 0,
        mean_parasites_per_vacuole=statistics.mean(ladder) if ladder else 0,
        mean_fraction_of_vacuoles=len(ladder) / len(counts))
    return result


def expected(rows):
    groups = defaultdict(list)
    for row in rows:
        if not 200 <= row['pathogen_area'] <= 800 or row['cell_id'] <= 0:
            raise ValueError('The preserved teaching rows no longer all pass the demonstrated filters')
        groups[(row['prcf'], row['vacuole_id'])].append(row)
    vacuoles, wells, conditions = {}, defaultdict(list), defaultdict(list)
    for (field, identity), observations in groups.items():
        first = observations[0]
        condition = {'c1': 'HeLa_vehicle', 'c2': 'HeLa_inhibitor'}[first['columnID']]
        well = '_'.join(first[k] for k in ('plateID', 'rowID', 'columnID'))
        n = len(observations)
        record = {k: first[k] for k in ('plateID', 'rowID', 'columnID', 'fieldID', 'prcf', 'cell_id')}
        bucket = str(n) if n in (1, 2, 4, 8, 16) else 'non_power_of_two'
        record.update(prc=well, condition=condition, n_parasites=n,
            total_parasite_area=sum(r['pathogen_area'] for r in observations),
            replication_bucket=bucket, is_power_of_two=bucket != 'non_power_of_two')
        if bucket != 'non_power_of_two':
            record['doublings'] = math.log2(n)
        vacuoles[field + '_v' + identity] = record
        wells[well].append(n)
        conditions[condition].append(n)
    well_records = {}
    for well, counts in wells.items():
        plate, row, column = well.split('_')
        well_records[well] = dict(summarize(counts), plateID=plate, rowID=row, columnID=column,
            condition={'c1': 'HeLa_vehicle', 'c2': 'HeLa_inhibitor'}[column])
    return {'vacuole_counts.csv': ('vacuole_id', vacuoles),
            'well_distribution.csv': ('prc', well_records),
            'condition_summary.csv': ('condition', {
                name: dict(summarize(counts), n_wells=sum(r['condition'] == name for r in well_records.values()))
                for name, counts in conditions.items()})}


def prepare(stage):
    stage = Path(stage)
    source = stage.parent / 'synthetic/replication/measurements/measurements.db'
    original_hash = digest(source)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM pathogen ORDER BY object_label')]
    wanted = expected(rows)
    if (len(rows) != 1992 or len(wanted['vacuole_counts.csv'][1]) != 480 or
            len(wanted['well_distribution.csv'][1]) != 12):
        raise ValueError('The known synthetic example has changed')
    runs = stage / 'replication_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-explicit-vacuoles-', dir=runs))
    destination = root / 'measurements/measurements.db'
    destination.parent.mkdir()
    shutil.copy2(source, destination)
    if digest(destination) != original_hash or digest(source) != original_hash:
        raise ValueError('The private copy or preserved database changed')
    manifest = dict(source=str(source), source_sha256=original_hash,
        database=str(destination), database_sha256=original_hash, expected=wanted,
        synthetic=True, images_available=False, biological_effect_claim=False,
        rows=len(rows), vacuoles=480, wells=12,
        host_groups=len({(r['prcf'], r['cell_id']) for r in rows}))
    write(root / 'manifest.json', manifest)
    return manifest


def verify_records(actual, key, expected_records):
    identities = [row[key] for row in actual]
    if len(identities) != len(set(identities)) or set(identities) != set(expected_records):
        raise ValueError('Output identities are missing, duplicated or unexpected')
    checked = 0
    for row in actual:
        for name, wanted in expected_records[row[key]].items():
            got = row[name]
            if isinstance(wanted, bool):
                matches = str(got) == str(wanted)
            elif isinstance(wanted, (int, float)):
                matches = math.isclose(float(got), wanted, rel_tol=1e-12, abs_tol=1e-12)
            else:
                matches = str(got) == str(wanted)
            if not matches:
                raise ValueError(f'{row[key]} / {name} differs from independent input counts')
            checked += 1
    return checked


def verify_files(directory, expectations):
    checks = {}
    for filename, (key, reference) in expectations.items():
        with (Path(directory) / filename).open(newline='') as stream:
            records = list(csv.DictReader(stream))
        checks[filename] = dict(rows=len(records), fields_checked=verify_records(records, key, reference))
    return checks
