"""Disclosed synthetic campaign and independent, bounded CSV checks.

The generator adapts tests/qt/test_control_chart_screen.py:campaign (seed 11).
Rows are simulated wells, NOT acquired measurements, cells or biological data.
Only the software workflow is validated. Nothing imports the chart engine.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import statistics


FIXTURE = Path(__file__).parent / 'fixtures/control_chart_campaign.csv'
COLUMNS = ('plate', 'order', 'value', 'n', 'subgroup_sd', 'centre', 'lower',
           'upper', 'sigma', 'z', 'in_baseline', 'flagged', 'rules')


def generate(path):
    """Create one new, explicitly synthetic tutorial fixture; never overwrite."""
    import numpy as np
    rng = np.random.default_rng(11)
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('plateID', 'run_date', 'well_type', 'signal'))
        for index in range(30):
            drift = 0.0 if index < 20 else 2.4 * (index - 19)
            for _ in range(3):
                for name, mean, scale in (('neg', 100 + drift, 2),
                                           ('pos', 20, 2), ('sample', 60, 9)):
                    writer.writerow((f'P{index+1:02}', f'2026-01-{index+1:02}',
                                     name, mean + rng.normal(0.0, scale)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_points(path, level='neg', baseline=20):
    """Independent stdlib arithmetic for this equal-size X-bar/S fixture."""
    with Path(path).open(newline='') as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 270 or baseline not in (12, 20):
        raise ValueError('Unexpected synthetic campaign shape or baseline')
    groups = {}
    for row in source:
        if row['well_type'] == level:
            groups.setdefault((row['plateID'], row['run_date']), []).append(float(row['signal']))
    if len(groups) != 30 or any(len(values) != 3 for values in groups.values()):
        raise ValueError('The selected control does not have three wells per plate')
    points = [{'plate': p, 'order': d, 'value': statistics.mean(v), 'n': 3,
               'subgroup_sd': statistics.stdev(v)} for (p, d), v in sorted(groups.items())]
    centre = statistics.mean(r['value'] for r in points[:baseline])
    # c4(3) = Gamma(3/2) / Gamma(1) = sqrt(pi)/2.
    sigma_within = statistics.mean(r['subgroup_sd'] for r in points[:baseline]) / (math.sqrt(math.pi)/2)
    sigma_mean = sigma_within / math.sqrt(3)
    for index, row in enumerate(points):
        z = (row['value'] - centre) / sigma_mean
        row.update(centre=centre, lower=centre-3*sigma_mean, upper=centre+3*sigma_mean,
                   sigma=sigma_mean, z=z, in_baseline=index < baseline,
                   flagged=abs(z) > 3, rules='1' if abs(z) > 3 else '')
    return points


def verify_points(actual, expected, *, limits_only=True):
    """Check every ordered identity and numeric value, not just the length."""
    if len(actual) != len(expected) or not actual:
        raise ValueError('Wrong chart point count')
    checked = 0
    for got, wanted in zip(actual, expected):
        for key in COLUMNS:
            if key not in got:
                raise ValueError(f'Missing chart column: {key}')
            if not limits_only and key in ('flagged', 'rules'):
                continue
            value = wanted[key]
            if key in ('plate', 'order', 'rules'):
                same = str(got[key]) == str(value)
            elif key in ('in_baseline', 'flagged'):
                same = str(got[key]) == str(value)
            else:
                same = math.isclose(float(got[key]), float(value), rel_tol=1e-10, abs_tol=1e-11)
            if not same:
                raise ValueError(f'Chart disagrees with independent CSV calculation: {wanted["plate"]}/{key}')
            checked += 1
    return {'rows': len(actual), 'checked_fields': checked,
            'ordered_identities_and_numbers_match': True,
            'rule_one_independently_checked': limits_only}


def verify_export(path, expected):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        records = list(reader)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError('Wrong exported chart schema')
    return {**verify_points(records, expected), 'sha256': digest(path), 'path': str(path)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', type=Path, required=True)
    args = parser.parse_args()
    generate(args.generate)
