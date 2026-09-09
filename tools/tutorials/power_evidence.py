"""Independently check captured power curves against every real fit record."""
from __future__ import annotations

import math


def check_recommendation(sentence, current_wells):
    """An unchanged well count must not be narrated as an improvement."""
    if f'Going to {current_wells:g} wells would take that to' in sentence:
        raise ValueError('The headline recommends the unchanged well count as an improvement')


def check_curves(records, threshold, replicates):
    """Reject omitted fits, invented detections and incomplete curve axes.

    These checks consume recorded output only. They never calculate or inject
    a GUI result, and passing them is not evidence of biological calibration.
    """
    total = 0
    for prefix, column in [('cells', 'imaging_n_cells_per_well_mu'),
                           ('wells', 'n_wells_per_screen')]:
        scans, curves = records[prefix + '_scan'], records[prefix + '_curve']
        values = {row[column] for row in scans}
        if not values or len(curves) != len(values) or {r['value'] for r in curves} != values:
            raise ValueError('Curve axes must include every simulated point exactly once')
        for curve in curves:
            members = [row for row in scans if row[column] == curve['value']]
            if len(members) != replicates or curve['n_replicates'] != replicates:
                raise ValueError('Every requested replicate belongs in the denominator')
            if any(row['status'] not in {'ok', 'not_converged', 'failed'} for row in members):
                raise ValueError('Unknown fit status cannot be interpreted as a detection')
            detected = sum(row['status'] == 'ok' and row['model_auroc'] is not None
                           and math.isfinite(row['model_auroc'])
                           and row['model_auroc'] >= threshold for row in members)
            if curve['n_detected'] != detected or curve['power'] != detected / replicates:
                raise ValueError('Detection probability must count only usable above-threshold fits')
            for status, count in [('ok', 'n_ok'), ('not_converged', 'n_not_converged'),
                                  ('failed', 'n_failed')]:
                if curve[count] != sum(row['status'] == status for row in members):
                    raise ValueError('Fit-status counts do not match the actual records')
        total += len(scans)
    return total
