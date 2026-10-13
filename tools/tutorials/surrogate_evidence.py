"""Independent arithmetic and display checks for the surrogate tutorial.

No spaCR fitting or scoring implementation is imported. A faithful surrogate
is still not a biological validation of the classifier it approximates.
"""
from collections import Counter
import math


def binary_counts(rows):
    if not rows:
        raise ValueError('No held-out predictions')
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        actual, predicted = row['cv_prediction'], row['surrogate_prediction']
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError('Expected finite binary class calls')
        matrix[int(actual)][int(predicted)] += 1
    n = len(rows)
    metrics = []
    for label in (0, 1):
        tp = matrix[label][label]
        support = sum(matrix[label])
        called = sum(row[label] for row in matrix)
        precision = tp / called if called else 0.
        recall = tp / support if support else 0.
        f1 = 2 * tp / (support + called) if support + called else 0.
        metrics.append(dict(class_id=label, support=support, precision=precision, recall=recall, f1=f1))
    present = [row['recall'] for row in metrics if row['support']]
    present_f1 = [metrics[i]['f1'] for i in (0,1)
                  if sum(matrix[i]) + sum(row[i] for row in matrix)]
    return dict(n=n, confusion=matrix, class_metrics=metrics,
        fidelity=(matrix[0][0] + matrix[1][1]) / n,
        balanced_accuracy=sum(present)/len(present),
        f1_macro=sum(present_f1)/len(present_f1))


def check_metric_values(actual, expected):
    for key, wanted in expected.items():
        value = float(actual[key])
        if not math.isfinite(value) or not math.isclose(value, float(wanted), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Saved surrogate metric differs: ' + key)
    return dict(passed=True, values=len(expected))


def check_formatted_table(actual, columns, rows):
    """Compare every five-significant-digit display cell, including repeats.

    The caller supplies the independently read source rows in original order;
    the GUI can sort them, but may neither lose nor duplicate any of them.
    The known 1,000-row GUI cap is explicit and never called a complete export.
    """
    if actual['columns'] != columns or len(columns) != len(set(columns)):
        raise ValueError('Displayed surrogate columns differ')
    def text(value):
        if value is None or isinstance(value, float) and math.isnan(value):
            return ''
        if isinstance(value, float):
            if not math.isfinite(value): raise ValueError('Nonfinite surrogate display value')
            return f'{value:.5g}'
        return str(value)
    wanted = [tuple(text(v) for v in row) for row in rows[:1000]]
    displayed = [tuple(row) for row in actual['rows']]
    if any(len(row) != len(columns) for row in wanted + displayed):
        raise ValueError('Incomplete surrogate display row')
    if Counter(displayed) != Counter(wanted):
        raise ValueError('Displayed surrogate cells or population differ')
    return dict(passed=True, displayed_rows=len(displayed), source_rows=len(rows),
        cells=len(displayed)*len(columns), truncated=len(rows)>1000)


def check_well_partition(population, held_out, report):
    """Verify a nonempty strict whole-well partition using database identities."""
    all_rows = {r['prcfo']: r for r in population}
    chosen = [r['prcfo'] for r in held_out]
    if len(all_rows) != len(population) or len(chosen) != len(set(chosen)):
        raise ValueError('Duplicate surrogate object identity')
    if not chosen or not set(chosen) < set(all_rows):
        raise ValueError('Unknown or non-held-out surrogate object population')
    group = lambda r: (r['plateID'], r['rowID'], r['columnID'])
    test_groups = {group(all_rows[k]) for k in chosen}
    expected = {k for k, row in all_rows.items() if group(row) in test_groups}
    if set(chosen) != expected:
        raise ValueError('A well is split across surrogate training and testing')
    all_groups = {group(row) for row in population}
    if report['group_by'] != 'well':
        raise ValueError('Wrong surrogate isolation unit')
    check_metric_values(report, dict(train_cells=len(population)-len(chosen),test_cells=len(chosen),
        train_groups=len(all_groups)-len(test_groups),test_groups=len(test_groups),
        total_groups=len(all_groups),cell_fraction=len(chosen)/len(population),
        group_fraction=len(test_groups)/len(all_groups)))
    return dict(passed=True,held_out_objects=len(chosen),training_objects=len(population)-len(chosen),
        held_out_wells=[list(g) for g in sorted(test_groups)],shared_wells=0)
