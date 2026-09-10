"""Small independent arithmetic reference for the recorded binary classifier.

No application, sklearn, pandas, model, or GUI imports. This checks values,
not the scientific adequacy of the known-overlap train/test split.
"""
from collections import Counter
import math
from statistics import mean


def binary_reference(rows, bins=5):
    if not rows or not isinstance(bins, int) or bins < 2:
        raise ValueError('Nonempty predictions and at least two bins are required')
    normal = []
    for row in rows:
        y = row['true_label']
        p = [float(row['prob_class_0']), float(row['prob_class_1'])]
        if y not in (0, 1) or any(not math.isfinite(v) or v < 0 for v in p) or sum(p) <= 0:
            raise ValueError('Invalid binary label or probability')
        p = [v / sum(p) for v in p]
        predicted = 0 if p[0] >= p[1] else 1
        if row['predicted_label'] != predicted:
            raise ValueError('The class call differs from probability argmax')
        normal.append(dict(y=y, probabilities=p, predicted=predicted, confidence=max(p)))
    n = len(normal)
    counts = Counter((r['y'], r['predicted']) for r in normal)
    support = [sum(r['y'] == k for r in normal) for k in (0, 1)]
    if not all(support):
        raise ValueError('The tutorial reference requires both recorded classes')
    recall, precision, f1 = [], [], []
    for k in (0, 1):
        tp = counts[k, k]
        predicted_n = sum(r['predicted'] == k for r in normal)
        rec, prec = tp / support[k], tp / predicted_n if predicted_n else 0
        recall.append(rec)
        precision.append(prec)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    calibration = []
    for k in (0, 1):
        for b in range(bins):
            low, high = b / bins, (b + 1) / bins
            members = [r for r in normal if low <= r['probabilities'][k] < high
                       or b == bins - 1 and r['probabilities'][k] == 1]
            if members:
                confidence = mean(r['probabilities'][k] for r in members)
                observed = mean(r['y'] == k for r in members)
                calibration.append(dict(class_index=k, bin=b + 1, bin_lower=low,
                    bin_upper=high, n=len(members), mean_confidence=confidence,
                    observed_frequency=observed, calibration_gap=observed-confidence))
    ece = 0
    for b in range(bins):
        members = [r for r in normal if b / bins <= r['confidence'] < (b + 1) / bins
                   or b == bins - 1 and r['confidence'] == 1]
        if members:
            ece += len(members) / n * abs(mean(r['y'] == r['predicted'] for r in members)
                                          - mean(r['confidence'] for r in members))
    summary = dict(n=n, accuracy=sum(r['y'] == r['predicted'] for r in normal) / n,
        balanced_accuracy=mean(recall), recall_macro=mean(recall), precision_macro=mean(precision),
        f1_macro=mean(f1), f1_weighted=sum(f1[k] * support[k] for k in (0, 1)) / n,
        log_loss=-mean(math.log(max(r['probabilities'][r['y']], 2.220446049250313e-16)) for r in normal),
        brier_multiclass=mean(sum((r['probabilities'][k] - (r['y'] == k)) ** 2 for k in (0, 1)) for r in normal),
        mean_confidence=mean(r['confidence'] for r in normal), expected_calibration_error=ece)
    return dict(summary=summary, calibration=calibration, observations=normal,
                confusion=[[counts[a, b] for b in (0, 1)] for a in (0, 1)])


def check_numbers(actual, expected, *, formatted=False):
    for key, value in expected.items():
        found = float(actual[key])
        if not math.isfinite(found) or not math.isclose(found, value,
                rel_tol=5e-5 if formatted else 1e-10, abs_tol=1e-8 if formatted else 1e-12):
            raise ValueError('Recorded numeric value differs: ' + key)


def verify_saved(rows, summary, calibration):
    reference = binary_reference(rows)
    check_numbers(summary, reference['summary'])
    actual = {(int(r['class_index']), int(r['bin'])): r for r in calibration}
    expected = {(r['class_index'], r['bin']): r for r in reference['calibration']}
    if len(actual) != len(calibration) or set(actual) != set(expected):
        raise ValueError('Calibration bin inventory differs')
    for key, values in expected.items():
        check_numbers(actual[key], values)
    return dict(passed=True, predictions=len(rows), summary_fields=len(reference['summary']),
                calibration_rows=len(calibration), confusion=reference['confusion'],
                biological_validation=False)


def verify_gui_matrix(table, reference, names=('infected_1', 'infected_2')):
    rows = {r[0]: dict(zip(table['columns'][1:], r[1:])) for r in table['rows']}
    if len(rows) != len(table['rows']) or set(rows) != set(names):
        raise ValueError('Actual confusion rows differ')
    for index, name in enumerate(names):
        if set(rows[name]) != set(names):
            raise ValueError('Actual confusion columns differ')
        check_numbers(rows[name], dict(zip(names, reference['confusion'][index])))
    return dict(passed=True, cells=4)
