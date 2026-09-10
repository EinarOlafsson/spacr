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


def verify_gui_predictions(table, saved, filter_text=''):
    """Compare the complete bounded GUI table with its real saved records."""
    if not saved or len(saved) > 2000:
        raise ValueError('This tutorial check requires a complete bounded prediction table')
    columns = list(saved[0])
    if table['columns'] != columns:
        raise ValueError('Prediction columns differ')
    terms = filter_text.casefold().split()
    expected = [r for r in saved if all(term in ' '.join(str(r[c]) for c in columns).casefold()
                                      for term in terms)]
    expected = {r['basename']: r for r in expected}
    if len(expected) != sum(all(term in ' '.join(str(r[c]) for c in columns).casefold()
                               for term in terms) for r in saved):
        raise ValueError('The saved prediction identities are duplicated')
    actual = {}
    for row in table['rows']:
        if len(row) != len(columns):
            raise ValueError('A GUI prediction row has missing cells')
        values = dict(zip(columns, row))
        if values['basename'] in actual:
            raise ValueError('A GUI prediction is duplicated')
        actual[values['basename']] = values
    if set(actual) != set(expected):
        raise ValueError('Filtered prediction identities differ')
    checked = 0
    for name, row in actual.items():
        for column in columns:
            value = expected[name][column]
            if column in {'fold','true_label','predicted_label','confidence'} or column.startswith(('prob_', 'raw_prob_')):
                check_numbers({column: row[column]}, {column: float(value)}, formatted=True)
            elif row[column] != str(value):
                raise ValueError('Prediction text differs: ' + column)
            checked += 1
    return dict(passed=True, rows=len(actual), checked_cells=checked, filter_text=filter_text)


def verify_error_lists(snapshot, saved, true_class='infected_1', predicted_class='infected_2'):
    """Independently check the unchanged error cell and both confidence lists."""
    threshold = float(snapshot['threshold'])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid inspection confidence threshold')
    rows = [r for r in saved if r['true_class'] == true_class and r['predicted_class'] == predicted_class]
    high = sorted([r for r in rows if float(r['confidence']) >= threshold], key=lambda r: -float(r['confidence']))
    low = sorted([r for r in rows if float(r['confidence']) < threshold], key=lambda r: float(r['confidence']))
    for key, expected in [('high',high),('low',low)]:
        actual = snapshot[key]
        if not expected:
            if actual != ['(none)']:
                raise ValueError('The empty confidence list differs')
            continue
        if len(actual) != len(expected):
            raise ValueError('Confidence list count differs')
        for text, row in zip(actual, expected):
            confidence, name = text.split(maxsplit=1)
            if name != row['basename'] or abs(float(confidence) - float(row['confidence'])) > .000500001:
                raise ValueError('Confidence list identity, order or rounded value differs')
    return dict(passed=True, error_cell_count=len(rows), high=len(high), low=len(low), threshold=threshold,
                causal_diagnosis_validated=False)
