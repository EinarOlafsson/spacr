"""Bounded native ML recording and independent saved-object lineage checks.

The two labels are plate columns, not independently validated phenotypes.
Only the GUI sets these values; this helper does not invoke the classifier.
"""
from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path


def bounded_settings():
    return {
        'classifier_family': 'ml', 'dataset_mode': 'metadata',
        'classes': {'column_c1': {'column': 'columnID', 'value': 'c1'},
                    'column_c2': {'column': 'columnID', 'value': 'c2'}},
        'model_type_ml': 'random_forest', 'channel_of_interest': 1,
        'n_estimators': 16, 'n_jobs': 2, 'n_repeats': 1, 'top_features': 10,
        # Match the merged form's shared test_split=0.1. Its current
        # normalization overrides the ML-specific test_size value; record
        # the actual 10% request, never narrate the discarded 50% value.
        'test_size': 0.1, 'cv_group_by': 'well', 'cross_validation': False,
        'prune_features': False, 'batch_correction': 'none', 'plot': True,
    }


def inspect_output(database, output, *, requested_fraction=0.1):
    """Resolve every saved identity against actual metadata; refuse overlap."""
    with sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True) as con:
        records = con.execute('SELECT prcfo, plateID, rowID, columnID FROM png_list').fetchall()
    identities = {}
    for identity, *well in records:
        well = tuple(str(value) for value in well)
        if str(identity) in identities and identities[str(identity)] != well:
            raise ValueError('Ambiguous source object identity')
        identities[str(identity)] = well
    with Path(output).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No saved ML objects')
    groups = {'train': set(), 'test': set()}
    classes = {'train': set(), 'test': set()}
    counts = {'train': 0, 'test': 0, 'not_used': 0}
    seen, reasons = set(), []
    for row in rows:
        identity = row['prcfo']
        if identity in seen or identity not in identities:
            raise ValueError('Duplicate or unknown saved ML identity')
        seen.add(identity)
        usage = row['data_usage']
        if usage not in counts:
            raise ValueError(f'Unknown data usage: {usage}')
        counts[usage] += 1
        if usage in groups:
            groups[usage].add(identities[identity])
            classes[usage].add(identities[identity][2])
        if row['data_usage_group_by'] != 'well':
            raise ValueError('The result does not describe a well-level split')
        if not math.isclose(float(row['split_requested_fraction']), requested_fraction):
            raise ValueError('Saved split request disagrees with the recorded setting')
        probabilities = [float(row[f'prediction_probability_class_{i}']) for i in (0, 1)]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
            raise ValueError('Non-finite or out-of-range class probability')
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-8):
            raise ValueError('Class probabilities do not sum to one')
        prediction = float(row['predictions'])
        if prediction not in (0, 1):
            raise ValueError('Non-binary saved prediction')
        if probabilities[int(prediction)] < max(probabilities) - 1e-8:
            raise ValueError('Prediction disagrees with maximum class probability')
    overlap = sorted(groups['train'] & groups['test'])
    if overlap:
        reasons.append('Actual database wells occur in both train and test')
    if any(values != {'c1', 'c2'} for values in classes.values()):
        reasons.append('Both plate-column labels must occur in each split')
    return {'accepted': not reasons, 'reasons': reasons, 'rows': len(rows),
            'sample_counts': counts, 'actual_wells': {k: sorted(v) for k, v in groups.items()},
            'overlapping_actual_wells': overlap,
            'labels_are_plate_columns_not_validated_phenotypes': True,
            'all_saved_probabilities_checked': True, 'output': str(output)}


def independent_metrics(database, output, metrics_file):
    """Recalculate the saved holdout report and check database write-back.

    This example's explicit c1/c2 metadata defines class zero/one. Neither
    the classifier nor sklearn's metric implementation is used here.
    """
    with Path(output).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True) as con:
        saved = {str(identity): (str(column), prediction, probability)
                 for identity, column, prediction, probability in con.execute(
                     'SELECT prcfo,columnID,predictions,ml_pred FROM png_list')}
    confusion = [[0, 0], [0, 0]]
    for row in rows:
        column, prediction, probability = saved[row['prcfo']]
        if (float(prediction) != float(row['predictions'])
                or not math.isclose(float(probability),
                                    float(row['prediction_probability_class_1']), abs_tol=1e-12)):
            raise ValueError('Database predictions disagree with results.csv')
        if row['data_usage'] == 'test':
            if column not in {'c1', 'c2'}:
                raise ValueError('Unknown held-out metadata label')
            confusion[int(column == 'c2')][int(float(row['predictions']))] += 1
    count = sum(map(sum, confusion))
    if not count:
        raise ValueError('No held-out predictions')
    expected = {}
    for label in (0, 1):
        true_positive = confusion[label][label]
        support = sum(confusion[label])
        predicted = sum(row[label] for row in confusion)
        precision = true_positive / predicted if predicted else 0
        recall = true_positive / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        expected[str(label)] = dict(precision=precision, recall=recall, **{'f1-score': f1}, support=support)
    accuracy = sum(confusion[i][i] for i in (0, 1)) / count
    with Path(metrics_file).open(newline='') as handle:
        metrics = {row['']: row for row in csv.DictReader(handle)}
    errors = [abs(float(metrics[label][field]) - value)
              for label, values in expected.items() for field, value in values.items()]
    errors.append(abs(float(metrics['accuracy']['precision']) - accuracy))
    if max(errors) > 1e-12:
        raise ValueError('Saved holdout metrics disagree with actual predictions')
    return {'confusion_matrix_true_rows_predicted_columns': confusion,
            'class_zero': 'columnID=c1', 'class_one': 'columnID=c2',
            'test_objects': count, 'accuracy': accuracy,
            'majority_class_accuracy_on_test': max(map(sum, confusion)) / count,
            'metrics_compared': len(errors), 'maximum_metric_error': max(errors),
            'database_predictions_checked': len(rows)}
