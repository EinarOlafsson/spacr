"""Check the actual exported CV inputs against authoritative source metadata."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import sqlite3

from prepare_classify_split import TRAIN_WELLS, TEST_WELLS, marker_profile, MARKER


def inspect_inputs(dataset):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / 'tutorial_input_manifest.json').read_text())
    if manifest.get('accepted') is not True or len(manifest.get('records', [])) != 64:
        raise ValueError('Expected the documented 64 unchanged example crops')
    database = Path(manifest['source_database'])
    if hashlib.sha256(database.read_bytes()).hexdigest() != manifest['source_database_sha256']:
        raise ValueError('Source database changed')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as con:
        rows = con.execute('SELECT prcfo,plateID,rowID,columnID,infected FROM png_list').fetchall()
    identities = {r[0]: (tuple(r[1:4]), r[4]) for r in rows}
    if len(identities) != len(rows):
        raise ValueError('Source identities are not unique')
    groups = {'train': set(), 'test': set()}
    counts = {'train': 0, 'test': 0}
    seen, image_hashes, expected = set(), set(), set()
    for row in manifest['records']:
        identity = row['prcfo']
        if identity in seen or identity not in identities:
            raise ValueError('Unknown or duplicate exported identity')
        seen.add(identity)
        well, label = identities[identity]
        if tuple(row['well']) != well or row['infected'] != label or row['class_name'] != f'infected_{label}':
            raise ValueError('Exported class or well differs from the database')
        split = row['split']
        if split not in groups:
            raise ValueError('Unknown split')
        target = dataset / split / row['class_name'] / (identity + '.png')
        if str(target) != row['target']:
            raise ValueError('Exported path differs from the canonical identity')
        for key in ('source', 'source_marker'):
            if hashlib.sha256(Path(row[key]).read_bytes()).hexdigest() != row[key + '_sha256']:
                raise ValueError('An original crop or format marker changed')
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('A prepared crop changed')
        if row['source_sha256'] in image_hashes:
            raise ValueError('Duplicate image content')
        image_hashes.add(row['source_sha256'])
        if marker_profile((target.parent / MARKER).read_bytes()) != marker_profile(Path(row['source_marker']).read_bytes()):
            raise ValueError('Exported storage format differs')
        groups[split].add(well)
        counts[split] += 1
        expected.add(target)
    actual = {p for split in ('train', 'test') for p in (dataset / split).glob('*/*.png')}
    if expected != actual or counts != {'train': 32, 'test': 32}:
        raise ValueError('The prepared split gained or lost images')
    if groups['train'] != TRAIN_WELLS or groups['test'] != TEST_WELLS or groups['train'] & groups['test']:
        raise ValueError('The prepared split has wrong or overlapping actual database wells')
    return {'accepted': True, 'source_database_sha256': manifest['source_database_sha256'],
            'objects_checked': len(seen), 'sample_counts': counts,
            'actual_wells': {k: sorted(map(list, v)) for k, v in groups.items()},
            'overlapping_database_wells': [], 'all_image_bytes_unchanged': True,
            'labels_preserved_from_example': True, 'filename_parser_fixed': False}


def inspect_finished(dataset):
    dataset = Path(dataset).resolve()
    proof = inspect_inputs(dataset)
    reports = {}
    for filename, counts in [('train_test_leakage_audit.json', (32, 32)),
                             ('train_validation_leakage_audit.json', (16, 16))]:
        matches = list((dataset / 'model').rglob(filename))
        if len(matches) != 1:
            raise ValueError('Expected one fresh native split audit: ' + filename)
        report = json.loads(matches[0].read_text())
        check_native_audit(report, counts)
        reports[filename] = report
    return dict(accepted=True, scope='Preserved canonical inputs and native grouped boundary audits',
                inputs=proof, native_audits=reports, saved_metrics_independently_checked=False,
                biological_validation=False, published=False)


def check_native_audit(report, counts):
    if (report.get('passed') is not True or report.get('group_by') != 'well'
            or report.get('critical_levels') or report.get('hash_errors')
            or report.get('unverifiable_counts')
            or (report.get('train_samples'), report.get('validation_samples')) != counts
            or any(report.get('overlap_counts', {}).get(k) != 0 for k in
                   ('well', 'field', 'object', 'exact', 'content_sha256', 'augmentation_family'))):
        raise ValueError('Native split audit did not pass the documented grouped boundaries')


def check_predictions(rows, metric, truth):
    """Recompute accuracy and macro F1; labels must match the actual manifest."""
    if len(rows) != len(truth) or not rows or len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Expected exactly one test prediction per original test object')
    confusion = [[0, 0], [0, 0]]
    for row in rows:
        if row['filename'] not in truth or int(row['true_label']) != truth[row['filename']]:
            raise ValueError('Test labels or identities differ from the original database')
        y, p = int(row['true_label']), int(row['predicted_label'])
        probs = [float(row[f'prob_class_{c}']) for c in (0, 1)]
        if (y not in (0, 1) or p not in (0, 1)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs)
                or not math.isclose(sum(probs), 1, abs_tol=1e-6)
                or probs[p] < max(probs) - 1e-8):
            raise ValueError('Invalid saved class probability or prediction')
        confusion[y][p] += 1
    support = list(map(sum, confusion))
    if min(support) <= 0:
        raise ValueError('Both saved example classes must occur in the test set')
    accuracy = sum(confusion[c][c] for c in (0, 1)) / len(rows)
    f1 = sum(2 * confusion[c][c] / (sum(confusion[c]) + sum(r[c] for r in confusion)) for c in (0, 1)) / 2
    if (not math.isclose(float(metric['accuracy']), accuracy, abs_tol=1e-9)
            or not math.isclose(float(metric['Accuracy']), accuracy, abs_tol=1e-9)
            or not math.isclose(float(metric['f1_macro']), f1, abs_tol=1e-9)
            or ast.literal_eval(metric['class_support']) != support):
        raise ValueError('Saved metrics differ from independently counted predictions')
    return {'predictions_checked': len(rows), 'confusion_matrix': confusion,
            'accuracy': accuracy, 'f1_macro': f1,
            'majority_accuracy': max(support) / len(rows),
            'all_metrics_validated': False, 'probabilities_calibrated': False}


def inspect_metrics(dataset):
    dataset = Path(dataset).resolve()
    inspect_inputs(dataset)
    manifest = json.loads((dataset / 'tutorial_input_manifest.json').read_text())
    truth = {r['target']: r['infected'] - 1 for r in manifest['records'] if r['split'] == 'test'}
    predictions = list((dataset / 'model').rglob('*_test_acc.csv'))
    metrics = list((dataset / 'model').rglob('*_test_result.csv'))
    if len(predictions) != 1 or len(metrics) != 1:
        raise ValueError('Expected one fresh saved evaluation, not mixed historical outputs')
    with predictions[0].open() as h:
        rows = list(csv.DictReader(h))
    with metrics[0].open() as h:
        saved = list(csv.DictReader(h))
    if len(saved) != 1:
        raise ValueError('Expected one actual test evaluation row')
    return check_predictions(rows, saved[0], truth)
