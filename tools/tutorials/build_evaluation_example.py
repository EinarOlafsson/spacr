"""Prepare an honest inspector bundle from the recorded classifier's predictions.

No model is trained and no probability is invented. New, byte-identical crop
copies get canonical database identities, so the API's filename parser sees
the actual wells. This exposes the OLD split's overlap; it does not repair the
split or turn its metrics into independent validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import pandas as pd

DEFAULT_STAGE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_identity(record):
    """Validate the database identity rather than guessing from legacy names."""
    expected = '_'.join(str(record[k]) for k in ('plateID', 'rowID', 'columnID', 'fieldID', 'cell_id'))
    if record['prcfo'] != expected or not re.fullmatch(r'[A-Za-z0-9-]+_r\d+_c\d+_f\d+_o\d+', expected):
        raise ValueError('Unsafe or inconsistent database object identity')
    return expected


def prepare(stage=DEFAULT_STAGE):
    from spacr.classifier_evaluation import (audit_split_leakage, evaluate_predictions,
                                             write_evaluation_bundle)
    stage = Path(stage).resolve()
    source = stage / 'annotate_fresh/example_data/plate1'
    dataset = source / 'datasets/training_1'
    run = dataset / 'model/resnet18/rgb/epochs_1'
    prediction_csv = run / 'resnet18_time_260909_test_acc.csv'
    database = source / 'measurements/measurements.db'
    original_audit = run / 'train_test_leakage_audit.json'
    before = {str(p): sha(p) for p in (prediction_csv, database, original_audit)}
    records = pd.read_csv(prediction_csv)
    if len(records) != 234 or records.filename.map(lambda v: Path(v).name).duplicated().any():
        raise ValueError('Expected the recorded 234 distinct test predictions')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        metadata = [dict(r) for r in conn.execute(
            'SELECT png_path,prcfo,plateID,rowID,columnID,fieldID,cell_id FROM png_list')]
    by_name = {Path(r['png_path']).name: r for r in metadata}
    if len(by_name) != len(metadata):
        raise ValueError('Duplicate source crop basenames')
    runs = stage / 'evaluation_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='REAL-predictions-known-well-overlap-', dir=runs))
    paths, lineage = {}, []
    for split in ('train', 'test'):
        paths[split] = {}
        for path in sorted((dataset / split).glob('*/*.png')):
            if path.is_symlink() or not path.resolve().is_relative_to(dataset) or path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError('Unexpected or external recorded crop')
            record = by_name[path.name]
            identity = canonical_identity(record)
            target = root / 'crops' / split / path.parent.name / (identity + '.png')
            if target.exists():
                raise ValueError('Two crops map to the same canonical object')
            before[str(path)] = sha(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if sha(target) != before[str(path)]:
                raise ValueError('The private crop copy differs')
            paths[split][path.name] = target
            lineage.append(dict(original=str(path), copy=str(target), sha256=sha(target),
                                database_identity=identity, split=split))
    if (len(paths['train']), len(paths['test'])) != (2102, 234):
        raise ValueError('The recorded dataset split changed')
    originals = [Path(p).name for p in records.filename]
    if set(originals) != set(paths['test']):
        raise ValueError('Predictions do not exactly cover the recorded test files')
    for original, label in zip(originals, records.true_label):
        if paths['test'][original].parent.name != {0: 'infected_1', 1: 'infected_2'}.get(int(label)):
            raise ValueError('The prediction label disagrees with the actual class folder')
    probabilities = records[['prob_class_0', 'prob_class_1']].to_numpy()
    evaluation = evaluate_predictions(records.true_label.tolist(), probabilities,
        [paths['test'][name] for name in originals], classes=['infected_1', 'infected_2'],
        calibration_method='none', calibration_bins=5)
    if evaluation['predictions'].predicted_label.tolist() != records.predicted_label.tolist():
        raise ValueError('Stored class calls differ from probability argmax')
    # The standard API records a single test fold as fold 0. This is not nested
    # or cross-validated inference; explicitly disclose that in its metadata.
    evaluation['summary']['tutorial_scope'] = (
        'Inspection of real saved one-epoch test predictions; actual database wells overlap training. '
        'Not independent biological validation, not cross-validated or nested predictions. '
        'Class names are inherited dataset labels, not newly validated infection states. '
        'Original probabilities preserved subject only to the API row-normalization of rounding error.')
    audit = audit_split_leakage(list(paths['train'].values()), list(paths['test'].values()),
        group_by='well', split_name='recorded_train_vs_test_with_database_identities',
        hash_content=True, require_identity=True)
    if audit.passed or audit.overlap_counts.get('well') != 3:
        raise ValueError('Expected the three independently observed shared wells')
    manifest = write_evaluation_bundle(root / 'evaluation', evaluation, leakage_reports=[audit])
    if any(sha(p) != value for p, value in before.items()):
        raise ValueError('An original prediction, database, audit or crop changed')
    proof = dict(lesson='39_classifier_evaluation', accepted=False, input_prepared=True,
        root=str(root), manifest=str(manifest), source_csv=str(prediction_csv),
        source_database=str(database), original_inputs=before, crop_lineage=lineage,
        prediction_count=234, class_names=['infected_1', 'infected_2'],
        original_leakage_audit=json.loads(original_audit.read_text()),
        database_identity_audit=audit.to_dict(), independent_validation=False,
        model_retrained=False, probability_values_invented=False,
        original_inputs_preserved=True, app_source_modified=False, published=False,
        next_gate='Independent numerical checks and actual saved-bundle GUI inspection')
    (root / 'preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    print(prepare(args.stage))
