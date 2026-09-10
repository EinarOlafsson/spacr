"""Make a portable, explicitly inspection-only copy of the real saved bundle.

Only the sample display paths change; numeric cells and database identities
are preserved. The archive does not include an annotation database or crops.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

FILES = ('calibration.csv', 'calibration.png', 'confusion_counts.csv',
         'confusion_matrix.png', 'confusion_normalized.csv', 'leakage.json',
         'per_plate_metrics.csv', 'oof_predictions.csv', 'summary.json',
         'evaluation_manifest.json')
ROOT = 'REAL_evaluation_known_overlap'
README = '''REAL SAVED PREDICTIONS — KNOWN TRAIN/TEST WELL OVERLAP

Extract into a NEW folder. In spaCR open Classify > Evaluation, then Browse
to REAL_evaluation_known_overlap (the folder containing evaluation_manifest.json).
This is a saved-results INSPECTOR, not a model-training or evaluation form.

234 real one-epoch test predictions: inherited classes infected_1 and infected_2.
The class names do not establish biological infection states. Counts are
[[104, 11], [8, 111]], giving 215/234 accuracy. All THREE test wells also occur
in training, so this is NOT independent validation. The earlier legacy filename
audit missed this overlap; the supplied audit uses actual database identities.

Files were produced with spacr.classifier_evaluation.evaluate_predictions and
write_evaluation_bundle. No model was retrained or probabilities invented.
The standard name oof_predictions.csv does NOT make this single test split
out-of-fold or nested validation. Calibration method is none; the table has
five bins per class. The confidence split in the inspector does not recalibrate
probabilities or change predicted classes.

Try Predictions filter r5_c2: 234 -> 128 rows, then clear it: 234 again.
In Confusion, true infected_1 / predicted infected_2 contains 11 disagreements.
Threshold 0.75 splits them 6 high / 5 low; 0.95 splits them 2 / 9.
High confidence does NOT prove a wrong annotation. Automatic model-versus-well
or plate-cause suggestions are hypotheses, not conclusions from this one plate.

Every numeric CSV cell and every database-derived identity is unchanged.
Only sample paths now read crops-not-included/<basename>. These are provenance
references, NOT usable image paths. Crops and the annotation database are NOT
included. For crop routing, open the matching original experiment in Annotate
first; do not substitute another experiment or expect this archive to supply it.
The tutorial records the six read-only inspector tabs, not working crop routing.

Two native PNG exports are preserved byte-for-byte. Their OOF titles are generic,
not a validation claim. The producing build warned about faint figure colours;
inspect exports before publication. The GUI provides readable numeric tables.
No annotation is changed and no spaCR AI provider call is needed for this lesson.
'''


def members(source):
    source = Path(source).resolve()
    content = {}
    for name in FILES:
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('Bundle inputs must be local regular files without links')
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('Unexpectedly large recorded inspector input')
        content[name] = path.read_bytes()
    manifest = json.loads(content['evaluation_manifest.json'])
    if set(manifest['files'].values()) != set(FILES):
        raise ValueError('The recorded manifest file inventory differs')
    if manifest.get('leakage_passed') is not False:
        raise ValueError('This example must disclose known leakage')
    rows = list(csv.DictReader(io.StringIO(content['oof_predictions.csv'].decode())))
    if len(rows) != 234 or len({r['basename'] for r in rows}) != 234:
        raise ValueError('Expected the 234 distinct recorded test predictions')
    for row in rows:
        name = row['basename']
        if (PurePosixPath(name).name != name or '\\' in name or
                not name.startswith('plate1_') or not name.endswith('.png')):
            raise ValueError('Unsafe recorded crop basename')
        row['sample'] = 'crops-not-included/' + name
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content['oof_predictions.csv'] = output.getvalue().encode()
    content['README.txt'] = README.encode()
    content['tutorial_provenance.json'] = json.dumps(dict(
        real_saved_predictions=True, independent_validation=False,
        original_prediction_csv_sha256=hashlib.sha256((source / 'oof_predictions.csv').read_bytes()).hexdigest(),
        changes=['sample display paths replaced by explicit crops-not-included references'],
        crops_included=False, annotation_database_included=False,
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in content.items()}),
        sort_keys=True, indent=2).encode() + b'\n'
    return content


def build_archive(source, destination):
    content = members(source)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            entry = zipfile.ZipInfo(ROOT + '/' + name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compresslevel=9)
    data = buffer.getvalue()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError('Refusing to replace a different tutorial archive')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
    if members(source) != content:
        raise ValueError('The recorded source changed while packaging')
    with zipfile.ZipFile(destination) as archive:
        if {n.removeprefix(ROOT + '/'): archive.read(n) for n in archive.namelist()} != content:
            raise ValueError('The download differs from its expected contents')
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                member_sha256={k: hashlib.sha256(v).hexdigest() for k, v in content.items()},
                independent_validation=False, crops_included=False)


def prepare_capture(source, archive, runs):
    """Extract only the just-verified fixed members into a new private run."""
    source = Path(source).resolve()
    expected = members(source / 'evaluation')
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != [ROOT + '/' + k for k in sorted(expected)]:
            raise ValueError('Archive inventory differs')
        if any(z.read(ROOT + '/' + k) != v for k, v in expected.items()):
            raise ValueError('Archive member differs')
        root = Path(tempfile.mkdtemp(prefix='REAL-portable-inspector-', dir=runs))
        for key in expected:
            target = root / 'evaluation' / key
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(z.read(ROOT + '/' + key))
    proof = json.loads((source / 'preparation.json').read_text())
    proof.update(root=str(root), manifest=str(root / 'evaluation/evaluation_manifest.json'),
                 archive=str(Path(archive).resolve()), archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                 portable_inspection_only=True, crops_included=False,
                 source_preparation=str(source / 'preparation.json'))
    (root / 'preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--capture-runs', type=Path)
    args = parser.parse_args()
    receipt = build_archive(args.source / 'evaluation', args.destination)
    if args.capture_runs:
        receipt['capture_root'] = str(prepare_capture(args.source, args.destination, args.capture_runs))
    print(json.dumps(receipt, indent=2))
