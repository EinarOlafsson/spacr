"""Join genuine preparation and Classify frames with their preserved inputs.

This demonstrates an explicit canonical-name preparation workaround, not a
repair to the application's legacy filename parser or a useful trained model.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from classify_split_evidence import inspect_finished, inspect_metrics
from compose_report_capture import _frame, _read, _same_hash
from stage_lesson import DEFAULT_STAGE, write


def compose(stage=DEFAULT_STAGE):
    stage = Path(stage).resolve()
    terminal = stage / 'captures/classify_canonical_preparation'
    gui = stage / 'classify_canonical_capture_v2/captures/classify_canonical_existing_v2'
    destination = stage / 'captures/classify_canonical_verified'
    if destination.exists():
        raise FileExistsError('Preserve the existing accepted composition')
    hashes = {}
    preparation = _read(terminal / 'scientific_acceptance.json', hashes)
    recorded = _read(gui / 'scientific_acceptance.json', hashes)
    batch = _read(gui / 'batch_acceptance.json', hashes)
    dataset = stage / 'classify_canonical_split_v2'
    proof = inspect_finished(dataset)
    metrics = inspect_metrics(dataset)
    if (not batch.get('accepted') or preparation.get('accepted') is not True
            or preparation.get('preparation_only') is not True
            or preparation.get('model_started') is not False
            or preparation.get('filename_parser_fixed') is not False
            or preparation.get('destination') != str(dataset)
            or preparation.get('inputs') != proof['inputs']
            or recorded.get('inputs') != proof['inputs']
            or recorded.get('native_audits') != proof['native_audits']
            or recorded.get('independent_test_metrics') != metrics):
        raise ValueError('Preparation, native run and independently checked outputs must agree')
    _same_hash(Path(__file__).with_name('prepare_classify_split.py'),
               preparation['helper_sha256'], hashes)
    frames, sources = {}, []
    for prefix, root in [('prepare', terminal), ('gui', gui)]:
        provenance = _read(root / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'classify_merged':
            raise ValueError('Expected completed genuine Classify captures')
        sources.append(provenance)
        for key, original in _read(root / 'frames.json', hashes).items():
            frame = deepcopy(original)
            path = _frame(root / frame['image'], frame['sha256'], root, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(root)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed while being read')
    destination.mkdir()
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[-1], sources=sources,
          composition_only=True, app_source_modified=False))
    write(destination / 'scientific_acceptance.json', dict(accepted=True,
          scope='Explicit canonical-name preparation and native existing-split CV run',
          preparation=preparation, native=proof, independent_test_metrics=metrics,
          batch_acceptance=batch, source_hashes=hashes, filename_parser_fixed=False,
          biological_validation=False, useful_model_claimed=False, published=False))
    print(destination)


if __name__ == '__main__':
    compose()
