"""Reuse verified CV/ML footage around the current genuine main-module route."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from compose_report_capture import _frame, _read
from stage_lesson import DEFAULT_STAGE, write


def check_navigation(proof):
    rows = proof.get('folds', [])
    expected = {'classifier_evaluation', 'explain_cv', 'activation', 'train_compare', 'feature_explorer'}
    if (proof.get('model_started') is not False or proof.get('family_restored') != 'cv'
            or proof.get('nested_feature_explorer_opened') is not True
            or len(rows) != 5 or {row['key'] for row in rows} != expected
            or any(not row['tooltip'] or row['tooltip'] != row['displayed_tooltip'] for row in rows)):
        raise ValueError('The exact five visible routes and navigation-only boundary must hold')


def compose(stage=DEFAULT_STAGE):
    stage = Path(stage).resolve()
    root = stage / 'captures'
    destination = root / 'classify_main_verified_overview_v2'
    if destination.exists():
        raise FileExistsError('Preserve the earlier composed overview')
    locations = [
        ('main', root / 'classify_main_family_and_folds_v4', None),
        ('ml', root / 'classify_ml_ten_percent_native',
         ['02_data_choice_load', '19_setting_classes', '23_batch_finished', '24_batch_figure']),
        ('cv', stage / 'classify_canonical_capture_v2/captures/classify_canonical_existing_v2',
         ['19_setting_generate_training_dataset', '24_batch_figure', '30_ai_unsent_question']),
    ]
    hashes, frames, sources, proofs = {}, {}, [], {}
    for prefix, source, wanted in locations:
        proof = _read(source / 'scientific_acceptance.json', hashes)
        if proof.get('accepted') is not True:
            raise ValueError('Every reused workflow needs its preserved acceptance evidence')
        if prefix == 'main':
            check_navigation(proof)
        else:
            batch = _read(source / 'batch_acceptance.json', hashes)
            if batch.get('accepted') is not True:
                raise ValueError('A reused result clip must be a completed native run')
        provenance = _read(source / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'classify_merged':
            raise ValueError('Expected the actual unified Classify module')
        sources.append(provenance); proofs[prefix] = proof
        available = _read(source / 'frames.json', hashes)
        for key in available if wanted is None else wanted:
            frame = deepcopy(available[key])
            path = _frame(source / frame['image'], frame['sha256'], source, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(source)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('An original recording changed during composition')
    destination.mkdir()
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[0], sources=sources,
          composition_only=True, app_source_modified=False))
    write(destination / 'scientific_acceptance.json', dict(accepted=True,
          scope='Current main-module navigation plus reused completed CV/ML result footage',
          sources=proofs, source_hashes=hashes, models_rerun_for_overview=False,
          different_examples_not_direct_model_comparison=True,
          filename_parser_fixed=False, biological_validation=False, published=False))
    print(destination)


if __name__ == '__main__':
    compose()
