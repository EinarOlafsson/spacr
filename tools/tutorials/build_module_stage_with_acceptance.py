#!/usr/bin/env python3
"""Stage lesson 20 only after rechecking its scoped, source-bound acceptance.

The original observations and capture failures remain unchanged. This creates
a local stage, not a publication, accuracy validation or listening review.
Selections come from the explicit acceptance record; --pick can only confirm
that same selection. No failed clip is silently omitted to complete a lesson.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/tutorials')]

from audit_staged_catalogs import CATALOGS
from compose_capture_set import compose
from stage_lesson import stage_lesson, write
from validate_cellpose_host import (
    ACCEPTANCE_SCOPE, Evidence, LESSON_ID, SCOPE_FLAGS, check_clip, check_verifier,
    child, require,
)


def captures_of(path):
    path = Path(path).resolve()
    if (path / 'frames.json').is_file():
        return path
    found = sorted(path.glob('captures/*/frames.json'))
    require(len(found) == 1, f'{path}: expected one capture folder, found {found}')
    return found[0].parent


def preflight(acceptance_path, lesson_path, sources, *, picks=None):
    """Recheck selected evidence before creating any staging destination."""
    import hashlib

    evidence = Evidence()
    acceptance = evidence.read(acceptance_path)
    lesson = evidence.read(lesson_path)
    canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()
    require(lesson.get('id') == LESSON_ID and acceptance.get('schema') == 1
            and acceptance.get('accepted') is True and acceptance.get('lesson') == LESSON_ID
            and acceptance.get('canonical_english_sha256') == canonical,
            'Scientific acceptance is not bound to the current English lesson 20')
    require(acceptance.get('scope') == ACCEPTANCE_SCOPE
            and all(acceptance.get(name) is False for name in SCOPE_FLAGS),
            'Acceptance exceeds its attribution/filter/selected-clip scope')
    hashes = acceptance.get('source_hashes')
    require(isinstance(hashes, dict) and hashes, 'Acceptance lacks source hash bindings')
    for path, digest in hashes.items():
        require(Path(path).is_absolute(), 'Acceptance source paths must be host absolute paths')
        evidence.bind(path, digest)
    selected = acceptance.get('selected_clips')
    require(isinstance(selected, list) and selected, 'Acceptance has no explicit selected clips')
    names = [row.get('scene') for row in selected]
    require(all(isinstance(name, str) for name in names) and len(names) == len(set(names)),
            'Acceptance repeats or omits a selected scene identity')
    visuals = [scene['visual'] for scene in lesson['scenes']]
    require(visuals and len(visuals) == len(set(visuals)) and set(names) == set(visuals),
            'Acceptance must cover every English scene exactly once')
    sources = [Path(source).resolve() for source in sources]
    require(sources and len(sources) == len(set(sources)), 'Capture sources must be distinct')
    groups = {str(source): [] for source in sources}
    for row in selected:
        require(isinstance(row.get('capture'), str) and Path(row['capture']).is_absolute()
                and row['capture'] in groups, 'Acceptance names an unsupplied capture source')
        groups[row['capture']].append(row['scene'])
    require(all(groups.values()), 'Every supplied source must contribute explicitly selected clips')
    if picks is not None:
        require(set(picks) == set(groups)
                and all(isinstance(picks[key], list)
                        and len(picks[key]) == len(set(picks[key]))
                        and set(picks[key]) == set(groups[key]) for key in groups),
                'Composition selection differs from scientific acceptance')
    observations = acceptance.get('native_observations')
    require(isinstance(observations, list) and len(observations) == len(sources)
            and {row.get('capture') for row in observations} == set(groups),
            'Acceptance must preserve each contributing raw observation exactly once')
    observations = {row['capture']: row for row in observations}
    selected_by_scene = {row['scene']: row for row in selected}
    expected_frames = {}
    for source in sources:
        source_key = str(source)
        observed = observations[source_key]
        required_json = [source / name for name in (
            'scientific_acceptance.json', 'independent_reference.json', 'frames.json',
            'provenance.json', 'host_gate_summary.json')]
        receipt_path = Path(observed.get('verifier_receipt_path', ''))
        require(receipt_path.is_absolute(), 'Raw observation lacks its verifier receipt path')
        required_json.append(receipt_path)
        require(all(str(path) in hashes for path in required_json),
                'Acceptance omits a required raw observation hash binding')
        proof = evidence.read(source / 'scientific_acceptance.json')
        provenance = evidence.read(source / 'provenance.json')
        summary = evidence.read(source / 'host_gate_summary.json')
        verifier = evidence.read(receipt_path)
        require(proof.get('accepted') is False and proof == observed.get('raw_observation')
                and provenance == observed.get('original_provenance')
                and summary == observed.get('host_gate_summary')
                and verifier == observed.get('verifier_receipt'),
                'Acceptance does not preserve the current original observations')
        require(provenance.get('completed_capture') is True
                or (provenance.get('completed_capture') is False
                    and provenance.get('status') == 'gate_failed'),
                'Source was not a completed or gate-failed full recording')
        reference = check_verifier(evidence, receipt_path, verifier, source,
                                   source.parent.parent / 'exact-source.json',
                                   sorted(source.glob('batch_figure_*_arrays.npz')))
        require(reference == observed.get('independent_reference'),
                'Acceptance contains a different independent reference')
        frames = evidence.read(source / 'frames.json')
        for name in groups[source_key]:
            require(name in frames, f'Selected scene is missing: {name}')
            frame = frames[name]
            row = selected_by_scene[name]
            clip = frame.get('clip')
            require(isinstance(clip, dict) and {'video', 'native_video', 'receipt'} <= set(clip),
                    f'Selected scene lacks a completed clip: {name}')
            bindings = (
                (child(source, frame['image']), frame['sha256'], row.get('image_sha256')),
                (child(source, clip['video']), clip['sha256'], row.get('video_sha256')),
                (child(source, clip['native_video']), clip['native_sha256'], row.get('native_sha256')),
                (child(source, clip['receipt']), clip['receipt_sha256'], row.get('receipt_sha256')),
                (child(source, summary['clips'][name]['file']), row.get('gate_sha256'), row.get('gate_sha256')),
            )
            require(all(str(path) in hashes and hashes[str(path)] == digest == declared
                        for path, digest, declared in bindings),
                    f'Acceptance omits or mismatches selected clip byte bindings: {name}')
            for suffix in ('host.json', 'pointer.json'):
                require(str(source / f'{name}.{suffix}') in hashes,
                        f'Acceptance omits selected clip telemetry: {name}')
            reviews = {}
            dialog = row.get('result_dialog_evidence') or {}
            if 'review' in dialog:
                review = dialog['review']
                require(hashes.get(review.get('image')) == review.get('image_sha256'),
                        'Acceptance omits terminal review image byte binding')
                reviews[(source_key, name)] = review
            checked = check_clip(evidence, source, name, frame, summary, reviews)
            require(checked == row, f'Acceptance decision differs from the recorded clip: {name}')
            expected_frames[name] = frame
    evidence.unchanged()
    return acceptance, lesson, groups, expected_frames, evidence


def build_stage(stage, *, lesson_path, acceptance_path, sources, catalogs,
                capture_module='cellpose_masks', picks=None, aliases=None,
                accept_verified_partial=False, recorder_delta_paths=(), recorder_delta_reason=None):
    stage = Path(stage).absolute()
    require(not stage.exists(), 'Staging requires a new destination')
    require(Path(capture_module).name == capture_module and capture_module not in ('', '.', '..'),
            'Capture module must be one safe directory name')
    sources = [captures_of(source) for source in sources]
    acceptance, lesson, groups, expected, evidence = preflight(
        Path(acceptance_path).resolve(), Path(lesson_path).resolve(), sources, picks=picks)
    require(all(not stage.resolve().is_relative_to(source) for source in sources),
            'Staging destination must stay outside raw capture directories')
    aliases = aliases or {}
    require(all(key == value for key, value in aliases.items()),
            'Lesson 20 acceptance binds original scene identities; renaming needs new acceptance')
    if not accept_verified_partial:
        require(all(row['original_provenance']['completed_capture'] is True
                    for row in acceptance['native_observations']),
                'Gate-failed recordings require --accept-verified-partial and explicit selections')
    catalog_sources = [Path(catalogs) / 'catalog' / name for name in CATALOGS]
    for path in catalog_sources:
        evidence.bind(path)
    stage.mkdir(parents=True)
    try:
        (stage / 'catalog').mkdir()
        for path in catalog_sources:
            shutil.copyfile(path, stage / 'catalog' / path.name)
        destination = stage / 'captures' / capture_module
        frames = compose(sources, destination, aliases=aliases,
                         accept_verified_partial=accept_verified_partial,
                         source_scenes=groups, recorder_delta_paths=recorder_delta_paths,
                         recorder_delta_reason=recorder_delta_reason)
        require(set(frames) == set(expected), 'Composition changed the accepted scene inventory')
        for name, frame in frames.items():
            row = next(item for item in acceptance['selected_clips'] if item['scene'] == name)
            require(frame['composition_source'] == {'capture': row['capture'], 'visual': name}
                    and frame['clip']['sha256'] == row['video_sha256']
                    and frame['clip']['native_sha256'] == row['native_sha256'],
                    'Composition changed the accepted recorded clip identity')
        evidence.unchanged()
        write(destination / 'scientific_acceptance.json', acceptance)
        stage_lesson(Path(lesson_path).resolve(), capture_module, stage, check_only=True)
        stage_lesson(Path(lesson_path).resolve(), capture_module, stage)
        evidence.unchanged()
        write(stage / 'build-module-stage-report.json', {
            'lesson': lesson['id'], 'scenes': len(frames), 'sources': [str(s) for s in sources],
            'aliases': aliases, 'picks': groups, 'scientific_acceptance_sha256':
                evidence.hashes[str(Path(acceptance_path).resolve())],
            'published': False,
        })
    except BaseException:
        shutil.rmtree(stage)
        raise
    return stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('--lesson', required=True, choices=[LESSON_ID])
    parser.add_argument('--scientific-acceptance', type=Path, required=True)
    parser.add_argument('--capture-module', required=True)
    parser.add_argument('--source', action='append', required=True, type=Path)
    parser.add_argument('--alias', action='append', default=[])
    parser.add_argument('--pick', action='append', default=[])
    parser.add_argument('--accept-verified-partial', action='store_true')
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--recorder-delta-path', action='append', default=[])
    parser.add_argument('--recorder-delta-reason')
    args = parser.parse_args(argv)
    picks = {}
    for value in args.pick:
        run, scenes = value.split('=', 1)
        key = str(captures_of(run))
        require(key not in picks, 'Repeated source selection')
        picks[key] = scenes.split(',')
    aliases = dict(value.split('=', 1) for value in args.alias)
    result = build_stage(args.stage, lesson_path=REPO / 'tools/tutorials/lessons' / f'{args.lesson}.json',
                         acceptance_path=args.scientific_acceptance, sources=args.source,
                         catalogs=args.catalogs, capture_module=args.capture_module,
                         picks=picks or None, aliases=aliases,
                         accept_verified_partial=args.accept_verified_partial,
                         recorder_delta_paths=args.recorder_delta_path,
                         recorder_delta_reason=args.recorder_delta_reason)
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
