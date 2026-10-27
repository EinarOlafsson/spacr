#!/usr/bin/env python3
"""Combine completed, equal-source capture runs without changing their evidence.

A split recording can retain completed clips while recording only missing
scenes. Each original receipt, image and video is copied byte-for-byte; the
new manifest records every source run and exact input manifest digest.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import math
from pathlib import Path
import re
import shutil
from stage_lesson import read, write


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activation_cohort(proof):
    """Validate recorded science, without loading a model or recomputing maps."""
    from activation_evidence import check_saved_runs
    if (proof.get('accepted') is not True or proof.get('lesson') != '16_activation'
            or proof.get('app_source_modified') is not False
            or proof.get('active_jobs_after_close') != 0
            or proof.get('provider_contacted') is not False
            or not re.fullmatch(r'cuda(?::\d+)?', str(proof.get('reference_device_resolution', {}).get('device', '')))):
        raise ValueError('Every used activation source needs its accepted CUDA scientific proof')
    preserved = proof.get('preservation', {})
    required = {'original_database_and_sidecars_unchanged',
                'original_crops_model_card_and_marker_unchanged',
                'private_archive_model_and_card_unchanged'}
    if not required <= preserved.keys() or any(value is not True for value in preserved.values()):
        raise ValueError('Activation source preservation failed')
    inputs = proof['inputs']
    def hashes(values):
        if (not isinstance(values, dict) or not values or any(
                not isinstance(name, str) or not isinstance(value, str)
                or not re.fullmatch('[0-9a-f]{64}', value) for name, value in values.items())):
            raise ValueError('Invalid activation input hash inventory')
        return values
    original = hashes(inputs['original_hashes'])
    private = hashes(inputs['private_input_hashes'])
    by_name = {Path(path).name: value for path, value in private.items()}
    model, archive = inputs['model'], inputs['archive']
    card = str(Path(model).with_suffix('.card.json'))
    if (set(private) != {model, archive, card} or len(by_name) != 3
            or len({Path(path).name for path in original}) != len(original)):
        raise ValueError('Activation private archive/model/card need unique exact identities')
    originals_by_name = {Path(path).name: value for path, value in original.items()}
    if any(by_name[Path(path).name] != originals_by_name.get(Path(path).name)
           for path in (model, card)):
        raise ValueError('Private activation model/card differ from the original')
    crops = inputs['crops']
    if (len(crops) != 4 or len({row['name'] for row in crops}) != 4
            or any(original.get(row['source']) != row['sha256'] or not row.get('identity')
                   for row in crops)
            or inputs.get('model_validation_claim') is not False
            or inputs.get('biological_claim') is not False):
        raise ValueError('Require the exact four original activation crops and scientific scope')
    reference = proof['independent_reference']
    requested_device = proof['reference_device_resolution']['device']
    error = reference.get('training_input_max_error')
    if (not re.fullmatch(r'cuda:\d+', reference.get('reference_device', ''))
            or reference.get('source_names') != [row['name'] for row in crops]
            or reference.get('original_channel_order_preserved') is not True
            or (requested_device != 'cuda' and requested_device != reference.get('reference_device'))
            or reference.get('training_input_pixels') != 4 * 3 * 128 * 128
            or type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1):
        raise ValueError('Activation reference/input comparison is not accepted CUDA evidence')
    database = inputs['database']
    bundle = inputs['database_bundle']
    if (not isinstance(bundle, dict) or database not in bundle
            or not set(bundle) <= {database, database + '-wal', database + '-shm'}
            or any(not isinstance(row, dict) or set(row) != {'sha256', 'signature'}
                   or not isinstance(row['signature'], list) or len(row['signature']) != 5
                   or any(type(value) is not int for value in row['signature']) for row in bundle.values())):
        raise ValueError('Activation database bundle must bind its original database and sidecars')
    database_hashes = hashes({Path(path).name: row['sha256'] for path, row in bundle.items()})
    runs = proof['runs']
    check_saved_runs(runs)
    settings = {}
    for run in runs:
        current = run['settings']
        if (current.get('dataset') != archive or current.get('model_path') != model
                or run['cuda_execution']['model_path'] != model
                or current.get('cam_type') != run['method']
                or any(Path(row['path']).name != crop['name'] for row, crop in zip(run['maps'], crops))
                or any(device != reference['reference_device']
                       for loaded in run['cuda_execution']['loads'] for forward in loaded['forwards']
                       for key in ('parameter_devices', 'buffer_devices', 'input_devices')
                       for device in forward[key])):
            raise ValueError('Activation run does not bind its shared model, crops and CUDA reference')
        settings[run['method']] = {key: value for key, value in current.items()
                                   if key not in ('dataset', 'model_path')}
    return dict(original_hashes=copy.deepcopy(original), private_inputs_by_basename=by_name,
                crops=[{key: copy.deepcopy(row[key]) for key in ('name', 'sha256', 'identity')} for row in crops],
                database_bundle=database_hashes,
                crop_format=copy.deepcopy(inputs['crop_format']), model_classes=inputs['model_classes'],
                model_epochs=inputs['model_epochs'], settings=settings)


def compose(sources, destination, *, aliases=None, accept_verified_partial=False,
            recorder_delta_paths=(), recorder_delta_reason=None, source_scenes=None,
            scientific_lesson=None):
    """Write one verified capture folder from disjoint completed source runs."""
    aliases = aliases or {}
    source_scenes = source_scenes or {}
    if scientific_lesson not in (None, '16_activation'):
        raise ValueError('Scientific composition is explicit only for lesson 16')
    science_sources, cohort = [], None
    known_sources = {str(source.resolve()) for source in sources}
    if set(source_scenes) - known_sources:
        raise ValueError('Scene selection names an unknown capture source')
    allowed = set(recorder_delta_paths)
    if allowed and (not recorder_delta_reason or any(
            not name.startswith('tools/tutorials/capture') or not name.endswith('.py')
            or '..' in Path(name).parts for name in allowed)):
        raise ValueError('Recorder differences need explicit capture-tool paths and a reason')
    observed = set()
    if destination.exists():
        raise ValueError('Capture composition requires a new destination')
    frames, inputs, source_identity = {}, [], None
    pending = []
    for source in sources:
        provenance = read(source / 'provenance.json')
        partial = not provenance.get('completed_capture')
        if partial and not accept_verified_partial:
            raise ValueError(f'Incomplete capture: {source}')
        identity_path = source / 'exact-source.json'
        if not identity_path.exists():
            identity_path = source.parent.parent / 'exact-source.json'
        identity = read(identity_path)
        if source_identity is None:
            source_identity = identity['sha256']
        else:
            if set(identity['sha256']) != set(source_identity):
                raise ValueError('Capture source inventories differ')
            changed = {name for name, digest in identity['sha256'].items()
                       if source_identity[name] != digest}
            if changed - allowed:
                raise ValueError('Capture runs use different application or recorder sources')
            observed.update(changed)
        manifest = read(source / 'frames.json')
        selected = set(source_scenes.get(str(source.resolve()), manifest))
        if not selected or selected - set(manifest):
            raise ValueError('Scene selection must name existing source frames')
        if scientific_lesson is not None:
            path = source / 'scientific_acceptance.json'
            if path.is_symlink() or not path.is_file():
                raise ValueError('Every used activation capture needs its original scientific proof')
            proof = read(path)
            current = activation_cohort(proof)
            if cohort is not None and current != cohort:
                raise ValueError('Selected activation sources use different scientific cohorts/settings')
            cohort = current
            science_sources.append(dict(source=str(source.resolve()), path=str(path.resolve()),
                sha256=sha256(path), selected_scenes=sorted(selected), original_proof=copy.deepcopy(proof)))
        inputs.append({'source': str(source.resolve()),
                       'manifest_sha256': sha256(source / 'frames.json'),
                       'provenance_sha256': sha256(source / 'provenance.json'),
                       'exact_source_sha256': sha256(identity_path),
                       'source_files_sha256': identity['sha256'],
                       'source_run_completed': not partial,
                       'verified_completed_clips_only': partial,
                       'selected_scenes': sorted(selected),
                       'omitted_source_scenes': sorted(set(manifest) - selected)})
        for name, original in manifest.items():
            if name not in selected:
                continue
            target = aliases.get(name, name)
            frame = copy.deepcopy(original)
            if partial and not frame.get('clip'):
                raise ValueError('A failed run can contribute only individually verified completed clips')
            if target in frames:
                raise ValueError(f'Duplicate capture scene: {target}')
            files = [(frame['image'], frame['sha256'])]
            if frame.get('clip'):
                clip = frame['clip']
                files += [(clip['video'], clip['sha256']),
                          (clip['receipt'], clip['receipt_sha256'])]
                receipt = read(source / clip['receipt'])
                if (receipt.get('sha256') != clip['sha256']
                        or not receipt.get('full_decode_passed')
                        or receipt.get('audio_streams') != 0):
                    raise ValueError('Clip receipt does not verify actual video')
            for relative, expected in files:
                path = source / relative
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('Capture files must stay within their source folder')
                if sha256(path) != expected:
                    raise ValueError(f'Capture evidence changed: {path}')
                pending.append((path, relative, expected))
            frame['composition_source'] = {'capture': str(source.resolve()),
                                           'visual': name}
            frames[target] = frame
    unknown = set(aliases) - {frame['composition_source']['visual'] for frame in frames.values()}
    if unknown:
        raise ValueError(f'Alias source scenes are absent: {sorted(unknown)}')
    if allowed != observed:
        raise ValueError('Explicit recorder differences must match the actual source delta')
    destination.mkdir(parents=True)
    try:
        for source, relative, expected in pending:
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and sha256(output) != expected:
                raise ValueError(f'Conflicting capture filename: {relative}')
            shutil.copyfile(source, output)
        write(destination / 'exact-source.json',
              {'sha256': {name: digest for name, digest in source_identity.items()
                          if name not in observed}, 'composition': inputs,
               'recorder_delta_paths': sorted(observed),
               'recorder_delta_reason': recorder_delta_reason})
        write(destination / 'frames.json', frames)
        if scientific_lesson is not None:
            write(destination / 'scientific_acceptance.json', dict(accepted=True, lesson=scientific_lesson,
                kind='composed_activation_source_science', shared_cohort=cohort, sources=science_sources,
                scope='Bound original accepted scientific runs; no merged or newly executed CUDA run'))
        write(destination / 'provenance.json',
              {'completed_capture': True, 'composition': inputs,
               'scene_aliases': aliases, 'evidence_copied_without_edits': True,
               'recorder_delta_paths': sorted(observed),
               'recorder_delta_reason': recorder_delta_reason,
               'native_compositor_claim': False, 'native_mac_claim': False})
    except BaseException:
        shutil.rmtree(destination)
        raise
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--alias', action='append', default=[], metavar='SOURCE=DESTINATION')
    parser.add_argument('--accept-verified-partial', action='store_true',
                        help='Retain completed, hash-verified clips from a failed later scene; preserve the failed-run receipt')
    parser.add_argument('--recorder-delta-path', action='append', default=[])
    parser.add_argument('--recorder-delta-reason')
    parser.add_argument('--scientific-lesson', choices=['16_activation'],
                        help='Bind each used original CUDA proof and require one shared scientific cohort')
    parser.add_argument('--source-scenes', type=Path,
                        help='Explicit JSON map of absolute capture paths to selected scene names; originals remain unchanged')
    args = parser.parse_args()
    aliases = dict(pair.split('=', 1) for pair in args.alias)
    frames = compose(args.sources, args.destination, aliases=aliases,
                     accept_verified_partial=args.accept_verified_partial,
                     recorder_delta_paths=args.recorder_delta_path,
                     recorder_delta_reason=args.recorder_delta_reason,
                     source_scenes=read(args.source_scenes) if args.source_scenes else None,
                     scientific_lesson=args.scientific_lesson)
    print(f'Composed {len(frames)} verified scenes; original recordings unchanged.')


if __name__ == '__main__':
    main()
