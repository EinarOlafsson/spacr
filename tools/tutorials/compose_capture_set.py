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
from pathlib import Path
import shutil
from stage_lesson import read, write


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(sources, destination, *, aliases=None, accept_verified_partial=False,
            recorder_delta_paths=(), recorder_delta_reason=None, source_scenes=None):
    """Write one verified capture folder from disjoint completed source runs."""
    aliases = aliases or {}
    source_scenes = source_scenes or {}
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
    parser.add_argument('--source-scenes', type=Path,
                        help='Explicit JSON map of absolute capture paths to selected scene names; originals remain unchanged')
    args = parser.parse_args()
    aliases = dict(pair.split('=', 1) for pair in args.alias)
    frames = compose(args.sources, args.destination, aliases=aliases,
                     accept_verified_partial=args.accept_verified_partial,
                     recorder_delta_paths=args.recorder_delta_path,
                     recorder_delta_reason=args.recorder_delta_reason,
                     source_scenes=read(args.source_scenes) if args.source_scenes else None)
    print(f'Composed {len(frames)} verified scenes; original recordings unchanged.')


if __name__ == '__main__':
    main()
