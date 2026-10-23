"""Stage exact published narration for an editorial visual change, without TTS.

The explicit record selects immutable published-source validation. It never
claims current synthesis-runtime freshness, a new translation or listening review.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from check_completed_matrix import voice_matrix
from retain_narration import catalog_lesson, digest, require_timing, require_unchanged_lesson
from stage_lesson import REPO, read, write

RECORD = 'published-retained-narration.json'
CATALOG_COMPOSITION = 'published-retained-catalog-composition.json'


def published_sources(baseline, identity, *, stage=None, catalog_composition=None):
    """Require the exact published receipt, catalogs, fifty tracks and native proofs."""
    from audit_staged_catalogs import CATALOGS
    from published_native_reuse import _published_inputs, interval_spec

    if Path(identity).name != identity or identity in {'.', '..'}:
        raise ValueError('Expected one published lesson identity')
    baseline, native, bindings, source = _published_inputs(baseline, identity)
    inventory = voice_matrix(REPO / 'tools/tutorials/authoring/tools/render_all_voices.py')
    expected = {(language, voice) for language, voices in inventory.items() for voice in voices}
    manifest = read(baseline / 'release-manifest.json')
    records = {row['path']: row for row in manifest['files']}
    files = []
    if catalog_composition is not None:
        _require_composition(stage, identity, catalog_composition, CATALOGS)

    def verified(relative, destination, *, exact_stage=True):
        path = baseline / relative
        row = records.get(relative)
        if (not row or path.is_symlink() or not path.resolve().is_relative_to(baseline)
                or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']):
            raise ValueError('Published narration source differs from manifest: ' + relative)
        if stage is not None and exact_stage:
            target = (_stage_file(stage, destination) if catalog_composition is not None
                      else Path(stage) / destination)
            if (target.is_symlink() or not target.resolve().is_relative_to(Path(stage).resolve())
                    or target.stat().st_size != row['bytes'] or digest(target) != row['sha256']):
                raise ValueError('Staged retained narration differs from published bytes: ' + destination)
        files.append((path, destination))
        return row

    english = read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
    interval_spec(english, native)
    catalogs = []
    localized = {}
    for name in CATALOGS:
        row = verified('web/catalog/' + name, 'catalog/' + name,
                       exact_stage=catalog_composition is None)
        original = catalog_lesson(baseline / 'web/catalog' / name, identity)
        if catalog_composition is not None:
            _require_composed_catalog(stage, identity, catalog_composition, name, original, row)
        current = catalog_lesson(REPO / 'tools/tutorials/authoring/catalog' / name, identity)
        require_unchanged_lesson(current, original, original, voice_inventory=inventory)
        if name == 'lessons_en.json':
            require_unchanged_lesson(english, original, original, voice_inventory=inventory)
        if name.startswith('lessons_'):
            localized[name[len('lessons_'):-len('.json')]] = original
        catalogs.append(dict(path=name, sha256=row['sha256'], bytes=row['bytes']))
    proofs = native.get('source_tracks', [])
    if (not isinstance(proofs, list) or len(proofs) != 50
            or any(not isinstance(row, dict) for row in proofs)):
        raise ValueError('Expected all fifty published native narration proofs')
    by_voice = {(row.get('language'), row.get('voice')): row for row in proofs}
    if len(by_voice) != 50 or set(by_voice) != expected:
        raise ValueError('Published native narration proof inventory differs')
    prefix = f'media_host/{identity}/audio/'
    audio_paths = {name for name in records if name.startswith(prefix)}
    exact_paths = {prefix + f'{language}/{voice}{suffix}'
                   for language, voice in expected for suffix in ('.m4a', '.json')}
    if audio_paths != exact_paths:
        raise ValueError('Published audio/timing manifest inventory differs')
    tracks, checked = [], []
    for language, voice in sorted(expected):
        relative = prefix + f'{language}/{voice}'
        destination = f'production/{identity}/audio/{language}/{voice}'
        audio = verified(relative + '.m4a', destination + '.m4a')
        metadata = verified(relative + '.json', destination + '.json')
        timing = read(baseline / (relative + '.json'))
        require_timing(timing, localized[language], language, voice, audio['sha256'])
        inputs = timing['render_inputs']
        fingerprint = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True,
                                                separators=(',', ':')).encode()).hexdigest()
        speed = inputs.get('speed')
        proof = dict(language=language, voice=voice, audio_sha256=audio['sha256'],
                     timing_sha256=metadata['sha256'], render_fingerprint=fingerprint)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
                or not math.isfinite(speed) or speed <= 0
                or timing.get('speed') != speed or timing.get('media_bytes') != audio['bytes']
                or timing.get('render_fingerprint') != fingerprint or by_voice[(language, voice)] != proof
                or len(timing.get('scenes', [])) != len(english['scenes'])
                or [row.get('scene') for row in timing['scenes']] != list(range(1, len(english['scenes']) + 1))
                or [row.get('text') for row in timing['scenes']] != [row['narration'] for row in localized[language]['scenes']]):
            raise ValueError('Published native proof, fingerprint or scene source differs')
        tracks.append(timing)
        checked.append(proof)
    if stage is not None:
        root = Path(stage) / 'production' / identity
        if read(root / 'lesson.en.json') != english:
            raise ValueError('Staged canonical English changed')
        actual = {path.relative_to(root / 'audio').as_posix()
                  for path in (root / 'audio').rglob('*') if path.is_file()}
        if actual != {f'{language}/{voice}{suffix}' for language, voice in expected
                      for suffix in ('.m4a', '.json')}:
            raise ValueError('Staged retained narration inventory differs')
    record = dict(schema=1, kind='published_retained_narration', lesson=identity, **source,
                  published_master=bindings['master'], published_timing=bindings['timing'],
                  catalogs=catalogs, tracks=checked, resynthesized=False, fresh_capture=False,
                  current_runtime_freshness_claimed=False, new_translation_review_claimed=False,
                  human_listening_review=False)
    return record, tracks, checked, files, english


def checked_published_narration(stage, identity):
    """Explicit retention is valid only while every immutable source still agrees."""
    if Path(identity).name != identity or identity in {'.', '..'}:
        raise ValueError('Expected one published lesson identity')
    if Path(stage).is_symlink():
        raise ValueError('Retained stage must not be a symlink')
    stage = Path(stage).resolve()
    path = stage / 'production' / identity / RECORD
    if path.is_symlink():
        raise ValueError('Retained narration record must be an independent file')
    record = read(path)
    if (record.get('schema') != 1 or record.get('kind') != 'published_retained_narration'
            or record.get('lesson') != identity):
        raise ValueError('Expected explicit published retained narration provenance')
    sidecar = stage / 'production' / identity / CATALOG_COMPOSITION
    composition = read(_stage_file(stage, sidecar.relative_to(stage))) if os.path.lexists(sidecar) else None
    expected, tracks, proofs, _, _ = published_sources(record['baseline_root'], identity, stage=stage,
                                                       catalog_composition=composition)
    if record != expected:
        raise ValueError('Published retained narration bindings changed')
    return tracks, proofs


def _stage_file(stage, relative):
    """Resolve a required regular file without accepting any symlink component."""
    relative = Path(relative)
    stage = Path(stage)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Invalid composed catalog path')
    path = stage
    if path.is_symlink():
        raise ValueError('Composed stage must not be a symlink')
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Composed catalog path must not use symlinks')
    if not path.is_file() or not path.resolve().is_relative_to(stage.resolve()):
        raise ValueError('Missing composed catalog file')
    return path


def _object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _unique_lesson(path, identity):
    value = read(path)
    lessons = value.get('lessons') if isinstance(value, dict) else None
    if (not isinstance(lessons, list) or any(not isinstance(row, dict)
            or not isinstance(row.get('id'), str) for row in lessons)):
        raise ValueError('Invalid composed lesson catalog')
    ids = [row['id'] for row in lessons]
    if any(not name or Path(name).name != name or name in {'.', '..'} for name in ids):
        raise ValueError('Invalid composed catalog identity')
    if len(ids) != len(set(ids)) or ids.count(identity) != 1:
        raise ValueError('Duplicate or missing composed catalog identity')
    return lessons[ids.index(identity)]


def _require_composition(stage, identity, composition, names):
    required = {'schema', 'kind', 'lesson', 'retained_record_sha256', 'catalogs'}
    if (stage is None or not isinstance(composition, dict) or set(composition) != required
            or type(composition['schema']) is not int or composition['schema'] != 1
            or composition['kind'] != 'published_retained_catalog_composition'
            or composition['lesson'] != identity):
        raise ValueError('Invalid explicit retained catalog composition')
    record = _stage_file(stage, Path('production') / identity / RECORD)
    if digest(record) != composition['retained_record_sha256']:
        raise ValueError('Original retained record changed')
    rows = composition['catalogs']
    keys = {'path', 'sha256', 'bytes', 'source_sha256', 'selected_lesson_sha256'}
    if (not isinstance(rows, list) or len(rows) != len(names)
            or any(not isinstance(row, dict) or set(row) != keys for row in rows)
            or [row['path'] for row in rows] != list(names)):
        raise ValueError('Composed catalog inventory changed')


def _require_composed_catalog(stage, identity, composition, name, original, source):
    row = next(row for row in composition['catalogs'] if row['path'] == name)
    path = _stage_file(stage, Path('catalog') / name)
    selected = _unique_lesson(path, identity)
    if (type(row['bytes']) is not int or path.stat().st_size != row['bytes']
            or digest(path) != row['sha256'] or row['source_sha256'] != source['sha256']
            or selected != original or _object_digest(selected) != row['selected_lesson_sha256']):
        raise ValueError('Composed catalog bytes or retained selected lesson changed')


def bind_composed_catalogs(stage, identity):
    """Explicitly bind merged catalogs after assembly, without changing source proof.

    This validates the retained lesson only. Other merged lessons require their own
    normal fresh/retained validators; this is not an acceptance claim for them.
    """
    from audit_staged_catalogs import CATALOGS
    if Path(identity).name != identity or identity in {'.', '..'}:
        raise ValueError('Expected one published lesson identity')
    stage = Path(stage)
    record_path = _stage_file(stage, Path('production') / identity / RECORD)
    destination = record_path.with_name(CATALOG_COMPOSITION)
    if os.path.lexists(destination):
        raise FileExistsError('Preserve existing catalog composition')
    record = read(record_path)
    if record.get('schema') != 1 or record.get('kind') != 'published_retained_narration' or record.get('lesson') != identity:
        raise ValueError('Expected original schema1 retained record')
    expected, _, _, _, _ = published_sources(record['baseline_root'], identity)
    if record != expected:
        raise ValueError('Original published retention proof changed')
    rows = []
    for source in expected['catalogs']:
        name = source['path']
        path = _stage_file(stage, Path('catalog') / name)
        original = _unique_lesson(Path(record['baseline_root']) / 'web/catalog' / name, identity)
        selected = _unique_lesson(path, identity)
        if selected != original:
            raise ValueError('Retained selected lesson changed during catalog composition')
        rows.append(dict(path=name, sha256=digest(path), bytes=path.stat().st_size,
                         source_sha256=source['sha256'], selected_lesson_sha256=_object_digest(selected)))
    composition = dict(schema=1, kind='published_retained_catalog_composition', lesson=identity,
                       retained_record_sha256=digest(record_path), catalogs=rows)
    # Full stage/audio validation happens BEFORE creating any sidecar.
    published_sources(record['baseline_root'], identity, stage=stage, catalog_composition=composition)
    with destination.open('x') as output:
        output.write(json.dumps(composition, ensure_ascii=False, indent=2) + '\n')
    return destination


def prepare(stage, identity, baseline):
    """Copy only verified published catalogs/audio into a wholly fresh private stage."""
    if os.path.lexists(stage):
        raise FileExistsError('Retained narration requires a fresh independent stage')
    stage, baseline = Path(stage).resolve(), Path(baseline).resolve()
    if (stage == baseline or stage.is_relative_to(baseline)
            or baseline.is_relative_to(stage)):
        raise FileExistsError('Retained narration requires a fresh independent stage')
    record, _, _, files, english = published_sources(baseline, identity)
    stage.mkdir(parents=True)
    for source, relative in files:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output)
    root = stage / 'production' / identity
    write(root / 'lesson.en.json', deepcopy(english))
    write(root / RECORD, record)
    checked_published_narration(stage, identity)
    return root / RECORD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.stage, args.lesson, args.baseline))


if __name__ == '__main__':
    main()
