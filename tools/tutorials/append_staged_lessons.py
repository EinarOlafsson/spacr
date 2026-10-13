#!/usr/bin/env python3
"""Selectively append source catalogs for lessons 74/75; never publish media.

The CLI only reports a plan unless --write is explicitly supplied. Existing
JSON and JavaScript lesson objects are preserved independently, including
historical differences between them. Navigation is built with the existing
local build_navigation.build; no publisher, upload or media tool is invoked.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping, Sequence


CATALOGS = tuple('lessons_' + lang + '.json' for lang in
                 ('en', 'es', 'fr', 'hi', 'it', 'ja', 'pt-BR', 'zh-CN')) + tuple(
    'captions_' + lang + '.json' for lang in ('da', 'de', 'is', 'ko', 'nb', 'sv'))
ALLOWED_IDS = ('74_import_images', '75_regression_diagnostics')
JS_PREFIX = '"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze('
JS_SUFFIX = ');\n'
NAV_PREFIX = '"use strict";\nwindow.SPACR_TUTORIAL_NAVIGATION = Object.freeze('
TEXT_FIELDS = frozenset(('title', 'description', 'objectives', 'prerequisite'))
ROUTE_FIELDS = ('number', 'slug', 'app_key', 'host_app_key')


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f'Duplicate JSON member: {key}')
        result[key] = value
    return result


def _loads(text: str):
    return json.loads(text, object_pairs_hook=_object_pairs,
                      parse_constant=lambda value: _fail(f'Nonfinite JSON value: {value}'))


def parse_javascript(text: str) -> dict:
    """Parse the current data-only wrapper, never execute JavaScript."""
    if not text.startswith(JS_PREFIX) or not text.endswith(JS_SUFFIX):
        _fail('Unexpected lesson_catalog.js wrapper; refusing to rewrite it')
    return _loads(text[len(JS_PREFIX):-len(JS_SUFFIX)])


def _lessons(catalog: dict, context: str, *, sparse: bool = False) -> list:
    if not isinstance(catalog, dict) or not isinstance(catalog.get('lessons'), list):
        _fail(f'{context}: missing lessons list')
    lessons = catalog['lessons']
    identities, routes, slugs = set(), set(), set()
    for position, lesson in enumerate(lessons, 1):
        if not isinstance(lesson, dict):
            _fail(f'{context}: lesson is not an object')
        identity = lesson.get('id')
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity):
            _fail(f'{context}: invalid lesson identity {identity!r}')
        if identity in identities:
            _fail(f'{context}: duplicate lesson ID {identity}')
        identities.add(identity)
        if not sparse or 'number' in lesson:
            if type(lesson.get('number')) is not int or lesson['number'] != position:
                _fail(f'{context}/{identity}: numbers must be contiguous in catalog order')
        for field, seen in (('app_key', routes), ('slug', slugs)):
            value = lesson.get(field)
            if value is None and (field == 'app_key' or sparse):
                continue
            if not isinstance(value, str) or not value:
                _fail(f'{context}/{identity}: invalid {field}')
            if value in seen:
                _fail(f'{context}: duplicate {field} route {value}')
            seen.add(value)
        scenes = lesson.get('scenes')
        if not isinstance(scenes, list) or not scenes:
            _fail(f'{context}/{identity}: missing or empty scenes')
        for scene in scenes:
            if (not isinstance(scene, dict) or not isinstance(scene.get('narration'), str)
                    or not scene['narration'].strip()):
                _fail(f'{context}/{identity}: missing scene narration')
            if 'speech_text' in scene and (not isinstance(scene['speech_text'], str)
                                           or not scene['speech_text'].strip()):
                _fail(f'{context}/{identity}: invalid speech_text')
    return lessons


def _ids(lessons: list) -> list[str]:
    return [lesson['id'] for lesson in lessons]


def _shape(lesson: dict) -> dict:
    """All nontranslated fields, including ordered scene/focus/link structure."""
    shape = {key: value for key, value in lesson.items() if key not in TEXT_FIELDS}
    shape['scenes'] = [{key: value for key, value in scene.items()
                        if key not in ('narration', 'speech_text')}
                       for scene in lesson['scenes']]
    return shape


def _new_translation(lesson: dict, english: dict, context: str) -> None:
    for key in ('title', 'description', 'prerequisite'):
        if not isinstance(lesson.get(key), str) or not lesson[key].strip():
            _fail(f'{context}: missing translated {key}')
    objectives = lesson.get('objectives')
    if (not isinstance(objectives, list) or len(objectives) != len(english.get('objectives', []))
            or not objectives or any(not isinstance(v, str) or not v.strip() for v in objectives)):
        _fail(f'{context}: missing or mismatched objectives')
    if _shape(lesson) != _shape(english):
        _fail(f'{context}: new lesson structure/scenes/order differs from English')
    if any(not isinstance(scene.get('visual'), str) or not scene['visual']
           for scene in lesson['scenes']):
        _fail(f'{context}: missing scene visual')


def _align(lessons: list, english: list, context: str, *, sparse=False,
           compare_scene_counts=True) -> None:
    if _ids(lessons) != _ids(english):
        _fail(f'{context}: lesson identities/order differs from English')
    for lesson, reference in zip(lessons, english):
        for key in ROUTE_FIELDS:
            if sparse and key not in lesson:
                continue
            if lesson.get(key) != reference.get(key):
                _fail(f'{context}/{lesson["id"]}: {key} route differs from English')
        if compare_scene_counts and len(lesson['scenes']) != len(reference['scenes']):
            _fail(f'{context}/{lesson["id"]}: scene count differs from English')


def merge_catalogs(existing: Mapping[str, dict], staged: Mapping[str, dict],
                   lesson_ids: Sequence[str], javascript_catalog: dict) -> dict:
    """Pure merge: return independent copies; never mutate any supplied object."""
    selected = list(lesson_ids)
    if (not selected or len(selected) != len(set(selected))
            or any(identity not in ALLOWED_IDS for identity in selected)):
        _fail('Explicit, unique lesson IDs from 74_import_images/75_regression_diagnostics required')
    for name in CATALOGS:
        if name not in existing or name not in staged:
            _fail(f'Missing locale catalog: {name}')
    original_en = _lessons(existing['lessons_en.json'], 'existing English')
    staged_en = _lessons(staged['lessons_en.json'], 'staged English')
    original_ids = _ids(original_en)
    if set(selected) & set(original_ids):
        _fail('Refusing replacement of an existing lesson ID')
    if _ids(staged_en)[:len(original_ids)] != original_ids:
        _fail('Staged English does not retain original lesson identities/order')
    staged_index = {lesson['id']: lesson for lesson in staged_en}
    if any(identity not in staged_index for identity in selected):
        _fail('Selected lesson missing from staged English')
    ordered_selected = [identity for identity in _ids(staged_en) if identity in selected]
    if selected != ordered_selected:
        _fail('Selected lesson order must match staged English')
    additions = [staged_index[identity] for identity in selected]
    expected_numbers = list(range(len(original_en) + 1, len(original_en) + len(selected) + 1))
    if [lesson['number'] for lesson in additions] != expected_numbers:
        _fail('Appended lesson numbers must immediately continue the existing catalog')
    merged = {}
    for name in CATALOGS:
        sparse = name.startswith('captions_')
        before = _lessons(existing[name], f'existing {name}', sparse=sparse)
        source = _lessons(staged[name], f'staged {name}', sparse=sparse)
        _align(before, original_en, f'existing {name}', sparse=sparse)
        # Unselected staged narration/scenes may be refreshed. They are ignored.
        _align(source, staged_en, f'staged {name}', sparse=sparse,
               compare_scene_counts=False)
        by_id = {lesson['id']: lesson for lesson in source}
        for reference in additions:
            _new_translation(by_id[reference['id']], reference, f'{name}/{reference["id"]}')
        result = copy.deepcopy(existing[name])
        result['lessons'].extend(copy.deepcopy(by_id[identity]) for identity in selected)
        _lessons(result, f'merged {name}', sparse=sparse)
        merged[name] = result
    # Preserve the original JS objects, not English JSON copies. Historical
    # scene/content discrepancies are reported and intentionally not repaired.
    js_lessons = _lessons(javascript_catalog, 'existing JavaScript')
    _align(js_lessons, original_en, 'existing JavaScript', compare_scene_counts=False)
    merged_js = copy.deepcopy(javascript_catalog)
    for lesson in additions:
        value = copy.deepcopy(lesson)
        value['poster'] = f'{value["id"]}/poster.jpg'
        value['silent'] = f'{value["id"]}/video/{value["id"]}_silent.mp4'
        merged_js['lessons'].append(value)
    _lessons(merged_js, 'merged JavaScript')
    count = lambda lessons: sum(len(lesson['scenes']) for lesson in lessons)
    return {'catalogs': merged, 'javascript_catalog': merged_js, 'report': {
        'selected_lesson_ids': selected, 'catalog_count': len(CATALOGS),
        'existing_lesson_count': len(original_en), 'merged_lesson_count': len(original_en) + len(selected),
        'existing_json_scene_count': count(original_en),
        'existing_javascript_scene_count': count(js_lessons),
        'appended_scene_count': count(additions),
        'merged_json_scene_count': count(merged['lessons_en.json']['lessons']),
        'merged_javascript_scene_count': count(merged_js['lessons']),
        'preserved_javascript_scene_count_differences': [
            {'lesson_id': a['id'], 'json': len(a['scenes']), 'javascript': len(b['scenes'])}
            for a, b in zip(original_en, js_lessons) if len(a['scenes']) != len(b['scenes'])],
        'existing_lesson_objects_and_catalog_metadata_preserved': True,
        'unselected_staged_lesson_objects_copied': False,
        'media_read_or_written': False, 'published': False}}


@dataclass(frozen=True)
class AppendPlan:
    root: Path
    outputs: dict[str, str]
    input_sha256: dict[Path, str]
    report: dict


def _local_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        _fail(f'Refusing symlink/out-of-root catalog path: {path}')
    if not path.is_file():
        _fail(f'Missing catalog input: {path}')
    return path


def plan_append(root: Path, stage: Path, lesson_ids: Sequence[str], *,
                navigation_builder: Callable | None = None) -> AppendPlan:
    """Read and validate all inputs, build navigation, and return unwritten text."""
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if root == stage:
        _fail('Stage and destination root must be different')
    fingerprints = {}
    def read(folder, relative):
        path = _local_file(folder, relative)
        raw = path.read_bytes()
        fingerprints[path] = _sha(raw)
        return raw.decode('utf-8')
    existing = {name: _loads(read(root, 'catalog/' + name)) for name in CATALOGS}
    staged = {name: _loads(read(stage, 'catalog/' + name)) for name in CATALOGS}
    javascript = parse_javascript(read(root, 'lesson_catalog.js'))
    # The existing navigation is an overwrite target too; pin it before planning.
    read(root, 'module_navigation.js')
    result = merge_catalogs(existing, staged, lesson_ids, javascript)
    if navigation_builder is None:
        from build_navigation import build
        navigation_builder = build
    navigation = navigation_builder(copy.deepcopy(result['catalogs']['lessons_en.json']))
    merged_ids = _ids(result['catalogs']['lessons_en.json']['lessons'])
    if navigation.get('preserved_lesson_ids') != merged_ids:
        _fail('Navigation does not preserve all merged lesson identities/order')
    missing = navigation.get('missing_tutorials')
    if not isinstance(missing, list) or [row.get('app_key') for row in missing] != ['ops']:
        _fail('Navigation must report only ops as missing; inspect unexpected routes')
    for lesson in result['catalogs']['lessons_en.json']['lessons']:
        if lesson['id'] in lesson_ids:
            route = navigation.get('routes', {}).get(lesson['id'], {})
            if any(route.get(key) != lesson.get(key) for key in ('app_key', 'host_app_key')):
                _fail(f'Navigation route differs for appended lesson {lesson["id"]}')
    outputs = {'catalog/' + name: json.dumps(catalog, ensure_ascii=False, indent=2) + '\n'
               for name, catalog in result['catalogs'].items()}
    outputs['lesson_catalog.js'] = JS_PREFIX + json.dumps(
        result['javascript_catalog'], ensure_ascii=False, separators=(',', ':')) + JS_SUFFIX
    outputs['module_navigation.js'] = NAV_PREFIX + json.dumps(
        navigation, ensure_ascii=False, indent=2) + JS_SUFFIX
    report = dict(result['report'], missing_tutorials=missing,
                  planned_files=list(outputs), written=False)
    return AppendPlan(root, outputs, fingerprints, report)


def write_plan(plan: AppendPlan) -> dict:
    """Apply only this validated local plan; refuse changed inputs before writes."""
    allowed = {'catalog/' + name for name in CATALOGS} | {'lesson_catalog.js', 'module_navigation.js'}
    if set(plan.outputs) != allowed:
        _fail('Unexpected output paths in append plan')
    for path, expected in plan.input_sha256.items():
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != expected:
            _fail(f'Input changed since planning; nothing written: {path}')
    targets = {relative: _local_file(plan.root, relative) for relative in plan.outputs}
    # Every parse/merge/navigation/staleness guard has passed before any write.
    # Each file replacement is atomic; the sixteen-file set is not a transaction.
    for relative, content in plan.outputs.items():
        target = targets[relative]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                             prefix='.' + target.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.chmod(target.stat().st_mode & 0o777)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return dict(plan.report, written=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Existing tutorial web root (contains catalog/ and both JavaScript files)')
    parser.add_argument('--stage', type=Path, required=True, help='Stage root containing catalog/')
    parser.add_argument('--lesson-ids', nargs='+', required=True, choices=ALLOWED_IDS)
    parser.add_argument('--write', action='store_true', help='Explicitly apply the validated local plan')
    args = parser.parse_args(argv)
    try:
        plan = plan_append(args.root, args.stage, args.lesson_ids)
        report = write_plan(plan) if args.write else plan.report
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Append refused: {exc}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
