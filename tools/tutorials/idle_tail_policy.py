"""Bind explicit safe idle tails to unchanged published media and scene clocks."""
from copy import deepcopy
import math
import json
from pathlib import Path
import re


def bind_policy(policy, records, lessons, load_timing=None):
    """Return catalog objects with only source-matched explicit idle-tail fences."""
    if not isinstance(policy, dict) or set(policy) != {'schema', 'lessons'} or policy['schema'] != 1 or not isinstance(policy['lessons'], list):
        raise ValueError('Invalid idle-tail policy schema')
    result = deepcopy(lessons)
    by_id = {lesson['id']: lesson for lesson in result}
    files = {record['path']: record for record in records}
    for lesson in result:
        lesson.pop('native_idle_tails', None)
    inherited = []
    for lesson in lessons:
        if 'native_idle_tails' in lesson:
            previous = deepcopy(lesson['native_idle_tails'])
            if previous.pop('schema', None) != 1:
                raise ValueError('Invalid inherited idle-tail policy schema')
            inherited.append(previous)
    requested = [entry.get('lesson') for entry in policy['lessons'] if isinstance(entry, dict)]
    if len(requested) != len(set(requested)):
        raise ValueError('Duplicate idle-tail policy lesson')
    entries = [entry for entry in inherited if entry['lesson'] not in requested] + policy['lessons']
    selected = set()
    for entry in entries:
        required = {'lesson', 'master_sha256', 'visual_timing_sha256', 'scenes'}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError('Invalid idle-tail lesson binding')
        identity = entry['lesson']
        if identity in selected or identity not in by_id:
            raise ValueError('Unknown or duplicate idle-tail lesson')
        selected.add(identity)
        lesson = by_id[identity]
        master = 'media_host/' + lesson.get('silent', '')
        timing = 'media_host/' + lesson.get('visual_timings', '')
        for path, field in [(master, 'master_sha256'), (timing, 'visual_timing_sha256')]:
            if (not isinstance(entry[field], str) or not re.fullmatch('[0-9a-f]{64}', entry[field])
                    or path not in files or files[path]['sha256'] != entry[field]):
                raise ValueError('Idle-tail policy source bytes changed: ' + path)
        # Thin Git checkpoints retain immutable file identities, not hosted JSON.
        # Their schema/hash binding is checked without claiming a fresh clock read.
        visual = load_timing(timing) if load_timing is not None else None
        if visual is not None and (visual.get('lesson') != identity or visual.get('kind') != 'native_visual_timing'
                or visual.get('native_live_video') is not True or not visual.get('coverage', {}).get('accepted')):
            raise ValueError('Idle-tail policy needs accepted native scene clocks')
        scenes = visual['scenes'] if visual is not None else None
        if not isinstance(entry['scenes'], list) or not entry['scenes']:
            raise ValueError('Idle-tail policy needs explicit scenes')
        used = set()
        for fence in entry['scenes']:
            if (not isinstance(fence, dict) or set(fence) != {'visual', 'source_start', 'source_end', 'idle_tail_start'}
                    or fence['visual'] in used):
                raise ValueError('Invalid or duplicate idle-tail scene')
            used.add(fence['visual'])
            values = [fence[key] for key in ('source_start', 'source_end', 'idle_tail_start')]
            if (not isinstance(fence['visual'], str) or not fence['visual']
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                    or fence['source_start'] < 0
                    or not 0 <= fence['idle_tail_start'] < fence['source_end'] - fence['source_start']):
                raise ValueError('Invalid idle-tail scene clock')
            if scenes is not None:
                found = [scene for scene in scenes if scene['visual'] == fence['visual']]
                if (len(found) != 1 or abs(fence['source_start'] - found[0]['speech_start']) > 1e-7
                        or abs(fence['source_end'] - found[0]['scene_end']) > 1e-7):
                    raise ValueError('Idle-tail policy scene clock changed')
        lesson['native_idle_tails'] = {'schema': 1, **deepcopy(entry)}
    return result


def validate_catalog_policy(records, lessons, load_timing=None):
    """Validate embedded product metadata without changing historical baselines."""
    expected = bind_policy({'schema': 1, 'lessons': []}, records, lessons, load_timing)
    if expected != lessons:
        raise ValueError('Idle-tail catalog binding differs from its lesson')


def load_policy(repo, path=None):
    """An explicit empty test/custom policy is different from a missing default."""
    source = Path(path) if path is not None else Path(repo) / 'tools/tutorials/idle_tail_policy.json'
    return json.loads(source.read_text(encoding='utf-8'))
