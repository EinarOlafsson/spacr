"""Project the redundant generated scene14 speech field onto published67.

This explicit producer operation never changes narration, audio, unrelated
catalog records, or the strict retained-narration comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pronunciation import spoken_form

IDENTITY = '67_experiment_design'
INDEX = 13
VISUAL = '12_real_export_complete'


def selected(catalog):
    matches = [row for row in catalog['lessons'] if row.get('id') == IDENTITY]
    if len(matches) != 1:
        raise ValueError('Expected exactly one Experiment Design lesson')
    return matches[0]


def project(catalog, canonical, published):
    result = deepcopy(catalog)
    lesson = selected(result)
    if (canonical.get('id') != IDENTITY or published.get('id') != IDENTITY
            or len(lesson['scenes']) != 14 or len(canonical['scenes']) != 14
            or len(published['scenes']) != 14):
        raise ValueError('Experiment Design identity or scene count changed')
    scene, authored, baseline = (item['scenes'][INDEX]
                                 for item in (lesson, canonical, published))
    if (any(row.get('visual') != VISUAL for row in (scene, authored, baseline))
            or scene.get('narration') != authored.get('narration')
            or scene.get('narration') != baseline.get('narration')):
        raise ValueError('Experiment Design scene14 identity or narration changed')
    if 'speech_text' not in authored and 'speech_text' in scene:
        if scene['speech_text'] != spoken_form(authored['narration'], 'en'):
            raise ValueError('Scene14 speech is not the redundant generated form')
        # Remove only this un-authored generated property in the private copy.
        del scene['speech_text']
    elif 'speech_text' in authored and scene.get('speech_text') != authored['speech_text']:
        raise ValueError('Explicit authored speech override changed')
    if lesson != published:
        raise ValueError('Projected Experiment Design differs from published lesson')
    return result


def pinned_json(path, expected):
    if path.is_symlink():
        raise ValueError('Expected a regular catalog source')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Catalog source hash changed')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'canonical', 'published-catalog'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    current = pinned_json(args.catalog, args.catalog_sha256)
    canonical = pinned_json(args.canonical, args.canonical_sha256)
    baseline = selected(pinned_json(args.published_catalog, args.published_catalog_sha256))
    projected = project(current, canonical, baseline)
    # Fresh output only. Root can review/integrate after the frozen batch ends.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(projected, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
