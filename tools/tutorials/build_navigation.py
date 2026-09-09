#!/usr/bin/env python3
"""Derive tutorial navigation from the same hierarchy the spaCR GUI draws.

Only navigation metadata is generated: this never rewrites a lesson's scenes,
translated narration, identifiers, media or timings merely because it moved.
Missing lessons are reported explicitly, not replaced with invented videos.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
KEY_ALIASES = {'cellpose_all': 'cellpose_masks'}
HOSTED_MODES = {'classify': 'classify_merged', 'ml_analyze': 'classify_merged',
                'parameter_sweep': 'regression'}
LABELS = {
    'en': ['Main modules', 'Submodules', 'Getting started', 'Help and utilities'],
    'es': ['Módulos principales', 'Submódulos', 'Primeros pasos', 'Ayuda y utilidades'],
    'fr': ['Modules principaux', 'Sous-modules', 'Premiers pas', 'Aide et utilitaires'],
    'hi': ['मुख्य मॉड्यूल', 'उपमॉड्यूल', 'शुरुआत करें', 'सहायता और उपयोगिताएँ'],
    'it': ['Moduli principali', 'Sottomoduli', 'Primi passi', 'Guida e utilità'],
    'pt-BR': ['Módulos principais', 'Submódulos', 'Primeiros passos', 'Ajuda e utilitários'],
    'ja': ['メインモジュール', 'サブモジュール', 'はじめに', 'ヘルプとユーティリティ'],
    'zh-CN': ['主模块', '子模块', '入门', '帮助与实用工具'],
    'da': ['Hovedmoduler', 'Undermoduler', 'Kom godt i gang', 'Hjælp og værktøjer'],
    'de': ['Hauptmodule', 'Untermodule', 'Erste Schritte', 'Hilfe und Werkzeuge'],
    'is': ['Aðaleiningar', 'Undireiningar', 'Fyrstu skrefin', 'Hjálp og verkfæri'],
    'ko': ['주요 모듈', '하위 모듈', '시작하기', '도움말 및 유틸리티'],
    'nb': ['Hovedmoduler', 'Undermoduler', 'Kom i gang', 'Hjelp og verktøy'],
    'sv': ['Huvudmoduler', 'Undermoduler', 'Kom igång', 'Hjälp och verktyg'],
}


def build(catalog: dict) -> dict:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    sys.path.insert(0, str(REPO))
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import app
    from spacr.qt.widgets.fold_strip import folded_modules

    lessons = catalog['lessons']
    by_key = {}
    for lesson in lessons:
        if lesson.get('app_key'):
            key = lesson['app_key']
            if key in by_key:
                raise ValueError(f'Duplicate tutorial route {key}')
            by_key[key] = lesson
    tiles = app.tiled_apps()
    tile_keys = {row[0] for row in tiles}
    names = {row[0]: row[1] for row in app.APPS}
    names.update({k: v[0] for k, v in folded_modules().items()})
    folds = app.folded_children()
    parents = {}
    for host, children in folds.items():
        for child in children:
            parents[KEY_ALIASES.get(child, child)] = host
    parents.update(HOSTED_MODES)
    intro = {'id': 'getting-started', 'kind': 'intro', 'label_index': 2,
             'lessons': [l['id'] for l in lessons if not l.get('app_key')]}
    groups = []
    for section, rows in app.home_bands(tiles):
        groups.append({'id': section.lower(), 'title': section,
                       'kind': 'main', 'module_keys': [r[0] for r in rows],
                       'lessons': [by_key[r[0]]['id'] for r in rows
                                   if r[0] in by_key]})
    subgroups = []
    routes = {}
    for key, lesson in by_key.items():
        if key in tile_keys:
            routes[lesson['id']] = {'kind': 'main', 'app_key': key}
        elif key in parents:
            host = parents[key]
            routes[lesson['id']] = {'kind': 'submodule', 'app_key': key,
                                    'host_app_key': host,
                                    'host_lesson': by_key.get(host, {}).get('id'),
                                    'host_title': names.get(host, host)}
        elif key in app.TILELESS_APPS:
            routes[lesson['id']] = {'kind': 'help', 'app_key': key}
        else:
            raise ValueError(f'Lesson has no current GUI location: {key}')
    hosts = list(dict.fromkeys([r[0] for r in tiles] + list(folds)))
    for host in hosts:
        members = [l['id'] for l in lessons if parents.get(l.get('app_key')) == host]
        if members:
            subgroups.append({'id': host, 'kind': 'host',
                              'title': names.get(host, host),
                              'host_lesson': by_key.get(host, {}).get('id'),
                              'help_host': host not in tile_keys,
                              'lessons': members})
    utilities = [l['id'] for l in lessons if routes.get(l['id'], {}).get('kind') == 'help']
    if utilities:
        subgroups.append({'id': 'help', 'kind': 'help', 'label_index': 3,
                          'lessons': utilities})
    sections = [{'id': 'main', 'label_index': 0, 'groups': groups},
                {'id': 'submodules', 'label_index': 1, 'groups': subgroups}]
    assigned = intro['lessons'] + [identity for section in sections for group in section['groups']
                                  for identity in group['lessons']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {l['id'] for l in lessons}:
        raise ValueError('Navigation must place each existing lesson exactly once')
    expected = set(names) | set(HOSTED_MODES)
    expected = {KEY_ALIASES.get(k, k) for k in expected}
    uncovered = [{'app_key': k, 'title': names.get(k, k), 'host_app_key': parents.get(k),
                  'status': 'deferred_unvalidated_workflow' if k == 'ops' else 'needs_tutorial'}
                 for k in sorted(expected - set(by_key))]
    return {'schema': 1, 'source_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'labels': LABELS, 'intro': intro, 'sections': sections, 'routes': routes,
            'missing_tutorials': uncovered,
            'preserved_lesson_ids': [l['id'] for l in lessons]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=WORKSPACE / 'catalog/lessons_en.json')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'web/module_navigation.js')
    args = parser.parse_args()
    result = build(json.loads(args.catalog.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('"use strict";\nwindow.SPACR_TUTORIAL_NAVIGATION = Object.freeze('
                          + json.dumps(result, ensure_ascii=False, indent=2) + ');\n',
                          encoding='utf-8')
    print(f"Placed {len(result['preserved_lesson_ids'])} existing lessons without changing media.")
    print('Missing lessons:', ', '.join(x['app_key'] for x in result['missing_tutorials']))
