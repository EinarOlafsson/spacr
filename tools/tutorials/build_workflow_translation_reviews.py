"""Compose source-bound tutorial reviews from reviewed workflow-map phrases.

Module names, API symbols, table names and column names remain the identifiers
shown in the English recording. A missing or stale phrase is an error, never
an English fallback. Composition does not stage audio or publish a lesson.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from string import Formatter
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import build_module_workflows as workflow

LESSONS = ('79_module_inputs_outputs', '80_image_analysis_pathways')
LANGUAGES = {'da', 'de', 'es', 'fr', 'hi', 'is', 'it', 'ja', 'ko', 'nb', 'pt-BR', 'sv', 'zh-CN'}
TOKENS = ('None', 'max_workers', 'png_list', 'measurements.db', 'FASTQ',
          'CSV', 'OPS', 'CV', 'ML', 'TIFF', 'PNG', 'FEATURES', 'log2')
FILE_FORMATS = {'FASTQ', 'CSV', 'TIFF', 'PNG'}


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_hash(value):
    return text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False))


def required_sources(data, lessons=LESSONS):
    """Collect exactly the phrases the shared lesson generator consumes."""
    found = set()

    def collect(source):
        found.add(source)
        return source

    for lesson in lessons:
        workflow.lesson_document(data, lesson, translate=collect)
    return found


def placeholders(text):
    return sorted(field for _, field, _, _ in Formatter().parse(text) if field is not None)


def reviewed_phrases(bank, data, map_sha256, lessons=LESSONS):
    """Require current sources, intact placeholders and literal data identifiers."""
    if bank.get('language') not in LANGUAGES:
        raise ValueError('Unsupported workflow translation language')
    if bank.get('workflow_map_sha256') != map_sha256:
        raise ValueError('Workflow map changed after phrase review')
    required = required_sources(data, lessons)
    translated = {}
    for record in bank['records']:
        source, target = record['source'], record['translation']
        if record.get('source_sha256') != text_hash(source):
            raise ValueError('Stale phrase source hash')
        if not isinstance(target, str) or not target.strip():
            raise ValueError('Empty phrase translation')
        if source in translated:
            raise ValueError('Duplicate phrase source')
        if placeholders(source) != placeholders(target):
            raise ValueError('Changed translation placeholders')
        for token in TOKENS:
            plural = r'(?:s)?' if token in FILE_FORMATS else ''
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(token) + plural + r'(?![A-Za-z0-9_])'
            if len(re.findall(pattern, source)) != len(re.findall(pattern, target)):
                raise ValueError(f'Changed technical token: {token}')
        for symbol in re.findall(r'\bspacr(?:\.[A-Za-z_]\w*)+', source):
            if symbol not in target:
                raise ValueError(f'Changed API symbol: {symbol}')
        translated[source] = target
    missing = required - translated.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} reviewed workflow phrases')
    return translated


def compose_reviews(bank, data, map_sha256, lessons=LESSONS):
    """Use the English generator for localized narration and unchanged links."""
    phrases = reviewed_phrases(bank, data, map_sha256, lessons)
    reviews = {}
    for lesson in lessons:
        english = workflow.lesson_document(data, lesson)
        localized = workflow.lesson_document(data, lesson, translate=phrases.__getitem__)
        for source, target in zip(english['scenes'], localized['scenes'], strict=True):
            if {k: v for k, v in source.items() if k != 'narration'} != {
                    k: v for k, v in target.items() if k != 'narration'}:
                raise ValueError('Translated lesson changed capture or link structure')
        reviews[lesson] = dict(
            language=bank['language'], lesson=lesson,
            english_sha256=canonical_hash(english),
            **{key: localized[key] for key in
               ('title', 'section', 'description', 'objectives', 'prerequisite')},
            scenes=[scene['narration'] for scene in localized['scenes']],
            review=dict(
                kind='AI-assisted phrase review composed by the shared workflow-map generator; module names match the English recording.',
                native_speaker_signoff=False, workflow_map_sha256=map_sha256,
                phrase_bank_sha256=canonical_hash(bank),
                corrections=['Use the same reviewed input/output and handoff phrases for the GUI and tutorial scripts.',
                             'Preserve literal API, table and column identifiers and the original capture/link structure.']))
    return reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'tools/tutorials/lessons/reviews')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    data = workflow.load()
    workflow.validate(data)
    digest = hashlib.sha256((ROOT / workflow.MAP_PATH).read_bytes()).hexdigest()
    for lesson, review in compose_reviews(bank, data, digest).items():
        path = args.output_dir / f"{lesson}.{bank['language']}.json"
        content = json.dumps(review, ensure_ascii=False, indent=2) + '\n'
        if args.check:
            if path.read_text() != content:
                raise ValueError(f'Generated translation review differs: {path}')
        else:
            if path.exists():
                raise FileExistsError(f'Refusing to replace an existing review: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        print(f"{lesson}/{bank['language']}: {len(review['scenes'])} source-bound scenes")


if __name__ == '__main__':
    main()
