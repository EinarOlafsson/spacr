#!/usr/bin/env python3
"""Promote explicit, source-pinned editorial corrections into staged catalogs.

An editorial review is not a claim of native-speaker or listening sign-off.
Published catalogs stay unchanged until all release requirements are met.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, REPO, read, write

authoring_tools = DEFAULT_STAGE.parent / 'tools'
if not authoring_tools.is_dir():
    authoring_tools = Path(__file__).resolve().parent / 'authoring/tools'
sys.path.insert(0, str(authoring_tools))
from pronunciation import spoken_form, assert_pronunciation_safe

SPOKEN = {'es', 'fr', 'hi', 'it', 'pt-BR', 'ja', 'zh-CN'}
CAPTION_ONLY = {'da', 'de', 'is', 'ko', 'nb', 'sv'}


def promote(review, stage):
    source = read(stage / 'catalog/lessons_en.json')
    english = next(item for item in source['lessons'] if item['id'] == review['lesson'])
    digest = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != review['english_sha256']:
        raise ValueError('English changed after the translation review')
    if len(review['scenes']) != len(english['scenes']) or not all(
            isinstance(text, str) and text.strip() for text in review['scenes']):
        raise ValueError('Review must explicitly provide every scene')
    language = review['language']
    if language not in SPOKEN | CAPTION_ONLY:
        raise ValueError('Review language is not in the preserved translation matrix')
    filename = f"{'lessons' if language in SPOKEN else 'captions'}_{language}.json"
    destination = stage / 'catalog' / filename
    baseline = REPO / 'docs/source/_extra/tutorials/catalog' / filename
    target = read(destination if destination.exists() else baseline)
    translated = copy.deepcopy(english)
    for key in ('title', 'description', 'objectives', 'prerequisite'):
        translated[key] = review[key]
    for scene, narration in zip(translated['scenes'], review['scenes']):
        scene['narration'] = narration
        if language in SPOKEN:
            scene['speech_text'] = spoken_form(narration, language)
            assert_pronunciation_safe(narration, scene['speech_text'])
        else:
            scene.pop('speech_text', None)
    existing = {item['id']: item for item in target['lessons']}
    existing[english['id']] = translated
    target['lessons'] = [existing[item['id']] for item in source['lessons']]
    write(destination, target)
    write(stage / 'production' / english['id'] / f'review.{language}.json', review)
    print(f"Staged editorial correction for {language}/{english['id']}; not published or listening-reviewed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    promote(read(args.review), args.stage)
