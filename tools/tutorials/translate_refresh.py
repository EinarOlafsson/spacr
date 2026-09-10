#!/usr/bin/env python3
"""Translate only selected refreshed lessons into review-only draft catalogs.

Reuse the established local translators and pronunciation rules. Drafts never
replace the working narration catalogs automatically: semantic review is a
separate required step, not inferred from a successful model call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, REPO, read, write

WORKSPACE = DEFAULT_STAGE.parent
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(WORKSPACE / 'tools'))
import translate_catalog as spoken
from narration_audio import split_english_sentences

spec = importlib.util.spec_from_file_location('caption_translator', REPO / 'tools/translate_caption_catalogs.py')
captions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(captions)


def sentence_chunks(partial):
    """Prevent multi-sentence scene omissions by translating each sentence."""
    expanded = copy.deepcopy(partial)
    counts = {}
    for lesson in expanded['lessons']:
        chunks = []
        counts[lesson['id']] = []
        for scene in lesson['scenes']:
            sentences = split_english_sentences(scene['narration'])
            if not sentences or any(len(text.split()) > 120 for text in sentences):
                raise ValueError('Split an empty/overlong narration sentence before translation')
            counts[lesson['id']].append(len(sentences))
            chunks.extend(dict(scene, narration=text) for text in sentences)
        lesson['scenes'] = chunks
    return expanded, counts


def rejoin_sentences(translated, partial, counts, language):
    originals = {item['id']: item for item in partial['lessons']}
    for lesson in translated['lessons']:
        sizes = counts[lesson['id']]
        if len(lesson['scenes']) != sum(sizes):
            raise ValueError('The translator did not return every input sentence')
        scenes = []
        cursor = 0
        for original, size in zip(originals[lesson['id']]['scenes'], sizes):
            text = [row['narration'].strip() for row in lesson['scenes'][cursor:cursor + size]]
            if not all(text):
                raise ValueError('A source sentence has an empty translation')
            narration = (' ' if language not in {'ja', 'zh-CN'} else '').join(text)
            if language in spoken.LANGUAGES:
                scenes.append(dict(original, narration=narration,
                                   speech_text=spoken.speech_text(narration, language)))
            else:
                scenes.append({'narration': narration})
            cursor += size
        lesson['scenes'] = scenes
    return translated


def merge_selected(existing, translated, source):
    """Preserve every unselected reviewed entry, retaining English identity order."""
    records = {lesson['id']: lesson for lesson in existing['lessons']}
    records.update({lesson['id']: lesson for lesson in translated['lessons']})
    identities = [lesson['id'] for lesson in source['lessons']]
    missing = set(identities) - set(records)
    if missing:
        raise ValueError(f'Missing translations must be selected too: {sorted(missing)}')
    result = copy.deepcopy(existing)
    result['lessons'] = [records[identity] for identity in identities]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--lessons', nargs='+', required=True, help='Stable lesson IDs')
    parser.add_argument('--languages', nargs='+', default=list(spoken.LANGUAGES) + list(captions.LANGUAGES),
                        choices=list(spoken.LANGUAGES) + list(captions.LANGUAGES))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='Use CUDA only when no other tutorial GPU writer is active')
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or not 1 <= args.batch_size <= 4:
        parser.error('Keep this background draft pass to 1–4 threads and 1–4 strings per batch')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable; no silent device fallback')
        torch.cuda.set_per_process_memory_fraction(0.15, 0)
    english = args.stage / 'catalog/lessons_en.json'
    source = read(english)
    selected = set(args.lessons)
    partial = dict(source, series=[], translation_protocol='sentence-chunks-v2',
                   lessons=[lesson for lesson in source['lessons'] if lesson['id'] in selected])
    if selected != {lesson['id'] for lesson in partial['lessons']}:
        parser.error('Every selected lesson must exist in the staged English catalog')
    digest = hashlib.sha256(json.dumps(partial, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    destination = args.stage / 'catalog-drafts'
    manifest_path = destination / 'review-manifest.json'
    manifest = read(manifest_path) if manifest_path.exists() else {}
    model = Path(os.environ.get('SPACR_NLLB_MODEL', WORKSPACE / 'project/translation_models/nllb-200-distilled-600M'))
    if not model.is_dir():
        parser.error(f'The local translation model is unavailable: {model}')
    spoken.MODEL = model
    expanded, sentence_counts = sentence_chunks(partial)
    baseline = REPO / 'docs/source/_extra/tutorials/catalog'
    for language in args.languages:
        filename = f"{'lessons' if language in spoken.LANGUAGES else 'captions'}_{language}.json"
        target = destination / filename
        if (target.is_file() and manifest.get(language, {}).get('source_sha256') == digest
                and manifest[language].get('draft_sha256') == hashlib.sha256(target.read_bytes()).hexdigest()):
            print(f'{language}: existing draft still needs semantic review', flush=True)
            continue
        started = time.monotonic()
        reviewed_path = args.stage / 'catalog' / filename
        reviewed = read(reviewed_path if reviewed_path.exists() else baseline / filename)
        # Older draft files may predate safely appended lessons. Seed missing
        # identities from the reviewed catalog before preserving existing drafts;
        # never discover this only after an expensive translation has finished.
        current = merge_selected(reviewed, read(target), source) if target.exists() else reviewed
        if language in spoken.LANGUAGES:
            translated = spoken.translate_language(expanded, language, args.batch_size, args.threads, args.device)
        else:
            translated = captions.translate(expanded, language, model, args.batch_size, args.threads, args.device)
        translated = rejoin_sentences(translated, partial, sentence_counts, language)
        # Translation is long-running; never label output against edited English.
        latest = read(english)
        latest = dict(latest, series=[], translation_protocol='sentence-chunks-v2',
                      lessons=[lesson for lesson in latest['lessons'] if lesson['id'] in selected])
        if hashlib.sha256(json.dumps(latest, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != digest:
            raise RuntimeError('Selected English lessons changed during translation; refusing mixed drafts')
        write(target, merge_selected(current, translated, source))
        manifest[language] = {'state': 'machine_draft_requires_semantic_review',
                             'source_sha256': digest, 'selected_lessons': args.lessons,
                             'draft_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                             'device': args.device,
                             'elapsed_seconds': round(time.monotonic() - started, 2)}
        write(manifest_path, manifest)
        print(f'{language}: saved review-only draft ({manifest[language]["elapsed_seconds"]}s)', flush=True)


if __name__ == '__main__':
    main()
