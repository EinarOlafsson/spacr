#!/usr/bin/env python3
"""Reconcile the one-voice refresh against its complete preserved lesson."""
import argparse
from pathlib import Path

from check_completed_matrix import check, digest
from stage_lesson import DEFAULT_STAGE, read, write

LESSON = '04_platform_installers'
PAIR = {Path('audio/en/af_heart.m4a'), Path('audio/en/af_heart.json')}


def verify(backup, stage=DEFAULT_STAGE):
    old, current = Path(backup) / LESSON, Path(stage) / 'production' / LESSON
    old_audio = {p.relative_to(old) for p in (old / 'audio').rglob('*') if p.is_file()}
    new_audio = {p.relative_to(current) for p in (current / 'audio').rglob('*') if p.is_file()}
    if len(old_audio) != 100 or new_audio != old_audio:
        raise ValueError('The fifty-voice audio/metadata inventory changed')
    changed = {p for p in old_audio if digest(old / p) != digest(current / p)}
    if changed != PAIR:
        raise ValueError(f'Expected only the Heart audio/metadata pair to change: {changed}')
    protected = []
    for path in sorted(old.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(old)
        if (relative in PAIR or relative == Path('captions/en/af_heart.vtt')
                or path.name.startswith('audio-checks.') or path.name == 'final-artifact-checks.json'):
            continue
        if not (current / relative).is_file() or digest(path) != digest(current / relative):
            raise ValueError(f'Unrelated lesson artifact changed: {relative}')
        protected.append({'path': relative.as_posix(), 'sha256': digest(path)})
    before, after = [read(folder / 'audio/en/af_heart.json') for folder in (old, current)]
    changes = []
    if len(before['scenes']) != len(after['scenes']):
        raise ValueError('Scene count changed')
    for scene_index, (previous, updated) in enumerate(zip(before['scenes'], after['scenes']), 1):
        if previous['text'] != updated['text'] or len(previous['sentences']) != len(updated['sentences']):
            raise ValueError('Display narration or sentence count changed')
        for sentence_index, (a, b) in enumerate(zip(previous['sentences'], updated['sentences']), 1):
            if a['text'] != b['text']:
                raise ValueError('Caption text changed')
            if (a['speech_text'], a['phonemes']) != (b['speech_text'], b['phonemes']):
                if (a['speech_text'].replace('[CUDA](/kˈuːdᵊ/)', '[CUDA](/kˈudə/)') != b['speech_text']
                        or 'kˈudə' not in b['phonemes'] or 'kˈuːdᵊ' in b['phonemes']):
                    raise ValueError('Unexpected speech or synthesis phoneme change')
                changes.append({'scene': scene_index, 'sentence': sentence_index, 'text': b['text']})
    if changes != [{'scene': 5, 'sentence': 2, 'text':
                    'Here the request was auto and the selected backend is CUDA on NVIDIA hardware.'}]:
        raise ValueError('The intended first CUDA sentence was not the sole speech-plan change')
    proof = check(stage, LESSON, DEFAULT_STAGE.parent / 'tools/render_all_voices.py')
    return {'lesson': LESSON, 'scope': 'One regenerated private track, not the older live recording',
            'original_backup': str(old.resolve()), 'source_audio_sha256': before['media_sha256'],
            'audio_sha256': after['media_sha256'], 'metadata_sha256': digest(current / 'audio/en/af_heart.json'),
            'original_duration': before['total_duration'], 'duration': after['total_duration'],
            'speech_plan_changes': changes, 'other_voice_pairs_byte_identical': 49,
            'shared_movie_unchanged': True, 'all_display_narration_unchanged': True,
            'full_heart_track_regenerated': True, 'protected_files': protected,
            'artifact_reconciliation': proof, 'passed': True,
            'human_listening_review': False, 'published': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.backup)
    write(args.output, result)
    print(f"Verified one changed track, 49 unchanged voice pairs, {len(result['protected_files'])} unchanged artifacts")
