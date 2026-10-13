#!/usr/bin/env python3
"""Repair one hosted Heart sentence, with no changes to the shared movie.

Never run the bulk tutorial publisher for this repair: the local authoring
track is a different revision. Download and hash-check the exact hosted pair,
preserve a backup, and keep the replacement inside its existing time window.
Publication is a separate, explicit, optimistic-locking two-file commit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ID = 'einarolafsson/spacr-tutorials'
MEDIA_PATH = '04_platform_installers/audio/en/af_heart'
EXPECTED_AUDIO = '854bf0ff5e881e449b7bbfb7da1d1da629d9a91437ac296f91c611ebc32c78e3'
WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
SAMPLE_RATE = 24000
OLD_PHONEMES = '/kˈuːdᵊ/'
NEW_PHONEMES = '/kˈudə/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def select_sentence(metadata):
    if metadata.get('language') != 'en' or metadata.get('voice') != 'af_heart':
        raise ValueError('Only the reported English Heart track is authorized')
    for scene_index, scene in enumerate(metadata['scenes']):
        for sentence_index, sentence in enumerate(scene['sentences']):
            if 'CUDA' in sentence['text']:
                if 'CUDA support' not in sentence['text']:
                    raise ValueError('First CUDA occurrence is no longer the reported sentence')
                return scene_index, sentence_index, sentence
    raise ValueError('Reported CUDA sentence is missing')


def replace_samples(source, replacement, start, end):
    """No sample outside the authorized interval may change before AAC encoding."""
    import numpy as np
    if not (0 <= start < end <= len(source)) or len(replacement) != end - start:
        raise ValueError('Replacement must fit the existing sentence exactly')
    result = source.copy()
    result[start:end] = replacement
    if not np.array_equal(result[:start], source[:start]) or not np.array_equal(result[end:], source[end:]):
        raise ValueError('Unrelated narration samples changed')
    return result


def decode(path):
    import numpy as np
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-'])
    return np.frombuffer(data, dtype='<f4').copy()


def build(output):
    from huggingface_hub import HfApi, hf_hub_download
    import numpy as np
    import soundfile as sf

    output.mkdir(parents=True, exist_ok=False)
    original, repaired = output / 'original', output / 'repaired'
    original.mkdir()
    repaired.mkdir()
    revision = HfApi().dataset_info(REPO_ID).sha
    for suffix in ('.m4a', '.json'):
        cached = hf_hub_download(REPO_ID, MEDIA_PATH + suffix, repo_type='dataset', revision=revision)
        (original / ('af_heart' + suffix)).write_bytes(Path(cached).read_bytes())
    source_bytes = (original / 'af_heart.m4a').read_bytes()
    metadata = json.loads((original / 'af_heart.json').read_text())
    if digest(source_bytes) != EXPECTED_AUDIO or metadata['media_sha256'] != EXPECTED_AUDIO:
        raise ValueError('Hosted Heart narration changed; do not repair an unverified revision')
    scene_index, sentence_index, sentence = select_sentence(metadata)

    # Set the existing cache before importing the pinned renderer/torch stack.
    os.environ['HF_HOME'] = str(WORKSPACE / 'project/kokoro_models')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(WORKSPACE / 'tools'))
    from render_kokoro_narration import _build_synthesis_stack, _model_snapshot
    from narration_audio import assemble_sentence_audio
    from kokoro import KModel, KPipeline
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20260911)
    snapshot = _model_snapshot(WORKSPACE / 'project/kokoro_models')
    pipeline, voice, runtime = _build_synthesis_stack(KModel, KPipeline, snapshot, 'af_heart', 'a', 'cpu')
    speech = sentence['speech_text']
    if speech.count(OLD_PHONEMES) != 1:
        raise ValueError('The expected single pronunciation override changed')
    speech = speech.replace(OLD_PHONEMES, NEW_PHONEMES)
    generated = list(pipeline(speech, voice=voice, speed=1.0))
    phonemes = ' '.join(result.phonemes for result in generated)
    if NEW_PHONEMES.strip('/') not in phonemes or OLD_PHONEMES.strip('/') in phonemes:
        raise ValueError('Synthesizer did not honor the full final schwa override')
    raw = np.concatenate([r.audio.detach().cpu().numpy() for r in generated if r.audio is not None])
    assembled, segments = assemble_sentence_audio([raw], np)
    sf.write(output / 'replacement-natural.wav', assembled, SAMPLE_RATE, subtype='FLOAT')
    source = decode(original / 'af_heart.m4a')
    start = round(sentence['speech_start'] * SAMPLE_RATE)
    end = round(sentence['speech_end'] * SAMPLE_RATE)
    ratio = len(assembled) / (end - start)
    if not .90 <= ratio <= 1.10:
        raise ValueError(f'Replacement needs excessive speed change: {ratio}')
    fitted_bytes = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output / 'replacement-natural.wav'),
        '-af', f'atempo={ratio:.12f},apad,atrim=end_sample={end-start}',
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-'])
    fitted = np.frombuffer(fitted_bytes, dtype='<f4').copy()
    gain = float(np.sqrt(np.mean(source[start:end] ** 2)) / np.sqrt(np.mean(fitted ** 2)))
    fitted *= gain
    if float(np.max(np.abs(fitted))) >= .89:
        raise ValueError('Replacement gain would consume the codec headroom')
    result = replace_samples(source, fitted, start, end)
    sf.write(output / 'replacement-fitted.wav', fitted, SAMPLE_RATE, subtype='PCM_16')
    sf.write(output / 'repaired-master.wav', result, SAMPLE_RATE, subtype='FLOAT')
    audio = repaired / 'af_heart.m4a'
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(output / 'repaired-master.wav'),
        '-t', str(metadata['total_duration']), '-c:a', 'aac', '-b:a', '48k', '-ar', str(SAMPLE_RATE),
        '-ac', '1', '-movflags', '+faststart', str(audio)], check=True)
    encoded = audio.read_bytes()
    final_pcm = decode(audio)
    if len(final_pcm) != len(source) or not np.all(np.isfinite(final_pcm)):
        raise ValueError('Encoded duration or finite-sample check failed')
    peak = float(20 * np.log10(np.max(np.abs(final_pcm))))
    if peak > -1:
        raise ValueError(f'Decoded AAC peak is too high: {peak} dBFS')
    updated = deepcopy(metadata)
    target_scene = updated['scenes'][scene_index]
    target = target_scene['sentences'][sentence_index]
    target['speech_text'], target['phonemes'] = speech, phonemes
    target['audible_start'] = sentence['speech_start'] + segments[0]['audible_start'] / ratio
    target['audible_end'] = sentence['speech_start'] + segments[0]['audible_end'] / ratio
    target_scene['speech_text'] = ' '.join(s['speech_text'] for s in target_scene['sentences'])
    target_scene['phonemes'] = ' '.join(s['phonemes'] for s in target_scene['sentences'])
    repair = {'kind': 'one_sentence_pronunciation_repair', 'source_revision': revision,
        'source_audio_sha256': EXPECTED_AUDIO, 'source_metadata_sha256': digest((original/'af_heart.json').read_bytes()),
        'scene': scene_index+1, 'sentence': sentence_index+1,
        'start': sentence['speech_start'], 'end': sentence['speech_end'],
        'old_phonemes': OLD_PHONEMES, 'new_phonemes': NEW_PHONEMES,
        'tempo_ratio': ratio, 'rms_gain': gain, 'seed': 20260911,
        'outside_interval_pcm_identical_before_encoding': True,
        'all_caption_text_and_sentence_boundaries_unchanged': True,
        'whole_track_reencoded': True, 'movie_changed': False,
        'decoded_sample_peak_dbfs': peak, 'runtime': runtime,
        'repair_script_sha256': digest(Path(__file__).read_bytes())}
    updated['source_render_fingerprint'] = updated.get('render_fingerprint')
    updated['source_render_inputs'] = updated.get('render_inputs')
    updated['render_inputs'] = repair
    updated['render_fingerprint'] = digest(json.dumps(repair, sort_keys=True).encode())
    updated['media_sha256'], updated['media_bytes'] = digest(encoded), len(encoded)
    for before, after in zip(metadata['scenes'], updated['scenes']):
        assert before['text'] == after['text']
        assert [(s['text'], s['speech_start'], s['speech_end']) for s in before['sentences']] == [
            (s['text'], s['speech_start'], s['speech_end']) for s in after['sentences']]
    write_json(repaired / 'af_heart.json', updated)
    write_json(output / 'repair-receipt.json', {'repair': repair, 'media_sha256': digest(encoded),
        'metadata_sha256': digest((repaired/'af_heart.json').read_bytes()), 'published': False})
    print(json.dumps({'output':str(output), 'scene':scene_index+1, 'sentence':sentence_index+1,
                      'tempo_ratio':ratio,'peak_dbfs':peak,'media_sha256':digest(encoded)}), flush=True)


def publish(output):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    receipt = json.loads((output / 'repair-receipt.json').read_text())
    api = HfApi()
    revision = api.dataset_info(REPO_ID).sha
    expected = {'.m4a': EXPECTED_AUDIO, '.json': receipt['repair']['source_metadata_sha256']}
    operations = []
    for suffix in ('.m4a', '.json'):
        current = Path(hf_hub_download(REPO_ID, MEDIA_PATH+suffix, repo_type='dataset', revision=revision))
        if digest(current.read_bytes()) != expected[suffix]:
            raise ValueError('Concurrent change to the target audio pair; publication refused')
        replacement = output / 'repaired' / ('af_heart'+suffix)
        key = 'media_sha256' if suffix == '.m4a' else 'metadata_sha256'
        if digest(replacement.read_bytes()) != receipt[key]:
            raise ValueError('Repair output changed after verification')
        operations.append(CommitOperationAdd(path_in_repo=MEDIA_PATH+suffix, path_or_fileobj=str(replacement)))
    commit = api.create_commit(repo_id=REPO_ID, repo_type='dataset', parent_commit=revision,
        operations=operations, commit_message='Correct the first CUDA in Platform Installers English Heart')
    receipt.update(published=True, media_host_commit=commit.oid)
    write_json(output / 'repair-receipt.json', receipt)
    print('Published only the Heart M4A/JSON pair:', commit.oid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    (publish if args.publish else build)(args.output)
