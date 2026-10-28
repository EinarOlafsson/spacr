"""Remove only Home's published terminal Spaceout scene, without synthesis.

Published prefix PCM is retained exactly. A bounded, recorded uniform codec gain
may prevent AAC overshoot; no normalization or TTS is performed. Failed attempts
are preserved. This is never a current whole-track TTS fingerprint.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

from stage_lesson import REPO, read, write
from check_completed_matrix import digest, voice_matrix
from published_scene_audio_refresh import regular

LESSON = '05_home'
KIND = 'published_terminal_scene_removal'
RECORD = 'published-terminal-scene-removal.json'
WORK = 'published-terminal-prefix-pcm'
RATE = 24000
CANONICAL_SHA256 = '3a1433580ecf272e574eaec9d4451f2a889a41df465c948e8c7bd8bfeba16b19'


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def terminal_prefix(previous, current):
    """Only the exact terminal visual may disappear; all retained fields agree."""
    expected = deepcopy(previous)
    scenes = expected.get('scenes')
    if (not isinstance(scenes, list) or len(scenes) != 39
            or scenes[-1].get('visual') != '24_spaceout_field'):
        raise ValueError('Require original39 Home scenes and exact removed terminal visual')
    expected['scenes'] = scenes[:38]
    if expected != current:
        raise ValueError('Only terminal scene39 may change; preserve exact first38 and metadata')


def require_voice_inventory(voices, inventory):
    """Require every normal offering once; retain published presentation order."""
    if (not isinstance(voices, dict) or set(voices) != set(inventory)
            or any(not isinstance(voices[lang], list)
                   or len(voices[lang]) != len(set(voices[lang]))
                   or set(voices[lang]) != set(expected)
                   for lang, expected in inventory.items())):
        raise ValueError('Preserve exact normal fifty published offerings')


def _source(stage, baseline):
    """Validate actual published sources and adopted38-scene metadata; no decode."""
    from audit_staged_catalogs import CATALOGS
    from published_native_reuse import _published_inputs
    from published_retained_narration import _authoring_without_generated_aliases
    from retain_narration import catalog_lesson, require_timing
    from apply_translation_review import translated_lesson
    if Path(stage).is_symlink() or Path(baseline).is_symlink():
        raise ValueError('Independent regular stage and genuine published baseline required')
    stage, baseline = Path(stage).resolve(), Path(baseline).resolve()
    if stage == baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):
        raise ValueError('Independent terminal-removal stage required')
    baseline, native, bindings, published = _published_inputs(baseline, LESSON)
    if (len(native.get('scenes', [])) != 39
            or native['scenes'][-1].get('visual') != '24_spaceout_field'):
        raise ValueError('Published native source must contain exact39 Home chapters')
    manifest = read(regular(baseline, 'release-manifest.json'))
    records = {row['path']: row for row in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate source manifest paths')
    def bound(relative):
        path = regular(baseline, relative)
        row = records.get(relative)
        if not row or (path.stat().st_size, digest(path)) != (row['bytes'], row['sha256']):
            raise ValueError('Published terminal-removal input differs from manifest: ' + relative)
        return path
    canonical = read(regular(REPO, f'tools/tutorials/lessons/{LESSON}.json'))
    if object_digest(canonical) != CANONICAL_SHA256 or read(regular(stage, f'production/{LESSON}/lesson.en.json')) != canonical:
        raise ValueError('Require exact reviewed Home terminal-deletion canonical')
    inventory = voice_matrix(REPO / 'tools/tutorials/authoring/tools/render_all_voices.py')
    expected_voices = {(lang, voice) for lang, voices in inventory.items() for voice in voices}
    if len(expected_voices) != 50:
        raise ValueError('Require normal fifty-offering inventory')
    localized, originals, catalogs, reviews = {}, {}, {}, {}
    english = None
    for name in CATALOGS:
        old_catalog = read(bound('web/catalog/' + name))
        new_catalog = read(regular(stage, 'catalog/' + name))
        old = catalog_lesson(baseline / 'web/catalog' / name, LESSON)
        current = catalog_lesson(stage / 'catalog' / name, LESSON)
        language = name.split('_', 1)[1].removesuffix('.json')
        prefix = deepcopy(old); prefix['scenes'] = prefix['scenes'][:38]
        compared = _authoring_without_generated_aliases(current, prefix, language)
        terminal_prefix(old, compared)
        expected_catalog = deepcopy(old_catalog)
        for index, row in enumerate(expected_catalog['lessons']):
            if row['id'] == LESSON:
                expected_catalog['lessons'][index] = current
        if expected_catalog != new_catalog:
            raise ValueError('Preserve every other catalog object, global field and order')
        require_voice_inventory(current.get('narration_voices'), inventory)
        if name == 'lessons_en.json':
            english = current
            if {k: v for k, v in compared.items() if k != 'narration_voices'} != canonical:
                raise ValueError('Current catalog differs from canonical Home source')
        else:
            path = regular(stage, f'production/{LESSON}/review.{language}.json')
            if path.read_bytes() != regular(REPO, f'tools/tutorials/lessons/reviews/{LESSON}.{language}.json').read_bytes():
                raise ValueError('Require actual adopted current full translation review')
            reviews[language] = digest(path)
        originals[language], localized[language] = old, current
        catalogs[name] = digest(regular(stage, 'catalog/' + name))
    for language in reviews:
        review = read(stage / f'production/{LESSON}/review.{language}.json')
        if translated_lesson(review, english, originals[language]) != localized[language]:
            raise ValueError('Current full localized review differs')
    tracks = []
    prefix = f'media_host/{LESSON}/audio/'
    expected_paths = {prefix + f'{lang}/{voice}{suffix}' for lang, voice in expected_voices for suffix in ('.m4a', '.json')}
    if {p for p in records if p.startswith(prefix)} != expected_paths:
        raise ValueError('Require exact published100 audio/timing files')
    for language, voice in sorted(expected_voices):
        stem = prefix + f'{language}/{voice}'
        audio, timing = bound(stem + '.m4a'), bound(stem + '.json')
        old = read(timing)
        require_timing(old, originals[language], language, voice, digest(audio))
        if (len(old.get('scenes', [])) != 39 or old.get('sample_rate') != RATE
                or [s.get('scene') for s in old['scenes']] != list(range(1, 40))
                or [s.get('text') for s in old['scenes']] != [s['narration'] for s in originals[language]['scenes']]
                or old.get('media_bytes') != audio.stat().st_size):
            raise ValueError('Original published narration timing/source differs')
        tracks.append(dict(language=language, voice=voice, audio=stem + '.m4a', timing=stem + '.json',
                           audio_sha256=digest(audio), timing_sha256=digest(timing)))
    source = dict(**published, published_bindings=bindings,
                  original_visual_clock=dict(path=bindings['timing']['path'], sha256=bindings['timing']['sha256']),
                  canonical_sha256=object_digest(canonical), catalogs=catalogs, reviews=reviews, tracks=tracks)
    return source, english, native


def retained_prefix(source_pcm, old):
    """Keep exact spoken prefix; append only silence for the declared final1s."""
    scenes = deepcopy(old.get('scenes', []))
    if len(scenes) != 39 or len(source_pcm) % 2:
        raise ValueError('Require original39 scenes and mono s16 PCM')
    cursor = 0
    for index, scene in enumerate(scenes):
        values = [scene.get(k) for k in ('speech_start', 'speech_end', 'scene_end')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Require finite original scene bounds')
        start, speech, end = [round(v * RATE) for v in values]
        if scene.get('scene') != index + 1 or start != cursor or not start < speech <= end or end > len(source_pcm) // 2:
            raise ValueError('Original scenes must cover contiguous decoded source PCM')
        cursor = end
    scenes = scenes[:38]
    final = scenes[-1]
    sentences = final.get('sentences')
    if not isinstance(sentences, list) or not sentences:
        raise ValueError('Final retained scene requires original validated audible metadata')
    ends = [s.get('audible_end') for s in sentences]
    starts = [s.get('audible_start') for s in sentences]
    if any(type(a) not in (int, float) or type(b) not in (int, float)
           or not math.isfinite(a) or not math.isfinite(b) or not final['speech_start'] <= a < b <= final['speech_end']
           for a, b in zip(starts, ends)):
        raise ValueError('Invalid original retained audible bounds')
    target = round((max(ends) + 1.0) * RATE)
    if target < round(final['speech_end'] * RATE):
        raise ValueError('One-second final gap cannot truncate original full speech')
    stop = min(target, round(final['scene_end'] * RATE))
    kept = source_pcm[:stop * 2]
    padding = target - stop
    joined = kept + b'\0\0' * padding
    final['scene_end'] = target / RATE
    final['duration'] = final['scene_end'] - final['speech_start']
    final['hold_after'] = final['scene_end'] - final['speech_end']
    proof = dict(source_samples=[0, stop], output_samples=[0, stop],
                 retained_pcm_sha256=hashlib.sha256(kept).hexdigest(),
                 final_silence_samples=padding, output_sample_count=target,
                 final_audible_end=max(ends), final_gap_seconds=target / RATE - max(ends))
    return joined, scenes, proof


def timing_metadata(old, scenes, audio, samples, *, gain=0.):
    result = dict(schema=1, language=old['language'], voice=old['voice'], lang_code=old['lang_code'],
                dialect=old['dialect'], speed=old['speed'], sample_rate=RATE, total_duration=samples / RATE,
                media_sha256=digest(audio), media_bytes=audio.stat().st_size,
                render_inputs=dict(kind=KIND, lesson=LESSON, language=old['language'], voice=old['voice'],
                    scenes=[dict(narration=s['text']) for s in scenes]),
                scenes=scenes, published_terminal_scene_removal=True)
    if gain:
        result['codec_compensation_db'] = gain
    return result


LEGACY_PRODUCER = '945cac64d13d13a64063e836747bffe0db9550d019f3d8b5f319abb9b41ef2a6'
LEGACY_FAILED_SHA = 'd37cc81b2e734a4c4c8761fb7cad4236d34329e7eb80b8a3dc266a93174b1af2'
CODEC_GAINS = (0., -.25, -.5, -.75, -1.)


def codec_pcm(pcm, gain):
    """Explicit bounded codec headroom only; no normalization or limiter."""
    if type(gain) not in (int, float) or gain not in CODEC_GAINS or len(pcm) % 2:
        raise ValueError('Only reviewed quarter-dB codec compensation through -1dB')
    if gain == 0:
        return pcm
    import numpy as np
    samples = np.frombuffer(pcm, dtype='<i2')
    return np.rint(samples.astype('float64') * (10. ** (gain / 20.))).astype('<i2').tobytes()


def codec_path(work, gain):
    return work / ('prefix.s16le' if gain == 0 else f'codec-input-{round(-gain * 100):03}.s16le')


def encode_prefix(prefix, audio, work, renderer, *, legacy_failed=False):
    """Retain every failed AAC; compensate only if normal decoded peak fails."""
    def encode(pcm, target):
        subprocess.run(['ffmpeg', '-nostdin', '-n', '-v', 'error', '-threads', '2', '-f', 's16le',
            '-ar', str(RATE), '-ac', '1', '-i', str(pcm), '-c:a', 'aac', '-b:a', '48k',
            '-movflags', '+faststart', str(target)], check=True)
    if not legacy_failed:
        encode(prefix, audio)
        peak = renderer.true_peak_dbfs(audio)
        if type(peak) not in (int, float) or not math.isfinite(peak):
            raise ValueError('Normal true-peak measurement unavailable; no compensation allowed')
        if peak <= renderer.MAX_RELEASE_TRUE_PEAK_DBFS:
            return 0., prefix, peak
    preserved = work / 'codec-uncompensated.m4a'
    if os.path.lexists(preserved):
        raise FileExistsError('Preserve earlier codec attempts; do not overwrite')
    audio.rename(preserved)
    for gain in CODEC_GAINS[1:]:
        pcm = codec_path(work, gain)
        if os.path.lexists(pcm):
            raise FileExistsError('Preserve earlier codec-input PCM')
        pcm.write_bytes(codec_pcm(prefix.read_bytes(), gain))
        attempt = work / f'codec-minus{round(-gain * 100):03}.m4a'
        encode(pcm, attempt)
        peak = renderer.true_peak_dbfs(attempt)
        if type(peak) not in (int, float) or not math.isfinite(peak):
            raise ValueError('Normal true-peak measurement unavailable; preserve codec attempt')
        if peak <= renderer.MAX_RELEASE_TRUE_PEAK_DBFS:
            attempt.rename(audio)
            return gain, pcm, peak
    raise ValueError('Normal decoded peak still fails after bounded codec compensation; preserve all attempts')


def compare_source_decode(audio, saved):
    """Legacy partials lacked journals: verify original AAC once, never encode."""
    decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(audio),
        '-vn', '-ac', '1', '-ar', str(RATE), '-c:a', 'pcm_s16le', '-f', 's16le', '-'],
        check=True, capture_output=True)
    if decoded.stderr or decoded.stdout != saved.read_bytes():
        raise ValueError('Legacy saved PCM differs from strict original published AAC decode')


def track_proof(original, work, audio, peak, gain, encoded):
    from narration_audio import ffmpeg_runtime_identity
    return dict(**original, source_pcm_sha256=digest(work / 'source.s16le'),
        source_decoded_samples=(work / 'source.s16le').stat().st_size // 2,
        decoder=ffmpeg_runtime_identity(), prefix_pcm_sha256=digest(work / 'prefix.s16le'),
        intervals=None, codec_gain_db=gain, codec_pcm_sha256=digest(encoded),
        output_audio_sha256=digest(audio), output_timing_sha256=digest(audio.with_suffix('.json')),
        decoded_true_peak_dbfs=peak)


def recover_legacy(stage, source, log, renderer):
    """Only the observed12 complete + Michael failure; validate all before edits."""
    root = stage / 'production' / LESSON
    partials = source['tracks'][:13]
    if len(partials) != 13 or (partials[-1]['language'], partials[-1]['voice']) != ('en', 'am_michael'):
        raise ValueError('Unknown legacy partial ordering')
    log = regular(Path(log).resolve().parent, Path(log).name)
    text = log.read_text()
    if (str(stage) not in text or source['baseline_root'] not in text
            or 'Normal final true-peak check failed' not in text or 'Terminal exit status: 1' not in text):
        raise ValueError('Require the actual failed legacy producer log')
    expected_audio = {f"{r['language']}/{r['voice']}{suffix}" for r in partials[:-1] for suffix in ('.m4a', '.json')}
    expected_audio.add('en/am_michael.m4a')
    if {p.relative_to(root / 'audio').as_posix() for p in (root / 'audio').rglob('*') if p.is_file()} != expected_audio:
        raise ValueError('Unknown or incomplete legacy output inventory')
    expected_work = {f"{r['language']}/{r['voice']}/{name}.s16le" for r in partials for name in ('source', 'prefix')}
    if {p.relative_to(root / WORK).as_posix() for p in (root / WORK).rglob('*') if p.is_file()} != expected_work:
        raise ValueError('Unknown or incomplete legacy PCM inventory')
    failed = regular(root, 'audio/en/am_michael.m4a')
    if digest(failed) != LEGACY_FAILED_SHA:
        raise ValueError('Actual previously measured failed AAC changed')
    rows, verified = [], []
    for index, original in enumerate(partials):
        language, voice = original['language'], original['voice']
        work = root / WORK / language / voice
        pcm, prefix = regular(work, 'source.s16le'), regular(work, 'prefix.s16le')
        old = read(Path(source['baseline_root']) / original['timing'])
        joined, scenes, intervals = retained_prefix(pcm.read_bytes(), old)
        if prefix.read_bytes() != joined:
            raise ValueError('Legacy retained prefix PCM differs')
        audio = regular(root, Path('audio') / language / (voice + '.m4a'))
        if index < 12 and read(regular(root, Path('audio') / language / (voice + '.json'))) != timing_metadata(old, scenes, audio, len(joined) // 2):
            raise ValueError('Legacy output timing/source/audio hash differs')
        compare_source_decode(Path(source['baseline_root']) / original['audio'], pcm)
        verified.append(dict(language=language, voice=voice, audio_sha256=original['audio_sha256'], source_pcm_sha256=digest(pcm)))
        if index < 12:
            peak = renderer.true_peak_dbfs(audio)
            if type(peak) not in (int, float) or not math.isfinite(peak) or peak > renderer.MAX_RELEASE_TRUE_PEAK_DBFS:
                raise ValueError('Previously complete legacy AAC fails normal peak; no silent rewrite')
            row = track_proof(original, work, audio, peak, 0., prefix)
            row['intervals'] = intervals; rows.append(row)
    recovery = dict(producer_sha256=LEGACY_PRODUCER, log_path=str(log), log_sha256=digest(log),
        source_sha256=object_digest(source), verified_source_decodes=verified,
        retained_complete_tracks=12, failed_audio_sha256=LEGACY_FAILED_SHA,
        failed_audio_true_peak_dbfs=-.9)
    return rows, recovery


def render(stage, baseline, *, resume_log=None):
    """Published prefix only; guarded AAC gain and strict legacy partial recovery."""
    source, english, _ = _source(stage, baseline)
    stage = Path(stage).resolve(); root = stage / 'production' / LESSON
    if os.path.lexists(root / RECORD):
        raise FileExistsError('Preserve completed terminal-removal record')
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import render_all_voices as renderer
    if resume_log is None:
        if os.path.lexists(root / WORK) or os.path.lexists(root / 'audio'):
            raise FileExistsError('Use a fresh stage; only explicit verified legacy resume accepts partials')
        rows, recovery = [], None
    else:
        rows, recovery = recover_legacy(stage, source, resume_log, renderer)
        for row in rows:
            work = root / WORK / row['language'] / row['voice']
            write(work / 'track-proof.json', dict(producer_sha256=digest(__file__), source_sha256=object_digest(source),
                recovered_original_producer=LEGACY_PRODUCER, track=row))
    for index, original in enumerate(source['tracks'][len(rows):], start=len(rows)):
        language, voice = original['language'], original['voice']; legacy_failed = recovery is not None and index == 12
        work = root / WORK / language / voice
        pcm = work / 'source.s16le'; prefix = work / 'prefix.s16le'
        if not legacy_failed:
            work.mkdir(parents=True, exist_ok=False)
            subprocess.run(['ffmpeg', '-nostdin', '-n', '-v', 'error', '-threads', '2', '-i',
                str(Path(source['baseline_root']) / original['audio']), '-vn', '-ac', '1', '-ar', str(RATE),
                '-c:a', 'pcm_s16le', '-f', 's16le', str(pcm)], check=True)
        old = read(Path(source['baseline_root']) / original['timing'])
        joined, scenes, intervals = retained_prefix(pcm.read_bytes(), old)
        if not legacy_failed: prefix.write_bytes(joined)
        elif prefix.read_bytes() != joined: raise ValueError('Failed track prefix changed after verification')
        audio = root / 'audio' / language / (voice + '.m4a'); audio.parent.mkdir(parents=True, exist_ok=True)
        gain, encoded, peak = encode_prefix(prefix, audio, work, renderer, legacy_failed=legacy_failed)
        write(audio.with_suffix('.json'), timing_metadata(old, scenes, audio, len(joined) // 2, gain=gain))
        row = track_proof(original, work, audio, peak, gain, encoded); row['intervals'] = intervals
        write(work / 'track-proof.json', dict(producer_sha256=digest(__file__), source_sha256=object_digest(source),
            recovered_original_producer=None, track=row))
        rows.append(row)
    if _source(stage, baseline)[0] != source:
        raise ValueError('Frozen source/catalog/review changed during terminal removal')
    write(root / RECORD, dict(schema=1, kind=KIND, lesson=LESSON, source=source,
          producer_sha256=digest(__file__), tracks=rows, recovery=recovery, resynthesized=False,
          current_runtime_freshness_claimed=False, fresh_capture=False, human_listening_review=False))
    return root / RECORD


def check_recovery(root, source, rows, recovery):
    if recovery is None:
        return set()
    fields = {'producer_sha256', 'log_path', 'log_sha256', 'source_sha256', 'verified_source_decodes',
              'retained_complete_tracks', 'failed_audio_sha256', 'failed_audio_true_peak_dbfs'}
    if (not isinstance(recovery, dict) or set(recovery) != fields
            or recovery['producer_sha256'] != LEGACY_PRODUCER
            or recovery['source_sha256'] != object_digest(source) or recovery['retained_complete_tracks'] != 12
            or recovery['failed_audio_sha256'] != LEGACY_FAILED_SHA or recovery['failed_audio_true_peak_dbfs'] != -.9):
        raise ValueError('Legacy recovery provenance changed')
    log = Path(recovery['log_path'])
    if digest(regular(log.parent, log.name)) != recovery['log_sha256']:
        raise ValueError('Original failed producer log changed')
    partials = source['tracks'][:13]
    if (len(partials) != 13 or (partials[-1]['language'], partials[-1]['voice']) != ('en', 'am_michael')
            or digest(regular(root, Path(WORK) / 'en/am_michael/codec-uncompensated.m4a')) != LEGACY_FAILED_SHA):
        raise ValueError('Preserved actual failed AAC or source order changed')
    by_voice = {(r['language'], r['voice']): r for r in rows}
    expected = [dict(language=r['language'], voice=r['voice'], audio_sha256=r['audio_sha256'],
                     source_pcm_sha256=by_voice[r['language'], r['voice']]['source_pcm_sha256']) for r in partials]
    if recovery['verified_source_decodes'] != expected:
        raise ValueError('Strict legacy source-decode verification identities changed')
    return {(r['language'], r['voice']) for r in partials[:12]}


def checked_terminal_narration(stage, identity=LESSON, baseline=None, *, decode=False):
    """Reconstruct retained PCM without source re-decode; optional normal final QA."""
    if identity != LESSON:
        raise ValueError('Terminal removal is explicit only for Home')
    stage = Path(stage).resolve(); root = stage / 'production' / identity
    record = read(regular(root, RECORD))
    source, english, _ = _source(stage, baseline or record['source']['baseline_root'])
    if (set(record) != {'schema', 'kind', 'lesson', 'source', 'producer_sha256', 'tracks',
                        'resynthesized', 'current_runtime_freshness_claimed', 'fresh_capture', 'human_listening_review', 'recovery'}
            or record.get('schema') != 1 or record.get('kind') != KIND or record.get('lesson') != identity
            or record.get('source') != source or record.get('producer_sha256') != digest(__file__)
            or any(record.get(k) is not False for k in ('resynthesized', 'current_runtime_freshness_claimed', 'fresh_capture', 'human_listening_review'))):
        raise ValueError('Explicit retained terminal-prefix provenance changed')
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import render_all_voices as renderer
    from narration_audio import ffmpeg_runtime_identity
    rows = record.get('tracks')
    expected = {(r['language'], r['voice']) for r in source['tracks']}
    if not isinstance(rows, list) or len(rows) != 50 or {(r['language'], r['voice']) for r in rows} != expected:
        raise ValueError('Require exact fifty terminal-prefix proofs')
    wanted = {f'{l}/{v}{s}' for l, v in expected for s in ('.m4a', '.json')}
    if {p.relative_to(root / 'audio').as_posix() for p in (root / 'audio').rglob('*') if p.is_file()} != wanted:
        raise ValueError('Output audio inventory must remain exact100 files')
    source_rows = {(r['language'], r['voice']): r for r in source['tracks']}
    recovered = check_recovery(root, source, rows, record['recovery'])
    tracks, proofs = [], []
    for row in rows:
        language, voice = row['language'], row['voice']; original = source_rows[language, voice]
        if (set(row) != set(original) | {'source_pcm_sha256', 'source_decoded_samples', 'decoder',
                'prefix_pcm_sha256', 'intervals', 'output_audio_sha256', 'output_timing_sha256', 'decoded_true_peak_dbfs',
                'codec_gain_db', 'codec_pcm_sha256'}
                or any(row.get(k) != v for k, v in original.items())):
            raise ValueError('Original published track binding changed')
        work = Path(WORK) / language / voice
        pcm, joined_path = regular(root, work / 'source.s16le'), regular(root, work / 'prefix.s16le')
        old = read(Path(source['baseline_root']) / original['timing'])
        joined, scenes, intervals = retained_prefix(pcm.read_bytes(), old)
        audio = regular(root, Path('audio') / language / (voice + '.m4a'))
        timing = regular(root, Path('audio') / language / (voice + '.json'))
        peak = row.get('decoded_true_peak_dbfs')
        gain = row.get('codec_gain_db')
        expected_encoded = codec_pcm(joined, gain)
        encoded = regular(root, work / codec_path(Path('.'), gain))
        if (row.get('source_pcm_sha256') != digest(pcm) or row.get('source_decoded_samples') != pcm.stat().st_size // 2
                or row.get('prefix_pcm_sha256') != digest(joined_path) or joined_path.read_bytes() != joined
                or row.get('codec_pcm_sha256') != digest(encoded) or encoded.read_bytes() != expected_encoded
                or row.get('intervals') != intervals or row.get('output_audio_sha256') != digest(audio)
                or row.get('output_timing_sha256') != digest(timing)
                or row.get('decoder') != ffmpeg_runtime_identity()
                or type(peak) not in (int, float) or not math.isfinite(peak) or peak > renderer.MAX_RELEASE_TRUE_PEAK_DBFS
                or read(timing) != timing_metadata(old, scenes, audio, len(joined) // 2, gain=gain)):
            raise ValueError('Retained PCM, output timing/bytes or original sound changed')
        if decode:
            sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
            import verify_audio_release as audio_checks
            errors = audio_checks.check_track(audio)
            if errors:
                raise ValueError('Normal final AAC check failed: ' + str(errors))
        journal = read(regular(root, work / 'track-proof.json'))
        if journal != dict(producer_sha256=digest(__file__), source_sha256=object_digest(source),
                recovered_original_producer=LEGACY_PRODUCER if (language, voice) in recovered else None, track=row):
            raise ValueError('Per-success track proof changed')
        tracks.append(read(timing))
        proofs.append(dict(language=language, voice=voice, audio_sha256=digest(audio), timing_sha256=digest(timing)))
    return tracks, proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--render', action='store_true', required=True)
    parser.add_argument('--resume-log', type=Path, help='Explicit legacy12complete/Michael-failed recovery log')
    args = parser.parse_args()
    print(render(args.stage, args.baseline, resume_log=args.resume_log))


if __name__ == '__main__':
    main()
