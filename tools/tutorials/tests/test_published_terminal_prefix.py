"""Synthetic prefix/source guard tests; no producer, decoder, browser or models."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import published_terminal_prefix as terminal
import build_appended_candidate as candidate


def authored():
    return {'id': '05_home', 'title': 'Home', 'scenes': [
        {'visual': f'v{i}', 'narration': f'Words {i}.'} for i in range(38)
    ] + [{'visual': '24_spaceout_field', 'narration': 'Removed words.'}]}


def timed():
    return dict(language='en', voice='af_heart', lang_code='a', dialect='American English', speed=1,
                scenes=[dict(scene=i + 1, text=f'Words {i}.', speech_start=i * 2.,
                             speech_end=i * 2. + 1., scene_end=(i + 1) * 2., duration=2.,
                             hold_after=1., sentences=[dict(sentence=1, speech_start=i * 2.,
                                speech_end=i * 2. + 1., audible_start=i * 2. + .1,
                                audible_end=i * 2. + .8)]) for i in range(39)])


def test_only_exact_terminal_scene_and_prefix_metadata_may_change():
    old = authored(); new = deepcopy(old); new['scenes'].pop()
    terminal.terminal_prefix(old, new)
    for mutation in ('scene', 'title', 'removed_visual', 'nonterminal'):
        a, b = deepcopy(old), deepcopy(new)
        if mutation == 'scene': b['scenes'][3]['narration'] = 'Other words.'
        elif mutation == 'title': b['title'] = 'Different'
        elif mutation == 'removed_visual': a['scenes'][-1]['visual'] = 'another'
        else: b['scenes'].pop(4)
        with pytest.raises(ValueError): terminal.terminal_prefix(a, b)


def test_retains_complete_original_pcm_prefix_without_mastering_or_removed_words(monkeypatch):
    monkeypatch.setattr(terminal, 'RATE', 100)
    old = timed(); pcm = bytes(range(250)) * 64
    joined, scenes, proof = terminal.retained_prefix(pcm, old)
    stop = proof['source_samples'][1]
    assert joined == pcm[:stop * 2]
    assert scenes[:37] == old['scenes'][:37]
    assert scenes[-1]['speech_end'] == old['scenes'][37]['speech_end']
    assert proof['final_gap_seconds'] == pytest.approx(1.)
    assert stop < round(old['scenes'][38]['speech_start'] * 100)
    assert len(scenes) == 38


def test_final_hold_adds_only_zero_samples_never_removed_scene_pcm(monkeypatch):
    monkeypatch.setattr(terminal, 'RATE', 100)
    old = timed(); old['scenes'][37]['sentences'][0]['audible_end'] = 75.
    old['scenes'][37]['speech_end'] = 75.8
    pcm = b'\x01\x02' * 8000
    joined, scenes, proof = terminal.retained_prefix(pcm, old)
    assert proof['final_silence_samples'] == 0
    # A valid shorter old hold requires padding, not copying any removed words.
    old['scenes'][37]['scene_end'] = 75.9
    old['scenes'][38]['speech_start'] = 75.9
    joined, scenes, proof = terminal.retained_prefix(pcm, old)
    assert proof['final_silence_samples'] == 10
    assert joined[:7590 * 2] == pcm[:7590 * 2]
    assert joined[7590 * 2:] == b'\0\0' * 10
    assert scenes[-1]['scene_end'] == 76.


@pytest.mark.parametrize('fault', ['missing', 'nonfinite', 'outside', 'full_speech', 'source_gap'])
def test_bad_original_audible_or_source_metadata_fails_closed(monkeypatch, fault):
    monkeypatch.setattr(terminal, 'RATE', 100)
    old = timed(); final = old['scenes'][37]
    if fault == 'missing': final['sentences'][0].pop('audible_end')
    elif fault == 'nonfinite': final['sentences'][0]['audible_end'] = float('nan')
    elif fault == 'outside': final['sentences'][0]['audible_end'] = 1000.
    elif fault == 'full_speech': final['speech_end'] = 76.; final['sentences'][0]['audible_end'] = 74.8
    else: old['scenes'][4]['speech_start'] += .1
    with pytest.raises(ValueError): terminal.retained_prefix(b'\x01\x02' * 8000, old)


def test_candidate_terminal_selection_is_only_home_and_no_parallel_changes():
    candidate.require_terminal_selection(['05_home'], ['05_home'], [], [], [], [], [])
    for args in [(['06_api'], ['06_api'], [], [], [], [], []),
                 (['05_home', '06_api'], ['05_home', '06_api'], [], [], [], [], []),
                 (['05_home'], ['05_home'], ['05_home'], [], [], [], []),
                 (['05_home'], ['05_home'], [], [], [], ['64_gate_editor'], [])]:
        with pytest.raises(ValueError): candidate.require_terminal_selection(*args)


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal, 'RATE', 100)
    monkeypatch.setitem(sys.modules, 'render_all_voices', SimpleNamespace(MAX_RELEASE_TRUE_PEAK_DBFS=-1.))
    monkeypatch.setitem(sys.modules, 'narration_audio', SimpleNamespace(ffmpeg_runtime_identity=lambda: {'decoder': 'fixture'}))
    old = timed(); pcm = b'\x01\x02' * 8000
    joined, scenes, intervals = terminal.retained_prefix(pcm, old)
    baseline = tmp_path / 'baseline'; baseline.mkdir(); stage = tmp_path / 'stage'; stage.mkdir()
    root = stage / 'production/05_home'; root.mkdir(parents=True)
    originals, rows = [], []
    for i in range(50):
        voice = f'voice{i}'; original = dict(language='en', voice=voice,
            audio=f'{voice}.m4a', timing=f'{voice}.json', audio_sha256='source-audio', timing_sha256='source-timing')
        terminal.write(baseline / original['timing'], old)
        work = root / terminal.WORK / 'en' / voice; work.mkdir(parents=True)
        (work / 'source.s16le').write_bytes(pcm); (work / 'prefix.s16le').write_bytes(joined)
        audio = root / 'audio/en' / (voice + '.m4a'); audio.parent.mkdir(parents=True, exist_ok=True); audio.write_bytes(b'output-' + voice.encode())
        metadata = terminal.timing_metadata(old, scenes, audio, len(joined) // 2)
        terminal.write(audio.with_suffix('.json'), metadata)
        originals.append(original)
        rows.append(dict(**original, source_pcm_sha256=terminal.digest(work / 'source.s16le'),
            source_decoded_samples=len(pcm) // 2, decoder={'decoder': 'fixture'}, prefix_pcm_sha256=terminal.digest(work / 'prefix.s16le'),
            intervals=intervals, codec_gain_db=0., codec_pcm_sha256=terminal.digest(work / 'prefix.s16le'), output_audio_sha256=terminal.digest(audio), output_timing_sha256=terminal.digest(audio.with_suffix('.json')),
            decoded_true_peak_dbfs=-1.5))
    source = dict(baseline_root=str(baseline), tracks=originals)
    monkeypatch.setattr(terminal, '_source', lambda *args: (source, {}, {}))
    record = dict(schema=1, kind=terminal.KIND, lesson='05_home', source=source,
        producer_sha256=terminal.digest(terminal.__file__), tracks=rows, recovery=None, resynthesized=False,
        current_runtime_freshness_claimed=False, fresh_capture=False, human_listening_review=False)
    terminal.write(root / terminal.RECORD, record)
    for row in rows:
        terminal.write(root / terminal.WORK / row['language'] / row['voice'] / 'track-proof.json',
            dict(producer_sha256=terminal.digest(terminal.__file__), source_sha256=terminal.object_digest(source),
                 recovered_original_producer=None, track=row))
    return stage, root, record


@pytest.mark.parametrize('fault', ['none', 'record', 'language', 'pcm', 'fingerprint', 'unknown_claim', 'codec_gain', 'journal'])
def test_checker_reconstructs_saved_pcm_and_rejects_tamper_without_decode(output, fault):
    stage, root, record = output
    if fault == 'record': record['resynthesized'] = True
    elif fault == 'unknown_claim': record['render_fingerprint'] = 'fake-fresh'
    elif fault == 'codec_gain': record['tracks'][0]['codec_gain_db'] = .25
    elif fault == 'journal': (root / terminal.WORK / 'en/voice0/track-proof.json').write_text('{}')
    elif fault == 'language': record['tracks'][0]['language'] = 'foreign'
    elif fault == 'pcm': (root / terminal.WORK / 'en/voice0/prefix.s16le').write_bytes(b'bad')
    elif fault == 'fingerprint':
        path = root / 'audio/en/voice0.json'; meta = terminal.read(path); meta['render_fingerprint'] = 'fake-fresh'; terminal.write(path, meta)
        record['tracks'][0]['output_timing_sha256'] = terminal.digest(path)
    terminal.write(root / terminal.RECORD, record)
    if fault == 'none':
        tracks, proofs = terminal.checked_terminal_narration(stage, decode=False)
        assert len(tracks) == len(proofs) == 50
        assert all('render_fingerprint' not in t for t in tracks)
    else:
        with pytest.raises(ValueError): terminal.checked_terminal_narration(stage, decode=False)


@pytest.mark.parametrize('fault', ['none', 'kind', 'baseline', 'audio_record'])
def test_candidate_requires_same_terminal_visual_join_before_any_codec(tmp_path, monkeypatch, fault):
    root = tmp_path / 'stage/production/05_home'; root.mkdir(parents=True)
    baseline = tmp_path / 'baseline'; baseline.mkdir()
    (root / terminal.RECORD).write_text('explicit-audio-proof')
    source = dict(baseline_root=str(baseline), baseline_manifest_sha256='original-manifest')
    visual = dict(**source, kind='terminal-only', terminal_scene_removal_record_sha256=terminal.digest(root / terminal.RECORD))
    if fault == 'kind': visual['kind'] = 'ordinary-native'
    elif fault == 'baseline': visual['baseline_root'] = str(tmp_path / 'foreign')
    elif fault == 'audio_record': visual['terminal_scene_removal_record_sha256'] = 'old-proof'
    terminal.write(root / 'published-native-edit.json', visual)
    calls = []
    monkeypatch.setitem(sys.modules, 'published_native_edit', SimpleNamespace(RECORD='published-native-edit.json',
        TERMINAL_KIND='terminal-only', checked_edited_native_timing=lambda *args: calls.append(args) or 'checked-path'))
    if fault == 'none':
        assert candidate.require_terminal_visual_join(tmp_path / 'stage', baseline, source) == 'checked-path'
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError): candidate.require_terminal_visual_join(tmp_path / 'stage', baseline, source)
        assert calls == []


def test_normal_offering_sets_allow_published_order_but_not_drop_or_duplicates():
    normal = {'en': ['af_heart', 'af_aoede'], 'es': ['ef_dora']}
    published = {'en': ['af_aoede', 'af_heart'], 'es': ['ef_dora']}
    terminal.require_voice_inventory(published, normal)
    for bad in ({'en': ['af_heart'], 'es': ['ef_dora']},
                {'en': ['af_aoede', 'af_heart', 'af_heart'], 'es': ['ef_dora']},
                {'en': ['af_aoede', 'af_heart'], 'fr': ['ff_siwis']}):
        with pytest.raises(ValueError): terminal.require_voice_inventory(bad, normal)


def test_codec_headroom_is_only_bounded_uniform_gain():
    import numpy as np
    original = np.array([-30000, -123, 0, 123, 30000], dtype='<i2').tobytes()
    assert terminal.codec_pcm(original, 0.) is original
    adjusted = np.frombuffer(terminal.codec_pcm(original, -.25), dtype='<i2')
    assert adjusted.tolist() == np.rint(np.array([-30000, -123, 0, 123, 30000]) * 10 ** (-.25 / 20)).tolist()
    for gain in (.25, -1.25, None):
        with pytest.raises(ValueError): terminal.codec_pcm(original, gain)


def test_codec_retry_preserves_failed_output_and_original_prefix(tmp_path, monkeypatch):
    pcm = tmp_path / 'prefix.s16le'; pcm.write_bytes(b'\x01\x02' * 12)
    audio = tmp_path / 'result.m4a'; calls = []
    def encode(command, **kwargs):
        calls.append(command); Path(command[-1]).write_bytes(('attempt' + str(len(calls))).encode())
    monkeypatch.setattr(terminal.subprocess, 'run', encode)
    peaks = iter([-.9, -1.2]); renderer = SimpleNamespace(true_peak_dbfs=lambda p: next(peaks), MAX_RELEASE_TRUE_PEAK_DBFS=-1.)
    before = pcm.read_bytes(); gain, encoded, peak = terminal.encode_prefix(pcm, audio, tmp_path, renderer)
    assert gain == -.25 and peak == -1.2
    assert audio.read_bytes() == b'attempt2'
    assert (tmp_path / 'codec-uncompensated.m4a').read_bytes() == b'attempt1'
    assert pcm.read_bytes() == before
    assert encoded.read_bytes() == terminal.codec_pcm(before, -.25)
    assert len(calls) == 2 and all('-af' not in c for c in calls)


def test_missing_peak_is_not_an_excuse_to_apply_gain(tmp_path, monkeypatch):
    pcm = tmp_path / 'prefix.s16le'; pcm.write_bytes(b'\x01\x02' * 12); audio = tmp_path / 'result.m4a'
    monkeypatch.setattr(terminal.subprocess, 'run', lambda command, **kwargs: Path(command[-1]).write_bytes(b'first'))
    renderer = SimpleNamespace(true_peak_dbfs=lambda p: None, MAX_RELEASE_TRUE_PEAK_DBFS=-1.)
    with pytest.raises(ValueError, match='unavailable'): terminal.encode_prefix(pcm, audio, tmp_path, renderer)
    assert audio.read_bytes() == b'first' and not (tmp_path / 'codec-uncompensated.m4a').exists()


@pytest.fixture
def legacy_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal, 'RATE', 100)
    stage = tmp_path / 'stage'; root = stage / 'production/05_home'; root.mkdir(parents=True)
    baseline = tmp_path / 'baseline'; baseline.mkdir(); source = dict(baseline_root=str(baseline), tracks=[])
    pcm = b'\x01\x02' * 8000
    names = [f'v{i:02}' for i in range(12)] + ['am_michael']
    hashes = {}
    for voice in names:
        old = timed(); old['voice'] = voice
        original = dict(language='en', voice=voice, audio=f'{voice}.m4a', timing=f'{voice}.json',
                        audio_sha256='original-' + voice, timing_sha256='timing-' + voice)
        source['tracks'].append(original); terminal.write(baseline / original['timing'], old)
        (baseline / original['audio']).write_bytes(b'original-aac')
        joined, scenes, proof = terminal.retained_prefix(pcm, old)
        work = root / terminal.WORK / 'en' / voice; work.mkdir(parents=True)
        (work / 'source.s16le').write_bytes(pcm); (work / 'prefix.s16le').write_bytes(joined)
        audio = root / 'audio/en' / (voice + '.m4a'); audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(('completed-' + voice).encode())
        if voice != 'am_michael': terminal.write(audio.with_suffix('.json'), terminal.timing_metadata(old, scenes, audio, len(joined) // 2))
        hashes[str(audio)] = terminal.digest(audio)
    monkeypatch.setattr(terminal, 'LEGACY_FAILED_SHA', terminal.digest(root / 'audio/en/am_michael.m4a'))
    log = tmp_path / 'failed.log'; log.write_text(str(stage) + '\n' + str(baseline) + '\nNormal final true-peak check failed\nTerminal exit status: 1\n')
    monkeypatch.setitem(sys.modules, 'narration_audio', SimpleNamespace(ffmpeg_runtime_identity=lambda: {'fixture': 'decoder'}))
    return stage, root, source, log, hashes


def test_strict_legacy_preflight_verifies13_and_preserves12_without_encoding(legacy_partial, monkeypatch):
    stage, root, source, log, hashes = legacy_partial; decodes = []
    monkeypatch.setattr(terminal, 'compare_source_decode', lambda audio, pcm: decodes.append((audio, pcm)))
    renderer = SimpleNamespace(true_peak_dbfs=lambda p: -2., MAX_RELEASE_TRUE_PEAK_DBFS=-1.)
    rows, recovery = terminal.recover_legacy(stage, source, log, renderer)
    assert len(rows) == 12 and len(decodes) == 13
    assert recovery['retained_complete_tracks'] == 12 and len(recovery['verified_source_decodes']) == 13
    assert all(terminal.digest(Path(p)) == h for p, h in hashes.items())
    assert not list(root.rglob('track-proof.json'))


@pytest.mark.parametrize('fault', ['unknown_output', 'pcm_mismatch', 'source_decode'])
def test_legacy_unknown_or_mismatched_partials_fail_before_any_rewrite(legacy_partial, monkeypatch, fault):
    stage, root, source, log, hashes = legacy_partial
    if fault == 'unknown_output': (root / 'audio/en/foreign.m4a').write_bytes(b'unknown')
    elif fault == 'pcm_mismatch': (root / terminal.WORK / 'en/v02/prefix.s16le').write_bytes(b'changed')
    def compare(audio, pcm):
        if fault == 'source_decode': raise ValueError('wrong source decode')
    monkeypatch.setattr(terminal, 'compare_source_decode', compare)
    renderer = SimpleNamespace(true_peak_dbfs=lambda p: -2., MAX_RELEASE_TRUE_PEAK_DBFS=-1.)
    with pytest.raises(ValueError): terminal.recover_legacy(stage, source, log, renderer)
    assert all(terminal.digest(Path(p)) == h for p, h in hashes.items())
    assert not list(root.rglob('track-proof.json'))
