from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sentence_audio_cache import cached_sentence


def test_sentence_reuse_is_bound_to_speech_voice_speed_and_verified_pcm(tmp_path):
    calls = []
    def synthesize():
        calls.append(True)
        return np.linspace(-.1, .1, 1000, dtype=np.float32), 'lˈIv'
    inputs = {'speech_text': '[live](/lˈIv/)', 'voice': 'af_heart', 'speed': 1,
              'runtime': {'model_sha256': 'pinned'}}
    first, phonemes, reused = cached_sentence(tmp_path, inputs, synthesize, np)
    second, _, reused_second = cached_sentence(tmp_path, inputs, synthesize, np)
    assert not reused and reused_second and len(calls) == 1
    assert np.array_equal(first, second) and phonemes == 'lˈIv'
    for key, value in [('speech_text', 'different'), ('voice', 'af_sky'), ('speed', .9),
                       ('runtime', {'model_sha256': 'changed'})]:
        assert not cached_sentence(tmp_path, {**inputs, key: value}, synthesize, np)[2]
    original = next(path for path in tmp_path.glob('*.json') if inputs['speech_text'] in path.read_text())
    original.with_suffix('.npy').write_bytes(b'corrupt')
    assert not cached_sentence(tmp_path, inputs, synthesize, np)[2]
