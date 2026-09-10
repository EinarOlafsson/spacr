"""Whole-media retention rejects overrides and mismatched old master timing."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_retained_media as retained


def test_no_override_preserves_original_paths(tmp_path, monkeypatch):
    original, stage = tmp_path / 'original', tmp_path / 'stage'
    media = original / 'production/lesson'
    (media / 'video').mkdir(parents=True)
    (media / 'video/lesson_silent.mp4').write_bytes(b'original video')
    (media / 'poster.jpg').write_bytes(b'original poster')
    calls = []
    def sources(*args, **kwargs):
        calls.append((args, kwargs))
        return {'tracks': [], 'catalogs': []}
    monkeypatch.setattr(retained, 'retained_sources', sources)
    result = retained.retained_media_sources(stage, original, tmp_path / 'baseline', 'lesson', {})
    assert result['media']['video']['path'] == str(media / 'video/lesson_silent.mp4')
    assert result['media_regenerated'] is False and result['media_copied'] is False
    assert calls[0][1] == {'require_staged': False}
    override = stage / 'production/lesson'
    override.mkdir(parents=True)
    (override / 'poster.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='staged production override'):
        retained.retained_media_sources(stage, original, tmp_path / 'baseline', 'lesson', {})


@pytest.fixture
def probe():
    return {'streams': [{'codec_type': 'video', 'width': 3840, 'height': 2160,
                         'r_frame_rate': '30/1', 'duration': '50.2'}]}


def test_matching_silent_master(probe):
    retained.require_master(probe, 50.205)


@pytest.mark.parametrize('key,value', [('width', 1920), ('height', 1080),
                                      ('duration', '51'), ('duration', 'nan'),
                                      ('codec_type', 'audio'), ('r_frame_rate', '24/1')])
def test_wrong_master_refused(probe, key, value):
    probe['streams'][0][key] = value
    with pytest.raises(ValueError, match='4K/30'):
        retained.require_master(probe, 50.205)


def test_audio_stream_is_not_a_silent_master(probe):
    probe['streams'].append({'codec_type': 'audio'})
    with pytest.raises(ValueError, match='one silent video stream'):
        retained.require_master(probe, 50.205)
