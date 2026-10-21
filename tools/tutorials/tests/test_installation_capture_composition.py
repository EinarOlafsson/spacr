"""Distinct installation evidence roles remain source-bound and immutable."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compose_installation_capture import compose


def fixture(tmp_path):
    source = tmp_path / 'source/captures/installer'
    source.mkdir(parents=True)
    image = source / 'frame.png'
    Image.new('RGB', (3840, 2160), 'black').save(image)
    frame = {'image': image.name, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    (source / 'frames.json').write_text(json.dumps({'original': frame}), encoding="utf-8")
    proof = {'completed_capture': True, 'installed_identity': {
        'version': '1.5.1.3', 'package': '/home/user/env/site-packages/spacr/__init__.py'}}
    (source / 'provenance.json').write_text(json.dumps(proof), encoding="utf-8")
    lesson = {'id': '02_install_spacr', 'scenes': [{'visual': 'current', 'narration': 'Published package output.'}]}
    digest = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    plan = {'lesson': lesson['id'], 'english_sha256': digest, 'scenes': {'current': {
        'capture': str(source), 'source_visual': 'original', 'kind': 'published_installation'}}}
    return source, lesson, plan


def test_composition_retains_original_native_pixels_and_public_identity(tmp_path):
    source, lesson, plan = fixture(tmp_path)
    original = (source / 'frame.png').read_bytes()
    output = tmp_path / 'result'
    compose(lesson, plan, output)
    assert (output / 'current.png').read_bytes() == original
    assert (source / 'frame.png').read_bytes() == original
    proof = json.loads((output / 'provenance.json').read_text(encoding="utf-8"))
    assert proof['not_one_application_session']
    assert proof['frame_origins']['current']['installed_identity']['version'] == '1.5.1.3'


@pytest.mark.parametrize('failure', ['source', 'bytes', 'incomplete', 'identity', 'card', 'nightly'])
def test_invalid_role_or_source_is_rejected_before_writing(tmp_path, failure):
    source, lesson, plan = fixture(tmp_path)
    if failure == 'source': lesson['scenes'][0]['narration'] = 'Changed narration.'
    elif failure == 'bytes': (source / 'frame.png').write_bytes(b'changed')
    elif failure in ('incomplete', 'identity'):
        proof = json.loads((source / 'provenance.json').read_text(encoding="utf-8"))
        if failure == 'incomplete': proof['completed_capture'] = False
        else: proof.pop('installed_identity')
        (source / 'provenance.json').write_text(json.dumps(proof), encoding="utf-8")
    elif failure == 'card': plan['scenes']['current']['kind'] = 'reference_guidance'
    else: plan['scenes']['current']['kind'] = 'current_nightly'
    output = tmp_path / 'result'
    with pytest.raises(ValueError): compose(lesson, plan, output)
    assert not output.exists()


def test_current_ui_keeps_feature_narration_and_requires_verified_clip(tmp_path):
    source, lesson, plan = fixture(tmp_path)
    video = source / 'clip.mp4'
    video.write_bytes(b'fixture-only-video')
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    receipt = source / 'clip.capture.json'
    receipt.write_text(json.dumps({'sha256': video_sha, 'full_decode_passed': True,
                                  'audio_streams': 0, 'size': [3840, 2160]}))
    frames = json.loads((source / 'frames.json').read_text())
    frames['original']['clip'] = {'video': video.name, 'sha256': video_sha,
        'receipt': receipt.name, 'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    (source / 'frames.json').write_text(json.dumps(frames))
    plan['scenes']['current']['kind'] = 'current_nightly'
    output = tmp_path / 'result'
    compose(lesson, plan, output)
    assert (output / 'current.mp4').read_bytes() == video.read_bytes()
    assert lesson['scenes'][0]['narration'] == 'Published package output.'
    proof = json.loads((output / 'provenance.json').read_text())
    assert proof['frame_origins']['current']['kind'] == 'current_nightly'
