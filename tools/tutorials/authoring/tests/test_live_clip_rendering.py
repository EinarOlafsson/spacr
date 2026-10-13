"""Actual footage must retain elapsed timing and verified capture evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import render_visual_master as renderer


def verified_clip(tmp_path,duration=2.0):
    video=tmp_path/'actual.mp4';video.write_bytes(b'captured-video-test-double')
    receipt=tmp_path/'actual.capture.json'
    receipt.write_text(json.dumps({'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
                                  'full_decode_passed':True,'audio_streams':0,
                                  'duration':duration}))
    return {'video':video.name,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
            'receipt':receipt.name,'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}


def test_live_render_trims_original_elapsed_time_without_loop_or_speed_change(tmp_path,monkeypatch):
    clip=verified_clip(tmp_path);commands=[];monkeypatch.setattr(renderer,'run',commands.append)
    renderer.encode_live_clip(clip,tmp_path,1.25,tmp_path/'output.mp4',30)
    command=commands[0]
    assert command[command.index('-t')+1]=='1.250000'
    assert command[command.index('-vf')+1]=='fps=30'
    assert '-stream_loop' not in command and '-loop' not in command
    assert not any('setpts' in item or 'tpad' in item for item in command)


def test_missing_genuine_capture_time_requires_recapture(tmp_path,monkeypatch):
    clip=verified_clip(tmp_path,duration=1);monkeypatch.setattr(renderer,'run',lambda _:pytest.fail('must fail before encoding'))
    with pytest.raises(ValueError,match='record a longer genuine hold'):
        renderer.encode_live_clip(clip,tmp_path,1.1,tmp_path/'output.mp4',30)


@pytest.mark.parametrize('changed',['video','receipt'])
def test_changed_capture_bytes_fail_before_encoding(tmp_path,monkeypatch,changed):
    clip=verified_clip(tmp_path);monkeypatch.setattr(renderer,'run',lambda _:pytest.fail('must fail before encoding'))
    (tmp_path/clip[changed]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        renderer.encode_live_clip(clip,tmp_path,1,tmp_path/'output.mp4',30)
