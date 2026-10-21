"""Affected poster contract: real first frame, ordinary images, fail-closed reuse."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

from PIL import Image
import pytest


@pytest.fixture
def poster_tool():
    path = Path(__file__).resolve().parents[1] / 'authoring/tools/prepare_web_media.py'
    # Private draft cohort lives beside its production draft; repo test is portable.
    if not path.is_file():
        path = Path(__file__).with_name('prepare_web_media.py')
    spec = importlib.util.spec_from_file_location('poster_contract_under_test', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


@pytest.fixture
def stage(tmp_path, poster_tool, monkeypatch):
    identity = '04_platform_installers'
    root = tmp_path / 'production' / identity
    (root / 'video').mkdir(parents=True)
    monkeypatch.setattr(poster_tool, 'PRODUCTION', tmp_path / 'production')
    monkeypatch.setattr(sys, 'argv', ['prepare_web_media.py', '--lessons', identity])
    return tmp_path, root, identity


def test_reuse_cli_poster_is_actual_first_decoded_frame(poster_tool, stage, monkeypatch):
    if not shutil.which('ffmpeg'):
        pytest.skip('Requires normal ffmpeg decoder')
    base, root, identity = stage
    (root / 'scenes.json').write_text(json.dumps({'kind': 'published_native_intervals',
                                               'scenes': [{'visual': 'held_card'}]}))
    master = root / 'video' / (identity + '_silent.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=red:s=64x48:r=30:d=0.1', '-an', '-c:v', 'libx264',
                    '-threads', '1', '-pix_fmt', 'yuv420p', str(master)], check=True, timeout=30)
    checked = []
    def validate(stage_path, lesson):
        checked.append((stage_path, lesson))
        return root / 'video/native-live-timings.json'
    monkeypatch.setitem(sys.modules, 'published_native_reuse',
                        types.SimpleNamespace(checked_published_native_timing=validate))
    assert poster_tool.main() == 0
    assert checked == [(base, identity)]
    with Image.open(root / 'poster.jpg') as poster:
        assert poster.size == (64, 48)
        red, green, blue = poster.convert('RGB').getpixel((32, 24))
        assert red > 230 and green < 20 and blue < 20


def test_ordinary_cli_keeps_existing_image_poster_route(poster_tool, stage, monkeypatch):
    _, root, identity = stage
    Image.new('RGB', (40, 30), 'blue').save(root / 'actual.png')
    (root / 'scenes.json').write_text(json.dumps({'scenes': [{'image': 'actual.png'}]}))
    (root / 'video' / (identity + '_silent.mp4')).write_bytes(b'not decoded on ordinary route')
    def forbid(*args, **kwargs):
        raise AssertionError('Ordinary route must not decode a master or invoke reuse validation')
    monkeypatch.setattr(poster_tool.subprocess, 'run', forbid)
    monkeypatch.setitem(sys.modules, 'published_native_reuse',
                        types.SimpleNamespace(checked_published_native_timing=forbid))
    assert poster_tool.main() == 0
    with Image.open(root / 'poster.jpg') as poster:
        assert poster.size == (40, 30)
        red, green, blue = poster.convert('RGB').getpixel((20, 15))
        assert blue > 230 and red < 20 and green < 20


def test_invalid_reuse_binding_fails_before_extraction_or_poster_write(poster_tool, stage, monkeypatch):
    _, root, identity = stage
    (root / 'scenes.json').write_text(json.dumps({'kind': 'published_native_intervals',
                                               'scenes': [{'visual': 'held_card'}]}))
    (root / 'video' / (identity + '_silent.mp4')).write_bytes(b'wrong master')
    original = b'preserved existing poster'
    (root / 'poster.jpg').write_bytes(original)
    def reject(*args):
        raise ValueError('Published native source differs from its verified manifest')
    def forbid(*args, **kwargs):
        raise AssertionError('Unvalidated media must never be decoded')
    monkeypatch.setitem(sys.modules, 'published_native_reuse',
                        types.SimpleNamespace(checked_published_native_timing=reject))
    monkeypatch.setattr(poster_tool.subprocess, 'run', forbid)
    with pytest.raises(ValueError, match='verified manifest'):
        poster_tool.main()
    assert (root / 'poster.jpg').read_bytes() == original
