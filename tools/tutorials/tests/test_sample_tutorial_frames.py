"""Offline tests for the item-447 sampled-frame path and theme sweep (no OCR model)."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'sample_tutorial_frames.py'
spec = importlib.util.spec_from_file_location('sample_tutorial_frames', MODULE)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.mark.parametrize('text, kind', [
    ('Output directory 2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09/s', 'maintainer_project'),
    ('Output directory 2/Claude/toxoplasma projects/tutorials', 'maintainer_project'),
    ('/tmp/x/refresh 2026-09-09/', 'refresh_stage'),
    ('Look in: /home/someone/data', 'unix_home'),
    ('/mnt/disk/data', 'mounted_volume'),
    ('/Users/me/Desktop', 'macos_users'),
    ('C:\\Users\\me', 'windows_drive'),
])
def test_path_patterns_catch_local_paths_even_with_ocr_spacing(text, kind):
    assert kind in {k for k, _ in sweep.path_hits([text])}


@pytest.mark.parametrize('text', [
    '/tmp/spacr-tutorials/profiler_runs/SYNTHETIC-OLS-x/example/SYNTHETIC_OLS_coefficients.csv',
    'Control gRNA/Gene 000000', 'Look in: /tmp/spacr-code', 'Home', 'input_a: 0 -> 1'])
def test_neutral_paths_and_gui_text_are_not_flagged(text):
    assert sweep.path_hits([text]) == []


def test_theme_separates_dark_and_light_frames():
    assert sweep.theme(Image.new('RGB', (64, 36), (20, 18, 30)))['dark'] is True
    assert sweep.theme(Image.new('RGB', (64, 36), (240, 240, 240)))['dark'] is False


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg unavailable')
def test_sweep_item_samples_poster_and_frames_and_reports_hits(tmp_path):
    poster = tmp_path / 'poster.jpg'
    Image.new('RGB', (64, 36), (10, 10, 10)).save(poster)
    video = tmp_path / 'lesson.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=white:s=64x36:d=2:r=5', '-pix_fmt', 'yuv420p', str(video)], check=True)
    seen = []

    def fake_ocr(image):
        seen.append(image.size)
        return ['/home/someone/project'] if len(seen) == 1 else ['Regression']

    item = {'lesson': 'x', 'poster': str(poster), 'video': str(video), 'video_source': 'explicit'}
    result = sweep.sweep_item(item, frames=3, ocr=fake_ocr)
    assert len(result['images']) == 4 and len(seen) == 4 and not result['errors']
    assert result['path_offender'] is True
    assert result['light_images'] == [r['image'] for r in result['images'][1:]]
    receipt = sweep.summarize([result], ocr_used=True)
    assert receipt['path_offenders'] == ['x'] and receipt['light_frame_lessons'] == ['x']


@pytest.mark.parametrize('text', ['https://github.com/EinarOlafsson/spacr', 'rolafsson.github.io/spacr/',
                                  'https://einarolafsson.glthub.io/spacr/api/index.html', 'EinarOlafsson/ spacr',
                                  'Author: Einar Birnir Olafsson', 'b.com/EinarOlafsson/'])
def test_public_project_urls_are_not_account_leaks(text):
    assert sweep.path_hits([text]) == []


@pytest.mark.parametrize('text', ['olafsson', 'Look in: /home/olafsson', '/tmp/pytest-of-olafsson/x',
                                  'https://github.com/EinarOlafsson/spacr and olafsson'])
def test_account_name_outside_public_urls_is_flagged(text):
    assert 'maintainer_account' in {k for k, _ in sweep.path_hits([text])}


def test_generic_sandbox_home_is_reported_apart_from_offenders():
    hits = [{'kind': k, 'text': t} for k, t in sweep.path_hits(['Launcher:/home/user/.local/bin/spacr'])]
    assert [h['kind'] for h in hits] == ['generic_home'] and sweep.offending(hits) == []
    assert sweep.offending([{'kind': k, 'text': t} for k, t in
                            sweep.path_hits(['/home/olafsson/.cache/spacr'])])
