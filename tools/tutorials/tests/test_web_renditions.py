"""Downscaling must preserve the shared video's frame clock in every language."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_web_renditions import require_timestamps, require_video


@pytest.fixture
def videos():
    source = {'streams': [{'codec_type': 'video', 'width': 3840, 'height': 2160,
                           'r_frame_rate': '30/1', 'nb_frames': '3'}]}
    target = deepcopy(source)
    target['streams'][0].update(width=2560, height=1440)
    return source, target


def test_actual_scaled_dimensions_and_same_clock_pass(videos):
    require_video(*videos)
    require_timestamps([0., 1/30, 2/30], [0., .033333, .066667], 3)


@pytest.mark.parametrize('change', [
    lambda p: p['streams'].append({'codec_type': 'audio'}),
    lambda p: p['streams'][0].update(width=1920),
    lambda p: p['streams'][0].update(codec_type='audio'),
    lambda p: p['streams'][0].update(nb_frames='2'),
    lambda p: p['streams'][0].update(r_frame_rate='24/1'),
])
def test_changed_streams_or_geometry_fail(videos, change):
    source, target = videos
    change(target)
    with pytest.raises(ValueError):
        require_video(source, target)


@pytest.mark.parametrize('times', [[0., 1/30], [0., 0., 2/30],
    [0., .034, 2/30], [0., float('nan'), 2/30], [0., float('inf'), 2/30]])
def test_missing_duplicated_shifted_or_invalid_times_fail(times):
    with pytest.raises(ValueError):
        require_timestamps([0., 1/30, 2/30], times, 3)


def test_encoded_video_has_frequent_seek_points_without_changing_frame_clock(tmp_path):
    """Narration corrections must not decode eight seconds of preceding video."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('The actual video encoding contract requires ffmpeg and ffprobe')
    path = Path(__file__).resolve().parents[1] / 'authoring/tools/publish_tutorials.py'
    spec = importlib.util.spec_from_file_location('tutorial_publisher', path)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    video = tmp_path / 'seekable.mp4'
    subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=320x180:rate=30:duration=6.1',
        '-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS, str(video),
    ], check=True, capture_output=True, timeout=30)
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=key_frame,best_effort_timestamp_time',
        '-of', 'json', str(video),
    ], check=True, capture_output=True, text=True, timeout=30)
    frames = json.loads(result.stdout)['frames']
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in frames]
    require_timestamps([index / 30 for index in range(183)], timestamps, 183)
    keyframes = [float(frame['best_effort_timestamp_time']) for frame in frames
                 if frame['key_frame']]
    assert keyframes[0] == 0
    boundaries = [*keyframes, timestamps[-1] + 1 / 30]
    assert max(right - left for left, right in zip(boundaries, boundaries[1:])) <= 2.000001


def test_only_the_pathway_overviews_use_an_exact_1080p_web_copy(videos):
    from types import SimpleNamespace
    from stage_web_renditions import REDUCED_WEB_DIMENSIONS, encoder_arguments, web_dimensions
    assert set(REDUCED_WEB_DIMENSIONS) == {'78_spacr_screens', '79_module_inputs_outputs',
                                           '80_image_analysis_pathways', '81_sequencing_pathways'}
    assert web_dimensions('79_module_inputs_outputs') == (1920, 1080)
    assert web_dimensions('07_mask') == (2560, 1440)
    source, target = videos
    reduced = deepcopy(target)
    reduced['streams'][0].update(width=1920, height=1080)
    require_video(source, reduced, web_dimensions('79_module_inputs_outputs'))
    for lesson, copy in (('79_module_inputs_outputs', target), ('07_mask', reduced)):
        with pytest.raises(ValueError):
            require_video(source, copy, web_dimensions(lesson))
    publisher = SimpleNamespace(ENCODE_ARGS=['-vf', "scale='min(2560,iw)':-2", '-g', '60'])
    assert encoder_arguments(publisher, '80_image_analysis_pathways') == ['-vf', "scale='min(1920,iw)':-2", '-g', '60']
    assert encoder_arguments(publisher, '07_mask') == publisher.ENCODE_ARGS
