"""Downscaling must preserve the shared video's frame clock in every language."""
from copy import deepcopy
from pathlib import Path
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
