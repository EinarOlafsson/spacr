"""Real tiny videos keep frame clocks through scene joins and web encoding."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('psf_visual_renderer', TOOLS / 'render_visual_master.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def _times(path):
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time', '-of', 'json', str(path),
    ]))
    return sorted(float(packet['pts_time']) for packet in result['packets'])


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                    reason='FFmpeg tools are required for actual video timing')
def test_scene_joins_preserve_frame_timestamps_through_normal_web_encoding(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (64, 64), '#4080c0').save(image)
    parts = []
    counts = [10, 20, 10, 14]
    for index, frames in enumerate(counts):
        part = tmp_path / f'part{index}.mp4'
        renderer.encode_still(image, frames / 30, part, 30)
        parts.append(part)
    master = tmp_path / 'master.mp4'
    renderer.concat(parts, master, tmp_path / 'concat.txt')
    master_times = _times(master)
    assert len(master_times) == sum(counts)
    assert master_times == pytest.approx([i / 30 for i in range(sum(counts))], abs=1 / 90000)

    sys.path.insert(0, str(TOOLS.parent.parent))
    try:
        import stage_web_renditions
        publisher_spec = importlib.util.spec_from_file_location('psf_test_publisher', TOOLS / 'publish_tutorials.py')
        publisher = importlib.util.module_from_spec(publisher_spec)
        publisher_spec.loader.exec_module(publisher)
        output = tmp_path / 'web.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(master),
                        *publisher.ENCODE_ARGS, str(output)], check=True)
        stage_web_renditions.require_timestamps(master_times, _times(output), sum(counts))
    finally:
        sys.path.pop(0)


def test_many_scene_listing_keeps_every_start_on_the_exact_frame_clock():
    from fractions import Fraction
    # 61 scenes of n/30 s, none a whole microsecond, like a long native lesson.
    counts = [451 + 3 * index + 1 for index in range(61)]
    durations = renderer.concat_durations([(count, Fraction(30)) for count in counts])
    start = Fraction(0)
    exact = Fraction(0)
    for count, text in zip(counts, durations):
        start += Fraction(text)
        exact += Fraction(count, 30)
        # The demuxer reads whole microseconds; each running start must stay
        # within half a microsecond of the frame clock (one 1/90000 tick is 11 us).
        assert Fraction(text) * 1_000_000 == int(Fraction(text) * 1_000_000)
        assert abs(start - exact) <= Fraction(1, 2_000_000)
    with pytest.raises(ValueError):
        renderer.concat_durations([(0, Fraction(30))])
