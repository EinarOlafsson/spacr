from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import time

from PIL import Image

sys.path.insert(0, 'tools/tutorials/authoring/tools')
from render_visual_master import frame_aligned_durations, spotlight

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-home-measure-ui-current-r1'
deadline = time.monotonic() + 3600
reports = {}
for identity in ('05_home', '08_measure'):
    folder = stage / 'production' / identity
    while not (folder / 'current-video-acceptance.json').is_file():
        assert time.monotonic() < deadline, 'No completed video receipt; nothing accepted'
        time.sleep(15)
    receipt = json.loads((folder / 'current-video-acceptance.json').read_text())
    assert receipt['master_full_decode_passed']
    visual = json.loads((folder / 'visual.json').read_text())
    timing = json.loads((folder / 'audio/en/af_heart.json').read_text())
    durations = frame_aligned_durations(timing['scenes'], 30)
    frames = []
    position = 0
    for duration in durations:
        count = round(duration * 30)
        frames.append(position + count // 2)
        position += count
    out = stage / 'frame-review' / identity
    out.mkdir(parents=True, exist_ok=True)
    video = stage / 'web-renditions' / identity / 'video' / (identity + '_silent.mp4')
    selection = '+'.join(f'eq(n,{frame})' for frame in frames)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-filter_threads', '2', '-i', str(video),
                    '-vf', 'select=' + selection.replace(',', '\\,'), '-vsync', '0',
                    str(out / 'scene-%02d.png')], check=True)
    decoded = sorted(out.glob('scene-*.png'))
    assert len(decoded) == len(visual['scenes'])
    records = []
    for index, (scene, frame) in enumerate(zip(visual['scenes'], decoded)):
        source = (folder / scene['image']).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == scene['capture_sha256']
        assert scene.get('pointer') is False
        expected = Image.open(source).convert('RGBA')
        if scene.get('focus'):
            spotlight(expected, scene['focus'])
        size = Image.open(frame).size
        expected = expected.convert('RGB').resize(size, Image.Resampling.LANCZOS)
        comparison = out / f'expected-{index + 1:02d}.png'
        expected.save(comparison)
        result = subprocess.run(['ffmpeg', '-v', 'info', '-threads', '2', '-filter_complex_threads', '2', '-i', str(frame),
                                 '-i', str(comparison), '-lavfi', 'ssim', '-frames:v', '1',
                                 '-f', 'null', '-'], check=True, capture_output=True, text=True)
        match = re.search(r'SSIM .*All:([0-9.]+)', result.stderr)
        assert match, result.stderr[-2000:]
        score = float(match.group(1))
        assert score > .97, (identity, index + 1, score, 'decoded scene does not match current native capture')
        records.append({'scene': index + 1, 'video_frame': frames[index],
                        'native_capture_sha256': scene['capture_sha256'],
                        'decoded_frame_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
                        'native_spotlight_downsample_ssim': score})
    reports[identity] = {'passed': True, 'scenes': records, 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest()}
    print(identity, len(records), 'decoded scenes agree with their current native captures', flush=True)
(stage / 'current-frame-fidelity.json').write_text(json.dumps({'passed': True, 'lessons': reports}, indent=2) + '\n')
