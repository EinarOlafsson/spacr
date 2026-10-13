from pathlib import Path
import hashlib
import json
import subprocess
import sys

from PIL import Image

sys.path.insert(0, 'tools/tutorials/authoring/tools')
from render_visual_master import encode_still, frame_aligned_durations, spotlight

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r2'
reports = {}
for identity in ('01_pypi_github', '03_pip_install', '04_platform_installers'):
    folder = stage / 'production' / identity
    receipt = json.loads((folder / 'current-video-acceptance.json').read_text())
    assert receipt['master_full_decode_passed']
    visual = json.loads((folder / 'visual.json').read_text())
    timing = json.loads((folder / 'audio/en/af_heart.json').read_text())
    durations = frame_aligned_durations(timing['scenes'], 30)
    frames = []
    position = 0
    for duration in durations:
        assert duration >= 2
        frames.append(position)
        position += round(duration * 30)
    out = stage / 'frame-review' / identity
    out.mkdir(parents=True, exist_ok=True)
    video = folder / 'video' / (identity + '_silent.mp4')
    selection = '+'.join(f'eq(n,{frame})' for frame in frames)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-filter_threads', '2', '-i', str(video),
                    '-vf', 'select=' + selection.replace(',', '\\,'), '-vsync', '0',
                    str(out / 'master-start-%02d.png')], check=True)
    decoded = sorted(out.glob('master-start-*.png'))
    assert len(decoded) == len(visual['scenes'])
    records = []
    for index, (scene, frame) in enumerate(zip(visual['scenes'], decoded)):
        source = (folder / scene['image']).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == scene['capture_sha256']
        assert scene.get('pointer') is False
        expected = Image.open(source).convert('RGBA')
        if scene.get('focus'):
            spotlight(expected, scene['focus'])
        native = out / f'coded-reference-native-{index + 1:02d}.png'
        expected.convert('RGB').save(native)
        reference_video = out / f'coded-reference-{index + 1:02d}.mp4'
        encode_still(native, 2, reference_video, 30)
        reference_frame = out / f'coded-reference-{index + 1:02d}.png'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '2', '-i', str(reference_video),
                        '-frames:v', '1', str(reference_frame)], check=True)
        actual_image = Image.open(frame).convert('RGB')
        reference_image = Image.open(reference_frame).convert('RGB')
        assert actual_image.size == reference_image.size == (3840, 2160)
        actual_pixels = actual_image.tobytes()
        reference_pixels = reference_image.tobytes()
        assert actual_pixels == reference_pixels, (identity, index + 1, '4K scene does not match its source-bound native capture under the approved codec')
        records.append({'scene': index + 1, 'video_frame': frames[index],
                        'native_capture_sha256': scene['capture_sha256'],
                        'decoded_frame_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
                        'actual_decoded_pixels_sha256': hashlib.sha256(actual_pixels).hexdigest(),
                        'reference_decoded_pixels_sha256': hashlib.sha256(reference_pixels).hexdigest(),
                        'all_decoded_pixels_identical': True})
    reports[identity] = {'passed': True, 'scenes': records, 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest()}
    print(identity, len(records), '4K scene-start frames match every native capture pixel under the approved codec', flush=True)
(stage / 'current-frame-fidelity.json').write_text(json.dumps(
    {'passed': True, 'scope': 'All 30 current installation 4K scene-start frames match exact decoded pixels from source-bound native captures composed and encoded with the approved renderer. Web geometry, timestamps, full decode and playback have separate normal receipts.',
     'lessons': reports}, indent=2) + '\n')
