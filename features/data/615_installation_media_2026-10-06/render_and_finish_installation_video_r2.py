import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from check_completed_matrix import digest
from stage_lesson import read, write
from render_visual_master import frame_aligned_durations
from stage_web_renditions import probe, stage_one

parser = argparse.ArgumentParser()
parser.add_argument('identity', choices=['01_pypi_github', '03_pip_install', '04_platform_installers'])
args = parser.parse_args()
stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r2')
folder = stage / 'production' / args.identity
visual_path = folder / 'visual.json'
timing_path = folder / 'audio/en/af_heart.json'
visual, timing = read(visual_path), read(timing_path)
assert len(visual['scenes']) == len(timing['scenes'])
for scene in visual['scenes']:
    source = (folder / scene['image']).resolve()
    assert source.is_relative_to(stage)
    assert digest(source) == scene['capture_sha256']
master = folder / 'video' / (args.identity + '_silent.mp4')
subprocess.run([sys.executable, 'tools/tutorials/authoring/tools/render_visual_master.py', str(visual_path),
                '--timings', str(timing_path), '--output', str(master)], check=True)
master_hash = digest(master)
metadata = probe(master)
assert len(metadata['streams']) == 1
stream = metadata['streams'][0]
assert stream['codec_type'] == 'video'
assert (stream['width'], stream['height'], stream['r_frame_rate']) == (3840, 2160, '30/1')
expected_frames = sum(round(duration * 30) for duration in frame_aligned_durations(timing['scenes'], 30))
assert int(stream['nb_frames']) == expected_frames
subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2', '-i', str(master), '-f', 'null', '-'], check=True)
if (folder / 'scenes.json').exists():
    assert (folder / 'scenes.json').read_bytes() == visual_path.read_bytes()
else:
    shutil.copyfile(visual_path, folder / 'scenes.json')
import prepare_web_media
prepare_web_media.PRODUCTION = stage / 'production'
sys.argv = ['prepare_web_media.py', '--lessons', args.identity]
prepare_web_media.main()
spec = importlib.util.spec_from_file_location('normal_tutorial_encoder', 'tools/tutorials/authoring/tools/publish_tutorials.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
rendition = stage_one(stage, {'lesson': args.identity, 'scope': 'current actual staged bytes',
                            'reconciliation': {'master_sha256': master_hash}}, publisher)
assert digest(master) == master_hash
write(folder / 'current-video-acceptance.json', {'lesson': args.identity, 'source_frame_count': len(visual['scenes']),
      'source_frame_hashes_verified': True, 'visual_sha256': digest(visual_path),
      'english_timing_sha256': digest(timing_path), 'master_sha256': master_hash,
      'master_probe': metadata, 'master_full_decode_passed': True,
      'master_frames_match_quantized_english_timing': True,
      'poster_sha256': digest(folder / 'poster.jpg'), 'web_rendition': rendition,
      'browser_verified': False, 'visual_review_complete': False, 'published': False})
print('PASS:', args.identity, '4K master and web rendition: native frame hashes, full decode and exact timing; browser and publication remain open', flush=True)
