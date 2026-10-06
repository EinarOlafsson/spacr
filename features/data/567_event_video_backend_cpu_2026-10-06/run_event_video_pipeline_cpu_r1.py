from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from spacr import _segmentation_backends as backend
from spacr import timelapse as tl

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
output = scratch / '567-event-video-pipeline-cpu-r1'
output.mkdir(exist_ok=False)
root = scratch / '567-event-video-backend-cpu-r1/backends'
model_dir = scratch / '567-videomae-preparation-r1/model'
os.environ['SPACR_BACKENDS_DIR'] = str(root)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_available()
tracks_dir = output / 'tracks'
tracks_dir.mkdir()
annotations = []
for column in (1, 2):
    field = f'plate1_r1_c{column}_f1'
    tracks = pd.DataFrame([
        {'frame': frame, 'track_id': track, 'x': 8 + 15 * track,
         'y': 12 + frame, 'area': 40 + frame, 'speed': float(frame)}
        for frame in range(5) for track in (1, 2)])
    base = tracks_dir / f'trackpy_tracks_cell_{field}'
    tracks.to_csv(str(base) + '.csv', index=False)
    features = tracks[['frame', 'track_id', 'area']].copy()
    features['intensity_mean_c0'] = [0.2 + 0.1 * f for f in features['frame']]
    features.to_csv(str(base) + '_features.csv', index=False)
    y, x = np.mgrid[0:16, 0:16]
    crops = np.stack([
        ((x + y + frame * 5 + track * 11 + column * 3) % 64)[None] / 63
        for frame, track in zip(features['frame'], features['track_id'])]).astype(np.float16)
    np.savez_compressed(str(base) + '_crops.npz',
        index=features[['frame', 'track_id']].to_numpy(), crops=crops)
    annotations += [{'field': field, 'track_id': 1, 'frame': 2, 'event': 'mitosis'},
                    {'field': field, 'track_id': 2, 'frame': 3, 'event': 'host_death'}]
ann = output / 'annotations.csv'
pd.DataFrame(annotations).to_csv(ann, index=False)
started = time.time()
try:
    result = tl._event_detection(str(tracks_dir), 'cell', 'trackpy',
        annotations=str(ann), encoder='videomae', video_checkpoint=str(model_dir),
        video_channels=[0, 0, 0], video_device='cpu', window=3, epochs=2,
        threshold=0.9, plot=False, conditions=['mock=c1', 'drug=c2'])
    model = torch.load(result['paths']['model'], weights_only=True, map_location='cpu')
    assert model['video_width'] == 768 and model['channels'] == 1
    assert model['video']['channel_map'] == [0, 0, 0]
    assert not any(k.startswith('_') for k in model['video'])
    assert model['video_provenance']['files_sha256'] == dict(backend._EVENT_VIDEO_FILES)
    original = result['events'].copy()
    restored = tl._event_detection(str(tracks_dir), 'cell', 'trackpy',
        model_path=result['paths']['model'], encoder='videomae', video_device='cpu',
        threshold=0.9, plot=False, conditions=['mock=c1', 'drug=c2'])
    pd.testing.assert_frame_equal(original, restored['events'])
    rejection = ''
    try:
        tl._event_detection(str(tracks_dir), 'cell', 'trackpy',
            model_path=result['paths']['model'], encoder='videomae',
            video_checkpoint=str(model_dir), video_channels=[0, 0, 1],
            video_device='cpu', plot=False)
    except ValueError as exc:
        rejection = str(exc)
    assert 'source-channel indices' in rejection, rejection
    report = {'item': 567, 'passed': True, 'CPU_only': True,
        'scope': 'Declared synthetic two-field plumbing test; not biological accuracy',
        'actual_model_classes': model['classes'], 'video_feature_width': model['video_width'],
        'actual_saved_provenance': model['video_provenance'],
        'held_out_scores': result['scores'].to_dict(orient='records'),
        'saved_model_weights_only_reload': True, 'trained_reload_events_exact': True,
        'wrong_source_channel_rejected': rejection, 'seconds': time.time() - started,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path('spacr/_segmentation_backends.py'), Path('spacr/timelapse.py'))}}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS actual installed pretrained encoder: full held-out/train/save/reload pipeline; synthetic CPU scope only.', flush=True)
finally:
    backend._shutdown_workers()
