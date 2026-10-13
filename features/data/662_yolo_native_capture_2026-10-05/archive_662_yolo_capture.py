import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image

repo = Path.cwd()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-662-yolo-current'
capture = stage / 'captures/make_masks_yolo_662_r1'
native = json.loads((capture / 'yolo_acceptance.json').read_text())
frames = json.loads((capture / 'frames.json').read_text())
provenance = json.loads((capture / 'provenance.json').read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
    return stage / Path(path).relative_to('/tmp/spacr-tutorials')

assert provenance['completed_capture'] and native['accepted']
assert native['demonstration_boxes_are_not_biological_ground_truth']
assert native['source_images_and_masks_unchanged']
assert len(frames) == 19
for name, row in frames.items():
    path = capture / row['image']
    assert sha(path) == row['sha256']
    assert Image.open(path).size == (3840, 2160)
    assert row['appearance']['theme'] == 'dark'
    assert row['appearance']['backdrop'] == 'blobs'
    assert row['capture_surface'] == 'application_window'
    assert not row['hidden_decorative_backdrops']
for row in native['inputs']:
    for key in ('source', 'image', 'mask'):
        assert sha(local(row[key])) == row[key + '_sha256']
for key in ('project', 'labels', 'class_names', 'negative_labels'):
    assert sha(local(native[key])) == native[key + '_sha256']
assert json.loads(local(native['class_names']).read_text())['classes'] == native['classes']
assert local(native['negative_labels']).read_bytes() == b''
height, width = native['inputs'][0]['shape']
rows = [line.split() for line in local(native['labels']).read_text().splitlines()]
assert len(rows) == len(native['boxes']) == 2
for fields, (class_id, x0, y0, x1, y1) in zip(rows, native['boxes']):
    assert int(fields[0]) == class_id
    expected = [(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height]
    assert np.allclose(list(map(float, fields[1:])), expected, atol=1e-6, rtol=0)
folder = repo / 'features/data/662_yolo_native_capture_2026-10-05'
assert not folder.exists()
folder.mkdir()
for name in ('provenance.json', 'frames.json', 'yolo_acceptance.json', 'yolo_inputs.json'):
    (folder / name).write_bytes((capture / name).read_bytes())
for key in ('project', 'labels', 'class_names', 'negative_labels'):
    source = local(native[key])
    (folder / source.name).write_bytes(source.read_bytes())
with zipfile.ZipFile(folder / 'native-frames.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for row in frames.values():
        archive.write(capture / row['image'], row['image'])
with zipfile.ZipFile(folder / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ('spacr/qt/screens/make_masks.py', 'spacr/qt/mask_engine.py',
                 'tools/tutorials/capture_make_masks.py', 'tools/tutorials/capture_refresh.py',
                 'tools/tutorials/run_neutral_capture.sh', 'tools/tutorials/capture_policy.py'):
        archive.write(repo / name, name)
(folder / 'archive_662_yolo_capture.py').write_bytes(Path(__file__).read_bytes())
source_files = ('spacr/qt/screens/make_masks.py', 'spacr/qt/mask_engine.py',
                'tools/tutorials/capture_make_masks.py', 'tools/tutorials/capture_refresh.py')
receipt = {'schema': 1, 'item': 'N662', 'date': '2026-10-05',
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'accepted_native_capture': True,
           'scope': 'Fresh real Make Masks Box gestures and actual file exports on unchanged acquired planes; tutorial lesson authoring, narration, video and publication remain open.',
           'capture': str(capture), 'native_frames': 19, 'resolution': [3840, 2160],
           'native_acceptance': native, 'provenance': provenance,
           'independent_output_and_frame_hash_verification': True,
           'independent_yolo_full_image_coordinate_verification': True,
           'source_files': {name: sha(repo / name) for name in source_files},
           'artifacts': {str(path.relative_to(repo)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
                         for path in sorted(folder.iterdir())},
           'lesson_authoring_narration_video_and_publication_complete': False}
destination = repo / 'features/data/662_yolo_native_capture_2026-10-05.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print('Accepted 19 fresh native frames, exact source/mask bytes, independent YOLO coordinates, class companion, negative labels and exact reload.')
