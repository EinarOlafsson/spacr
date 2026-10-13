from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image

repo = Path.cwd()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-662-yolo-r2-current'
capture = stage / 'captures/make_masks_yolo_662_r2'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
    return stage / Path(path).relative_to('/tmp/spacr-tutorials')

native, frames, provenance = [read(capture / name) for name in ('yolo_acceptance.json', 'frames.json', 'provenance.json')]
assert provenance['completed_capture'] and not provenance['app_source_modified']
assert native['accepted'] and native['source_images_and_masks_unchanged']
assert native['demonstration_boxes_are_not_biological_ground_truth'] and native['undo_redo_and_reload_exact']
assert len(frames) == 19
for name, row in frames.items():
    assert sha(capture / row['image']) == row['sha256']
    assert Image.open(capture / row['image']).size == (3840, 2160)
    assert row['capture_surface'] == 'application_window'
    assert row['appearance']['theme'] == 'dark' and row['appearance']['backdrop'] == 'blobs'
    assert not row['hidden_decorative_backdrops']
for row in native['inputs']:
    for key in ('source', 'image', 'mask'):
        assert sha(local(row[key])) == row[key + '_sha256']
for key in ('project', 'labels', 'class_names', 'negative_labels'):
    assert sha(local(native[key])) == native[key + '_sha256']
assert read(local(native['class_names']))['classes'] == native['classes']
assert local(native['negative_labels']).read_bytes() == b''
rows = [line.split() for line in local(native['labels']).read_text().splitlines()]
assert len(rows) == len(native['boxes']) == 2
height, width = native['inputs'][0]['shape']
for fields, (class_id, x0, y0, x1, y1) in zip(rows, native['boxes']):
    assert int(fields[0]) == class_id
    assert np.allclose(list(map(float, fields[1:])), [(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height], atol=1e-6, rtol=0)
previous = read(repo / 'features/data/662_make_masks_current_yolo_2026-10-05.json')
application = previous['application_source_sha256']
assert len(application) == 6 and all(sha(repo / name) == value for name, value in application.items())
pixels = read(scratch / 'yolo-full-resolution-proof.json')
assert pixels['accepted'] and len(pixels['captures'][capture.name]) == 10
folder = repo / 'features/data/662_yolo_native_capture_r2_2026-10-05'
assert not folder.exists()
folder.mkdir()
for name in ('provenance.json', 'frames.json', 'yolo_acceptance.json', 'yolo_inputs.json'):
    shutil.copyfile(capture / name, folder / name)
for key in ('project', 'labels', 'class_names', 'negative_labels'):
    shutil.copyfile(local(native[key]), folder / local(native[key]).name)
for name in ('check_yolo_full_resolution.py', 'yolo-full-resolution-proof.json', 'yolo-full-resolution-r1.log'):
    shutil.copyfile(scratch / name, folder / name)
shutil.copyfile(Path(__file__), folder / Path(__file__).name)
with zipfile.ZipFile(folder / 'native-frames.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for row in frames.values():
        archive.write(capture / row['image'], row['image'])
receipt = {
 'accepted_native_capture': True, 'capture': str(capture), 'native_acceptance': native,
 'native_capture_source_commit': provenance['commit'],
 'independent_current_application_sha256': application,
 'all_original_image_and_mask_bytes_exact': True,
 'full_image_normalized_XYWH_and_class_map_independently_verified': True,
 'ten_full_resolution_annotation_states_have_every_expected_box_edge': True,
 'visual_findings': 'Thin native two-pixel outlines are present in the original capture; resized previews obscured some. Desktop/repaint experiments are private diagnostics only and no recorder or application change remains.',
 'native_speaker_signoff': False, 'narration_video_publication_complete': False,
 'artifacts': {str(path.relative_to(repo)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(folder.iterdir())},
}
destination = repo / 'features/data/662_yolo_native_capture_r2_2026-10-05.json'
destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
adapter = {
 'accepted': True, 'scope': 'Original native Box interaction recording, independent complete output verification and full-resolution outline readback; narration/video/publication remain open',
 'capture': str(capture), 'provenance': provenance, 'native_acceptance': native,
 'frame_sha256': {name: row['sha256'] for name, row in frames.items()},
 'application_source_sha256': application,
 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
 'source_file_sha256': {str(capture / row['image']): row['sha256'] for row in frames.values()} | {str(repo / name): value for name, value in application.items()},
 'canonical_acceptance_receipt': str(destination.relative_to(repo)),
 'previous_capture_receipt_retained': 'features/data/662_yolo_native_capture_2026-10-05.json',
 'narration_video_publication_complete': False, 'biological_ground_truth_claimed': False,
}
(repo / 'features/data/662_make_masks_current_yolo_2026-10-05.json').write_text(json.dumps(adapter, ensure_ascii=False, indent=2) + '\n')
print('Accepted original native r2 frames and independent full-resolution box edges; prior receipt retained', flush=True)
