from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import zipfile

import numpy as np
from PIL import Image
import requests

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
root.mkdir(exist_ok=False)
responses = json.loads((scratch / 'pbc-primary-public-network.json').read_text())
metadata = next(row['data'][0] for row in responses if '/files?' in row['url'])
assert metadata['filename'] == 'PBC_dataset_normal_DIB.zip'
details = metadata['content_details']
path = root / metadata['filename']
sha = hashlib.sha256()
size = 0
cached = scratch / '565-pbc-retrieval-r1' / metadata['filename']
shutil.copyfile(cached, path)
with path.open('rb') as stream:
    while block := stream.read(1024 * 1024):
        sha.update(block)
        size += len(block)
assert size == details['size'] and sha.hexdigest() == details['sha256_hash']
expected = {'basophil': 1218, 'eosinophil': 3117, 'erythroblast': 1551,
            'ig': 2895, 'lymphocyte': 1214, 'monocyte': 1420,
            'neutrophil': 3329, 'platelet': 2348}
expected_geometry = {(360,363):16639,(366,369):250,(360,360):198,(360,361):2,(361,360):1,(359,360):1,(362,360):1}
geometry_counts = {}
rows = []
counts = {name: 0 for name in expected}
image_root = root / 'original-images'
image_root.mkdir()
with zipfile.ZipFile(path) as archive:
    members = [r for r in archive.infolist() if r.filename.lower().endswith(('.jpg', '.jpeg'))
               and not any(part == '__MACOSX' or part.startswith('.') for part in PurePosixPath(r.filename).parts)]
    assert len(members) == 17092
    for member in sorted(members, key=lambda m: m.filename):
        parts = PurePosixPath(member.filename).parts
        assert '..' not in parts and not PurePosixPath(member.filename).is_absolute()
        label = next((part.lower() for part in parts if part.lower() in expected), None)
        assert label is not None
        output = image_root / label / parts[-1]
        output.parent.mkdir(exist_ok=True)
        assert not output.exists()
        payload = archive.read(member)
        output.write_bytes(payload)
        with Image.open(output) as image:
            assert image.mode == 'RGB'
            array = np.array(image)
        geometry = (array.shape[1], array.shape[0])
        assert array.ndim == 3 and array.shape[2] == 3 and geometry in expected_geometry and array.dtype == np.uint8
        geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1
        rows.append({'key': label + '/' + parts[-1], 'label': label, 'path': str(output),
                     'archive_member': member.filename,
                     'sha256': hashlib.sha256(payload).hexdigest(),
                     'RGB_pixels_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                     'shape': list(array.shape)})
        counts[label] += 1
assert geometry_counts == expected_geometry
assert counts == expected and len({r['key'] for r in rows}) == 17092
groups = {}
for row in rows:
    groups.setdefault(row['RGB_pixels_sha256'], []).append(row)
conflicts = {h: group for h, group in groups.items() if len({r['label'] for r in group}) > 1}
unique_rows = []
duplicates = []
for pixel_hash, group in groups.items():
    if pixel_hash in conflicts:
        continue
    unique_rows.append(group[0])
    duplicates.extend({'retained': group[0]['key'], 'excluded_duplicate': row['key']} for row in group[1:])
inventory = json.loads((scratch / '565-pbc-independent-archive-inventory.json').read_text())
assert len(conflicts) == len(inventory['cross_label_pixel_conflicts']) == 1
assert set(conflicts) == set(inventory['cross_label_pixel_conflicts'])
assert sum(len(group) for group in conflicts.values()) == inventory['conflicting_original_images'] == 2
assert len(duplicates) == inventory['same_label_extra_original_images'] == 16
assert len(unique_rows) == 17074
retrieval_counts = {name: sum(r['label'] == name for r in unique_rows) for name in expected}
shutil.copyfile(scratch / '565-pbc-independent-archive-inventory.json', root / 'independent-archive-inventory.json')
for source in ('pbc-primary-public-network.json', 'pbc-primary-dataset.html', 'pbc-primary-rendered-text.txt'):
    shutil.copyfile(scratch / source, root / source)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = {'prepared': True, 'dataset_primary_url': 'https://data.mendeley.com/datasets/snkd93bnjr/1',
        'dataset_DOI': '10.17632/snkd93bnjr.1', 'annotation_origin': 'Expert clinical pathologists, as stated by original dataset authors',
        'license': 'CC BY 4.0', 'dataset_archive_sha256': sha.hexdigest(), 'dataset_archive_bytes': size,
        'all_original_geometry_counts_width_height': {str(k):v for k,v in geometry_counts.items()}, 'all_original_class_counts': counts, 'all_original_images': rows,
        'cross_label_pixel_conflicts_excluded_entire_groups': conflicts, 'retrieval_class_counts': retrieval_counts, 'pixel_identical_duplicates_excluded': duplicates, 'retrieval_images': unique_rows,
        'embedding_policy': {'backbone': 'vit_small_patch14_dinov2.lvd142m', 'channel_policy': 'project',
                             'channels': [0, 1, 2], 'channel_scale': [255.0, 255.0, 255.0],
                             'resize': 'Whole RGB image resized to 224x224 with PIL bilinear; no crop or synthetic pixels',
                             'streaming': 'Normal spaCR embed_array in bounded chunks, using its real pretrained encoder cached once',
                             'batch_size': 64, 'device': 'cuda'},
        'retrieval_policy': 'All unique unambiguous human-labelled cells; every cross-label pixel-identical group excluded before embedding; exact normalized FAISS search with self excluded; per-class precision@10 against class prevalence; no tuning on labels',
        'million_scale_policy': {'rows': 1200000, 'dimensions': 128, 'seed': 0,
                                'data': 'Explicit random float32 timing fixture, not acquired cell embeddings or biological validation',
                                'queries': 20, 'k': 100},
        'no_patient_split_screen_wide_GUI_or_million_real_cell_claim': True,
        'actual_GPU_embeddings_FAISS_timing_and_human_class_agreement_pending': True,
        'source_sha256': {p: digest(p) for p in ['spacr/active_learning.py', 'spacr/embeddings.py', 'spacr/schema.py', 'spacr/agreement.py']},
        'prepared_script_sha256': digest(__file__)}
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print('PASS: primary archive SHA256 and all 17092 original clinician-labelled RGB cells/class counts verified; unique retrieval images', len(unique_rows), 'same-label pixel duplicates excluded', len(duplicates), 'cross-label conflicting original images excluded', 2, '; actual GPU embedding/retrieval remains pending.', flush=True)
