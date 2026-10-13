from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import zipfile

import numpy as np
from PIL import Image
import requests

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r2'
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
            array = np.array(image.convert('RGB'))
        assert array.shape == (363, 360, 3) and array.dtype == np.uint8
        rows.append({'key': label + '/' + parts[-1], 'label': label, 'path': str(output),
                     'archive_member': member.filename,
                     'sha256': hashlib.sha256(payload).hexdigest(),
                     'RGB_pixels_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                     'shape': list(array.shape)})
        counts[label] += 1
assert counts == expected and len({r['key'] for r in rows}) == 17092
unique = {}
duplicates = []
for row in rows:
    pixel_hash = row['RGB_pixels_sha256']
    if pixel_hash in unique:
        assert unique[pixel_hash]['label'] == row['label']
        duplicates.append({'retained': unique[pixel_hash]['key'], 'excluded_duplicate': row['key']})
    else:
        unique[pixel_hash] = row
unique_rows = list(unique.values())
for source in ('pbc-primary-public-network.json', 'pbc-primary-dataset.html', 'pbc-primary-rendered-text.txt'):
    shutil.copyfile(scratch / source, root / source)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = {'prepared': True, 'dataset_primary_url': 'https://data.mendeley.com/datasets/snkd93bnjr/1',
        'dataset_DOI': '10.17632/snkd93bnjr.1', 'annotation_origin': 'Expert clinical pathologists, as stated by original dataset authors',
        'license': 'CC BY 4.0', 'dataset_archive_sha256': sha.hexdigest(), 'dataset_archive_bytes': size,
        'all_original_class_counts': counts, 'all_original_images': rows,
        'pixel_identical_duplicates_excluded': duplicates, 'retrieval_images': unique_rows,
        'embedding_policy': {'backbone': 'vit_small_patch14_dinov2.lvd142m', 'channel_policy': 'project',
                             'channels': [0, 1, 2], 'channel_scale': [255.0, 255.0, 255.0],
                             'resize': 'Whole RGB image resized to 224x224 with PIL bilinear; no crop or synthetic pixels',
                             'streaming': 'Normal spaCR embed_array in bounded chunks, using its real pretrained encoder cached once',
                             'batch_size': 64, 'device': 'cuda'},
        'retrieval_policy': 'All unique human-labelled cells; exact normalized FAISS search with self excluded; per-class precision@10 against class prevalence; no tuning on labels',
        'million_scale_policy': {'rows': 1200000, 'dimensions': 128, 'seed': 0,
                                'data': 'Explicit random float32 timing fixture, not acquired cell embeddings or biological validation',
                                'queries': 20, 'k': 100},
        'no_patient_split_screen_wide_GUI_or_million_real_cell_claim': True,
        'actual_GPU_embeddings_FAISS_timing_and_human_class_agreement_pending': True,
        'source_sha256': {p: digest(p) for p in ['spacr/active_learning.py', 'spacr/embeddings.py', 'spacr/schema.py', 'spacr/agreement.py']},
        'prepared_script_sha256': digest(__file__)}
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print('PASS: primary archive SHA256 and all 17092 original clinician-labelled RGB cells/class counts verified; unique retrieval images', len(unique_rows), 'exact pixel duplicates excluded', len(duplicates), '; actual GPU embedding/retrieval remains pending.', flush=True)
