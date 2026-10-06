from pathlib import Path, PurePosixPath
from collections import Counter
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
download = json.loads((root / 'archive-download.json').read_text())
archive_path = root / 'HEp-2-ExpD.zip'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == download['original_archive_sha256']
expected = {'Interphase': 2500, 'Anaphase': 1250, 'Artefact': 1250,
            'Metaphase': 1250, 'Prophase': 1250, 'Telophase': 1250, 'Triple': 1250}
destination = root / 'original-images'
destination.mkdir(exist_ok=False)
pixels_path = root / 'original-DAPI-pixels.npy'
pixels = np.lib.format.open_memmap(pixels_path, mode='w+', dtype=np.uint8, shape=(10000, 256, 256))
rows = []
with zipfile.ZipFile(archive_path) as archive:
    members = sorted((r for r in archive.infolist() if r.filename.lower().endswith('.webp')), key=lambda r: r.filename)
    assert len(members) == 10000
    for index, member in enumerate(members):
        parts = PurePosixPath(member.filename)
        assert not parts.is_absolute() and '..' not in parts.parts and len(parts.parts) == 2
        label = parts.parts[0]
        assert label in expected and parts.stem.startswith(label + '_')
        payload = archive.read(member)
        assert payload[:4] == b'RIFF' and payload[8:12] == b'WEBP'
        assert payload[12:16] == b'VP8L'
        with Image.open(io.BytesIO(payload)) as image:
            assert image.mode == 'RGB' and image.size == (256, 256)
            array = np.array(image)
        assert array.dtype == np.uint8 and array.shape == (256, 256, 3)
        np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
        np.testing.assert_array_equal(array[:, :, 0], array[:, :, 2])
        pixels[index] = array[:, :, 0]
        target = destination / member.filename
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)
        rows.append({'key': member.filename, 'label': label, 'original_row': index,
                     'original_file_sha256': hashlib.sha256(payload).hexdigest(),
                     'DAPI_pixels_sha256': hashlib.sha256(array[:, :, 0].tobytes()).hexdigest(),
                     'original_path': str(target)})
pixels.flush()
assert dict(Counter(r['label'] for r in rows)) == expected
groups = {}
for row in rows:
    groups.setdefault(row['DAPI_pixels_sha256'], []).append(row)
conflicts = {sha: group for sha, group in groups.items() if len({r['label'] for r in group}) > 1}
cohort, duplicates = [], []
for sha, group in groups.items():
    if sha in conflicts:
        continue
    cohort.append(group[0])
    duplicates.extend({'retained': group[0]['key'], 'excluded_duplicate': r['key']}
                      for r in group[1:])
receipt = {'dataset': 'Original expert-classified HEp-2 ExpD',
           'resolved_version_DOI': download['resolved_version_DOI'],
           'archive_download': download,
           'all_10000_original_lossless_WebP_preserved': True,
           'all_three_stored_RGB_planes_exactly_identical_DAPI': True,
           'stored_RGB_not_three_biological_stains': True,
           'original_shape': [256, 256], 'original_dtype': 'uint8',
           'original_class_counts': expected, 'all_original_images': rows,
           'cross_label_pixel_conflicts_excluded_entire_groups': conflicts,
           'same_label_pixel_duplicates_excluded': duplicates,
           'expert_retrieval_images': cohort,
           'expert_retrieval_class_counts': dict(Counter(r['label'] for r in cohort)),
           'original_DAPI_array_sha256': hashlib.sha256(pixels_path.read_bytes()).hexdigest(),
           'original_DAPI_array_bytes': pixels_path.stat().st_size,
           'no_CNN_D_labels_G1_S_G2_M_truth_manual_mask_or_patient_split_claim': True,
           'normal_CUDA_encoder_comparisons_not_yet_run': True,
           'preparation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'original-image-inventory.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS all 10000 original expert-labelled lossless DAPI images/pixels verified; unique unambiguous cohort', len(cohort), 'same-label extras', len(duplicates), 'cross-label groups', len(conflicts), flush=True)
