from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import h5py
import numpy as np

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
classes = ['Early G1', 'Late G1', 'S/G2', 'Metaphase', 'Anaphase', 'Telophase', 'Abberent', 'Over_seg', 'Anaphase_defect']
target = root / 'CycleNET-original-five-plane-pixels.npy'
assert not target.exists()
total = 12717
pixels = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=(total, 64, 64, 5))
records = []
offset = 0
for split in ('train', 'test'):
    with h5py.File(root / f'cellcycle_{split}_set.hdf5') as source:
        data, labels, info = source['data1'], source['Index1'][:], source['Info1'][:]
        assert data.shape[1] == 5 * 64 * 64 and data.dtype == np.float16
        assert np.all(labels.sum(axis=1) == 1) and set(np.unique(labels)) == {0, 1}
        for start in range(0, len(data), 128):
            chunk = data[start:start + 128].reshape(-1, 5, 64, 64).transpose(0, 2, 3, 1)
            assert np.isfinite(chunk).all()
            pixels[offset + start:offset + start + len(chunk)] = chunk
            for j, image in enumerate(chunk):
                row = start + j
                metadata = [value.decode() for value in info[row]]
                records.append({'key': f'{split}:{row:05d}', 'original_split': split, 'original_row': row,
                                'pixel_row': offset + row, 'label_index': int(labels[row].argmax()),
                                'label': classes[int(labels[row].argmax())], 'original_info1': metadata,
                                'cell_identity_first_four_info_columns': metadata[:4],
                                'plate_group': '|'.join(metadata[:2]), 'field_group': '|'.join(metadata[:3]),
                                'full_original_five_plane_sha256': hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                                'biological_first_three_plane_sha256': hashlib.sha256(np.ascontiguousarray(image[..., :3]).tobytes()).hexdigest()})
        offset += len(data)
assert offset == total and len(records) == total
pixels.flush()
groups = defaultdict(list)
for record in records:
    groups[record['biological_first_three_plane_sha256']].append(record)
retained = []
excluded = []
conflicts = []
for digest, members in groups.items():
    if len({row['label'] for row in members}) > 1:
        conflicts.append({'sha256': digest, 'original_keys': [row['key'] for row in members], 'labels': [row['label'] for row in members]})
        excluded.extend({'key': row['key'], 'reason': 'identical biological pixels with conflicting manual labels'} for row in members)
    else:
        retained.append(members[0])
        excluded.extend({'key': row['key'], 'reason': 'same-label exact biological-pixel duplicate', 'retained_key': members[0]['key']} for row in members[1:])
retained.sort(key=lambda row: row['pixel_row'])
train = [r for r in records if r['original_split'] == 'train']
test = [r for r in records if r['original_split'] == 'test']
overlap = {}
for key in ('plate_group', 'field_group', 'biological_first_three_plane_sha256'):
    left, right = {r[key] for r in train}, {r[key] for r in test}
    overlap[key] = {'train_unique': len(left), 'test_unique': len(right), 'intersection': len(left & right)}
identity = Counter(tuple(r['cell_identity_first_four_info_columns']) for r in records)
sha = hashlib.sha256()
with target.open('rb') as stream:
    while block := stream.read(1024 * 1024):
        sha.update(block)
manifest = {'original_archive_sha256': '38790933cfe5c9bddd0e2976457f9a0bb105112b62a056d48222cae886328fb1',
            'label_map_primary_source': 'https://github.com/CellProfiling/subcell-analysis/blob/91b1e151c739eb23fb32fe99e1685903cf9915f2/utils/dataset.py',
            'original_stored_dtype': 'float16', 'original_shape': [total, 64, 64, 5], 'raw_photon_values_claimed': False,
            'biological_channel_order': ['F-RFP nuclear and bud-neck reference', 'GFP-tagged protein', 'RFP cytoplasmic reference'],
            'derived_planes_not_encoded': [3, 4], 'all_original_stored_pixels_preserved': True,
            'original_class_counts': dict(Counter(r['label'] for r in records)),
            'eligible_class_counts': dict(Counter(r['label'] for r in retained)),
            'preinference_dedup_rule': 'Retain first original row for same-label identical first-three biological planes; exclude all members of cross-label identical biological-pixel groups.',
            'original_rows': records, 'eligible_rows': retained, 'excluded_rows': excluded, 'conflicting_pixel_groups': conflicts,
            'duplicate_first_four_info_identity_groups': sum(v > 1 for v in identity.values()),
            'provided_train_test_overlap': overlap,
            'all_nine_original_label_classes_retained_without_post_result_exclusion': True,
            'original_pixel_array': {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha.hexdigest()},
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'CycleNET-original-cohort-inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({key: manifest[key] for key in ('original_class_counts', 'eligible_class_counts', 'provided_train_test_overlap', 'duplicate_first_four_info_identity_groups')}, indent=2))
print('PASS original rows', total, 'eligible', len(retained), 'excluded', len(excluded), 'cross-label pixel conflicts', len(conflicts), flush=True)
