from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import hashlib
import io
import json
import zipfile

import numpy as np
import requests
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
original = scratch / '557-noisy-nuclei-primary-inspection-r1'
target = scratch / '557-noisy-nuclei-pair-audit-r1'
target.mkdir(exist_ok=False)
inspection = json.loads((original / 'inspection.json').read_text())
digest = lambda x: hashlib.sha256(x).hexdigest()
archive = original / 'Denoising_Dataset.zip'
assert digest(archive.read_bytes()) == inspection['sha256']
arrays = {}
groups = defaultdict(dict)
with zipfile.ZipFile(archive) as source:
    for row in inspection['TIFF_originals']:
        member = PurePosixPath(row['archive_member'])
        payload = source.read(str(member))
        assert digest(payload) == row['sha256']
        image = tifffile.imread(io.BytesIO(payload))
        assert image.dtype == np.uint16 and image.shape == (1024, 1024)
        assert digest(image.tobytes()) == row['original_pixels_sha256']
        split = member.parts[1].lower()
        arm = member.parts[2].lower()
        assert split in ('training', 'test') and arm in ('low', 'high')
        assert arm not in groups[(split, member.name)]
        groups[(split, member.name)][arm] = row
        arrays[str(member)] = image
assert Counter(split for split, _ in groups) == {'training': 20, 'test': 5}
records = []
for (split, name), arms in sorted(groups.items()):
    assert set(arms) == {'low', 'high'}
    high = arrays[arms['high']['archive_member']].astype(np.float32)
    low = arrays[arms['low']['archive_member']].astype(np.float32)
    smooth_high = gaussian_filter(high, 2)
    smooth_low = gaussian_filter(low, 2)
    shift, error, phase = phase_cross_correlation(smooth_high, smooth_low, upsample_factor=10, normalization=None)
    correlations = {str(sigma): float(np.corrcoef(gaussian_filter(high, sigma).ravel(), gaussian_filter(low, sigma).ravel())[0, 1]) for sigma in (0, 2, 4)}
    records.append({'split': split, 'basename': name, 'originals': arms, 'diagnostic_blur_sigma2_phase_correlation_shift_YX_px': shift.tolist(), 'phase_correlation_normalized_error': float(error), 'phase_difference': float(phase), 'unshifted_Pearson_by_Gaussian_sigma_px': correlations, 'original_high_saturated_pixels': int((high == 65535).sum()), 'original_low_saturated_pixels': int((low == 65535).sum()), 'no_images_resampled_or_shifted': True})
pixels = defaultdict(list)
for row in inspection['TIFF_originals']:
    pixels[row['original_pixels_sha256']].append(row['archive_member'])
duplicates = {key: value for key, value in pixels.items() if len(value) > 1}
assert digest(archive.read_bytes()) == inspection['sha256']
pair_record = {'source_DOI': inspection['original_DOI'], 'archive_sha256': inspection['sha256'], 'all_50_original_pixels_reverified': True, 'pairs': records, 'exact_original_pixel_duplicate_groups': duplicates, 'shift_is_diagnostic_only_not_a_mask_registration_or_physical_calibration': True, 'no_manual_masks_exposure_laser_or_photon_settings_in_original_archive': True, 'no_training_GPU_F1_or_calibrated_low_light_acceptance_claim': True}
(target / 'pairs.json').write_text(json.dumps(pair_record, indent=2) + '\n')
print('PASS original paired image audit; counts', Counter(r['split'] for r in records), 'max absolute diagnostic shift', max(abs(v) for r in records for v in r['diagnostic_blur_sigma2_phase_correlation_shift_YX_px']), 'exact duplicates', len(duplicates), flush=True)
url = 'https://zenodo.org/records/3715492'
response = requests.get(url, timeout=90)
response.raise_for_status()
(target / 'manual-mask-primary-record.html').write_text(response.text)
annotated = target / 'Stardist_v2.zip'
md5 = hashlib.md5()
sha = hashlib.sha256()
size = 0
with requests.get(url + '/files/Stardist_v2.zip?download=1', stream=True, timeout=90) as response:
    response.raise_for_status()
    with annotated.open('wb') as stream:
        for block in response.iter_content(1024 * 1024):
            stream.write(block)
            md5.update(block)
            sha.update(block)
            size += len(block)
assert md5.hexdigest() == '5b7490336e5c5b8035e43cd8302e80a3'
assert sha.hexdigest() == 'aec767afae76942b7c97e31c500284f8b5862150d8e81b57f513e66d7258c05e'
inventory = []
matches = []
with zipfile.ZipFile(annotated) as source:
    for member in source.infolist():
        path = PurePosixPath(member.filename)
        assert '..' not in path.parts and not path.is_absolute()
        if member.is_dir() or any(p == '__MACOSX' or p.startswith('.') for p in path.parts) or path.suffix.lower() not in ('.tif', '.tiff'):
            continue
        payload = source.read(member)
        image = tifffile.imread(io.BytesIO(payload))
        pixel_sha = digest(image.tobytes())
        record = {'member': member.filename, 'bytes': len(payload), 'sha256': digest(payload), 'shape': list(image.shape), 'dtype': str(image.dtype), 'original_pixels_sha256': pixel_sha}
        inventory.append(record)
        if pixel_sha in pixels:
            matches.append({'annotated_member': record, 'denoising_originals': pixels[pixel_sha]})
match_record = {'manual_annotation_primary_URL': url, 'DOI': '10.5281/zenodo.3715492', 'published_MD5_verified': True, 'MD5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'bytes': size, 'all_original_TIFF_inventory': inventory, 'exact_original_pixel_matches_with_noisy_or_high_signal_pairs': matches, 'same_species_and_microscope_are_insufficient_to_transfer_masks': True, 'no_inferred_image_registration_or_manual_labels_assigned_to_unmatched_pairs': True}
(target / 'manual-mask-identity-check.json').write_text(json.dumps(match_record, indent=2) + '\n')
print('PASS annotated archive original MD5/SHA verified; TIFFs', len(inventory), 'exact denoising image pixel matches', len(matches), flush=True)
