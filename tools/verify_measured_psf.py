"""Verify a measured optical PSF on public fluorescent beads, CPU only.

Pass a folder containing zenodo-record.json (the record API response),
psf-200nm.tif and 30x1000_FS-025_FW-60_AC-170-04.tif from
https://doi.org/10.5281/zenodo.14203207. Run under tools/run_capped.sh 6G.
Writes a prepared PSF, resampled image, restored volume and JSON receipt into
that folder. Original downloaded files are verified by MD5 and never changed.
This checks calibrated processing, not resolution or biological accuracy.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import affine_transform

from spacr.point_spread import apply_psf, load_psf


def main():
    """Run the fixed public-data acceptance procedure and record its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Local Zenodo files and zenodo-record.json; derived outputs are written here.')
    root = parser.parse_args().folder
    record = json.loads((root / 'zenodo-record.json').read_text())
    files = {item['key']: item for item in record['files']}
    image_path = root / '30x1000_FS-025_FW-60_AC-170-04.tif'
    psf_path = root / 'psf-200nm.tif'


    def digest(path, algorithm='sha256'):
        """Hash one bounded local source or derived file."""
        return hashlib.new(algorithm, path.read_bytes()).hexdigest()


    def read_calibrated(path):
        """Read ImageJ spatial calibration without rounding its rational values."""
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            spacing = [float(tif.imagej_metadata['spacing'])]
            for tag in ('YResolution', 'XResolution'):
                numerator, denominator = page.tags[tag].value
                spacing.append(denominator / numerator)
            return tif.asarray(), tuple(spacing), tif.imagej_metadata['unit']


    for path in (image_path, psf_path):
        assert files[path.name]['checksum'] == 'md5:' + digest(path, 'md5')
    original_hashes = {path.name: digest(path) for path in (image_path, psf_path)}
    image, image_spacing, image_unit = read_calibrated(image_path)
    measured, psf_spacing, psf_unit = read_calibrated(psf_path)
    peak = np.unravel_index(int(measured.argmax()), measured.shape)
    radius = [max(p, size - 1 - p) for p, size in zip(peak, measured.shape)]
    padding = [(r - p, r - (size - 1 - p))
               for r, p, size in zip(radius, peak, measured.shape)]
    prepared = np.pad(measured, padding, mode='constant')
    assert np.array_equal(prepared[tuple(slice(a, a + n) for (a, _), n in
                                         zip(padding, measured.shape))], measured)
    assert math.fsum(prepared.ravel()) == math.fsum(measured.ravel())
    assert np.unravel_index(int(prepared.argmax()), prepared.shape) == tuple(radius)
    derivative = root / 'psf-200nm-explicit-odd-peak-origin.tif'
    tifffile.imwrite(derivative, prepared, imagej=True, metadata={
        'axes': 'ZYX', 'unit': 'um', 'spacing': psf_spacing[0]},
        resolution=(1 / psf_spacing[2], 1 / psf_spacing[1]))
    kernel = load_psf(derivative, sampling_um=psf_spacing)

    try:
        apply_psf(image, kernel, operation='deconvolve', image_sampling_um=image_spacing)
    except ValueError as error:
        assert 'sampling differ' in str(error)
        mismatch_refusal = str(error)
    else:
        raise AssertionError('The original mismatched calibration must be refused')

    ratio = np.asarray(psf_spacing) / image_spacing
    center = (np.asarray(image.shape) - 1) / 2
    matrix = np.diag(ratio)
    offset = center - matrix @ center
    resampled = affine_transform(image, matrix, offset=offset, output_shape=image.shape,
                                 output=np.float32, order=1, mode='reflect', prefilter=False)
    assert np.isfinite(resampled).all()
    input_digest = hashlib.sha256(resampled.tobytes()).hexdigest()
    started = time.monotonic()
    progress = []


    def update(channel, completed, total):
        """Retain each iteration callback for the completion assertion."""
        progress.append([channel, completed, total])
        print(f'channel {channel}: {completed}/{total}', flush=True)


    result = apply_psf(resampled, kernel, operation='deconvolve',
                       image_sampling_um=psf_spacing, iterations=10, progress=update)
    elapsed = time.monotonic() - started
    assert result.image.shape == resampled.shape
    assert result.image.dtype == np.float32
    assert np.isfinite(result.image).all() and (result.image >= 0).all()
    assert not np.array_equal(result.image, resampled)
    assert input_digest == hashlib.sha256(resampled.tobytes()).hexdigest()
    assert original_hashes == {path.name: digest(path) for path in (image_path, psf_path)}
    assert progress == [[0, number, 10] for number in range(1, 11)]
    np.save(root / 'beads-matched-grid.npy', resampled)
    np.save(root / 'beads-measured-psf-restored.npy', result.image)
    receipt = {
        'item': 490, 'date_utc': datetime.now(timezone.utc).date().isoformat(), 'device': 'CPU', 'execution_note': 'Use tools/run_capped.sh 6G; memory limits are enforced by the caller.',
        'validation_script_sha256': digest(Path(__file__)),
        'source': 'https://doi.org/10.5281/zenodo.14203207',
        'source_license': 'CC-BY-4.0',
        'source_authors': ['Nicolas Chiaruttini', 'Romain Guiet'],
        'source_sha256': original_hashes, 'source_md5_verified_against_record': True,
        'image_shape_zyx': list(image.shape), 'image_dtype': str(image.dtype),
        'image_sampling_um': image_spacing, 'image_metadata_unit': image_unit,
        'psf_original_shape_zyx': list(measured.shape), 'psf_sampling_um': psf_spacing,
        'psf_metadata_unit_literal': psf_unit, 'psf_peak_origin_zyx': [int(x) for x in peak],
        'psf_zero_padding_zyx': [[int(x) for x in pair] for pair in padding],
        'psf_derived_shape_zyx': list(prepared.shape), 'psf_derived_sha256': digest(derivative),
        'psf_values_and_total_preserved_before_normalization': True,
        'original_grid_refusal': mismatch_refusal,
        'explicit_image_resampling': {
            'matrix_output_to_input': matrix.tolist(), 'offset': offset.tolist(),
            'interpolation_order': 1, 'boundary': 'reflect', 'center_aligned': True,
            'output_sampling_um': psf_spacing,
            'purpose': 'Explicitly match image sampling to the measured PSF; no runtime tolerance changes.',
        },
        'saturated_input_voxels': int(np.count_nonzero(image == np.iinfo(image.dtype).max)),
        'seconds': elapsed, 'iterations': 10, 'output_dtype': str(result.image.dtype),
        'output_shape_preserved': True, 'output_finite_nonnegative': True,
        'original_files_and_processing_input_unchanged': True,
        'output_sha256': digest(root / 'beads-measured-psf-restored.npy'),
        'input_quantiles': np.percentile(resampled, [0, 50, 99, 99.9, 100]).tolist(),
        'output_quantiles': np.percentile(result.image, [0, 50, 99, 99.9, 100]).tolist(),
        'processing_provenance': result.provenance,
        'limits': [
            'The record calls this the best-settings PSF; an exact optical-setting match to this bead acquisition is not verified.',
            'Real fluorescent-bead data and a measured optical PSF, not biological recovery ground truth.',
            'Both arrays remain in the published raw, non-deskewed LLS7 coordinates.',
            'Explicit image resampling and zero-padding/peak-origin preparation are acceptance-script steps, not new automatic application behavior.',
            'Saturated voxels, finite 200nm bead size, background and unregularized RL preclude a resolution or accuracy claim.',
            'This validates a CPU 3D measured-PSF application; it does not revalidate every GUI or establish applicability to another microscope.',
        ],
    }
    (root / 'measured-application-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'seconds': elapsed, 'shape': result.image.shape, 'status': 'passed'}), flush=True)


if __name__ == "__main__":
    main()
