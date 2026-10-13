from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
original = scratch / '557-noisy-nuclei-primary-inspection-r1'
audit = scratch / '557-noisy-nuclei-pair-audit-r1'
inspection = json.loads((original / 'inspection.json').read_text())
pairs = json.loads((audit / 'pairs.json').read_text())
manual = json.loads((audit / 'manual-mask-identity-check.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert pairs['all_50_original_pixels_reverified'] and len(pairs['pairs']) == 25
assert digest(original / 'Denoising_Dataset.zip') == inspection['sha256']
assert digest(audit / 'Stardist_v2.zip') == manual['sha256']
assert manual['published_MD5_verified']
out = Path('features/data/557_public_paired_image_audit_2026-10-06')
out.mkdir(exist_ok=False)
sources = [original / 'inspection.json', original / 'primary-record.html', audit / 'pairs.json', audit / 'manual-mask-identity-check.json', audit / 'manual-mask-primary-record.html', scratch / 'inspect_real_noisy_nuclei_metadata_r1.py', scratch / 'audit_noisy_nuclei_pairs_r1.py', scratch / 'real-noisy-nuclei-inspection-r1.log', scratch / 'noisy-nuclei-pair-audit-r1.log', Path(__file__)]
for source in sources:
    target = out / source.name
    if source.suffix in ('.log', '.html') or source.name == 'inspection.json':
        target = target.with_name(target.name + '.gz')
        target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == source.read_bytes()
    else:
        shutil.copyfile(source, target)
summary = {'item': 557, 'public_paired_image_original_metadata_alignment_and_manual_mask_identity_audit_complete': True, 'paired_dataset_URL': inspection['original_primary_URL'], 'paired_dataset_DOI': inspection['original_DOI'], 'original_archive_sha256': inspection['sha256'], 'original_archive_MD5': inspection['MD5'], 'original_archive_bytes': inspection['bytes'], 'original_TIFFs': 50, 'training_pairs': 20, 'test_pairs': 5, 'all_original_TIFF_pixels_file_hashes_and_pair_basenames_exact': True, 'exact_pixel_duplicate_groups': pairs['exact_original_pixel_duplicate_groups'], 'diagnostic_shift_max_absolute_px': max(abs(v) for r in pairs['pairs'] for v in r['diagnostic_blur_sigma2_phase_correlation_shift_YX_px']), 'diagnostic_unshifted_blurred_Pearson_range': [min(r['unshifted_Pearson_by_Gaussian_sigma_px']['2'] for r in pairs['pairs']), max(r['unshifted_Pearson_by_Gaussian_sigma_px']['2'] for r in pairs['pairs'])], 'no_masks_exposure_laser_photon_or_gain_calibration_in_original_archive': True, 'related_manual_annotation_dataset_URL': manual['manual_annotation_primary_URL'], 'related_manual_annotation_archive_sha256': manual['sha256'], 'related_manual_annotation_original_TIFFs': len(manual['all_original_TIFF_inventory']), 'related_manual_annotation_exact_original_pixel_match_count': len(manual['exact_original_pixel_matches_with_noisy_or_high_signal_pairs']), 'same_cell_line_and_microscope_do_not_authorize_transferring_masks': True, 'diagnostic_alignment_only_no_resampling_or_transferred_mask_claim': True, 'archive_originals_retained_in_scratch_primary_immutable_urls_and_exact_hashes_in_repository': True, 'no_training_GPU_denoising_F1_or_calibrated_low_light_acceptance_claim': True, 'second_calibrated_low_light_dataset_requirement_remains_open': True, 'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}}
out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
note = f'''\n2026-10-06 workstation additional real denoising dataset audit: the primary noisy-nuclei archive DOI 10.5281/zenodo.5750174 matches its published MD5 and all 50 acquired 1024x1024 uint16 TIFFs are independently reverified. Twenty original training pairs and five original test pairs match by basename; exact pixel duplicates and unshifted/phase-correlation diagnostics are recorded without modifying, registering or denoising any original. Maximum diagnostic shift is {summary['diagnostic_shift_max_absolute_px']} px. There are no manual masks, exposure/laser/gain/photon settings, OME metadata or acquisition-calibration text in the original archive; ImageJ display limits are not calibration. The related manual-mask StarDist archive DOI 10.5281/zenodo.3715492 is independently downloaded and verified against both published MD5 and SHA256, inventorying {summary['related_manual_annotation_original_TIFFs']} original TIFFs. Exact pixel matches with the noisy/high-signal dataset: {summary['related_manual_annotation_exact_original_pixel_match_count']}. Matching species/microscope descriptions do not justify mask transfer to unmatched images. Receipt 557_public_paired_image_audit_2026-10-06.json retains source records, complete original file/pixel inventories, all pair diagnostics, identity comparison, scripts and terminal logs; original archives remain intact in scratch. This advances actual dataset evidence but does not satisfy the still-open second calibrated low-light segmentation-F1 requirement or claim new training/GPU/biological accuracy. The optional request for a local real low-light/manual-mask acquisition with recorded exposure remains pending. Current DINOv2 normal retry continues through the unchanged scheduler; Home retains CPU CI/Qt/source and workstation retains all GPU/documentation lanes. Protected jobs remain untouched.\n'''
for path in (Path('features/future/557_self_supervised_denoising.txt'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp')):
    with path.open('a') as stream:
        stream.write(note)
print('PASS complete public paired-original audit and independent annotated-image identity result archived; item 557 scientific requirement stays open.', flush=True)
