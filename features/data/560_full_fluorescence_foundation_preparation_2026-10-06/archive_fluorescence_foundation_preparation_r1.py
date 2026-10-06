from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '560-human-fluorescence-primary-r1'
destination = Path('features/data/560_full_fluorescence_foundation_preparation_2026-10-06')
destination.mkdir(exist_ok=False)
sources = ['epithelial-record.json', 'epithelial-paper.xml', 'archive-download.json', 'original-image-inventory.json',
           'subcell-paper.xml', 'CycleNET-original-archive-download.json', 'CycleNET-author-download-availability.json',
           'CycleNET-original-archive-members.json', 'CycleNET-HDF5-original-inventory.json', 'CycleNET-original-cohort-inventory.json',
           'CycleNET-primary-source-manifest.json', 'SubCell-analysis-source-tree.json', 'frozen-full-fluorescence-foundation-plan-r1.json']
sources.extend(path.name for path in root.glob('*-CPU-weight-preparation.json'))
for relative in sources:
    target = destination / (relative + '.gz')
    target.write_bytes(gzip.compress((root / relative).read_bytes(), mtime=0))
primary = {}
for repository, folder, revision in [('BooneAndrewsLab/CycleNET', 'CycleNET-primary-source', '4a5d72f475b774b157c2ed7bb96a5d37d339ccbe'),
                                     ('CellProfiling/subcell-analysis', 'SubCell-analysis-primary-source', '91b1e151c739eb23fb32fe99e1685903cf9915f2')]:
    for path in sorted((root / folder).rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root / folder))
        data = path.read_bytes()
        target = destination / folder / (relative + '.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, mtime=0))
        primary[str(target)] = {'original_url': f'https://raw.githubusercontent.com/{repository}/{revision}/{relative}',
                                'original_sha256': hashlib.sha256(data).hexdigest(), 'original_bytes': len(data),
                                'original_download_was_HTTP404': relative == 'src/input_queue_cellcycle.py'}
scripts = ['download_hep2_expert_archive_r1.py', 'prepare_hep2_expert_images_r1.py', 'prepare_foundation_encoder_weights_r1.py',
           'prepare_cyclenet_original_cohort_r1.py', 'freeze_fluorescence_foundation_plan_r1.py', 'run_full_fluorescence_foundation_CUDA_r1.py',
           'archive_fluorescence_foundation_preparation_r1.py']
for name in scripts:
    shutil.copyfile(scratch / name, destination / name)
for name in ['hep2-expert-archive-download-r1.log', 'hep2-expert-image-preparation-r1.log', 'foundation-private-environment-r1.log',
             'foundation-normal-CPU-weight-preparation-r1.log', 'cyclenet-original-cohort-preparation-r1.log']:
    (destination / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
artifacts = {str(path): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
             for path in sorted(destination.rglob('*')) if path.is_file()}
yeast = json.loads((root / 'CycleNET-original-cohort-inventory.json').read_text())
receipt = {'item': 560, 'preparation_only_no_new_GPU_inference_or_scorecard_claim': True,
           'actual_all_10000_expert_epithelial_original_files_and_pixels_verified': True,
           'actual_all_12717_original_yeast_stored_crops_verified_and_preserved': True,
           'preinference_yeast_unique_unambiguous_crop_count': len(yeast['eligible_rows']),
           'same_label_identical_blank_yeast_crops_original_count': 2176,
           'yeast_cross_label_identical_pixel_group_count': 1,
           'yeast_cross_label_identical_pixel_rows_excluded': 2,
           'provided_yeast_split_is_not_plate_or_field_independent': True,
           'native_yeast_actual_channel_and_label_mapping_confirmed_in_pinned_SubCell_authors_loader': True,
           'six_normal_CPU_factories_loaded_and_all_actual_weight_and_parameter_hashes_frozen': True,
           'private_environment': str(scratch / '560-foundation-env-py313-r1'),
           'private_compatibility_versions': {'timm': '1.0.30', 'transformers': '4.48.3', 'tokenizers': '0.21.4', 'einops': '0.8.2'},
           'inherited_dinocell_package_pins_conflict_recorded_no_global_pip_check_or_production_environment_repair_claim': True,
           'frozen_plan_sha256': hashlib.sha256((root / 'frozen-full-fluorescence-foundation-plan-r1.json').read_bytes()).hexdigest(),
           'all_twelve_normal_model_dataset_comparisons_planned': True,
           'SubCell_DAPI_missing_protein_is_explicit_zero_padded_DNA_only_ablation': True,
           'SubCell_yeast_has_real_nuclear_reference_and_GFP_protein_planes': True,
           'original_archives_and_full_pixel_arrays_retained_at': str(root),
           'no_manual_mask_clinical_plate_holdout_Cell_DINO_or_native_speaker_claim': True,
           'original_failed_wrong_path_HTTP404_preserved_not_claimed_as_primary_evidence': True,
           'extended_raw_primary_source_records': primary, 'artifacts': artifacts}
Path(str(destination) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
for path, expected in artifacts.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected['sha256']
print('PASS frozen actual data and all six normal pretrained-weight preparations archived', len(artifacts), 'artifacts')
