from pathlib import Path
import hashlib
import json
import subprocess

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
hep = json.loads((root / 'original-image-inventory.json').read_text())
yeast = json.loads((root / 'CycleNET-original-cohort-inventory.json').read_text())
models = ('resnet18', 'resnet50', 'vit_small_patch14_dinov2.lvd142m', 'openphenom', 'chada_vit', 'subcell')
cohorts = {
    'HEp2-expert-DAPI': {'rows': [{'key': row['key'], 'label': row['label'], 'pixel_row': row['original_row']} for row in hep['expert_retrieval_images']],
                        'pixel_path': str(root / 'original-DAPI-pixels.npy'), 'pixel_sha256': hep['original_DAPI_array_sha256'],
                        'inventory_path': str(root / 'original-image-inventory.json'), 'inventory_sha256': digest(root / 'original-image-inventory.json'),
                        'class_counts': hep['expert_retrieval_class_counts'], 'original_shape': [10000, 256, 256], 'original_dtype': 'uint8',
                        'actual_biological_markers': ['DAPI'], 'scale': 255.0,
                        'SubCell_interpretation': 'DNA-only ablation: project one DAPI plane and zero-pad the absent protein input; never duplicate DAPI as protein.',
                        'class_scope': 'All seven original expert classes, including Artefact and Triple; no CNN-D or inferred G1/S/G2 labels.'},
    'CycleNET-expert-native': {'rows': [{'key': row['key'], 'label': row['label'], 'pixel_row': row['pixel_row'], 'original_split': row['original_split'], 'plate_group': row['plate_group'], 'field_group': row['field_group']} for row in yeast['eligible_rows']],
                             'pixel_path': yeast['original_pixel_array']['path'], 'pixel_sha256': yeast['original_pixel_array']['sha256'],
                             'inventory_path': str(root / 'CycleNET-original-cohort-inventory.json'), 'inventory_sha256': digest(root / 'CycleNET-original-cohort-inventory.json'),
                             'class_counts': yeast['eligible_class_counts'], 'original_shape': yeast['original_shape'], 'original_dtype': 'float16',
                             'actual_biological_markers': yeast['biological_channel_order'], 'scale': 1.0,
                             'SubCell_interpretation': 'Published nuclear-reference/protein mapping: stored planes 0 and 1. F-RFP is a nuclear/bud-neck marker, not a DNA stain.',
                             'class_scope': 'All nine original manual classes, including QC classes. No result-based exclusions.',
                             'provided_train_test_overlap': yeast['provided_train_test_overlap'],
                             'preinference_dedup_rule': yeast['preinference_dedup_rule'],
                             'stored_pixels_already_author_processed_not_raw_photon_calibration': True}
}
matrix = []
weights = {}
for backbone in models:
    prep_path = root / (backbone + '-CPU-weight-preparation.json')
    weights[backbone] = {'preparation_path': str(prep_path), 'preparation_sha256': digest(prep_path), 'receipt': json.loads(prep_path.read_text())}
    for cohort, data in cohorts.items():
        channels = [0] if cohort == 'HEp2-expert-DAPI' else ([0, 1] if backbone == 'subcell' else [0, 1, 2])
        policy = 'project' if cohort == 'CycleNET-expert-native' or backbone == 'subcell' else 'per_channel'
        size = weights[backbone]['receipt']['normal_input_size']
        timm = backbone in models[:3]
        matrix.append({'backbone': backbone, 'cohort': cohort, 'channels': channels, 'channel_policy': policy,
                       'channel_scale': [data['scale']] * len(channels), 'batch_size': 16, 'device': 'cuda', 'normalize': True,
                       'external_resize': {'size': size, 'method': 'PIL whole-crop bilinear; HEp2 uint8 L, CycleNET float32 F per biological plane'} if timm else None,
                       'internal_resize': None if timm else {'size': size, 'method': 'unchanged normal foundation factory torch bilinear align_corners=False'},
                       'input_clipping': 'unchanged normal embed_array clips scaled inputs to [0,1]',
                       'no_added_timm_ImageNet_normalization': True,
                       'normal_model_state_sha256': weights[backbone]['receipt']['complete_ordered_model_state_sha256']})
plan = {'item': 560, 'application_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'application_source_sha256': digest(Path('spacr/embeddings.py')), 'cohorts': cohorts, 'models': weights, 'comparisons': matrix,
        'random_seed_before_each_normal_factory': 0, 'CUDA_profiles': 'Actual first and last complete forward batch for every model/cohort; require CUDA kernels.',
        'execution': 'Normal _backbone_encoder and embed_array, cached real encoder reused across chunks; no mock or surrogate.',
        'scientific_comparison': 'Existing spaCR model pipelines with explicit model-appropriate adapters, not a causal architecture-only comparison.',
        'retrieval': {'normal_callable': '_scored_encoder_entry', 'k': 10, 'all_eligible_rows': True, 'full_rank_AP': True, 'self_excluded': True,
                      'independent_replay': 'All-row cosine ranking, stable ties, complete AP and neighbour votes/chance reconstructed from original saved features.'},
        'classifier': {'normal_callable': '_embedding_classifier_scorecard', 'folds': 5, 'seed': 0, 'scope': 'Within-cohort stratified crop diagnostic; scaler and fit inside folds.'},
        'no_patient_plate_gene_unseen_generalisation_clinical_or_manual_mask_accuracy_claim': True,
        'Cell_DINO_status': 'Official checkpoint not acquired; application version unsupported. No invented model result.',
        'freeze_script_sha256': digest(Path(__file__))}
target = root / 'frozen-full-fluorescence-foundation-plan-r1.json'
assert not target.exists()
target.write_text(json.dumps(plan, indent=2) + '\n')
print('PASS frozen twelve comparisons, expert cohort sizes', {name: len(value['rows']) for name, value in cohorts.items()}, 'plan SHA256', digest(target))
