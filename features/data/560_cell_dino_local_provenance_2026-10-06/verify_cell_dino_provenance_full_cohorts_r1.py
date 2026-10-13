from pathlib import Path
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from spacr.embeddings import EmbeddingSpec, _scored_encoder_entry, encoder_entry

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '560-human-fluorescence-primary-r1'
encoded = root / 'full-fluorescence-foundation-CUDA-r1'
previous = root / 'full-fluorescence-foundation-CPU-scorecards-r2'
source_sha = hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest()
assert source_sha == json.loads((scratch / 'Home-current-documentation-inventory-r1.json').read_text())['source_sha256']

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

baseline_path = scratch / 'foundation-api-refresh-baseline-r1/embeddings.py'
module_spec = importlib.util.spec_from_file_location('spacr._foundation_baseline', baseline_path)
baseline = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = baseline
module_spec.loader.exec_module(baseline)
four = pd.DataFrame(np.eye(4), index=['a0', 'a1', 'b0', 'b1'])
labels = dict(zip(four.index, ['a', 'a', 'b', 'b']))
old_four = baseline._retrieval_scorecard(four, labels)
new_four = _scored_encoder_entry(None, four, labels).metrics
np.testing.assert_allclose(old_four['chance_map'], 1 / 3, rtol=0, atol=1e-12)
np.testing.assert_allclose(new_four['chance_map'], 11 / 18, rtol=0, atol=1e-12)
for name in ('openphenom', 'chada_vit', 'subcell'):
    original = baseline.encoder_entry(baseline.EmbeddingSpec(backbone=name))
    assert not original.sha256 and original.uri == 'timm:' + name
    assert encoder_entry(EmbeddingSpec(backbone=name)).sha256

execution = json.loads((encoded / 'complete-execution.json').read_text())
plan = json.loads((root / 'frozen-full-fluorescence-foundation-plan-r1.json').read_text())
results = []
for run in execution['comparisons']:
    name = run['backbone'] + '-' + run['cohort']
    assert digest(run['full_features']['path']) == run['full_features']['sha256']
    old = json.loads((previous / (name + '-scorecard.json')).read_text())
    values = np.load(run['full_features']['path'], mmap_mode='r')
    keys = [row['key'] for row in run['ordered_rows']]
    named = {row['key']: row['label'] for row in run['ordered_rows']}
    assert run['ordered_rows'] == plan['cohorts'][run['cohort']]['rows']
    frame = pd.DataFrame(values, index=keys, columns=run['columns'])
    settings = dict(run['normal_spec'])
    for key in ('channels', 'channel_scale'):
        settings[key] = tuple(settings[key])
    spec = EmbeddingSpec(**settings)
    started = time.perf_counter()
    entry = _scored_encoder_entry(spec, frame, named, k=10)
    elapsed = time.perf_counter() - started
    assert set(entry.metrics) == set(old['normal_retrieval'])
    for metric, value in old['normal_retrieval'].items():
        if metric != 'chance_map':
            np.testing.assert_allclose(entry.metrics[metric], value, rtol=0, atol=1e-12,
                                       err_msg=name + '/' + metric)
    np.testing.assert_allclose(entry.metrics['chance_map'],
                               old['analytical_random_ranking_expected_map'],
                               rtol=0, atol=1e-12, err_msg=name)
    assert entry.sha256 and digest(entry.path) == entry.sha256
    actual_files = plan['models'][run['backbone']]['receipt']['actual_cached_files']
    assert any(record['sha256'] == entry.sha256 for record in actual_files.values())
    assert entry.size_bytes == Path(entry.path).stat().st_size
    assert entry.trained_on and not entry.verified
    results.append({'backbone': run['backbone'], 'cohort': run['cohort'],
                    'normal_model_zoo_entry': asdict(entry), 'elapsed_seconds': elapsed,
                    'every_original_metric_except_corrected_chance_map_exact_atol_1e_12': True,
                    'new_chance_map_matches_independent_original_exact_random_AP_atol_1e_12': True,
                    'actual_weight_bytes_match_frozen_GPU_plan': True,
                    'full_features_sha256': run['full_features']['sha256']})
    print('PASS current normal full-cohort API', name, entry.metrics,
          'checkpoint', entry.sha256, 'seconds', elapsed, flush=True)
    del frame, values
assert digest('spacr/embeddings.py') == source_sha
output = scratch / 'cell-dino-provenance-full-cohort-replay-r1.json'
assert not output.exists()
output.write_text(json.dumps({'passed': True, 'comparisons': results,
                             'original_four_crop_diagnostic': old_four,
                             'fixed_four_crop_diagnostic': new_four,
                             'source_sha256': source_sha,
                             'baseline_source_sha256': digest(baseline_path),
                             'script_sha256': digest(__file__),
                             'no_new_GPU_inference_or_classifier_fit': True}, indent=2) + '\n')
print('PASS all twelve full current-source scorecards and actual checkpoint hashes', flush=True)
