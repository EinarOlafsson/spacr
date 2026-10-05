#!/usr/bin/env python3
"""Replay the cyst/puncta reference and compare candidates and native measurements.

Run under tools/run_capped.sh with CUDA hidden. Original images and parent
masks are read only. Use a new output directory; the tool refuses to replace
existing evidence. Reference rows are compared at their published six-digit
precision and the first field is also compared with the reference detector.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from spacr import tabular
from spacr.qt import cpu_modes
from spacr.tiff_io import write_tiff

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-data', required=True, type=Path, help='Directory containing channel1 and cyst_masks')
parser.add_argument('--reference-data', required=True, type=Path, help='Directory containing the three reference CSV tables')
parser.add_argument('--reference-detector', required=True, type=Path, help='Authoritative center_puncta_detector.py')
parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
args = parser.parse_args()
source, reference, ref_module, out = args.source_data, args.reference_data, args.reference_detector, args.output
out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('analysis_reference', ref_module)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

meta = tabular.read_table(reference/'input_cyst_metadata.csv', canonicalise=False)
gold = tabular.read_table(reference/'all_detection_candidates_including_below_threshold.csv', canonicalise=False)
accepted_gold = tabular.read_table(reference/'puncta_centre_measurements_all_variants.csv', canonicalise=False)
rows, manifests = [], []
started = time.monotonic()
first_exact = None
for position, well in enumerate(sorted(meta.well.unique())):
    name = f'plate1_{well}_T0001F001L01A01Z01C02.tif'
    image_path, parent_path = source/'channel1'/name, source/'cyst_masks'/name
    before = [digest(image_path), digest(parent_path)]
    image, parents = tifffile.imread(image_path), tifffile.imread(parent_path)
    labels, candidates = cpu_modes.puncta(image, parents, measurements=True)
    if position == 0:
        expected = ref.detect_centers(image, parents, ref.CenterSettings(k=2.5, center_pixels=20))
        pd.testing.assert_frame_equal(candidates[expected.columns].reset_index(drop=True), expected,
                                      check_exact=True, check_dtype=False)
        first_exact = {'well': well, 'candidates': len(expected), 'all_reference_columns_bit_exact': True}
    assert before == [digest(image_path), digest(parent_path)], 'Scientific inputs changed'
    accepted = candidates[candidates.included]
    assert np.count_nonzero(np.unique(labels)) == len(accepted)
    assert np.all(labels[parents == 0] == 0)
    masks = out/'masks'
    masks.mkdir(exist_ok=True)
    write_tiff(masks/name, labels.astype(np.uint16))
    candidates['well'] = well
    rows.append(candidates)
    manifests.append(dict(well=well, image=str(image_path), parent=str(parent_path),
                          image_sha256=before[0], parent_sha256=before[1],
                          candidates=len(candidates), included=len(accepted),
                          partial_center_masks=int((accepted.mask_pixels < 20).sum()),
                          mask_path=str(masks/name), mask_sha256=digest(masks/name)))
    print(f'{well}: candidates={len(candidates)} included={len(accepted)} elapsed={time.monotonic()-started:.1f}s', flush=True)

all_rows = pd.concat(rows, ignore_index=True)
keys = ['well', 'cyst_id', 'y', 'x']
actual = all_rows.sort_values(keys).reset_index(drop=True)
expected = gold.sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(actual[keys], expected[keys], check_dtype=False)
errors = {}
reference_columns = ['z', 'sigma', 'peak', 'center3', 'disc13', 'center20', 'corrected20',
                     'cy', 'cx', 'local_bg', 'cyst_bg', 'corrected', 'corrected_cyst', 'noise_mad',
                     'center10', 'center40', 'corr10', 'corr40']
for column in reference_columns:
    np.testing.assert_allclose(actual[column], expected[column], rtol=6e-6, atol=6e-6,
                               err_msg=f'Published six-significant-digit column {column}')
    errors[column] = float(np.max(np.abs(actual[column]-expected[column])))
np.testing.assert_array_equal(actual.included.astype(bool), expected.included.astype(bool))
included = all_rows[all_rows.included].merge(meta[['well', 'object_label', 'experiment', 'cohort', 'condition']],
                  left_on=['well', 'cyst_id'], right_on=['well', 'object_label'], suffixes=('', '_parent'),
                  validate='many_to_one')
gold_keys = accepted_gold.sort_values(keys).reset_index(drop=True)[keys]
pd.testing.assert_frame_equal(included.sort_values(keys).reset_index(drop=True)[keys], gold_keys, check_dtype=False)
counts = {str(k): int(v) for k, v in included.groupby('condition').size().items()}
assert len(all_rows) == 62317 and len(included) == 24145
assert counts == {'ATG2 KO': 10085, "ATG2' KO": 1303, 'ME49': 12757}
tabular.write_table(all_rows, out/'all_candidates.csv', canonicalise=False)
tabular.write_table(included, out/'included_centres.csv', canonicalise=False)
receipt = dict(schema=1, feature=661, fields=len(manifests), input_parent_rows=int(meta.shape[0]), parents_with_included_puncta=int(included[keys[:2]].drop_duplicates().shape[0]),
               candidates=len(all_rows), included=len(included), condition_counts=counts,
               first_field_exact_reference=first_exact, published_reference_keys_exact=True,
               published_numeric_comparison='rtol=6e-6, atol=6e-6; reference CSV rounded to six significant digits',
               maximum_absolute_errors=errors, input_hashes_preserved=True,
               reference_module=str(ref_module), reference_module_sha256=digest(ref_module),
               reference_candidates_sha256=digest(reference/'all_detection_candidates_including_below_threshold.csv'),
               reference_included_sha256=digest(reference/'puncta_centre_measurements_all_variants.csv'),
               implementation_sha256={p: digest(root/p) for p in ('spacr/qt/cpu_modes.py', 'spacr/qt/mask_engine.py', 'spacr/qt/screens/make_masks.py')},
               parameters=cpu_modes.provenance(cpu_modes.PUNCTA, cpu_modes.DEFAULT_PARAMS),
               elapsed_seconds=time.monotonic()-started, fields_provenance=manifests,
               figure_final_statistics_rerun=False, whole_measure_pipeline_rerun=False)
(out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'fields_provenance'}, ensure_ascii=False), flush=True)
