import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-first-open-serial-20261008')
scratch = Path('/mnt/wd4tb/scratch/ci-4133-fast-min-20261008')
out = root / 'features/data/43_plaque_fail_fast_signature_cpu_2026-10-08'
out.mkdir(parents=True, exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
test = 'tests/test_plaque_segmentation_diagnostics.py'
base = '4133beafcd0ae427795617a4a01e295fd40539b7'
(out / 'before-test.py.gz').write_bytes(gzip.compress(subprocess.check_output(['git', 'show', base + ':' + test], cwd=root), mtime=0))
(out / 'after-test.py.gz').write_bytes(gzip.compress((root / test).read_bytes(), mtime=0))
(out / 'focused.log.gz').write_bytes(gzip.compress((scratch / 'full-signature-focused.log').read_bytes(), mtime=0))
for name in ['signature-probe.py', 'signature-probe.json', 'signature-probe.log', 'signature-probe-initial-path-error.log']:
    (out / name).write_bytes((scratch / name).read_bytes())
raw = (scratch / 'fast1.log').read_bytes()
lines = raw.decode().splitlines()
(out / 'hosted-failure-excerpt.txt').write_text('\n'.join(lines[6105:6142]) + '\n')
receipt = {
    'required_head': base, 'required_run': 37728397690, 'failed_job': 113154247469,
    'failed_node': 'tests/test_cellpose_api_contract.py::test_every_converted_double_declares_the_installed_signature',
    'sole_offender': test + ':258 Model.eval',
    'complete_raw_log_sha256': sha(raw), 'complete_raw_log_bytes': len(raw),
    'raw_log_location': '/mnt/wd4tb/scratch/ci-4133-fast-min-20261008/fast1.log',
    'failed_batch': 62, 'selected_batches': 122,
    'repair': 'Declare full literal Cellpose 4.2 signature, preserving sentinel/check and hard fail if model evaluation is reached.',
    'production_changed': False, 'guard_changed': False,
    'focused': {'passed': 4, 'warnings': 3, 'seconds': 30.84, 'memory_cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen',
        'nodes': ['tests/test_cellpose_api_contract.py::test_every_converted_double_declares_the_installed_signature',
                  'tests/test_test_suite_hygiene.py::test_the_cellpose_mocks_do_not_swallow_channel_axis',
                  'tests/test_test_suite_hygiene.py::test_the_cellpose_mock_ratchet_is_empty_and_stays_that_way',
                  'tests/test_plaque_segmentation_diagnostics.py::test_incompatible_flow_and_metric_returns_refuse_before_model_evaluation']},
    'actual_cellpose_version': '4.2.1.1',
    'native_signature_probe': json.loads((scratch / 'signature-probe.json').read_text()),
    'fail_before_evaluation': 'Original pytest.raises(ValueError, match="not both") remains. segment_plaque_image rejects before eval and does not catch evaluation exceptions. Model helper AssertionError/pytest.fail cannot become that expected ValueError.',
    'source_bindings': {p: sha((root / p).read_bytes()) for p in [test, 'tests/test_cellpose_api_contract.py', 'tests/test_test_suite_hygiene.py', 'tests/cellpose_api_contract.py', 'spacr/plaque.py']},
    'limitations': ['Local focused repair only; required run remains genuinely failed.', 'No GPU/model weights or whole-suite acceptance.', 'Initial metadata probe omitted PYTHONPATH and failed before imports; retained error is harness setup only. Corrected PYTHONPATH=. probe passes.']
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(out / 'README.txt').write_text('Cellpose fail-fast double complete-signature repair, 2026-10-08\n\n'
    'Corrected 4133 Fast1 really failed the strict signature sweep: the previous\n'
    'axis-aware double still had **kwargs. Declare all real parameters/defaults\n'
    'without changing the axis sentinel/check, original refusal assertion or\n'
    'hard failure upon any unexpected model call. No production/guard edits.\n'
    'Four exact focused checks pass on actual installed Cellpose 4.2.1.1;\n'
    'separate native metadata comparison proves names/order/defaults, with the\n'
    'intentional axis sentinel substitution. Full failed hosted raw log is\n'
    'preserved in scratch; its complete byte count/SHA and exact excerpt are\n'
    'bound here, with whole-phase archival assigned separately. No hosted\n'
    'green or hardware acceptance is claimed. Initial probe path error retained.\n')
(out / 'reproduce.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
python_bin=${SPACR_PROOF_PYTHON:-python}
tools/run_capped.sh 4G "$python_bin" -m pytest -q -p no:randomly tests/test_cellpose_api_contract.py::test_every_converted_double_declares_the_installed_signature tests/test_test_suite_hygiene.py::test_the_cellpose_mocks_do_not_swallow_channel_axis tests/test_test_suite_hygiene.py::test_the_cellpose_mock_ratchet_is_empty_and_stays_that_way tests/test_plaque_segmentation_diagnostics.py::test_incompatible_flow_and_metric_returns_refuse_before_model_evaluation
PYTHONPATH=. tools/run_capped.sh 4G "$python_bin" features/data/43_plaque_fail_fast_signature_cpu_2026-10-08/signature-probe.py
''')
verify = (root / 'features/data/43_641_first_open_serial_cpu_2026-10-08/verify.py').read_text()
verify = verify.replace("{**receipt['after_bindings'], **receipt['production_bindings']}.items()", "receipt['source_bindings'].items()")
(out / 'verify.py').write_text(verify)
(out / 'assemble.py').write_bytes(Path(__file__).read_bytes())
manifest = {'schema': 1, 'payloads': {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
    for p in sorted(out.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'}}
(out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'payloads': len(manifest['payloads']), 'bytes': sum(p['bytes'] for p in manifest['payloads'].values())}))
