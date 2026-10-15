from pathlib import Path
import hashlib
import importlib.util
import json
import os

ROOT = Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/installed-cprofile-diagnostic')
SITE = ROOT.parent / 'smoke-preflight/venv/lib/python3.12/site-packages'
spec = importlib.util.spec_from_file_location('unchanged_startup_diagnostic_driver', ROOT / 'spacr_startup_benchmark.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
original_worker = driver.WORKER
prefix = (ROOT / 'worker-prefix.py').read_text()
driver.WORKER = prefix + '\n' + original_worker
(ROOT / 'exact-profiled-worker.py').write_text(driver.WORKER)
(ROOT / 'worker-binding.json').write_text(json.dumps({
    'normal_driver_sha256': hashlib.sha256((ROOT / 'spacr_startup_benchmark.py').read_bytes()).hexdigest(),
    'worker_prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
    'original_worker_sha256': hashlib.sha256(original_worker.encode()).hexdigest(),
    'profiled_worker_sha256': hashlib.sha256(driver.WORKER.encode()).hexdigest(),
    'normal_worker_suffix_exact': driver.WORKER.endswith(original_worker),
    'runs': 1, 'timeout_s': 30, 'offscreen': True, 'record_only': False,
    'release_acceptance': False, 'diagnostic_only': True}, indent=2) + '\n')
os.environ['SPACR_DIAGNOSTIC_PROFILE_DIR'] = str(ROOT)
artifact = driver.run_benchmark(ROOT / 'profiled-navigation.json', runs=1, timeout_s=30,
                                offscreen=True, package_root=SITE)
print(json.dumps({'passed': artifact['passed'], 'registry_count': artifact['registry_count'],
                  'violations': artifact['violations']}, indent=2), flush=True)
raise SystemExit(0 if artifact['passed'] else 1)
