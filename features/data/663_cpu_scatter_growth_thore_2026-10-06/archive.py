import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
SCRATCH = Path('/mnt/wd4tb/scratch')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_source(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def compressed(target, data):
    target.write_bytes(gzip.compress(data, mtime=0))


def freeze(target, commit, name, label):
    data = git_source(commit, name)
    compressed(target / (label + '.py.gz'), data)
    return {'commit': commit, 'path': name,
            'sha256': hashlib.sha256(data).hexdigest()}


def copy_files(target, source, names):
    for name in names:
        path = source / name
        if name in ("prepare.py", "measure.py") and (target / name).exists():
            continue
        if path.suffix == '.log':
            compressed(target / (name + '.gz'), path.read_bytes())
        else:
            shutil.copyfile(path, target / name)


def finish(target, receipt, instructions):
    (target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (target / 'REPRODUCE.txt').write_text(instructions)
    shutil.copyfile(Path(__file__), target / 'archive.py')
    manifest = {'sha256': {p.name: digest(p) for p in sorted(target.iterdir())
                           if p.is_file() and p.name != 'manifest.json'}}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


memory = ROOT / 'features/data/548_native_memory_cpu_2026-10-06'
memory.mkdir(parents=True, exist_ok=True)
source = SCRATCH / 'f548-native-memory-20261006'
copy_files(memory, source, ['prepare.py', 'measure.py', 'baseline.log',
                          'optimized.log', 'focused.log', 'coverage.json',
                          'coverage.log'])
identities = {key: freeze(memory, commit, 'spacr/io.py', 'io-' + key)
              for key, commit in [('baseline', 'df7cb31225^'),
                                  ('optimized', 'df7cb31225')]}
outputs = {}
for relative in ['stack/plate1_A01_1_1.npy', 'stack/plate1_A01_1_2.npy',
                 'masks/plate1_A01_1_norm_timelapse.npz']:
    left, right = source / 'baseline' / relative, source / 'optimized' / relative
    assert left.read_bytes() == right.read_bytes(), relative
    outputs[relative] = digest(left)
measurements = {label: json.loads((source / (label + '.log')).read_text().splitlines()[-1])
                for label in ['baseline', 'optimized']}
finish(memory, {
    'sources': identities, 'fixture': {'T': 2, 'Z': 2, 'C': 2,
                                    'YX': [2048, 2048], 'dtype': 'uint16'},
    'measurements': measurements, 'identical_output_sha256': outputs,
    'peak_rss_reduction_mib': round(measurements['baseline']['peak_rss_mib'] -
                                   measurements['optimized']['peak_rss_mib'], 2),
    'scope': 'CUDA-hidden CPU ingest; inference not exercised',
    'limits': ['Whole-field normalization still scales with T/Z/Y/X/C.',
               'This fixture does not qualify arbitrary large acquisitions.',
               'Shared-host wall times are not a speed improvement claim.']},
    'Frozen source and actual paired measurement logs are included.\n'
    'The prepare script regenerates the original scratch fixture and replaces\n'
    'its baseline/optimized directories; run it only in a dedicated scratch copy.\n'
    'Use worktrees at the two recorded source commits and PYTHONPATH=. when\n'
    'running measure.py, CUDA_VISIBLE_DEVICES=\'\', QT_QPA_PLATFORM=offscreen,\n'
    'tools/run_capped.sh 4G python. Compare the three recorded output SHA256s.\n'
    'Run tests/test_native_tzyx_batch_f548.py for channel-subset regressions.\n')

themes = ROOT / 'features/data/663_cpu_scatter_growth_thore_2026-10-06'
themes.mkdir(parents=True, exist_ok=True)
numba = SCRATCH / 'theme-refinement-20261006/numba-integration'
growth = SCRATCH / 'theme-growth-thore-20261006'
thore = SCRATCH / 'thore-fix-20261006'
copy_files(themes, numba, ['parity_perf.json', 'worker_gui_perf.json',
                         'new-source-coverage.json', 'coverage-ambient.json',
                         'parity_perf.py', 'worker_perf.py', 'worker_perf.log',
                         'pytest-01.log', 'pytest-04.log'])
compressed(themes / 'scatter-baseline.py.gz', (numba / 'ambient.py').read_bytes())
sources = {label: freeze(themes, commit, 'spacr/qt/widgets/ambient.py', label)
           for label, commit in [('scatter-production', '204c16be16'),
                                ('growth-footprint', 'bc393abfd2'),
                                ('thore-fix', '539058cc05'),
                                ('live-widget-catalog', '16d073c21b')]}
assert sources['scatter-production']['sha256'] == json.loads(
    (numba / 'new-source-coverage.json').read_text())['source_sha256']
compressed(themes / 'growth-coverage.json.gz', (growth / 'coverage-bc.json').read_bytes())
compressed(themes / 'thore-fix-coverage.json.gz', (thore / 'coverage-final.json').read_bytes())
if not (themes / 'live_widget_probe.py').exists():
    shutil.copyfile(growth / 'live_widget_probe.py', themes / 'live_widget_probe.py')
for path in sorted((growth / 'stills').iterdir()):
    shutil.copyfile(path, themes / path.name)
for name in ['fungal-preview.mp4', 'thore-preview.mp4']:
    shutil.copyfile(growth / name, themes / name)
compressed(themes / 'root-integrated-488-tests.log.gz', (
    SCRATCH / 'theme-refinement-20261006/root-final-ambient-20261006.log').read_bytes())
finish(themes, {
    'sources': sources,
    'exact_native_4k_frame_pairs': 36, 'duplicate_border_primitive_pairs': 6,
    'integrated_root_test_source': '1fe95be155', 'integrated_passes': 488,
    'growth_coverage': {'added_statements': 143, 'touching_arcs': 47},
    'thore_fix_coverage': {'added_statements': 15, 'touching_arcs': 6},
    'limits': ['Native 4K atlas/active lens remain below 24 FPS.',
               'Offline preview videos are not live cadence acceptance.',
               'Gallery source precedes the separately recorded Thore fix.',
               'Reported preference Save native crash is not resolved.',
               'Live-widget probes include capture and control-setter stalls.']},
    'Frozen renderer sources and exact pixel/performance receipts are included.\n'
    'The production scatter receipt binds e00d4027e243e400525983dd12d8d32e57468074655ad1e94b8d85594aec39b8.\n'
    'parity_perf.py and worker_perf.py record original scratch/worktree paths;\n'
    'to relocate, point their baseline/production paths at the matching frozen\n'
    'source files and verify SHA256 before running. Do not use current HEAD\n'
    'as the old tested revision. Use CUDA-hidden offscreen capped 4G Python.\n'
    'Coverage is added-source coverage, not global/full-module acceptance.\n'
    'Gallery images/clips bind growth-footprint bc393abfd2. Live widget script\n'
    'binds catalog16d; later Thore fix539058 changes geometry/speed behavior.\n'
    'Native4K frame targets and the actual Save crash remain OPEN.\n')
print(json.dumps({'memory_artifacts': len(list(memory.iterdir())),
                  'theme_artifacts': len(list(themes.iterdir())),
                  'memory_output_parity': True}))
