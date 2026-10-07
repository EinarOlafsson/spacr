import gzip
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
scratch = Path('/mnt/wd4tb/scratch/ci-final-repairs-20261006')
target = root / 'features/data/43_cpu_replay_repairs_2026-10-06'
target.mkdir(parents=True, exist_ok=True)
for name in ('combined-repairs.log', 'layout-boundary-cohort.log',
             'layout-fixed-width-before.log', 'layout-responsive.log',
             'flow-boundary-before.log', 'fungal-py39-compatible.log'):
    shutil.copyfile(scratch / name, target / name)
sources = (
    'spacr/qt/widgets/timelapse_preview.py',
    'spacr/qt/screens/make_masks.py',
    'spacr/qt/widgets/flow.py',
    'tests/qt/test_data_art_gravity_radius.py',
    'tests/qt/test_600_popup_layout_and_selection.py',
    'tests/qt/test_the_results_header_wraps_instead_of_overlapping.py',
    'tests/qt/test_fungal_growth_engine.py',
)
bindings = {}
for name in sources:
    data = (root / name).read_bytes()
    bindings[name] = hashlib.sha256(data).hexdigest()
    (target / (Path(name).name + '.gz')).write_bytes(gzip.compress(data, mtime=0))
receipt = {
    'date': '2026-10-06 America/Detroit',
    'source_sha256': bindings,
    'checks': {
        'combined_preview_mask_density': '81 passed in 18.41s',
        'layout_and_shared_flow_neighbors': '190 passed in 12.31s',
        'fungal_lineage': '9 passed; unchanged adjacent-edge assertions use Python3.9-compatible zip',
        'fatal_ruff': 'passed',
        'diff_check': 'passed',
    },
    'scope': [
        'Preview uses established scaled_for with its actual target widget.',
        'Missing mask returns before manual identity hashing; valid-mask history preserved.',
        'Density counts all sampled values and retains strict increasing population and equal shapes.',
        'QRect right is inclusive; exact-fit flow items no longer wrap one pixel early.',
        'One-row test reserves actual preferred width, including 20px wider captions; all ordering, parent and alignment assertions retained.',
        'Narrow windows still wrap; no promise of one row at1600px with every alpha control and arbitrary font.',
    ],
    'hosted_acceptance': 'OPEN; local checks do not establish all-green GitHub',
}
(target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(target / 'README.md').write_text('CPU replay repairs, 2026-10-06\n\n'
    'The six source/test snapshots are hashed in receipt.json. All local checks '
    'ran with CUDA hidden, offscreen Qt and a 4 GiB cap. The 81-case preview, '
    'missing-mask and density cohort and 190-case layout/shared-flow cohort '
    'pass. The original flow method fails both exact-fit boundary cases. '
    'Wider captions also reproduce the original fixed-width layout failure. '
    'The final layout test reserves actual one-row preferred width and keeps '
    'every ordering, parent and alignment assertion; genuinely narrow windows '
    'continue to wrap. Final hosted CI, numerical coverage and uninterrupted '
    'serial acceptance remain open.\n')
shutil.copyfile(Path(__file__), target / 'archive.py')
files = []
for path in sorted(target.iterdir()):
    if path.name == 'manifest.json':
        continue
    data = path.read_bytes()
    files.append({'path': path.name, 'size': len(data),
                  'sha256': hashlib.sha256(data).hexdigest()})
(target / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'payloads': len(files), 'source_sha256': bindings}))
