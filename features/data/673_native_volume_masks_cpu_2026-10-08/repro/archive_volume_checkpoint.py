import ast
import difflib
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/gate-anchors-20261008')
out = root / 'features/data/673_native_volume_masks_cpu_2026-10-08'
out.mkdir(parents=True, exist_ok=True)
before, checkpoint, after = 'ec50d0bbb1e', 'ecf6125a077', '2720b462e8a'
apps = ['spacr/qt/mask_engine.py', 'spacr/qt/screens/make_masks.py']
tests = ['tests/test_make_masks_native_volume.py', 'tests/qt/test_make_masks_native_volume_editor.py']


def frozen(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path])


def write(path, data):
    destination = out / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def packed(data):
    return gzip.compress(data, mtime=0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


bindings = {}
for path in apps:
    for label, commit in [('before', before), ('checkpoint', checkpoint), ('after', after)]:
        write('source/' + label + '/' + path + '.gz', packed(frozen(commit, path)))
    bindings[path] = digest(frozen(after, path))
for path in tests:
    bindings[path] = digest(frozen(after, path))
    write('source/after/' + path + '.gz', packed(frozen(after, path)))

logs = [
    'volume-model-first.log', 'volume-model-final.log', 'volume-model-doc-guard.log',
    'volume-model-guards.log', 'volume-gui-first.log', 'volume-gui-second.log',
    'volume-model-gui-followup.log', 'volume-real-interaction.log', 'volume-checkpoint.log',
    'volume-dirty-close.log', 'volume-close-traced.log', 'volume-visual.log',
    'extract-volume-delta.log', 'root-model-ecd381-612.log',
    'root-model-docs-ecd381-612.log', 'root-volume-14e4-612.log',
]
for name in logs:
    write('logs/' + name + '.gz', packed((scratch / name).read_bytes()))
for name in ['volume-model-coverage.json', 'volume-checkpoint-coverage.json', 'volume-close-coverage.json']:
    write('coverage/' + name + '.gz', packed((scratch / name).read_bytes()))
for name in ['volume_visual.py', 'extract_volume_delta.py', 'archive_volume_checkpoint.py']:
    write('repro/' + name, (scratch / name).read_bytes())
for name in ['volume-api-ui-delta.json', 'volume-visual-receipt.json']:
    write(name, (scratch / name).read_bytes())
for name in ['volume-native-anchors.png', 'volume-selected-surfaces.png']:
    write('stills/' + name, (scratch / name).read_bytes())

coverage = json.loads((scratch / 'volume-checkpoint-coverage.json').read_text())
close = json.loads((scratch / 'volume-close-coverage.json').read_text())
gaps = {}
for path in apps:
    baseline = frozen(before, path).decode().splitlines()
    previous = frozen(checkpoint, path).decode().splitlines()
    current = frozen(after, path).decode().splitlines()
    rows = coverage['files'][path]
    if path in close['files']:
        new_rows = close['files'][path]
        mapping = {}
        for tag, a, b, c, d in difflib.SequenceMatcher(a=previous, b=current, autojunk=False).get_opcodes():
            if tag == 'equal':
                mapping.update({a + offset + 1: c + offset + 1 for offset in range(b - a)})

        def mapped_line(line):
            if line == 0:
                return 0
            mapped = mapping.get(abs(line))
            return None if mapped is None else mapped if line > 0 else -mapped

        executed = set(new_rows['executed_lines']) | {mapping[n] for n in rows['executed_lines'] if n in mapping}
        executed_arcs = {tuple(arc) for arc in new_rows['executed_branches']}
        for a, b in rows['executed_branches']:
            pair = mapped_line(a), mapped_line(b)
            if None not in pair:
                executed_arcs.add(pair)
        universe = set(new_rows['executed_lines']) | set(new_rows['missing_lines'])
        arcs = {tuple(arc) for arc in new_rows['executed_branches'] + new_rows['missing_branches']}
        missing = universe - executed
        missing_arcs = arcs - executed_arcs
    else:
        assert previous == current
        missing = set(rows['missing_lines'])
        missing_arcs = {tuple(arc) for arc in rows['missing_branches']}
    changed = set()
    for tag, a, b, c, d in difflib.SequenceMatcher(a=baseline, b=current, autojunk=False).get_opcodes():
        if tag != 'equal':
            changed.update(range(c + 1, d + 1))
    gaps[path] = {
        'changed_missing_lines': sorted(missing & changed),
        'changed_missing_incident_arcs': sorted([list(arc) for arc in missing_arcs if any(n in changed for n in arc)]),
        'policy': 'Checkpoint executions mapped only across byte-identical lines, plus direct final close-case executions; no replaced-line inheritance. Not a whole-package ratchet acceptance.',
    }

prefix = "CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=.:/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly "
commands = {
    'checkpoint_116': prefix + 'tests/test_make_masks_native_volume.py tests/qt/test_make_masks_native_volume_editor.py tests/test_make_masks_yolo_box_backend.py tests/qt/test_make_masks_yolo_boxes.py tests/qt/test_mask_engine_v2.py tests/test_every_callable_in_the_package_is_documented.py --cov=spacr.qt.mask_engine --cov=spacr.qt.screens.make_masks --cov-branch --cov-report=json:/mnt/wd4tb/scratch/gate-anchors-20261008/volume-checkpoint-coverage.json',
    'final_gui_13': prefix + 'tests/qt/test_make_masks_native_volume_editor.py',
    'final_close_3': prefix + 'tests/qt/test_make_masks_native_volume_editor.py::test_dirty_busy_close_requires_explicit_discard_and_cancel_preserves_completed_edit tests/qt/test_make_masks_native_volume_editor.py::test_pending_edit_cancellation_destroys_dialog_without_late_gui_publication tests/qt/test_make_masks_native_volume_editor.py::test_unsaved_close_cancel_preserves_native_dialog_and_geometry --cov=spacr.qt.screens.make_masks --cov-branch --cov-report=json:/mnt/wd4tb/scratch/gate-anchors-20261008/volume-close-coverage.json',
}
write('commands.json', (json.dumps(commands, indent=2) + '\n').encode())
receipt = {
    'feature': 'N673', 'before_checkpoint': before, 'checkpoint_phase_source': checkpoint,
    'source_checkpoint': after, 'source_commits': ['3dc450de27b', '54351ca8439', checkpoint, after],
    'current_bindings': bindings,
    'dependency': {'feature': 'N672', 'source_commits': ['6b89b49b80c', '1ad258e625f'], 'inherited_doc_correction': '22f8c0046041cec731344cb5df9505daeaad2c30', 'local_doc_cherry': before},
    'phases': [
        {'source': checkpoint, 'passed': 116, 'seconds': 48.75, 'log': 'logs/volume-checkpoint.log.gz', 'coverage': 'coverage/volume-checkpoint-coverage.json.gz', 'scope': 'Six selected owning model/GUI, legacy 2D/YOLO and callable-documentation test files; exact command in commands.json.'},
        {'source': after, 'passed': 13, 'seconds': 10.88, 'log': 'logs/volume-dirty-close.log.gz', 'scope': 'Full final native volume editor GUI file, after dirty/busy close guard.'},
        {'source': after, 'passed': 3, 'seconds': 8.23, 'log': 'logs/volume-close-traced.log.gz', 'coverage': 'coverage/volume-close-coverage.json.gz', 'scope': 'Direct branch-traced dirty busy Cancel/Discard, fresh busy native teardown and unsaved idle Cancel tests.'},
    ],
    'phase_count_policy': 'Phases overlap. Do not add counts or relabel the 116-case run as the final changed-close source.',
    'independent_root_phases': [
        {'source': 'ecd38110b55', 'passed': 42, 'seconds': 1.25, 'log': 'logs/root-model-ecd381-612.log.gz', 'scope': 'Root model checkpoint before GUI integration.'},
        {'source': 'ecd38110b55', 'passed': 3, 'seconds': 28.24, 'log': 'logs/root-model-docs-ecd381-612.log.gz', 'scope': 'Root required/doc/ghost guards before GUI integration.'},
        {'source': '14e4bbcb2a9', 'passed': 60, 'seconds': 30.54, 'log': 'logs/root-volume-14e4-612.log.gz', 'scope': 'Root GUI/model selected integration before final dirty/busy close follow-up.'},
    ],
    'environment': {'python': '3.12.13', 'pytest': '8.4.2 overlay', 'Qt': '6.12.0 overlay', 'cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen'},
    'changed_source_gap_audit': gaps,
    'bounded_work': {'max_source_voxels': 16777216, 'max_source_decoded_bytes': 268435456,
                     'membership_chunk_points': 4096, 'max_history_bytes': 67108864,
                     'max_history_steps': 25, 'display_preview_points': 4000,
                     'max_boundary_vertices': 8192, 'max_boundary_triangles': 16384},
    'demonstrated': [
        'Independent original-pixel voxel oracle over all six XYZ storage permutations and anisotropic physical spacing.',
        'Native rotated x/y/z anchor edits and shared multi-surface wheel edits change the full mask; exact undo/redo.',
        'Actual rotated rectangle-through-view drawing and native dialog source-bound save/reopen.',
        'Original source bytes, integer label IDs, dtype, array shape/storage axes and spacing are preserved.',
        'Wrong mask header shape/dtype is refused before decoded allocation; external source/mask/sidecar changes during staging refuse publication.',
        'Boundary/voxel disagreement, overlapping labels, invalid/nonfinite geometry, end-on dragging and unsupported existing-label topology refuse without scientific mutation.',
        'Pending-worker cancellation and dirty busy Cancel/Discard preserve completed voxels and prevent late GUI publication.',
        'Actual offscreen screenshots leave zero top-level widgets, native dialog destroyed and worker idle.',
    ],
    'limits': [
        'No GPU, native-display aesthetics, native4K animation/FPS, whole Qt suite, whole numerical-ratchet or absolute RSS/runtime performance acceptance.',
        'Memory/work budgets are structural limits, not measured end-to-end RAM savings; no RSS benchmark was run.',
        'Supported input is one real numeric 3D TIFF series with explicit spatial axes XYZ permutation; T/C/RGB, guessed axes and oversized inputs are refused.',
        'Existing labels need a bounded connected manifold editable boundary; unrepresentable/disconnected/overdetailed shapes are refused, never replaced with a bounding box.',
        'The point cloud is a deterministic bounded display preview only. Actual scientific counts/export use every original voxel centre and the same edited mesh.',
        'Two-filename TIFF/JSON publication is guarded against staged file changes and rolls back owned publication failures; it is not universally atomic against arbitrary uncooperative concurrent writers.',
        'Normal API/UI delta is scoped to two changed files. Full canonical catalogs/tutorials and hosted acceptance remain workstation/root responsibilities.',
        'Historical failed harness logs are preserved: missing QMessageBox stub, too-small voxel-motion witness and already-deleted fixture teardown. Earlier unrelated required-doc omissions were fixed by root; not waived.',
    ],
}
write('receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
write('README.txt', b'''N673 native 3D Make Masks CPU checkpoint

The committed model and real Qt editor use N672's shared physical boundary for
rendering, voxel containment, edits, undo and persistence. Ordinary 2D and YOLO
paths remain separate and passed the selected legacy regression files.

Evidence is explicitly phased: 116 tests at ecf, followed by 13 final GUI tests
and three branch-traced close tests at 2720. These overlap; do not sum them.
The final close follow-up protects completed unsaved edits while another edit
is busy. Cancel preserves the native dialog/worker/session; explicit Discard
cancels the pending operation. All actual regression files are frozen here.

The two 1300x900 stills are actual offscreen Qt captures. Capture saved/reopened
exact anisotropic ZYX labels and geometry, preserved the source bytes and
ended with a destroyed native dialog, idle worker and zero top-level widgets.
There is no GPU, aesthetics, native4K FPS, whole-suite or measured-RSS claim.

The receipt lists explicit unsupported topology/axis/size cases, publication
race limitations, structural memory bounds and remaining changed-source
coverage gaps. Full normal API/UI regeneration belongs to the workstation;
the included +11 API/+41 UI/+10 objectname delta is scoped to two files only.

Run python verify.py --git after integration to verify immutable payloads,
frozen source/test bytes and current Git source/test bindings. --frozen verifies
only the historical checkpoint if current app bytes have legitimately changed.
Original agent commit objects are not required by the verifier.
''')
write('verify.py', b'''import argparse, gzip, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--git', action='store_true')
p.add_argument('--frozen', action='store_true')
args = p.parse_args()
root = Path(__file__).resolve().parent
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip())
relative = root.relative_to(repo)

def read(path):
    local = root / path
    if local.exists():
        return local.read_bytes()
    if args.git:
        return subprocess.check_output(['git', 'show', 'HEAD:' + str(relative / path)], cwd=repo)
    raise FileNotFoundError(local)

m = json.loads(read('manifest.json'))
r = json.loads(read('receipt.json'))
for item in m['payloads']:
    data = read(item['path'])
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
for path, digest in r['current_bindings'].items():
    frozen = gzip.decompress(read('source/after/' + path + '.gz'))
    assert hashlib.sha256(frozen).hexdigest() == digest, path
    if not args.frozen:
        current = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=repo) if args.git else (repo / path).read_bytes()
        assert hashlib.sha256(current).hexdigest() == digest, path
v = json.loads(read('volume-visual-receipt.json'))
assert v['source_bytes_unchanged'] and v['exact_mask_geometry_roundtrip']
assert v['dialog_native_destroyed'] and v['worker_idle'] and v['top_level_widgets_after'] == 0
print('Verified', len(m['payloads']), 'payloads,', len(r['current_bindings']), 'frozen bindings;', 'current bindings checked' if not args.frozen else 'historical checkpoint only')
''')
payloads = []
for path in sorted(out.rglob('*')):
    if path.is_file() and path.name != 'manifest.json':
        data = path.read_bytes()
        payloads.append({'path': str(path.relative_to(out)), 'bytes': len(data), 'sha256': digest(data)})
write('manifest.json', (json.dumps({'payloads': payloads}, indent=2) + '\n').encode())
print(json.dumps({'payloads': len(payloads), 'bytes': sum(item['bytes'] for item in payloads),
                  'gaps': {path: [len(rows['changed_missing_lines']), len(rows['changed_missing_incident_arcs'])] for path, rows in gaps.items()}}))
