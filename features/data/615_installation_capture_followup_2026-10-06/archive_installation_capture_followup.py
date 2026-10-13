from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-standard-install-neutral-current-r1'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dest = Path('features/data/615_installation_capture_followup_2026-10-06')
dest.mkdir(exist_ok=True)
linux = stage / 'captures/linux_public_1513_r3'
current = stage / 'current-ui-reference-r2/captures/current_install_openings_r2'
public = read(linux / 'provenance.json')
nightly = read(current / 'provenance.json')
assert public['completed_capture'] and public['gui']['returncode'] == 0
assert public['gui']['normal_window_close'] and not public['gui']['analysis_started']
assert public['installed_identity']['version'] == '1.5.1.3'
assert not public['application_source_modified']
assert nightly['completed_capture'] and not nightly['app_source_modified']
records = {}
with zipfile.ZipFile(dest / 'original-captures.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, root in [('linux-public-1513-r3', linux), ('current-nightly-ui-reference-r2', current)]:
        frames = read(root / 'frames.json')
        for row in frames.values():
            assert digest(root / row['image']) == row['sha256']
        for path in sorted(root.iterdir()):
            if path.is_file():
                archive.write(path, name + '/' + path.name)
        records[name] = {'frames': len(frames), 'provenance_sha256': digest(root / 'provenance.json'),
                         'frame_sha256': {key: row['sha256'] for key, row in frames.items()}}
buttons = read(current / 'frames.json')['00_home']['buttons']
modules = sorted({row['module_key'] for row in buttons if row.get('module_key') and row['module_key'] != '__home__'})
assert len(modules) == 19, modules
assert {'umap', 'embeddings', 'power', 'dose_response'} <= set(modules), modules
source_hashes = {}
frozen = '3053b081512d6f598a1aba76dac2393de31ba48a'
for path in ['spacr/qt/widgets/home.py', 'spacr/qt/preferences.py', 'spacr/qt/app.py']:
    original = subprocess.check_output(['git', 'show', frozen + ':' + path])
    assert Path(path).read_bytes() == original, path
    source_hashes[path] = digest(Path(path))
for name in ['current-neutral-public-linux-capture-r3.log', 'current-public-linux-cards-r3.log',
             'current-installation-ui-openings-r1.log', 'current-installation-ui-openings-r2.log',
             'installation-capture-followup-archive-r1.log', 'installation-capture-followup-archive-r2.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
for name in ['run_installation_namespace.sh', 'capture_current_install_openings.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
report = {'item': 615, 'linux_capture_r3_terminal_returncode': 0,
          'linux_gui_normal_close_returncode': 0, 'public_installed_identity': public['installed_identity'],
          'previous_r2_outer_returncode_143_retained_as_historical_failed_attempt': True,
          'current_ui_reference_capture_terminal_returncode': 0,
          'current_ui_reference_source_frozen_commit': frozen, 'source_hashes': source_hashes,
          'current_ui_visible_modules': modules, 'current_home_and_performance_original_frames_visually_reviewed': True,
          'public_release_home_21_tiles_not_admitted_to_current_standard_lessons': True,
          'current_nightly_ui_19_tile_reference_is_separate_from_public_installation': True,
          'narration_must_identify_current_nightly_reference_and_released_build_layout_difference': True,
          'windows_macos_native_capture_claimed': False, 'installation_media_publication_pending': True,
          'Mask_07_and_Conda_02_unchanged': True, 'captures': records}
report['artifacts'] = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}
Path('features/data/615_installation_capture_followup_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation installation recording follow-up: fresh Linux capture r3 finishes with outer rc=0 and a normal GUI close at rc=0; this closes the r2 outer-shutdown gap while retaining the earlier rc=143 attempt as historical failed evidence. Seven genuine native verification/privacy/Home frames and five normal command-reference/release frames remain source-bound to the actual public 1.5.1.3 installation. A separate fresh current-nightly recording captures the actual nineteen-module alpha-off Home and Preferences Performance, with unchanged application source and visually reviewed original 4K frames. The public release still has an older twenty-one-tile Home; it is not admitted to current standard tutorials. Upcoming installation narration must explicitly identify the current-nightly UI reference rather than imply it came from the public install. Receipt 615_installation_capture_followup_2026-10-06.json archives both original captures, source hashes, wrappers and terminal logs. Full installation lesson staging/narration/publication remains open; Mask 07 and the accepted Conda 02 correction remain unchanged. Home retains CPU/Qt/CI source ownership; all GPU tasks remain workstation-owned through the normal queue.\n'
for path in ['features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: terminal Linux recording and separately identified current-nightly UI reference archived', flush=True)
