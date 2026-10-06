from pathlib import Path
import ast
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-standard-install-neutral-current-r1'
dest = Path('features/data/615_public_installation_checkpoint_2026-10-06')
dest.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
installations = {}
for route, folder, count in [('pip', 'public-pypi-1x7rw565', 6), ('linux_cpu', 'linux-runtime-6pafwa0_', 3)]:
    root = stage / 'installation_runs' / folder
    receipt = read(root / 'receipt.json')
    assert receipt['accepted'] and len(receipt['steps']) == count
    assert all(row['returncode'] == 0 and row['completed'] for row in receipt['steps'])
    target = dest / route
    target.mkdir(exist_ok=True)
    shutil.copyfile(root / 'receipt.json', target / 'receipt.json')
    for path in root.glob('*.log'):
        (target / (path.name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    installations[route] = {'passed': count, 'receipt': str(target / 'receipt.json')}
release = stage / 'installation_runs/linux-installer-frzg4gdf/receipt.json'
assert read(release)['accepted']
shutil.copyfile(release, dest / 'verified-release-installer.json')
captures = {}
with zipfile.ZipFile(dest / 'native-captures.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('pip_public_1513_r1', 'linux_public_1513_r2', 'release_1513_r1'):
        root = stage / 'captures' / name
        receipt = read(root / 'provenance.json')
        assert receipt['completed_capture']
        frames = read(root / 'frames.json')
        for row in frames.values():
            image = root / row['image']
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row['sha256']
        for path in sorted(root.iterdir()):
            if path.is_file():
                archive.write(path, name + '/' + path.name)
        captures[name] = {'frames': len(frames), 'inner_capture_complete': True,
                          'application_modified': False, 'published': False}
assert '18 passed' in (scratch / 'current-installation-helper-tests-r1.log').read_text()
for name in ('current-neutral-public-pip-installation-r1.log', 'current-neutral-public-linux-installation-r2.log',
             'current-neutral-public-pip-capture-r1.log', 'current-neutral-public-linux-capture-r1.log',
             'current-neutral-public-linux-capture-r2.log', 'current-public-linux-installation-r1.log',
             'current-public-release-capture-r1.log', 'current-public-linux-cards-r1.log',
             'current-public-linux-cards-r2.log', 'current-neutral-public-pip-cards-r1.log',
             'current-installation-helper-tests-r1.log'):
    path = scratch / name
    assert path.is_file()
    (dest / (name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
helpers = ['tools/tutorials/check_linux_installation.py', 'tools/tutorials/capture_pip_installation.py',
           'tools/tutorials/render_installer_instruction_cards.py', 'tools/tutorials/render_pip_instruction_cards.py']
for name in helpers:
    ast.parse(Path(name).read_text())
shutil.copyfile(__file__, dest / Path(__file__).name)
receipt = {'accepted_clean_installations': installations, 'public_version': '1.5.1.3',
           'actual_recorded_capture_provenance': captures, 'scoped_helper_tests_passed': 18,
           'helper_source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in helpers},
           'linux_backend_explicitly_requested': 'cpu', 'cuda_install_acceptance': False,
           'linux_capture_outer_returncode': 143, 'linux_inner_gui_returncode': 0,
           'linux_outer_shutdown_unresolved': True,
           'linux_capture_receipt_saved_before_outer_exit': True,
           'public_release_home_has_21_tiles': True, 'current_standard_home_has_19_tiles': True,
           'public_home_native_frames_admitted_to_standard_tutorials': False,
           'tutorial_source_narration_and_publication_pending': True,
           'windows_or_macos_native_install_claimed': False,
           'Mask_07_and_Conda_02_unchanged': True,
           'artifacts': {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                         for p in sorted(dest.rglob('*')) if p.is_file()}}
Path('features/data/615_public_installation_checkpoint_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation public-installation checkpoint: current public 1.5.1.3 installs pass normally in isolated neutral paths: six pip steps and three Linux installer steps, with CPU explicitly selected for the latter. The Linux runner keeps the real home/root read-only and confines writes to its private installation; the native verifier now pins the actual requested backend. Normal cards use the established repository style and the verified installed version; eighteen scoped helper checks pass. Pip recording closes normally. Linux recording saves a completed seven-frame native receipt and closes its GUI at zero, but the outer recording exits 143 afterwards; outer shutdown remains unresolved and is not claimed successful. Genuine public release Home shows 21 tiles including older organism cards, so these Home frames are not admitted to current standard tutorials, which require the actual 19-tile alpha-off interface. Windows/macOS are explicitly source-reviewed guidance only. Receipt 615_public_installation_checkpoint_2026-10-06.json retains exact installs, terminal/capture/cards, checksums, failed attempts and the public-version layout distinction. Full lesson staging/narration/publication remains open; Mask 07 and Conda 02 stay unchanged. API repair resumes only the remaining is/fr/ko locales on CPU; all spaCR GPU work remains workstation-owned through the normal turn queue. Home retains CPU CI/coverage/serial Qt ownership.\n'
for path in ('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('Archived normal public installation and bounded helper acceptance, with explicit unresolved outer shutdown and layout/publication scope.', flush=True)
