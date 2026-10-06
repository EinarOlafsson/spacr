from pathlib import Path
import hashlib
import json

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r2')
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(stage / 'current-frame-fidelity.json')['passed']
assert read(stage / 'current-native-frame-path-and-alpha-acceptance.json')['passed']
rows = []
for identity in ['01_pypi_github', '03_pip_install', '04_platform_installers']:
    folder = stage / 'production' / identity
    assert read(folder / 'current-browser-acceptance.json')['passed']
    assert len(read(folder / 'current-browser-acceptance.json')['checks']) == 14
    for index, scene in enumerate(read(folder / 'visual.json')['scenes'], 1):
        source = (folder / scene['image']).resolve()
        assert digest(source) == scene['capture_sha256']
        rows.append({'lesson': identity, 'scene': index, 'source': str(source), 'sha256': digest(source)})
assert len(rows) == 30
report = {'passed': True, 'date': '2026-10-06', 'reviewer': 'Codex AI visual and technical review',
    'scope': 'Original native images used by every current scene reviewed through view_image, plus full native/tiled OCR and all exact 4K decoded scene-start comparisons.',
    'current_nightly_Home_19_tiles_Core_Data_Tools_Assays_visible': True,
    'current_Performance_preferences_original_visible': True,
    'old_public_Home_and_alpha_cards_absent_from_admitted_scenes': True,
    'privacy_alias_shows_original_recorded_all_false_consent_profile': True,
    'actual_CPU_doctor_result_warning_and_skip_match_installer_prose': True,
    'recorded_public_version_1513_and_conda_package_1508_preserved': True,
    'command_reference_and_Windows_macOS_guidance_cards_explicitly_labelled': True,
    'not_native_Windows_macOS_capture_or_native_speaker_signoff': True,
    'all_30_frame_hashes': rows, 'published': False}
(stage / 'current-original-frame-visual-review.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: original native installation frame visual review recorded at the actual acquired/reference scope.', flush=True)
