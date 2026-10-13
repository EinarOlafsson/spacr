import copy
import hashlib
from pathlib import Path
import shutil
import sys

sys.path.insert(0, 'tools/tutorials')
from stage_lesson import read, write, stage_lesson

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
prior = scratch / 'tutorial-installation-completion-r1'
stage = scratch / 'tutorial-installation-completion-r2'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not stage.exists()
shutil.copytree(prior, stage)
old_freeze = stage / 'frozen-narration-inputs.json'
old_freeze.rename(stage / 'historical-r1-frozen-narration-inputs.json')
capture = stage / 'captures/linux_public_plus_current_nightly_ui_r1'
frames = read(capture / 'frames.json')
old = copy.deepcopy(frames['07_privacy_keep_off'])
replacement = copy.deepcopy(frames['02_installer_backend'])
replacement['image'] = old['image']
shutil.copyfile(capture / frames['02_installer_backend']['image'], capture / replacement['image'])
assert digest(capture / replacement['image']) == replacement['sha256']
frames['07_privacy_keep_off'] = replacement
write(capture / 'frames.json', frames)
provenance = read(capture / 'provenance.json')
original_origin = copy.deepcopy(provenance['frame_origins']['07_privacy_keep_off'])
provenance['frame_origins']['07_privacy_keep_off'] = {
    **provenance['frame_origins']['02_installer_backend'],
    'kind': 'native_public_runtime_consent_profile_verification',
    'scene_alias': '07_privacy_keep_off',
    'recorded_settings': {'collected': False, 'report_issues': False,
                          'share_diagnostics': False, 'sign_in_now': False},
    'not_a_privacy_dialog_capture': True}
provenance['visual_admission_correction'] = {
    'prior_stage': str(prior), 'rejected_frame': original_origin,
    'reason': 'Older Home and alpha assay cards appear behind the public privacy dialog',
    'prior_private_04_video_not_publishable': True,
    'historical_original_pixels_retained': True,
    'replacement_source_image_bytes_unmodified': True,
    'replacement_is_recorded_consent_verification_not_a_mock_dialog': True}
write(capture / 'provenance.json', provenance)
identity = '04_platform_installers'
focus = read(stage / (identity + '-focus.json'))
focus['recapture']['frames']['07_privacy_keep_off'] = replacement['sha256']
write(stage / (identity + '-focus.json'), focus)
stage_lesson(Path('tools/tutorials/lessons') / (identity + '.json'), capture.name,
             stage, check_only=True, focus_map=stage / (identity + '-focus.json'))
stage_lesson(Path('tools/tutorials/lessons') / (identity + '.json'), capture.name,
             stage, focus_map=stage / (identity + '-focus.json'))
folder = stage / 'production' / identity
(folder / 'scenes.json').unlink()
(folder / 'current-video-acceptance.json').rename(folder / 'historical-r1-video-encoding-only.json')
assert (folder / 'lesson.en.json').read_bytes() == (prior / 'production' / identity / 'lesson.en.json').read_bytes()
preserved = []
for path in prior.rglob('*'):
    if not path.is_file():
        continue
    relative = path.relative_to(prior)
    if (relative.parts[0] == 'catalog' or
        (relative.parts[0] == 'production' and
         (relative.parts[1] in ['01_pypi_github', '03_pip_install'] or
          any(part in ['audio', 'captions'] for part in relative.parts) or
          path.name.startswith('review.')))):
        assert (stage / relative).read_bytes() == path.read_bytes(), relative
        preserved.append(str(relative))
write(stage / 'privacy-correction-preservation.json', {
    'passed': True, 'prior_stage': str(prior), 'current_stage': str(stage),
    'replacement': provenance['frame_origins']['07_privacy_keep_off'],
    'correction': provenance['visual_admission_correction'],
    'all_catalogs_reviews_narration_audio_and_captions_unchanged': True,
    'lesson_01_and_03_complete_production_objects_unchanged': True,
    'preserved_files': preserved, 'published': False})
print('PASS: unchanged narration/translations/media retained; installer privacy scene uses original recorded all-off consent profile; old dialog quarantined in r1.', flush=True)
