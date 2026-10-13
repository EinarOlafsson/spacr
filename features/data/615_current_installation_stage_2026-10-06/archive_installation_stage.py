from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r1'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dest = Path('features/data/615_current_installation_stage_2026-10-06')
dest.mkdir(exist_ok=False)
audit = read(stage / 'catalog-preservation.json')
assert audit['passed'] and audit['retained_count'] == 82 and audit['catalog_count'] == 14
preflight = read(stage / 'narration-preflight.json')
assert preflight['passed'] and len(preflight['all_150_registry_track_sentence_plans']) == 150
assert '206 passed' in (scratch / 'installation-current-tutorial-checks-r1.log').read_text()
identities = ['01_pypi_github', '03_pip_install', '04_platform_installers']
files = []
for folder in ['captures', 'catalog', 'production']:
    files.extend(p for p in sorted((stage / folder).rglob('*')) if p.is_file())
files.extend([stage / 'catalog-preservation.json', stage / 'narration-preflight.json'])
files.extend(sorted(stage.glob('*-focus.json')))
with zipfile.ZipFile(dest / 'source-bound-captures-stage-and-preservation.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path.relative_to(stage)))
for name in ['installation-official-source-capture-r1.log', 'installation-completion-stage-r1.log',
             'installation-narration-preflight-r1.log', 'installation-narration-preflight-r2.log',
             'installation-current-tutorial-checks-r1.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
for name in ['stage_installation_completion.py', 'installation_review_delta.json',
             'preflight_installation_narration.py', 'run_installation_tutorial_checks.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
freeze = {str(path): digest(path) for path in files}
freeze.update({str(p): digest(p) for p in [Path('tools/tutorials/authoring/tools/render_all_voices.py'),
    Path('tools/tutorials/authoring/tools/narration_audio.py'), Path('tools/tutorials/authoring/tools/pronunciation.py')]})
write_path = stage / 'frozen-narration-inputs.json'
write_path.write_text(json.dumps(freeze, indent=2) + '\n')
report = {'item': 615, 'current_installation_lessons_staged': identities, 'source_bound_scenes': 30,
          'reviewed_languages_per_lesson': 13, 'all_39_reviews_source_bound': True,
          'new_changed_scene_translation_reviews': 26, 'unchanged_source_01_reviews_retained': True,
          'all_other_82_complete_lesson_objects_preserved_in_each_of_14_catalogs': True,
          'tutorial_and_translation_guard_checks_passed': 206, 'all_150_track_sentence_plans_passed': True,
          'public_installation_recorded_version': '1.5.1.3',
          'actual_current_public_conda_version': '1.5.0.8',
          'public_and_current_nightly_UI_explicitly_distinguished_in_narration': True,
          'actual_cpu_doctor_backend_pass_GPU_skip_optional_CZI_warning_narration_corrected': True,
          'no_21_tile_public_home_pixels_admitted': True,
          'Mask_07_Conda_02_Home_05_Measure_08_unchanged': True,
          'normal_gpu_turn_label_planned': '615-current-installation-narration-20261006-r1',
          'audio_video_candidate_publication_and_live_acceptance_pending': True,
          'native_windows_macos_installation_or_native_speaker_review_claimed': False,
          'frozen_narration_inputs': freeze,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_current_installation_stage_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation current installation staging: lessons 01/03/04 now use thirty source-bound scenes from fresh official PyPI/GitHub/conda pages, accepted actual public pip/Linux verification and separately identified current-nightly Home/Performance references. Public version 1.5.1.3 and actual conda-forge 1.5.0.8 are verified rather than assumed equal. Narration explicitly explains the older released Home layout; no twenty-one-tile public Home pixels enter the standard lessons. Installer doctor prose now matches its actual CPU backend PASS/GPU SKIP/optional CZI warning rather than claiming failed GPU/PATH checks. Normal staging promotes all thirty-nine source-bound reviews (twenty-six changed-scene review files, thirteen unchanged lesson 01 reviews), and preserves every other eighty-two complete lesson objects in every fourteen-language catalog. Tutorial/translation guards pass 206 cases and all 150 current sentence/voice plans pass preflight. Receipt 615_current_installation_stage_2026-10-06.json archives exact captures, per-frame public/nightly origins, stage, reviews, full preservation, source freeze and terminal evidence. Planned normal GPU turn: 615-current-installation-narration-20261006-r1 for precisely lessons 01/03/04, all eight spoken languages and fifty registry voices per lesson; no other GPU work is queued here. Normal audio/video/candidate/publication/deployed acceptance remains open. Mask 07, Conda 02, Home 05 and Measure 08 stay unchanged. Home retains CPU/Qt/CI/source ownership; protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ['features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: accepted installation stage and exact source freeze archived; publication remains pending', flush=True)
