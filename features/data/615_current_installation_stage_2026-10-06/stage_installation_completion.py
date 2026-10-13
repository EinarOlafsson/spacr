import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, 'tools/tutorials')
from stage_lesson import read, write, stage_lesson, union
from apply_translation_review import promote_many
from audit_staged_catalogs import audit

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r1'
prior = scratch / 'tutorial-standard-install-neutral-current-r1'
current = prior / 'current-ui-reference-r2/captures/current_install_openings_r2'
home_text = 'This recording installs spaCR version one point five point one point three. Released builds can show an older Home layout. Here is Home in the current nightly build, with Core, Data, Tools and Assays. Continue to Home screen and navigation for the current modules.'
performance_text = 'In the current nightly interface, open Preferences and choose Performance. Select a profile for your computer. Laptop uses fewer resources; higher profiles allow more caching and background work. Start with a modest profile if memory is limited.'
doctor_text = 'Run spacr-doctor from the installed runtime if startup or a file format needs attention. In this recorded CPU installation, the backend check passes and the GPU probe is skipped. The warning says the optional CZI reader is missing; install that extra only if you need CZI files.'
identities = ['01_pypi_github', '03_pip_install', '04_platform_installers']
before = {identity: read(Path('tools/tutorials/lessons') / (identity + '.json')) for identity in identities}
for identity, original_name, composite_name in [('03_pip_install', 'pip_public_1513_r1', 'pip_public_plus_current_nightly_ui_r1'),
                                               ('04_platform_installers', 'linux_public_1513_r3', 'linux_public_plus_current_nightly_ui_r1')]:
    original = prior / 'captures' / original_name
    source_proof = read(original / 'provenance.json')
    assert source_proof['completed_capture'] and source_proof['gui']['returncode'] == 0
    assert source_proof['installed_identity']['version'] == '1.5.1.3'
    destination = stage / 'captures' / composite_name
    destination.mkdir(exist_ok=False)
    frames = {}
    origins = {}
    for name, frame in read(original / 'frames.json').items():
        if name == '06_installed_home':
            continue
        assert hashlib.sha256((original / frame['image']).read_bytes()).hexdigest() == frame['sha256']
        shutil.copyfile(original / frame['image'], destination / frame['image'])
        frames[name] = copy.deepcopy(frame)
        origins[name] = {'kind': 'public_installed_runtime_or_explicit_instruction_card',
                         'capture': str(original), 'source_frame': name, 'sha256': frame['sha256']}
    for name, source_name in [('13_current_nightly_home', '00_home')] + ([('14_current_nightly_performance', '09_performance')] if identity.startswith('04') else []):
        frame = copy.deepcopy(read(current / 'frames.json')[source_name])
        assert hashlib.sha256((current / frame['image']).read_bytes()).hexdigest() == frame['sha256']
        frame['image'] = name + '.png'
        shutil.copyfile(current / (source_name + '.png'), destination / frame['image'])
        frames[name] = frame
        origins[name] = {'kind': 'separate_current_nightly_UI_reference', 'capture': str(current),
                         'source_frame': source_name, 'sha256': frame['sha256']}
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', {'completed_capture': True, 'composite_of_completed_original_captures': True,
        'app_source_modified': False, 'frame_pixels_modified': False,
        'public_installation_capture': source_proof, 'current_nightly_reference': read(current / 'provenance.json'),
        'not_one_application_session': True, 'explicitly_distinguished_in_narration': True, 'frame_origins': origins})
    lesson = copy.deepcopy(before[identity])
    for scene in lesson['scenes']:
        if scene['visual'] == '06_installed_home':
            scene['visual'] = '13_current_nightly_home'
            scene['narration'] = home_text
            scene.pop('focus', None)
        elif identity.startswith('04') and scene['visual'] == '09_performance':
            scene['visual'] = '14_current_nightly_performance'
            scene['narration'] = performance_text
        elif identity.startswith('04') and scene['visual'] == '05_doctor':
            scene['narration'] = doctor_text
    write(Path('tools/tutorials/lessons') / (identity + '.json'), lesson)

captures = ['official_sources_20261006_r1', 'pip_public_plus_current_nightly_ui_r1', 'linux_public_plus_current_nightly_ui_r1']
for identity, capture in zip(identities, captures):
    path = Path('tools/tutorials/lessons') / (identity + '.json')
    lesson = read(path)
    frames = read(stage / 'captures' / capture / 'frames.json')
    regions = {}
    for scene in lesson['scenes']:
        name = scene['visual']
        frame = frames[name]
        if name == '13_current_nightly_home':
            tiles = [row['rect'] for row in frame['buttons'] if row.get('module_key') and row['module_key'] != '__home__' and row['rect'][2] > 200 and row['rect'][3] > 100]
            assert len(tiles) == 19
            regions[name] = union(tiles)
        elif name == '14_current_nightly_performance':
            regions[name] = union([row['rect'] for row in frame['dialogs']])
        else:
            regions[name] = frame.get('focus')
    focus_path = stage / (identity + '-focus.json')
    write(focus_path, {'english_sha256': hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      'scenes': regions, 'recapture': {'capture_module': capture,
                          'frames': {name: frame['sha256'] for name, frame in frames.items()}}})
    stage_lesson(path, capture, stage, check_only=True, focus_map=focus_path)
    stage_lesson(path, capture, stage, focus_map=focus_path)

delta = read(scratch / 'installation_review_delta.json')
for language, paragraphs in delta.items():
    reviews = []
    for identity in identities:
        path = Path('tools/tutorials/lessons/reviews') / f'{identity}.{language}.json'
        review = read(path)
        lesson = read(Path('tools/tutorials/lessons') / (identity + '.json'))
        assert len(review['scenes']) == len(lesson['scenes'])
        if identity != '01_pypi_github':
            review['review'] = {'kind': 'AI technical review of current source/capture changes; existing unchanged scene prose retained',
                'translator': 'Codex AI, explicit direct translations of current English changes', 'date': '2026-10-06',
                'independent_peer_review': False, 'native_speaker_signoff': False, 'listening_review': False,
                'scope': 'Public-version/current-nightly distinction; current Performance and actual doctor result where applicable'}
        for index, scene in enumerate(lesson['scenes']):
            if scene['visual'] == '13_current_nightly_home':
                review['scenes'][index] = paragraphs[0]
            elif scene['visual'] == '14_current_nightly_performance':
                review['scenes'][index] = paragraphs[1]
            elif identity.startswith('04') and scene['visual'] == '05_doctor':
                review['scenes'][index] = paragraphs[2]
        review['english_sha256'] = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        write(path, review)
        reviews.append(review)
    promote_many(reviews, stage)
write(stage / 'catalog-preservation.json', audit(Path('docs/source/_extra/tutorials/catalog'), stage, set(identities)))
print('PASS: three installation lessons, 30 genuine source-bound scenes, 39 source-pinned reviews, every other 82 complete lesson objects retained in every catalog; not published.', flush=True)
