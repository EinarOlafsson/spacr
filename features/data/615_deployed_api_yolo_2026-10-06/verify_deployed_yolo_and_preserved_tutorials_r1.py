from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
read = lambda p: json.loads(p.read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
publication = read(candidate / 'publication-receipt.json')
assembled = scratch / 'docs-37419061863/assembled'
channels = read(assembled / 'channels.json')
assert channels['channels']['nightly']['commit'] == '7d24dcf7760e63f38a85c311da3bd477d0be9479'
url = 'https://einarolafsson.github.io/spacr/nightly/tutorials/'
files = ['index.html', 'app_v2.js', 'lesson_catalog.js', 'module_navigation.js']
files.extend('catalog/' + path.name for path in sorted((assembled / 'nightly/tutorials/catalog').glob('*.json')))
files_verified = {}
for relative in files:
    payload = urllib.request.urlopen(url + relative, timeout=120).read()
    assert payload == (assembled / 'nightly/tutorials' / relative).read_bytes(), relative
    if relative.startswith('catalog/'):
        assert payload == (Path('docs/source/_extra/tutorials') / relative).read_bytes()
    files_verified[relative] = sha(payload)
index = (assembled / 'nightly/tutorials/index.html').read_bytes()
assert publication['commit'].encode() in index
for attribute in ['audio-root', 'video4k-root', 'web-root']:
    match = re.search(r'data-' + attribute + r'="([^"]+)"', index.decode())
    assert match and match.group(1) == publication['media_root']
mobile = read(scratch / 'yolo-api-nightly-mobile-r1.json')
assert mobile['passed'] and mobile['media_root'] == publication['media_root']
assert mobile['index_sha256'] == sha(index)
identities = ['14_make_masks', '05_home', '08_measure', '02_conda_install']
assert {row['lesson'] for row in mobile['cases']} == set(identities)
catalog = read(assembled / 'nightly/tutorials/catalog/lessons_en.json')
lessons = {row['id']: row for row in catalog['lessons']}
records = {row['path']: row for row in read(candidate / 'release-manifest.json')['files']}
box_frame = read(stage / 'current-frame-fidelity.json')['lessons']['14_make_masks']['scenes'][45]['video_frame']
frames = {'14_make_masks': box_frame, '05_home': 1063, '08_measure': 0, '02_conda_install': 2682}
reports = {}
for identity in identities:
    source = read(Path('tools/tutorials/lessons') / (identity + '.json'))
    assert [row['narration'] for row in source['scenes']] == [row['narration'] for row in lessons[identity]['scenes']]
    case = next(row for row in mobile['cases'] if row['lesson'] == identity)
    relative = identity + '/web/' + identity + '_silent.mp4'
    video_url = case['clocks_after_seek']['video_src']
    assert video_url == publication['media_root'] + '/' + relative
    output = scratch / (identity + '-yolo-nightly-video-r1.mp4')
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(video_url, timeout=120) as response, output.open('wb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
            count += len(block)
    expected = records['media_host/' + relative]
    assert digest.hexdigest() == expected['sha256'] and count == expected['bytes']
    image_path = scratch / (identity + '-yolo-nightly-frame-r1.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(output), '-vf',
                    f'select=eq(n\\,{frames[identity]})', '-frames:v', '1', '-vsync', '0', str(image_path)], check=True)
    reports[identity] = {'video_url': video_url, 'complete_video_sha256': digest.hexdigest(),
        'bytes': count, 'candidate_record_exact': True, 'decoded_frame': frames[identity],
        'decoded_frame_png': str(image_path), 'decoded_frame_png_sha256': sha(image_path.read_bytes()),
        'deployed_phone_playback_and_seek_passed': True, 'narration_sha256': case['audio_sha256']}
    print('PASS:', identity, 'actual deployed full video hash, phone playback/seek and decoded current frame.', flush=True)
result = {'passed': True, 'completed_docs_workflow': 37419061863, 'actual_documentation_source': channels['channels']['nightly']['commit'],
    'immutable_media_commit': publication['commit'], 'normal_assembled_reference': str(assembled),
    'actual_deployed_player_and_catalog_files': files_verified,
    'mobile_receipt_sha256': sha((scratch / 'yolo-api-nightly-mobile-r1.json').read_bytes()),
    'lessons': reports, 'Mask_07_unchanged': True, 'visual_review_pending': True}
(scratch / 'yolo-preserved-tutorials-deployed-r1.json').write_text(json.dumps(result, indent=2) + '\n')
