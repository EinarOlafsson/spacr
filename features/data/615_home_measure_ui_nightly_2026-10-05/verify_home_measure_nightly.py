from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-home-measure-ui-current-r1'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
read = lambda path: json.loads(path.read_text())
digest = lambda data: hashlib.sha256(data).hexdigest()
publication = read(candidate / 'publication-receipt.json')
url = 'https://einarolafsson.github.io/spacr/nightly/tutorials/'
index = urllib.request.urlopen(url, timeout=120).read()
assembled = scratch / 'home-measure-published-assembled-r1'
channels = read(assembled / 'channels.json')
assert channels['channels']['nightly']['commit'] == '44443b0bc899928164e182f38f7ff1fcf9677507'
expected_index = (assembled / 'nightly/tutorials/index.html').read_bytes()
assert index == expected_index, 'Nightly has not deployed the exact current tutorial tree'
assert urllib.request.urlopen(url + 'app_v2.js', timeout=120).read() == (assembled / 'nightly/tutorials/app_v2.js').read_bytes()
assert publication['commit'].encode() in index
for attribute in ['audio-root', 'video4k-root', 'web-root']:
    match = re.search(r'data-' + attribute + r'="([^"]+)"', index.decode())
    assert match and match.group(1) == publication['media_root']
catalog = urllib.request.urlopen(url + 'catalog/lessons_en.json', timeout=120).read()
assert catalog == Path('docs/source/_extra/tutorials/catalog/lessons_en.json').read_bytes()
lessons = {row['id']: row for row in json.loads(catalog)['lessons']}
for identity in ['05_home', '08_measure']:
    source = read(Path('tools/tutorials/lessons') / (identity + '.json'))
    assert [row['narration'] for row in source['scenes']] == [row['narration'] for row in lessons[identity]['scenes']]
mobile = read(scratch / 'home-measure-ui-nightly-mobile-r1.json')
assert mobile['passed'] and mobile['media_root'] == publication['media_root']
assert mobile['index_sha256'] == digest(index)
assert {row['lesson'] for row in mobile['cases']} == {'05_home', '08_measure'}
records = {row['path']: row for row in read(candidate / 'release-manifest.json')['files']}
reports = {}
for identity, frame in [('05_home', 1063), ('08_measure', 0)]:
    row = next(row for row in mobile['cases'] if row['lesson'] == identity)
    video_url = row['clocks_after_seek']['video_src']
    relative = identity + '/web/' + identity + '_silent.mp4'
    assert video_url == publication['media_root'] + '/' + relative
    output = scratch / (identity + '-nightly-current-video.mp4')
    sha = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(video_url, timeout=120) as response, output.open('wb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            sha.update(block)
            count += len(block)
    expected = records['media_host/' + relative]
    assert sha.hexdigest() == expected['sha256'] and count == expected['bytes']
    image_path = scratch / (identity + '-nightly-current-frame.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(output), '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-vsync', '0', str(image_path)], check=True)
    reports[identity] = {'video_url_from_actual_deployed_player': video_url, 'complete_video_sha256': sha.hexdigest(), 'candidate_media_record_identical': True, 'decoded_frame_index': frame, 'decoded_frame_png': str(image_path), 'decoded_frame_png_sha256': digest(image_path.read_bytes()), 'mobile_playback_and_seek_passed': row['passed'], 'narration_sha256': row['audio_sha256']}
    print(identity, 'actual nightly complete video hash, chapter seek and decoded current frame PASS', flush=True)
result = {'passed': True, 'url': url, 'commit': publication['commit'], 'documentation_source_commit': channels['channels']['nightly']['commit'], 'completed_docs_workflow': 37401660267, 'normal_publisher_assembled_reference': str(assembled), 'index_sha256': digest(index), 'deployed_index_exact': True, 'deployed_player_script_exact': True, 'deployed_English_catalog_exact': True, 'current_authored_narration_exact': True, 'mobile_receipt_sha256': digest((scratch / 'home-measure-ui-nightly-mobile-r1.json').read_bytes()), 'lessons': reports, 'visual_review_pending': True}
(scratch / 'home-measure-ui-nightly-current-r1.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS: exact deployed Home/Measure tutorials; actual decoded frames await visual review', flush=True)
