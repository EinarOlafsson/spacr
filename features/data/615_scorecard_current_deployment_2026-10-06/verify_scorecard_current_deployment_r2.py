from pathlib import Path
import hashlib
import json
import re
import urllib.request

base = 'https://einarolafsson.github.io/spacr/'
root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
channels_bytes = urllib.request.urlopen(base + 'channels.json', timeout=60).read()
channels = json.loads(channels_bytes)
assert channels['channels']['nightly']['commit'] == '47a19ab6b6b4a605d394a299f58c6d4c4b9fceae'
verified = {}
api_files = sorted(Path('docs/source/_static/i18n/api').glob('*.json'))
tutorial_files = sorted(Path('docs/source/_extra/tutorials/catalog').glob('*.json'))
assert len(api_files) == 10 and len(tutorial_files) == 14
for local in api_files:
    url = base + 'nightly/_static/i18n/api/' + local.name
    payload = urllib.request.urlopen(url, timeout=60).read()
    assert payload == local.read_bytes(), local
    verified[url] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    print('PASS actual source-current deployed API', local.name, flush=True)
for local in tutorial_files:
    url = base + 'nightly/tutorials/catalog/' + local.name
    payload = urllib.request.urlopen(url, timeout=60).read()
    assert payload == local.read_bytes(), local
    verified[url] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
index = urllib.request.urlopen(base + 'nightly/tutorials/', timeout=60).read()
for attribute in ('audio-root', 'video4k-root', 'web-root'):
    url = re.search(r'data-' + attribute + r'="([^"]+)"', index.decode()).group(1)
    assert url == 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/0889ee14f5a7368a791df27c7331862a6c7741b4'
previous = json.loads(Path('features/data/615_installation_actual_nightly_2026-10-06.json').read_text())
assert previous['actual_decoded_current_installation_Home_Measure_Conda_and_unchanged_Mask_frames_visually_reviewed']
receipt = {'passed': True, 'exact_completed_workflow': 37438733539, 'actual_channels': channels,
           'actual_channels_sha256': hashlib.sha256(channels_bytes).hexdigest(), 'all_ten_API_catalog_bytes_match_current_source': True,
           'all_fourteen_tutorial_catalog_bytes_match_current_source': True, 'actual_deployed_files': verified,
           'live_player_sha256': hashlib.sha256(index).hexdigest(), 'media_commit_unchanged_from_accepted_full_video_and_playback_receipt': '0889ee14f5a7368a791df27c7331862a6c7741b4',
           'prior_full_media_playback_visual_receipt_sha256': hashlib.sha256(Path('features/data/615_installation_actual_nightly_2026-10-06.json').read_bytes()).hexdigest(),
           'no_repeated_full_video_download_or_new_native_host_speaker_claim': True,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'scorecard-current-deployment-r2.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS actual nightly 47a19ab6b deployment: all ten API and fourteen tutorial catalogs exact; unchanged accepted media remains deployed.', flush=True)
