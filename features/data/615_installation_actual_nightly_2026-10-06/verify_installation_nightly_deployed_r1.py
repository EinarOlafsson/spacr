from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('--assembled',type=Path,required=True)
parser.add_argument('--workflow',type=int,required=True)
args=parser.parse_args()
scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage=scratch/'tutorial-installation-completion-r2'
candidate=Path((stage/'current-candidate-path.txt').read_text().strip())
read=lambda p:json.loads(Path(p).read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
publication=read(candidate/'publication-receipt.json')
assert publication['commit']=='0889ee14f5a7368a791df27c7331862a6c7741b4'
assembled=args.assembled
channels=read(assembled/'channels.json')
url='https://einarolafsson.github.io/spacr/nightly/tutorials/'
assert urllib.request.urlopen('https://einarolafsson.github.io/spacr/channels.json',timeout=120).read()==(assembled/'channels.json').read_bytes()
files=['index.html','app_v2.js','lesson_catalog.js','module_navigation.js']
files.extend('catalog/'+p.name for p in sorted((assembled/'nightly/tutorials/catalog').glob('*.json')))
verified={}
for relative in files:
    payload=urllib.request.urlopen(url+relative,timeout=120).read()
    assert payload==(assembled/'nightly/tutorials'/relative).read_bytes(),relative
    if relative.startswith('catalog/'):
        assert payload==(Path('docs/source/_extra/tutorials')/relative).read_bytes(),relative
    verified[relative]=sha(payload)
index=(assembled/'nightly/tutorials/index.html').read_bytes()
for attribute in ('audio-root','video4k-root','web-root'):
    match=re.search(r'data-'+attribute+r'="([^"]+)"',index.decode())
    assert match and match.group(1)==publication['media_root']
mobile=read(scratch/'installation-current-nightly-mobile-r1.json')
assert mobile['passed'] and mobile['media_root']==publication['media_root']
assert mobile['index_sha256']==sha(index)
identities=['01_pypi_github','03_pip_install','04_platform_installers','02_conda_install','05_home','07_mask','08_measure']
assert {r['lesson'] for r in mobile['cases']}==set(identities)
catalog=read(assembled/'nightly/tutorials/catalog/lessons_en.json')
lessons={r['id']:r for r in catalog['lessons']}
records={r['path']:r for r in read(candidate/'release-manifest.json')['files']}
fidelity=read(stage/'current-frame-fidelity.json')
frames={identity:[0] for identity in identities}
frames['05_home']=[1063]
frames['02_conda_install']=[2682]
for identity in ('01_pypi_github','03_pip_install','04_platform_installers'):
    visual=read(stage/'production'/identity/'visual.json')
    for i,scene in enumerate(visual['scenes']):
        if 'privacy_keep_off' in scene['image']:
            frames[identity].append(fidelity['lessons'][identity]['scenes'][i]['video_frame'])
reports={}
for identity in identities:
    source=read(Path('tools/tutorials/lessons')/(identity+'.json'))
    assert [r['narration'] for r in source['scenes']]==[r['narration'] for r in lessons[identity]['scenes']]
    case=next(r for r in mobile['cases'] if r['lesson']==identity)
    relative=identity+'/web/'+identity+'_silent.mp4'
    video_url=case['clocks_after_seek']['video_src']
    assert video_url==publication['media_root']+'/'+relative
    output=scratch/(identity+'-installation-nightly-video-r1.mp4')
    assert not output.exists()
    digest=hashlib.sha256();count=0
    with urllib.request.urlopen(video_url,timeout=120) as response,output.open('wb') as stream:
        while block:=response.read(1024*1024):
            stream.write(block);digest.update(block);count+=len(block)
    expected=records['media_host/'+relative]
    assert digest.hexdigest()==expected['sha256'] and count==expected['bytes']
    decoded=[]
    for frame in frames[identity]:
        path=scratch/(identity+f'-installation-nightly-frame-{frame}-r1.png')
        subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-i',str(output),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-vsync','0',str(path)],check=True)
        decoded.append({'frame':frame,'PNG':str(path),'PNG_sha256':sha(path.read_bytes())})
    reports[identity]={'actual_video_url':video_url,'complete_video_sha256':digest.hexdigest(),'bytes':count,'candidate_media_exact':True,'deployed_phone_playback_and_seek_passed':True,'decoded_frames':decoded,'audio_sha256':case['audio_sha256']}
    print('PASS:',identity,'actual full-video SHA, phone playback/seek and decoded frame(s).',flush=True)
result={'passed':True,'completed_docs_workflow':args.workflow,'actual_resolved_nightly_source':channels['channels']['nightly']['commit'],'immutable_media_commit':publication['commit'],'normal_assembled_reference':str(assembled),'actual_deployed_player_and_catalog_files':verified,'mobile_receipt_sha256':sha((scratch/'installation-current-nightly-mobile-r1.json').read_bytes()),'new_installation_and_retained_Measure_Home_Conda_Mask_verified':True,'lessons':reports,'actual_decoded_frame_visual_review_pending':True}
(scratch/'installation-current-nightly-deployed-r1.json').write_text(json.dumps(result,indent=2)+'\n')
