#!/usr/bin/env python3
"""Export recorded05 checkpoint after actual hosted verification; never publish."""
from pathlib import Path, PurePosixPath
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import argparse,datetime,hashlib,json,math,os,re,shutil,subprocess
WORK=Path(__file__).resolve().parent
STAGE=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-05-selective-20261015')
BASELINE=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-04-selective-20261015/release-candidate-append-4lb27kuy')
BASELINE_SHA='aef7a3aeab1d0c792ee368bc6fe519377fd485f386c536498d2557b1b14fbe5f'
LESSON='05_home'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate',type=Path,default=os.environ.get('CANDIDATE'))
p.add_argument('--hf-name',default=os.environ.get('HF_NAME','n676-home-selective-20261015'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--acceptance',type=Path,required=True)
p.add_argument('--visual-review',type=Path,required=True)
p.add_argument('--visual-review-sha256',required=True)
p.add_argument('--evidence',type=Path,action='append',default=[])
p.add_argument('--browser-matrix',type=Path,required=True)
p.add_argument('--browser-matrix-sha256',required=True)
p.add_argument('--player-source-proof',type=Path)
p.add_argument('--player-source-proof-sha256')
a=p.parse_args()
if not a.candidate:p.error('Actual recorded candidate required')
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def bind(path):return {'path':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
source_pin=os.environ['SOURCE_COMMIT'];english_sha=os.environ['APPROVED_ENGLISH_SHA256']
capture_pin='87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
assert head()==source_pin and os.environ.get('CUDA_VISIBLE_DEVICES')==''
followup_paths=[]
if source_pin!=capture_pin:
 assert a.player_source_proof and a.player_source_proof_sha256 and sha(a.player_source_proof)==a.player_source_proof_sha256
 proof=read(a.player_source_proof)
 assert proof['status']=='pass' and proof['capture_source_commit']==capture_pin and proof['source_commit']==source_pin
 subprocess.run(['git','merge-base','--is-ancestor',capture_pin,source_pin],cwd=ROOT,check=True)
 changed=subprocess.check_output(['git','diff','--name-only',capture_pin,source_pin],cwd=ROOT,text=True).splitlines()
 assert changed==sorted(proof['changed_paths']) and len(changed)==2
 assert 'tools/tutorials/authoring/web/app_v2.js' in changed
 test=next(x for x in changed if x!='tools/tutorials/authoring/web/app_v2.js')
 assert test.startswith('tools/tutorials/tests/test_') and test.endswith('.py')
 assert set(proof['source_files'])==set(changed)
 for rel,b in proof['source_files'].items():
  assert bind(ROOT/rel)==b and hashlib.sha256(subprocess.check_output(['git','show',source_pin+':'+rel],cwd=ROOT)).hexdigest()==b['sha256']
 for key in ['owning_tests','independent_source_peer']:
  b=proof[key];path=Path(b['path']);assert bind(path)==b;followup_paths.append(path)
 followup_paths.append(a.player_source_proof)
else:assert a.player_source_proof is None and a.player_source_proof_sha256 is None

assert sha(a.acceptance)==os.environ['ACCEPTED_CAPTURE_SHA256']
acceptance=read(a.acceptance)
assert acceptance['lesson']==LESSON and acceptance['accepted'] is True and acceptance['all_selected_clips_accepted'] is True
assert acceptance['whole_fresh_capture'] is False and acceptance['source_commit']==capture_pin and acceptance['english_sha256']==english_sha
selection_path=Path(acceptance['selection']['path'])
assert bind(selection_path)==acceptance['selection']
selection=read(selection_path)
assert selection['accepted'] is True and selection['lesson']==LESSON and selection['whole_fresh_capture'] is False
assert set(acceptance['capture_source_commits'])=={'f9be2391fa3e7d4ef86676bdcf001563517be87a',capture_pin}
voice_freeze_path=Path('/media/carruthers/mnt3/claude/n676_stages/voices-05-selective-20261015/source-freeze.json')
voice_freeze=read(voice_freeze_path)
assert voice_freeze['source_commit']==capture_pin and voice_freeze['selected']==[LESSON] and voice_freeze['source_dirty'] is False
assert voice_freeze['lesson_sha256']=={LESSON:english_sha} and voice_freeze['voice_count']==voice_freeze['track_count']==50
def git_bytes(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
assert subprocess.check_output(['git','rev-parse','f9be2391fa3e7d4ef86676bdcf001563517be87a:spacr'],cwd=ROOT)==subprocess.check_output(['git','rev-parse',source_pin+':spacr'],cwd=ROOT)
dirty=subprocess.check_output(['git','diff','--name-only',source_pin,'--','spacr','tools/tutorials','tests'],cwd=ROOT,text=True).splitlines()
assert not [x for x in dirty if not x.startswith('tools/tutorials/release_candidate/')]
english_path=ROOT/'tools/tutorials/lessons'/f'{LESSON}.json';english=read(english_path)
assert sha(english_path)==english_sha=='d57ea978b09a03132fad3f1d97dc83eabcb2403e607adb55deaaa4ef94ef2800'
normalized=hashlib.sha256(json.dumps(english,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
assert normalized=='f8908880aff368137b2883715d0e2ea4caf7f694a18426d3da1b567451455b38'
assert acceptance.get('canonical_english_sha256',normalized)==normalized
visuals=[s['visual'] for s in english['scenes']];assert len(visuals)==len(set(visuals))==31
assert acceptance['selected_visuals']==selection['selected_visuals']==visuals
selected=selection['rows'];assert len(selected)==31 and [r['visual'] for r in selected]==visuals
assert all(r['accepted'] is True and not r['trimmed'] and not r['looped'] and r['playback_speed']==1 for r in selected)
rows={r['visual']:dict(r,video_sha256=r['source_video']['sha256'],receipt_sha256=r['capture_receipt']['sha256'],capture=r['source_capture_directory']) for r in selected}
native_frames=0
for r in selected:
 gate_path=Path(r['gate']['path']);receipt_path=Path(r['capture_receipt']['path'])
 assert bind(gate_path)==r['gate'] and bind(receipt_path)==r['capture_receipt']
 gate=read(gate_path);receipt=read(receipt_path)
 assert gate['passed'] is True and gate['reasons']==[]
 assert all(gate[k]==[] for k in ('app_freeze_seconds','app_slow_seconds','app_unknown_seconds','app_gui_sampler_gaps_s','app_stall_windows_s','adapter_bypasses'))
 assert gate['cursor']['jump_count']==0 and gate['top_bar']['hidden'] is True
 assert receipt['full_decode_passed'] is True and receipt['audio_streams']==0 and receipt['encoded_fps']=='30/1'
 assert len(receipt['frames'])==r['full_native_frames'] and receipt['sha256']==r['source_video']['sha256']
 assert r['source_start_frame']==0 and r['source_end_frame_exclusive']==r['full_native_frames']
 native_frames+=r['full_native_frames']
assert native_frames==selection['selected_full_native_frames']==23571
assert sum(r['source_commit']==capture_pin for r in selected)==1
assert rows['13h_preferences_apply']['full_native_frames']==984
# Rebind exact existing source/visual/translation/audio/caption/timing peer bytes; no duplicate native decode.
required_names=['clean-apply-native-independent-evidence-peer.json','root-bounded-new-apply-visual-review.json','previous-strict-pass-bounded-visual-peer.json','overview-translation-independent-technical-peer.json','overview-final-source-closure-peer.json','final-caption-source-consistency.json','english24-audio-independent-readback.json','all50-audio-catalog-independent-readback.json','generated-native-timing-independent-readback.json']
required_paths=[WORK/name for name in required_names]
assert sha(required_paths[0])==acceptance['independent_actual_source_peer_sha256']
assert sha(required_paths[1])==acceptance['root_new_apply_bounded_visual_review_sha256']
for path in required_paths:assert path.is_file() and not path.is_symlink()
# Bind all13 exact official reviews and their unchanged source text.
languages=('da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN');review_paths=[]
for lang in languages:
 path=ROOT/'tools/tutorials/lessons/reviews'/f'{LESSON}.{lang}.json';review=read(path)
 assert review['lesson']==LESSON and review['language']==lang and review['english_sha256']==normalized
 assert len(review['scenes'])==31
 relative=path.relative_to(ROOT).as_posix()
 captured_review=subprocess.check_output(['git','show',capture_pin+':'+relative],cwd=ROOT)
 assert sha(path)==hashlib.sha256(captured_review).hexdigest(), 'Official review bytes differ from immutable capture commit'
 review_paths.append(path)
# Finished complete-native media identity plus bounded actual visual review.
timing_path=STAGE/'production'/LESSON/'video/native-live-timings.json';timing=read(timing_path)
assert timing['complete_native_clips'] is True and timing['native_live_video'] is True
assert timing['playback_speed']==1 and timing['looped_or_stretched'] is False and timing['fps']==30
assert len(timing['scenes'])==31 and [r['visual'] for r in timing['scenes']]==visuals
assert len(timing['source_tracks'])==50 and timing['coverage']['accepted'] is True and timing['coverage']['gaps']==[]
assert len({(r['language'],r['voice']) for r in timing['source_tracks']})==50
qa_path=STAGE/'production'/LESSON/'audio-checks.en+es+fr+hi+it+pt-BR+ja+zh-CN.json';qa=read(qa_path)
assert qa['lesson']==LESSON and qa['passed'] is True and qa['tracks_checked']==len(qa['tracks'])==50
assert all(not r['errors'] for r in qa['tracks'])
qa_hashes={(r['language'],r['voice']):r['audio_sha256'] for r in qa['tracks']};assert len(qa_hashes)==50
for track in timing['source_tracks']:
 language,voice=track['language'],track['voice'];audio=STAGE/'production'/LESSON/'audio'/language/(voice+'.m4a');sidecar=audio.with_suffix('.json')
 assert sha(audio)==track['audio_sha256']==qa_hashes[(language,voice)] and sha(sidecar)==track['timing_sha256']
 metadata=read(sidecar);assert metadata['media_sha256']==track['audio_sha256'] and metadata['render_fingerprint']==track['render_fingerprint']

assert len(timing['native_clip_sources'])==timing['coverage']['native_scene_count']==31
assert math.isclose(timing['total_duration'],native_frames/30,abs_tol=1e-7)
for scene,source in zip(timing['scenes'],timing['native_clip_sources']):
 row=rows[scene['visual']];assert source['clip_sha256']==row['video_sha256'] and source['receipt_sha256']==row['receipt_sha256']
 receipt=read(Path(row['capture'])/read(Path(row['capture'])/'frames.json')[scene['visual']]['clip']['receipt'])
 assert math.isclose(scene['complete_native_duration'],len(receipt['frames'])/30,abs_tol=1e-7)
 assert math.isclose(scene['duration'],scene['complete_native_duration'],abs_tol=1e-7)
master_path=STAGE/'production'/LESSON/'video'/f'{LESSON}_silent.mp4'
web_checks=read(STAGE/'web-renditions'/LESSON/'rendition-checks.json')
assert web_checks['accepted'] is True and web_checks['full_decode_passed'] is True and web_checks['all_frame_presentation_times_match'] is True
assert web_checks['frames']==native_frames and web_checks['master_sha256']==sha(master_path)
sampled_path=a.visual_review.resolve();assert sha(sampled_path)==a.visual_review_sha256;sampled=read(sampled_path)
assert sampled['lesson']==LESSON and sampled['status']=='sampled_visual_review_pass' and sampled['master']['sha256']==sha(master_path)
assert sampled['timing']['sha256']==sha(timing_path) and sampled['complete_native_clips'] is True
assert sampled.get('human_listening_signoff') is not True and sampled.get('native_speaker_signoff') is not True
assert sha(a.browser_matrix)==a.browser_matrix_sha256
matrix=read(a.browser_matrix)
assert isinstance(matrix,list) and len(matrix)==73 and len({r['run'] for r in matrix})==73
assert all(r['returncode']==0 and r['run'].startswith(LESSON+'-') for r in matrix)
assert any('-cues-' in r['run'] for r in matrix) and any('-preserve-interval-boundary-eof-' in r['run'] for r in matrix)
assert any('-completed-matrix-' in r['run'] for r in matrix)
# Recorded checkpoint and immutable uploaded media/readback are authoritative.
source=ROOT/'tools/tutorials/release_candidate';candidate=a.candidate.resolve();assert candidate.is_relative_to(STAGE.resolve())
export=a.output.resolve();assert export.is_relative_to(WORK.resolve()) and not export.exists()
checkpoint=read(source/'checkpoint.json');receipt=read(source/'publication-receipt.json');browser=read(source/'published-media-browser-checks.json');manifest=read(candidate/'release-manifest.json');manifest_sha=sha(candidate/'release-manifest.json')
assert sha(BASELINE/'release-manifest.json')==BASELINE_SHA==manifest['baseline_manifest_sha256']
# Exact association: the newly selected candidate must contain THIS staged 05, not another candidate with coherent older receipts.
manifest_rows={r['path']:r for r in manifest['files']};assert len(manifest_rows)==len(manifest['files'])
def associated(relative,actual):
 row=manifest_rows[relative];b=bind(actual)
 assert row['bytes']==b['bytes'] and row['sha256']==b['sha256']
 target=candidate/relative;assert not target.is_symlink() and target.stat().st_size==b['bytes']
 # Actual candidate bytes are checked by the owning candidate verification/readback; association uses its immutable manifest.
 return relative
associated('media_host/'+LESSON+'/video/'+LESSON+'_silent.mp4',master_path)
associated('media_host/'+LESSON+'/video/native-live-timings.json',timing_path)
web_path=STAGE/'web-renditions'/LESSON/'video'/(LESSON+'_silent.mp4')
associated('media_host/'+LESSON+'/web/'+LESSON+'_silent.mp4',web_path)
assert web_checks['rendition_sha256']==manifest_rows['media_host/'+LESSON+'/web/'+LESSON+'_silent.mp4']['sha256']
expected_audio=set()
for track in timing['source_tracks']:
 for suffix in ['.m4a','.json']:
  rel='audio/'+track['language']+'/'+track['voice']+suffix
  expected_audio.add(associated('media_host/'+LESSON+'/'+rel,STAGE/'production'/LESSON/rel))
assert len(expected_audio)==100 and {x for x in manifest_rows if x.startswith('media_host/'+LESSON+'/audio/')}==expected_audio
associated('web/app_v2.js',ROOT/'tools/tutorials/authoring/web/app_v2.js')
assert sha(candidate/'web/app_v2.js')==sha(ROOT/'tools/tutorials/authoring/web/app_v2.js')
expected_inventory={}
for track in timing['source_tracks']:expected_inventory.setdefault(track['language'],[]).append(track['voice'])
expected_inventory={k:sorted(v) for k,v in expected_inventory.items()}

assert manifest['preserved_lessons']==83 and manifest['refreshed_lessons']==[LESSON]
assert manifest['appended_lessons']==[] and manifest['withdrawn_lessons']==[] and manifest['routes']==manifest['ready_lessons']==84
assert checkpoint['release_hold'] is False and checkpoint['media_uploaded'] is True and checkpoint['pages_tree_ready'] is True
assert checkpoint['published'] is False and Path(checkpoint['private_candidate']).resolve()==candidate
assert checkpoint['manifest_sha256']==receipt['manifest_sha256']==browser['manifest_sha256']==manifest_sha
assert sha(source/'release-manifest.json')==manifest_sha
assert sha(source/'publication-receipt.json')==sha(candidate/'publication-receipt.json')
assert sha(source/'published-media-browser-checks.json')==sha(candidate/'checks/published-media-browser-checks.json')
revision=receipt['commit'];assert re.fullmatch('[0-9a-f]{40}',revision)
tag=a.hf_name+'-verified';media_root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'+revision
assert receipt['readback']['commit']==checkpoint['media_revision']['commit']==revision
assert receipt['tag']==checkpoint['media_revision']['tag']==tag and receipt['branch']==a.hf_name
assert receipt['media_root']==browser['media_root']==checkpoint['media_revision']['media_root']==media_root
assert browser['passed'] is True and len(browser['ready_playback_cases'])==84 and all(r['passed'] is True for r in browser['ready_playback_cases'])
ready={r['id'] for r in read(candidate/'web/catalog/lessons_en.json')['lessons'] if r.get('status')!='coming_soon'}
assert {r['lesson'] for r in browser['ready_playback_cases']}==ready and len(ready)==84
pages=ROOT/'docs/source/_extra/tutorials';assert Path(browser['pages_tree']).resolve()==pages.resolve() and browser['index_sha256']==sha(pages/'index.html')
media=[r for r in manifest['files'] if r['path'].startswith('media_host/')];rb=receipt['readback']
assert rb['passed'] is True and rb['files_expected']==rb['metadata_matched']==rb['downloaded_sha256_matched']==receipt['media_files']==len(media)
assert rb['bytes_read_back']==receipt['media_bytes']==manifest['media_host_bytes']==sum(r['bytes'] for r in media)
assert not rb['metadata_failures'] and not rb['download_failures']
def other_media(m):return {r['path']:(r['sha256'],r['bytes']) for r in m['files'] if r['path'].startswith('media_host/') and not r['path'].startswith('media_host/'+LESSON+'/')}
assert other_media(manifest)==other_media(read(BASELINE/'release-manifest.json'))
preserved=[];names=sorted(p.name for p in (BASELINE/'web/catalog').glob('*.json'));assert len(names)==14
assert names==sorted(p.name for p in (candidate/'web/catalog').glob('*.json'))
for name in names:
 old=read(BASELINE/'web/catalog'/name)['lessons'];new=read(candidate/'web/catalog'/name)['lessons']
 assert [r['id'] for r in old]==[r['id'] for r in new] and len(new)==len({r['id'] for r in new})==84
 assert [r for r in old if r['id']!=LESSON]==[r for r in new if r['id']!=LESSON]
 current=next(r for r in new if r['id']==LESSON);authored=dict(current)
 inventory=authored.pop('narration_voices');assert all(len(v)==len(set(v)) for v in inventory.values())
 assert {k:sorted(v) for k,v in inventory.items()}==expected_inventory
 staged=next(r for r in read(STAGE/'catalog'/name)['lessons'] if r['id']==LESSON);staged=dict(staged);staged.pop('narration_voices',None)
 assert authored==staged and len(authored['scenes'])==31 and 'spaceout' not in json.dumps(authored).lower()
 assert sha(source/'web/catalog'/name)==sha(candidate/'web/catalog'/name);preserved.append(name)
files=checkpoint['files'];assert len(files)==len({r['path'] for r in files})==145
checked=[]
for row in files:
 rel=PurePosixPath(row['path']);assert rel.parts and not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==row['path']
 path=source/row['path'];assert path.resolve().is_relative_to(source.resolve())
 assert not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];checked.append(dict(row,role='checkpoint_recorded_file'))
for name in ('checkpoint.json','publication-receipt.json','published-media-browser-checks.json'):
 assert name not in {r['path'] for r in checked};b=bind(source/name);checked.append({'path':name,'bytes':b['bytes'],'sha256':b['sha256'],'role':'bound_record_evidence'})
assert len(checked)==148
for path in [a.acceptance,selection_path,english_path,timing_path,sampled_path,qa_path,voice_freeze_path,*review_paths,*required_paths,a.browser_matrix,*followup_paths,*a.evidence]:
 assert Path(path).is_file() and not Path(path).is_symlink() and Path(path).suffix in {'.json','.txt','.log','.patch','.py','.sh','.png','.csv','.md'} and Path(path).stat().st_size<20_000_000
# Every prerequisite above is read-only; create the export only after all pass.
snapshot=export/'checkpoint';snapshot.mkdir(parents=True)
for row in checked:
 target=snapshot/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/row['path'],target)
 assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'] and sha(source/row['path'])==row['sha256']
assert sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file())==sorted(r['path'] for r in checked)
evidence=[]
paths=[a.acceptance,selection_path,english_path,timing_path,sampled_path,qa_path,voice_freeze_path,*review_paths,*required_paths,a.browser_matrix,*followup_paths,*a.evidence]
for index,path in enumerate(dict.fromkeys(Path(p).resolve() for p in paths)):
 target=export/'review-evidence'/f'{index:02}-{path.name}';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 assert sha(path)==sha(target);evidence.append({'source':bind(path),'snapshot':bind(target)})
assert head()==source_pin and all(sha(source/r['path'])==r['sha256'] for r in checked)
report={'schema':1,'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lesson':LESSON,'source_commit':source_pin,'capture_and_voice_source_commit':capture_pin,'player_source_proof':bind(a.player_source_proof) if a.player_source_proof else None,'browser_matrix':bind(a.browser_matrix),'capture_source_commits':acceptance['capture_source_commits'],'whole_fresh_capture':False,'reused_strict_native_clips':30,'fresh_strict_native_clips':1,'english_sha256':english_sha,'normalized_english_sha256':normalized,'candidate_root':str(candidate),'snapshot_root':str(snapshot),'checkpoint_file_count':145,'evidence_extra_count':3,'snapshot_file_count':148,'checkpoint_bytes':sum(r['bytes'] for r in files),'snapshot_total_bytes':sum(r['bytes'] for r in checked),'files':checked,'review_evidence':evidence,'translation_review_count':13,'baseline_manifest_sha256':BASELINE_SHA,'preserved_other_lessons':83,'preserved_catalogs':preserved,'release_manifest_sha256':manifest_sha,'publication_revision':revision,'publication_tag':tag,'published_media_root':media_root,'native_frame_count':native_frames,'native_duration_seconds':native_frames/30,'timing':bind(timing_path),'master':bind(master_path),'checks':{'record_completed':True,'all_checkpoint_and_snapshot_hashes_and_lengths_match':True,'all84_hosted_playback_cases_passed':True,'complete_50_source_tracks':True,'complete_31_native_clips':True,'readback_metadata_matched':rb['metadata_matched'],'readback_downloaded_sha256_matched':rb['downloaded_sha256_matched']},'limits':['Existing immutable upload/readback/Pages-tree receipts; not a fresh remote verification or liveSpace deployment proof.','All84 route receipt coverage does not claim84current GNOME remakes.','Selection comprises30 reused strict full clips plus1 fresh strict Apply/Revert clip; no wholly fresh capture or causal product-fix claim.','Actual master content review is sampled; audioQA is not human listening/native-speaker acceptance.','Export contains complete generated145+3checkpoint files and bound reviewevidence, not hosted media, private configurations or app patch.']}
(export/'export-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
(export/'README.md').write_text('Recorded05 full-native component refresh:30 reused strict full clips plus1 fresh strict Apply/Revert clip;31scenes/23571frames/785.7seconds,50 current audio assemblies and13 official reviews. Spaceout removed. Other83 lessons preserved. Checkpoint145+3coherent files. No wholly fresh capture, causal product-fix, whole watching, human listening, phoneme-completeness, native-speaker or live Space deployment claim.\n')
print(json.dumps({'export_manifest':bind(export/'export-manifest.json'),'snapshot_file_count':148}))
