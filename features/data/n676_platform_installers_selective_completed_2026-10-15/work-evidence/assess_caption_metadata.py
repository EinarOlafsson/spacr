from pathlib import Path
import sys,os,fcntl,subprocess,json,hashlib,math
R=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(R),str(R/'tools'),str(R/'tools/tutorials'),str(R/'tools/tutorials/authoring/tools')]
from apply_translation_review import translated_lesson
from narration_audio import split_sentences
from pronunciation import spoken_form
N=Path('/media/carruthers/mnt3/claude/n676_stages');W=N/'work/lesson04-selective-20261015';B=N/'host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e';REPORT=Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013');L='04_platform_installers';A=B/'media_host'/L/'audio';ledger=N.parent/'ram_leases.md';lease_name=f'codex-lesson04-caption-metadata | source/review/all50smallJSON clocks and primary Git text only | 1 GB | owner_pid={os.getpid()} | removed on exit'
def sha(data):return hashlib.sha256(data).hexdigest()
def bind(p):
 data=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(data),sha256=sha(data))
def read(p):return json.loads(p.read_bytes())
def finite(value):return type(value) in (int,float) and math.isfinite(value)
with ledger.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+lease_name+'\n')
try:
 root_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert root_head=='50da1bd9bfd59f9042019a1e20b602fcfae86c24'
 report_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPORT,text=True).strip()
 manifest=read(B/'release-manifest.json');assert bind(B/'release-manifest.json')['sha256']=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439';files={row['path']:row for row in manifest['files']}
 p=R/'tools/tutorials/lessons'/f'{L}.json';english=read(p);norm=sha(json.dumps(english,sort_keys=True,ensure_ascii=False).encode());assert len(english['scenes'])==14
 sourcefiles=[];reviews={};catalogs=[];localized={}
 for source in [p,*sorted((R/'tools/tutorials/lessons/reviews').glob(f'{L}.*.json'))]:
  relative=source.relative_to(R).as_posix();other=subprocess.check_output(['git','show',report_head+':'+relative],cwd=REPORT);assert source.read_bytes()==other
  sourcefiles.append({**bind(source),'equals_REPORT_commit':report_head})
  if source!=p:
   review=read(source);assert review['english_sha256']==norm and len(review['scenes'])==14;reviews[review['language']]=review
 assert len(reviews)==13 and len(sourcefiles)==14
 for cp in sorted((B/'web/catalog').glob('*.json')):
  language=cp.stem.split('_',1)[1];matches=[x for x in read(cp)['lessons'] if x['id']==L];assert len(matches)==1;item=matches[0];base=dict(item);inventory=base.pop('narration_voices');assert sum(map(len,inventory.values()))==50
  expected=english if language=='en' else translated_lesson(reviews[language],english,base)
  assert base==expected,(cp,'catalog/current source mismatch');localized[language]=item
  catalogs.append(dict(language=language,catalog=bind(cp),lesson_object_sha256=sha(json.dumps(item,sort_keys=True,ensure_ascii=False).encode()),matches_current_source_review=True))
 assert len(catalogs)==14
 tracks=[];scenes_checked=0;sentences_checked=0
 for side in sorted(A.glob('*/*.json')):
  m=read(side);lang=m['language'];voice=m['voice'];b=bind(side);record=files[side.relative_to(B).as_posix()];assert(b['sha256'],b['bytes'])==(record['sha256'],record['bytes'])
  audio=side.with_suffix('.m4a');audio_row=files[audio.relative_to(B).as_posix()];assert(m['media_sha256'],m['media_bytes'])==(audio_row['sha256'],audio_row['bytes'])
  inputs=m['render_inputs'];assert inputs['lesson']==L and inputs['language']==lang and inputs['voice']==voice and inputs['dialect']==m['dialect'];assert sha(json.dumps(inputs,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())==m['render_fingerprint'];assert inputs['pronunciation']['sha256']==bind(R/'tools/tutorials/authoring/tools/pronunciation.py')['sha256']
  assert len(m['scenes'])==len(inputs['scenes'])==14;expected=localized[lang]['scenes'];previous_end=0.;utterances=0
  for index,(scene,authored,render_scene) in enumerate(zip(m['scenes'],expected,inputs['scenes']),1):
   text=authored['narration'];assert scene['scene']==index and scene['text']==text and render_scene['narration']==text
   assert scene['speech_text']==spoken_form(text,lang,m['dialect'])
   sentence_texts=split_sentences(text,lang);assert [s['text'] for s in scene['sentences']]==sentence_texts
   assert render_scene['speech_text']==[spoken_form(t,lang,m['dialect']) for t in sentence_texts]
   for key in ['speech_start','speech_end','scene_end','duration']:assert finite(scene[key])
   assert abs(scene['speech_start']-previous_end)<.002 and scene['speech_start']<scene['speech_end']<=scene['scene_end']
   assert abs(scene['duration']-(scene['scene_end']-scene['speech_start']))<.002
   previous_sentence_end=scene['speech_start']
   for si,sent in enumerate(scene['sentences'],1):
    assert sent['sentence']==si and sent['speech_text']==spoken_form(sent['text'],lang,m['dialect']) and sent['phonemes'].strip()
    for key in ['speech_start','speech_end','audible_start','audible_end','duration','trimmed_lead','trimmed_tail']:assert finite(sent[key])
    assert previous_sentence_end-.002<=sent['speech_start']<=sent['audible_start']<sent['audible_end']<=sent['speech_end']<=scene['speech_end']+.002
    assert abs(sent['duration']-(sent['speech_end']-sent['speech_start']))<.002 and min(sent['trimmed_lead'],sent['trimmed_tail'])>=0
    previous_sentence_end=sent['speech_end'];utterances+=1
   assert abs(previous_sentence_end-scene['speech_end'])<.002;previous_end=scene['scene_end']
  assert finite(m['total_duration']) and abs(previous_end-m['total_duration'])<.002
  tracks.append(dict(language=lang,voice=voice,sidecar=b,audio_manifest_sha256=audio_row['sha256'],audio_manifest_bytes=audio_row['bytes'],scenes=14,sentences=utterances,render_fingerprint=m['render_fingerprint'],pronunciation=inputs['pronunciation'],all_text_sentence_boundaries_and_identity_checks_passed=True));scenes_checked+=14;sentences_checked+=utterances
 assert len(tracks)==50
 source_snapshot=W/'official-release-source';source_snapshot.mkdir(exist_ok=False);release='444fdbf88bdf4891a130d68347cb4d98cb06d07a';primary=[]
 for relative in ['packaging/online/install_spacr_unix.sh','packaging/online/install_spacr_windows.ps1','packaging/online/spacr_online_installer.nsi','packaging/online/build_macos_online.sh','packaging/online/macos_launcher.c','spacr/install_profile.py']:
  data=subprocess.check_output(['git','show',release+':'+relative],cwd=R);dest=source_snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);primary.append(dict(url='https://github.com/EinarOlafsson/spacr/blob/'+release+'/'+relative,git_commit=release,git_blob=subprocess.check_output(['git','rev-parse',release+':'+relative],cwd=R,text=True).strip(),snapshot=bind(dest)))
 oldspec=N/'host/stage-04-r23/production'/L/'scenes.json';origins=read(oldspec)['capture_source']['frame_origins'];recorded=[dict(visual=k,kind=v['kind'],installed_identity=v.get('installed_identity')) for k,v in origins.items() if v.get('installed_identity')];assert all(x['installed_identity']['version']=='1.5.1.3' for x in recorded)
 j={'schema':1,'lesson':L,'status':'source_metadata_checks_pass','component':'CAPTION/NARRATION metadata and factual-source bindings; final visual-content disposition separate','root_commit':root_head,'REPORT_commit':report_head,'canonical':bind(p),'normalized_english_sha256':norm,'scene_count':14,'source_files':sourcefiles,'official_review_count':13,'published_catalogs':catalogs,'baseline_manifest':bind(B/'release-manifest.json'),'all50_sidecars':tracks,'scenes_checked':scenes_checked,'sentence_rows_checked':sentences_checked,'technical_checks_passed':True,'reused_TODAY_official_release_metadata':dict(url='https://api.github.com/repos/EinarOlafsson/spacr/releases/latest',snapshot=bind(N/'work/lesson01-selective-20261015/primary-github-latest.json')),'official_release_source_bindings':primary,'recorded_version_source':dict(specification=bind(oldspec),installed_origins=recorded),'current_nightly_source_bindings':[bind(R/'spacr/qt/preferences.py'),bind(R/'spacr/qt/app.py')],'limits':['JSON clocks/text/source identities only; no acoustic decode, synthesis or hearing/native-speaker review.','Stored synthesis-runtime identity is historical metadata, not a current new-render claim.','Current official source declarations do not establish a successful install/uninstall on every platform.','ActualpicturedHome category consistency is underseparate video review; sourceOrganism literal does not prove visibleincapture.']}
 out=W/'caption-source-metadata-checks.json';assert not out.exists();out.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':bind(out),'tracks':len(tracks),'scenes':scenes_checked,'sentence_rows':sentences_checked,'english_raw':bind(p)['sha256'],'english_normalized':norm}),flush=True)
finally:
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=lease_name);f.truncate()
