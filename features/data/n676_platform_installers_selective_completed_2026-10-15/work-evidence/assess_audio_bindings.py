"""Read-only retained audio comparison; no decoder, synthesizer or app imports."""
import collections,datetime,fcntl,hashlib,json,os,pathlib,re,subprocess,sys
ROOT=pathlib.Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path.insert(0,str(ROOT))
N=pathlib.Path('/media/carruthers/mnt3/claude/n676_stages');W=N/'work/lesson04-selective-20261015';C=N/'host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e';S=N/'host/stage-04-r23/production/04_platform_installers';V=N/'voices-04-r23';LESSON='04_platform_installers';L=pathlib.Path('/media/carruthers/mnt3/claude/ram_leases.md');TAG='codex-lesson04-audio-assessment'
def lease(add):
 with L.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);s=f.read();f.seek(0);f.truncate();f.write(s+f'{TAG} | retained audio hash readback | 2 GB | owner_pid={os.getpid()}\n' if add else ''.join(x for x in s.splitlines(True) if not x.startswith(TAG+' |')));f.flush()
def bind(p):
 p=pathlib.Path(p);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns),str(p)
 return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def read(p):return json.loads(pathlib.Path(p).read_text())
lease(True)
try:
 manifest=bind(C/'release-manifest.json');assert manifest['sha256']=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439';m=read(manifest['path']);mi={r['path']:r for r in m['files']}; qa_path=S/'audio-checks.en+es+fr+hi+it+pt-BR+ja+zh-CN.json';qa=read(qa_path);assert qa['passed'] is True and qa['tracks_checked']==len(qa['tracks'])==50
 freeze=read(V/'source-freeze.json');assert freeze['source_commit']=='77c57aae4773d767343d406a61a5c2f538c0c3d4' and freeze['source_dirty'] is False and freeze['track_count']==freeze['voice_count']==50
 adoption_path=N/'logs/adopt-04_platform_installers-r23.json';adoption=read(adoption_path);ar=next(r for r in adoption['lessons'] if r['lesson']==LESSON);am={(r['language'],r['voice']):r for r in ar['tracks']};qr={(r['language'],r['voice']):r for r in qa['tracks']};assert len(qr)==50==len(am)
 tracks=[];file_rows=[];sentence_counts=collections.Counter();durations=[];runtime_software=set();input_digests=[]
 for key,q in sorted(qr.items()):
  lang,voice=key;assert not q['errors'];files={}
  for ext in ['m4a','json']:
   rel=f'media_host/{LESSON}/audio/{lang}/{voice}.{ext}';actual=bind(C/rel);historic=bind(S/f'audio/{lang}/{voice}.{ext}');rendered=bind(V/f'production/{LESSON}/audio/{lang}/{voice}.{ext}');r=mi[rel]
   assert (actual['sha256'],actual['bytes'])==(r['sha256'],r['bytes'])==(historic['sha256'],historic['bytes'])==(rendered['sha256'],rendered['bytes']),rel
   files[ext]=actual;file_rows.append({'candidate':actual,'prior_owning_qa_stage':historic,'original_render_workspace':rendered,'release_manifest_record':r})
  md=read(files['json']['path']);assert md['media_sha256']==files['m4a']['sha256']==q['audio_sha256']==am[key]['audio_sha256'];assert md['media_bytes']==files['m4a']['bytes'];assert files['json']['sha256']==am[key]['timing_sha256'];assert md['voice']==voice and md['language']==lang
  payload=md['render_inputs'];fingerprint=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert fingerprint==md['render_fingerprint'];assert payload['lesson']==LESSON and payload['voice']==voice and payload['language']==lang
  assert len(md['scenes'])==len(payload['scenes'])==14;assert [r['scene'] for r in md['scenes']]==list(range(1,15));count=sum(len(r['sentences']) for r in md['scenes']);sentence_counts[lang]+=count;durations.append(md['total_duration']);runtime=payload['synthesis_runtime'];assert runtime['voice_pack']['tensor_sha256']==runtime['voice_pack']['loaded_tensor_sha256'];runtime_software.add(json.dumps(runtime['software'],sort_keys=True))
  input_digests.append(fingerprint);tracks.append({'language':lang,'voice':voice,'audio':files['m4a'],'sidecar':files['json'],'original_qa_errors':q['errors'],'render_fingerprint':fingerprint,'render_inputs_sha256_recomputed':True,'source_model_repository':runtime['model']['repository'],'source_model_snapshot':runtime['model']['snapshot'],'source_voice_pack_sha256':runtime['voice_pack']['sha256'],'source_loaded_voice_tensor_sha256':runtime['voice_pack']['loaded_tensor_sha256'],'duration':md['total_duration'],'scenes':14,'sentences':count})
 closed={str(p.relative_to(C)) for p in (C/f'media_host/{LESSON}/audio').glob('*/*') if p.is_file()};assert closed=={r['release_manifest_record']['path'] for r in file_rows} and len(closed)==100
 logs=[]
 for name in ['audio-04_platform_installers-r23.log','audio-04_platform_installers-r23-tool.log','onsets-04_platform_installers-r23.log','onsets-04_platform_installers-r23-tool.log']:
  p=N/'logs'/name;s=p.read_text();
  if '-tool' not in name:assert 'Terminal exit status: 0' in s and 'Source: '+freeze['source_commit'] in s
  if name.startswith('audio-'):assert len(re.findall(r'^\S+/\S+: 0 errors$',s,re.M))==50
  else:assert '"tracks": 50' in s and '"failed_tracks": 0' in s and '"failures": {}' in s
  logs.append(bind(p))
 part=read(N/'logs/voices-04-r23-partitions.json');assert part['passed'] is True and part['terminal_statuses']==[0,0,0,0] and part['source_commit']==freeze['source_commit'];assert set(v for group in part['partitions'] for v in group)=={v for l,v in qr}
 renderlogs=[]
 for i,count in enumerate([13,13,12,12],1):
  p=N/f'logs/voices-04-r23-part{i}.log';s=p.read_text();assert f'complete rendered={count} skipped=0 peak_failures=0' in s and 'Terminal exit status: 0' in s;renderlogs.append(bind(p))
 historical_sources={}
 for name in ['tools/tutorials/verify_staged_audio.py','tools/tutorials/check_narration_onsets.py','tools/tutorials/authoring/tools/verify_audio_release.py','tools/tutorials/authoring/tools/render_all_voices.py','tools/tutorials/authoring/tools/narration_audio.py','tools/tutorials/authoring/tools/sentence_audio_cache.py']:
  r=subprocess.run(['git','show',freeze['source_commit']+':'+name],cwd=ROOT,capture_output=True,check=True);historical_sources[name]={'git_commit':freeze['source_commit'],'bytes':len(r.stdout),'sha256':hashlib.sha256(r.stdout).hexdigest()}
 report={'schema':1,'lesson':LESSON,'component':'audio','status':'technical_readback_pass','decision':'KEEP','decision_scope':'Existing audio technical component only; root owns final content/presentation acceptance. No concrete technical audio repair found.','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':str(C),'candidate_manifest':manifest,'canonical_english':bind(ROOT/f'tools/tutorials/lessons/{LESSON}.json'),'historical_audio_source_commit':freeze['source_commit'],'source_freeze':bind(V/'source-freeze.json'),'owning_qa':bind(qa_path),'owning_qa_logs':logs,'render_partitions':bind(N/'logs/voices-04-r23-partitions.json'),'actual_render_logs':renderlogs,'adoption_receipt':bind(adoption_path),'historical_owning_source_bindings':historical_sources,'checks':{'published_tracks':50,'published_audio_and_sidecars':100,'all100_actual_bytes_match_C02_manifest_priorQA_stage_and_original_render_workspace':True,'all50_media_sha_match_owningQA_sidecar_and_adoption':True,'all50_sidecar_sha_match_adoption':True,'all50_render_fingerprints_match_their_recorded_inputs':True,'all50_recorded_voice_tensor_load_hashes_match':True,'all50_prior_full_decode_QA_rows_zero_errors':True,'prior_sentence_onset_gate_tracks':50,'prior_sentence_onset_gate_failed_tracks':0,'original_render_partitions_terminal':[0,0,0,0],'original_rendered_tracks':50,'original_peak_failures':0,'new_decode_or_synthesis_runs':0},'tracks':tracks,'file_bindings':file_rows,'summary':{'spoken_languages':dict(collections.Counter(l for l,v in qr)),'sentence_intervals_by_language':dict(sentence_counts),'total_sentence_intervals':sum(sentence_counts.values()),'duration_min':min(durations),'duration_max':max(durations),'recorded_runtime_software':[json.loads(x) for x in sorted(runtime_software)]},'limits':['Prior QA and onset evidence is reused only after actual byte identity verification; its then-current-runtime assertion is not restamped as current runtime freshness.','The source-freeze, render inputs, model/voice identifiers and self-fingerprints are bound; raw cached sentence arrays/model contents are not reloaded or independently rehashed.','No human listening, native-language acceptance, pronunciation correctness, complete first phonemes/words or inaudibility claim. RMS onset checks cannot prove linguistic completeness.','No new audio decode, browser playback, network check, synthesis, model inference or installation. Text/factual correctness and all narration clock/review comparisons are independently owned by model_schema_702; companion receipt can be bound separately.'],'human_listening_signoff':False,'native_speaker_signoff':False,'audible_word_completeness_verified':False,'current_runtime_audio_freshness_claimed':False,'resource_cap':'2G; owned lease removed in finally'}
 out=W/'audio-component-assessment.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(bind(out)));print(json.dumps(report['checks']));print(json.dumps(report['summary']))
finally:lease(False)
