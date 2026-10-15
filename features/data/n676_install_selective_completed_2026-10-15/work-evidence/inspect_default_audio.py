from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import os,fcntl,json,hashlib,subprocess,math
import numpy as np
N=Path('/media/carruthers/mnt3/claude/n676_stages');W=N/'work/lesson02-selective-20261015';C=N/'host/stage-23-r60-20261015/release-candidate-append-7ew0avek';A=C/'media_host/02_install_spacr/audio';CACHE=N/'sentence-audio-cache';Q=Path('/media/carruthers/mnt3/codex/scratch/tutorials-agent-20261008/tutorials-requested-live-r1/production/02_install_spacr/audio-checks.en+es+fr+hi+it+pt-BR+ja+zh-CN.json');SR=24000
ledger=N.parent/'ram_leases.md';lease_name=f'codex-lesson02-default-audio | reused exact50 owning QA; one AAC and52 raw cache pairs CPU | 1 GB | owner_pid={os.getpid()} | removed on exit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def corr(a,b):
 size=min(len(a),len(b));a=a[:size].astype(np.float64);b=b[:size].astype(np.float64);den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.dot(a,b)/den) if den else None
with ledger.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+lease_name+'\n')
try:
 manifest=json.loads((C/'release-manifest.json').read_text());assert sha(C/'release-manifest.json')=='8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4';files={r['path']:r for r in manifest['files']};q=json.loads(Q.read_text());assert q['passed'] is True and q['tracks_checked']==50 and all(not r['errors'] for r in q['tracks']);qa={(r['language'],r['voice']):r for r in q['tracks']};tracks=[]
 for p in sorted(A.glob('*/*.json')):
  m=json.loads(p.read_text());old=Q.parent/'audio'/p.relative_to(A);b=bind(p);r=files[p.relative_to(C).as_posix()];assert(b['sha256'],b['bytes'])==(r['sha256'],r['bytes']);assert old.read_bytes()==p.read_bytes();assert qa[(m['language'],m['voice'])]['audio_sha256']==m['media_sha256'];audio=p.with_suffix('.m4a');mr=files[audio.relative_to(C).as_posix()];assert(mr['sha256'],mr['bytes'])==(m['media_sha256'],m['media_bytes']);tracks.append({'language':m['language'],'voice':m['voice'],'sidecar':b,'historical_sidecar':str(old),'historical_sidecar_bytes_identical':True,'audio_manifest_sha256':mr['sha256'],'audio_manifest_bytes':mr['bytes'],'owning_qa_audio_sha_matches':True})
 assert len(tracks)==50
 side=A/'en/af_heart.json';m=json.loads(side.read_text());audio=side.with_suffix('.m4a');assert sha(audio)==m['media_sha256'];payload=m['render_inputs'];assert m['render_fingerprint']==hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert payload['pronunciation']['sha256']==sha(ROOT/'tools/tutorials/authoring/tools/pronunciation.py')
 d=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(audio),'-map','0:a:0','-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,timeout=45);assert d.returncode==0,d.stderr.decode(errors='replace');pcm=np.frombuffer(d.stdout,dtype='<f4');assert pcm.size and np.isfinite(pcm).all();problems=[];utterances=[];missing=[]
 for scene in m['scenes']:
  for sent in scene['sentences']:
   inputs={'schema':1,'speech_text':sent['speech_text'],'speed':scene['effective_speed'],'language':m['language'],'lang_code':m['lang_code'],'dialect':m['dialect'],'voice':m['voice'],'sample_rate':m['sample_rate'],'synthesis_runtime':payload['synthesis_runtime']};key=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();jp=CACHE/(key+'.json');cp=CACHE/(key+'.npy')
   if not jp.exists() or not cp.exists():missing.append({'scene':scene['scene'],'sentence':sent['sentence'],'key':key});continue
   cache=json.loads(jp.read_text());assert cache['inputs']==inputs and sha(cp)==cache['sha256'];raw=np.load(cp,allow_pickle=False);assert raw.ndim==1 and raw.dtype==np.float32 and len(raw)==cache['samples'] and np.isfinite(raw).all();assert cache['phonemes']==sent['phonemes'];threshold=max(.00002,float(np.max(np.abs(raw)))*.002);active=[(s,min(len(raw),s+120)) for s in range(0,len(raw),120) if np.max(np.abs(raw[s:s+120]))>=threshold];assert active;start=max(0,active[0][0]-2400);end=min(len(raw),active[-1][1]+4320);trim=raw[start:end];actual=pcm[round(sent['speech_start']*SR):round(sent['speech_end']*SR)];drift=max(abs(start/SR-sent['trimmed_lead']),abs((len(raw)-end)/SR-sent['trimmed_tail']),abs(len(trim)/SR-sent['duration']))
   if drift>.002:problems.append({'scene':scene['scene'],'sentence':sent['sentence'],'trim_drift_seconds':drift})
   utterances.append({'scene':scene['scene'],'sentence':sent['sentence'],'text':sent['text'],'speech_text':sent['speech_text'],'phonemes':sent['phonemes'],'cache_metadata':bind(jp),'cache_pcm':bind(cp),'raw_samples':len(raw),'trim_start_samples':start,'trim_end_samples':end,'trim_sidecar_max_error_seconds':drift,'head700ms_cosine':corr(trim[:16800:4],actual[:16800:4]),'tail700ms_cosine':corr(trim[-16800::4],actual[-16800::4]),'whole_utterance_cosine':corr(trim[::4],actual[::4])})
 duration=pcm.size/SR
 if abs(duration-m['total_duration'])>.1:problems.append({'decoded_duration_mismatch':duration-m['total_duration']})
 output={'schema':1,'lesson':'02_install_spacr','owning_qa':bind(Q),'owning_qa_passed':q['passed'],'owning_qa_current_runtime_freshness_claimed':q['current_runtime_freshness_claimed'],'all50_exact_owning_qa_audio_hashes_and_historical_sidecars':tracks,'default_audio':bind(audio),'default_sidecar':bind(side),'default_full_decode_finite':True,'default_decoded_samples':int(pcm.size),'default_decoded_duration':duration,'default_metadata_duration':m['total_duration'],'default_utterances_checked':utterances,'default_cache_missing':missing,'concrete_problems':problems,'baseline_manifest':bind(C/'release-manifest.json'),'human_listening_performed':False,'word_completeness_verified':False,'limits':['Only af_heart decoded anew; exact prior50-track QA and C23 immutable hash readback reused for other49.','Waveform correlation supports raw-to-assembled identity; it does not establish perceptual word/phoneme completeness. Quiet discarded prefixes are not automatically defects.','Historical QA freshness is its original claim, not a newly synthesized/current runtime claim. Historical source-freeze receipt was unavailable at the one known expected path.']};out=W/'default-audio-cache-inspection.json';assert not out.exists();out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':bind(out),'tracks_reused':len(tracks),'default_utterances':len(utterances),'missing_cache':len(missing),'problems':problems,'duration_difference':duration-m['total_duration'],'head_min':min([r['head700ms_cosine'] for r in utterances] or [0]),'tail_min':min([r['tail700ms_cosine'] for r in utterances] or [0])}),flush=True)
finally:
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=lease_name);f.truncate()
