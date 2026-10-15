from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import os,fcntl,json,hashlib,math,subprocess,collections
import numpy as np
N=Path('/media/carruthers/mnt3/claude/n676_stages');W=N/'work/lesson01-selective-20261015';C=N/'host/stage-23-r60-20261015/release-candidate-append-7ew0avek';A=C/'media_host/01_pypi_github/audio';CACHE=N/'sentence-audio-cache';L='01_pypi_github';SR=24000
ledger=Path('/media/carruthers/mnt3/claude/ram_leases.md');lease=f'codex-lesson01-selective-audio | existing50AAC plus712raw-cache bounded CPU inspection | 1 GB | owner_pid={os.getpid()} | removed on exit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def corr(a,b):
 n=min(len(a),len(b));a=a[:n].astype(np.float64);b=b[:n].astype(np.float64);den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.dot(a,b)/den) if den>0 else None
with ledger.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+lease+'\n')
try:
 manifest=json.loads((C/'release-manifest.json').read_text());assert sha(C/'release-manifest.json')=='8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4';rows={r['path']:r for r in manifest['files']}
 tracks=[];utterances=[];problems=[];lowcor=[]
 for sidecar in sorted(A.glob('*/*.json')):
  m=json.loads(sidecar.read_text());audio=sidecar.with_suffix('.m4a');language=m['language'];voice=m['voice'];ident=f'{language}/{voice}';audio_bind=bind(audio);timing_bind=bind(sidecar)
  for p,b in [(audio,audio_bind),(sidecar,timing_bind)]:
   row=rows[p.relative_to(C).as_posix()];assert (b['sha256'],b['bytes'])==(row['sha256'],row['bytes'])
  assert audio_bind['sha256']==m['media_sha256'] and audio_bind['bytes']==m['media_bytes']
  payload=m['render_inputs'];assert m['render_fingerprint']==hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert payload['pronunciation']['sha256']==sha(ROOT/'tools/tutorials/authoring/tools/pronunciation.py')
  decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(audio),'-map','0:a:0','-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,timeout=45)
  assert decoded.returncode==0,(ident,decoded.stderr.decode(errors='replace'));pcm=np.frombuffer(decoded.stdout,dtype='<f4');assert pcm.size and np.isfinite(pcm).all()
  duration=len(pcm)/SR
  if abs(duration-m['total_duration'])>.1:problems.append({'track':ident,'kind':'decoded_duration_mismatch','metadata':m['total_duration'],'decoded':duration})
  before=len(utterances);cache_bindings=[]
  for scene in m['scenes']:
   for sent in scene['sentences']:
    inputs={'schema':1,'speech_text':sent['speech_text'],'speed':scene['effective_speed'],'language':language,'lang_code':m['lang_code'],'dialect':m['dialect'],'voice':voice,'sample_rate':m['sample_rate'],'synthesis_runtime':payload['synthesis_runtime']}
    key=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();cp=CACHE/(key+'.npy');jp=CACHE/(key+'.json');cache=json.loads(jp.read_text());assert cache['inputs']==inputs and sha(cp)==cache['sha256'];raw=np.load(cp,allow_pickle=False);assert raw.ndim==1 and raw.dtype==np.float32 and len(raw)==cache['samples'] and np.isfinite(raw).all()
    assert cache['phonemes']==sent['phonemes'],(ident,scene['scene'],sent['sentence'])
    frame=120;peak=float(np.max(np.abs(raw)));threshold=max(.00002,peak*.002);active=[(start,min(len(raw),start+frame)) for start in range(0,len(raw),frame) if np.max(np.abs(raw[start:start+frame]))>=threshold]
    assert active,(ident,scene['scene'],sent['sentence'],'no raw activity');start=max(0,active[0][0]-2400);end=min(len(raw),active[-1][1]+4320);trim=raw[start:end]
    expected_start=round(sent['speech_start']*SR);expected_end=round(sent['speech_end']*SR);actual=pcm[expected_start:expected_end]
    drift=max(abs(start/SR-sent['trimmed_lead']),abs((len(raw)-end)/SR-sent['trimmed_tail']),abs(len(trim)/SR-sent['duration']))
    if drift>.002:problems.append({'track':ident,'scene':scene['scene'],'sentence':sent['sentence'],'kind':'raw_cache_trim_or_duration_mismatch','seconds':drift})
    head=corr(trim[:16800:4],actual[:16800:4]);tail=corr(trim[-16800::4],actual[-16800::4]);sample=min(len(trim),len(actual));whole=corr(trim[:sample:4],actual[:sample:4])
    prefix=raw[:start];prefpeak=float(np.max(np.abs(prefix))) if len(prefix) else 0.;prefix_q=int(np.count_nonzero(np.rint(prefix.astype(np.float64)*32768))) if len(prefix) else 0
    record={'language':language,'voice':voice,'scene':scene['scene'],'sentence':sent['sentence'],'text':sent['text'],'speech_text':sent['speech_text'],'phonemes':sent['phonemes'],'cache_key':key,'cache_json':bind(jp),'cache_pcm':bind(cp),'raw_samples':len(raw),'trim_start_samples':start,'trim_end_samples':end,'trim_matches_sidecar_within2ms':drift<=.002,'head_700ms_downsampled_cosine':head,'tail_700ms_downsampled_cosine':tail,'whole_utterance_downsampled_cosine':whole,'discarded_prefix_peak':prefpeak,'discarded_prefix_peak_dbfs':20*math.log10(prefpeak) if prefpeak>0 else None,'discarded_prefix_nonzero_pcm16_samples':prefix_q,'discarded_prefix_classification':'Unclassified low-energy raw samples; no hearing/phoneme completeness judgement.'}
    if head is None or tail is None or min(head,tail)<.85:lowcor.append({'track':ident,'scene':scene['scene'],'sentence':sent['sentence'],'head':head,'tail':tail,'whole':whole})
    utterances.append(record)
  tracks.append({'language':language,'voice':voice,'audio':audio_bind,'sidecar':timing_bind,'render_fingerprint':m['render_fingerprint'],'pronunciation':payload['pronunciation'],'decoded_finite_mono_24k':True,'decode_returncode':0,'decoded_samples':len(pcm),'decoded_duration':duration,'sidecar_duration':m['total_duration'],'duration_difference_seconds':duration-m['total_duration'],'scene_count':len(m['scenes']),'utterances':len(utterances)-before})
 assert len(tracks)==50 and len(utterances)==712
 result={'schema':1,'lesson':L,'scope':'One bounded CPU inspection of existing50AAC tracks and exact712raw utterance cache pairs; no synthesis/model/browser/video decode.','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'baseline_manifest':bind(C/'release-manifest.json'),'tracks':tracks,'utterances':utterances,'concrete_identity_or_duration_problems':problems,'low_raw_to_AAC_edge_correlation_observations':lowcor,'all50_full_AAC_decode_finite':True,'all712_verified_exact_cache_inputs_hashes_phonemes':True,'human_listening_performed':False,'word_completeness_verified':False,'limits':['AAC mastering/lossy coding can alter waveform correlation; a low score is a targeted observation, not automatic clipping or repair proof.','Raw-cache lineage and retained head/tail waveform support do not establish perceptual quality/complete initial consonants; discarded quiet prefixes remain unclassified.','No fullvideo/browser rerun, synthesis, package/model loading/download or source/asset mutation.']}
 out=W/'audio-waveform-cache-inspection.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':bind(out),'tracks':len(tracks),'utterances':len(utterances),'problems':problems,'lowcor_count':len(lowcor),'lowcor_first':lowcor[:4]}),flush=True)
finally:
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=lease);f.truncate()
