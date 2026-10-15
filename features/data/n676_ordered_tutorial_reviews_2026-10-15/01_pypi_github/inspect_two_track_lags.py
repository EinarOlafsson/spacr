from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import json,hashlib,subprocess,fcntl,os,time,math
import numpy as np
N=Path('/media/carruthers/mnt3/claude/n676_stages');W=N/'work/lesson01-selective-20261015';A=N/'host/stage-23-r60-20261015/release-candidate-append-7ew0avek/media_host/01_pypi_github/audio';CACHE=N/'sentence-audio-cache';SR=24000
ledger=Path('/media/carruthers/mnt3/claude/ram_leases.md');lease=f'codex-lesson01-two-track-lag | narrow observed-correlation CPU followup | 1 GB | owner_pid={os.getpid()} | removed on exit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def compare(template,pcm,where):
 step=4;lower=max(0,where-480);upper=min(len(pcm),where+len(template)+480);a=template[::step].astype(np.float64);b=pcm[lower:upper:step].astype(np.float64)
 if len(b)<len(a):return {'unavailable':True}
 cross=np.correlate(b,a,mode='valid');c=np.r_[0.,np.cumsum(b*b)];norm=np.sqrt(np.maximum(1e-30,(c[len(a):]-c[:-len(a)])*np.dot(a,a)));values=cross/norm;i=int(np.argmax(values));return {'cosine':float(values[i]),'lag_samples':lower+i*step-where,'lag_ms':(lower+i*step-where)/24}
with ledger.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+lease+'\n')
try:
 began=time.monotonic();results=[]
 for language,voice in [('en','bm_fable'),('zh-CN','zm_yunyang')]:
  p=A/language/(voice+'.json');audio=p.with_suffix('.m4a');m=json.loads(p.read_text());rawdecoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(audio),'-map','0:a:0','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,timeout=20);assert rawdecoded.returncode==0;pcm=np.frombuffer(rawdecoded.stdout,dtype='<f4');rows=[]
  for sc in m['scenes']:
   for sent in sc['sentences']:
    assert time.monotonic()-began<60
    inputs={'schema':1,'speech_text':sent['speech_text'],'speed':sc['effective_speed'],'language':language,'lang_code':m['lang_code'],'dialect':m['dialect'],'voice':voice,'sample_rate':m['sample_rate'],'synthesis_runtime':m['render_inputs']['synthesis_runtime']};key=hashlib.sha256(json.dumps(inputs,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();raw=np.load(CACHE/(key+'.npy'),allow_pickle=False);start=round(sent['trimmed_lead']*SR);end=len(raw)-round(sent['trimmed_tail']*SR);trim=raw[start:end];size=min(16800,len(trim));head=compare(trim[:size],pcm,round(sent['speech_start']*SR));tail=compare(trim[-size:],pcm,round(sent['speech_end']*SR)-size)
    rows.append({'scene':sc['scene'],'sentence':sent['sentence'],'text':sent['text'],'head_best':head,'tail_best':tail,'metadata_speech_start':sent['speech_start'],'metadata_speech_end':sent['speech_end'],'retained_raw_duration':len(trim)/SR})
  results.append({'language':language,'voice':voice,'audio':bind(audio),'sidecar':bind(p),'decoded_samples':len(pcm),'decoded_duration':len(pcm)/SR,'sidecar_duration':m['total_duration'],'utterances':rows})
 report={'schema':1,'lesson':'01_pypi_github','scope':'Only two observed low-correlation tracks; lag-only followup, no synthesis or perceptual classification.','initial_inspection':bind(W/'audio-waveform-cache-inspection.json'),'tracks':results,'cpu_wall_seconds':time.monotonic()-began,'lag_search_ms':[-20,20],'sample_step':4,'limits':['Cosine search distinguishes retained-waveform timing offset from gross missing edge support; it does not identify words or establish perceptual completeness.','No claim about unclassified quiet discarded raw prefixes. No audio/source edits.']};out=W/'two-track-lag-inspection.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(bind(out));print([{'voice':r['voice'],'head_min':min(s['head_best']['cosine'] for s in r['utterances']),'tail_min':min(s['tail_best']['cosine'] for s in r['utterances']),'head_lags_ms':sorted(set(s['head_best']['lag_ms'] for s in r['utterances'])),'tail_lags_ms':sorted(set(s['tail_best']['lag_ms'] for s in r['utterances'])),'AAC_extra_duration':r['decoded_duration']-r['sidecar_duration']} for r in results])
finally:
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=lease);f.truncate()
