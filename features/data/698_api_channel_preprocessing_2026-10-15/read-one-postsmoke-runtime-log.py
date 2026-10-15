from pathlib import Path
from datetime import datetime,timezone
import requests,json,time,hashlib,re,traceback
from urllib.parse import urlsplit,urlunsplit
from huggingface_hub import get_token
S=Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015');D=S/'one-postsmoke-runtime-log';D.mkdir(exist_ok=False)
SOURCE='f457356bbffa1c3166f01ea3e7a40c6ab44fca35';BASE='https://huggingface.co/api/spaces/einarolafsson/spacr-mask-generator'
s=requests.Session();s.trust_env=False;s.headers['Authorization']='Bearer '+get_token();state={'scope':'ONE read-only post-timeout exactsource runtime/log snapshot; no inference/retry/restart/forcedGPUstop','started_utc':datetime.now(timezone.utc).isoformat()}
try:
 r=s.get(BASE,timeout=(5,10),allow_redirects=False);r.raise_for_status();info=r.json();assert info['sha']==SOURCE
 r=s.get(BASE+'/runtime',timeout=(5,10),allow_redirects=False);r.raise_for_status();runtime=r.json();assert runtime['sha']==SOURCE and runtime['hardware']['requested']=='zero-a10g'and runtime['hardware']['current']in('zero-a10g',None)
 state['runtime']=runtime;state['expected_source']=SOURCE;received=bytearray();deadline=time.monotonic()+30
 try:
  with s.get(BASE+'/logs/run',stream=True,timeout=(5,8),allow_redirects=False)as response:
   response.raise_for_status()
   for chunk in response.iter_content(128):
    received.extend(chunk[:256*1024-len(received)])
    if len(received)>=256*1024 or time.monotonic()>deadline:break
 except requests.RequestException as error:state['log_stop_type']=type(error).__name__
 raw=bytes(received);(D/'runtime-observed-raw.log').write_bytes(raw)
 text=raw.decode('utf-8','replace')
 def redact(match):
  value=match.group();parsed=urlsplit(value)
  return urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'REDACTED_SIGNED_QUERY'if parsed.query else'',parsed.fragment))
 text=re.sub(r'https?://[^\s"<>]+',redact,text);text=re.sub(r'(?i)Bearer\s+[A-Za-z0-9_.-]+','Bearer REDACTED',text);text=re.sub(r'\bhf_[A-Za-z0-9]{15,}\b','HF_TOKEN_REDACTED',text)
 (D/'runtime-observed-sanitized.log').write_text(text);state.update({'status':'READ_COMPLETE','raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'sanitized_sha256':hashlib.sha256(text.encode()).hexdigest(),'redaction_changed_bytes':text.encode()!=raw,'active_remote_execution':'UNKNOWN; source timeout/cancellation is not proof ofGPUstop'})
except Exception as error:state.update({'status':'READ_FAILED_OR_REFUSED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
state['finished_utc']=datetime.now(timezone.utc).isoformat();(D/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
