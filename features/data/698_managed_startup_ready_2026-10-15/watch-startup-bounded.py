"""Sixteen read-only minute snapshots, exact source bound, no service mutation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import requests
from huggingface_hub import get_token
S=Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
REPO='einarolafsson/spacr-mask-generator';SOURCE='f457356bbffa1c3166f01ea3e7a40c6ab44fca35'
BASE='https://huggingface.co/api/spaces/'+REPO
session=requests.Session();session.headers['Authorization']='Bearer '+get_token()
last=None;failures=0
for index in range(16):
    row={'time_utc':datetime.now(timezone.utc).isoformat(),'round':index,'expected_source':SOURCE}
    try:
        response=session.get(BASE,timeout=(10,20),allow_redirects=False);response.raise_for_status();info=response.json();assert info['sha']==SOURCE
        response=session.get(BASE+'/runtime',timeout=(10,20),allow_redirects=False);response.raise_for_status();runtime=response.json();hardware=runtime.get('hardware')or{}
        assert hardware.get('requested')=='zero-a10g' and hardware.get('current')in(None,'zero-a10g')
        row.update({'sha':info['sha'],'stage':runtime.get('stage'),'hardware':hardware,'errorMessage':runtime.get('errorMessage')});failures=0
        if row['stage']!=last:
            print(json.dumps(row),flush=True);last=row['stage']
        terminal=row['stage']in('RUNNING','RUNTIME_ERROR','BUILD_ERROR','PAUSED','STOPPED','SLEEPING')
        if terminal:
            cap=2*1024*1024;received=0;deadline=time.monotonic()+20
            try:
                with session.get(BASE+'/logs/run',stream=True,timeout=(10,8),allow_redirects=False)as logs:
                    logs.raise_for_status()
                    with (S/'startup-watch-terminal-run.log').open('xb')as handle:
                        for chunk in logs.iter_content(chunk_size=256):
                            if time.monotonic()>deadline:break
                            allowed=chunk[:cap-received];handle.write(allowed);received+=len(allowed)
                            if received>=cap:break
            except requests.RequestException as error:row['log_capture_stop']=type(error).__name__
            row['run_log_bytes']=received;row['terminal_observation']=True
    except Exception as error:
        failures+=1;row.update({'read_error_type':type(error).__name__,'consecutive_errors':failures});terminal=isinstance(error,AssertionError)or failures>=3
    with(S/'startup-watch-snapshots.jsonl').open('a')as handle:
        handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
    (S/'startup-watch-latest.json').write_text(json.dumps(row,indent=2)+'\n')
    if terminal:
        print(json.dumps({'watch_stop':'terminal_phase_or_guard','observation':row}),flush=True);break
    if index<15:time.sleep(60)
else:
    print(json.dumps({'watch_stop':'bounded16rounds_complete','last':row}),flush=True)
