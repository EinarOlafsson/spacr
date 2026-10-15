"""One separately authorized anonymous public-input technical smoke, never parity."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
import argparse, hashlib, importlib.metadata, io, json, math, os, signal, stat, threading, time, traceback, zipfile

S = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
SOURCE = 'f457356bbffa1c3166f01ea3e7a40c6ab44fca35'
INPUT_SHA = 'e84718d522edef0d3430a15834c6b5cfbcc5ac062e56cb5d29301722a1e98ffb'
API = '/segment_nuclei_from_cellmask'
KEY = 'nuclei_from_cellmask_v1'
WEIGHTS_SHA = '2675553a46e97a7bc4bd2bfe3e954954194fe71ca4e94261e752a02bf0b6eb47'
INPUT = S/'approved-nuclei-image-acquisition/CSA_screen__screen_20250124_133156__plate1_A14_11_1.tif'
FLAGS = {'SPACR_WEB_ZERO_GPU': '1', 'SPACR_WEB_PROVISION_MODELS': '1'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    expected = {'decision': 'AUTHORIZE_ONE_ANONYMOUS_TECHNICAL_SMOKE', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'space_commit': SOURCE, 'input_sha256': INPUT_SHA, 'api_name': API, 'max_submissions': 1}
    assert json.loads(args.authorization.read_text()) == expected
    assert importlib.metadata.version('gradio-client') == '2.5.0'
    assert INPUT.is_file() and not INPUT.is_symlink() and INPUT.stat().st_size == 4132206
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == INPUT_SHA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    import requests, httpx
    from huggingface_hub import get_token
    from gradio_client import Client, handle_file
    D = S/'one-nuclei-hosted-smoke-execution'
    D.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic(); deadline = started + 300
    state = {'scope': 'ONE anonymous hosted technical smoke, not desktop/scientific parity', 'space_source': SOURCE, 'input_sha256': INPUT_SHA, 'model_key': KEY, 'submissions': 0, 'cancel_attempts': 0, 'started_utc': datetime.now(timezone.utc).isoformat(), 'privacy': 'Only exact approved public image; no service authentication/token, models or private data uploaded', 'remaining_anonymous_quota': 'UNKNOWN; one max120s request, no retry or paid fallback', 'events': []}
    token = get_token(); assert token
    hub = requests.Session(); hub.trust_env = False; hub.headers['Authorization'] = 'Bearer ' + token
    public = requests.Session(); public.trust_env = False
    client = job = None
    def checkpoint():
        if time.monotonic() >= deadline: raise TimeoutError('300s total deadline')
    def bounded_get(session, url, name, cap):
        checkpoint()
        with session.get(url, timeout=(5, min(20, max(1, deadline-time.monotonic()))), allow_redirects=False, stream=True) as response:
            response.raise_for_status(); assert response.status_code == 200
            parts=[]; count=0
            for chunk in response.iter_content(65536):
                checkpoint(); count += len(chunk)
                if count > cap: raise ValueError('response byte cap')
                parts.append(chunk)
        raw=b''.join(parts); (D/name).write_bytes(raw); return raw
    def guard(label):
        base='https://huggingface.co/api/spaces/einarolafsson/spacr-mask-generator'
        info=json.loads(bounded_get(hub,base,label+'-info.json',2*1024**2))
        runtime=json.loads(bounded_get(hub,base+'/runtime',label+'-runtime.json',100000))
        variables=json.loads(bounded_get(hub,base+'/variables',label+'-variables.json',100000))
        assert info['sha'] == SOURCE and runtime['sha'] == SOURCE and runtime['stage'] == 'RUNNING'
        assert runtime['hardware'] == {'current':'zero-a10g','requested':'zero-a10g'}
        assert {k:v['value'] for k,v in variables.items()} == FLAGS
        domains=[r['domain'] for r in runtime['domains'] if r['stage']=='READY']; assert len(domains)==1
        host=domains[0]; assert host.endswith('.hf.space') and '/' not in host and ':' not in host
        return 'https://'+host
    def cancel_once():
        if job is None or state['cancel_attempts']: return
        state['cancel_attempts']=1; state['remote_execution_state_after_timeout']='UNKNOWN; cancellation cannot prove active GPU stopped'
        def attempt():
            try: state['cancel_return']=job.cancel()
            except Exception as error: state['cancel_error_type']=type(error).__name__
        thread=threading.Thread(target=attempt,daemon=True);thread.start();thread.join(timeout=5)
        state['cancel_call_terminal']=not thread.is_alive()
    def alarm(_signum,_frame): raise TimeoutError('300s total deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
    code=1
    try:
        domain=guard('before');state['provider_derived_domain']=domain
        config=json.loads(bounded_get(public,domain+'/config','config.json',4*1024**2))
        schema=json.loads(bounded_get(public,domain+'/gradio_api/info','api-info.json',4*1024**2))
        approved=json.loads((S/'live-schema-f457/api-info.json').read_text())
        assert schema['named_endpoints'] == approved['named_endpoints'] and config['version']=='6.17.3'
        rows=json.loads((S/'hosted-parity-plan-f457/acceptance-matrix17.json').read_text())
        row=next(r for r in rows if r['model']['key']==KEY)
        names=row['parameter_order'];assert len(names)==24
        values=dict(row['request_template_named']);values['path']=handle_file(INPUT)
        assert values['chosen_channels']==[0] and values['chosen_magnification']==20 and values['chosen_diameter'] is None and values['accepted'] is True
        state['request_named']={k:v for k,v in values.items() if k!='path'}
        state['request_path']='Exact approved public local input bound above; normal SDK upload'
        client=Client(domain,token=False,max_workers=1,verbose=False,download_files=False,analytics_enabled=False,httpx_kwargs={'timeout':httpx.Timeout(connect=10,read=145,write=30,pool=5),'follow_redirects':False,'trust_env':False})
        assert not any('authorization' in k.lower() for k in client.headers)
        assert [p['parameter_name'] for p in client._info['named_endpoints'][API]['parameters']] == names
        assert client.config['dependencies'] == config['dependencies']
        assert client.src.startswith(domain+'/')
        assert guard('presubmit') == domain
        state['submissions']=1;state['submit_started_utc']=datetime.now(timezone.utc).isoformat()
        job=client.submit(*(values[n] for n in names),api_name=API)
        result=job.result(timeout=max(1,deadline-time.monotonic()-20))
        assert isinstance(result,(tuple,list)) and len(result)==2
        (D/'sdk-result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        state['submit_finished_utc']=datetime.now(timezone.utc).isoformat()
        def output_file(value,name,cap):
            assert isinstance(value,dict) and isinstance(value.get('url'),str) and not value.get('is_stream')
            url=urljoin(domain+'/',value['url']);parsed=urlparse(url);assert parsed.scheme=='https' and parsed.netloc==urlparse(domain).netloc and parsed.path.startswith('/gradio_api/file=') and not parsed.username and not parsed.password
            return bounded_get(public,url,name,cap)
        output_file(result[0],'result.zip',128*1024**2)
        output_file(result[1],'preview.png',32*1024**2)
        allowed={'input/'+INPUT.name,'image.tif','mask.tif','flows.npy','cell_probability.npy','objects.csv','README.txt','manifest.json'}
        with zipfile.ZipFile(D/'result.zip') as archive:
            entries=archive.infolist();paths=[entry.filename for entry in entries]
            assert len(paths)==len(set(paths)) and set(paths)<=allowed and len(paths)<=8
            assert {'input/'+INPUT.name,'image.tif','mask.tif','objects.csv','README.txt','manifest.json'}<=set(paths)
            assert sum(entry.file_size for entry in entries)<=512*1024**2
            for entry in entries:
                path=PurePosixPath(entry.filename)
                assert not path.is_absolute() and '..' not in path.parts and str(path)==entry.filename and not stat.S_ISLNK(entry.external_attr>>16) and entry.file_size<=256*1024**2
            manifest=json.loads(archive.read('manifest.json'));assert manifest['model']=={'id':KEY,'version':'1','weights_sha256':WEIGHTS_SHA}
            assert set(manifest['files'])==set(paths)-{'manifest.json'}
            for name,identity in manifest['files'].items():
                with archive.open(name) as handle:
                    digest=hashlib.sha256();count=0
                    for chunk in iter(lambda:handle.read(65536),b''):checkpoint();digest.update(chunk);count+=len(chunk)
                assert digest.hexdigest()==identity['sha256'] and count==identity['bytes']
            assert hashlib.sha256(archive.read('input/'+INPUT.name)).hexdigest()==INPUT_SHA
            (D/'result-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            import numpy as np,tifffile
            arrays={}
            for name in ['image.tif','mask.tif','flows.npy','cell_probability.npy']:
                if name not in paths:continue
                raw=archive.read(name)
                if name.endswith('.tif'):
                    with tifffile.TiffFile(io.BytesIO(raw))as handle:
                        series=handle.series[0];assert len(series.shape)<=3 and math.prod(series.shape)*np.dtype(series.dtype).itemsize<=128*1024**2;array=series.asarray()
                else:
                    handle=io.BytesIO(raw);version=np.lib.format.read_magic(handle);assert version in ((1,0),(2,0))
                    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(handle)if version==(1,0)else np.lib.format.read_array_header_2_0(handle));assert dtype.kind in 'buif'and shape==((2,1998,1998)if name=='flows.npy'else(1998,1998)) and math.prod(shape)*dtype.itemsize<=128*1024**2
                    array=np.load(io.BytesIO(raw),allow_pickle=False)
                assert array.dtype.kind in 'buif'and np.isfinite(array).all()
                expected_shape=(2,1998,1998)if name=='flows.npy'else(1998,1998)
                assert array.shape==expected_shape
                arrays[name]={'shape':list(array.shape),'dtype':str(array.dtype),'minimum':float(array.min()),'maximum':float(array.max()),'file_sha256':hashlib.sha256(raw).hexdigest()}
                if name=='image.tif':
                    with tifffile.TiffFile(INPUT)as source_handle:assert np.array_equal(array,source_handle.series[0].asarray())
                if name=='mask.tif':
                    assert array.dtype.kind in 'ui';state['actual_label_count']=int(np.count_nonzero(np.unique(array)))
                del array,raw
            state['arrays']=arrays;state['reported_object_count']=manifest['object_count']
            assert state['actual_label_count']==manifest['object_count']
            selection=manifest['settings']['input_selection'];assert selection['series']==0 and selection['channels']==[0]and selection['indices']=={}and selection['source_axes']=='YX'
            assert manifest['settings']['magnification']==20 and manifest['shape_yx']==[1998,1998]
            state['limits']=['Technical smoke only; no independent desktop reference or biological acceptance.','20x is explicit software default; actual acquisition magnification unverified.','No claim active cancellation/physical GPU placement/cost/cleanup/retention or wholeN698 parity.']
        assert guard('after')==domain
        assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==INPUT_SHA
        state['status']='TECHNICAL_SMOKE_PASS';code=0
    except (TimeoutError,requests.Timeout,httpx.TimeoutException) as error:
        state['status']='TIMEOUT';state['error']=str(error);cancel_once()
    except Exception as error:
        state['status']='FAILED_OR_REFUSED';state['error_type']=type(error).__name__;state['error']=str(error);state['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0)
        if client is not None:
            teardown = {}; finished = threading.Event()
            def close_client():
                try:client.close();client.executor.shutdown(wait=False,cancel_futures=True)
                except Exception as error:teardown['error_type']=type(error).__name__
                finally:finished.set()
            thread=threading.Thread(target=close_client,daemon=True);thread.start();thread.join(timeout=5)
            state['local_teardown']={**teardown,'terminal':finished.is_set(),'remaining_state':'terminal'if finished.is_set()else'UNKNOWN; local teardown exceeded5s'}
        state['finished_utc']=datetime.now(timezone.utc).isoformat();state['wall_seconds']=time.monotonic()-started
        state['files']={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in D.iterdir()if p.is_file()}
        payload=(json.dumps(state,indent=2,default=str)+'\n').encode()
        try:
            with(D/'receipt.json').open('xb')as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
        except Exception as error:
            code=1;state['receipt_integrity_error_type']=type(error).__name__
            print(json.dumps({'receipt_integrity_failure':state},default=str),flush=True)
        print(json.dumps({'status':state['status'],'submissions':state['submissions'],'cancel_attempts':state['cancel_attempts'],'receipt':str(D/'receipt.json'),'local_teardown':state.get('local_teardown')}),flush=True)
        os._exit(code)

if __name__=='__main__':main()
