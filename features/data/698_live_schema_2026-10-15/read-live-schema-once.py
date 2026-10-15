from datetime import datetime,timezone
from pathlib import Path
import requests,json,hashlib,ast,traceback
from urllib.parse import urlparse
from huggingface_hub import get_token
S=Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015');D=S/'live-schema-f457';D.mkdir(exist_ok=False)
R=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
SOURCE='f457356bbffa1c3166f01ea3e7a40c6ab44fca35';BASE='https://huggingface.co/api/spaces/einarolafsson/spacr-mask-generator'
session=requests.Session();session.headers['Authorization']='Bearer '+get_token();public=requests.Session()
result={'space_source':SOURCE,'started_utc':datetime.now(timezone.utc).isoformat(),'operations':[],'inference_calls':0,'service_mutations':0}
def get_json(client,url,name):
    response=client.get(url,timeout=(5,15),allow_redirects=False,stream=True)
    result['operations'].append({'method':'GET','url':url,'status':response.status_code})
    response.raise_for_status();parts=[];size=0
    for chunk in response.iter_content(65536):
        size+=len(chunk)
        if size>4*1024*1024:raise ValueError('response exceeds4MiB')
        parts.append(chunk)
    raw=b''.join(parts);(D/name).write_bytes(raw);return json.loads(raw)
def guard(prefix):
    info=get_json(session,BASE,prefix+'-info.json');runtime=get_json(session,BASE+'/runtime',prefix+'-runtime.json');variables=get_json(session,BASE+'/variables',prefix+'-variables.json')
    assert info['sha']==SOURCE and runtime['sha']==SOURCE
    assert runtime['stage']=='RUNNING' and runtime['hardware']=={'current':'zero-a10g','requested':'zero-a10g'}
    assert {k:v['value'] for k,v in variables.items()}=={'SPACR_WEB_ZERO_GPU':'1','SPACR_WEB_PROVISION_MODELS':'1'}
    ready=[r['domain'] for r in runtime['domains'] if r['stage']=='READY'];assert len(ready)==1
    host=ready[0];assert host.endswith('.hf.space') and '/'not in host and ':'not in host
    return 'https://'+host
try:
    domain=guard('before');result['provider_derived_domain']=domain
    config=get_json(public,domain+'/config','config.json')
    info=get_json(public,domain+'/gradio_api/info','api-info.json')
    assert guard('after')==domain
    app=(R/'tools/web_mask_generator/app.py').read_bytes();manifest=(R/'tools/web_mask_generator/space/models.json').read_bytes()
    tree=ast.parse(app);infer=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)and n.name=='infer')
    args=[a.arg for a in infer.args.args];assert len(args)==24
    families=sorted({m['family'] for m in json.loads(manifest)['models']});assert len(families)==8
    endpoints={('/segment_plaque' if f=='toxoplasma_plaque' else '/detect_wells' if f=='toxoplasma_well_detector' else '/segment_'+f):f for f in families}
    named=info['named_endpoints'];assert set(endpoints)<=set(named)
    dependencies={d['api_name']:d for d in config['dependencies'] if d.get('api_name')}
    components={c['id']:c for c in config['components']}
    rows=[]
    for endpoint,family in endpoints.items():
        schema=named[endpoint];dep=dependencies[endpoint[1:]];params=schema['parameters']
        assert len(params)==24 and len(dep['inputs'])==24
        names=[p['parameter_name'] for p in params];assert names==args,(endpoint,names,args)
        dropdown=components[dep['inputs'][1]]['props'];keys={pair[1]if isinstance(pair,list)else pair for pair in dropdown['choices']}
        expected={m['key'] for m in json.loads(manifest)['models'] if m['family']==family};assert keys==expected
        assert components[dep['inputs'][2]]['props']['value'] is False
        assert components[dep['inputs'][0]]['props']['interactive'] is False
        assert len(schema['returns'])==2
        rows.append({'family':family,'endpoint':endpoint,'input_count':len(params),'parameter_names':names,'model_keys':sorted(keys),'default_model_key':dropdown['value'],'output_count':len(schema['returns']),'consent_default_false':True,'upload_initially_disabled':True})
    result.update({'status':'PASS','families':rows,'source_app_sha256':hashlib.sha256(app).hexdigest(),'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),'gradio_config_version':config.get('version'),'queue_enabled':config.get('enable_queue'),'all_public_api_names':sorted(named),'limits':['Read-only API/config shape acceptance; not inference,weights,placement,parity,cancellation,scientific acceptance or website activation.']})
except Exception as error:
    result.update({'status':'FAILED_OR_REFUSED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
finally:
    result['finished_utc']=datetime.now(timezone.utc).isoformat();result['files']={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in D.iterdir()if p.is_file()};(D/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='PASS':raise SystemExit(1)
