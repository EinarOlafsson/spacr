"""One authorized source-archive-only atomic Space upload and immutable readback."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import requests
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import CommitOperationAdd, HfApi, configure_http_backend, get_token
from huggingface_hub.hf_api import RepoFile
S = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
TARGET = 'einarolafsson/spacr-mask-generator'
PARENT = '3371bce879bb15a6597d6f268071ebfb2df1e23c'
PATCH = '84615c61607759607c01429fe7d349cfb8242e9a'
class BoundedSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', (300, 90))
        return super().request(*args, **kwargs)
configure_http_backend(backend_factory=BoundedSession)
def note(step, **details):
    row = {'time_utc': datetime.now(timezone.utc).isoformat(), 'step': step, **details}
    with (S / 'source-archive-stage-a-events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n'); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)
def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024*1024): result.update(chunk)
    return result.hexdigest()
def expected_sources():
    manifest = json.loads((S / 'source-staging-35aeaa9ddc-no-weights.json').read_text())
    expected = {row['destination_path']: row.copy() for row in manifest['files']}
    for src, dest in [('tools/web_mask_generator/space/README.md', 'README.md'), ('tools/web_mask_generator/space/requirements.txt', 'requirements.txt')]:
        raw = subprocess.check_output(['git','-C',str(ROOT),'show',PATCH+':'+src])
        for path in (src, dest): expected[path].update({'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(), 'git_blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()})
    assert len(expected) == 1182
    return expected
def tree(api, rev):
    return {x.path:x for x in api.list_repo_tree(TARGET,repo_type='space',revision=rev,recursive=True) if isinstance(x,RepoFile)}
def check_sources(actual, expected):
    for path,row in expected.items():
        current=actual[path]; assert current.size==row['bytes']
        if current.lfs: assert current.lfs.size==row['bytes'] and current.lfs.sha256==row['sha256']
        else: assert current.blob_id==row['git_blob']
def public_bytes(path, revision, limit):
    response=requests.get('https://huggingface.co/spaces/'+TARGET+'/resolve/'+revision+'/'+path,timeout=(10,60));response.raise_for_status();assert len(response.content)<=limit;return response.content
state={'scope':'STAGE_A_ONLY','target':TARGET,'parent_commit':PARENT,'mutation_attempts':[],'status':'STARTED','authorization':'Parent/root explicitly authorized one artifact-only atomic add; stageB not authorized.'}
try:
    fd=os.open(S/'source-archive-stage-a-attempt.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    plan_path=S/'SOURCE-ARCHIVE-TWO-STAGE-PLAN.json';assert sha(plan_path)=='0cead1b0b4329b3ce0b3bf8d839b1fa94a4b4854f78e1165260c1b182c73bbdd'
    plan=json.loads(plan_path.read_text());peer=json.loads((S/'root-source-archive-stage-A-peer.json').read_text());assert peer['archive_sha256']==plan['artifact']['sha256']
    artifact=plan['artifact'];local=Path(artifact['local_path']);remote_path=artifact['remote_path']
    assert artifact['bytes']==local.stat().st_size==45028475 and sha(local)==artifact['sha256']=='a2068b97a8973a64fa13febbd81b87e933befb7684779be50d356fe969ce457f'
    state['artifact']=artifact
    token=get_token();assert token
    session=BoundedSession();session.headers['Authorization']='Bearer '+token
    response=session.get('https://huggingface.co/api/whoami-v2',timeout=(10,30),allow_redirects=False);response.raise_for_status();who=response.json()
    assert who.get('name')=='einarolafsson' and who.get('type')=='user' and who.get('isPro')is False and who.get('canPay')is False
    api=HfApi(token=token);info=api.space_info(TARGET);assert info.id==TARGET and info.sha==PARENT and info.private is False and info.sdk=='gradio'
    runtime=api.get_space_runtime(TARGET);assert runtime.requested_hardware=='zero-a10g' and runtime.hardware in(None,'zero-a10g')
    flags=api.get_space_variables(TARGET);assert {k:v.value for k,v in flags.items()}=={'SPACR_WEB_ZERO_GPU':'1','SPACR_WEB_PROVISION_MODELS':'1'}
    expected=expected_sources();before=tree(api,PARENT);assert set(before)==set(expected)|{'.gitattributes'};check_sources(before,expected);assert remote_path not in before
    attributes_before=public_bytes('.gitattributes',PARENT,65536);assert hashlib.sha1(b'blob '+str(len(attributes_before)).encode()+b'\0'+attributes_before).hexdigest()==before['.gitattributes'].blob_id
    state['fresh_guards']={'source_parent':PARENT,'all1182_source_identities':True,'requested':runtime.requested_hardware,'current':runtime.hardware,'flags_exact':True,'artifact_sha256':artifact['sha256']}
    note('fresh_stage_a_guards_verified',**state['fresh_guards'])
    state['mutation_attempts'].append({'operation':'add_exact_source_archive','path':remote_path,'parent_commit':PARENT})
    note('artifact_only_commit_requested',path=remote_path,bytes=artifact['bytes'])
    commit=api.create_commit(repo_id=TARGET,repo_type='space',parent_commit=PARENT,num_threads=1,commit_message='Stage reviewed normal spaCR source distribution for dependency build',commit_description='Source snapshot66ac /code35aeaa. Ordinary source archive SHA256 '+artifact['sha256']+'; no checkpoints, history or user data. StageA only; configuration unchanged.',operations=[CommitOperationAdd(path_in_repo=remote_path,path_or_fileobj=local)])
    state['space_commit']=commit.oid;note('artifact_commit_accepted',commit=commit.oid)
    after=tree(api,commit.oid);assert set(after)==set(expected)|{'.gitattributes',remote_path};check_sources(after,expected)
    uploaded=after[remote_path];assert uploaded.lfs and uploaded.lfs.sha256==artifact['sha256'] and uploaded.lfs.size==artifact['bytes'] and uploaded.size==artifact['bytes']
    attributes_after=public_bytes('.gitattributes',commit.oid,65536)
    oldlines=attributes_before.decode().splitlines();newlines=attributes_after.decode().splitlines();added=set(newlines)-set(oldlines);removed=set(oldlines)-set(newlines)
    assert not removed
    allowed={remote_path+' filter=lfs diff=lfs merge=lfs -text','*.tar.gz filter=lfs diff=lfs merge=lfs -text'}
    assert added<=allowed
    attribute_diff='\n'.join(difflib.unified_diff(oldlines,newlines,fromfile='parent/.gitattributes',tofile='stageA/.gitattributes',lineterm=''))
    state['provider_attribute_diff']=attribute_diff
    digest=hashlib.sha256();total=0
    with requests.get('https://huggingface.co/spaces/'+TARGET+'/resolve/'+commit.oid+'/'+remote_path,stream=True,timeout=(10,90))as response:
        response.raise_for_status()
        for chunk in response.iter_content(1024*1024):
            total+=len(chunk);assert total<=artifact['bytes'];digest.update(chunk)
    assert total==artifact['bytes'] and digest.hexdigest()==artifact['sha256']
    final=api.space_info(TARGET);assert final.sha==commit.oid
    runtime=api.get_space_runtime(TARGET);assert runtime.requested_hardware=='zero-a10g' and runtime.hardware in(None,'zero-a10g')
    flags=api.get_space_variables(TARGET);assert {k:v.value for k,v in flags.items()}=={'SPACR_WEB_ZERO_GPU':'1','SPACR_WEB_PROVISION_MODELS':'1'}
    readback={'space_commit':commit.oid,'unchanged_source_files':[{ 'path':p,'bytes':after[p].size,'blob':after[p].blob_id,'lfs_sha256':after[p].lfs.sha256 if after[p].lfs else None}for p in sorted(expected)],'archive':{'path':remote_path,'bytes':total,'lfs_sha256':uploaded.lfs.sha256,'streamed_public_sha256':digest.hexdigest()},'provider_attribute_diff':attribute_diff}
    (S/'source-archive-stage-a-readback.json').write_text(json.dumps(readback,indent=2)+'\n')
    state.update({'status':'STAGE_A_ARCHIVE_COMMITTED_AND_VERIFIED','unchanged_source_files':1182,'archive_streamed_public_bytes':total,'archive_streamed_sha256':digest.hexdigest(),'runtime':{'stage':runtime.stage,'current':runtime.hardware,'requested':runtime.requested_hardware},'stage_B_executed':False,'other_mutations':False})
    note('stage_a_verified',commit=commit.oid,unchanged_source_files=1182,public_archive_bytes=total,runtime=state['runtime'])
except BaseException as error:
    state['status']='STOPPED_NO_RETRY';state['error_type']=type(error).__name__;response=getattr(error,'response',None)
    if response is not None:state['error_http_status']=response.status_code
    note('stopped',error_type=type(error).__name__,mutation_attempts=len(state['mutation_attempts']));raise
finally:
    state['finished_at_utc']=datetime.now(timezone.utc).isoformat();(S/'source-archive-stage-a-result.json').write_text(json.dumps(state,indent=2)+'\n')
