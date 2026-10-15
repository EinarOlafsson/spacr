from pathlib import Path
from urllib.parse import urlparse,urljoin
import requests,hashlib,json,time,os,sys,traceback
from datetime import datetime,timezone
S=Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015');D=S/'approved-nuclei-image-acquisition';D.mkdir(mode=0o700,exist_ok=False)
R=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015');sys.path.insert(0,str(R))
record=next(r for r in json.loads((S/'hosted-parity-plan-f457/example-records-six-UNAPPROVED.json').read_text())if r['family']=='nuclei_from_cellmask')
assert record['revision']=='67c45ebd25000a36dc42120379e06285bdcb4b1d'and record['bytes']==4132206 and record['sha256']=='e84718d522edef0d3430a15834c6b5cfbcc5ac062e56cb5d29301722a1e98ffb'
record['credit']='Einar Olafsson / spaCR cross-channel nuclei-from-cellmask dataset, CC BY 4.0';record['source_url']='https://huggingface.co/datasets/'+record['repo_id']+'/blob/'+record['revision']+'/'+record['filename']
result={'scope':'Exactly one root operator-approved public DATASET input acquisition/header only; no image transformation, upload, inference, model or GPU action','approval':'Parent/root explicitly approved this exact immutable record after README CC-BY4.0+cell-mask input and both fields/by-well test rows(62objects) review; no allhistoricalmodel unseen claim','record':record,'started_utc':datetime.now(timezone.utc).isoformat(),'redirect_chain':[],'network_credentials':'NONE; requests Session trust_envFalse/no token'}
try:
 session=requests.Session();session.trust_env=False;url=record['source_url'].replace('/blob/','/resolve/');deadline=time.monotonic()+120
 for redirect in range(6):
  parsed=urlparse(url);host=parsed.hostname;assert parsed.scheme=='https'and(host=='huggingface.co'or host.endswith('.huggingface.co')or host.endswith('.hf.co'))
  response=session.get(url,stream=True,allow_redirects=False,timeout=(10,20));result['redirect_chain'].append({'host':host,'status':response.status_code})
  if response.status_code in (301,302,303,307,308):
   url=urljoin(url,response.headers['Location']);response.close();continue
  response.raise_for_status();break
 else:raise ValueError('redirect count exceeds6')
 pending=D/'input.tif.partial';sha=hashlib.sha256();total=0
 with pending.open('xb')as f:
  for chunk in response.iter_content(65536):
   total+=len(chunk)
   if total>record['bytes']or total>5*1024**2 or time.monotonic()>deadline:raise ValueError('approved byte/time limit exceeded')
   sha.update(chunk);f.write(chunk)
  f.flush();os.fsync(f.fileno())
 assert total==record['bytes']and sha.hexdigest()==record['sha256']
 image=D/Path(record['filename']).name;pending.rename(image)
 from tools.web_mask_generator.inputs import describe_upload
 header=describe_upload(image)
 assert hashlib.sha256(image.read_bytes()).hexdigest()==record['sha256']
 metadata=S/'parity-example-metadata/cross-channel-nuclei-from-cellmask'
 result.update({'status':'PASS','verified_bytes':total,'verified_sha256':sha.hexdigest(),'file':image.name,'description':header,'image_unchanged_after_header_read':True,'input_reader_source_sha256':hashlib.sha256((R/'tools/web_mask_generator/inputs.py').read_bytes()).hexdigest(),'source_metadata_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in metadata.iterdir()if p.is_file()},'semantics':'Dataset card declares cell-mask input for nuclei prediction. Channel index must follow actual header;20x remains an explicit software default, not a measured acquisition fact.','limits':['No selected-plane pixel read/transformation/inference yet.','Test split metadata agrees bywell/fields; not proof of unseen status for everyhistoricalmodel.','No preview/service/upload/GPU/model/website mutation or calls.']})
except Exception as error:result.update({'status':'FAILED_OR_REFUSED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
result['finished_utc']=datetime.now(timezone.utc).isoformat();(D/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='PASS':raise SystemExit(1)
