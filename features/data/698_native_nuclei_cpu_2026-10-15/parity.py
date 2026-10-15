import gc,hashlib,importlib.metadata,io,json,os,socket,subprocess,sys,time,zipfile
from pathlib import Path
repo=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015');sys.path.insert(0,str(repo))
work=Path('/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015');trust=json.loads((work/'trust.json').read_text());checkpoint=Path(trust['file'])
original_affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{original_affinity[0]})
network_attempts=[];original_connect=socket.socket.connect
def offline_connect(sock,address):
 if sock.family in (socket.AF_INET,socket.AF_INET6):
  network_attempts.append(repr(address));raise RuntimeError('Network prohibited during cache-only native acceptance')
 return original_connect(sock,address)
socket.socket.connect=offline_connect
import numpy as np
import tifffile
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
from tools.web_mask_generator import masks,registry
from spacr.io import _normalize_img_batch
from spacr.object import generate_cellpose_masks_sam
source_paths=['spacr/object.py','spacr/settings.py','spacr/object_settings_table.py','spacr/io.py','spacr/utils.py','spacr/model_zoo.py','spacr/spacr_cellpose.py','tools/web_mask_generator/masks.py','tools/web_mask_generator/registry.py','tools/web_mask_generator/results.py','tools/web_mask_generator/inputs.py']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(1048576),b''):h.update(block)
 return h.hexdigest()
before={name:sha(repo/name) for name in source_paths};assert sha(checkpoint)==trust['sha256'];checkpoint_stat=checkpoint.stat()
key=trust['key'];family='nuclei_from_cellmask';available=next(x for x in registry.model_families() if x['family']==family);assert available['default']==key
resolved=registry.resolve_trusted_model(family,{key:checkpoint});assert resolved['metadata']['key']==key and resolved['metadata']['sha256']==trust['sha256']
y,x=np.mgrid[:128,:160];image=(100+x*2+y).astype(np.uint16)
for cy,cx,ry,rx in [(38,42,25,30),(78,110,30,35),(104,40,17,22)]:
 inside=((y-cy)/ry)**2+((x-cx)/rx)**2<=1
 hole=((y-cy)/(ry*0.35))**2+((x-cx)/(rx*0.35))**2<=1
 image[inside]+=1800;image[hole]-=900
input_file=work/'declared-cellmask-like-synthetic.tif';tifffile.imwrite(input_file,image,metadata={'axes':'YX'});input_hash=sha(input_file)
settings,object_settings,point,diameter=masks._settings('nucleus',{'diameter':32.0},checkpoint)
settings.update({'plot':False,'save':True,'batch_size':1,'verbose':False})
project=work/'desktop';batches=project/'batches';batches.mkdir(parents=True);raw_dir=project/'stack';raw_dir.mkdir()
filename='synthetic_r1_c1_f1.npy';np.save(raw_dir/filename,image[...,None]);raw_hash=sha(raw_dir/filename)
normalized=_normalize_img_batch(image[None,...,None].copy(),[0],np.float32,settings)
np.savez_compressed(batches/'one-field.npz',data=normalized,filenames=np.asarray([filename]));npz_hash=sha(batches/'one-field.npz')
print(json.dumps({'stage':'actual-desktop-start','model':key,'input_shape':list(image.shape),'diameter':diameter,'threads':torch.get_num_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0))}),flush=True)
start=time.monotonic();generate_cellpose_masks_sam(str(batches),dict(settings),'nucleus',batch_paths=[str(batches/'one-field.npz')],run_qc=False);desktop_seconds=time.monotonic()-start
mask_path=batches/'nucleus_mask_stack'/filename;desktop=np.load(mask_path,allow_pickle=False);assert desktop.shape==image.shape;desktop_hash=sha(mask_path)
gc.collect()
print(json.dumps({'stage':'actual-desktop-finished','seconds':desktop_seconds,'objects':int(np.count_nonzero(np.unique(desktop))),'threads':torch.get_num_threads()}),flush=True)
start=time.monotonic();result=masks.run_mask_download(work/'web-result.zip',family=family,input_file=input_file,checkpoint=checkpoint,expected_sha256=trust['sha256'],model_id=key,model_version=resolved['metadata']['version'],effective_settings={'diameter':32.0});adapter_seconds=time.monotonic()-start
with zipfile.ZipFile(result) as archive:
 manifest=json.loads(archive.read('manifest.json'));actual=tifffile.imread(io.BytesIO(archive.read('mask.tif')));assert np.array_equal(actual,desktop)
 assert manifest['model']=={'id':key,'version':resolved['metadata']['version'],'weights_sha256':trust['sha256']}
 assert archive.read('input/'+input_file.name)==input_file.read_bytes()
 assert np.array_equal(tifffile.imread(io.BytesIO(archive.read('image.tif'))),image)
 for name,fingerprint in manifest['files'].items():
  data=archive.read(name);assert hashlib.sha256(data).hexdigest()==fingerprint['sha256'] and len(data)==fingerprint['bytes']
 optional={}
 for name in ['flows.npy','cell_probability.npy']:
  if name in archive.namelist():
   array=np.load(io.BytesIO(archive.read(name)),allow_pickle=False);assert np.isfinite(array).all();optional[name]={'shape':list(array.shape),'dtype':str(array.dtype)}
 assert set(optional)=={'flows.npy','cell_probability.npy'}
assert sha(checkpoint)==trust['sha256'];after_stat=checkpoint.stat();assert (after_stat.st_size,after_stat.st_mtime_ns)==(checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns)
assert sha(input_file)==input_hash and sha(raw_dir/filename)==raw_hash and sha(batches/'one-field.npz')==npz_hash and sha(mask_path)==desktop_hash
assert before=={name:sha(repo/name) for name in source_paths};assert not torch.cuda.is_initialized() and not network_attempts
objects=int(np.count_nonzero(np.unique(desktop)))
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'source_sha256':before,'checkpoint':trust,'input':{'declared_synthetic':True,'shape':list(image.shape),'dtype':str(image.dtype),'sha256':input_hash,'biological_groundtruth':False},'settings':{'diameter':32.0,'role':'nucleus','resolved_diameter':diameter,'flow_threshold':point['flow_threshold'],'cellprob_threshold':point['cellprob_threshold']},'actual_desktop':'spacr.object.generate_cellpose_masks_sam','actual_adapter':'tools.web_mask_generator.masks.run_mask_download','desktop_seconds':desktop_seconds,'adapter_seconds':adapter_seconds,'desktop_objects':objects,'pixel_exact_postfiltered_labels':True,'optional_outputs_actual_finite':optional,'runtime':{'cellpose':importlib.metadata.version('cellpose'),'torch':torch.__version__,'initial_threads_requested':1,'observed_final_threads':torch.get_num_threads(),'observed_final_interop_threads':torch.get_num_interop_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'cuda_initialized':False,'mps_available':bool(torch.backends.mps.is_available()),'network_attempts':network_attempts},'original_checkpoint_inputs_desktop_labels_and_source_unchanged':True,'limits':['One declared synthetic field; no biological quality or accuracy claim, including if masks empty','Exact current default nuclei_from_cellmask_v1 only; no other checkpoint/family acceptance','CPU/native desktop-adapter parity only, not hosted/browser/GPU acceptance','No model construction/prediction mocks, training or downloads']}
(work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'stage':'PASS','desktop_objects':objects,'desktop_seconds':desktop_seconds,'adapter_seconds':adapter_seconds,'threads':torch.get_num_threads()}),flush=True)
