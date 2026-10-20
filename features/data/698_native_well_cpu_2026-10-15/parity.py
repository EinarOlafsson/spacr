import hashlib,importlib.metadata,json,os,sys,time,zipfile,subprocess
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-scientific-backends-integration-20261014')
sys.path.insert(0,str(root))
import numpy as np
import torch
from PIL import Image
import spacr
from spacr import model_zoo,plaque
from tools.web_mask_generator import detector
assert Path(spacr.__file__).resolve().parent==root/'spacr'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
assert not torch.backends.mps.is_available()
torch.set_num_threads(1)
work=Path('/mnt/wd4tb/scratch/n698-native-well-cpu-20261015')
checkpoint=Path.home()/'.spacr/models/yolo_welldetect_v4.pt'
key='toxoplasma_well_detector_v2'
record=next(row for row in model_zoo.BUNDLED_REMOTE_MODELS if row['key']==key)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert record['name']==checkpoint.name and sha(checkpoint)==record['sha256']
sourcepaths=['spacr/plaque.py','spacr/model_zoo.py','tools/web_mask_generator/detector.py','tools/web_mask_generator/inputs.py','tools/web_mask_generator/results.py']
before={p:sha(root/p) for p in sourcepaths}
y,x=np.indices((256,320));gray=np.full(y.shape,210,dtype=np.uint8)
for cy,cx in [(80,75),(80,230),(190,155)]:
 r=np.sqrt((y-cy)**2+(x-cx)**2)
 gray[r<46]=55;gray[(r>=43)&(r<=47)]=15;gray[(r>=47)&(r<=50)]=245
rgb=np.stack([gray,np.clip(gray.astype(int)+12,0,255).astype(np.uint8),np.clip(gray.astype(int)-9,0,255).astype(np.uint8)],axis=-1)
settings={'confidence':0.15,'imgsz':320,'min_axis_ratio':0.7}
cases=[]
for name,image in [('gray',gray),('rgb',rgb)]:
 source=work/(name+'.png');Image.fromarray(image).save(source);inputsha=sha(source)
 started=time.perf_counter();desktop=plaque.detect_wells(image,str(checkpoint),**settings);desktopseconds=time.perf_counter()-started
 target=work/(name+'.zip');started=time.perf_counter()
 detector.run_well_download(target,input_file=source,checkpoint=checkpoint,expected_sha256=record['sha256'],model_id=key,model_version='v2',effective_settings=settings)
 adapterseconds=time.perf_counter()-started
 with zipfile.ZipFile(target) as z:
  actual=json.loads(z.read('boxes.json'));manifest=json.loads(z.read('manifest.json'))
  assert len(actual)==len(desktop)==manifest['detection_count']
  for row,well in zip(actual,desktop):
   assert all(row[k]==v for k,v in well.as_dict().items()),(name,row,well.as_dict())
  assert manifest['model']=={'id':key,'version':'v2','weights_sha256':record['sha256']}
  assert manifest['settings']['pipeline']=='spacr.plaque.detect_wells'
  assert set(manifest['unavailable'])=={'masks','flows','cell_probability'}
  assert not any(Path(p).name in {'masks.tif','flows.tif','cell_probability.tif'} for p in z.namelist())
  for path,fingerprint in manifest['files'].items():
   data=z.read(path);assert hashlib.sha256(data).hexdigest()==fingerprint['sha256'] and len(data)==fingerprint['bytes']
  assert z.read('input/'+source.name)==source.read_bytes()
 assert sha(source)==inputsha
 cases.append({'case':name,'input_shape':list(image.shape),'dtype':str(image.dtype),'input_sha256':inputsha,'actual_desktop_detections':len(desktop),'boxes_equal':True,'desktop_seconds':desktopseconds,'adapter_seconds':adapterseconds,'result_sha256':sha(target)})
 print(json.dumps(cases[-1]),flush=True)
assert sha(checkpoint)==record['sha256']
assert before=={p:sha(root/p) for p in sourcepaths}
assert not torch.cuda.is_initialized()
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':before,'checkpoint':{'key':key,'explicit_selection':True,'bytes':checkpoint.stat().st_size,'sha256':record['sha256']},'runtime':{'ultralytics':importlib.metadata.version('ultralytics'),'torch':torch.__version__,'cuda_available':False,'cuda_initialized':False,'mps_available':False,'threads':torch.get_num_threads()},'settings':settings,'cases':cases,'limits':['Two declared synthetic inputs; no biological accuracy acceptance','Explicit installed v2 identity only; no default substitution or other-family parity claim','CPU only; no native hosted/GPU/browser-service acceptance','No model/package downloads or protected-data reads']}
(work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
