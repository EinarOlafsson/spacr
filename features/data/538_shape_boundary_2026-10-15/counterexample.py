from pathlib import Path
import hashlib,json,sys
import numpy as np
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f538-shape-boundary-20261015');sys.path.insert(0,str(root))
from spacr.psf_pipeline import _prepare_unmixing
scratch=Path('/mnt/wd4tb/scratch/f538-shape-boundary-20261015');scratch.mkdir(exist_ok=True)
plane=np.arange(64,dtype=np.float64).reshape(8,8)
proper=[np.stack([20+3*plane,20+.6*plane],axis=-1),np.stack([20+.4*plane,20+4*plane],axis=-1)]
spatial=[]
for own,other in [(3,.6),(.4,4)]:
 value=np.full((8,8),20.0);value[:,0]=20+own*np.arange(8);value[:,1]=20+other*np.arange(8);spatial.append(value)
cases=[('invalid_2d_explicit',spatial,(0,1),'0:A01; 1:B01'),('invalid_2d_default',spatial,None,'0:A01; 1:B01'),('valid_yxc',proper,None,'0:A01; 1:B01'),('valid_zyxc',[np.stack([v,v]) for v in proper],None,'0:A01; 1:B01'),('valid_extra_leading_axes',[np.stack([np.stack([v,v]),np.stack([v,v])]) for v in proper],None,'0:A01; 1:B01'),('valid_single_channel_default',[proper[0][...,:1]],None,'0:A01')]
observations=[];plans={}
for case,fields,channels,controls in cases:
 folder=scratch/case;folder.mkdir(exist_ok=True)
 for index,field in enumerate(fields):np.save(folder/f'plate1_{"AB"[index]}01_1.npy',field)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npy')}
 plan=_prepare_unmixing({'unmix':True,'unmix_controls':controls,'unmix_background_percentile':5.0},folder,channels=channels);plans[case]=plan
 assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npy')}
 original=fields[0].copy();out=plan.apply(fields[0]);np.testing.assert_array_equal(fields[0],original)
 observations.append({'case':case,'accepted':True,'control_shape':list(fields[0].shape),'plan_channels':list(plan.channels),'applied_shape':list(out.shape),'input_hashes':hashes,'inputs_unchanged':True})
image=spatial[0].copy();before=image.copy();out=plans['valid_yxc'].apply(image);np.testing.assert_array_equal(image,before)
assert np.array_equal(out[:,2:],image[:,2:]) and not np.allclose(out[:,:2],image[:,:2])
single=proper[0][...,:1];np.testing.assert_array_equal(plans['valid_single_channel_default'].apply(single),single.astype(np.float32))
receipt={'source_pin':'fd908ba35','scope':'Private declared synthetic channel-last controls plus deliberately invalid2Dspatial arrays; actual preparation/estimation/plan application, no models/GPU/realbiology.','observations':observations,'valid_two_channel_plan_accepts_2d_spatial_image':{'accepted':True,'shape':list(image.shape),'only_first_two_spatial_columns_changed':True,'max_absolute_change':float(np.max(np.abs(out-image))),'original_image_unchanged':True},'single_explicit_channel_axis_default_preserved_exactly':True,'limits':['Valid3D/4D and higher leading axes all accepted; no proposed upper dimension bound.','Single-channel YXC(C=1) remains intentional valid input;2Dhas no explicit spectral axis.','No matrix/rank/conditioning/PSF/real-double-stain or biological efficacy acceptance.']};(scratch/'counterexample.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
