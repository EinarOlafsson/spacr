from pathlib import Path
import hashlib,json,sys
import numpy as np
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f538-channel-boundary-20261015');sys.path.insert(0,str(root))
from spacr.psf_pipeline import _prepare_measure_unmixing
scratch=Path('/mnt/wd4tb/scratch/f538-channel-boundary-20261015');stack=scratch/'synthetic'/'stack';stack.mkdir(parents=True,exist_ok=True)
plane=np.arange(64,dtype=np.float64).reshape(8,8)
fields={'plate1_A01_1.npy':np.stack([20+3*plane,20+.25*plane],axis=-1),'plate1_B01_1.npy':np.stack([20+.1*plane,20+4*plane],axis=-1)}
for name,value in fields.items():np.save(stack/name,value)
hashes={name:hashlib.sha256((stack/name).read_bytes()).hexdigest() for name in fields}
results=[];plans={}
for case,channels in [('unique',[0,1]),('fraction',[0.5,1]),('bool',[False,True]),('duplicate',[0,0,1]),('integral_float',[0.0,1.0]),('integral_string',['0','1']),('negative',[-1,1])]:
 settings={'src':str(stack),'unmix':True,'unmix_controls':'0:A01; 1:B01','unmix_background_percentile':5.0,'channels':channels}
 try:
  plan=_prepare_measure_unmixing(settings);plans[case]=plan
  result={'case':case,'input_channels':channels,'accepted':True,'planned_channels':list(plan.channels),'provenance':plan.provenance()}
 except (ValueError,TypeError,IndexError,OverflowError) as error:
  result={'case':case,'input_channels':channels,'accepted':False,'error':type(error).__name__+': '+str(error)}
 results.append(result)
 assert hashes=={name:hashlib.sha256((stack/name).read_bytes()).hexdigest() for name in fields}
raw=fields['plate1_A01_1.npy'];original=raw.copy();unique=plans['unique'].apply(raw);duplicate=plans['duplicate'].apply(raw);np.testing.assert_array_equal(raw,original)
maxdiff=float(np.max(np.abs(unique[...,0]-duplicate[...,0])));assert maxdiff>0
receipt={'source_pin':'41b7e452052','scope':'Declared8x8 synthetic2channel single-stain controls, actual normal preparation/estimation/application, no models/GPU/biological data.','input_file_sha256':hashes,'all_input_files_and_apply_array_unchanged':True,'cases':results,'duplicate_selection_actual_effect':{'channel0_max_absolute_difference_from_unique_selection':maxdiff,'unique_channel0_std':float(unique[...,0].std()),'duplicate_channel0_std':float(duplicate[...,0].std())},'limits':'Shows existing invalid physical-channel selectors truncate/duplicate rather than refusing. Does not establish real double-stain/PSF/registration or biological accuracy.'}
(scratch/'counterexample.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
