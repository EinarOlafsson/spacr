import json,hashlib
from pathlib import Path
import pandas as pd
from spacr.embeddings import _mil_bags
records=[]
for plate in ('plate_1','plate_2'):
 for well in range(4):
  for cell in range(2):
   records.append(dict(plateID=plate,rowID='r1',columnID='c'+str(well+1),well='A'+str(well+1),well_label=int(well<2),emb_0=float(well+cell)))
frame=pd.DataFrame(records);original=frame.copy(deep=True)
bags,labels,wells,rows,kept=_mil_bags(frame)
pd.testing.assert_frame_equal(frame,original)
r={'source':str(Path(__import__('spacr.embeddings',fromlist=['x']).__file__).resolve()),'input_cells':len(frame),'native_acquisition_wells':len(frame.groupby(['plateID','rowID','columnID'])),'returned_bags':len(bags),'returned_identifiers':wells,'bag_cells':[len(x) for x in bags],'cross_plate_bags':[{'well':well,'plates':sorted(kept.iloc[index].plateID.unique().tolist())} for well,index in zip(wells,rows) if kept.iloc[index].plateID.nunique()>1],'input_unchanged':True,'scope':'Read-only actual _mil_bags, no training/models/GPU/data files'}
Path('/mnt/wd4tb/scratch/f562-direct-table-identity-audit-20261015/receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,sort_keys=True))
