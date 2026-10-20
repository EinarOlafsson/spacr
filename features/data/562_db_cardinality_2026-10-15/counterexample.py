from pathlib import Path
import hashlib,json,sqlite3,sys
sys.path.insert(0,str(Path.cwd()))
import pandas as pd
from spacr.embeddings import _mil_frame_from_db
root=Path('/mnt/wd4tb/scratch/f562-db-cardinality-20261015')
root.mkdir(exist_ok=True)
results=[]
for case in ['crop_conflict','stored_conflict','stored_identical','crop_alternative_view']:
 db=root/(case+'.db')
 key='plate1_r1_c1_f1_o1'
 crop={'prcfo':key,'png_path':'/declared-synthetic/cell.png','plateID':'plate1','rowID':'r1','columnID':'c1','fieldID':'f1'}
 crops=[crop]
 stored=[{'prcfo':key,'emb_0':1.0,'emb_1':2.0}]
 if case=='crop_conflict': crops.append(dict(crop,plateID='other_plate'))
 if case=='crop_alternative_view': crops.append(dict(crop,png_path='/declared-synthetic/nucleus.png'))
 if case=='stored_conflict': stored.append(dict(stored[0],emb_0=99.0))
 if case=='stored_identical': stored.append(dict(stored[0]))
 with sqlite3.connect(db) as con:
  pd.DataFrame(crops).to_sql('png_list',con,index=False,if_exists='replace')
  pd.DataFrame(stored).to_sql('crop_embedding',con,index=False,if_exists='replace')
  indexes={name:con.execute('PRAGMA index_list('+name+')').fetchall() for name in ['png_list','crop_embedding']}
 before=hashlib.sha256(db.read_bytes()).hexdigest()
 frame=_mil_frame_from_db(db)
 after=hashlib.sha256(db.read_bytes()).hexdigest()
 assert before==after
 results.append({'case':case,'indexes':indexes,'input_crops':len(crops),'input_stored':len(stored),'result_rows':len(frame),'result':frame.to_dict('records'),'database_sha256':before,'unchanged':True})
(root/'counterexample.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
