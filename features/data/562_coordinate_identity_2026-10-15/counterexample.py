from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import hashlib,json,sqlite3,sys
import numpy as np
import pandas as pd
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-coordinate-identity-20261015')
sys.path.insert(0,str(root))
from spacr import active_learning as al
from spacr.embeddings import EmbeddingSpec,_mil_frame_from_db,_mil_bags
scratch=Path('/mnt/wd4tb/scratch/f562-coordinate-identity-20261015')
a=EmbeddingSpec(normalize=True);b=EmbeddingSpec(normalize=False)
assert a.fingerprint()!=b.fingerprint()
keys=[f'plate1_r1_c{well+1}_f1_o{cell+1}' for well in range(4) for cell in range(2)]
values=np.arange(16,dtype=np.float32).reshape(8,2)
crops=pd.DataFrame([{'prcfo':key,'png_path':f'/declared-synthetic/{key}.png','plateID':'plate1','rowID':'r1','columnID':f'c{well+1}','fieldID':'f1'} for well in range(4) for cell,key in enumerate(keys[2*well:2*well+2])])
labels=pd.DataFrame({'wellID':[f'plate1_r1_c{well+1}' for well in range(4)],'well_label':[0,0,1,1]})
entry=SimpleNamespace(sha256='a'*64,key='declared-synthetic-encoder',source='local')
result=SimpleNamespace(values=values,columns=('emb_0','emb_1'),spec=a)
observations=[]
for case in ['fingerprint','weights','consistent','missing','unknown']:
 database=scratch/(case+'.db')
 with sqlite3.connect(database) as con:crops.to_sql('png_list',con,index=False,if_exists='replace')
 al._store_crop_embeddings(str(database),keys,result,encoder_entry=entry)
 with sqlite3.connect(database) as con:
  if case=='fingerprint':con.execute('UPDATE crop_embedding SET _embedding_fingerprint=?,_embedding_spec=? WHERE rowid>4',(b.fingerprint(),json.dumps(asdict(b),sort_keys=True)))
  if case=='weights':con.execute('UPDATE crop_embedding SET _embedding_weights_sha256=? WHERE rowid>4',('b'*64,))
  if case=='missing':con.execute('UPDATE crop_embedding SET _embedding_weights_sha256=NULL,_embedding_fingerprint=NULL WHERE rowid>4')
  if case=='unknown':con.execute("UPDATE crop_embedding SET _embedding_weights_sha256='unknown',_embedding_fingerprint='unknown' WHERE rowid>4")
 before=hashlib.sha256(database.read_bytes()).hexdigest()
 frame=_mil_frame_from_db(str(database),labels)
 original=frame.copy(deep=True)
 bags,targets,wells,rows,kept=_mil_bags(frame)
 pd.testing.assert_frame_equal(frame,original)
 assert before==hashlib.sha256(database.read_bytes()).hexdigest()
 assert len(bags)==4 and [len(bag) for bag in bags]==[2]*4
 observations.append({'case':case,'accepted':True,'bag_count':len(bags),'cells':sum(map(len,bags)),'fingerprints':frame._embedding_fingerprint.astype(str).unique().tolist(),'weights':frame._embedding_weights_sha256.astype(str).unique().tolist(),'database_sha256':before,'database_and_frame_unchanged':True})
writer=scratch/'normal-writer.db'
with sqlite3.connect(writer) as con:crops.to_sql('png_list',con,index=False,if_exists='replace')
al._store_crop_embeddings(str(writer),keys,result,encoder_entry=entry)
al._store_crop_embeddings(str(writer),keys[:1],SimpleNamespace(values=values[:1],columns=('emb_0','emb_1'),spec=b),encoder_entry=entry)
with sqlite3.connect(writer) as con:
 replacement=pd.read_sql_query('SELECT * FROM crop_embedding',con)
assert len(replacement)==1 and replacement._embedding_fingerprint.eq(b.fingerprint()).all()
receipt={'input':'Declared synthetic numeric vectors and ModelEntry-like metadata; no model inference/checkpoint/training.','source_pin':'1d90f7c921b','actual_spec_fingerprints':[a.fingerprint(),b.fingerprint()],'normal_writer_replaces_incompatible_spec':True,'rows_after_different_spec_write':len(replacement),'observations':observations,'limits':['Proves current reader/bags accept observed incompatible coordinate identities; does not measure model accuracy or fabricate trusted checkpoint identities.','Missing/unknown cases accepted legacy control, not proof of compatible or verified vectors.']}
(scratch/'counterexample.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
