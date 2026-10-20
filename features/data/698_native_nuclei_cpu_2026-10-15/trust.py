import ast,hashlib,json
from pathlib import Path
repo=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015')
tree=ast.parse((repo/'spacr/model_zoo.py').read_text());bundled=next(n.value for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='BUNDLED_REMOTE_MODELS')
key='nuclei_from_cellmask_v1';record=next(ast.literal_eval(n) for n in bundled.elts if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='key' and isinstance(v,ast.Constant) and v.value==key for k,v in zip(n.keys,n.values)))
checkpoint=Path('/home/olafsson/.spacr/models')/record['name'];stat=checkpoint.stat();digest=hashlib.sha256()
with checkpoint.open('rb') as handle:
 for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
assert digest.hexdigest()==record['sha256'];after=checkpoint.stat();assert (after.st_size,after.st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
receipt={'key':key,'file':str(checkpoint),'bytes':stat.st_size,'sha256':digest.hexdigest(),'trusted_bundled_sha256_exact':True,'architecture':record['architecture'],'bounded_input':'One declared synthetic128x160uint16cellmask-like field, no biological groundtruth/accuracy claim','checkpoint_stat_unchanged':True}
Path('/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/trust.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,sort_keys=True))
