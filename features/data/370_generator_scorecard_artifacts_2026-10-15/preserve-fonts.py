from pathlib import Path
import hashlib,json,subprocess
from tools.build_score_cards import without_font_data
root=Path.cwd();kept=[]
for name in subprocess.check_output(['git','diff','--name-only','--','docs/models/cards'],text=True).splitlines():
 before=subprocess.check_output(['git','show','HEAD:'+name]);after=(root/name).read_bytes()
 assert without_font_data(before.decode()) == without_font_data(after.decode()),name
 kept.append({'path':name,'before_sha256':hashlib.sha256(before).hexdigest(),'generated_sha256':hashlib.sha256(after).hexdigest(),'proof':'Only font embedding serialization differs under normal without_font_data equality.'})
if kept:subprocess.run(['git','restore','--source=HEAD','--worktree','--',*[x['path'] for x in kept]],check=True)
Path('/mnt/wd4tb/scratch/home-b-f370-generator-20261015/font-byte-preservation.json').write_text(json.dumps(kept,indent=2)+'\n')
print('Preserved',len(kept),'unchanged SVGs')
