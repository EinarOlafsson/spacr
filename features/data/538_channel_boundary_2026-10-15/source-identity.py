from pathlib import Path
import ast,hashlib,json,subprocess
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f538-channel-boundary-20261015')
current=(root/'spacr/psf_pipeline.py').read_text()
def helper(text,name):
 tree=ast.parse(text)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(text,node)
proof={}
for name,pin in [('_prepare_unmixing','417e1dd3700'),('_prepare_measure_unmixing','417e1dd3700')]:
 approved=subprocess.check_output(['git','show',pin+':spacr/psf_pipeline.py'],cwd=root,text=True)
 actual=helper(current,name);assert actual==helper(approved,name),name
 proof[name]={'approved_pin':pin,'exact_source_segment_unchanged':True,'sha256':hashlib.sha256(actual.encode()).hexdigest()}
reviews=json.loads((root/'features/data/538_channel_boundary_2026-10-15/peer-adoption.json').read_text())
for row in reviews:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['final_sha256']
receipt={'current_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'helpers':proof,'nine_exact_review_files_match_peer_adoption':True}
(root/'features/data/538_channel_boundary_2026-10-15/source-before-closure.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
