import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
selected=[p for p in paths if p and (p.startswith(('spacr/','tools/','docs/i18n/','docs/source/','packaging/')) or p.startswith('README') or p in ('setup.py','pyproject.toml','MANIFEST.in'))]
rows={}
for name in selected:
 p=root/name
 if p.is_symlink(): rows[name]={'symlink':str(p.readlink())};continue
 if not p.is_file(): raise RuntimeError('Missing tracked input: '+name)
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 rows[name]=h.hexdigest()
record={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'files':rows}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print('Pinned',len(rows),'tracked source/resource/catalog/README inputs',flush=True)
