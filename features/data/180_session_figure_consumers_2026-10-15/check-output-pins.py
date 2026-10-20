import hashlib,json,pathlib
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-graph-api-20261015')
p=pathlib.Path('/mnt/wd4tb/scratch/home-b-n180-figure-producer-20261015/run-1/frozen-output-hashes.json');d=json.loads(p.read_text())
for name,wanted in d['artifacts'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted,name
assert hashlib.sha256(pathlib.Path(d['before_help']).read_bytes()).hexdigest()==d['before_help_sha256']
print('Frozen output hashes verified:',len(d['artifacts']))
