import json,runpy,sys
from pathlib import Path
out=Path('/mnt/wd4tb/scratch/n43-readme-size-maintenance-20261015/run')
def audit(event,args):
    if event in {'socket.connect','socket.getaddrinfo'}:
        with (out/'network-refusals.jsonl').open('a') as stream: stream.write(json.dumps({'event':event,'args':repr(args)})+'\n')
        raise RuntimeError('Offline README maintenance: network forbidden')
sys.addaudithook(audit)
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015');sys.path[:0]=[str(root),str(root/'tools')]
mode=sys.argv[1];args=sys.argv[2:]
if mode=='build':
    sys.argv=['tools/build_readme_images.py']+args
    runpy.run_path(str(root/'tools/build_readme_images.py'),run_name='__main__')
elif mode=='pytest':
    import pytest
    raise SystemExit(pytest.main(args))
else: raise ValueError(mode)
