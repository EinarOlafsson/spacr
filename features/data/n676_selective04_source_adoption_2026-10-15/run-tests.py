import sys, pathlib, socket, subprocess, pytest, json, hashlib, ast
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015')
sys.path.insert(0,str(root));out=pathlib.Path('/mnt/wd4tb/scratch/work04-source-implementation-20261015')
def audit(event,args):
    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('Network forbidden in source adoption tests')
    if event=='subprocess.Popen' and any(x in str(args[0]).lower() for x in ('ffmpeg','ffprobe','chrome','chromium')): raise RuntimeError('Encoding/browser forbidden in source adoption tests')
sys.addaudithook(audit)
pins=json.loads((out/'source-pins.json').read_text())
for row in pins['files']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256']
code=pytest.main(['-q','--disable-warnings','--junitxml='+str(out/'owning-tests.xml'),'tools/tutorials/tests/test_retained_mixed_visual_refresh.py','tools/tutorials/tests/test_retained_visual_refresh.py','tools/tutorials/tests/test_merged_install_command_routes.py'])
for row in pins['files']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256']
raise SystemExit(code)
