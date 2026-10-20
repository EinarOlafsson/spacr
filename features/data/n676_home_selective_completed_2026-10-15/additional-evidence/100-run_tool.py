#!/usr/bin/env python3
"""Execute an explicitly selected selective05 tool on bound integrated source."""
from pathlib import Path
import sys
ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'tools/tutorials'), str(ROOT/'tools/tutorials/authoring/tools')]
sys.meta_path[:] = [f for f in sys.meta_path if '__editable__' not in (getattr(f, '__module__', '') or type(f).__module__)]
import os, hashlib, subprocess, runpy, json
assert Path.cwd().resolve() == ROOT, 'Wrong worktree cwd'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU downstream only'
assert Path(os.environ['SPACR_TUTORIAL_WORKSPACE']).resolve()==Path(os.environ['STAGE']).resolve(), 'Stage-tools must target actual STAGE, never inherited voice workspace'
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == os.environ['SOURCE_COMMIT'], 'Final source pin changed'
english = ROOT/'tools/tutorials/lessons/05_home.json'
assert hashlib.sha256(english.read_bytes()).hexdigest() == os.environ['APPROVED_ENGLISH_SHA256'], 'Approved English pin changed'
accepted = Path(os.environ['ACCEPTED_CAPTURE_EVIDENCE'])
assert hashlib.sha256(accepted.read_bytes()).hexdigest() == os.environ['ACCEPTED_CAPTURE_SHA256'], 'Capture evidence pin changed'
proof=json.loads(accepted.read_text())
assert proof['lesson']=='05_home' and proof['accepted'] is True and proof['all_selected_clips_accepted'] is True
assert proof['selected_visuals']==[s['visual'] for s in json.loads(english.read_text())['scenes']]
assert proof['whole_fresh_capture'] is False, 'Selective reuse must not be restamped as whole-fresh capture'
capture_render_commit = '87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
assert proof['source_commit']==capture_render_commit and proof['english_sha256']==os.environ['APPROVED_ENGLISH_SHA256']
assert os.environ['APPROVED_ENGLISH_SHA256']=='d57ea978b09a03132fad3f1d97dc83eabcb2403e607adb55deaaa4ef94ef2800'
capture_commits=proof['capture_source_commits']
assert isinstance(capture_commits,list) and capture_commits and len(capture_commits)==len(set(capture_commits))
for commit in capture_commits:
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--quiet',commit,'--','spacr'],cwd=ROOT,check=True)
def source_guard():
    final = os.environ['SOURCE_COMMIT']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==final
    changed = subprocess.check_output(['git','diff','--name-only',capture_render_commit,final],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','merge-base','--is-ancestor',capture_render_commit,final],cwd=ROOT,check=True)
    expected = {'tools/tutorials/authoring/web/app_v2.js','tools/tutorials/tests/test_player_preserve_visual_intervals.py'}
    assert set(changed)==(set() if final==capture_render_commit else expected), 'Exact player/test follow-up only'
    dirty = subprocess.check_output(['git','status','--porcelain','--','spacr','tools/tutorials','tests',':(exclude)tools/tutorials/release_candidate'],cwd=ROOT,text=True)
    assert not dirty.strip(), 'Relevant implementation/canonical source must be clean'
    subprocess.run(['git','diff','--quiet',capture_render_commit,final,'--','spacr','tools/tutorials/lessons/05_home.json','tools/tutorials/lessons/reviews/05_home.*.json'],cwd=ROOT,check=True)
    selection_binding=proof['selection'];selection_path=Path(selection_binding['path'])
    assert selection_path.stat().st_size==selection_binding['bytes'] and hashlib.sha256(selection_path.read_bytes()).hexdigest()==selection_binding['sha256']
    selection=json.loads(selection_path.read_text())
    for binding in selection['source_inventory_verification']['inventory_bindings']:
        path=Path(binding['path']);assert path.stat().st_size==binding['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==binding['sha256']
        inventory=json.loads(path.read_text())['sha256']
        assert len(inventory)==502 and not (expected & set(inventory)), 'Player follow-up must be outside all captured502 source inventories'
    freeze=json.loads((Path(os.environ['VOICES'])/'source-freeze.json').read_text())
    assert freeze['source_commit']==capture_render_commit and freeze['source_dirty'] is False and freeze['selected']==['05_home']
    assert freeze['lesson_sha256']=={'05_home':os.environ['APPROVED_ENGLISH_SHA256']}
source_guard()
script = Path(sys.argv[1]).resolve()
assert script.is_relative_to(ROOT/'tools/tutorials') or script.parent == Path(__file__).resolve().parent, 'Unexpected script root'
sys.argv = [str(script), *sys.argv[2:]]
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    source_guard()
