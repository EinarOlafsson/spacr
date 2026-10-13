from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/tmp/spacr-implementation-20261001/f575-two-well')
repo=Path('/tmp/spacr-implementation-20261001/suggest-capture')
stage=Path(sys.argv[1]).resolve()
(stage/'source-revision.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'production_sources':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ['spacr/run_journal.py','spacr/qt/bridge.py','spacr/measure.py','spacr/qt/screens/app_screen.py','spacr/cli.py','spacr/cli_repro.py']},'acceptance_source':'Isolated base plus retained source.patch; no other agent edits this tree.'},indent=2)+'\n')
(stage/'source.patch').write_bytes(subprocess.check_output(['git','diff','--','spacr/run_journal.py','spacr/qt/bridge.py'],cwd=repo))
for name in ['gui_run.py','replay.py']:
    subprocess.run([sys.executable,str(root/name),str(stage)],check=True)
print('FINAL GUI TO SNAKEMAKE ACCEPTANCE PASS',flush=True)
