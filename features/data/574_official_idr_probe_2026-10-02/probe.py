import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

base = Path(__file__).parent
stage = base / 'run'
source = stage / 'source'
source.mkdir(exist_ok=True)
cached = Path('/home/carruthers/.cache/spacr/example_data/plate1')
images = [next(cached.glob('plate1_' + well + '_*.tif')) for well in ('E01', 'E02')]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path): sha(path) for path in images}
for path in images:
    shutil.copy2(path, source / path.name)
metadata = dict(title='spaCR example screen', description='Private technical validation of two cached example wells. Not a submission.',
                authors='Researcher Example', email='example@example.org', affiliation='Example Institute',
                organism='Homo sapiens', cell_line='HeLa', microscope='Yokogawa CellVoyager')
(stage / 'metadata.json').write_text(json.dumps(metadata))
command = [sys.executable, '-m', 'spacr.cli', 'archive-package', '--src', str(source), '--out', str(stage / 'before'), '--metadata', str(stage / 'metadata.json'), '--copy-images']
with (stage / 'cli-before.txt').open('w') as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
study = next((stage / 'before').glob('*/idr/*-study.txt'))
parser = base / 'idr-utils/pyidr/study_parser.py'
command = [sys.executable, str(parser), str(study), '--report']
result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(stage / 'idr-before.txt').write_text(result.stdout)
assert result.returncode != 0 and 'Study Publication Title' in result.stdout, result.stdout
assert before == {str(path): sha(path) for path in images}
provenance = {name: {'commit': subprocess.check_output(['git', '-C', str(base / name), 'rev-parse', 'HEAD'], text=True).strip()} for name in ('idr-utils', 'idr-template', 'page-tab-specification')}
provenance['idr-utils']['parser_sha256'] = sha(parser)
provenance['source_images'] = before
provenance['official_command'] = command
provenance['official_returncode'] = result.returncode
(stage / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(result.stdout)
