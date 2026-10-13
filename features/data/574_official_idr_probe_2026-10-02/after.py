import hashlib
import json
import subprocess
import sys
from pathlib import Path

base = Path(__file__).parent
stage = base / 'run'
with (stage / 'cli-after.txt').open('w') as log:
    subprocess.run([sys.executable, '-m', 'spacr.cli', 'archive-package', '--src', str(stage / 'source'),
                    '--out', str(stage / 'after'), '--metadata', str(stage / 'metadata.json'), '--copy-images'],
                   check=True, stdout=log, stderr=subprocess.STDOUT)
study = next((stage / 'after').glob('*/idr/*-study.txt'))
parser = base / 'idr-utils/pyidr/study_parser.py'
before = json.loads((stage / 'provenance.json').read_text())
assert hashlib.sha256(parser.read_bytes()).hexdigest() == before['idr-utils']['parser_sha256']
command = [sys.executable, str(parser), str(study), '--report']
result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(stage / 'idr-after.txt').write_text(result.stdout)
assert result.returncode != 0 and 'Unmatched name spacr-example-screen/screenA' in result.stdout, result.stdout
assert 'Study Publication Title\t\n' in study.read_text()
assert 'Comment[IDR Study Accession]\t\n' in study.read_text()
for path, digest in before['source_images'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
(stage / 'after-proof.json').write_text(json.dumps({
    'command': command, 'official_returncode': result.returncode,
    'publication_row_repaired': True, 'unknown_publication_title_blank': True,
    'official_parser_unchanged': True, 'source_images_unchanged': True,
    'full_official_acceptance': False,
    'remaining_blocker': 'IDR parser requires curator-assigned accession/study-screen naming; exported submission template intentionally leaves accession blank. No accession was fabricated.'
}, indent=2) + '\n')
print(result.stdout)
