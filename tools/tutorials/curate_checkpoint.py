"""Copy a saved Curate file AND its ledger into a new external checkpoint.

After Save, before reopening the mask:
    python curate_checkpoint.py /path/mask.tif /path/new-checkpoint
This is a separate backup step, NOT a repair or restoration of Curate history.
Do not edit the source while copying. Existing destinations are refused.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def checkpoint(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    ledger=Path(str(source)+'.curation.json')
    if destination.exists():raise FileExistsError('Choose a NEW checkpoint directory')
    if not source.is_file() or not ledger.is_file():
        raise ValueError('Both saved data and its curation ledger must exist')
    log=json.loads(ledger.read_text())
    if not isinstance(log.get('edits'),list):raise ValueError('Expected a recorded edit list')
    before={str(path):sha(path) for path in (source,ledger)}
    destination.mkdir(parents=True)
    copies={}
    for path in (source,ledger):
        target=destination/path.name;shutil.copy2(path,target)
        if sha(target)!=before[str(path)]:raise ValueError('Checkpoint copy differs')
        copies[str(target)]=sha(target)
    if any(sha(path)!=value for path,value in before.items()):
        raise ValueError('A source changed during backup; do not trust this checkpoint')
    receipt=dict(source_sha256=before,copied_sha256=copies,edit_count=len(log['edits']),
        scope='External exact-byte checkpoint, not automatic app history restoration',
        source_unchanged=True)
    (destination/'checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Preserved saved data and ledger:',destination)
    return receipt


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit(__doc__)
    checkpoint(sys.argv[1],sys.argv[2])
