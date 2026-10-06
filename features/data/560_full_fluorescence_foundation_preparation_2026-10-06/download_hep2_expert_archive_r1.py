from pathlib import Path
import hashlib
import json
import requests

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
metadata = json.loads((root / 'epithelial-record.json').read_text())
entry = next(row for row in metadata['files'] if row['key'] == 'HEp-2-ExpD.zip')
assert entry['size'] == 141882774
assert entry['checksum'] == 'md5:89b9228c62bb547bb7ab31234764ed74'
target = root / entry['key']
assert not target.exists()
url = entry['links']['self']
md5, sha = hashlib.md5(), hashlib.sha256()
with requests.get(url, stream=True, timeout=(30, 60)) as response:
    response.raise_for_status()
    with target.open('xb') as stream:
        for block in response.iter_content(1024 * 1024):
            stream.write(block)
            md5.update(block)
            sha.update(block)
assert target.stat().st_size == entry['size']
assert md5.hexdigest() == entry['checksum'].split(':')[1]
receipt = {'primary_concept_DOI': '10.5281/zenodo.18337703',
           'resolved_record_id': metadata['id'],
           'resolved_version_DOI': metadata.get('doi'),
           'primary_file_url': url,
           'published_MD5_verified': md5.hexdigest(),
           'original_archive_sha256': sha.hexdigest(),
           'original_archive_bytes': target.stat().st_size,
           'license': metadata['metadata']['license'],
           'dataset': 'Original manually expert-classified ExpD, not CNN-D',
           'paper_DOI': '10.1038/s41597-026-08216-w',
           'download_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'archive-download.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS original expert ExpD archive', receipt['original_archive_bytes'], receipt['original_archive_sha256'], 'version', receipt['resolved_version_DOI'], flush=True)
