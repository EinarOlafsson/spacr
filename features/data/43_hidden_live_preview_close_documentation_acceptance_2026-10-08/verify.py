from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for row in receipt['payloads']:
 stored=(root/row['path']).read_bytes();assert len(stored)==row['bytes'] and hashlib.sha256(stored).hexdigest()==row['sha256']
 raw=gzip.decompress(stored);assert len(raw)==row['raw_bytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256']
print('PASS',len(receipt['payloads']),'portable documentation payloads')
