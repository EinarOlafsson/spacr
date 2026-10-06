from pathlib import Path
import hashlib
import json
import urllib.request

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/552-instanseg-brightfield-r2')
root.mkdir(exist_ok=False)
def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'spacr-source-validation'})
    return urllib.request.urlopen(request, timeout=90).read()
head_url = 'https://api.github.com/repos/instanseg/instanseg/commits/main'
head_raw = fetch(head_url)
head = json.loads(head_raw)['sha']
tree_raw = fetch('https://api.github.com/repos/instanseg/instanseg/git/trees/' + head + '?recursive=1')
tree = json.loads(tree_raw)
assert not tree.get('truncated')
matches = [row for row in tree['tree'] if row['path'] == 'instanseg/examples/HE_example.tif']
assert len(matches) == 1, [row['path'] for row in tree['tree'] if row['path'].startswith('examples/')]
row = matches[0]
url = 'https://raw.githubusercontent.com/instanseg/instanseg/' + head + '/' + row['path']
data = fetch(url)
assert len(data) == row['size'] and not data.startswith(b'version https://git-lfs')
assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['sha']
readme = fetch('https://raw.githubusercontent.com/instanseg/instanseg/' + head + '/README.md')
assert b'HE_example.tif' in readme and b'brightfield_nuclei' in readme
(root / 'HE_example.tif').write_bytes(data)
(root / 'upstream-README.md').write_bytes(readme)
(root / 'upstream-head.json').write_bytes(head_raw)
(root / 'upstream-tree.json').write_bytes(tree_raw)
report = {'source_url': url, 'upstream_commit': head, 'upstream_git_blob': row['sha'],
          'source_sha256': hashlib.sha256(data).hexdigest(), 'source_bytes': len(data),
          'original_upstream_HE_example_bytes_unchanged': True,
          'README_designates_as_brightfield_example_for_brightfield_nuclei': True,
          'no_independent_human_reference_labels_claimed': True}
(root / 'source-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
