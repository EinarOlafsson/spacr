"""Verify this immutable N704 packet and an owner's isolated candidate tree."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--candidate', required=True, help='Git tree or commit containing this isolated promotion')
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    receipt = json.loads((packet / 'receipt.json').read_text())
    for name, expected in receipt['files'].items():
        data = (packet / name).read_bytes()
        assert len(data) == expected['bytes'] and sha(data) == expected['sha256'], name
    manifest = json.loads(gzip.decompress((packet / 'manifest.json.gz').read_bytes()))
    patch = gzip.decompress((packet / 'readme-docs-only.patch.gz').read_bytes())
    assert sha(patch) == manifest['patch_sha256']
    paths = (packet / 'files.txt').read_text().splitlines()
    assert len(paths) == 738 and paths == sorted(manifest['source_and_promoted_files'])
    assert set((packet / 'historical-678-files.txt').read_text().splitlines()) <= set(paths)

    def git(*command):
        return subprocess.check_output(['git', *command], cwd=args.repo)

    candidate = git('rev-parse', args.candidate + '^{tree}').decode().strip()
    changes = git('diff', '--name-only', manifest['base_main_commit'], candidate).decode().splitlines()
    assert changes == paths, 'Candidate changes differ from the exact packet scope'
    assert candidate == manifest['promotion_tree'], 'Candidate does not reproduce the proven tree'
    expected = dict(manifest['source_and_promoted_files'])
    expected.update(manifest['referenced_assets'])
    observed = {}
    for line in git('ls-tree', '-r', candidate).decode().splitlines():
        fields, path = line.split('\t', 1)
        mode, kind, oid = fields.split()
        if path in expected:
            observed[path] = (mode, kind, oid)
    assert set(observed) == set(expected)
    for path, record in expected.items():
        assert observed[path] == (record['mode'], record['kind'], record['oid']), path
    refs = [candidate + ':' + path for path in sorted(expected)]
    raw = subprocess.run(['git', 'cat-file', '--batch'], cwd=args.repo,
                         input=('\n'.join(refs) + '\n').encode(),
                         stdout=subprocess.PIPE, check=True).stdout
    offset = 0
    for path in sorted(expected):
        end = raw.index(b'\n', offset)
        fields = raw[offset:end].split()
        assert len(fields) == 3 and fields[1] == b'blob', path
        size = int(fields[2]); start = end + 1
        assert sha(raw[start:start + size]) == expected[path]['sha256'], path
        assert raw[start + size:start + size + 1] == b'\n'
        offset = start + size + 1
    assert offset == len(raw)
    print(json.dumps({'status': 'PASS', 'candidate_tree': candidate,
                      'exact_promoted_files': len(paths),
                      'referenced_image_assets': len(manifest['referenced_assets']),
                      'scope': 'Local immutable packet/candidate bytes only; no hosted/rendered acceptance'}))


if __name__ == '__main__':
    main()
