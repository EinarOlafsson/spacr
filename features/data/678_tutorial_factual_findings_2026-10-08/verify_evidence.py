"""Verify bounded audit evidence; optionally check its frozen catalog snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    receipt = json.loads((directory / 'receipt.json').read_text())
    archive = directory / receipt['archive']
    assert digest(archive.read_bytes()) == receipt['archive_sha256']
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        assert len(members) == len(receipt['payloads'])
        assert {member.name for member in members} == set(receipt['payloads'])
        for member in members:
            assert member.isfile()
            data = stream.extractfile(member).read()
            expected = receipt['payloads'][member.name]
            assert len(data) == expected['bytes']
            assert digest(data) == expected['sha256']
        provenance = json.loads(stream.extractfile('provenance.json').read())
    assert len(provenance['inventory']) == receipt['offered_lessons'] == 84
    assert sum(row['user_text_fields'] for row in provenance['inventory']) == receipt['user_text_fields']
    if args.candidate:
        assert digest((args.candidate / 'release-manifest.json').read_bytes()) == receipt['candidate_manifest_sha256']
        for name, expected in provenance['catalogs'].items():
            assert digest((args.candidate / 'web/catalog' / name).read_bytes()) == expected
    print(f"PASS: {len(members)} archive payloads; {receipt['offered_lessons']} offered lessons inventoried")


if __name__ == '__main__':
    main()
