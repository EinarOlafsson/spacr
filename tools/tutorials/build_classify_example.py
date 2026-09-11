"""Build the exact recorded preparation helper's small downloadable ZIP."""
import hashlib
from pathlib import Path
import zipfile

from stage_lesson import REPO, read


def build():
    root = Path(__file__).resolve().parent
    receipt = read(root / 'evidence/2026-09-11_classify_canonical_recording_checks.json')
    helper = root / 'prepare_classify_split.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != receipt['preparation_helper_sha256']:
        raise ValueError('The downloadable helper must be the exact recorded source')
    target = REPO / 'docs/source/_extra/tutorials/examples/Classify_existing_split_example.zip'
    contents = {'prepare_classify_split.py': helper.read_bytes(),
                'README.txt': (root / 'classify_existing_split_README.txt').read_bytes()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist()) != set(contents):
            raise ValueError('Archive integrity or members differ')
        if any(archive.read(name) != content for name, content in contents.items()):
            raise ValueError('Archive content differs from recorded sources')
    print(target, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    build()
