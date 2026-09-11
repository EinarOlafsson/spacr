"""Package the exact source used by the recorded per-array overlay command."""
import hashlib
from pathlib import Path
import zipfile

from stage_lesson import REPO, read


def build():
    root = Path(__file__).resolve().parent
    receipt = read(root / 'evidence/2026-09-11_mask_overlay_recorded_checks.json')
    helper = root / 'export_mask_overlays.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != receipt['helper_sha256']:
        raise ValueError('Download source must exactly match the recorded command')
    target = REPO / 'docs/source/_extra/tutorials/examples/Mask_overlay_API_workaround.zip'
    contents = {'export_mask_overlays.py': helper.read_bytes(),
                'README.txt': (root / 'mask_overlay_README.txt').read_bytes()}
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
