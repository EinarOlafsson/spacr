"""Package the exact saved masks and helper used in the actual API recording."""
from pathlib import Path
import json
import zipfile

from model_compare_example import EXPECTED
from ops_geometry_example import digest
from stage_lesson import DEFAULT_STAGE, REPO, read


def build():
    root = Path(__file__).resolve().parent
    capture = read(DEFAULT_STAGE / 'captures/model_compare_1507_api/scientific_acceptance.json')
    helper = root / 'model_compare_example.py'
    if capture.get('accepted') is not True or capture['helper_sha256'] != digest(helper):
        raise ValueError('Require the exact successful recorded API helper')
    if capture['run']['sources'] != EXPECTED or capture['run']['accuracy_validated'] is not False:
        raise ValueError('Require real saved-mask provenance without an accuracy claim')
    target = REPO / 'docs/source/_extra/tutorials/examples/Model_Compare_saved_masks_example.zip'
    if target.exists():
        raise FileExistsError('Preserve the existing example bundle')
    content = {'model_compare_example.py': helper.read_bytes(),
               'ops_geometry_example.py': (root / 'ops_geometry_example.py').read_bytes(),
               'README.txt': (root / 'model_compare_README.txt').read_bytes()}
    for name, expected in EXPECTED.items():
        path = DEFAULT_STAGE / 'model_compare_1507_inputs' / name
        if digest(path) != expected:
            raise ValueError('An actual recorded input changed')
        content['inputs/' + name] = path.read_bytes()
    content['source_manifest.json'] = (json.dumps({
        'sources': EXPECTED, 'field': 'cell_pair_02', 'helper_sha256': digest(helper),
        'original_capture': 'cellpose_apply_native_zoom_v2',
        'same_cpsam_weights_different_preprocessing': True,
        'ground_truth': False, 'accuracy_validated': False}, indent=2) + '\n').encode()
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist()) != set(content):
            raise ValueError('Invalid example archive')
        if any(archive.read(name) != data for name, data in content.items()):
            raise ValueError('Packaged files differ from their verified originals')
    print(target, digest(target), target.stat().st_size)


if __name__ == '__main__':
    build()
