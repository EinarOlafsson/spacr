"""Package the exact API helper used by the real Embeddings recording."""
from pathlib import Path
import zipfile

from embeddings_example import digest
from stage_lesson import DEFAULT_STAGE, REPO, read


def build():
    root = Path(__file__).resolve().parent
    proof = read(DEFAULT_STAGE / 'captures/embeddings_1507_api/scientific_acceptance.json')
    helper = root / 'embeddings_example.py'
    if proof.get('accepted') is not True or proof.get('helper_sha256') != digest(helper):
        raise ValueError('Only the exact successfully recorded helper can be packaged')
    target = REPO / 'docs/source/_extra/tutorials/examples/Embeddings_API_example.zip'
    sources = {'embeddings_example.py': helper, 'README.txt': root / 'embeddings_README.txt'}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist()) != set(sources):
            raise ValueError('Invalid example archive')
        for name, path in sources.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError('Example archive does not contain the recorded helper')
    print(target, digest(target))


if __name__ == '__main__':
    build()
