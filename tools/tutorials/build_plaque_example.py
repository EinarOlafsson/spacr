"""Package only the preserved SYNTHETIC teaching inputs, never old results/models."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

README = '''SYNTHETIC PLAQUE TUTORIAL EXAMPLE

These are invented grayscale images and hand-constructed labelled masks,
not acquired biological data. Control/treatment names are teaching labels;
they do not establish an experimental comparison or a drug effect.

Extract this archive into a NEW folder. Set Source to SYNTHETIC_plaque,
the folder containing the four TIFFs and its masks subfolder. With Masks
OFF, Plaque Analysis reuses the included masks rather than segmenting again.
Do not overwrite these masks when trying a new segmentation: use a copy.

Expected saved-mask object counts: A01=4, A02=3, B01=2, B02=1.
Areas are pixels; no well geometry or physical calibration is supplied.
The manifest gives the exact constructed object areas. A newly downloaded
model's preview is a SEPARATE computation and need not reproduce these masks.
No old database, saved results, settings, or model weights are included.
'''


def archive_members(source):
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    names = [row['file'] for row in manifest['records']]
    if len(names) != 4 or len(set(names)) != 4 or any(Path(n).name != n or not n.endswith('.tif') for n in names):
        raise ValueError('Expected exactly four distinct plain TIFF names')
    members = {'manifest.json': (source/'manifest.json').read_bytes(), 'README.txt': README.encode()}
    for name in names:
        for relative in (name, 'masks/'+name):
            members[relative] = (source/relative).read_bytes()
    return members


def build_archive(source, destination):
    members = archive_members(source)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo('SYNTHETIC_plaque/'+name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    data = buffer.getvalue()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError('Refusing to overwrite a different example archive')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
    if archive_members(source) != members:
        raise ValueError('The source inputs changed while packaging')
    with zipfile.ZipFile(destination) as archive:
        if {name.removeprefix('SYNTHETIC_plaque/'): archive.read(name) for name in archive.namelist()} != members:
            raise ValueError('The packaged example does not exactly preserve its inputs')
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), synthetic=True,
                members={k: hashlib.sha256(v).hexdigest() for k, v in members.items()},
                old_results_included=False, models_included=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_archive(args.source, args.destination), indent=2))
