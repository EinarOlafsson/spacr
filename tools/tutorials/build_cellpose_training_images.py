"""Add three separate Apply images to the public Cellpose training example.

The six training pairs in ``Cellpose_training_images_masks.zip`` stay
byte-identical. The new ``apply/`` folder holds three 512 x 512 crops of the
same cell channel from wells that the training pairs do not use, so the Train
lesson can apply its new checkpoint to images the model did not train on.
No masks are shipped for them: Apply writes its own into ``apply/masks``.
The reference label counts in the manifest come from the source label plane
and are provenance, not independently reviewed ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / 'docs/source/_extra/tutorials/examples/Cellpose_training_images_masks.zip'
SOURCE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/tutorials/merged')
IMAGE_CHANNEL = 1
MASK_PLANE = 4
CROP = 512
# Wells B03, B04 and B05; the training pairs come from B01 and B02 only.
APPLY = (
    ('cell_field_01.tif', 'plate1_B03_1_1.npy', 480, 960),
    ('cell_field_02.tif', 'plate1_B04_3_1.npy', 960, 640),
    ('cell_field_03.tif', 'plate1_B05_4_1.npy', 640, 1120),
)
README = '''CELLPOSE TRAINING IMAGES AND MASKS

Extract this archive, then open Home > Tools > Make Masks > Cellpose Workbench > Train.
Set Source to the extracted training folder. Leave mask_src empty to use its masks subfolder.
The six 512 x 512 microscopy images have matching integer cell-label masks.
Inspect and correct the supplied masks in Make Masks before using them for your own model.
Choose a model name, base model and output folder, review the training settings, then Run.
The console reports the checkpoint path.

The apply folder holds three more 512 x 512 images of the same channel, taken from
wells that are not in the training set. In the Apply tab, set src to the extracted
apply folder, check that Custom model points to your checkpoint and turn on save.
Preview one image, then Run: Apply writes one mask per image into apply/masks.
Use separately labelled fields to evaluate the result.

Full walkthrough: https://einarolafsson.github.io/spacr/nightly/cellpose_training.html
The source manifest records the original tutorial field identities and crop locations.
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crop(name, row, column):
    path = SOURCE / name
    before = sha(path.read_bytes())
    merged = np.load(path, mmap_mode='r', allow_pickle=False)
    image = np.array(merged[row:row + CROP, column:column + CROP, IMAGE_CHANNEL])
    labels = np.array(merged[row:row + CROP, column:column + CROP, MASK_PLANE])
    if image.shape != (CROP, CROP) or image.dtype != np.uint16:
        raise ValueError(f'Expected a full uint16 crop from {name}')
    if sha(path.read_bytes()) != before:
        raise ValueError('A source field changed while it was read')
    return image, labels, before


def tiff_bytes(image):
    import tifffile
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, image)
    return buffer.getvalue()


def build(output, source=TARGET):
    """Write the extended example to ``output``; ``source`` is only read."""
    import tifffile
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output path for the extended example')
    with zipfile.ZipFile(source) as old:
        entries = {name: old.read(name) for name in old.namelist()}
    manifest = json.loads(entries['source_manifest.json'])
    if 'apply_images' in manifest or any(n.startswith('apply/') for n in entries):
        raise ValueError('The example already contains Apply images')
    training_wells = {pair['source'].split('_')[1] for pair in manifest['pairs']}
    for pair in manifest['pairs']:
        for key, path in (('image_sha256', 'training/'), ('mask_sha256', 'training/masks/')):
            if sha(entries[path + pair['file']]) != pair[key]:
                raise ValueError('A published training pair differs from its manifest')
    records, added = [], {}
    for filename, name, row, column in APPLY:
        if name.split('_')[1] in training_wells:
            raise ValueError('Apply images must come from wells outside the training set')
        image, labels, source_sha = crop(name, row, column)
        data = tiff_bytes(image)
        if not np.array_equal(tifffile.imread(io.BytesIO(data)), image):
            raise ValueError('The written TIFF does not round-trip')
        added['apply/' + filename] = data
        records.append(dict(file=filename, source=name, source_sha256=source_sha,
                            image_sha256=sha(data), origin_yx=[row, column],
                            reference_label_objects=int(np.count_nonzero(np.unique(labels))),
                            image_range=[int(image.min()), int(image.max())]))
    manifest['apply_images'] = records
    manifest['apply_images_scope'] = ('Separate wells from the training pairs, for trying a '
        'trained checkpoint in Apply. No masks are included; reference label counts come '
        'from the source label plane and are not reviewed ground truth.')
    entries['source_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    entries['README.txt'] = README.encode()
    entries.update(added)
    order = ['README.txt', 'source_manifest.json',
             *sorted(n for n in entries if n.startswith('training/') and '/masks/' not in n),
             *sorted(n for n in entries if n.startswith('training/masks/')),
             *sorted(added)]
    if set(order) != set(entries) or len(order) != 17:
        raise ValueError('Expected README, manifest, twelve training TIFFs and three Apply images')
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip.part', delete=False) as part:
        temporary = Path(part.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name in order:
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, entries[name])
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() or any(check.read(n) != entries[n] for n in order):
                raise ValueError('The rebuilt archive failed its CRC or byte check')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(bundle=TARGET.name, source_bundle_sha256=sha(Path(source).read_bytes()),
                sha256=sha(output.read_bytes()),
                bytes=output.stat().st_size, training_pairs=len(manifest['pairs']),
                training_bytes_unchanged=True, apply_images=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New file; install it over the published example only after review')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema=1, date='2026-09-25', **build(args.output))
    args.receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
