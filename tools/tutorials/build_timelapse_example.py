"""Package the exact synthetic preview planes without changing recorded results."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

README = '''SYNTHETIC TIMELAPSE PREVIEW EXAMPLE

These images are computer-generated teaching data, NOT an acquired experiment.
The masks are actual Cellpose results from the recorded spaCR batch run, NOT
ground-truth annotations or a segmentation-accuracy benchmark.

Extract into a NEW folder. In Mask, enable Time / Timelapse, then open Track
preview. Choose sequence -> image_sequences/field_A01, then Masks ->
label_sequences/field_A01. Keep image and label sequences under separate parents
so the image field selector cannot treat a mask folder as another image field.

Use Channel 0 here: each exported image contains ONLY the recorded merged
cell-intensity plane (original channel 1). These are eight exact 256x256 uint16
planes. Each accompanying label plane is the recorded merged cell mask.
Set Frames previewed to 8, Fields to 1, Mode to iou, IoU threshold to 0.1,
Min track length to 3, and Keep only full-length tracks OFF. Leave Propagate
settings OFF: the preview's IoU threshold is NOT among its propagated keys.
Run preview reads the existing masks; it does not benchmark a model.

The recorded example gives 16 tracks of 8 frames. IoU 1.0 requires exact
pixel overlap and gives 128 one-frame tracks. Restore 0.1, wait until idle, and
click Re-link explicitly. Typing can launch a pass for an intermediate value;
do not assume the last result used the final text. Fragmentation/jump
indicators are not biological measurements.
Playback fps is not the acquisition interval and does not calibrate velocity.

For the full batch demonstration use Help -> Demos -> Timelapse demo in a
new folder. Turn Plot and Keep original images on before Run. The accompanying
tutorial records current application warnings rather than hiding them.
No saved database, model weights, or user configuration is included here.
'''


def members(source):
    source = Path(source).resolve()
    result = {'README.txt': README.encode()}
    for kind in ('image_sequences', 'label_sequences'):
        for frame in range(8):
            name = f'{kind}/field_A01/frame_{frame:02}.npy'
            path = source / name
            if not path.resolve().is_relative_to(source) or path.is_symlink():
                raise ValueError('A preview source must not be a symlink or leave its folder')
            data = path.read_bytes()
            array = np.load(io.BytesIO(data), allow_pickle=False)
            if array.shape != (256, 256) or array.dtype != np.uint16:
                raise ValueError('Expected the recorded 256 by 256 uint16 preview plane')
            result[name] = data
    result['manifest.json'] = json.dumps(dict(synthetic=True,
        acquired_biological_data=False, source_image_plane=1, source_mask_plane=2,
        source_layout='Recorded merged arrays: nucleus intensity, cell intensity, cell mask, nucleus mask',
        preview_channel=0, frames=8, width=256, height=256,
        segmentation_accuracy_claim=False,
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in result.items() if k.endswith('.npy')}),
        sort_keys=True, indent=2).encode() + b'\n'
    return result


def build_archive(source, destination):
    content = members(source)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            entry = zipfile.ZipInfo('SYNTHETIC_timelapse_preview/' + name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compresslevel=9)
    data = buffer.getvalue()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError('Refusing to replace a different tutorial archive')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
    if members(source) != content:
        raise ValueError('A preview source changed during packaging')
    with zipfile.ZipFile(destination) as archive:
        if {name.removeprefix('SYNTHETIC_timelapse_preview/'): archive.read(name)
                for name in archive.namelist()} != content:
            raise ValueError('The archive does not exactly preserve the recorded planes')
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), synthetic=True,
                members={name: hashlib.sha256(value).hexdigest() for name, value in content.items()},
                acquired_biological_data=False, segmentation_accuracy_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_archive(args.source, args.destination), indent=2))
