"""Package only the exact synthetic merged arrays used by the Motility lesson."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

README = '''SYNTHETIC MOTILITY TRACKS — NOT AN ACQUIRED EXPERIMENT

Eight exact outputs of the recorded Help > Demos > Timelapse demo.
The 256x256 uint16 arrays contain nucleus intensity (0), cell intensity (1),
tracked cell masks (2), and nucleus masks (3). Computed masks are NOT ground
truth. Sixteen cell IDs persist across eight frames; there is no pathogen
plane or acquired physical calibration. No old database or results are included.

Extract into a NEW folder. Open Measure > Time > Motility Assay. Choose
SYNTHETIC_motility_tracks, the ROOT containing merged, for Source and for
Live > Choose plate folder. Do not substitute an ordinary static test plate.

Live preview: cell; intensity channels 2; tracked mask plane 2; pathogen
mask plane -1 (none); frames 8; minimum length 3; max displacement 50;
Drop over-straight tracks off; Pixels per um 0 and Seconds per frame 0
(unknown). Leave Propagate settings OFF. Run preview should read 128 points,
with 16 usable tracks and mean speed about 1.4001441642 px/frame.
Length 9 excludes all 16 from the usable summary; restore 3 to recover them.
Drop over-straight tracks at 0.95 leaves one; switching it off restores 16.
The label uninfected here means no supplied pathogen plane, not a validated
negative biological group. No classifier or infection effect was validated.

Unit exercise ONLY: 2 pixels/um with interval 0 stays px/frame. Add 60 seconds
per frame and the mean becomes 0.7000720821 um/min. These two values are
HYPOTHETICAL, not acquisition metadata. Return both to 0 for unknown units.
The trajectory axes remain in pixels. Playback speed is not acquisition time.

For the optional recorded batch: channels [0,1], nucleus 0, cell 1, pathogen
unset, tracked object cell; Reuse existing measurements OFF; n_jobs 1;
max displacement 50; straightness filter OFF; infection QC scope none;
QC graphs ON; motility_xlim and motility_ylim [-30,30]. Batch numeric widgets
cannot express unknown calibration: the recording explicitly uses hypothetical
2 pixels/um and 60 seconds/frame, never a measured physical scale.
Run writes 128 frame observations, a 16-track well summary, CSVs and PDFs.
PDFs are not queued in the GUI and this build has faint export marks. Screen
export did NOT repair the contrast; inspect files before using them in reports.
The tutorial demonstrates genuine live plots, not publication-ready exports.
API: spacr.timelapse.automated_motility_assay. AI is not needed for this example.
'''


def members(source):
    source = Path(source).resolve()
    content = {'README.txt': README.encode()}
    names = [f'merged/plate1_A01_1_{i}.npy' for i in range(1, 9)]
    names.append('merged/.spacr_plane_layout.json')
    for name in names:
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('The recorded input must stay inside its source, without links')
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('Unexpectedly large teaching input')
        data = path.read_bytes()
        if name.endswith('.npy'):
            array = np.load(io.BytesIO(data), allow_pickle=False)
            if array.shape != (256, 256, 4) or array.dtype != np.uint16:
                raise ValueError('Expected a recorded 256x256x4 uint16 array')
        else:
            layout = json.loads(data)
            if layout.get('intensity_channels') != [0, 1] or layout.get('mask_dims') != {'cell': 2, 'nucleus': 3}:
                raise ValueError('The layout differs from the recorded plane mapping')
        content[name] = data
    content['manifest.json'] = json.dumps(dict(synthetic=True,
        acquired_biological_data=False, acquired_calibration=False, frames=8,
        shape=[256, 256, 4], intensity_channels=[0, 1], cell_mask=2, nucleus_mask=3,
        pathogen_mask=None, segmentation_accuracy_claim=False,
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in content.items() if k.startswith('merged/')}),
        sort_keys=True, indent=2).encode() + b'\n'
    return content


def build_archive(source, destination):
    content = members(source)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            entry = zipfile.ZipInfo('SYNTHETIC_motility_tracks/' + name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compresslevel=9)
    data = buffer.getvalue()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError('Refusing to overwrite a different tutorial archive')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
    if members(source) != content:
        raise ValueError('The recorded source changed while packaging')
    with zipfile.ZipFile(destination) as archive:
        if {n.removeprefix('SYNTHETIC_motility_tracks/'): archive.read(n) for n in archive.namelist()} != content:
            raise ValueError('The download does not preserve its exact inputs')
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        members={k: hashlib.sha256(v).hexdigest() for k, v in content.items()},
        synthetic=True, acquired_calibration=False, old_results_included=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_archive(args.source, args.destination), indent=2))
