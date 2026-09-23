"""Deterministic synthetic Host–Pathogen project with independently known truth."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


VERSION = 2
SETTINGS_FILE = 'host_pathogen_settings.csv'


def example_folder():
    """Return the private cache directory for the synthetic example."""
    from .example_archives import example_plate_folder

    return example_plate_folder().parent / f'host_pathogen_v{VERSION}'


def is_present(folder):
    """Recognize a complete generated project without replacing user modifications.

    :param folder: candidate synthetic project directory.
    :returns: True when its manifest version, synthetic marker, listed files
        and nonempty measurement database are present. File contents are not
        compared with the original recipe, so user edits remain intact.
    """
    folder = Path(folder)
    try:
        record = json.loads((folder / 'example_manifest.json').read_text())
        return (record['version'] == VERSION and record['synthetic'] is True
                and all((folder / name).is_file() for name in record['files'])
                and (folder / 'measurements/measurements.db').stat().st_size > 0)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_csv(path, rows):
    """Write records with stable column order and UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _field(column, field):
    """Draw labelled compartments, measure them, and return independent truth."""
    import numpy as np

    yy, xx = np.mgrid[:256, :384]
    cell = np.zeros(yy.shape, dtype=np.uint16)
    nucleus = np.zeros_like(cell)
    vacuole = np.zeros_like(cell)
    parasite = np.zeros_like(cell)
    image = np.zeros((*yy.shape, 4), dtype=np.uint16)
    centers = {i + 1: (x, y) for i, (x, y) in enumerate(
        [(64, 64), (176, 64), (288, 64), (64, 192), (176, 192), (288, 192)])}
    for label, (x, y) in centers.items():
        inside = ((xx - x) / 47) ** 2 + ((yy - y) / 46) ** 2 <= 1
        cell[inside] = label
        nuclear = (xx - (x - 18)) ** 2 + (yy - (y - 12)) ** 2 <= 8 ** 2
        nucleus[nuclear] = label
        image[inside, 0] = 0 if label == 4 else 100
        image[inside, 1] = 120
        image[nuclear, 2] = 1800
    recipe = [
        (1, 1, 1, .8 if column == 1 else 2.5, .5, 73, 75),
        (2, 2, 2, 2.5, .5, 158, 78),
        (3, 2, 4, .8, 2.5, 195, 78),
        (4, 3, 8, 2.5, 2.5, 297, 75),
        (5, 4, 0, 2.5, .5, 73, 203),
        (6, None, 2, 2.5, 2.5, 367, 128),
        (7, 5, 4, .8, .5, 185, 203),
    ]
    identity = dict(plateID='synthetic_hp', rowID='r1', columnID=f'c{column}',
                    fieldID=str(field), timeID='1')
    rows = {'cell': [], 'cytoplasm': [], 'pathogen': [], 'organelle': []}
    truth = []
    parasite_id = 0
    for label, host, count, r0, r1, x, y in recipe:
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= 15 ** 2
        vacuole[inside] = label
        image[inside, 0] = round(100 * r0)
        image[inside, 1] = round(120 * r1)
        image[inside, 3] = 250
        for index in range(count):
            angle = index * 2 * np.pi / count
            px = x + (0 if count == 1 else 8 * np.cos(angle))
            py = y + (0 if count == 1 else 8 * np.sin(angle))
            obj = (xx - px) ** 2 + (yy - py) ** 2 <= 2 ** 2
            parasite_id += 1
            parasite[obj] = parasite_id
            image[obj, 3] = 2400
            rows['organelle'].append(dict(identity, object_label=parasite_id,
                                           cell_id=host, pathogen_id=label))
        rows['pathogen'].append(dict(identity, object_label=label, cell_id=host,
            pathogen_channel_0_mean_intensity=float(image[inside, 0].mean()),
            pathogen_channel_1_mean_intensity=float(image[inside, 1].mean())))
        state0 = 'unknown' if host in (None, 4, 5) else ('positive' if r0 >= 2 else 'negative')
        state1 = 'unknown' if host in (None, 5) else ('positive' if r1 >= 2 else 'negative')
        truth.append(dict(identity, vacuole_id=label, host_id=host,
                           parasite_count=count, channel_0_state=state0,
                           channel_1_state=state1,
                           channel_0_recruitment_ratio='' if state0 == 'unknown' else r0,
                           channel_1_recruitment_ratio='' if state1 == 'unknown' else r1))
    orphan = (xx - 365) ** 2 + (yy - 225) ** 2 <= 2 ** 2
    parasite[orphan] = parasite_id + 1
    image[orphan, 3] = 2400
    rows['organelle'].append(dict(identity, object_label=parasite_id + 1,
                                   cell_id=None, pathogen_id=None))
    for label in centers:
        rows['cell'].append(dict(identity, object_label=label))
        if label == 5:
            continue
        reference = (cell == label) & (nucleus == 0) & (vacuole == 0)
        rows['cytoplasm'].append(dict(identity, object_label=label,
            cytoplasm_channel_0_mean_intensity=float(image[reference, 0].mean()),
            cytoplasm_channel_1_mean_intensity=float(image[reference, 1].mean())))
    stack = np.concatenate([image, np.stack([cell, nucleus, vacuole, parasite], axis=-1)], axis=-1)
    return stack, rows, truth


def build_example(destination, *, progress=None, cancelled=None):
    """Create a ready-to-run synthetic project, preserving existing user files.

    :param destination: new/empty project directory or this complete cached sample.
    :param progress: optional callable receiving completed and total field counts.
    :param cancelled: optional callable; True aborts before the next field/commit.
    :returns: absolute project Path. A complete cached copy is reused unchanged.
    :raises FileExistsError: a nonempty destination is not a complete sample.
    :raises RuntimeError: generation was cancelled; no partial project is published.
    """
    import numpy as np
    import pandas as pd
    import tifffile
    from PIL import Image
    from .database_schema import ensure_database_schema
    from .host_pathogen import default_settings

    destination = Path(destination).expanduser().resolve()
    if is_present(destination):
        return destination
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'Use a new empty folder for the synthetic example: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.spacr-host-pathogen-', dir=destination.parent) as temporary:
        root = Path(temporary) / 'project'
        for name in ('measurements', 'merged', 'images', 'masks', 'previews', 'settings', 'expected'):
            (root / name).mkdir(parents=True, exist_ok=True)
        tables = {name: [] for name in ('cell', 'cytoplasm', 'pathogen', 'organelle')}
        truth = []
        for position, (column, field) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)], 1):
            if cancelled and cancelled():
                raise RuntimeError('Synthetic example generation cancelled.')
            stack, rows, expected = _field(column, field)
            stem = f'synthetic_hp_A0{column}_{field}_1'
            np.save(root / 'merged' / f'{stem}.npy', stack)
            tifffile.imwrite(root / 'images' / f'{stem}.tif', stack[..., :4].transpose(2, 0, 1),
                              metadata={'axes': 'CYX'}, photometric='minisblack')
            for offset, name in enumerate(('cell', 'nucleus', 'vacuole', 'parasite'), 4):
                tifffile.imwrite(root / 'masks' / f'{stem}_{name}.tif', stack[..., offset])
            rgb = np.stack([stack[..., 0] / 300, stack[..., 1] / 300,
                            np.maximum(stack[..., 2], stack[..., 3]) / 2400], axis=-1)
            Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(root / 'previews' / f'{stem}.png')
            for name in tables:
                tables[name].extend(rows[name])
            truth.extend(expected)
            if progress:
                progress(position, 4)
        from .tabular import write_database

        for name, rows in tables.items():
            write_database(pd.DataFrame(rows), root / 'measurements' / 'measurements.db',
                           name, if_exists='fail')
        ensure_database_schema(root / 'measurements' / 'measurements.db')
        (root / 'merged' / '.spacr_plane_layout.json').write_text(json.dumps(dict(
            version=1, intensity_channels=[0, 1, 2, 3],
            mask_plane_order=['cell', 'nucleus', 'pathogen', 'organelle'],
            mask_dims=dict(cell=4, nucleus=5, pathogen=6, organelle=7)), indent=2) + '\n')
        _write_csv(root / 'expected' / 'vacuoles.csv', truth)
        _write_csv(root / 'expected' / 'wells.csv', [dict(
            plateID='synthetic_hp', rowID='r1', columnID=f'c{column}',
            host_cells=12, infected_cells=10, multiply_infected_cells=2,
            infection_fraction=10 / 12, vacuoles=14, linked_vacuoles=12,
            vacuoles_with_counts=14, orphan_parasites=2) for column in (1, 2)])
        settings = default_settings(dict(src=str(destination), hp_marker_channels=[0, 1],
            hp_marker_thresholds={0: 2., 1: 2.}, hp_parasite_table='organelle'))
        _write_csv(root / 'settings' / SETTINGS_FILE,
                   [dict(Key=k, Value=v if isinstance(v, str) else repr(v)) for k, v in settings.items()])
        (root / 'README.md').write_text(_README, encoding='utf-8')
        manifest = dict(version=VERSION, synthetic=True, seed='deterministic geometry; no random sampling',
                        fields=4, host_cells=24, vacuoles=28, parasites=88,
                        files=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()),
                        merged_planes=['marker_0', 'marker_1', 'DNA', 'parasites',
                                       'cell_mask', 'nucleus_mask', 'vacuole_mask', 'parasite_mask'])
        (root / 'example_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if cancelled and cancelled():
            raise RuntimeError('Synthetic example generation cancelled.')
        if destination.exists():
            destination.rmdir()
        os.rename(root, destination)
    return destination


_README = """# Synthetic Host–Pathogen test project

This is generated test data, not acquired microscopy, curated biological ground
truth or evidence of model accuracy. No trained network or GPU is needed.

Open Toxoplasma → Host–Pathogen Analysis → Load test data…, then Run. The
settings select whole vacuoles in `pathogen`, host references in `cytoplasm`,
and individually labelled parasites in `organelle` with explicit `pathogen_id`
links. `organelle` is the storage table here; its objects represent parasites.
The two marker thresholds are illustrative ratios of 2, not calibrated biology.

Four synthetic fields cover two wells. Each field has six hosts, five infected
hosts, one host with two vacuoles, one uninfected host and seven vacuoles. One
vacuole is extracellular. Parasites per vacuole are 1, 2, 4, 8, 0, 2 and 4.
One additional parasite has no parent and must appear in the orphan report.
Host 4 has a zero channel-0 reference; host 5 deliberately lacks a reference
row. Their affected marker calls must remain unknown. Unknown is not negative.
Well A02 changes the first vacuole's channel-0 recruitment to positive.

The image intensities and compartment means are generated consistently. Mask
labels, host links and count truth come from the drawing recipe. Expected
results in `expected/` are independently specified from that recipe, rather
than copied from the analysis output. Each well should report 12 hosts, 10
infected hosts (10/12), two multiply infected hosts, 14 vacuoles, 12 host-linked
vacuoles and two orphan parasites. The extracellular vacuole contributes to
vacuole/count summaries, but never the host infection denominator.

`images/` contains four-channel uint16 TIFFs, `masks/` the drawn object labels,
`merged/` the eight-plane stacks listed in `example_manifest.json`, and
`previews/` RGB views. DNA and parasite fluorescence share blue in the preview;
marker 0 is red and marker 1 green. Sources and expected results are preserved
when Run writes `results/host_pathogen/`. The settings file is ready for the
GUI or `spacr-run host_pathogen --help`. For a custom location use
`python -m spacr.host_pathogen_example --out /path/to/new/project`.
"""


def main(argv=None):
    """Build the sample from the command line without requiring Qt or network.

    :param argv: optional argument list; None reads the process arguments.
    :returns: zero after printing the generated or reused project directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    print(build_example(args.out or example_folder()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
